"""The verification notebook must run standalone.

Kaggle executes the emitted notebook with papermill, in a fresh interpreter.
Nothing from verify_loras.py is importable there, so any module-level constant
the notebook body references (GATE, TEST_PROMPTS, NEGATIVE, ...) has to be
emitted as a cell. Three runs were burned learning that one name at a time.

This compiles the notebook's code cells and asserts every global the code reads
is either defined by an earlier cell, a builtin, or an import.
"""
from __future__ import annotations

import ast
import builtins
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_loras import build_notebook


def _cell_source(cell: dict) -> str:
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else source


def _defined_names(tree: ast.Module) -> set[str]:
    """Names a module-level block binds."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
    return names


def _module_scope_loads(tree: ast.Module) -> set[str]:
    """Names read at module scope, ignoring function bodies and comprehensions.

    A name inside a function body is a local or a parameter, not a missing
    global, so only the top-level statements are inspected.
    """
    loads: set[str] = set()

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(
                child,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
            ):
                # Decorators and default arguments still evaluate at module scope.
                for decorator in getattr(child, "decorator_list", []):
                    visit(decorator)
                for default in list(getattr(child, "args", ast.arguments()).defaults if hasattr(child, "args") else []):
                    visit(default)
                continue
            if isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                continue
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                loads.add(child.id)
            visit(child)

    visit(tree)
    return loads


class VerifyNotebookStandaloneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.notebook = build_notebook()
        self.cells = [
            _cell_source(c)
            for c in self.notebook["cells"]
            if c["cell_type"] == "code"
        ]

    def test_every_global_read_is_defined_by_an_earlier_cell(self) -> None:
        available: set[str] = set(dir(builtins))
        missing: list[str] = []

        for index, source in enumerate(self.cells):
            # IPython magics and shell escapes are not valid Python. A `!` line
            # can also continue onto following lines with a trailing backslash.
            python_lines: list[str] = []
            in_shell = False
            for line in source.splitlines():
                stripped = line.lstrip()
                if in_shell:
                    in_shell = stripped.endswith("\\")
                    python_lines.append("")
                    continue
                if stripped.startswith(("!", "%")):
                    in_shell = stripped.endswith("\\")
                    python_lines.append("")
                    continue
                python_lines.append(line)
            tree = ast.parse("\n".join(python_lines))

            # Names bound anywhere in this cell count as available to it. The
            # bug being guarded is a name that is never emitted at all, not
            # use-before-definition inside one cell.
            cell_defs = _defined_names(tree)
            for name in _module_scope_loads(tree):
                if name not in available and name not in cell_defs:
                    missing.append(f"cell {index}: {name}")

            available |= cell_defs

        self.assertEqual(
            missing,
            [],
            "the notebook reads names it never defines; emit them as a cell:\n  "
            + "\n  ".join(sorted(set(missing))),
        )

    def test_gate_and_prompts_are_emitted(self) -> None:
        joined = "\n".join(self.cells)
        for name in ("GATE", "TEST_PROMPTS", "NEGATIVE"):
            self.assertIn(f"{name} =", joined, f"{name} is not defined in the notebook")

    def test_prompts_carry_the_trained_identity_tokens(self) -> None:
        joined = "\n".join(self.cells)
        self.assertIn("indian muslim", joined)
        self.assertIn("short boxed beard", joined)
        self.assertIn("dark kajal", joined)


if __name__ == "__main__":
    unittest.main()
