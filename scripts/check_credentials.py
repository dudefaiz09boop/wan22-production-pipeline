#!/usr/bin/env python3
"""Verify every credential in .env against its live service.

Reads .env, then makes one authenticated call per service and reports
PASS/FAIL. Never prints secret values.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ENV_PATH = Path(".env")


def load_env(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def call(url: str, headers: dict | None = None, timeout: int = 25) -> tuple[int, bytes]:
    merged = {"User-Agent": BROWSER_UA, "Accept": "application/json"}
    merged.update(headers or {})
    request = urllib.request.Request(url, headers=merged)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()[:400]


def check(name: str, fn) -> bool:
    try:
        ok, detail = fn()
    except Exception as error:  # noqa: BLE001 - report and continue
        print(f"  FAIL {name}: {error}")
        return False
    print(f"  {'PASS' if ok else 'FAIL'} {name}: {detail}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--env-file",
        default=str(ENV_PATH),
        help="env file to read (default: .env; use instance.env for the GPU box)",
    )
    args = ap.parse_args()
    load_env(Path(args.env_file))
    results: list[bool] = []

    def hf():
        token = os.environ.get("HF_TOKEN", "")
        if not token:
            return False, "HF_TOKEN empty"
        status, body = call(
            "https://huggingface.co/api/whoami-v2",
            {"Authorization": f"Bearer {token}"},
        )
        if status != 200:
            return False, f"HTTP {status}"
        return True, json.loads(body).get("name", "?")

    def hf_model():
        token = os.environ.get("HF_TOKEN", "")
        status, body = call(
            "https://huggingface.co/api/models/Comfy-Org/Wan-2.2",
            {"Authorization": f"Bearer {token}"},
        )
        if status != 200:
            return False, f"HTTP {status}"
        return True, f"{len(json.loads(body).get('siblings', []))} files visible"

    def civitai():
        token = os.environ.get("CIVITAI_API_KEY", "")
        if not token:
            return False, "CIVITAI_API_KEY empty"
        status, body = call(
            "https://civitai.com/api/v1/models?limit=1",
            {"Authorization": f"Bearer {token}"},
        )
        if status != 200:
            return False, f"HTTP {status}"
        return True, "authenticated"

    def civitai_model():
        token = os.environ.get("CIVITAI_API_KEY", "")
        status, body = call(
            "https://civitai.com/api/v1/model-versions/2377035",
            {"Authorization": f"Bearer {token}"},
        )
        if status != 200:
            return False, f"HTTP {status}"
        data = json.loads(body)
        files = [f["name"] for f in data.get("files", []) if f["name"].endswith(".safetensors")]
        return True, f"penis_insert high -> {files[:1]}"

    def aic():
        token = os.environ.get("AIC_API_KEY", "")
        base = os.environ.get("AIC_API_BASE", "https://api.aiccloud.in").rstrip("/")
        if not token:
            return False, "AIC_API_KEY empty"
        status, body = call(
            f"{base}/api/v1/billing/wallet",
            {"Authorization": f"Bearer {token}"},
        )
        if status != 200:
            return False, f"HTTP {status} {body[:120].decode(errors='replace')}"
        data = json.loads(body)
        balance = data.get("balance", 0) / 100
        return True, f"wallet INR {balance:.2f}"

    def kaggle():
        # The CLI authenticates with its OAuth credentials file. KAGGLE_KEY in
        # .env is a KGAT token that the legacy auth path rejects, and the CLI
        # prefers the env var over the file, so strip it for this call.
        import subprocess
        env = {k: v for k, v in os.environ.items() if k != "KAGGLE_KEY"}
        result = subprocess.run(
            ["kaggle", "datasets", "list", "--user", "sofiya009", "--page-size", "20"],
            capture_output=True, text=True, timeout=90, env=env,
        )
        if result.returncode != 0:
            return False, (result.stderr or result.stdout).strip()[:120]
        text = result.stdout
        needed = ["sofiya009/lora-train-faiz", "sofiya009/lora-train-juveriya"]
        missing = [name for name in needed if name not in text]
        if missing:
            return False, f"missing datasets: {missing}"
        return True, "both training datasets present"

    def gemini():
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            return False, "GEMINI_API_KEY empty"
        status, body = call(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
        )
        if status != 200:
            return False, f"HTTP {status}"
        return True, f"{len(json.loads(body).get('models', []))} models"

    def openai():
        # Optional and unused by the pipeline; report as skipped when absent.
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            return True, "not set (optional, unused)"
        status, body = call(
            "https://api.openai.com/v1/models",
            {"Authorization": f"Bearer {key}"},
        )
        if status != 200:
            return False, f"HTTP {status}"
        return True, f"{len(json.loads(body).get('data', []))} models"

    print("Credential checks")
    print("-" * 50)
    results.append(check("HF_TOKEN identity", hf))
    results.append(check("HF model access", hf_model))
    results.append(check("CIVITAI_API_KEY", civitai))
    results.append(check("Civitai anatomy LoRA", civitai_model))
    results.append(check("AIC_API_KEY wallet", aic))
    results.append(check("KAGGLE credentials", kaggle))
    results.append(check("GEMINI_API_KEY", gemini))
    results.append(check("OPENAI_API_KEY", openai))

    print("-" * 50)
    passed = sum(results)
    print(f"{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
