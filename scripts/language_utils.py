#!/usr/bin/env python3
"""Helpers for selecting a single spoken-text representation."""
from __future__ import annotations


def select_language_text(
    romanized: str,
    script: str,
    language: str,
    requested_language: str,
) -> str:
    if not script:
        return romanized
    language_code = language.strip().lower().split("-", 1)[0]
    if requested_language == "mixed" or requested_language == language_code:
        return script
    return romanized
