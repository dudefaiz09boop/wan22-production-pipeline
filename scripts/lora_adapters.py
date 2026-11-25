"""Adapter selection for the SDXL stills pipeline.

Kept free of torch/diffusers imports so the selection rule can be unit tested
without a GPU stack installed.
"""
from __future__ import annotations


def select_adapters(
    loaded: list[str],
    characters: list[str],
) -> list[str]:
    """Return the adapters to activate for a beat with this cast.

    The combined adapter is a fallback only. Preferring it unconditionally (the
    previous behaviour) meant the per-character LoRAs were loaded and then never
    used, so a single-person beat still got both faces blended in.
    """
    if not loaded:
        return []

    wanted = [char for char in characters if char in loaded]
    if not wanted and "both" in loaded:
        wanted = ["both"]
    return wanted
