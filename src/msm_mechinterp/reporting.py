"""Lightweight JSON-serialization helpers for script outputs.

Keeps result dicts JSON-safe (integer layer keys as strings) so run outputs
can be written to disk and turned into figures later without re-parsing
human-readable log text.
"""

from __future__ import annotations

import json
import os
from typing import Any


def layer_dict_to_json_safe(d: dict[int, float]) -> dict[str, float]:
    """Convert a ``{layer_idx: float}`` mapping to ``{str(layer_idx): float}`` for JSON."""
    return {str(k): v for k, v in d.items()}


def write_json(path: str, data: dict[str, Any]) -> None:
    """Write ``data`` to ``path`` as indented JSON, creating parent dirs as needed.

    Args:
        path: Destination file path.
        data: JSON-serializable structure (use :func:`layer_dict_to_json_safe`
            first for any ``dict[int, float]`` values).
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
