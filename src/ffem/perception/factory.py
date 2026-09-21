"""Checkpoint discovery and perception-backend construction."""
from __future__ import annotations

import os
from pathlib import Path

from .segmentation import (
    NumpyFallbackSegmenter,
    ProjectionConfig,
    TorchRangeSegmenter,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def discover_checkpoint(
    explicit: str = "",
    search_root: str = "models/checkpoints",
) -> str | None:
    """Resolve a checkpoint deterministically."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        return str(path.resolve())

    env = os.environ.get("FFEM_CHECKPOINT", "").strip()
    if env:
        path = Path(env).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"FFEM_CHECKPOINT does not exist: {path}"
            )
        return str(path.resolve())

    requested = Path(search_root).expanduser()
    roots = [
        requested if requested.is_absolute() else Path.cwd() / requested,
        _repo_root() / search_root,
    ]

    seen: set[Path] = set()
    candidates: list[Path] = []
    for root in roots:
        root = root.resolve()
        if root in seen or not root.exists():
            continue
        seen.add(root)
        candidates.extend(sorted(root.glob("*.pt")))

    preferred = (
        "semanticposs_range_model.pt",
        "semantic_model.pt",
        "semanticposs_first_model.pt",
    )
    for name in preferred:
        for candidate in candidates:
            if candidate.name == name:
                return str(candidate.resolve())

    return str(candidates[0].resolve()) if candidates else None


def build_segmenter(
    backend: str = "auto",
    checkpoint: str = "",
    num_classes: int = 7,
    height: int = 32,
    width: int = 1024,
    max_range: float = 80.0,
):
    selected = backend.strip().lower()
    found = discover_checkpoint(checkpoint)

    if selected == "auto":
        if found is None:
            raise FileNotFoundError(
                "No FFEM checkpoint was found. Provide checkpoint:=... or "
                "FFEM_CHECKPOINT=/absolute/path/model.pt. "
                "Use model_backend:=fallback only for smoke tests."
            )
        selected = "torch_range"

    if selected == "torch_range":
        if found is None:
            raise FileNotFoundError(
                "torch_range requires a checkpoint; pass checkpoint:=... "
                "or set FFEM_CHECKPOINT."
            )
        return (
            TorchRangeSegmenter(
                found,
                ProjectionConfig(
                    height=int(height),
                    width=int(width),
                    max_range=float(max_range),
                ),
                num_classes=int(num_classes),
            ),
            found,
        )

    if selected == "fallback":
        return NumpyFallbackSegmenter(int(num_classes)), None

    raise ValueError(
        f"Unknown perception backend '{backend}'. "
        "Expected one of: auto, torch_range, fallback."
    )
