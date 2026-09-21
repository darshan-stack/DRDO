"""LiDAR semantic segmentation interfaces and range-image backend."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import numpy as np


@dataclass
class ProjectionConfig:
    height: int = 32
    width: int = 1024
    min_range: float = 1.0
    max_range: float = 80.0
    fov_up_deg: float = 10.0
    fov_down_deg: float = -30.0


class RangeImageProjector:
    def __init__(self, config: ProjectionConfig | None = None):
        self.cfg = config or ProjectionConfig()
        if self.cfg.height <= 0 or self.cfg.width <= 0:
            raise ValueError("Range-image height and width must be positive")
        if self.cfg.max_range <= self.cfg.min_range:
            raise ValueError("max_range must exceed min_range")

    def project(
        self,
        points: np.ndarray,
        intensity: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        p = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        n = len(p)
        if intensity is None:
            inten = np.zeros(n, dtype=np.float32)
        else:
            inten = np.asarray(intensity, dtype=np.float32).reshape(-1)
            if len(inten) != n:
                raise ValueError("intensity length must match points")

        depth = np.linalg.norm(p, axis=1)
        yaw = np.arctan2(p[:, 1], p[:, 0])
        pitch = np.arcsin(
            np.clip(p[:, 2] / np.maximum(depth, 1e-6), -1.0, 1.0)
        )
        col = (
            (yaw + np.pi) / (2.0 * np.pi) * self.cfg.width
        ).astype(np.int32) % self.cfg.width
        vfov = np.deg2rad(self.cfg.fov_up_deg - self.cfg.fov_down_deg)

        valid = (
            (depth >= self.cfg.min_range)
            & (depth <= self.cfg.max_range)
            & np.isfinite(p).all(axis=1)
        )

        point_index = np.full(
            (self.cfg.height, self.cfg.width),
            -1,
            dtype=np.int64,
        )
        depth_image = np.zeros(
            (self.cfg.height, self.cfg.width),
            dtype=np.float32,
        )
        intensity_image = np.zeros_like(depth_image)

        idx = np.flatnonzero(valid)
        if len(idx):
            rows = np.clip(
                (
                    (np.deg2rad(self.cfg.fov_up_deg) - pitch[idx])
                    / vfov
                    * self.cfg.height
                ).astype(np.int32),
                0,
                self.cfg.height - 1,
            )
            cols = col[idx]
            flat = rows * self.cfg.width + cols
            order = np.lexsort((depth[idx], flat))
            sorted_flat = flat[order]
            first = np.empty(len(order), dtype=bool)
            first[0] = True
            first[1:] = sorted_flat[1:] != sorted_flat[:-1]
            chosen = idx[order[first]]
            chosen_flat = sorted_flat[first]
            rr = chosen_flat // self.cfg.width
            cc = chosen_flat % self.cfg.width
            point_index[rr, cc] = chosen
            depth_image[rr, cc] = depth[chosen]
            intensity_image[rr, cc] = inten[chosen]

        return {
            "depth": depth_image,
            "intensity": intensity_image,
            "point_index": point_index,
            "valid": valid,
        }


class SemanticSegmenter:
    num_classes: int

    def predict(self, points: np.ndarray, intensity: np.ndarray | None = None):
        raise NotImplementedError


class NumpyFallbackSegmenter(SemanticSegmenter):
    """Non-neural fallback for smoke tests only."""

    def __init__(self, num_classes: int = 7):
        self.num_classes = int(num_classes)

    def predict(self, points, intensity=None):
        p = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if intensity is None:
            inten = np.zeros(len(p), dtype=np.float32)
        else:
            inten = np.asarray(intensity, dtype=np.float32).reshape(-1)
            if len(inten) != len(p):
                raise ValueError("intensity length must match points")

        labels = np.zeros(len(p), dtype=np.int64)
        labels[p[:, 2] > 0.25] = 2
        labels[inten > 0.72] = 1
        labels[p[:, 2] > 0.8] = 4
        probs = np.full(
            (len(p), self.num_classes),
            0.12 / max(self.num_classes - 1, 1),
            dtype=np.float32,
        )
        if len(p):
            probs[np.arange(len(p)), labels] = 0.88
        return labels, probs


class TorchRangeSegmenter(SemanticSegmenter):
    """Point-wise adapter around the trained compact range-image model."""

    def __init__(
        self,
        checkpoint: str,
        projection: ProjectionConfig | None = None,
        num_classes: int = 7,
        device: str = "auto",
    ):
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise RuntimeError("Install torch to use TorchRangeSegmenter.") from exc

        self.torch = torch
        self.projector = RangeImageProjector(projection)
        self.num_classes = int(num_classes)
        self.device = (
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else device
            if device != "auto"
            else "cpu"
        )

        if self.device == "cpu":
            try:
                torch.set_num_threads(
                    int(os.environ.get("FFEM_TORCH_THREADS", "4")
                ))
            except Exception:
                pass

        self.model = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, self.num_classes, 1),
        ).to(self.device)

        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")

        if isinstance(state, dict) and "classes" in state:
            if int(state["classes"]) != self.num_classes:
                raise ValueError(
                    f"Checkpoint has {state['classes']} classes but FFEM expects {self.num_classes}"
                )
        if isinstance(state, dict) and "class_names" in state:
            if len(state["class_names"]) != self.num_classes:
                raise ValueError("Checkpoint class_names length does not match model output")

        weights = state.get("model", state) if isinstance(state, dict) else state
        if not isinstance(weights, dict):
            raise ValueError("Unsupported checkpoint format: expected a state_dict")
        self.model.load_state_dict(weights, strict=True)
        self.model.eval()

    def predict(self, points: np.ndarray, intensity: np.ndarray | None = None):
        t = self.torch
        p = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if intensity is None:
            intensity = np.zeros(len(p), dtype=np.float32)
        else:
            intensity = np.asarray(intensity, dtype=np.float32).reshape(-1)
            if len(intensity) != len(p):
                raise ValueError("intensity length must match points")

        labels = np.zeros(len(p), dtype=np.int64)
        probs = np.zeros((len(p), self.num_classes), dtype=np.float32)
        if not len(p):
            return labels, probs

        image = self.projector.project(p, intensity)
        x = np.stack(
            (
                image["depth"] / self.projector.cfg.max_range,
                image["intensity"],
            ),
            axis=0,
        )[None]
        xt = t.from_numpy(x).float().to(self.device)

        with t.inference_mode():
            logits = self.model(xt)[0].cpu().numpy()

        pixel_labels = np.argmax(logits, axis=0)
        shifted = logits - logits.max(axis=0, keepdims=True)
        pixel_probs = np.exp(shifted)
        pixel_probs /= pixel_probs.sum(axis=0, keepdims=True) + 1e-8

        # Every point starts as unknown. Only points represented in the range
        # image are replaced by neural predictions.
        probs[:, 0] = 1.0
        point_index = image["point_index"]
        rows, cols = np.where(point_index >= 0)
        original = point_index[rows, cols]
        labels[original] = pixel_labels[rows, cols]
        probs[original] = pixel_probs[:, rows, cols].T
        return labels, probs
