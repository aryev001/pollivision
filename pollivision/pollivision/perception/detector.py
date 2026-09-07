"""Ensemble detector: runs every enabled backend and fuses their output.

This is the entry point to the perception stack. It owns backend construction,
per-frame execution, box fusion, and the association of ovary detections with
their parent flower - the last of which feeds a cue the sex head relies on.
"""

from __future__ import annotations

import time

import numpy as np

from ..backends.base import PromptBank
from ..backends.yoloe import YoloEBackend
from ..fusion import wbf
from ..logging_utils import get_logger
from ..types import BBox, Detection

LOGGER = get_logger(__name__)


class EnsembleDetector:
    """Multi-backend flower detector with weighted box fusion."""

    def __init__(self, cfg, device: str = "cpu") -> None:
        self.cfg = cfg
        self.device = device
        self.prompts = PromptBank.from_config(cfg)
        self.backends = self._build_backends()
        if not self.backends:
            raise RuntimeError(
                "No detector backend could be constructed. Check detector.backends "
                "in your config and that weights are reachable or cached."
            )
        LOGGER.info("Detector ensemble: %s", [b.name for b in self.backends])

        fusion = cfg.section("detector.fusion")
        self.iou_threshold = float(fusion.get("iou_threshold", 0.55))
        self.score_threshold = float(fusion.get("score_threshold", 0.12))
        self.require_votes = int(fusion.get("require_votes", 1))

    # -- construction ------------------------------------------------------ #

    def _build_backends(self) -> list:
        backends = []
        for spec in self.cfg.get("detector.backends", []) or []:
            if not spec.get("enabled", False):
                continue
            name = spec.get("name")
            try:
                backends.append(self._make_backend(name, spec))
            except Exception as exc:  # noqa: BLE001
                # One unavailable backend must not take the whole rover down;
                # the ensemble is explicitly designed to run degraded.
                LOGGER.warning("Backend '%s' unavailable: %s", name, exc)
        return backends

    def _make_backend(self, name: str, spec: dict):
        common = dict(
            conf=float(spec.get("conf", 0.10)),
            device=self.device,
            weight=float(spec.get("weight", 1.0)),
        )
        if name == "yoloe":
            return YoloEBackend(
                prompts=self.prompts,
                weights=spec.get("weights", "yoloe-11s-seg"),
                iou=float(spec.get("iou", 0.6)),
                imgsz=int(spec.get("imgsz", 640)),
                max_det=int(spec.get("max_det", 60)),
                **common,
            )
        if name == "yolo_world":
            from ..backends.extra import YoloWorldBackend

            return YoloWorldBackend(
                prompts=self.prompts,
                weights=spec.get("weights", "yolov8s-worldv2.pt"),
                iou=float(spec.get("iou", 0.6)),
                imgsz=int(spec.get("imgsz", 640)),
                max_det=int(spec.get("max_det", 60)),
                **common,
            )
        if name == "owlv2":
            from ..backends.extra import Owlv2Backend

            return Owlv2Backend(
                prompts=self.prompts,
                weights=spec.get("weights", "google/owlv2-base-patch16-ensemble"),
                **common,
            )
        if name == "finetuned":
            from ..backends.extra import FineTunedBackend

            return FineTunedBackend(
                weights=spec.get("weights"),
                iou=float(spec.get("iou", 0.6)),
                imgsz=int(spec.get("imgsz", 640)),
                max_det=int(spec.get("max_det", 60)),
                prompts=self.prompts,
                **common,
            )
        raise ValueError(f"Unknown detector backend '{name}'")

    # -- inference --------------------------------------------------------- #

    def detect(self, frame: np.ndarray) -> tuple[list[Detection], dict[str, float]]:
        """Detect and fuse.

        Returns the fused detections plus per-backend latencies in milliseconds.
        """
        raw: list[Detection] = []
        latency: dict[str, float] = {}

        for backend in self.backends:
            start = time.perf_counter()
            try:
                found = backend.detect(frame)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Backend '%s' failed on this frame: %s", backend.name, exc)
                found = []
            latency[backend.name] = (time.perf_counter() - start) * 1000.0
            for det in found:
                det.meta["backend_weight"] = backend.weight
            raw.extend(found)

        start = time.perf_counter()
        fused = wbf.fuse(
            raw,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            require_votes=self.require_votes,
            n_backends=max(len(self.backends), 1),
        )
        latency["fusion"] = (time.perf_counter() - start) * 1000.0
        return fused, latency

    def warmup(self, shape: tuple[int, int] = (480, 640)) -> None:
        for backend in self.backends:
            backend.warmup(shape)


def split_roles(detections: list[Detection]) -> dict[str, list[Detection]]:
    """Group fused detections by canonical role."""
    out: dict[str, list[Detection]] = {}
    for det in detections:
        out.setdefault(det.label, []).append(det)
    return out


def associate_ovaries(
    flowers: list[Detection],
    ovaries: list[Detection],
    max_gap_ratio: float = 0.9,
) -> dict[int, tuple[BBox, float]]:
    """Attach each detected ovary to the flower it sits beneath.

    A pistillate cucurbit flower carries its inferior ovary directly below the
    corolla, so the association test is a vertical one: the ovary must sit at or
    below the corolla centre, overlap it horizontally, and be close enough that
    the gap is a fraction of the corolla's own size. Detections that satisfy all
    three are strong evidence of a female flower.

    Args:
        flowers: Fused flower detections.
        ovaries: Fused ovary detections.
        max_gap_ratio: Largest vertical gap, as a multiple of corolla height.

    Returns:
        ``{flower_index: (ovary_box, association_score)}`` for matched flowers.
    """
    matches: dict[int, tuple[BBox, float]] = {}
    if not flowers or not ovaries:
        return matches

    used: set[int] = set()
    for flower_index, flower in enumerate(flowers):
        best_score, best_box, best_ovary = 0.0, None, -1
        for ovary_index, ovary in enumerate(ovaries):
            if ovary_index in used:
                continue

            # Horizontal overlap, normalised by the narrower of the two boxes.
            overlap = min(flower.box.x2, ovary.box.x2) - max(flower.box.x1, ovary.box.x1)
            narrower = max(min(flower.box.width, ovary.box.width), 1e-6)
            horizontal = float(np.clip(overlap / narrower, 0.0, 1.0))
            if horizontal < 0.25:
                continue

            # The ovary must be below the corolla centre, and near it.
            gap = ovary.box.cy - flower.box.cy
            if gap < 0:
                continue
            gap_ratio = gap / max(flower.box.height, 1e-6)
            if gap_ratio > max_gap_ratio + 0.5:
                continue
            proximity = float(np.clip(1.0 - gap_ratio / (max_gap_ratio + 0.5), 0.0, 1.0))

            score = horizontal * proximity * ovary.score
            if score > best_score:
                best_score, best_box, best_ovary = score, ovary.box, ovary_index

        if best_box is not None and best_score > 0.05:
            matches[flower_index] = (best_box, float(best_score))
            used.add(best_ovary)
    return matches
