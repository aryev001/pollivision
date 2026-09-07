"""The perception pipeline: one frame in, analysed flowers out.

This assembles the stages of the report's Sec. III-C and III-D in order:
detection and fusion, quality gating, sex and stage classification, pollen and
stigma analysis, localisation into the rover frame, and temporal tracking. The
act-and-verify half of Fig. 1 lives in :mod:`pollivision.runtime.mission`, which
drives this class.

Ordering here is chosen to do the cheap work first. The quality gate runs before
any vision-language scoring so that occluded, blurred and truncated candidates
are discarded before they cost a model inference - on a 4-core SBC that
reordering alone is the difference between a usable frame rate and a slideshow.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ..backends.clip_embed import get_vlm
from ..geometry.camera import CameraIntrinsics, Extrinsics
from ..geometry.localizer import Localizer
from ..logging_utils import get_logger
from ..perception import regions
from ..perception.anthesis import AnthesisClassifier
from ..perception.detector import EnsembleDetector, associate_ovaries, split_roles
from ..perception.occlusion import QualityGate
from ..perception.pollen import PollenEstimator
from ..perception.sex import SexClassifier
from ..perception.stigma import StigmaLocator
from ..tracking.tracker import FlowerTracker
from ..types import Detection, FlowerObservation, FlowerSex, FrameResult

LOGGER = get_logger(__name__)


class PerceptionPipeline:
    """End-to-end per-frame flower perception."""

    def __init__(self, cfg, device: Optional[str] = None) -> None:
        self.cfg = cfg
        self.device = device or str(cfg.get("runtime.device", "cpu"))
        self._configure_threads()

        self.detector = EnsembleDetector(cfg, device=self.device)
        self.vlm = get_vlm(cfg, device=self.device)

        self.quality = QualityGate(cfg)
        self.sex = SexClassifier(cfg, vlm=self.vlm)
        self.anthesis = AnthesisClassifier(cfg, vlm=self.vlm)
        self.pollen = PollenEstimator(cfg, vlm=self.vlm)
        self.stigma = StigmaLocator(cfg)

        self.intrinsics = CameraIntrinsics.from_config(cfg.section("camera"))
        self.extrinsics = Extrinsics.from_config(cfg.section("extrinsics"))
        self.localizer = Localizer(cfg, self.intrinsics, self.extrinsics)

        self.tracker = FlowerTracker(cfg)

        self.detect_every = max(1, int(cfg.get("runtime.detect_every_n_frames", 1)))
        self.max_flowers = int(cfg.get("runtime.max_flowers_per_frame", 12))
        self.warn_latency = float(cfg.get("runtime.warn_latency_ms", 800))

        self._frame_index = 0
        self._last_result: Optional[FrameResult] = None

    def _configure_threads(self) -> None:
        threads = int(self.cfg.get("runtime.torch_threads", 0))
        if threads > 0:
            try:
                import torch

                torch.set_num_threads(threads)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ #

    def process(self, frame: np.ndarray,
                depth: Optional[np.ndarray] = None) -> FrameResult:
        """Analyse a single BGR frame."""
        self._frame_index += 1
        started = time.perf_counter()
        height, width = frame.shape[:2]

        result = FrameResult(frame_index=self._frame_index, width=width, height=height)

        # Detection can be skipped on intermediate frames; the tracker carries
        # identity across the gap and the previous frame's analysis stands.
        if (self._frame_index - 1) % self.detect_every != 0 and self._last_result is not None:
            result.flowers = self._last_result.flowers
            result.rejected = self._last_result.rejected
            result.depth = self._last_result.depth
            result.latency_ms["skipped_detection"] = 0.0
            return result

        detections, detect_latency = self.detector.detect(frame)
        result.latency_ms.update(detect_latency)

        by_role = split_roles(detections)
        flowers_raw: list[Detection] = by_role.get("flower", []) + by_role.get("bud", [])
        ovaries: list[Detection] = by_role.get("ovary", [])

        if not flowers_raw:
            result.latency_ms["total"] = (time.perf_counter() - started) * 1000.0
            self._last_result = result
            return result

        flowers_raw.sort(key=lambda d: d.score, reverse=True)
        flowers_raw = flowers_raw[: self.max_flowers]

        # ---- Quality gate first, so rejected candidates cost no inference ---
        stage = time.perf_counter()
        accepted: list[Detection] = []
        rejected: list[FlowerObservation] = []
        for detection in flowers_raw:
            assessment = self.quality.assess(frame, detection.box,
                                             detection.score, detection.mask)
            if assessment.accepted:
                detection.meta["quality"] = assessment
                accepted.append(detection)
            else:
                observation = FlowerObservation(
                    box=detection.box, score=detection.score, mask=detection.mask,
                    quality=assessment, sources=[detection.source],
                )
                rejected.append(observation)
        result.latency_ms["quality"] = (time.perf_counter() - stage) * 1000.0
        result.rejected = rejected

        if not accepted:
            result.latency_ms["total"] = (time.perf_counter() - started) * 1000.0
            self._last_result = result
            return result

        boxes = [d.box for d in accepted]
        masks = [d.mask for d in accepted]
        labels = [d.label for d in accepted]

        # ---- Shape: ellipse fit drives ranging, orientation and region masks --
        ellipses = []
        for detection in accepted:
            ellipse = None
            if detection.mask is not None:
                local = regions.mask_in_box(detection.mask, detection.box)
                if local is not None and local.sum() > 32:
                    fitted = regions.fit_ellipse(regions.largest_component(local))
                    if fitted is not None:
                        # fit_ellipse works in crop coordinates; lift back to
                        # frame coordinates so every consumer shares one frame.
                        fitted.cx += max(0.0, np.floor(detection.box.x1))
                        fitted.cy += max(0.0, np.floor(detection.box.y1))
                        ellipse = fitted
            if ellipse is None:
                ellipse = regions.ellipse_from_box(detection.box)
            ellipses.append(ellipse)

        # ---- Depth ---------------------------------------------------------
        # Resolved before the perception heads because the sex head's
        # morphology cue needs it to separate an inferior ovary from the canopy
        # behind it; a monocular run simply gets None here and the cue abstains.
        stage = time.perf_counter()
        probe_flowers = [
            FlowerObservation(box=d.box, score=d.score, mask=d.mask, ellipse=e)
            for d, e in zip(accepted, ellipses)
        ]
        depth = self.localizer.resolve_depth(frame, probe_flowers, depth)
        result.latency_ms["depth"] = (time.perf_counter() - stage) * 1000.0

        # ---- Sex classification -------------------------------------------
        stage = time.perf_counter()
        ovary_matches = associate_ovaries(accepted, ovaries)
        sex_hints = {
            i: str(d.meta["sex_hint"]) for i, d in enumerate(accepted)
            if "sex_hint" in d.meta
        }
        sex_estimates = self.sex.classify(frame, boxes, masks, ovary_matches,
                                          sex_hints, depth=depth)
        result.latency_ms["sex"] = (time.perf_counter() - stage) * 1000.0

        # ---- Anthesis staging ---------------------------------------------
        stage = time.perf_counter()
        anthesis_estimates = self.anthesis.classify(frame, boxes, masks, labels)
        result.latency_ms["anthesis"] = (time.perf_counter() - stage) * 1000.0

        # ---- Pollen availability -------------------------------------------
        stage = time.perf_counter()
        pollen_estimates = self.pollen.estimate(frame, boxes, masks, ellipses)
        result.latency_ms["pollen"] = (time.perf_counter() - stage) * 1000.0

        # ---- Assemble observations -----------------------------------------
        flowers: list[FlowerObservation] = []
        for index, detection in enumerate(accepted):
            observation = FlowerObservation(
                box=detection.box,
                score=detection.score,
                mask=detection.mask,
                ellipse=ellipses[index],
                sex=sex_estimates[index],
                anthesis=anthesis_estimates[index],
                pollen=pollen_estimates[index],
                quality=detection.meta.get("quality"),
                sources=list(detection.meta.get("sources", [detection.source])),
                meta={"prompts": detection.meta.get("prompts", [])},
            )

            # The aiming point depends on which structure the probe must reach:
            # the anther on a staminate flower, the stigma on a pistillate one.
            if observation.sex.sex is FlowerSex.FEMALE:
                stigma_box, aim, confidence = self.stigma.locate(
                    frame, detection.box, detection.mask, ellipses[index])
                observation.aim_point = aim
                observation.meta["stigma_box"] = stigma_box
                observation.meta["stigma_confidence"] = confidence
            elif observation.pollen.anther_box is not None:
                observation.aim_point = observation.pollen.anther_box.center
            else:
                observation.aim_point = (ellipses[index].cx, ellipses[index].cy)

            flowers.append(observation)

        # ---- Localisation ---------------------------------------------------
        stage = time.perf_counter()
        result.depth = self.localizer.localize(frame, flowers, depth)
        result.latency_ms["localize"] = (time.perf_counter() - stage) * 1000.0

        # ---- Tracking and temporal smoothing --------------------------------
        stage = time.perf_counter()
        flowers = self.tracker.update(flowers)
        result.latency_ms["tracking"] = (time.perf_counter() - stage) * 1000.0

        # Re-apply the receptivity requirement after smoothing: the tracker can
        # legitimately change a flower's stage once several frames agree.
        if self.quality.require_receptive:
            for observation in flowers:
                if not observation.anthesis.stage.is_viable:
                    from ..types import RejectReason

                    observation.quality.reject(RejectReason.NOT_RECEPTIVE)

        result.flowers = flowers
        result.latency_ms["total"] = (time.perf_counter() - started) * 1000.0

        if result.latency_ms["total"] > self.warn_latency:
            LOGGER.warning(
                "Frame %d took %.0f ms (budget %.0f ms). Consider raising "
                "runtime.detect_every_n_frames or lowering detector imgsz.",
                self._frame_index, result.latency_ms["total"], self.warn_latency,
            )

        self._last_result = result
        return result

    def warmup(self, shape: tuple[int, int] = (480, 640)) -> None:
        """Run one dummy frame so the first real one is not pathologically slow."""
        self.detector.warmup(shape)

    def reset(self) -> None:
        self.tracker.reset()
        self._frame_index = 0
        self._last_result = None
