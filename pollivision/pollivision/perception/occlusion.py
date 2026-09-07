"""Detection quality gate (report Sec. III-C).

The report specifies that detections are "subsequently filtered to remove
partially occluded or low-confidence detections". This module implements that
filter, and it does more work than the phrase suggests: every downstream stage
assumes it is looking at a cleanly imaged corolla. Sex classification needs an
unobstructed view beneath the flower, pollen estimation needs unclipped chroma
and texture, and orientation estimation needs a complete outline - a corolla
half-hidden behind a leaf yields a truncated mask whose fitted ellipse implies a
tilt that is not there, which would then feed a wrong voltage into the
electrostatic controller.

Rejecting early is therefore not conservatism for its own sake; it prevents a
segmentation artefact from propagating into an actuation decision.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..types import BBox, QualityAssessment, RejectReason
from . import regions


class QualityGate:
    """Scores and filters flower candidates."""

    def __init__(self, cfg) -> None:
        section = cfg.section("perception.quality")
        self.min_score = float(section.get("min_score", 0.20))
        self.min_box_px = float(section.get("min_box_px", 24))
        self.max_truncation = float(section.get("max_truncation", 0.18))
        self.max_occlusion = float(section.get("max_occlusion", 0.45))
        self.min_sharpness = float(section.get("min_sharpness", 0.12))
        self.min_exposure = float(section.get("min_exposure", 0.35))
        self.max_clipped = float(section.get("max_clipped_fraction", 0.25))
        self.require_receptive = bool(section.get("require_receptive", True))
        # Cucurbit corollas are deeply lobed, so even a perfectly clear flower
        # has a convexity deficit. This is the deficit treated as normal petal
        # shape rather than as something lying across the flower; it is
        # species-dependent (a pumpkin corolla is far more lobed than a
        # cucumber one) and worth tuning if clean flowers are being rejected.
        self.petal_lobe_allowance = float(section.get("petal_lobe_allowance", 0.15))

        foliage = cfg.section("perception.sex.morphology")
        self.foliage_hue = tuple(foliage.get("hue_range", [30, 95]))
        self.foliage_saturation = float(foliage.get("min_saturation", 45))

    def assess(
        self,
        frame: np.ndarray,
        box: BBox,
        score: float,
        mask: Optional[np.ndarray] = None,
    ) -> QualityAssessment:
        height, width = frame.shape[:2]
        assessment = QualityAssessment()

        if score < self.min_score:
            assessment.reject(RejectReason.LOW_CONFIDENCE)

        if min(box.width, box.height) < self.min_box_px:
            assessment.reject(RejectReason.TOO_SMALL)

        assessment.truncation = self._truncation(box, width, height)
        if assessment.truncation > self.max_truncation:
            assessment.reject(RejectReason.TRUNCATED)

        patch = regions.crop(frame, box)
        if patch.size == 0:
            assessment.reject(RejectReason.TOO_SMALL)
            return assessment

        local_mask = None
        if mask is not None and mask.shape[:2] == frame.shape[:2]:
            candidate = regions.mask_in_box(mask, box)
            if candidate is not None and candidate.shape[:2] == patch.shape[:2]:
                local_mask = candidate

        assessment.sharpness = regions.sharpness(patch)
        if assessment.sharpness < self.min_sharpness:
            assessment.reject(RejectReason.BLURRED)

        assessment.exposure, clipped = regions.exposure_quality(patch, local_mask)
        if assessment.exposure < self.min_exposure or clipped > self.max_clipped:
            assessment.reject(RejectReason.EXPOSURE)

        assessment.occlusion = self._occlusion(patch, local_mask)
        if assessment.occlusion > self.max_occlusion:
            assessment.reject(RejectReason.OCCLUDED)

        return assessment

    @staticmethod
    def _truncation(box: BBox, width: int, height: int) -> float:
        """Fraction of the box that falls outside the frame.

        A flower clipped by the frame edge has an incomplete outline, so its
        fitted ellipse - and therefore its range and orientation estimates - are
        unreliable even though the visible part may look perfectly sharp.
        """
        clipped = box.clip(width, height)
        if box.area <= 1e-6:
            return 1.0
        return float(np.clip(1.0 - clipped.area / box.area, 0.0, 1.0))

    def _occlusion(self, patch: np.ndarray, mask: Optional[np.ndarray]) -> float:
        """Estimate how much of the corolla is covered.

        With a mask: the shortfall between the mask and its convex hull. A leaf
        lying across a flower carves a bite out of the mask, and that bite shows
        up directly as a convexity deficit.

        Without a mask: the fraction of the box reading as foliage green, which
        is cruder but still catches a flower buried in leaves.
        """
        if mask is not None and mask.sum() > 32:
            component = regions.largest_component(mask)
            deficit = 1.0 - regions.solidity(component)
            # Subtract the deficit that lobed petals produce on their own, so
            # only the excess reads as something covering the flower.
            excess = deficit - self.petal_lobe_allowance
            return float(np.clip(excess / 0.5, 0.0, 1.0))

        hsv = regions.hsv_of(patch)
        green = regions.green_fraction(hsv, None, self.foliage_hue, self.foliage_saturation)
        return float(np.clip((green - 0.35) / 0.45, 0.0, 1.0))
