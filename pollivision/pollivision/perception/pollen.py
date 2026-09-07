"""Pollen availability estimation (report Sec. III-E-iii).

The report's third adaptive input is "an estimated measure of pollen
availability derived from the visual appearance of the male flower's anther".
This module produces that estimate, and it is what makes the electrostatic
controller adaptive rather than merely reactive: a heavily loaded anther needs
a short dwell, a nearly-spent one needs a long one or should be skipped
entirely so the rover does not waste a visit.

Four independent measurements are combined:

* **Area fraction** - how much of the inner corolla disc reads as anther-coloured
  rather than petal-coloured. Measured *relative to the petal ring* rather than
  against fixed thresholds, because the anther and the petals are both yellow
  and their absolute hues shift with sunlight and white balance. The relative
  comparison is what makes this survive an ESP32-CAM's drifting auto white
  balance.
* **Granularity** - pollen is a powder of discrete grains, and it produces
  high-frequency texture that a bare, dehisced anther does not have.
* **Chroma excess** - loaded anthers are markedly more saturated and more orange
  than the surrounding corolla.
* **Vision-language score** - a zero-shot loaded-vs-depleted judgement on the
  anther crop, which catches cases the colour statistics miss.

The output is a 0-1 availability with a confidence, plus the located anther box
which doubles as the probe's aiming point for collection.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..fusion.calibration import temperature_scale
from ..logging_utils import get_logger
from ..types import BBox, Ellipse, PollenEstimate, ScoreBreakdown
from . import regions

LOGGER = get_logger(__name__)


class PollenEstimator:
    """Estimates anther pollen load from colour, texture and semantics."""

    def __init__(self, cfg, vlm=None) -> None:
        self.cfg = cfg
        self.vlm = vlm
        section = cfg.section("perception.pollen")
        self.enabled = bool(section.get("enabled", True))

        self.inner_ratio = float(section.get("inner_radius_ratio", 0.45))
        self.ring_ratio = float(section.get("petal_ring_ratio", 0.80))
        self.hue_range = tuple(section.get("hue_range", [14, 42]))
        self.min_saturation = float(section.get("min_saturation", 110))
        self.min_value = float(section.get("min_value", 90))
        self.saturation_margin = float(section.get("saturation_margin", 25))
        self.granularity_scale = float(section.get("granularity_scale", 400.0))

        weights = section.section("weights")
        self.w_area = float(weights.get("area", 1.0))
        self.w_granularity = float(weights.get("granularity", 0.6))
        self.w_chroma = float(weights.get("chroma", 0.8))
        self.w_vlm = float(weights.get("vlm", 0.9))

        prompts = section.section("prompts")
        self.prompt_groups = {
            "loaded": list(prompts.get("loaded", [])),
            "depleted": list(prompts.get("depleted", [])),
        }

    def estimate(
        self,
        frame: np.ndarray,
        boxes: list[BBox],
        masks: list[Optional[np.ndarray]],
        ellipses: list[Optional[Ellipse]],
    ) -> list[PollenEstimate]:
        if not boxes:
            return []
        if not self.enabled:
            return [PollenEstimate() for _ in boxes]

        measurements = [
            self._measure(frame, box,
                          masks[index] if index < len(masks) else None,
                          ellipses[index] if index < len(ellipses) else None)
            for index, box in enumerate(boxes)
        ]

        vlm_scores = self._score_vlm(frame, [m["anther_box"] or boxes[i]
                                             for i, m in enumerate(measurements)])

        estimates: list[PollenEstimate] = []
        for index, measurement in enumerate(measurements):
            breakdown = ScoreBreakdown()
            total, weight_sum = 0.0, 0.0

            for name, value, weight in (
                ("area", measurement["area_fraction"], self.w_area),
                ("granularity", measurement["granularity"], self.w_granularity),
                ("chroma", measurement["chroma"], self.w_chroma),
            ):
                total += weight * value
                weight_sum += weight
                breakdown.add(name, float(value), weight)

            if vlm_scores is not None:
                # Soften the zero-shot softmax before it competes with the
                # physical measurements.
                score = temperature_scale(float(vlm_scores[index]), 2.0)
                total += self.w_vlm * score
                weight_sum += self.w_vlm
                breakdown.add("vlm", score, self.w_vlm)

            availability = float(np.clip(total / max(weight_sum, 1e-6), 0.0, 1.0))
            # Browning anthers hold pollen that is largely non-viable, so
            # freshness scales the final availability rather than being a
            # separate additive cue.
            availability *= measurement["freshness"]

            estimates.append(PollenEstimate(
                availability=float(np.clip(availability, 0.0, 1.0)),
                confidence=float(measurement["confidence"]),
                anther_box=measurement["anther_box"],
                area_fraction=float(measurement["area_fraction"]),
                granularity=float(measurement["granularity"]),
                freshness=float(measurement["freshness"]),
                breakdown=breakdown,
            ))
        return estimates

    # ------------------------------------------------------------------ #

    def _score_vlm(self, frame: np.ndarray, boxes: list[BBox]) -> Optional[np.ndarray]:
        if self.vlm is None or not self.prompt_groups["loaded"]:
            return None
        crops = [regions.crop(frame, box.scaled(1.4, frame.shape[1], frame.shape[0]))
                 for box in boxes]
        try:
            scored = self.vlm.score_groups(crops, self.prompt_groups)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Vision-language pollen scoring failed: %s", exc)
            return None
        return np.array([row.get("loaded", 0.5) for row in scored], dtype=np.float32)

    def _measure(self, frame: np.ndarray, box: BBox, mask: Optional[np.ndarray],
                 ellipse: Optional[Ellipse]) -> dict:
        """Colour/texture measurement of the anther region."""
        empty = {
            "area_fraction": 0.0, "granularity": 0.0, "chroma": 0.0,
            "freshness": 1.0, "confidence": 0.0, "anther_box": None,
        }

        patch = regions.crop(frame, box)
        if patch.size == 0 or patch.shape[0] < 6 or patch.shape[1] < 6:
            return empty

        offset = (max(0.0, np.floor(box.x1)), max(0.0, np.floor(box.y1)))
        local_mask = None
        if mask is not None and mask.shape[:2] == frame.shape[:2]:
            candidate = regions.mask_in_box(mask, box)
            if candidate is not None and candidate.shape[:2] == patch.shape[:2]:
                local_mask = candidate

        region = regions.corolla_regions(
            patch.shape[:2], ellipse=ellipse, inner_ratio=self.inner_ratio,
            ring_ratio=self.ring_ratio, offset=offset, mask=local_mask,
        )
        if region.inner.sum() < 24:
            return empty

        hsv = regions.hsv_of(patch)
        hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        # Petal baseline from the outer annulus. If the ring is too small to be
        # reliable (heavy occlusion, tight crop), fall back to fixed thresholds.
        if region.ring.sum() >= 24:
            petal_saturation = float(np.percentile(saturation[region.ring], 60))
            petal_hue = float(np.percentile(hue[region.ring], 50))
            have_baseline = True
        else:
            petal_saturation, petal_hue, have_baseline = self.min_saturation, 30.0, False

        # An anther pixel is inside the anther hue gate AND meaningfully more
        # saturated than the petals around it.
        gate = regions.hue_mask(hsv, self.hue_range, 0, self.min_value)
        threshold = max(self.min_saturation,
                        petal_saturation + self.saturation_margin) if have_baseline \
            else self.min_saturation
        anther = gate & (saturation >= threshold) & region.inner

        area_fraction = float(anther.sum() / max(region.inner.sum(), 1))
        # Rescale: an anther occupying much more than half the inner disc is
        # already a fully loaded one, so saturate the score there.
        area_score = float(np.clip(area_fraction / 0.55, 0.0, 1.0))

        anther_box = None
        granularity, chroma, freshness = 0.0, 0.0, 1.0
        confidence = 0.25 if not have_baseline else 0.5

        if anther.sum() >= 20:
            cleaned = cv2.morphologyEx(anther.astype(np.uint8), cv2.MORPH_OPEN,
                                       np.ones((3, 3), np.uint8))
            component = regions.largest_component(cleaned)
            if component.sum() >= 12:
                ys, xs = np.nonzero(component)
                anther_box = BBox(
                    float(offset[0] + xs.min()), float(offset[1] + ys.min()),
                    float(offset[0] + xs.max() + 1), float(offset[1] + ys.max() + 1),
                )
                granularity = regions.texture_energy(patch, component, self.granularity_scale)

                anther_saturation = float(np.percentile(saturation[component], 60))
                # Excess saturation over the petal baseline, normalised by a
                # margin beyond which the anther is unambiguous.
                chroma = float(np.clip(
                    (anther_saturation - petal_saturation) / 90.0, 0.0, 1.0))

                # Browning: hue drifting below the yellow band with falling value.
                anther_hue = float(np.percentile(hue[component], 50))
                anther_value = float(np.percentile(value[component], 60))
                brown = (anther_hue < self.hue_range[0] / 2.0 - 2.0) and (anther_value < 110)
                freshness = 0.55 if brown else 1.0

                confidence = float(np.clip(
                    0.35 + 0.4 * min(component.sum() / 200.0, 1.0)
                    + (0.15 if have_baseline else 0.0), 0.0, 1.0))

        quality, clipped = regions.exposure_quality(patch, region.inner)
        # Blown highlights destroy both chroma and texture, so the measurement
        # is reported as less trustworthy rather than silently wrong.
        confidence *= float(np.clip(1.0 - clipped * 1.5, 0.1, 1.0))

        return {
            "area_fraction": area_score,
            "granularity": granularity,
            "chroma": chroma,
            "freshness": freshness,
            "confidence": confidence,
            "anther_box": anther_box,
        }
