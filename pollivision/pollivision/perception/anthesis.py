"""Anthesis staging: is this flower actually receptive right now?

Cucurbit flowers are anthetic for a single morning - typically opening around
dawn and closing by midday, with stigma receptivity and pollen viability both
falling away over a few hours. A rover that treats every detected flower as a
target therefore spends most of its energy budget on buds that are not yet open
and on spent flowers that can no longer set fruit.

Staging combines a zero-shot vision-language judgement with a geometric check.
The geometry is the useful corrective: a fully open corolla is large, round and
convex, while buds are small and elongated and senescent flowers are collapsed
and ragged. When the two disagree the confidence drops, and a low-confidence
stage is treated as not-receptive by the quality gate - the conservative
direction, since a missed flower costs one visit but a mistimed pollination
costs the fruit.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..logging_utils import get_logger
from ..types import AnthesisEstimate, AnthesisStage, BBox
from . import regions

LOGGER = get_logger(__name__)

_STAGES = ("bud", "opening", "receptive", "senescent")


class AnthesisClassifier:
    """Stages flowers as bud / opening / receptive / senescent."""

    def __init__(self, cfg, vlm=None) -> None:
        self.cfg = cfg
        self.vlm = vlm
        self.enabled = bool(cfg.get("perception.anthesis.enabled", True))

        prompts = cfg.section("perception.anthesis.prompts")
        self.prompt_groups = {
            stage: list(prompts.get(stage, [])) for stage in _STAGES
        }
        self.prompt_groups = {k: v for k, v in self.prompt_groups.items() if v}

        self.min_circularity = float(
            cfg.get("perception.anthesis.min_circularity_receptive", 0.55))
        self.min_saturation = float(
            cfg.get("perception.anthesis.min_saturation_fresh", 60))
        self.vlm_weight = float(cfg.get("perception.anthesis.vlm_weight", 1.0))
        self.geometry_weight = float(cfg.get("perception.anthesis.geometry_weight", 0.6))

    def classify(
        self,
        frame: np.ndarray,
        boxes: list[BBox],
        masks: list[Optional[np.ndarray]],
        labels: Optional[list[str]] = None,
    ) -> list[AnthesisEstimate]:
        if not boxes:
            return []
        if not self.enabled:
            return [AnthesisEstimate(stage=AnthesisStage.RECEPTIVE, confidence=0.0)
                    for _ in boxes]

        vlm_probabilities = self._score_vlm(frame, boxes)

        estimates: list[AnthesisEstimate] = []
        for index, box in enumerate(boxes):
            geometry = self._score_geometry(
                frame, box, masks[index] if index < len(masks) else None
            )

            combined = {stage: 0.0 for stage in _STAGES}
            total_weight = 0.0
            if vlm_probabilities is not None:
                for stage in _STAGES:
                    combined[stage] += self.vlm_weight * vlm_probabilities[index].get(stage, 0.0)
                total_weight += self.vlm_weight
            for stage in _STAGES:
                combined[stage] += self.geometry_weight * geometry.get(stage, 0.0)
            total_weight += self.geometry_weight

            if total_weight > 0:
                combined = {k: v / total_weight for k, v in combined.items()}

            # A detector that explicitly called this a bud overrides the rest:
            # it is a direct observation rather than an inference.
            if labels is not None and index < len(labels) and labels[index] == "bud":
                combined = {"bud": 0.8, "opening": 0.15, "receptive": 0.04, "senescent": 0.01}

            best = max(combined, key=combined.get)
            ordered = sorted(combined.values(), reverse=True)
            # Confidence is the margin over the runner-up: a near-tie between
            # "receptive" and "senescent" should not read as a confident call.
            margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)

            estimates.append(AnthesisEstimate(
                stage=AnthesisStage(best),
                probabilities={k: float(v) for k, v in combined.items()},
                confidence=float(np.clip(margin * 2.0, 0.0, 1.0)),
            ))
        return estimates

    def _score_vlm(self, frame: np.ndarray, boxes: list[BBox]):
        if self.vlm is None or not self.prompt_groups:
            return None
        # A tight crop here, unlike the sex head: staging is about the corolla
        # itself, and surrounding foliage only adds noise.
        crops = [regions.crop(frame, box.scaled(1.15, frame.shape[1], frame.shape[0]))
                 for box in boxes]
        try:
            return self.vlm.score_groups(crops, self.prompt_groups)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Vision-language anthesis scoring failed: %s", exc)
            return None

    def _score_geometry(self, frame: np.ndarray, box: BBox,
                        mask: Optional[np.ndarray]) -> dict[str, float]:
        """Shape and colour evidence for each stage.

        Two measurements carry the signal: how round and convex the corolla
        outline is (open flowers are discs, buds are cones, spent flowers are
        crumpled), and how saturated the petals still are (senescence drains
        chroma toward brown).
        """
        patch = regions.crop(frame, box)
        if patch.size == 0:
            return {stage: 0.25 for stage in _STAGES}

        local_mask = None
        if mask is not None and mask.shape[:2] == frame.shape[:2]:
            local_mask = regions.mask_in_box(mask, box)
            if local_mask is not None and local_mask.shape[:2] != patch.shape[:2]:
                local_mask = None

        if local_mask is not None and local_mask.sum() > 32:
            shape_mask = regions.largest_component(local_mask)
            round_score = regions.circularity(shape_mask)
            convexity = regions.solidity(shape_mask)
        else:
            # Without a mask, fall back on the box aspect ratio: an open corolla
            # is close to square in projection, a bud is tall and narrow.
            aspect = min(box.width, box.height) / max(max(box.width, box.height), 1e-6)
            round_score, convexity = aspect, aspect

        hsv = regions.hsv_of(patch)
        saturation = regions.robust_mean(hsv[..., 1], local_mask, percentile=60)
        value = regions.robust_mean(hsv[..., 2], local_mask, percentile=60)
        freshness = float(np.clip(saturation / max(self.min_saturation * 1.6, 1.0), 0.0, 1.0))
        brightness = float(np.clip(value / 180.0, 0.0, 1.0))

        openness = float(np.clip((round_score - 0.25) / 0.55, 0.0, 1.0))
        integrity = float(np.clip((convexity - 0.55) / 0.35, 0.0, 1.0))

        scores = {
            "receptive": openness * integrity * (0.4 + 0.6 * freshness) * (0.5 + 0.5 * brightness),
            "opening": openness * (1.0 - integrity) + 0.25 * (1.0 - openness) * freshness,
            "bud": (1.0 - openness) * freshness,
            # Senescence shows as an open-ish but ragged, desaturated corolla.
            "senescent": (1.0 - freshness) * (0.4 + 0.6 * (1.0 - integrity)),
        }
        total = sum(scores.values())
        if total <= 1e-6:
            return {stage: 0.25 for stage in _STAGES}
        return {k: v / total for k, v in scores.items()}
