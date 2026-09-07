"""Stigma localisation on pistillate flowers.

Deposition needs a more precise aiming point than "the middle of the flower".
The receptive surface of a pistillate cucurbit flower is the stigma - a short,
lobed, sticky structure at the centre of the corolla - and the electrostatic
field falls off steeply with distance, so aiming at the corolla centroid rather
than the stigma wastes a large part of the deposited dose on petal tissue.

The stigma is found as the compact, saturated, non-petal structure occupying the
inner disc. It is the same family of measurement as the anther search in
``pollen.py``, and deliberately so: on a pistillate flower the inner disc holds
a stigma instead of an anther, and the discriminating work has already been done
by the sex head.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..types import BBox, Ellipse
from . import regions


class StigmaLocator:
    """Finds the deposition aiming point inside a female corolla."""

    def __init__(self, cfg) -> None:
        section = cfg.section("perception.stigma")
        self.enabled = bool(section.get("enabled", True))
        self.inner_ratio = float(section.get("inner_radius_ratio", 0.40))
        self.hue_range = tuple(section.get("hue_range", [18, 45]))
        self.min_saturation = float(section.get("min_saturation", 90))

    def locate(
        self,
        frame: np.ndarray,
        box: BBox,
        mask: Optional[np.ndarray] = None,
        ellipse: Optional[Ellipse] = None,
    ) -> tuple[Optional[BBox], tuple[float, float], float]:
        """Return ``(stigma_box, aim_point_px, confidence)``.

        The aim point always falls back to the corolla centre, so a caller can
        use the returned point unconditionally; the confidence says how much
        better than that fallback it is.
        """
        fallback = (ellipse.cx, ellipse.cy) if ellipse is not None else box.center
        if not self.enabled:
            return None, fallback, 0.0

        patch = regions.crop(frame, box)
        if patch.size == 0 or patch.shape[0] < 6 or patch.shape[1] < 6:
            return None, fallback, 0.0

        offset = (max(0.0, np.floor(box.x1)), max(0.0, np.floor(box.y1)))
        local_mask = None
        if mask is not None and mask.shape[:2] == frame.shape[:2]:
            candidate = regions.mask_in_box(mask, box)
            if candidate is not None and candidate.shape[:2] == patch.shape[:2]:
                local_mask = candidate

        region = regions.corolla_regions(
            patch.shape[:2], ellipse=ellipse, inner_ratio=self.inner_ratio,
            ring_ratio=0.80, offset=offset, mask=local_mask,
        )
        if region.inner.sum() < 20:
            return None, fallback, 0.0

        hsv = regions.hsv_of(patch)
        saturation = hsv[..., 1]

        if region.ring.sum() >= 24:
            petal_saturation = float(np.percentile(saturation[region.ring], 60))
        else:
            petal_saturation = self.min_saturation

        gate = regions.hue_mask(hsv, self.hue_range, 0, 60)
        threshold = max(self.min_saturation, petal_saturation + 15)
        candidate_mask = gate & (saturation >= threshold) & region.inner

        if candidate_mask.sum() < 12:
            return None, fallback, 0.0

        cleaned = cv2.morphologyEx(candidate_mask.astype(np.uint8), cv2.MORPH_OPEN,
                                   np.ones((3, 3), np.uint8))
        component = regions.largest_component(cleaned)
        if component.sum() < 8:
            return None, fallback, 0.0

        ys, xs = np.nonzero(component)
        stigma_box = BBox(
            float(offset[0] + xs.min()), float(offset[1] + ys.min()),
            float(offset[0] + xs.max() + 1), float(offset[1] + ys.max() + 1),
        )
        aim = (float(offset[0] + xs.mean()), float(offset[1] + ys.mean()))

        # A credible stigma is small relative to the corolla and near its centre;
        # a large blob is more likely a mis-segmented petal highlight.
        size_ratio = component.sum() / max(region.inner.sum(), 1)
        plausible_size = float(np.clip(1.0 - abs(size_ratio - 0.25) / 0.45, 0.0, 1.0))
        offset_px = np.hypot(aim[0] - offset[0] - region.centre[0],
                             aim[1] - offset[1] - region.centre[1])
        centrality = float(np.clip(1.0 - offset_px / max(region.radius, 1e-6), 0.0, 1.0))
        confidence = float(np.clip(0.5 * plausible_size + 0.5 * centrality, 0.0, 1.0))

        return stigma_box, aim, confidence


def stigma_pollen_load(frame: np.ndarray, stigma_box: Optional[BBox],
                       hue_range: tuple[float, float] = (14, 45),
                       min_saturation: float = 110) -> float:
    """Fraction of a stigma region carrying pollen-coloured material.

    Used by the verification stage: a successful deposition leaves visible
    yellow grains on the stigma, so this value should rise across a transfer.
    """
    if stigma_box is None:
        return 0.0
    patch = regions.crop(frame, stigma_box, pad=2)
    if patch.size == 0:
        return 0.0
    hsv = regions.hsv_of(patch)
    gate = regions.hue_mask(hsv, hue_range, min_saturation, 90)
    return float(gate.mean()) if gate.size else 0.0
