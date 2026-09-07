"""Flower sex classification (report Sec. III-C).

Cucurbits are monoecious, and only a male-to-female transfer produces fruit, so
getting this wrong does not merely waste an actuation - it can waste the whole
visit. The report identifies the decisive cue explicitly: pistillate (female)
flowers carry an *inferior ovary*, a miniature fruit sitting directly beneath
the corolla, which staminate (male) flowers lack entirely; a male flower instead
sits on a long, thin, bare pedicel.

Rather than trusting one model, this head fuses up to four independent cues in
log-odds space:

1. **Morphology** - an explicit geometric search beneath the corolla for a
   convex green blob of plausible size. This is the physically correct cue, it
   needs no learned weights at all, and it fails in ways that are easy to
   understand and tune.
2. **Vision-language scoring** - a zero-shot prompt comparison on a downward-
   biased context crop, which captures overall gestalt the geometry misses.
3. **Detector association** - an ovary detected as its own object and matched to
   this flower by the open-vocabulary detector.
4. **Linear probe** - an optional trained head over VLM embeddings, for users
   who label a few dozen crops of their own cultivar.

Cue 1 is weighted highest because it is the one grounded in the plant's actual
morphology; the others corroborate it. Any subset can be missing and the head
still produces a calibrated posterior over the cues that remain.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..fusion.calibration import (
    LinearProbe,
    agreement_confidence,
    confidence_from_logit,
    fuse_logits,
    temperature_scale,
)
from ..logging_utils import get_logger
from ..types import BBox, Detection, FlowerSex, SexEstimate
from . import regions

LOGGER = get_logger(__name__)

# Zero-shot softmaxes are systematically overconfident; soften before fusing.
_VLM_TEMPERATURE = 2.5


class SexClassifier:
    """Fuses morphological, vision-language and detector evidence."""

    def __init__(self, cfg, vlm=None) -> None:
        self.cfg = cfg
        self.vlm = vlm
        self.enabled = bool(cfg.get("perception.sex.enabled", True))

        weights = cfg.section("perception.sex.weights")
        self.w_vlm = float(weights.get("vlm", 1.0))
        self.w_morphology = float(weights.get("morphology", 1.4))
        self.w_detector = float(weights.get("detector", 1.1))
        self.w_probe = float(weights.get("probe", 1.6))

        self.threshold = float(cfg.get("perception.sex.decision_threshold", 0.5))
        self.min_confidence = float(cfg.get("perception.sex.min_confidence", 0.25))

        prompts = cfg.section("perception.sex.prompts")
        self.prompt_groups = {
            "female": list(prompts.get("female", [])),
            "male": list(prompts.get("male", [])),
        }

        morphology = cfg.section("perception.sex.morphology")
        self.search_scale = float(morphology.get("search_scale", 1.9))
        self.min_width_ratio = float(morphology.get("min_width_ratio", 0.22))
        self.max_width_ratio = float(morphology.get("max_width_ratio", 0.95))
        self.min_solidity = float(morphology.get("min_solidity", 0.72))
        self.min_green_fraction = float(morphology.get("min_green_fraction", 0.35))
        self.hue_range = tuple(morphology.get("hue_range", [30, 95]))
        self.min_saturation = float(morphology.get("min_saturation", 45))
        self.min_value = float(morphology.get("min_value", 30))
        # Depth half-window used to separate the ovary from the canopy behind
        # it. Scaled from the species' ovary length, since a structure deeper
        # than the ovary itself is by definition not the ovary.
        self.ovary_depth_window_m = float(
            morphology.get("depth_window_m", cfg.get("species.ovary_length_m", 0.05)))

        self.probe: Optional[LinearProbe] = None
        probe_path = cfg.get("perception.sex.probe_path")
        if probe_path:
            try:
                self.probe = LinearProbe.load(probe_path)
                LOGGER.info("Loaded sex linear probe from %s", probe_path)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Could not load sex probe from %s: %s", probe_path, exc)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def classify(
        self,
        frame: np.ndarray,
        boxes: list[BBox],
        masks: list[Optional[np.ndarray]],
        ovary_matches: Optional[dict[int, tuple[BBox, float]]] = None,
        sex_hints: Optional[dict[int, str]] = None,
        depth: Optional[np.ndarray] = None,
    ) -> list[SexEstimate]:
        """Classify a batch of flowers.

        Batched because the vision-language model amortises much better over a
        batch of crops than over one crop at a time.
        """
        if not boxes:
            return []
        if not self.enabled:
            return [SexEstimate() for _ in boxes]

        ovary_matches = ovary_matches or {}
        sex_hints = sex_hints or {}

        vlm_scores = self._score_vlm(frame, boxes)
        probe_scores = self._score_probe(frame, boxes)

        estimates: list[SexEstimate] = []
        for index, box in enumerate(boxes):
            evidence: dict[str, tuple[float, float]] = {}

            morphology_p, ovary_box = self._score_morphology(
                frame, box, masks[index] if index < len(masks) else None, depth
            )
            evidence["morphology"] = (morphology_p, self.w_morphology)

            if vlm_scores is not None:
                evidence["vlm"] = (
                    temperature_scale(vlm_scores[index], _VLM_TEMPERATURE),
                    self.w_vlm,
                )

            if index in ovary_matches:
                _, association = ovary_matches[index]
                # A confident association is strong evidence for female; a weak
                # one should barely move the posterior, hence the 0.5 floor.
                evidence["detector"] = (0.5 + 0.45 * float(np.clip(association, 0.0, 1.0)),
                                        self.w_detector)
                if ovary_box is None:
                    ovary_box = ovary_matches[index][0]

            if probe_scores is not None:
                evidence["probe"] = (float(probe_scores[index]), self.w_probe)

            if index in sex_hints:
                # A supervised detector that predicts sex directly is treated as
                # a strong but not infallible cue.
                hint = 0.85 if sex_hints[index] == "female" else 0.15
                evidence["detector_class"] = (hint, self.w_detector)

            p_female, breakdown = fuse_logits(evidence, prior=0.5)
            confidence = min(
                confidence_from_logit(breakdown.fused_logit, scale=2.5),
                max(agreement_confidence(breakdown), 0.15),
            )

            if confidence < self.min_confidence:
                sex = FlowerSex.UNKNOWN
            elif p_female >= self.threshold:
                sex = FlowerSex.FEMALE
            else:
                sex = FlowerSex.MALE

            estimates.append(SexEstimate(
                sex=sex,
                p_female=float(p_female),
                confidence=float(confidence),
                breakdown=breakdown,
                ovary_box=ovary_box,
            ))
        return estimates

    # ------------------------------------------------------------------ #
    # Individual cues
    # ------------------------------------------------------------------ #

    def _score_vlm(self, frame: np.ndarray, boxes: list[BBox]) -> Optional[np.ndarray]:
        if self.vlm is None or not self.prompt_groups["female"]:
            return None
        scale = float(self.cfg.get("vlm.crop_scale", 1.6))
        bias = float(self.cfg.get("vlm.crop_bias_down", 0.55))
        crops = [regions.context_crop(frame, box, scale, bias) for box in boxes]
        try:
            scored = self.vlm.score_groups(crops, self.prompt_groups)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Vision-language sex scoring failed: %s", exc)
            return None
        return np.array([row.get("female", 0.5) for row in scored], dtype=np.float32)

    def _score_probe(self, frame: np.ndarray, boxes: list[BBox]) -> Optional[np.ndarray]:
        if self.probe is None or self.vlm is None:
            return None
        scale = float(self.cfg.get("vlm.crop_scale", 1.6))
        bias = float(self.cfg.get("vlm.crop_bias_down", 0.55))
        crops = [regions.context_crop(frame, box, scale, bias) for box in boxes]
        try:
            features = self.vlm.encode_images(crops).cpu().numpy()
            return self.probe.predict_proba(features)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Sex probe scoring failed: %s", exc)
            return None

    def _score_morphology(
        self,
        frame: np.ndarray,
        box: BBox,
        mask: Optional[np.ndarray],
        depth: Optional[np.ndarray] = None,
    ) -> tuple[float, Optional[BBox]]:
        """Search beneath the corolla for an inferior ovary.

        Returns ``(p_female, ovary_box)``, where 0.5 means "no usable evidence".

        What separates the sexes here is the **width profile of the structure
        attached to the corolla base**: a staminate flower hangs on a pedicel,
        thin and near-constant in width, while a pistillate flower carries an
        ovary that bulges to a substantial fraction of the corolla width before
        narrowing into the stem.

        Measuring that profile requires isolating the structure from the canopy
        behind it, and this cue is deliberately restricted to the cases where
        that isolation is actually achievable:

        * **With depth** (the report's RGB-D baseline, Sec. III-B) the ovary sits
          at essentially the flower's own range while the canopy behind it does
          not, so a depth window around the corolla segments it cleanly. This is
          the path this cue is designed for.
        * **With a segmented ovary instance** the work is already done, and the
          detector-association cue covers it - so morphology stands aside rather
          than double-counting the same evidence.

        On a bare monocular camera with neither, this cue returns *neutral*. That
        is a deliberate limitation, not an oversight: an immature cucurbit fruit
        is green foliage-coloured against green foliage, and colour-threshold and
        edge-growing segmentations were both tried and both merged the ovary into
        the background. A cue that fires unreliably is worse than one that
        abstains, because the fusion cannot tell a confident mistake from a
        confident success. In that configuration the sex decision rests on the
        vision-language cue and the detector's ovary class, and adding a depth
        channel is the single highest-value upgrade to sex accuracy.
        """
        if depth is None:
            return 0.5, None

        height, width = frame.shape[:2]
        if depth.shape[:2] != (height, width):
            return 0.5, None

        band_height = box.height * (self.search_scale - 1.0)
        if band_height < 6 or box.width < 8:
            return 0.5, None

        # Reference range: the corolla's own depth, taken robustly.
        corolla = self._corolla_depth(depth, box, mask)
        if corolla is None:
            return 0.5, None

        half_width = box.width * 1.1
        x1 = int(np.clip(box.cx - half_width, 0, width - 1))
        x2 = int(np.clip(box.cx + half_width, 0, width))
        # Strictly below the corolla: a band overlapping the petals measures the
        # flower rather than the structure beneath it.
        y1 = int(np.clip(box.y2 + max(2.0, box.height * 0.02), 0, height - 1))
        y2 = int(np.clip(box.y2 + band_height, 0, height))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return 0.5, None

        band_depth = depth[y1:y2, x1:x2]
        band_h, band_w = band_depth.shape[:2]
        centre_x = int(np.clip(box.cx - x1, 0, band_w - 1))

        # Foreground = whatever sits at the flower's own range. The window scales
        # with the corolla's physical size so it works for a 3 cm cucumber
        # blossom and a 10 cm pumpkin one alike.
        tolerance = max(0.02, 0.6 * self.ovary_depth_window_m)
        finite = np.isfinite(band_depth)
        region = finite & (np.abs(band_depth - corolla) <= tolerance)

        if mask is not None and mask.shape[:2] == (height, width):
            region &= ~mask[y1:y2, x1:x2]

        region = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_OPEN,
                                  np.ones((3, 3), np.uint8)).astype(bool)
        if region.sum() < 24:
            # Nothing at the flower's range below it: a thin pedicel can fall
            # below the depth sensor's resolution, so this stays neutral rather
            # than being read as evidence of a male.
            return 0.5, None

        profile = self._width_profile(region, centre_x)
        if profile is None:
            return 0.5, None

        widths, region_mask = profile
        ratios = widths / max(box.width, 1e-6)

        # Skip the topmost rows: they sit in the shadow of the petals, where the
        # corolla boundary itself distorts the measured width.
        skip = max(1, int(0.12 * len(ratios)))
        usable = ratios[skip:]
        usable = usable[usable > 0.0]
        if usable.size < 3:
            return 0.5, None

        peak = float(np.percentile(usable, 90))
        typical = float(np.median(usable))
        coverage = float(usable.size / max(len(ratios) - skip, 1))

        if peak > self.max_width_ratio * 1.8:
            # Too wide to be an ovary: the depth window has caught a leaf lying
            # at the same range as the flower.
            return 0.5, None

        if peak < self.min_width_ratio:
            # A consistently thin stalk across a good span of the band is a
            # pedicel, and that is real evidence of a staminate flower.
            if coverage < 0.35:
                return 0.5, None
            return float(np.clip(0.5 - 0.32 * coverage, 0.18, 0.5)), None

        ys, xs = np.nonzero(region_mask)
        ovary_box = BBox(float(x1 + xs.min()), float(y1 + ys.min()),
                         float(x1 + xs.max() + 1), float(y1 + ys.max() + 1))

        # Three things make a width profile look like an ovary.
        # 1. The peak reaches a plausible fraction of the corolla width.
        span = max(self.max_width_ratio - self.min_width_ratio, 1e-6)
        size_score = float(np.clip((peak - self.min_width_ratio) / span, 0.0, 1.0))

        # 2. It *bulges*: wider at its middle than a stem is anywhere. A
        #    constant-width region scores nothing here, which is what rules out
        #    a pedicel that happens to be thick.
        bulge = float(np.clip((peak - typical) / max(peak, 1e-6), 0.0, 1.0))
        bulge_score = float(np.clip(bulge / 0.45, 0.0, 1.0))

        # 3. It is convex and compact, as fruit are and leaf clusters are not.
        component_solidity = regions.solidity(region_mask)
        solidity_score = float(np.clip(
            (component_solidity - self.min_solidity) / max(1.0 - self.min_solidity, 1e-6),
            0.0, 1.0))

        score = 0.45 * size_score + 0.30 * bulge_score + 0.25 * solidity_score
        score *= float(np.clip(0.4 + 0.6 * coverage, 0.0, 1.0))

        # Ceiling below 1: a convincing blob is strong evidence, never proof.
        p_female = float(np.clip(0.5 + 0.42 * score, 0.5, 0.92))
        return p_female, ovary_box

    @staticmethod
    def _corolla_depth(depth: np.ndarray, box: BBox,
                       mask: Optional[np.ndarray]) -> Optional[float]:
        """Robust range to the corolla, used as the depth-window reference."""
        x1, y1, x2, y2 = box.as_int()
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(depth.shape[1], max(x2, x1 + 1))
        y2 = min(depth.shape[0], max(y2, y1 + 1))
        patch = depth[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        if mask is not None and mask.shape[:2] == depth.shape[:2]:
            local = mask[y1:y2, x1:x2]
            if local.shape == patch.shape and local.sum() > 16:
                patch = patch[local]
        values = patch[np.isfinite(patch)]
        values = values[values > 1e-3]
        if values.size < 8:
            return None
        return float(np.median(values))

    @staticmethod
    def _width_profile(region: np.ndarray,
                       centre_x: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Per-row width of the run containing the flower's vertical axis.

        Measuring only the *contiguous run through the centre* rather than the
        total foreground per row is what keeps an unrelated leaf sitting beside
        the stem from inflating the width and faking an ovary.
        """
        band_h, band_w = region.shape[:2]
        widths = np.zeros(band_h, dtype=np.float32)
        kept = np.zeros_like(region)

        cursor = int(np.clip(centre_x, 0, band_w - 1))
        for row in range(band_h):
            line = region[row]
            if not line.any():
                continue
            # Track the run through the previous row's centre, so the profile
            # follows a stem that leans rather than jumping to another object.
            if not line[cursor]:
                candidates = np.nonzero(line)[0]
                nearest = candidates[np.argmin(np.abs(candidates - cursor))]
                if abs(int(nearest) - cursor) > band_w * 0.30:
                    continue
                cursor = int(nearest)

            left = cursor
            while left > 0 and line[left - 1]:
                left -= 1
            right = cursor
            while right < band_w - 1 and line[right + 1]:
                right += 1

            widths[row] = right - left + 1
            kept[row, left:right + 1] = True
            cursor = (left + right) // 2

        if not kept.any() or float(widths.max()) < 2:
            return None
        return widths, kept
