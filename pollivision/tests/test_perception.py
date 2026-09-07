"""Perception tests: fusion, calibration, regions, tracking and the sex cue.

The sex tests use synthetic renders with exact ground truth. They verify
*logic*, not real-world accuracy: that the depth-based morphology cue fires on
an ovary and not on a pedicel, and - just as importantly - that it abstains
rather than guessing when it has no depth to work with.
"""

from __future__ import annotations

import numpy as np
import pytest

from pollivision.config import load_config
from pollivision.fusion import wbf
from pollivision.fusion.calibration import (
    LinearProbe,
    agreement_confidence,
    fuse_logits,
    to_logit,
    to_probability,
)
from pollivision.perception import regions
from pollivision.perception.detector import associate_ovaries
from pollivision.perception.occlusion import QualityGate
from pollivision.perception.sex import SexClassifier
from pollivision.tracking.ledger import PollinationLedger
from pollivision.tracking.tracker import FlowerTracker
from pollivision.types import (
    AnthesisEstimate,
    AnthesisStage,
    BBox,
    Detection,
    FlowerObservation,
    FlowerSex,
    PollenEstimate,
    SexEstimate,
)
from tests.synthetic import SyntheticFlower, render_scene


@pytest.fixture(scope="module")
def cfg():
    return load_config("default", ["species/pumpkin"])


# --------------------------------------------------------------------------- #


class TestBBox:
    def test_iou_of_identical_boxes(self):
        box = BBox(0, 0, 10, 10)
        assert box.iou(box) == pytest.approx(1.0)

    def test_disjoint_boxes_have_zero_iou(self):
        assert BBox(0, 0, 10, 10).iou(BBox(20, 20, 30, 30)) == 0.0

    def test_downward_bias_extends_the_crop_below(self):
        """Sex classification needs to see beneath the corolla."""
        box = BBox(100, 100, 200, 200)
        biased = box.scaled(2.0, 640, 480, bias_down=1.0)
        above = box.y1 - biased.y1
        below = biased.y2 - box.y2
        assert below > above


class TestCalibration:
    def test_logit_round_trip(self):
        for probability in (0.01, 0.3, 0.5, 0.87, 0.99):
            assert to_probability(to_logit(probability)) == pytest.approx(probability, abs=1e-6)

    def test_neutral_cues_do_not_move_the_posterior(self):
        posterior, _ = fuse_logits({"a": (0.5, 1.0), "b": (0.5, 5.0)}, prior=0.5)
        assert posterior == pytest.approx(0.5, abs=1e-6)

    def test_zero_weight_cue_is_ignored(self):
        posterior, breakdown = fuse_logits({"a": (0.99, 0.0)}, prior=0.5)
        assert posterior == pytest.approx(0.5, abs=1e-6)
        assert "a" not in breakdown.cues

    def test_agreeing_cues_reinforce(self):
        one, _ = fuse_logits({"a": (0.8, 1.0)}, prior=0.5)
        two, _ = fuse_logits({"a": (0.8, 1.0), "b": (0.8, 1.0)}, prior=0.5)
        assert two > one

    def test_conflicting_cues_cancel(self):
        posterior, breakdown = fuse_logits({"a": (0.9, 1.0), "b": (0.1, 1.0)}, prior=0.5)
        assert posterior == pytest.approx(0.5, abs=1e-6)
        assert agreement_confidence(breakdown) == pytest.approx(0.0, abs=1e-6)

    def test_agreement_is_one_when_cues_align(self):
        _, breakdown = fuse_logits({"a": (0.8, 1.0), "b": (0.7, 1.0)}, prior=0.5)
        assert agreement_confidence(breakdown) == pytest.approx(1.0, abs=1e-6)

    def test_linear_probe_learns_a_separable_split(self):
        rng = np.random.default_rng(0)
        positive = rng.normal(1.0, 0.25, (60, 8))
        negative = rng.normal(-1.0, 0.25, (60, 8))
        features = np.vstack([positive, negative]).astype(np.float32)
        labels = np.hstack([np.ones(60), np.zeros(60)])
        probe = LinearProbe.fit(features, labels, ["male", "female"])
        predictions = probe.predict_proba(features) > 0.5
        assert (predictions == labels.astype(bool)).mean() > 0.95


class TestBoxFusion:
    def test_single_detection_survives(self):
        detection = Detection(BBox(10, 10, 50, 50), 0.9, "flower", "yoloe")
        assert len(wbf.fuse([detection])) == 1

    def test_overlapping_detections_merge(self):
        fused = wbf.fuse([
            Detection(BBox(10, 10, 50, 50), 0.9, "flower", "yoloe"),
            Detection(BBox(12, 12, 52, 52), 0.8, "flower", "owlv2"),
        ], n_backends=2)
        assert len(fused) == 1
        assert set(fused[0].meta["sources"]) == {"yoloe", "owlv2"}

    def test_different_roles_never_merge(self):
        """A corolla and the ovary beneath it overlap by construction."""
        fused = wbf.fuse([
            Detection(BBox(10, 10, 50, 50), 0.9, "flower", "yoloe"),
            Detection(BBox(11, 11, 51, 51), 0.8, "ovary", "yoloe"),
        ])
        assert len(fused) == 2

    def test_distant_detections_stay_separate(self):
        fused = wbf.fuse([
            Detection(BBox(10, 10, 50, 50), 0.9, "flower", "yoloe"),
            Detection(BBox(200, 200, 240, 240), 0.8, "flower", "yoloe"),
        ])
        assert len(fused) == 2

    def test_vote_requirement_drops_unconfirmed_boxes(self):
        detections = [Detection(BBox(10, 10, 50, 50), 0.9, "flower", "yoloe")]
        assert wbf.fuse(detections, require_votes=2, n_backends=2) == []

    def test_fused_box_lies_between_its_members(self):
        fused = wbf.fuse([
            Detection(BBox(10, 10, 50, 50), 0.9, "flower", "a"),
            Detection(BBox(20, 20, 60, 60), 0.9, "flower", "b"),
        ], n_backends=2)
        assert 10 <= fused[0].box.x1 <= 20

    def test_mask_is_carried_from_the_best_member(self):
        mask = np.ones((100, 100), dtype=bool)
        fused = wbf.fuse([
            Detection(BBox(10, 10, 50, 50), 0.6, "flower", "a"),
            Detection(BBox(11, 11, 51, 51), 0.9, "flower", "b", mask=mask),
        ], n_backends=2)
        assert fused[0].mask is not None


class TestOvaryAssociation:
    def test_ovary_below_a_flower_is_matched(self):
        flowers = [Detection(BBox(100, 100, 200, 200), 0.9, "flower", "yoloe")]
        ovaries = [Detection(BBox(120, 205, 180, 260), 0.7, "ovary", "yoloe")]
        assert 0 in associate_ovaries(flowers, ovaries)

    def test_ovary_above_a_flower_is_rejected(self):
        """An inferior ovary is below the corolla, never above it."""
        flowers = [Detection(BBox(100, 100, 200, 200), 0.9, "flower", "yoloe")]
        ovaries = [Detection(BBox(120, 20, 180, 90), 0.7, "ovary", "yoloe")]
        assert associate_ovaries(flowers, ovaries) == {}

    def test_laterally_offset_ovary_is_rejected(self):
        flowers = [Detection(BBox(100, 100, 200, 200), 0.9, "flower", "yoloe")]
        ovaries = [Detection(BBox(400, 205, 460, 260), 0.7, "ovary", "yoloe")]
        assert associate_ovaries(flowers, ovaries) == {}

    def test_each_ovary_is_claimed_once(self):
        flowers = [Detection(BBox(100, 100, 200, 200), 0.9, "flower", "yoloe"),
                   Detection(BBox(105, 105, 205, 205), 0.8, "flower", "yoloe")]
        ovaries = [Detection(BBox(120, 210, 180, 265), 0.7, "ovary", "yoloe")]
        assert len(associate_ovaries(flowers, ovaries)) == 1


class TestRegions:
    def test_ellipse_recovers_known_axes(self):
        import cv2

        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.ellipse(mask, (100, 100), (60, 30), 0, 0, 360, 1, -1)
        ellipse = regions.fit_ellipse(mask.astype(bool))
        assert ellipse is not None
        assert ellipse.major == pytest.approx(120, rel=0.08)
        assert ellipse.minor == pytest.approx(60, rel=0.08)
        assert ellipse.axis_ratio == pytest.approx(0.5, abs=0.05)

    def test_solidity_separates_convex_from_concave(self):
        import cv2

        disc = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(disc, (50, 50), 30, 1, -1)
        assert regions.solidity(disc.astype(bool)) > 0.95

        crescent = disc.copy()
        cv2.circle(crescent, (65, 50), 22, 0, -1)
        assert regions.solidity(crescent.astype(bool)) < 0.9

    def test_circularity_of_a_disc_is_near_one(self):
        import cv2

        disc = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(disc, (50, 50), 30, 1, -1)
        assert regions.circularity(disc.astype(bool)) > 0.85

    def test_largest_component_removes_speckle(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:60, 10:60] = True
        mask[90:95, 90:95] = True
        assert regions.largest_component(mask).sum() == 50 * 50

    def test_corolla_regions_are_disjoint(self):
        result = regions.corolla_regions((100, 100), None, 0.45, 0.80)
        assert not (result.inner & result.ring).any()
        assert result.inner.sum() > 0 and result.ring.sum() > 0

    def test_hue_mask_handles_wraparound(self):
        hsv = np.zeros((10, 10, 3), dtype=np.uint8)
        hsv[..., 0] = 2      # ~4 degrees, inside a 350->20 wrapped range
        hsv[..., 1] = 200
        hsv[..., 2] = 200
        assert regions.hue_mask(hsv, (350, 20), 50, 50).all()

    def test_sharpness_ranks_blur_correctly(self):
        import cv2

        rng = np.random.default_rng(0)
        sharp = (rng.random((100, 100, 3)) * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(sharp, (21, 21), 0)
        assert regions.sharpness(sharp) > regions.sharpness(blurred)

    def test_exposure_detects_clipping(self):
        blown = np.full((50, 50, 3), 255, dtype=np.uint8)
        quality, clipped = regions.exposure_quality(blown)
        assert clipped > 0.9 and quality < 0.1


class TestQualityGate:
    def test_frame_edge_truncation_is_rejected(self, cfg):
        gate = QualityGate(cfg)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        assessment = gate.assess(frame, BBox(-60, 100, 60, 220), 0.9)
        assert not assessment.accepted

    def test_tiny_detections_are_rejected(self, cfg):
        gate = QualityGate(cfg)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        assert not gate.assess(frame, BBox(100, 100, 110, 110), 0.9).accepted

    def test_low_confidence_is_rejected(self, cfg):
        gate = QualityGate(cfg)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        assert not gate.assess(frame, BBox(100, 100, 200, 200), 0.01).accepted


class TestTracker:
    def _flower(self, x, y, p_female=0.9, size=40) -> FlowerObservation:
        flower = FlowerObservation(box=BBox(x, y, x + size, y + size), score=0.9)
        flower.sex = SexEstimate(sex=FlowerSex.FEMALE, p_female=p_female, confidence=0.7)
        flower.anthesis = AnthesisEstimate(stage=AnthesisStage.RECEPTIVE, confidence=0.8)
        flower.pollen = PollenEstimate(availability=0.5, confidence=0.7)
        return flower

    def test_identity_is_preserved_across_motion(self, cfg):
        tracker = FlowerTracker(cfg)
        ids = []
        for step in range(8):
            flowers = tracker.update([self._flower(100 + step * 9, 100)])
            ids.append(flowers[0].track_id)
        assert len(set(ids)) == 1

    def test_distinct_flowers_get_distinct_ids(self, cfg):
        tracker = FlowerTracker(cfg)
        flowers = tracker.update([self._flower(100, 100), self._flower(400, 300)])
        assert flowers[0].track_id != flowers[1].track_id

    def test_smoothing_suppresses_a_single_noisy_frame(self, cfg):
        """One bad frame must not flip a well-established classification."""
        tracker = FlowerTracker(cfg)
        for _ in range(6):
            tracker.update([self._flower(100, 100, p_female=0.95)])
        flowers = tracker.update([self._flower(100, 100, p_female=0.02)])
        assert flowers[0].sex.sex is FlowerSex.FEMALE
        assert flowers[0].sex.p_female > 0.5

    def test_stale_tracks_are_retired(self, cfg):
        tracker = FlowerTracker(cfg)
        tracker.update([self._flower(100, 100)])
        for _ in range(int(cfg.get("tracking.max_age", 15)) + 3):
            tracker.update([])
        assert tracker.tracks == {}


class TestLedger:
    def test_pollinated_flowers_are_skipped(self, cfg):
        ledger = PollinationLedger(cfg)
        ledger.record_attempt(1, FlowerSex.FEMALE)
        ledger.record_outcome(1, True)
        skip, reason = ledger.should_skip(1)
        assert skip and "pollinated" in reason

    def test_attempts_are_capped(self, cfg):
        ledger = PollinationLedger(cfg)
        for _ in range(int(cfg.get("ledger.max_attempts", 3))):
            ledger.record_attempt(2, FlowerSex.FEMALE)
            ledger.record_outcome(2, False)
        skip, reason = ledger.should_skip(2)
        assert skip and "exhausted" in reason

    def test_unknown_flowers_are_not_skipped(self, cfg):
        assert PollinationLedger(cfg).should_skip(999)[0] is False

    def test_flower_is_reacquired_by_position_after_track_loss(self, cfg):
        """A dropped and remade track must not defeat the pollination memory."""
        ledger = PollinationLedger(cfg)
        position = np.array([0.5, 0.1, 0.3])
        ledger.record_attempt(1, FlowerSex.FEMALE, position)
        ledger.record_outcome(1, True)
        skip, _ = ledger.should_skip(77, position + 0.01)
        assert skip

    def test_distant_flower_is_not_confused_with_a_serviced_one(self, cfg):
        ledger = PollinationLedger(cfg)
        ledger.record_attempt(1, FlowerSex.FEMALE, np.array([0.5, 0.1, 0.3]))
        ledger.record_outcome(1, True)
        assert ledger.should_skip(77, np.array([0.9, 0.4, 0.3]))[0] is False

    def test_probe_charge_is_spent_by_deposition(self, cfg):
        ledger = PollinationLedger(cfg)
        ledger.load_probe(1, 1.0)
        ledger.consume_probe(0.25)
        assert ledger.probe_charge == pytest.approx(0.75)

    def test_probe_source_clears_when_exhausted(self, cfg):
        ledger = PollinationLedger(cfg)
        ledger.load_probe(1, 0.2)
        ledger.consume_probe(0.5)
        assert ledger.probe_charge == 0.0 and ledger.pollen_source is None


class TestSexMorphology:
    """The depth-based ovary cue, on renders with exact ground truth."""

    def _score(self, cfg, sex, radius=64, use_depth=True, seed=5):
        flower = SyntheticFlower(cx=320, cy=200, radius=radius, sex=sex)
        scene = render_scene([flower], seed=seed, with_depth=True)
        classifier = SexClassifier(cfg, vlm=None)
        return classifier._score_morphology(
            scene.image, BBox(*[float(v) for v in flower.box]), None,
            scene.depth if use_depth else None,
        )

    @pytest.mark.parametrize("radius", [42, 64, 86])
    def test_ovary_is_evidence_of_a_female(self, cfg, radius):
        probability, ovary_box = self._score(cfg, "female", radius)
        assert probability > 0.5
        assert ovary_box is not None

    @pytest.mark.parametrize("radius", [42, 64, 86])
    def test_bare_pedicel_is_evidence_of_a_male(self, cfg, radius):
        probability, ovary_box = self._score(cfg, "male", radius)
        assert probability < 0.5
        assert ovary_box is None

    def test_cue_abstains_without_depth(self, cfg):
        """Monocular colour cannot separate a green ovary from green foliage.

        The cue must return exactly neutral rather than guessing, so that a
        measurement it cannot make does not outvote the cues that worked.
        """
        for sex in ("male", "female"):
            probability, ovary_box = self._score(cfg, sex, use_depth=False)
            assert probability == 0.5
            assert ovary_box is None

    def test_depth_cue_separates_the_sexes_across_a_sweep(self, cfg):
        classifier = SexClassifier(cfg, vlm=None)
        correct = 0
        for index in range(10):
            sex = "male" if index % 2 == 0 else "female"
            flower = SyntheticFlower(cx=150 + (index % 4) * 130,
                                     cy=140 + (index % 3) * 50,
                                     radius=42 + (index % 5) * 11, sex=sex)
            scene = render_scene([flower], seed=30 + index, with_depth=True)
            probability, _ = classifier._score_morphology(
                scene.image, BBox(*[float(v) for v in flower.box]), None, scene.depth)
            correct += (probability > 0.5) == (sex == "female")
        assert correct == 10
