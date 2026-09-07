"""End-to-end tests: config, planning and the closed mission loop.

The mission tests drive :class:`MissionController` with hand-built perception
results rather than real inference, so they run in milliseconds and assert on
the *decision logic* - collect before deposit, never re-pollinate, always verify
- independently of how good the detector happens to be.
"""

from __future__ import annotations

import numpy as np
import pytest

from pollivision.config import Config, load_config, parse_overrides
from pollivision.planning.target import TargetSelector
from pollivision.runtime.mission import MissionController, MissionState
from pollivision.types import (
    AnthesisEstimate,
    AnthesisStage,
    BBox,
    EnvironmentReading,
    FlowerObservation,
    FlowerSex,
    FrameResult,
    PollenEstimate,
    Pose3D,
    ProbeMode,
    QualityAssessment,
    SexEstimate,
)


@pytest.fixture
def cfg():
    return load_config("default", ["species/pumpkin"])


def make_flower(sex: FlowerSex, track_id: int, range_m: float = 0.3,
                pollen: float = 0.8, x: float = 300.0) -> FlowerObservation:
    """A fully-populated, actionable observation."""
    flower = FlowerObservation(box=BBox(x, 200, x + 80, 280), score=0.9)
    flower.sex = SexEstimate(sex=sex, confidence=0.8,
                             p_female=0.9 if sex is FlowerSex.FEMALE else 0.1)
    flower.anthesis = AnthesisEstimate(stage=AnthesisStage.RECEPTIVE, confidence=0.9)
    flower.pollen = PollenEstimate(availability=pollen, confidence=0.8)
    flower.quality = QualityAssessment(accepted=True, sharpness=0.8, occlusion=0.05)
    flower.range_m = range_m
    flower.incidence_deg = 10.0
    flower.track_id = track_id
    flower.pose = Pose3D(position=np.array([0.0, 0.0, range_m]),
                         normal=np.array([0.0, 0.0, -1.0]))
    flower.rover_pose = Pose3D(position=np.array([range_m, 0.0, 0.25]),
                               normal=np.array([-1.0, 0.0, 0.0]), frame="rover")
    return flower


def make_result(flowers, frame_index: int = 1) -> FrameResult:
    return FrameResult(frame_index=frame_index, width=640, height=480,
                       flowers=list(flowers))


# --------------------------------------------------------------------------- #


class TestConfig:
    def test_species_overlay_wins(self, cfg):
        assert cfg.get("species.name") == "pumpkin"
        assert cfg.get("species.corolla_diameter_m") == pytest.approx(0.105)

    def test_base_values_survive_the_overlay(self, cfg):
        assert cfg.get("runtime.device") == "cpu"

    def test_dotted_default_for_a_missing_key(self, cfg):
        assert cfg.get("nope.not.here", "fallback") == "fallback"

    def test_cli_overrides_parse_into_nested_dicts(self):
        parsed = parse_overrides(["a.b.c=3", "d=true", "e=[1,2]"])
        assert parsed == {"a": {"b": {"c": 3}}, "d": True, "e": [1, 2]}

    def test_override_beats_overlay(self):
        cfg = load_config("default", ["species/cucumber"],
                          {"species": {"corolla_diameter_m": 0.5}})
        assert cfg.get("species.corolla_diameter_m") == 0.5

    def test_esp32_overlay_switches_to_size_ranging(self):
        cfg = load_config("default", ["esp32cam"])
        assert cfg.get("geometry.depth.mode") == "size"

    def test_bad_override_is_rejected(self):
        with pytest.raises(ValueError):
            parse_overrides(["not-an-assignment"])

    def test_set_and_get_round_trip(self):
        cfg = Config({})
        cfg.set("a.b.c", 7)
        assert cfg.get("a.b.c") == 7


class TestTargetSelection:
    def test_empty_probe_prefers_a_male(self, cfg):
        selector = TargetSelector(cfg)
        plan = selector.select(
            [make_flower(FlowerSex.FEMALE, 1, x=100),
             make_flower(FlowerSex.MALE, 2, x=400)],
            probe_charge=0.0)
        assert plan is not None
        assert plan.mode is ProbeMode.COLLECT
        assert plan.flower.sex.sex is FlowerSex.MALE

    def test_loaded_probe_will_service_a_female(self, cfg):
        selector = TargetSelector(cfg)
        plan = selector.select([make_flower(FlowerSex.FEMALE, 1)], probe_charge=0.9)
        assert plan is not None and plan.mode is ProbeMode.DEPOSIT

    def test_empty_probe_cannot_service_a_female(self, cfg):
        selector = TargetSelector(cfg)
        assert selector.select([make_flower(FlowerSex.FEMALE, 1)], probe_charge=0.0) is None

    def test_non_receptive_flowers_are_never_targeted(self, cfg):
        selector = TargetSelector(cfg)
        flower = make_flower(FlowerSex.MALE, 1)
        flower.anthesis = AnthesisEstimate(stage=AnthesisStage.SENESCENT, confidence=0.9)
        assert selector.select([flower], probe_charge=0.0) is None

    def test_out_of_range_flowers_are_skipped(self, cfg):
        selector = TargetSelector(cfg)
        far = make_flower(FlowerSex.MALE, 1, range_m=5.0)
        assert selector.select([far], probe_charge=0.0) is None

    def test_steeply_angled_flowers_are_skipped(self, cfg):
        """Past the incidence limit the field projection is not worth the drive."""
        selector = TargetSelector(cfg)
        flower = make_flower(FlowerSex.MALE, 1)
        flower.incidence_deg = 80.0
        assert selector.select([flower], probe_charge=0.0) is None

    def test_depleted_anther_is_not_worth_a_visit(self, cfg):
        selector = TargetSelector(cfg)
        assert selector.select([make_flower(FlowerSex.MALE, 1, pollen=0.02)],
                               probe_charge=0.0) is None

    def test_nearer_flower_wins_all_else_equal(self, cfg):
        selector = TargetSelector(cfg)
        plan = selector.select([make_flower(FlowerSex.MALE, 1, range_m=0.9),
                                make_flower(FlowerSex.MALE, 2, range_m=0.2)],
                               probe_charge=0.0)
        assert plan.flower.track_id == 2

    def test_richer_anther_wins_at_equal_range(self, cfg):
        selector = TargetSelector(cfg)
        plan = selector.select([make_flower(FlowerSex.MALE, 1, pollen=0.25),
                                make_flower(FlowerSex.MALE, 2, pollen=0.95)],
                               probe_charge=0.0)
        assert plan.flower.track_id == 2

    def test_probe_is_placed_at_the_configured_standoff(self, cfg):
        selector = TargetSelector(cfg)
        plan = selector.select([make_flower(FlowerSex.MALE, 1)], probe_charge=0.0)
        offset = np.linalg.norm(plan.probe_position - plan.flower.rover_pose.position)
        assert offset == pytest.approx(plan.standoff_m, abs=1e-6)


class TestMissionLoop:
    def test_idle_when_nothing_is_visible(self, cfg):
        mission = MissionController(cfg)
        step = mission.step(make_result([]), EnvironmentReading())
        assert step.state is MissionState.SEARCH
        assert "no flowers detected" in step.message

    def test_full_collection_cycle(self, cfg):
        """search -> approach -> actuate -> verify, then the probe is loaded."""
        mission = MissionController(cfg)
        male = make_flower(FlowerSex.MALE, 1)
        frame = np.full((480, 640, 3), 120, dtype=np.uint8)

        step = mission.step(make_result([male], 1), EnvironmentReading(), frame,
                            probe_at_pose=False)
        assert step.state is MissionState.APPROACH

        fired = []
        step = mission.step(make_result([male], 2), EnvironmentReading(), frame,
                            probe_at_pose=True, fire=lambda c: fired.append(c) is None)
        assert step.state is MissionState.VERIFY
        assert len(fired) == 1
        assert fired[0].mode is ProbeMode.COLLECT
        assert fired[0].voltage_kv > 0

        # Verification needs the flower to settle before it will judge.
        depleted = make_flower(FlowerSex.MALE, 1, pollen=0.1)
        for index in range(3, 8):
            step = mission.step(make_result([depleted], index), EnvironmentReading(),
                                frame, probe_at_pose=True)
            if step.verification is not None and not step.message.startswith("waiting"):
                break
        assert step.verification is not None
        assert step.verification.success
        assert mission.ledger.probe_charge > 0

    def test_serviced_flower_is_not_revisited(self, cfg):
        mission = MissionController(cfg)
        mission.ledger.record_attempt(5, FlowerSex.FEMALE)
        mission.ledger.record_outcome(5, True)
        mission.ledger.load_probe(9, 1.0)

        female = make_flower(FlowerSex.FEMALE, 5)
        step = mission.step(make_result([female]), EnvironmentReading())
        assert step.state is MissionState.SEARCH
        assert step.plan is None

    def test_actuation_failure_routes_to_recovery(self, cfg):
        mission = MissionController(cfg)
        male = make_flower(FlowerSex.MALE, 1)
        frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        mission.step(make_result([male], 1), EnvironmentReading(), frame)
        step = mission.step(make_result([male], 2), EnvironmentReading(), frame,
                            probe_at_pose=True, fire=lambda c: False)
        assert step.state is MissionState.RECOVER

    def test_lost_target_is_eventually_abandoned(self, cfg):
        mission = MissionController(cfg)
        male = make_flower(FlowerSex.MALE, 1)
        mission.step(make_result([male], 1), EnvironmentReading())
        for index in range(2, 2 + MissionController.LOST_TARGET_PATIENCE + 2):
            step = mission.step(make_result([], index), EnvironmentReading())
        assert step.state in (MissionState.RECOVER, MissionState.SEARCH)

    def test_humidity_reaches_the_fired_command(self, cfg):
        """Sensing must actually parameterise actuation, not just be logged."""
        frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        voltages = []
        for humidity in (35.0, 90.0):
            mission = MissionController(cfg)
            male = make_flower(FlowerSex.MALE, 1)
            reading = EnvironmentReading(humidity=humidity, tof_distance_m=0.04)
            mission.step(make_result([male], 1), reading, frame)
            step = mission.step(make_result([male], 2), reading, frame,
                                probe_at_pose=True)
            voltages.append(step.command.voltage_kv)
        assert voltages[1] > voltages[0]

    def test_wire_format_is_serialisable(self, cfg):
        import json

        mission = MissionController(cfg)
        male = make_flower(FlowerSex.MALE, 1)
        frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        mission.step(make_result([male], 1), EnvironmentReading(), frame)
        step = mission.step(make_result([male], 2), EnvironmentReading(), frame,
                            probe_at_pose=True)
        payload = json.loads(json.dumps(step.to_wire()))
        assert payload["hv"]["kv"] > 0
        assert payload["goal"]["mode"] == "collect"

    def test_rejected_candidates_are_explained(self, cfg):
        from pollivision.types import RejectReason

        mission = MissionController(cfg)
        rejected = make_flower(FlowerSex.MALE, 1)
        rejected.quality = QualityAssessment(accepted=False,
                                             reasons=[RejectReason.OCCLUDED])
        result = FrameResult(frame_index=1, width=640, height=480,
                             flowers=[], rejected=[rejected])
        step = mission.step(result, EnvironmentReading())
        assert "occluded" in step.message
