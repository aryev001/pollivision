"""Adaptive electrostatic controller and safety-limit tests.

The safety assertions here are the most important tests in the suite: this is
the only component that drives kilovolts near living tissue, and its limits must
hold for *every* combination of sensor readings, including physically impossible
ones produced by a failing sensor.
"""

from __future__ import annotations

import itertools
import math

import pytest

from pollivision.config import load_config
from pollivision.control import safety
from pollivision.control.electrostatic import AdaptiveElectrostaticController
from pollivision.types import (
    BBox,
    EnvironmentReading,
    FlowerObservation,
    FlowerSex,
    PollenEstimate,
    ProbeMode,
    SexEstimate,
)


@pytest.fixture
def cfg():
    return load_config("default")


@pytest.fixture
def controller(cfg):
    return AdaptiveElectrostaticController(cfg)


def make_flower(incidence=0.0, pollen=0.8, range_m=0.04,
                sex=FlowerSex.MALE) -> FlowerObservation:
    flower = FlowerObservation(box=BBox(0, 0, 50, 50), score=0.9)
    flower.sex = SexEstimate(sex=sex, p_female=0.9 if sex is FlowerSex.FEMALE else 0.1,
                             confidence=0.8)
    flower.pollen = PollenEstimate(availability=pollen, confidence=0.8)
    flower.incidence_deg = incidence
    flower.range_m = range_m
    return flower


def env(humidity=55.0, standoff=0.04) -> EnvironmentReading:
    return EnvironmentReading(humidity=humidity, tof_distance_m=standoff)


class TestFieldModels:
    def test_corona_onset_rises_for_sharper_tips(self):
        """Peek's law: a sharper electrode tolerates a higher surface field."""
        assert (safety.corona_onset_kv_per_m(0.0005)
                > safety.corona_onset_kv_per_m(0.006))

    def test_corona_onset_exceeds_bulk_breakdown(self):
        for radius in (0.0005, 0.001, 0.003, 0.006):
            assert (safety.corona_onset_kv_per_m(radius)
                    > safety.AIR_BREAKDOWN_KV_PER_M)

    def test_tip_field_exceeds_target_field(self):
        """Field concentrates at the electrode and is far weaker at the flower."""
        tip = safety.tip_field_kv_per_m(6.0, 0.04, 0.003)
        target = safety.target_field_kv_per_m(6.0, 0.04, 0.003)
        assert tip > target * 10

    def test_target_field_falls_with_standoff(self):
        near = safety.target_field_kv_per_m(6.0, 0.02, 0.003)
        far = safety.target_field_kv_per_m(6.0, 0.08, 0.003)
        assert near > far

    def test_max_safe_voltage_inverts_the_tip_field_model(self):
        voltage = safety.max_safe_voltage(0.04, 0.003, 2500.0)
        assert safety.tip_field_kv_per_m(voltage, 0.04, 0.003) == pytest.approx(2500.0, rel=1e-6)


class TestGainTerms:
    def test_reference_point_has_unit_gain(self, controller):
        assert controller.distance_factor(controller.d_reference) == pytest.approx(1.0)
        assert controller.humidity_factor(controller.rh_reference) == pytest.approx(1.0)
        assert controller.incidence_factor(0.0) == pytest.approx(1.0)

    def test_humidity_gain_is_monotonic(self, controller):
        values = [controller.humidity_factor(rh) for rh in range(20, 100, 5)]
        assert all(b >= a for a, b in zip(values, values[1:]))

    def test_distance_gain_is_monotonic(self, controller):
        values = [controller.distance_factor(d) for d in (0.01, 0.02, 0.04, 0.08)]
        assert all(b > a for a, b in zip(values, values[1:]))

    def test_incidence_gain_is_floored_at_grazing_angles(self, controller):
        """Drive must not diverge as the corolla tips edge-on."""
        assert controller.incidence_factor(89.0) < 10.0

    def test_sparse_anther_gets_a_longer_dwell(self, controller):
        _, empty = controller.pollen_factors(0.02)
        _, full = controller.pollen_factors(1.0)
        assert empty > full


class TestSafetyInvariants:
    """These must hold across the whole reachable input space."""

    def test_limits_hold_over_a_full_sweep(self, controller):
        limits = controller.limits
        ceiling = limits.tip_field_ceiling(controller.tip_radius)

        humidities = (5.0, 30.0, 55.0, 80.0, 100.0)
        standoffs = (0.001, 0.005, 0.015, 0.04, 0.10, 0.5)
        incidences = (0.0, 30.0, 60.0, 89.0)
        pollens = (0.0, 0.5, 1.0)

        for humidity, standoff, incidence, pollen in itertools.product(
                humidities, standoffs, incidences, pollens):
            for mode in (ProbeMode.COLLECT, ProbeMode.DEPOSIT):
                command = controller.compute(
                    make_flower(incidence, pollen, standoff),
                    env(humidity, standoff), mode,
                )
                assert limits.voltage_min_kv - 1e-6 <= command.voltage_kv \
                    <= limits.voltage_max_kv + 1e-6
                assert limits.exposure_min_ms - 1e-6 <= command.exposure_ms \
                    <= limits.exposure_max_ms + 1e-6
                assert command.tip_field_kv_per_m <= ceiling + 1e-3
                assert command.standoff_m >= limits.min_standoff_m - 1e-9
                assert math.isfinite(command.voltage_kv)

    def test_standoff_floor_is_enforced_and_reported(self, controller):
        command = controller.compute(make_flower(range_m=0.001), env(55.0, 0.001),
                                     ProbeMode.COLLECT)
        assert command.standoff_m >= controller.limits.min_standoff_m
        assert any("standoff" in reason for reason in command.clamped)

    def test_corona_ceiling_binds_for_a_sharp_tip(self):
        cfg = load_config("default", overrides={
            "electrostatic": {"probe_tip_radius_m": 0.0005,
                              "limits": {"voltage_kv": [1.5, 30.0]}}})
        controller = AdaptiveElectrostaticController(cfg)
        command = controller.compute(make_flower(70.0, 0.05, 0.08),
                                     env(95.0, 0.08), ProbeMode.COLLECT)
        assert any("corona" in reason for reason in command.clamped)
        assert command.tip_field_kv_per_m <= controller.limits.tip_field_ceiling(0.0005) + 1e-3

    def test_weak_command_is_flagged_not_silently_boosted(self):
        """A safe-but-useless command must be visible, never quietly amplified."""
        cfg = load_config("default", overrides={
            "electrostatic": {"probe_tip_radius_m": 0.0005,
                              "limits": {"voltage_kv": [1.5, 30.0]}}})
        controller = AdaptiveElectrostaticController(cfg)
        command = controller.compute(make_flower(70.0, 0.05, 0.08),
                                     env(95.0, 0.08), ProbeMode.COLLECT)
        assert any("useful minimum" in reason for reason in command.clamped)

    def test_inverted_limits_are_rejected_at_construction(self):
        with pytest.raises(ValueError):
            safety.SafetyLimits(voltage_min_kv=10.0, voltage_max_kv=2.0)
        with pytest.raises(ValueError):
            safety.SafetyLimits(corona_margin=0.0)


class TestModes:
    def test_deposit_is_gentler_than_collect(self, controller):
        collect = controller.compute(make_flower(), env(), ProbeMode.COLLECT)
        deposit = controller.compute(make_flower(), env(), ProbeMode.DEPOSIT)
        assert deposit.voltage_kv < collect.voltage_kv
        assert deposit.polarity == +1 and collect.polarity == -1
        assert deposit.discharge_delay_ms > 0

    def test_idle_produces_no_output(self, controller):
        command = controller.compute(make_flower(), env(), ProbeMode.IDLE)
        assert command.voltage_kv == 0.0 and command.mode is ProbeMode.IDLE

    def test_disabled_controller_is_inert(self):
        cfg = load_config("default", overrides={"electrostatic": {"enabled": False}})
        controller = AdaptiveElectrostaticController(cfg)
        command = controller.compute(make_flower(), env(), ProbeMode.COLLECT)
        assert command.voltage_kv == 0.0

    def test_empty_probe_cannot_deposit(self, controller):
        female = make_flower(sex=FlowerSex.FEMALE)
        assert controller.decide_mode(female, probe_charge=0.0) is ProbeMode.IDLE
        assert controller.decide_mode(female, probe_charge=0.6) is ProbeMode.DEPOSIT

    def test_depleted_anther_is_not_worth_collecting(self, controller):
        male = make_flower(pollen=0.02)
        assert controller.decide_mode(male, probe_charge=0.0) is ProbeMode.IDLE

    def test_tof_reading_overrides_the_vision_range(self, controller):
        """The ToF sensor refines standoff during final approach (Sec. III-D)."""
        flower = make_flower(range_m=0.09)
        command = controller.compute(
            flower, EnvironmentReading(humidity=55.0, tof_distance_m=0.03),
            ProbeMode.COLLECT)
        assert command.standoff_m == pytest.approx(0.03)

    def test_invalid_telemetry_falls_back_to_vision_range(self, controller):
        flower = make_flower(range_m=0.05)
        reading = EnvironmentReading(humidity=55.0, tof_distance_m=0.03, valid=False)
        command = controller.compute(flower, reading, ProbeMode.COLLECT)
        assert command.standoff_m != pytest.approx(0.03)

    def test_rationale_is_recorded_for_every_command(self, controller):
        command = controller.compute(make_flower(30.0, 0.4, 0.05),
                                     env(75.0, 0.05), ProbeMode.COLLECT)
        for key in ("distance_gain", "humidity_gain", "incidence_gain",
                    "pollen_availability", "tip_field_kv_per_m"):
            assert key in command.rationale
