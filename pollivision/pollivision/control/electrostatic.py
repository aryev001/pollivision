"""Adaptive electrostatic parameter controller (report Sec. III-E and III-F).

This is the component the report identifies as its central contribution.
Existing electrostatic pollination work operates at fixed charging parameters;
here the probe voltage, exposure duration and discharge timing are recomputed
for every attempt from four live quantities:

* ambient relative humidity (Sec. III-E-i),
* probe-to-flower distance (Sec. III-E-ii),
* corolla orientation, as the incidence angle of the probe axis (Sec. III-E-ii),
* visually estimated pollen availability (Sec. III-E-iii).

Following Sec. III-F this is a rule-based closed-form mapping, not a learned
policy. That is the right choice for three reasons: it runs in microseconds on
an embedded controller, every coefficient is physically interpretable and can be
fitted from a modest calibration campaign, and - most importantly for a device
that drives kilovolts near living tissue - its behaviour outside the calibration
envelope is predictable rather than merely untested.

Physical reasoning behind each term
-----------------------------------
**Distance.** The field from a small probe falls roughly as 1/d^2, so holding a
constant field at the flower requires voltage to rise as d^2. A real probe is
neither an ideal sphere nor an infinite plane, so the exponent is a tunable
parameter defaulting to 1.6 - between the plane (1.0) and sphere (2.0) cases.

**Humidity.** Adsorbed water raises the surface conductivity of the pollen
exine, so induced charge bleeds away faster as RH climbs; the effect is mild at
moderate RH and steep past roughly 60-70%. A logistic in RH captures that shape,
whereas a linear correction would either under-compensate in humid conditions or
over-drive in dry ones.

**Orientation.** Only the field component normal to the corolla face does useful
work on grains sitting in it, so the useful field scales with cos(incidence) and
drive is raised by its reciprocal. The reciprocal is floored, because past a
certain obliquity the correct response is to reposition the rover rather than to
keep raising the voltage.

**Pollen availability.** A sparse anther yields less per unit time, so exposure
is extended when availability is low. Voltage is raised only slightly, because
the limiting factor on a depleted anther is how much pollen is present, not how
hard it is being pulled.

Every computed command then passes through :mod:`pollivision.control.safety`,
which enforces the field ceiling and standoff floor independently of anything
here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..logging_utils import get_logger
from ..types import (
    ElectrostaticCommand,
    EnvironmentReading,
    FlowerObservation,
    FlowerSex,
    ProbeMode,
)
from .safety import SafetyLimits, apply_limits, target_field_kv_per_m, tip_field_kv_per_m

LOGGER = get_logger(__name__)


@dataclass
class ControllerGains:
    """Tunable coefficients of the adaptive mapping.

    Defaults are physically motivated starting points, not measured constants.
    ``tools/calibrate_electrostatic.py`` fits them to experimental transfer
    efficiency data, which is the empirical calibration Sec. III-F describes.
    """

    distance_exponent: float = 1.6
    humidity_midpoint: float = 68.0
    humidity_steepness: float = 0.09
    humidity_max_boost: float = 0.55
    incidence_min_cos: float = 0.35
    incidence_gain: float = 0.8
    pollen_exposure_gain: float = 0.9
    pollen_voltage_gain: float = 0.25

    @classmethod
    def from_config(cls, cfg) -> "ControllerGains":
        section = cfg.section("electrostatic.gains")
        return cls(
            distance_exponent=float(section.get("distance_exponent", 1.6)),
            humidity_midpoint=float(section.get("humidity_midpoint", 68.0)),
            humidity_steepness=float(section.get("humidity_steepness", 0.09)),
            humidity_max_boost=float(section.get("humidity_max_boost", 0.55)),
            incidence_min_cos=float(section.get("incidence_min_cos", 0.35)),
            incidence_gain=float(section.get("incidence_gain", 0.8)),
            pollen_exposure_gain=float(section.get("pollen_exposure_gain", 0.9)),
            pollen_voltage_gain=float(section.get("pollen_voltage_gain", 0.25)),
        )


class AdaptiveElectrostaticController:
    """Computes probe voltage, exposure and timing for each pollination attempt."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.enabled = bool(cfg.get("electrostatic.enabled", True))
        self.gains = ControllerGains.from_config(cfg)
        self.limits = SafetyLimits.from_config(cfg)

        reference = cfg.section("electrostatic.reference")
        self.v_reference = float(reference.get("voltage_kv", 6.0))
        self.d_reference = float(reference.get("standoff_m", 0.04))
        self.rh_reference = float(reference.get("humidity_pct", 55.0))
        self.t_reference = float(reference.get("exposure_ms", 350.0))

        self.tip_radius = float(cfg.get("electrostatic.probe_tip_radius_m", 0.003))

        deposit = cfg.section("electrostatic.deposit")
        self.deposit_voltage_scale = float(deposit.get("voltage_scale", 0.65))
        self.deposit_exposure_scale = float(deposit.get("exposure_scale", 0.8))
        self.deposit_discharge_delay = float(deposit.get("discharge_delay_ms", 60.0))

        self.default_standoff = float(cfg.get("planning.approach_standoff_m", 0.04))

    # ------------------------------------------------------------------ #
    # Individual gain terms
    # ------------------------------------------------------------------ #

    def distance_factor(self, standoff_m: float) -> float:
        """Voltage multiplier holding the surface field roughly constant."""
        ratio = max(standoff_m, 1e-4) / max(self.d_reference, 1e-4)
        return float(ratio ** self.gains.distance_exponent)

    def humidity_factor(self, humidity_pct: float) -> float:
        """Voltage multiplier compensating for humidity-driven charge decay.

        Normalised so the reference humidity yields exactly 1.0, which keeps the
        reference operating point meaningful when the midpoint is retuned.
        """
        def logistic(rh: float) -> float:
            return 1.0 / (1.0 + math.exp(-self.gains.humidity_steepness *
                                         (rh - self.gains.humidity_midpoint)))

        boost = 1.0 + self.gains.humidity_max_boost * logistic(humidity_pct)
        baseline = 1.0 + self.gains.humidity_max_boost * logistic(self.rh_reference)
        return float(boost / max(baseline, 1e-6))

    def incidence_factor(self, incidence_deg: Optional[float]) -> float:
        """Voltage multiplier compensating for an obliquely presented corolla."""
        if incidence_deg is None:
            return 1.0
        cosine = math.cos(math.radians(float(np.clip(incidence_deg, 0.0, 89.0))))
        cosine = max(cosine, self.gains.incidence_min_cos)
        # At normal incidence this is exactly 1; it grows as the flower tips away.
        return float(1.0 + self.gains.incidence_gain * (1.0 / cosine - 1.0))

    def pollen_factors(self, availability: float) -> tuple[float, float]:
        """Return ``(voltage_factor, exposure_factor)`` for a pollen load."""
        deficit = float(np.clip(1.0 - availability, 0.0, 1.0))
        voltage = 1.0 + self.gains.pollen_voltage_gain * deficit
        exposure = 1.0 + self.gains.pollen_exposure_gain * deficit
        return float(voltage), float(exposure)

    # ------------------------------------------------------------------ #
    # Command synthesis
    # ------------------------------------------------------------------ #

    def compute(
        self,
        flower: FlowerObservation,
        environment: EnvironmentReading,
        mode: ProbeMode,
        standoff_m: Optional[float] = None,
    ) -> ElectrostaticCommand:
        """Compute the electrostatic command for one pollination attempt."""
        if not self.enabled or mode is ProbeMode.IDLE:
            return ElectrostaticCommand(mode=ProbeMode.IDLE)

        # Prefer the ToF reading over the vision estimate when it is available:
        # Sec. III-D has the ToF sensor refining range during the final approach,
        # and it is both more accurate and more current than the vision estimate.
        standoff = self._resolve_standoff(flower, environment, standoff_m)

        availability = float(np.clip(flower.pollen.availability, 0.0, 1.0))
        if mode is ProbeMode.DEPOSIT:
            # On deposition the relevant quantity is how loaded the *probe* is,
            # which the caller passes in via the observation's pollen field.
            availability = float(np.clip(flower.meta.get("probe_charge", availability), 0.0, 1.0))

        distance_gain = self.distance_factor(standoff)
        humidity_gain = self.humidity_factor(environment.humidity)
        incidence_gain = self.incidence_factor(flower.incidence_deg)
        pollen_voltage_gain, pollen_exposure_gain = self.pollen_factors(availability)

        voltage = (self.v_reference * distance_gain * humidity_gain
                   * incidence_gain * pollen_voltage_gain)
        exposure = self.t_reference * pollen_exposure_gain * humidity_gain

        if mode is ProbeMode.DEPOSIT:
            voltage *= self.deposit_voltage_scale
            exposure *= self.deposit_exposure_scale

        voltage, exposure, standoff, clamped = apply_limits(
            voltage, exposure, standoff, self.tip_radius, self.limits
        )

        tip_field = tip_field_kv_per_m(voltage, standoff, self.tip_radius)
        target_field = target_field_kv_per_m(voltage, standoff, self.tip_radius)
        if target_field < self.limits.min_useful_target_field_kv_per_m:
            # Safe, but too weak to plausibly move pollen. Surfaced rather than
            # corrected: the fix is a shorter standoff or a recalibrated
            # reference point, and silently raising the voltage to reach a
            # target field would defeat the corona ceiling that just clamped it.
            clamped.append(
                f"target field {target_field:.1f}kV/m below useful minimum "
                f"{self.limits.min_useful_target_field_kv_per_m:.1f}kV/m"
            )

        return ElectrostaticCommand(
            mode=mode,
            voltage_kv=voltage,
            exposure_ms=exposure,
            discharge_delay_ms=self.deposit_discharge_delay
            if mode is ProbeMode.DEPOSIT else 0.0,
            # Collection attracts grains onto the probe; deposition reverses the
            # polarity so they are repelled onto the stigma.
            polarity=-1 if mode is ProbeMode.COLLECT else +1,
            standoff_m=standoff,
            predicted_field_kv_per_m=target_field,
            tip_field_kv_per_m=tip_field,
            clamped=clamped,
            rationale={
                "distance_gain": round(distance_gain, 4),
                "humidity_gain": round(humidity_gain, 4),
                "incidence_gain": round(incidence_gain, 4),
                "pollen_voltage_gain": round(pollen_voltage_gain, 4),
                "pollen_exposure_gain": round(pollen_exposure_gain, 4),
                "humidity_pct": round(environment.humidity, 2),
                "standoff_mm": round(standoff * 1000, 2),
                "incidence_deg": round(flower.incidence_deg or 0.0, 2),
                "pollen_availability": round(availability, 4),
                "tip_field_kv_per_m": round(tip_field, 1),
            },
        )

    def _resolve_standoff(self, flower: FlowerObservation,
                          environment: EnvironmentReading,
                          override: Optional[float]) -> float:
        if override is not None:
            return float(override)
        if environment.tof_distance_m is not None and environment.valid:
            return float(environment.tof_distance_m)
        # Falling back on the vision range means the standoff is only as good as
        # the size prior, which is precisely why the safety module enforces a
        # floor independently.
        if flower.range_m is not None:
            return float(min(flower.range_m, self.default_standoff * 3.0))
        return self.default_standoff

    # ------------------------------------------------------------------ #

    def decide_mode(self, flower: FlowerObservation, probe_charge: float,
                    min_charge_to_deposit: float = 0.15) -> ProbeMode:
        """Choose collect or deposit for a given flower and probe state.

        The rover cannot deposit what it has not collected, so an empty probe
        makes every male flower a collection opportunity and every female one a
        flower to remember and come back to.
        """
        if flower.sex.sex is FlowerSex.MALE:
            # Topping up from a richer anther than the current load is worth it.
            if probe_charge < 0.85 and flower.pollen.availability > 0.15:
                return ProbeMode.COLLECT
            return ProbeMode.IDLE
        if flower.sex.sex is FlowerSex.FEMALE:
            if probe_charge >= min_charge_to_deposit:
                return ProbeMode.DEPOSIT
            return ProbeMode.IDLE
        return ProbeMode.IDLE
