"""Field models and hard safety limits for the electrostatic end-effector.

Kept separate from the adaptive controller, and applied *after* it, so that no
amount of gain tuning or bad sensor data can talk the system into an unsafe
output. The controller proposes; this disposes.

Two different fields matter, at two different places, and conflating them is the
easiest way to build a system whose safety limit never actually engages:

* **Tip field** - the field at the probe electrode itself. This is what governs
  corona onset, and corona is actively harmful here: it generates ionic wind and
  ozone that disperse the very grains the probe is trying to hold, on top of the
  usual electrical hazards. This is the quantity the safety ceiling is set on,
  because it is the one that binds first as voltage rises.

* **Target field** - the much weaker field out at the flower surface, which is
  what actually exerts force on pollen grains. This one is reported as an
  *efficacy* figure rather than a safety limit: if it falls too low the command
  is physically incapable of moving pollen regardless of being perfectly safe,
  and the operator needs to know that during calibration.

Both are analytic approximations for a sphere-tipped probe near a plant, which
is neither an isolated sphere nor a parallel plate. They are the right shape and
the right order of magnitude; the absolute coefficients belong to the empirical
calibration Sec. III-F calls for, and ``tools/calibrate_electrostatic.py``
fits them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Dielectric strength of dry air at sea level and standard temperature, kV/m.
AIR_BREAKDOWN_KV_PER_M = 3000.0

#: Peek's-law surface-roughness coefficient for a smooth electrode, in
#: sqrt(metres). Corona begins above the bulk breakdown value on a curved
#: electrode because the field falls away steeply from the surface.
PEEK_COEFFICIENT = 0.0308


def corona_onset_kv_per_m(tip_radius_m: float,
                          relative_air_density: float = 1.0) -> float:
    """Corona inception field at a curved electrode, after Peek's law.

    ``E0 = 3000 * delta * (1 + 0.0308 / sqrt(delta * r))`` in kV/m. Sharper tips
    tolerate a higher *surface* field before ionising, which is why a fine probe
    can be driven harder than a blunt one - but they also concentrate the field,
    so the net effect on the usable voltage is not obvious a priori and is worth
    computing rather than guessing.
    """
    radius = max(tip_radius_m, 1e-5)
    delta = max(relative_air_density, 1e-3)
    return float(AIR_BREAKDOWN_KV_PER_M * delta *
                 (1.0 + PEEK_COEFFICIENT / math.sqrt(delta * radius)))


def tip_field_kv_per_m(voltage_kv: float, standoff_m: float,
                       tip_radius_m: float) -> float:
    """Field magnitude at the probe tip surface.

    Sphere-to-plane geometry with the plant treated as the counter-electrode.
    The logarithmic form ``E = V / (r * ln(1 + d/r))`` is the standard
    approximation for this configuration and reduces sensibly at both extremes:
    it approaches the parallel-plate value ``V/d`` when the gap is small
    relative to the tip, and grows only logarithmically once the gap is large,
    which matches the weak dependence a real probe shows.
    """
    radius = max(tip_radius_m, 1e-5)
    gap = max(standoff_m, 1e-4)
    return float(voltage_kv / (radius * math.log1p(gap / radius)))


def target_field_kv_per_m(voltage_kv: float, standoff_m: float,
                          tip_radius_m: float) -> float:
    """Field magnitude at the flower surface.

    Charged sphere above a grounded plane, evaluated on the plane directly
    beneath the tip. The sphere carries ``Q = 4*pi*eps0*r*V``, and including its
    image charge the field at the plane is ``2*V*r/h^2`` with ``h = d + r`` the
    centre-to-plane distance. The 1/d^2 falloff this predicts is why standoff is
    the single most influential variable in the whole controller.
    """
    radius = max(tip_radius_m, 1e-5)
    centre_distance = max(standoff_m, 1e-4) + radius
    return float(2.0 * voltage_kv * radius / (centre_distance ** 2))


def max_safe_voltage(standoff_m: float, tip_radius_m: float,
                     max_tip_field_kv_per_m: float) -> float:
    """Largest probe voltage keeping the tip field within its ceiling."""
    radius = max(tip_radius_m, 1e-5)
    gap = max(standoff_m, 1e-4)
    return float(max_tip_field_kv_per_m * radius * math.log1p(gap / radius))


@dataclass
class SafetyLimits:
    """Bounds applied to every command before it reaches the hardware."""

    voltage_min_kv: float = 1.5
    voltage_max_kv: float = 12.0
    exposure_min_ms: float = 80.0
    exposure_max_ms: float = 1500.0
    #: Ceiling on the *tip* field, expressed as a fraction of the corona onset
    #: field for the fitted tip radius. Staying meaningfully below onset leaves
    #: margin for a roughened or contaminated electrode, which ionises earlier
    #: than a clean one.
    corona_margin: float = 0.55
    #: Optional absolute ceiling on the tip field. The effective limit is the
    #: lower of this and the corona-derived one.
    max_tip_field_kv_per_m: float = 4000.0
    min_standoff_m: float = 0.015
    #: Advisory floor on the *target* field. Below this a command is safe but
    #: too weak to plausibly move pollen; flagged, never clamped.
    min_useful_target_field_kv_per_m: float = 8.0

    @classmethod
    def from_config(cls, cfg) -> "SafetyLimits":
        section = cfg.section("electrostatic.limits")
        voltage = section.get("voltage_kv", [1.5, 12.0])
        exposure = section.get("exposure_ms", [80.0, 1500.0])
        return cls(
            voltage_min_kv=float(voltage[0]),
            voltage_max_kv=float(voltage[1]),
            exposure_min_ms=float(exposure[0]),
            exposure_max_ms=float(exposure[1]),
            corona_margin=float(section.get("corona_margin", 0.55)),
            max_tip_field_kv_per_m=float(section.get("max_tip_field_kv_per_m", 4000.0)),
            min_standoff_m=float(section.get("min_standoff_m", 0.015)),
            min_useful_target_field_kv_per_m=float(
                section.get("min_useful_target_field_kv_per_m", 8.0)),
        )

    def __post_init__(self) -> None:
        if self.voltage_min_kv > self.voltage_max_kv:
            raise ValueError("electrostatic.limits.voltage_kv bounds are inverted")
        if self.exposure_min_ms > self.exposure_max_ms:
            raise ValueError("electrostatic.limits.exposure_ms bounds are inverted")
        if not 0.0 < self.corona_margin <= 1.0:
            raise ValueError("electrostatic.limits.corona_margin must be in (0, 1]")

    def tip_field_ceiling(self, tip_radius_m: float) -> float:
        """Effective tip-field ceiling for a given electrode radius."""
        return min(
            self.max_tip_field_kv_per_m,
            self.corona_margin * corona_onset_kv_per_m(tip_radius_m),
        )


def apply_limits(voltage_kv: float, exposure_ms: float, standoff_m: float,
                 tip_radius_m: float,
                 limits: SafetyLimits) -> tuple[float, float, float, list[str]]:
    """Clamp a proposed command.

    Returns ``(voltage_kv, exposure_ms, standoff_m, reasons)``. The standoff is
    returned because enforcing its floor changes the geometry every downstream
    field calculation depends on, and silently reporting a field computed at a
    standoff the system refused to use would be worse than useless.

    Every clamp that fires is named, so a rover quietly pinned against a ceiling
    on every cycle shows up in the logs rather than looking like a tuning issue.
    """
    reasons: list[str] = []

    if standoff_m < limits.min_standoff_m:
        reasons.append(
            f"standoff {standoff_m * 1000:.0f}mm raised to floor "
            f"{limits.min_standoff_m * 1000:.0f}mm"
        )
        standoff_m = limits.min_standoff_m

    ceiling = limits.tip_field_ceiling(tip_radius_m)
    voltage_ceiling = max_safe_voltage(standoff_m, tip_radius_m, ceiling)
    if voltage_kv > voltage_ceiling:
        reasons.append(
            f"corona limit: {voltage_kv:.2f}kV -> {voltage_ceiling:.2f}kV "
            f"(tip field ceiling {ceiling:.0f}kV/m)"
        )
        voltage_kv = voltage_ceiling

    if voltage_kv > limits.voltage_max_kv:
        reasons.append(f"voltage ceiling {limits.voltage_max_kv:.1f}kV")
        voltage_kv = limits.voltage_max_kv
    elif voltage_kv < limits.voltage_min_kv:
        reasons.append(f"voltage floor {limits.voltage_min_kv:.1f}kV")
        voltage_kv = limits.voltage_min_kv

    clamped_exposure = float(np.clip(exposure_ms, limits.exposure_min_ms,
                                     limits.exposure_max_ms))
    if abs(clamped_exposure - exposure_ms) > 1e-6:
        reasons.append(f"exposure clamped to {clamped_exposure:.0f}ms")

    return float(voltage_kv), clamped_exposure, float(standoff_m), reasons
