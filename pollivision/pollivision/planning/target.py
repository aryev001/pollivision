"""Target selection and approach planning (report Sec. III-D).

Given every flower the perception stack accepted this frame, decide which one to
approach next and where to put the probe. This module deliberately does *not*
plan paths - obstacle avoidance and differential-drive trajectory generation
belong to the rover's navigation stack. What it produces is the goal that stack
is given: a flower, a probe pose in the rover frame, and the reason it was
chosen.

Selection is a weighted score rather than nearest-first, because nearest-first
is a poor policy here. The rover's throughput is limited by pollen logistics: it
must collect from a staminate flower before any pistillate flower is worth
visiting, and a collection from a depleted anther wastes the trip. Encoding that
as an explicit ``pollen_need`` term lets the same scoring function express both
"go find pollen" and "go spend it" without a separate mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..geometry import transforms
from ..logging_utils import get_logger
from ..types import FlowerObservation, FlowerSex, ProbeMode

LOGGER = get_logger(__name__)


@dataclass
class TargetPlan:
    """The chosen flower and how to approach it."""

    flower: FlowerObservation
    mode: ProbeMode
    score: float
    probe_position: np.ndarray            # rover frame, metres
    approach_direction: np.ndarray        # rover frame, unit vector
    standoff_m: float
    reason: str = ""
    scores: dict[str, float] = None       # per-term contributions, for logging

    @property
    def range_m(self) -> float:
        return float(np.linalg.norm(self.probe_position))


class TargetSelector:
    """Scores candidate flowers and picks the next pollination target."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        weights = cfg.section("planning.weights")
        self.w_receptivity = float(weights.get("receptivity", 2.0))
        self.w_quality = float(weights.get("quality", 1.2))
        self.w_proximity = float(weights.get("proximity", 1.0))
        self.w_pollen_need = float(weights.get("pollen_need", 1.5))
        self.w_centrality = float(weights.get("centrality", 0.4))
        self.w_occlusion = float(weights.get("occlusion_penalty", 1.4))

        self.max_range = float(cfg.get("planning.max_range_m", 1.2))
        self.min_range = float(cfg.get("planning.min_range_m", 0.05))
        self.standoff = float(cfg.get("planning.approach_standoff_m", 0.04))
        self.max_incidence = float(cfg.get("planning.max_incidence_deg", 55.0))
        self.probe_capacity = int(cfg.get("planning.probe_capacity", 4))

    # ------------------------------------------------------------------ #

    def select(
        self,
        flowers: list[FlowerObservation],
        probe_charge: float,
        ledger=None,
        frame_shape: Optional[tuple[int, int]] = None,
    ) -> Optional[TargetPlan]:
        """Choose the next target, or None if nothing is worth approaching."""
        best: Optional[TargetPlan] = None

        for flower in flowers:
            plan = self._score(flower, probe_charge, ledger, frame_shape)
            if plan is None:
                continue
            if best is None or plan.score > best.score:
                best = plan

        if best is not None:
            LOGGER.debug("Selected track %s (%s, %s) score %.3f: %s",
                         best.flower.track_id, best.flower.sex.sex.value,
                         best.mode.value, best.score, best.reason)
        return best

    def _score(
        self,
        flower: FlowerObservation,
        probe_charge: float,
        ledger,
        frame_shape: Optional[tuple[int, int]],
    ) -> Optional[TargetPlan]:
        if not flower.is_actionable:
            return None
        if flower.rover_pose is None or flower.range_m is None:
            return None

        if not (self.min_range <= flower.range_m <= self.max_range):
            return None

        # An obliquely presented corolla needs an impractical drive voltage and
        # is better approached from a different rover position later.
        if flower.incidence_deg is not None and flower.incidence_deg > self.max_incidence:
            return None

        mode = self._mode_for(flower, probe_charge)
        if mode is ProbeMode.IDLE:
            return None

        if ledger is not None:
            skip, why = ledger.should_skip(
                flower.track_id,
                flower.rover_pose.position if flower.rover_pose else None,
            )
            if skip:
                return None

        terms: dict[str, float] = {}

        terms["receptivity"] = self.w_receptivity * float(
            np.clip(flower.anthesis.confidence, 0.0, 1.0))

        quality = 0.5 * float(np.clip(flower.quality.sharpness, 0.0, 1.0)) + \
            0.5 * float(np.clip(flower.sex.confidence, 0.0, 1.0))
        terms["quality"] = self.w_quality * quality

        # Nearer is better, normalised over the usable range window.
        proximity = 1.0 - (flower.range_m - self.min_range) / \
            max(self.max_range - self.min_range, 1e-6)
        terms["proximity"] = self.w_proximity * float(np.clip(proximity, 0.0, 1.0))

        terms["pollen_need"] = self.w_pollen_need * self._pollen_need(
            flower, mode, probe_charge)

        terms["centrality"] = self.w_centrality * self._centrality(flower, frame_shape)

        terms["occlusion"] = -self.w_occlusion * float(
            np.clip(flower.quality.occlusion, 0.0, 1.0))

        score = float(sum(terms.values()))

        probe_position, approach = transforms.approach_vector(
            flower.rover_pose, self.standoff)

        return TargetPlan(
            flower=flower,
            mode=mode,
            score=score,
            probe_position=probe_position,
            approach_direction=approach,
            standoff_m=self.standoff,
            reason=self._describe(flower, mode, probe_charge),
            scores={k: round(v, 4) for k, v in terms.items()},
        )

    def _mode_for(self, flower: FlowerObservation, probe_charge: float) -> ProbeMode:
        if flower.sex.sex is FlowerSex.MALE:
            # Only worth stopping at an anther that still carries usable pollen
            # and only if the probe has room for it.
            if probe_charge < 0.85 and flower.pollen.availability > 0.15:
                return ProbeMode.COLLECT
            return ProbeMode.IDLE
        if flower.sex.sex is FlowerSex.FEMALE:
            if probe_charge >= 1.0 / max(self.probe_capacity, 1):
                return ProbeMode.DEPOSIT
            return ProbeMode.IDLE
        return ProbeMode.IDLE

    def _pollen_need(self, flower: FlowerObservation, mode: ProbeMode,
                     probe_charge: float) -> float:
        """How badly the rover wants to visit this flower *right now*.

        This is the term that gives the rover its collect-then-spend rhythm. An
        empty probe makes a loaded anther urgent; a full probe makes a receptive
        female urgent. Both fall away as the need is met.
        """
        if mode is ProbeMode.COLLECT:
            deficit = float(np.clip(1.0 - probe_charge, 0.0, 1.0))
            richness = float(np.clip(flower.pollen.availability, 0.0, 1.0))
            return deficit * richness
        if mode is ProbeMode.DEPOSIT:
            # Rises with how much pollen is on hand to spend.
            return float(np.clip(probe_charge, 0.0, 1.0))
        return 0.0

    @staticmethod
    def _centrality(flower: FlowerObservation,
                    frame_shape: Optional[tuple[int, int]]) -> float:
        """Prefer targets near the optical axis: less steering, less occlusion."""
        if frame_shape is None:
            return 0.0
        height, width = frame_shape[:2]
        dx = (flower.box.cx - width / 2.0) / max(width / 2.0, 1e-6)
        dy = (flower.box.cy - height / 2.0) / max(height / 2.0, 1e-6)
        return float(np.clip(1.0 - np.hypot(dx, dy), 0.0, 1.0))

    @staticmethod
    def _describe(flower: FlowerObservation, mode: ProbeMode,
                  probe_charge: float) -> str:
        if mode is ProbeMode.COLLECT:
            return (f"collect from staminate flower "
                    f"(anther load {flower.pollen.availability:.2f}, "
                    f"probe at {probe_charge:.2f})")
        return (f"deposit onto pistillate flower "
                f"(probe carrying {probe_charge:.2f})")
