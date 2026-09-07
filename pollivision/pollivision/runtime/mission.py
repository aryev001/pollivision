"""Mission state machine: the closed perceive-plan-act-verify loop of Fig. 1.

The report describes the rover as running a closed loop in which "visual
perception drives navigation, navigation triggers environmental sensing, sensing
parameterizes the electrostatic controller, and the outcome of each pollination
attempt is fed back to guide the rover toward the next target flower". This
class is that loop.

It is deliberately a pure state machine over perception results and sensor
readings: it emits *commands* (approach this pose, fire these electrostatic
parameters) and consumes *outcomes*, but never touches motors or high-voltage
hardware itself. That separation is what makes the whole loop testable offline
against recorded video, which matters a great deal for a system whose actuator
is a kilovolt probe operating near living tissue.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np

from ..control.electrostatic import AdaptiveElectrostaticController
from ..control.verification import OutcomeVerifier
from ..logging_utils import get_logger
from ..planning.target import TargetPlan, TargetSelector
from ..tracking.ledger import PollinationLedger
from ..types import (
    ElectrostaticCommand,
    EnvironmentReading,
    FrameResult,
    ProbeMode,
    VerificationResult,
)

LOGGER = get_logger(__name__)


class MissionState(str, Enum):
    """Where the rover is in the pollination cycle."""

    SEARCH = "search"        # sweeping the bed, nothing selected
    APPROACH = "approach"    # driving the probe toward a chosen flower
    ACTUATE = "actuate"      # probe in position, firing
    VERIFY = "verify"        # checking the outcome
    RECOVER = "recover"      # backing off after a failure or a lost target


@dataclass
class MissionStep:
    """What the rover should do next, plus why."""

    state: MissionState
    plan: Optional[TargetPlan] = None
    command: Optional[ElectrostaticCommand] = None
    verification: Optional[VerificationResult] = None
    message: str = ""
    stats: dict = field(default_factory=dict)

    def to_wire(self) -> dict:
        """Compact dict for the link to the rover's motion controller."""
        payload: dict = {"state": self.state.value, "msg": self.message}
        if self.plan is not None:
            payload["goal"] = {
                "track": self.plan.flower.track_id,
                "sex": self.plan.flower.sex.sex.value,
                "mode": self.plan.mode.value,
                "pos_m": [round(float(v), 4) for v in self.plan.probe_position],
                "dir": [round(float(v), 4) for v in self.plan.approach_direction],
                "standoff_m": round(self.plan.standoff_m, 4),
            }
        if self.command is not None:
            payload["hv"] = self.command.to_wire()
        return payload


class MissionController:
    """Drives the perceive-plan-act-verify cycle."""

    #: How close the probe must be to its commanded pose before firing.
    POSE_TOLERANCE_M = 0.012

    #: Frames a target may go unseen during approach before it is abandoned.
    LOST_TARGET_PATIENCE = 12

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.selector = TargetSelector(cfg)
        self.controller = AdaptiveElectrostaticController(cfg)
        self.verifier = OutcomeVerifier(cfg)
        self.ledger = PollinationLedger(cfg)

        self.state = MissionState.SEARCH
        self.active: Optional[TargetPlan] = None
        self._lost_frames = 0
        self._fired_at_frame: Optional[int] = None
        self._probe_capacity = int(cfg.get("planning.probe_capacity", 4))

    # ------------------------------------------------------------------ #

    def step(
        self,
        result: FrameResult,
        environment: EnvironmentReading,
        frame: Optional[np.ndarray] = None,
        probe_at_pose: bool = False,
        fire: Optional[Callable[[ElectrostaticCommand], bool]] = None,
    ) -> MissionStep:
        """Advance the mission by one perception frame.

        Args:
            result: This frame's perception output.
            environment: Current humidity / temperature / ToF reading.
            frame: The BGR frame, needed for verification measurements.
            probe_at_pose: Whether the motion controller reports the probe has
                reached the commanded pose. Supplied by the rover; the vision
                stack cannot observe its own end-effector.
            fire: Callback that actuates the hardware and returns success. When
                omitted the step is simulated, which is what lets the whole loop
                run against recorded video.

        Returns:
            The action the rover should take next.
        """
        if self.state is MissionState.SEARCH:
            return self._search(result)
        if self.state is MissionState.APPROACH:
            return self._approach(result, environment, probe_at_pose, frame, fire)
        if self.state is MissionState.ACTUATE:
            return self._actuate(result, environment, frame, fire)
        if self.state is MissionState.VERIFY:
            return self._verify(result, frame)
        return self._recover(result)

    # ------------------------------------------------------------------ #

    def _search(self, result: FrameResult) -> MissionStep:
        plan = self.selector.select(
            result.flowers,
            probe_charge=self.ledger.probe_charge,
            ledger=self.ledger,
            frame_shape=(result.height, result.width),
        )
        if plan is None:
            return MissionStep(
                state=MissionState.SEARCH,
                message=self._idle_reason(result),
                stats=self.ledger.stats(),
            )

        self.active = plan
        self._lost_frames = 0
        self.state = MissionState.APPROACH
        return MissionStep(
            state=MissionState.APPROACH, plan=plan,
            message=f"target acquired: {plan.reason}",
            stats=self.ledger.stats(),
        )

    def _approach(self, result, environment, probe_at_pose, frame, fire) -> MissionStep:
        plan = self._refresh_target(result)
        if plan is None:
            self.state = MissionState.RECOVER
            return MissionStep(state=MissionState.RECOVER,
                               message="lost the target during approach")

        if not probe_at_pose:
            return MissionStep(
                state=MissionState.APPROACH, plan=plan,
                message=f"approaching, {plan.range_m * 100:.1f} cm to probe pose",
            )

        self.state = MissionState.ACTUATE
        return self._actuate(result, environment, frame, fire)

    def _actuate(self, result, environment, frame, fire) -> MissionStep:
        plan = self._refresh_target(result)
        if plan is None:
            self.state = MissionState.RECOVER
            return MissionStep(state=MissionState.RECOVER,
                               message="target lost before actuation")

        flower = plan.flower
        # Deposition parameters depend on how loaded the *probe* is, not on the
        # pollen visible on the pistillate flower being approached.
        if plan.mode is ProbeMode.DEPOSIT:
            flower.meta["probe_charge"] = self.ledger.probe_charge

        command = self.controller.compute(flower, environment, plan.mode,
                                          standoff_m=plan.standoff_m)

        if frame is not None:
            self.verifier.capture_baseline(frame, flower, plan.mode, result.frame_index)

        if flower.track_id is not None:
            self.ledger.record_attempt(
                flower.track_id, flower.sex.sex,
                flower.rover_pose.position if flower.rover_pose else None,
            )

        actuated = True
        if fire is not None:
            try:
                actuated = bool(fire(command))
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Actuation callback raised: %s", exc)
                actuated = False

        if not actuated:
            self.state = MissionState.RECOVER
            return MissionStep(state=MissionState.RECOVER, plan=plan, command=command,
                               message="actuation reported failure")

        self._fired_at_frame = result.frame_index
        self.state = MissionState.VERIFY
        return MissionStep(
            state=MissionState.VERIFY, plan=plan, command=command,
            message=(f"fired {command.mode.value} at {command.voltage_kv:.2f} kV "
                     f"for {command.exposure_ms:.0f} ms"),
        )

    def _verify(self, result: FrameResult, frame) -> MissionStep:
        plan = self.active
        if plan is None:
            self.state = MissionState.SEARCH
            return MissionStep(state=MissionState.SEARCH, message="nothing to verify")

        current = self._find_track(result, plan.flower.track_id) or plan.flower

        if frame is None:
            # No imagery to verify against: assume success rather than block the
            # mission, but say so, since the ledger entry is then unverified.
            outcome = VerificationResult(success=True, confidence=0.0,
                                         note="no frame available to verify against")
        else:
            outcome = self.verifier.verify(frame, current, result.frame_index)
            if outcome.note.startswith("waiting"):
                return MissionStep(state=MissionState.VERIFY, plan=plan,
                                   verification=outcome, message=outcome.note)

        self._settle_outcome(plan, outcome)

        self.state = MissionState.SEARCH
        self.active = None
        return MissionStep(
            state=MissionState.SEARCH, plan=plan, verification=outcome,
            message=f"{'success' if outcome.success else 'failed'}: {outcome.note}",
            stats=self.ledger.stats(),
        )

    def _settle_outcome(self, plan: TargetPlan, outcome: VerificationResult) -> None:
        """Update the ledger and the probe's pollen budget after an attempt."""
        track_id = plan.flower.track_id
        if track_id is not None:
            self.ledger.record_outcome(track_id, outcome.success, outcome.note)
            self.verifier.clear(track_id)

        if not outcome.success:
            return

        if plan.mode is ProbeMode.COLLECT and track_id is not None:
            # The probe now carries roughly what the anther lost, capped at one.
            collected = min(1.0, max(plan.flower.pollen.availability,
                                     abs(outcome.pollen_delta)))
            self.ledger.load_probe(track_id, collected)
        elif plan.mode is ProbeMode.DEPOSIT:
            # Each deposition spends a share of the load; capacity sets how many
            # pistillate flowers one collection is expected to serve.
            self.ledger.consume_probe(1.0 / max(self._probe_capacity, 1))

    def _recover(self, result: FrameResult) -> MissionStep:
        self.active = None
        self._lost_frames = 0
        self.state = MissionState.SEARCH
        return MissionStep(state=MissionState.SEARCH,
                           message="recovered; resuming search",
                           stats=self.ledger.stats())

    # ------------------------------------------------------------------ #

    def _refresh_target(self, result: FrameResult) -> Optional[TargetPlan]:
        """Re-locate the active target in the current frame.

        The rover is moving, so the target's pose must be re-read every frame
        rather than trusted from acquisition. A target that disappears briefly
        (a leaf swings across it) is tolerated for a few frames before being
        abandoned, since giving up instantly would make the rover thrash.
        """
        if self.active is None:
            return None

        refreshed = self._find_track(result, self.active.flower.track_id)
        if refreshed is None:
            self._lost_frames += 1
            if self._lost_frames > self.LOST_TARGET_PATIENCE:
                return None
            return self.active

        self._lost_frames = 0
        plan = self.active
        plan.flower = refreshed
        if refreshed.rover_pose is not None:
            from ..geometry import transforms

            position, direction = transforms.approach_vector(
                refreshed.rover_pose, plan.standoff_m)
            plan.probe_position = position
            plan.approach_direction = direction
        self.active = plan
        return plan

    @staticmethod
    def _find_track(result: FrameResult, track_id: Optional[int]):
        if track_id is None:
            return None
        for flower in result.flowers:
            if flower.track_id == track_id:
                return flower
        return None

    @staticmethod
    def _idle_reason(result: FrameResult) -> str:
        """Explain why nothing was selected - the most common field question."""
        if not result.flowers and not result.rejected:
            return "searching: no flowers detected"
        if not result.flowers:
            reasons: dict[str, int] = {}
            for observation in result.rejected:
                for reason in observation.quality.reasons:
                    reasons[reason.value] = reasons.get(reason.value, 0) + 1
            summary = ", ".join(f"{k}x{v}" for k, v in sorted(reasons.items()))
            return f"searching: all {len(result.rejected)} candidates rejected ({summary})"
        return (f"searching: {len(result.flowers)} flowers visible, "
                "none currently actionable")

    def reset(self) -> None:
        self.state = MissionState.SEARCH
        self.active = None
        self._lost_frames = 0
        self._fired_at_frame = None
        self.verifier.reset()
