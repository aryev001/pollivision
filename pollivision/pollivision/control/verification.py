"""Pollination outcome verification (report Sec. III-G).

Fig. 1 closes its loop with a verification step: "a verification step confirms
the outcome and feeds the result back into the navigation module to select the
next target flower". Without it the rover is running open-loop and cannot tell a
successful transfer from a probe that fired into empty air.

Verification is differential. Absolute appearance is far too sensitive to
viewpoint, exposure and white balance to support a threshold, but the *change*
across a single attempt - same flower, same camera, seconds apart - cancels most
of that out:

* **After collection** from a male anther, the anther should read measurably
  emptier: the pollen-availability estimate should drop.
* **After deposition** onto a female stigma, the stigma should read measurably
  more loaded: pollen-coloured material should appear on it.

Both comparisons are made against a baseline captured immediately before the
attempt, and both are gated on the frames being comparable in the first place -
if the flower moved out of view or the exposure swung, the result is reported as
inconclusive rather than as a failure, since punishing a flower for a bad frame
would make the rover give up on perfectly good targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..logging_utils import get_logger
from ..perception.stigma import stigma_pollen_load
from ..types import BBox, FlowerObservation, ProbeMode, VerificationResult

LOGGER = get_logger(__name__)


@dataclass
class AttemptBaseline:
    """State captured just before an attempt, for later comparison."""

    track_id: int
    mode: ProbeMode
    pollen_availability: float = 0.0
    stigma_load: float = 0.0
    stigma_box: Optional[BBox] = None
    exposure: float = 1.0
    frame_index: int = 0
    meta: dict = field(default_factory=dict)


class OutcomeVerifier:
    """Compares before/after observations to judge a pollination attempt."""

    def __init__(self, cfg) -> None:
        section = cfg.section("verification")
        self.enabled = bool(section.get("enabled", True))
        self.min_pollen_drop = float(section.get("min_pollen_drop", 0.08))
        self.min_stigma_gain = float(section.get("min_stigma_gain", 0.05))
        self.settle_frames = int(section.get("settle_frames", 3))
        self._baselines: dict[int, AttemptBaseline] = {}

    # ------------------------------------------------------------------ #

    def capture_baseline(self, frame: np.ndarray, flower: FlowerObservation,
                         mode: ProbeMode, frame_index: int = 0) -> Optional[AttemptBaseline]:
        """Record the pre-attempt state of a flower."""
        if not self.enabled or flower.track_id is None:
            return None

        stigma_box = flower.meta.get("stigma_box")
        baseline = AttemptBaseline(
            track_id=flower.track_id,
            mode=mode,
            pollen_availability=float(flower.pollen.availability),
            stigma_load=stigma_pollen_load(frame, stigma_box),
            stigma_box=stigma_box,
            exposure=float(flower.quality.exposure),
            frame_index=frame_index,
        )
        self._baselines[flower.track_id] = baseline
        return baseline

    def verify(self, frame: np.ndarray, flower: FlowerObservation,
               frame_index: int = 0) -> VerificationResult:
        """Judge the outcome of the attempt on this flower."""
        if not self.enabled:
            return VerificationResult(success=True, confidence=0.0,
                                      note="verification disabled")
        if flower.track_id is None:
            return VerificationResult(note="no track identity to verify against")

        baseline = self._baselines.get(flower.track_id)
        if baseline is None:
            return VerificationResult(note="no baseline captured")

        if frame_index - baseline.frame_index < self.settle_frames:
            # Firing the probe disturbs the flower; measuring before it settles
            # reads motion blur as a change in pollen load.
            return VerificationResult(note="waiting for the flower to settle")

        # A large exposure swing between the two frames invalidates the
        # comparison, since both measurements are chroma-based.
        exposure_shift = abs(flower.quality.exposure - baseline.exposure)
        if exposure_shift > 0.35:
            return VerificationResult(
                confidence=0.0,
                note=f"inconclusive: exposure shifted by {exposure_shift:.2f}",
            )

        if baseline.mode is ProbeMode.COLLECT:
            return self._verify_collection(flower, baseline)
        if baseline.mode is ProbeMode.DEPOSIT:
            return self._verify_deposition(frame, flower, baseline)
        return VerificationResult(note="nothing to verify for an idle attempt")

    # ------------------------------------------------------------------ #

    def _verify_collection(self, flower: FlowerObservation,
                           baseline: AttemptBaseline) -> VerificationResult:
        delta = float(flower.pollen.availability - baseline.pollen_availability)
        success = delta <= -self.min_pollen_drop

        # Confidence scales with how far past the threshold the drop went, and
        # is discounted by how much we trust the two measurements themselves.
        magnitude = min(abs(delta) / max(self.min_pollen_drop * 3.0, 1e-6), 1.0)
        reliability = float(np.clip(flower.pollen.confidence, 0.0, 1.0))
        confidence = float(np.clip(magnitude * (0.4 + 0.6 * reliability), 0.0, 1.0))

        if success:
            note = f"anther depleted by {abs(delta):.3f}"
        elif delta > self.min_pollen_drop:
            # Pollen apparently increased. Almost always a viewpoint change
            # revealing more of the anther, not an actual gain.
            note = f"pollen reading rose by {delta:.3f}; likely a viewpoint change"
            confidence *= 0.5
        else:
            note = f"no measurable depletion (delta {delta:+.3f})"

        return VerificationResult(success=success, confidence=confidence,
                                  pollen_delta=delta, note=note)

    def _verify_deposition(self, frame: np.ndarray, flower: FlowerObservation,
                           baseline: AttemptBaseline) -> VerificationResult:
        stigma_box = flower.meta.get("stigma_box") or baseline.stigma_box
        if stigma_box is None:
            return VerificationResult(note="stigma not localised; cannot verify")

        current = stigma_pollen_load(frame, stigma_box)
        delta = float(current - baseline.stigma_load)
        success = delta >= self.min_stigma_gain

        magnitude = min(abs(delta) / max(self.min_stigma_gain * 3.0, 1e-6), 1.0)
        stigma_confidence = float(np.clip(flower.meta.get("stigma_confidence", 0.5), 0.0, 1.0))
        confidence = float(np.clip(magnitude * (0.4 + 0.6 * stigma_confidence), 0.0, 1.0))

        note = (f"stigma pollen load rose by {delta:.3f}" if success
                else f"no measurable deposition (delta {delta:+.3f})")
        return VerificationResult(success=success, confidence=confidence,
                                  stigma_delta=delta, note=note)

    def clear(self, track_id: int) -> None:
        self._baselines.pop(track_id, None)

    def reset(self) -> None:
        self._baselines.clear()
