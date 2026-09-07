"""Annotated overlay rendering.

Debugging a multi-stage perception stack from numbers alone is slow. This
renderer draws every intermediate the pipeline produces - the corolla mask and
its fitted ellipse, the located ovary, the anther or stigma aim point, the sex
posterior with its per-cue breakdown, range and incidence - so that a single
frame shows which stage went wrong rather than only that something did.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..types import (
    ElectrostaticCommand,
    FlowerObservation,
    FlowerSex,
    FrameResult,
)

# BGR. Female/male are the two colours the eye must separate instantly.
COLOR_FEMALE = (180, 90, 235)
COLOR_MALE = (235, 180, 60)
COLOR_UNKNOWN = (150, 150, 150)
COLOR_REJECT = (90, 90, 90)
COLOR_OVARY = (90, 220, 120)
COLOR_AIM = (60, 235, 255)
COLOR_TEXT = (245, 245, 245)
COLOR_PANEL = (28, 28, 28)


def sex_color(sex: FlowerSex) -> tuple[int, int, int]:
    if sex is FlowerSex.FEMALE:
        return COLOR_FEMALE
    if sex is FlowerSex.MALE:
        return COLOR_MALE
    return COLOR_UNKNOWN


def draw_frame(
    frame: np.ndarray,
    result: FrameResult,
    command: Optional[ElectrostaticCommand] = None,
    state: str = "",
    message: str = "",
    show_rejected: bool = True,
    show_masks: bool = True,
) -> np.ndarray:
    """Render a fully annotated copy of ``frame``."""
    canvas = frame.copy()

    if show_rejected:
        for observation in result.rejected:
            x1, y1, x2, y2 = observation.box.as_int()
            cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_REJECT, 1)
            reasons = ",".join(r.value for r in observation.quality.reasons[:2])
            _label(canvas, f"rejected: {reasons}", (x1, max(12, y1 - 4)),
                   COLOR_REJECT, scale=0.35)

    if show_masks:
        canvas = _blend_masks(canvas, result.flowers)

    for observation in result.flowers:
        _draw_flower(canvas, observation)

    _draw_panel(canvas, result, command, state, message)
    return canvas


def _blend_masks(canvas: np.ndarray, flowers: list[FlowerObservation]) -> np.ndarray:
    """Tint each corolla mask by sex, in one blend to keep it cheap."""
    overlay = np.zeros_like(canvas)
    painted = False
    for observation in flowers:
        if observation.mask is None or observation.mask.shape[:2] != canvas.shape[:2]:
            continue
        overlay[observation.mask] = sex_color(observation.sex.sex)
        painted = True
    if not painted:
        return canvas
    return cv2.addWeighted(canvas, 1.0, overlay, 0.25, 0.0)


def _draw_flower(canvas: np.ndarray, observation: FlowerObservation) -> None:
    color = sex_color(observation.sex.sex)
    x1, y1, x2, y2 = observation.box.as_int()
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

    if observation.ellipse is not None:
        ellipse = observation.ellipse
        cv2.ellipse(
            canvas,
            (int(ellipse.cx), int(ellipse.cy)),
            (max(1, int(ellipse.major / 2)), max(1, int(ellipse.minor / 2))),
            ellipse.angle_deg, 0, 360, color, 1,
        )

    if observation.sex.ovary_box is not None:
        ox1, oy1, ox2, oy2 = observation.sex.ovary_box.as_int()
        cv2.rectangle(canvas, (ox1, oy1), (ox2, oy2), COLOR_OVARY, 1)
        _label(canvas, "ovary", (ox1, oy2 + 12), COLOR_OVARY, scale=0.35)

    if observation.aim_point is not None:
        ax, ay = int(observation.aim_point[0]), int(observation.aim_point[1])
        cv2.drawMarker(canvas, (ax, ay), COLOR_AIM, cv2.MARKER_CROSS, 14, 2)

    track = f"#{observation.track_id} " if observation.track_id is not None else ""
    header = (f"{track}{observation.sex.sex.value} "
              f"p={observation.sex.p_female:.2f} c={observation.sex.confidence:.2f}")
    _label(canvas, header, (x1, max(12, y1 - 18)), color)

    detail = f"{observation.anthesis.stage.value}"
    if observation.sex.sex is FlowerSex.MALE:
        detail += f" pollen={observation.pollen.availability:.2f}"
    if observation.range_m is not None:
        detail += f" {observation.range_m * 100:.0f}cm"
    if observation.incidence_deg is not None:
        detail += f" {observation.incidence_deg:.0f}deg"
    _label(canvas, detail, (x1, max(24, y1 - 5)), color, scale=0.4)


def _draw_panel(canvas: np.ndarray, result: FrameResult,
                command: Optional[ElectrostaticCommand],
                state: str, message: str) -> None:
    """Status panel across the top of the frame."""
    lines = [
        f"frame {result.frame_index}  "
        f"flowers {len(result.flowers)} (M{len(result.males)}/F{len(result.females)})  "
        f"rejected {len(result.rejected)}  "
        f"{result.latency_ms.get('total', 0.0):.0f} ms",
    ]
    if state or message:
        lines.append(f"{state}: {message}"[:110])
    if command is not None and command.voltage_kv > 0:
        clamp = f"  [{'; '.join(command.clamped)}]" if command.clamped else ""
        lines.append(
            f"HV {command.mode.value} {command.voltage_kv:.2f} kV  "
            f"{command.exposure_ms:.0f} ms  pol {command.polarity:+d}  "
            f"gap {command.standoff_m * 1000:.0f} mm  "
            f"target {command.predicted_field_kv_per_m:.0f} kV/m{clamp}"[:110]
        )

    height = 8 + 18 * len(lines)
    panel = canvas[0:height, 0:canvas.shape[1]]
    cv2.rectangle(panel, (0, 0), (canvas.shape[1], height), COLOR_PANEL, -1)
    cv2.addWeighted(panel, 0.75, canvas[0:height, 0:canvas.shape[1]], 0.25, 0,
                    canvas[0:height, 0:canvas.shape[1]])
    for index, line in enumerate(lines):
        _label(canvas, line, (8, 16 + 18 * index), COLOR_TEXT, scale=0.45)


def _label(canvas: np.ndarray, text: str, origin: tuple[int, int],
           color: tuple[int, int, int], scale: float = 0.45) -> None:
    """Draw text with a dark outline so it stays legible over any background."""
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def draw_breakdown(canvas: np.ndarray, observation: FlowerObservation,
                   origin: tuple[int, int] = (8, 120)) -> np.ndarray:
    """Render the per-cue evidence behind one flower's sex posterior.

    This is the view that makes a misclassification diagnosable: it shows
    whether the morphology cue, the vision-language cue or the detector
    association was the one that pulled the decision the wrong way.
    """
    x, y = origin
    lines = [f"track #{observation.track_id} sex evidence (log-odds):"]
    for name, value in observation.sex.breakdown.cues.items():
        weight = observation.sex.breakdown.weights.get(name, 1.0)
        arrow = "F" if value > 0 else ("M" if value < 0 else "-")
        lines.append(f"  {name:<16} {value:+.3f} (w={weight:.2f}) -> {arrow}")
    lines.append(f"  fused {observation.sex.breakdown.fused_logit:+.3f} "
                 f"= p(female) {observation.sex.p_female:.3f}")

    for index, line in enumerate(lines):
        _label(canvas, line, (x, y + 16 * index), COLOR_TEXT, scale=0.42)
    return canvas
