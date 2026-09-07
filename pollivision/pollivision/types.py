"""Core data types shared across the PolliVision perception and control stack.

Everything that crosses a module boundary is one of these frozen-ish dataclasses
so that the pipeline stages stay decoupled and individually testable.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class FlowerSex(str, Enum):
    """Sex of a cucurbit flower.

    Cucurbits are monoecious: a single plant carries separate staminate (male,
    pollen-bearing) and pistillate (female, ovary-bearing) flowers. Selective
    pollen transfer therefore requires the rover to tell them apart, which is
    the discriminative task described in Sec. III-C of the report.
    """

    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


class AnthesisStage(str, Enum):
    """Developmental / receptivity stage of a flower.

    Cucurbit flowers are typically anthetic for only a single morning, so a
    pollination rover that ignores stage wastes most of its actuation budget.
    """

    BUD = "bud"                # closed, not yet receptive
    OPENING = "opening"        # partially open, approaching receptivity
    RECEPTIVE = "receptive"    # fully open, the only pollination-worthy state
    SENESCENT = "senescent"    # wilting / closed again, no longer viable
    UNKNOWN = "unknown"

    @property
    def is_viable(self) -> bool:
        return self is AnthesisStage.RECEPTIVE


class ProbeMode(str, Enum):
    """What the electrostatic end-effector is being asked to do."""

    COLLECT = "collect"  # charge probe, attract pollen off a male anther
    DEPOSIT = "deposit"  # discharge/reverse near a female stigma
    IDLE = "idle"


class RejectReason(str, Enum):
    """Why a candidate detection was dropped by the quality gate."""

    LOW_CONFIDENCE = "low_confidence"
    OCCLUDED = "occluded"
    TRUNCATED = "truncated"
    BLURRED = "blurred"
    EXPOSURE = "exposure"
    TOO_SMALL = "too_small"
    OUT_OF_RANGE = "out_of_range"
    NOT_RECEPTIVE = "not_receptive"


# --------------------------------------------------------------------------- #
# Geometry primitives
# --------------------------------------------------------------------------- #


@dataclass
class BBox:
    """Axis-aligned bounding box in pixel coordinates (x1, y1, x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    @property
    def center(self) -> tuple[float, float]:
        return (self.cx, self.cy)

    def as_array(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)

    def as_int(self) -> tuple[int, int, int, int]:
        return (int(round(self.x1)), int(round(self.y1)),
                int(round(self.x2)), int(round(self.y2)))

    def clip(self, w: int, h: int) -> "BBox":
        return BBox(
            float(np.clip(self.x1, 0, w - 1)),
            float(np.clip(self.y1, 0, h - 1)),
            float(np.clip(self.x2, 0, w - 1)),
            float(np.clip(self.y2, 0, h - 1)),
        )

    def scaled(self, factor: float, w: Optional[int] = None,
               h: Optional[int] = None, bias_down: float = 0.0) -> "BBox":
        """Grow the box about its centre by ``factor``.

        ``bias_down`` shifts the expansion downward as a fraction of the added
        height. This matters for sex classification: the ovary that identifies a
        female flower sits *below* the corolla, so the context crop must reach
        further down than up.
        """
        cx, cy = self.cx, self.cy
        nw, nh = self.width * factor, self.height * factor
        dy = bias_down * (nh - self.height) * 0.5
        box = BBox(cx - nw / 2, cy - nh / 2 + dy, cx + nw / 2, cy + nh / 2 + dy)
        if w is not None and h is not None:
            box = box.clip(w, h)
        return box

    def iou(self, other: "BBox") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 1e-9 else 0.0

    @staticmethod
    def from_xyxy(a) -> "BBox":
        return BBox(float(a[0]), float(a[1]), float(a[2]), float(a[3]))


@dataclass
class Ellipse:
    """Fitted corolla ellipse, used for orientation and tilt-invariant ranging."""

    cx: float
    cy: float
    major: float   # full length of the major axis, in pixels
    minor: float   # full length of the minor axis, in pixels
    angle_deg: float  # orientation of the major axis, CCW from +x

    @property
    def axis_ratio(self) -> float:
        """minor/major in [0, 1]; equals cos(tilt) for a circular corolla."""
        return float(np.clip(self.minor / max(self.major, 1e-6), 0.0, 1.0))

    @property
    def tilt_rad(self) -> float:
        """Angle between the corolla face normal and the camera optical axis."""
        return float(math.acos(np.clip(self.axis_ratio, 0.0, 1.0)))


@dataclass
class Pose3D:
    """A point plus a surface normal, expressed in a named reference frame."""

    position: np.ndarray          # (3,) metres
    normal: Optional[np.ndarray] = None   # (3,) unit vector, corolla face direction
    frame: str = "camera"

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)
        if self.normal is not None:
            n = np.asarray(self.normal, dtype=np.float64).reshape(3)
            norm = np.linalg.norm(n)
            self.normal = n / norm if norm > 1e-9 else None

    @property
    def range_m(self) -> float:
        return float(np.linalg.norm(self.position))


# --------------------------------------------------------------------------- #
# Perception outputs
# --------------------------------------------------------------------------- #


@dataclass
class Detection:
    """A raw candidate emitted by one detector backend, before fusion."""

    box: BBox
    score: float
    label: str
    source: str = "unknown"                  # which backend produced it
    mask: Optional[np.ndarray] = None        # bool array, full-frame resolution
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreBreakdown:
    """Per-cue evidence behind a fused probability.

    Kept alongside every classification so that field failures can be traced to
    the cue that misfired rather than to an opaque ensemble score.
    """

    cues: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    fused_logit: float = 0.0

    def add(self, name: str, logit: float, weight: float) -> None:
        self.cues[name] = float(logit)
        self.weights[name] = float(weight)


@dataclass
class SexEstimate:
    sex: FlowerSex = FlowerSex.UNKNOWN
    p_female: float = 0.5
    confidence: float = 0.0
    breakdown: ScoreBreakdown = field(default_factory=ScoreBreakdown)
    ovary_box: Optional[BBox] = None   # located inferior ovary, when found


@dataclass
class AnthesisEstimate:
    stage: AnthesisStage = AnthesisStage.UNKNOWN
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class PollenEstimate:
    """Visual estimate of pollen availability on a male anther (Sec. III-E-iii)."""

    availability: float = 0.0     # 0 = depleted, 1 = fully loaded
    confidence: float = 0.0
    anther_box: Optional[BBox] = None
    area_fraction: float = 0.0    # fraction of the inner corolla that reads as anther
    granularity: float = 0.0      # texture energy; fresh pollen is grainy
    freshness: float = 1.0        # 1 = bright yellow, lower = browning/spent
    breakdown: ScoreBreakdown = field(default_factory=ScoreBreakdown)


@dataclass
class QualityAssessment:
    """Output of the occlusion / image-quality gate (Sec. III-C)."""

    accepted: bool = True
    reasons: list[RejectReason] = field(default_factory=list)
    occlusion: float = 0.0     # fraction of the corolla judged to be covered
    truncation: float = 0.0    # fraction of the box clipped by the frame edge
    sharpness: float = 1.0     # normalised Laplacian energy
    exposure: float = 1.0      # 1 = well exposed, lower = clipped highlights/shadows

    def reject(self, reason: RejectReason) -> None:
        self.accepted = False
        if reason not in self.reasons:
            self.reasons.append(reason)


@dataclass
class FlowerObservation:
    """A fully-analysed flower in a single frame.

    This is the unit the navigation and electrostatic-control stages consume.
    """

    box: BBox
    score: float
    mask: Optional[np.ndarray] = None
    ellipse: Optional[Ellipse] = None

    sex: SexEstimate = field(default_factory=SexEstimate)
    anthesis: AnthesisEstimate = field(default_factory=AnthesisEstimate)
    pollen: PollenEstimate = field(default_factory=PollenEstimate)
    quality: QualityAssessment = field(default_factory=QualityAssessment)

    pose: Optional[Pose3D] = None          # in the camera frame
    rover_pose: Optional[Pose3D] = None    # transformed into the rover frame
    range_m: Optional[float] = None
    incidence_deg: Optional[float] = None  # angle between probe axis and corolla normal
    aim_point: Optional[tuple[float, float]] = None  # pixel target (anther or stigma)

    track_id: Optional[int] = None
    sources: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        """True when this flower is a legitimate pollination target."""
        return (
            self.quality.accepted
            and self.anthesis.stage.is_viable
            and self.sex.sex is not FlowerSex.UNKNOWN
        )


@dataclass
class FrameResult:
    """Everything the perception stack knows after processing one frame."""

    frame_index: int = 0
    timestamp: float = field(default_factory=time.time)
    width: int = 0
    height: int = 0
    flowers: list[FlowerObservation] = field(default_factory=list)
    rejected: list[FlowerObservation] = field(default_factory=list)
    depth: Optional[np.ndarray] = None
    latency_ms: dict[str, float] = field(default_factory=dict)

    @property
    def males(self) -> list[FlowerObservation]:
        return [f for f in self.flowers if f.sex.sex is FlowerSex.MALE]

    @property
    def females(self) -> list[FlowerObservation]:
        return [f for f in self.flowers if f.sex.sex is FlowerSex.FEMALE]


# --------------------------------------------------------------------------- #
# Sensing and control
# --------------------------------------------------------------------------- #


@dataclass
class EnvironmentReading:
    """Ambient sensing that parameterises the electrostatic controller.

    ``humidity`` is relative humidity in percent; it is the dominant
    environmental term because surface conduction on the pollen exine rises
    steeply with RH, bleeding off induced charge (Sec. III-E-i).
    """

    humidity: float = 55.0
    temperature_c: float = 25.0
    tof_distance_m: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    valid: bool = True


@dataclass
class ElectrostaticCommand:
    """Actuation parameters produced by the adaptive controller (Sec. III-F)."""

    mode: ProbeMode = ProbeMode.IDLE
    voltage_kv: float = 0.0
    exposure_ms: float = 0.0
    discharge_delay_ms: float = 0.0
    polarity: int = -1            # -1 attract/collect, +1 release/deposit
    standoff_m: float = 0.0
    predicted_field_kv_per_m: float = 0.0   # at the flower surface; drives transfer
    tip_field_kv_per_m: float = 0.0         # at the electrode; governs corona onset
    clamped: list[str] = field(default_factory=list)
    rationale: dict[str, float] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        """Compact dict for the serial/MQTT link to the rover MCU."""
        return {
            "mode": self.mode.value,
            "kv": round(self.voltage_kv, 3),
            "ms": round(self.exposure_ms, 1),
            "delay": round(self.discharge_delay_ms, 1),
            "pol": self.polarity,
        }


@dataclass
class VerificationResult:
    """Outcome check closing the loop in Fig. 1 (Sec. III-G)."""

    success: bool = False
    confidence: float = 0.0
    pollen_delta: float = 0.0    # negative after a successful collection
    stigma_delta: float = 0.0    # positive after a successful deposition
    note: str = ""
