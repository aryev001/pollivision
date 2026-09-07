"""Frame conversions between the camera and the rover body."""

from __future__ import annotations

import numpy as np

from ..types import Pose3D
from .camera import Extrinsics


def to_rover_frame(pose: Pose3D, extrinsics: Extrinsics) -> Pose3D:
    """Express a camera-frame pose in the rover body frame.

    The normal is rotated but not translated - it is a direction, not a point.
    Getting this wrong is a classic source of silently plausible bugs, since a
    translated normal still looks like a unit vector.
    """
    position = extrinsics.apply_point(pose.position)
    normal = extrinsics.apply_vector(pose.normal) if pose.normal is not None else None
    return Pose3D(position=position, normal=normal, frame="rover")


def approach_vector(pose: Pose3D, standoff_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Probe placement for a given flower pose.

    Returns ``(probe_position, approach_direction)``. The probe is placed along
    the corolla's outward normal at the requested standoff so that it looks
    straight into the face of the flower, which is the geometry that maximises
    the useful component of the electrostatic field.
    """
    if pose.normal is None:
        # Without an orientation estimate, back off along the line of sight.
        direction = pose.position / max(np.linalg.norm(pose.position), 1e-9)
        return pose.position - direction * standoff_m, direction

    # The normal points back toward the observer, so moving along it steps away
    # from the flower - exactly where the probe should sit.
    probe_position = pose.position + pose.normal * standoff_m
    approach = -pose.normal
    return probe_position, approach / max(np.linalg.norm(approach), 1e-9)


def horizontal_bearing_deg(position: np.ndarray) -> float:
    """Bearing to a rover-frame point, in degrees; positive is to the left."""
    x, y = float(position[0]), float(position[1])
    return float(np.degrees(np.arctan2(y, max(abs(x), 1e-9) * np.sign(x) if x else 1e-9)))


def ground_distance(position: np.ndarray) -> float:
    """Horizontal distance from the rover origin, ignoring height."""
    return float(np.hypot(float(position[0]), float(position[1])))
