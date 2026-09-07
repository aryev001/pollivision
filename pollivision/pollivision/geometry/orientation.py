"""Corolla orientation estimation.

Flower orientation is one of the three real-time inputs to the adaptive
electrostatic controller (report Sec. III-E-ii), because only the component of
the field normal to the corolla face does useful work on pollen sitting in it.
A flower tilted 60 degrees away from the probe sees roughly half the effective
field of one facing it head-on, and the controller must raise its drive to
compensate.

The estimate comes from projective geometry rather than a learned model. An open
cucurbit corolla is close to a planar disc; a circle viewed obliquely projects to
an ellipse whose axis ratio is the cosine of the tilt angle and whose minor axis
lies along the direction of tilt. That gives the full surface normal up to one
sign ambiguity (a disc tilted toward the camera and one tilted away project
identically), which is resolved from the depth gradient when depth is available
and otherwise by assuming the flower faces the observer - correct for the great
majority of flowers a rover approaches, since it approaches the ones it can see.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..types import BBox, Ellipse, Pose3D
from .camera import CameraIntrinsics


def normal_from_ellipse(
    ellipse: Ellipse,
    min_axis_ratio: float = 0.15,
    max_tilt_deg: float = 75.0,
) -> tuple[np.ndarray, float]:
    """Surface normal of a disc from its projected ellipse.

    Returns the unit normal in the camera frame (+x right, +y down, +z forward)
    and a 0-1 reliability. The normal is returned pointing back toward the
    camera, i.e. with a negative z component.
    """
    ratio = float(np.clip(ellipse.axis_ratio, 0.0, 1.0))
    # A near-degenerate ellipse means an almost edge-on flower, where the fit is
    # numerically unstable and the flower is not a viable target anyway.
    reliability = float(np.clip((ratio - min_axis_ratio) / (1.0 - min_axis_ratio), 0.0, 1.0))

    tilt = math.acos(ratio)
    max_tilt = math.radians(max_tilt_deg)
    if tilt > max_tilt:
        tilt = max_tilt
        reliability *= 0.5

    # The corolla tips about its major axis, so the in-plane component of the
    # normal lies along the *minor* axis direction, 90 degrees from the major.
    major_angle = math.radians(ellipse.angle_deg)
    tilt_direction = major_angle + math.pi / 2.0

    normal = np.array([
        math.sin(tilt) * math.cos(tilt_direction),
        math.sin(tilt) * math.sin(tilt_direction),
        -math.cos(tilt),
    ], dtype=np.float64)
    return normal / np.linalg.norm(normal), reliability


def refine_normal_with_depth(
    normal: np.ndarray,
    depth: Optional[np.ndarray],
    box: BBox,
    intrinsics: CameraIntrinsics,
    mask: Optional[np.ndarray] = None,
    min_points: int = 60,
) -> tuple[np.ndarray, float]:
    """Refine a normal by fitting a plane to the corolla's depth points.

    Where depth is available this both resolves the ellipse method's sign
    ambiguity and improves accuracy on flowers whose outline is partly occluded.
    Returns the refined normal and a 0-1 confidence; falls back to the input
    normal unchanged when there is not enough usable depth.
    """
    if depth is None:
        return normal, 0.0

    x1, y1, x2, y2 = box.as_int()
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(depth.shape[1], max(x2, x1 + 1))
    y2 = min(depth.shape[0], max(y2, y1 + 1))

    patch = depth[y1:y2, x1:x2]
    if patch.size < min_points:
        return normal, 0.0

    valid = np.isfinite(patch) & (patch > 1e-3)
    if mask is not None and mask.shape[:2] == depth.shape[:2]:
        local = mask[y1:y2, x1:x2]
        if local.shape == patch.shape:
            valid &= local
    if int(valid.sum()) < min_points:
        return normal, 0.0

    ys, xs = np.nonzero(valid)
    zs = patch[valid].astype(np.float64)
    us = xs + x1
    vs = ys + y1

    points = np.stack([
        (us - intrinsics.cx) * zs / intrinsics.fx,
        (vs - intrinsics.cy) * zs / intrinsics.fy,
        zs,
    ], axis=1)

    centroid = points.mean(axis=0)
    centred = points - centroid
    # The plane normal is the least-significant principal direction of the
    # point cloud; its singular value relative to the others says how planar
    # the cloud actually is.
    try:
        _, singular_values, vt = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        return normal, 0.0

    fitted = vt[2]
    fitted = fitted / max(np.linalg.norm(fitted), 1e-9)
    if fitted[2] > 0:  # point it back toward the camera
        fitted = -fitted

    if singular_values[1] < 1e-9:
        return normal, 0.0
    planarity = float(np.clip(1.0 - singular_values[2] / singular_values[1], 0.0, 1.0))
    if planarity < 0.3:
        # Too scattered to be a plane; the depth here is probably foliage.
        return normal, 0.0

    return fitted, planarity


def incidence_angle(normal: np.ndarray, approach: np.ndarray) -> float:
    """Angle in degrees between the corolla normal and the probe approach axis.

    Zero means the probe is aimed straight into the face of the flower, which is
    the geometry that maximises the useful field component.
    """
    normal = np.asarray(normal, dtype=np.float64).reshape(3)
    approach = np.asarray(approach, dtype=np.float64).reshape(3)
    normal_norm = np.linalg.norm(normal)
    approach_norm = np.linalg.norm(approach)
    if normal_norm < 1e-9 or approach_norm < 1e-9:
        return 0.0
    # The probe travels along +approach while the normal points back toward the
    # camera, so the two are anti-parallel in the ideal head-on case.
    cosine = float(np.dot(-normal / normal_norm, approach / approach_norm))
    return float(np.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0)))))


def estimate_pose(
    ellipse: Optional[Ellipse],
    box: BBox,
    range_m: float,
    intrinsics: CameraIntrinsics,
    depth: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    aim_point: Optional[tuple[float, float]] = None,
    min_axis_ratio: float = 0.15,
    max_tilt_deg: float = 75.0,
    use_depth_normal: bool = True,
    depth_min_points: int = 60,
) -> tuple[Pose3D, float]:
    """Full 3D pose of a flower in the camera frame.

    Returns the pose (position plus corolla face normal) and a 0-1 confidence in
    the normal. The position is taken at ``aim_point`` when one is supplied, so
    the rover approaches the anther or stigma rather than the corolla centroid.
    """
    u, v = aim_point if aim_point is not None else (
        (ellipse.cx, ellipse.cy) if ellipse is not None else box.center
    )
    position = intrinsics.backproject_range(u, v, range_m)

    if ellipse is None:
        # No shape information: assume the flower faces the camera and say so
        # with a zero confidence, which the controller treats conservatively.
        return Pose3D(position=position, normal=np.array([0.0, 0.0, -1.0]),
                      frame="camera"), 0.0

    normal, confidence = normal_from_ellipse(ellipse, min_axis_ratio, max_tilt_deg)

    if use_depth_normal and depth is not None:
        refined, planarity = refine_normal_with_depth(
            normal, depth, box, intrinsics, mask, depth_min_points
        )
        if planarity > 0.0:
            # Blend rather than replace: the ellipse fit is reliable on a clean
            # outline, the plane fit on a dense depth patch, and averaging them
            # is more robust than trusting either alone.
            blend = float(np.clip(planarity, 0.0, 1.0))
            combined = (1.0 - blend) * normal + blend * refined
            norm = np.linalg.norm(combined)
            if norm > 1e-9:
                normal = combined / norm
                confidence = float(np.clip(max(confidence, blend), 0.0, 1.0))

    return Pose3D(position=position, normal=normal, frame="camera"), confidence
