"""Pinhole camera model, intrinsics handling and pixel/ray conversions.

The pipeline must work with two very different sensors: a calibrated RGB-D head
(the report's baseline, Sec. III-B) and a bare ESP32-CAM with an uncalibrated
OV2640. Both are handled here so that nothing downstream needs to know which is
in use — an uncalibrated camera simply gets intrinsics synthesised from a
horizontal field-of-view estimate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics with optional Brown-Conrady distortion."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: Optional[np.ndarray] = None  # (k1, k2, p1, p2, k3)
    name: str = "camera"

    def __post_init__(self) -> None:
        if self.distortion is not None:
            self.distortion = np.asarray(self.distortion, dtype=np.float64).reshape(-1)

    # -- construction ------------------------------------------------------ #

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float,
                 name: str = "camera") -> "CameraIntrinsics":
        """Synthesise intrinsics from a horizontal field of view.

        Accurate enough for target selection and for size-prior ranging within
        roughly 10-15%; run ``pollivision calibrate`` for metric-grade work.
        """
        hfov = np.deg2rad(hfov_deg)
        fx = (width / 2.0) / np.tan(hfov / 2.0)
        # Square pixels are a safe assumption for the small CMOS sensors used here.
        return cls(fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0,
                   width=width, height=height, name=name)

    @classmethod
    def from_config(cls, cfg) -> "CameraIntrinsics":
        """Build from a ``Config`` section, preferring explicit intrinsics."""
        width = int(cfg.get("width", 640))
        height = int(cfg.get("height", 480))
        name = str(cfg.get("name", "camera"))
        fx = cfg.get("fx")
        if fx is not None:
            return cls(
                fx=float(fx),
                fy=float(cfg.get("fy", fx)),
                cx=float(cfg.get("cx", width / 2.0)),
                cy=float(cfg.get("cy", height / 2.0)),
                width=width,
                height=height,
                distortion=cfg.get("distortion"),
                name=name,
            )
        return cls.from_fov(width, height, float(cfg.get("hfov_deg", 60.0)), name)

    @classmethod
    def load(cls, path: str | Path) -> "CameraIntrinsics":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(
            fx=data["fx"], fy=data["fy"], cx=data["cx"], cy=data["cy"],
            width=data["width"], height=data["height"],
            distortion=data.get("distortion"), name=data.get("name", "camera"),
        )

    def save(self, path: str | Path) -> None:
        payload = {
            "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
            "width": self.width, "height": self.height, "name": self.name,
        }
        if self.distortion is not None:
            payload["distortion"] = self.distortion.tolist()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    # -- derived quantities ------------------------------------------------ #

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    @property
    def hfov_deg(self) -> float:
        return float(np.rad2deg(2.0 * np.arctan((self.width / 2.0) / self.fx)))

    @property
    def mean_focal(self) -> float:
        """Geometric mean focal length, used for isotropic size-prior ranging."""
        return float(np.sqrt(self.fx * self.fy))

    def scaled_to(self, width: int, height: int) -> "CameraIntrinsics":
        """Rescale intrinsics when frames are resized (common with ESP32-CAM)."""
        if width == self.width and height == self.height:
            return self
        sx, sy = width / self.width, height / self.height
        return CameraIntrinsics(
            fx=self.fx * sx, fy=self.fy * sy,
            cx=self.cx * sx, cy=self.cy * sy,
            width=width, height=height,
            distortion=self.distortion, name=self.name,
        )

    # -- projection -------------------------------------------------------- #

    def pixel_to_ray(self, u: float, v: float) -> np.ndarray:
        """Unit direction in the camera frame (+x right, +y down, +z forward)."""
        ray = np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0])
        return ray / np.linalg.norm(ray)

    def backproject(self, u: float, v: float, depth_m: float) -> np.ndarray:
        """3D point for a pixel at a given *depth along the optical axis*."""
        return np.array([
            (u - self.cx) * depth_m / self.fx,
            (v - self.cy) * depth_m / self.fy,
            depth_m,
        ])

    def backproject_range(self, u: float, v: float, range_m: float) -> np.ndarray:
        """3D point for a pixel at a given *euclidean range* from the camera."""
        return self.pixel_to_ray(u, v) * range_m

    def project(self, point: np.ndarray) -> tuple[float, float]:
        x, y, z = np.asarray(point, dtype=np.float64).reshape(3)
        z = max(z, 1e-6)
        return (self.fx * x / z + self.cx, self.fy * y / z + self.cy)

    def pixels_per_metre_at(self, range_m: float) -> float:
        """Apparent scale: how many pixels a 1 m object spans at this range."""
        return self.mean_focal / max(range_m, 1e-6)


@dataclass
class Extrinsics:
    """Rigid transform from the camera frame into the rover body frame.

    The rover frame follows the usual ground-robot convention: +x forward,
    +y left, +z up, origin at the drive centre on the ground plane.
    """

    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        self.rotation = np.asarray(self.rotation, dtype=np.float64).reshape(3, 3)
        self.translation = np.asarray(self.translation, dtype=np.float64).reshape(3)

    @classmethod
    def from_config(cls, cfg) -> "Extrinsics":
        """Build from mount angles: pitch/yaw/roll in degrees plus an offset.

        Also folds in the axis change between the optical convention
        (+x right, +y down, +z forward) and the rover body convention
        (+x forward, +y left, +z up).
        """
        pitch = np.deg2rad(float(cfg.get("pitch_deg", 0.0)))
        yaw = np.deg2rad(float(cfg.get("yaw_deg", 0.0)))
        roll = np.deg2rad(float(cfg.get("roll_deg", 0.0)))
        offset = np.asarray(cfg.get("offset_m", [0.0, 0.0, 0.0]), dtype=np.float64)

        # Optical -> body axis permutation.
        axis_swap = np.array([[0.0, 0.0, 1.0],
                              [-1.0, 0.0, 0.0],
                              [0.0, -1.0, 0.0]])
        rot = _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll) @ axis_swap
        return cls(rotation=rot, translation=offset)

    def apply_point(self, point: np.ndarray) -> np.ndarray:
        return self.rotation @ np.asarray(point, dtype=np.float64).reshape(3) + self.translation

    def apply_vector(self, vector: np.ndarray) -> np.ndarray:
        return self.rotation @ np.asarray(vector, dtype=np.float64).reshape(3)


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
