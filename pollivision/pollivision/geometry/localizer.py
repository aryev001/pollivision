"""Flower localisation: ties ranging, orientation and frame transforms together.

This is the module that turns a 2D detection into the 3D goal the navigation
module in the report's Sec. III-D consumes: "detected flower coordinates are
transformed from image space to the rover's local reference frame".
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..logging_utils import get_logger
from ..types import FlowerObservation
from . import depth as depth_mod
from . import orientation as orientation_mod
from . import transforms
from .camera import CameraIntrinsics, Extrinsics

LOGGER = get_logger(__name__)


class Localizer:
    """Assigns metric position, orientation and range to flower observations."""

    def __init__(self, cfg, intrinsics: CameraIntrinsics, extrinsics: Extrinsics) -> None:
        self.cfg = cfg
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

        self.mode = str(cfg.get("geometry.depth.mode", "auto"))
        self.valid_range = tuple(cfg.get("geometry.depth.valid_range_m", [0.05, 3.0]))
        self.depth_percentile = float(cfg.get("geometry.depth.depth_patch_percentile", 35))

        self.use_major_axis = bool(cfg.get("geometry.ranging.use_major_axis", True))
        self.size_sigma = float(cfg.get("geometry.ranging.size_sigma_frac", 0.30))

        orientation_cfg = cfg.section("geometry.orientation")
        self.min_axis_ratio = float(orientation_cfg.get("min_axis_ratio", 0.15))
        self.use_depth_normal = bool(orientation_cfg.get("use_depth_normal", True))
        self.depth_min_points = int(orientation_cfg.get("depth_normal_min_points", 60))
        self.max_tilt_deg = float(orientation_cfg.get("max_tilt_deg", 75.0))

        self.corolla_diameter = float(cfg.get("species.corolla_diameter_m", 0.10))
        self.standoff = float(cfg.get("planning.approach_standoff_m", 0.04))

        self._midas = None
        if self.mode in {"midas", "auto"} and cfg.get("geometry.depth.midas.enabled", False):
            try:
                self._midas = depth_mod.MonocularDepth(cfg)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Monocular depth unavailable (%s); "
                               "using size-prior ranging only", exc)

    # ------------------------------------------------------------------ #

    def resolve_depth(
        self,
        frame: np.ndarray,
        flowers: list[FlowerObservation],
        depth: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Return the metric depth map to use for this frame.

        Called *before* the perception heads rather than as part of
        localisation, because the sex head's morphology cue needs depth to
        separate an ovary from the canopy behind it. A supplied RGB-D frame is
        passed through unchanged; otherwise a monocular map is recovered and
        anchored to metric scale using the size-prior ranges.
        """
        if depth is not None or self._midas is None or not flowers:
            return depth
        return self._recover_depth(frame, flowers, self._size_estimates(frame, flowers))

    def _size_estimates(self, frame: np.ndarray,
                        flowers: list[FlowerObservation]) -> list:
        intrinsics = self.intrinsics.scaled_to(frame.shape[1], frame.shape[0])
        return [
            depth_mod.range_from_size(
                flower.ellipse, flower.box, intrinsics, self.corolla_diameter,
                self.size_sigma, self.use_major_axis,
            )
            for flower in flowers
        ]

    def localize(
        self,
        frame: np.ndarray,
        flowers: list[FlowerObservation],
        depth: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Populate pose, range and incidence on each flower, in place.

        Returns the depth map used, so callers can keep it for obstacle checks.
        """
        if not flowers:
            return depth

        intrinsics = self.intrinsics.scaled_to(frame.shape[1], frame.shape[0])
        size_estimates = self._size_estimates(frame, flowers)

        if depth is None:
            depth = self.resolve_depth(frame, flowers, None)

        # Fuse the available range sources, then solve pose.
        for flower, size_estimate in zip(flowers, size_estimates):
            candidates = []
            if self.mode != "size" and depth is not None:
                candidates.append(depth_mod.range_from_depth(
                    depth, flower.box, intrinsics, flower.mask,
                    self.depth_percentile, self.valid_range,
                ))
            if self.mode != "rgbd" or depth is None:
                candidates.append(size_estimate)

            fused = depth_mod.fuse_ranges(candidates, self.valid_range)
            if fused is None:
                flower.range_m = None
                flower.meta["range_source"] = "none"
                continue

            flower.range_m = float(fused.range_m)
            flower.meta["range_source"] = fused.source
            flower.meta["range_sigma_m"] = float(fused.sigma_m)

            pose, normal_confidence = orientation_mod.estimate_pose(
                ellipse=flower.ellipse,
                box=flower.box,
                range_m=fused.range_m,
                intrinsics=intrinsics,
                depth=depth,
                mask=flower.mask,
                aim_point=flower.aim_point,
                min_axis_ratio=self.min_axis_ratio,
                max_tilt_deg=self.max_tilt_deg,
                use_depth_normal=self.use_depth_normal,
                depth_min_points=self.depth_min_points,
            )
            flower.pose = pose
            flower.meta["normal_confidence"] = float(normal_confidence)
            flower.rover_pose = transforms.to_rover_frame(pose, self.extrinsics)

            _, approach = transforms.approach_vector(pose, self.standoff)
            flower.incidence_deg = orientation_mod.incidence_angle(
                pose.normal if pose.normal is not None else np.array([0.0, 0.0, -1.0]),
                approach,
            )
            if flower.ellipse is not None:
                flower.meta["tilt_deg"] = float(np.degrees(flower.ellipse.tilt_rad))

        return depth

    def _recover_depth(self, frame, flowers, size_estimates) -> Optional[np.ndarray]:
        """Run monocular depth and anchor it with the size-prior ranges."""
        try:
            relative = self._midas.infer(frame)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Monocular depth inference failed: %s", exc)
            return None

        anchors = [
            (flower.box.cx, flower.box.cy, estimate.range_m)
            for flower, estimate in zip(flowers, size_estimates)
            if estimate.valid and self.valid_range[0] <= estimate.range_m <= self.valid_range[1]
        ]
        metric = depth_mod.MonocularDepth.anchor_to_metric(
            relative, anchors, self.valid_range
        )
        if metric is None:
            LOGGER.debug("Not enough anchors to scale monocular depth this frame")
        return metric
