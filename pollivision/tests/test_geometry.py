"""Geometry tests: ranging, orientation and frame transforms.

These check closed-form mathematics against analytically-known answers, so they
are exact rather than approximate and will catch a sign error or a convention
flip immediately.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pollivision.config import Config
from pollivision.geometry import depth as depth_mod
from pollivision.geometry import orientation, transforms
from pollivision.geometry.camera import CameraIntrinsics, Extrinsics
from pollivision.types import BBox, Ellipse, Pose3D


def _ellipse_bbox(a: float, b: float, roll_deg: float,
                  cx: float = 320, cy: float = 240) -> BBox:
    """Exact axis-aligned bounding box of a rotated ellipse."""
    theta = math.radians(roll_deg)
    half_w = math.hypot(a * math.cos(theta), b * math.sin(theta))
    half_h = math.hypot(a * math.sin(theta), b * math.cos(theta))
    return BBox(cx - half_w, cy - half_h, cx + half_w, cy + half_h)


class TestIntrinsics:
    def test_fov_round_trip(self):
        intrinsics = CameraIntrinsics.from_fov(640, 480, 62.0)
        assert intrinsics.hfov_deg == pytest.approx(62.0, abs=1e-6)

    def test_backproject_project_round_trip(self):
        intrinsics = CameraIntrinsics.from_fov(800, 600, 65.0)
        point = intrinsics.backproject(500.0, 300.0, 0.42)
        u, v = intrinsics.project(point)
        assert (u, v) == pytest.approx((500.0, 300.0), abs=1e-6)

    def test_rescaling_preserves_fov(self):
        intrinsics = CameraIntrinsics.from_fov(1600, 1200, 65.0)
        scaled = intrinsics.scaled_to(800, 600)
        assert scaled.hfov_deg == pytest.approx(intrinsics.hfov_deg, abs=1e-6)


class TestSizePriorRanging:
    """The major-axis estimator must be invariant to corolla tilt.

    Perspective foreshortening compresses a tilted disc along one axis only, so
    the ellipse's major axis stays proportional to the true corolla diameter.
    A bounding-box-side estimator does not have this property and drifts badly
    once the flower is both tilted and rotated in-plane.
    """

    @pytest.mark.parametrize("tilt_deg", [0, 30, 50, 65])
    @pytest.mark.parametrize("roll_deg", [0, 30, 45, 70])
    def test_major_axis_is_tilt_invariant(self, tilt_deg, roll_deg):
        intrinsics = CameraIntrinsics.from_fov(640, 480, 62.0)
        true_range, diameter = 0.5, 0.10
        major_px = intrinsics.mean_focal * diameter / true_range

        a = major_px / 2.0
        b = a * math.cos(math.radians(tilt_deg))
        ellipse = Ellipse(320, 240, 2 * a, 2 * b, roll_deg)
        box = _ellipse_bbox(a, b, roll_deg)

        estimate = depth_mod.range_from_size(ellipse, box, intrinsics, diameter)
        assert estimate.range_m == pytest.approx(true_range, rel=1e-6)

    def test_bbox_estimator_degrades_where_major_axis_does_not(self):
        """Documents *why* the major axis is used, not just that it works."""
        intrinsics = CameraIntrinsics.from_fov(640, 480, 62.0)
        true_range, diameter = 0.5, 0.10
        major_px = intrinsics.mean_focal * diameter / true_range
        a = major_px / 2.0
        b = a * math.cos(math.radians(65))
        box = _ellipse_bbox(a, b, 45)

        naive = depth_mod.range_from_size(None, box, intrinsics, diameter,
                                          use_major_axis=False)
        # A tilted, rotated flower reads as much further away than it is.
        assert naive.range_m > true_range * 1.25

    def test_uncertainty_tracks_the_size_prior(self):
        intrinsics = CameraIntrinsics.from_fov(640, 480, 62.0)
        ellipse = Ellipse(320, 240, 100.0, 100.0, 0.0)
        box = BBox(270, 190, 370, 290)
        loose = depth_mod.range_from_size(ellipse, box, intrinsics, 0.10,
                                          sigma_fraction=0.40)
        tight = depth_mod.range_from_size(ellipse, box, intrinsics, 0.10,
                                          sigma_fraction=0.10)
        assert loose.sigma_m > tight.sigma_m


class TestRangeFusion:
    def test_fusion_favours_the_precise_estimate(self):
        loose = depth_mod.RangeEstimate(0.50, 0.15, "size")
        tight = depth_mod.RangeEstimate(0.44, 0.03, "depth")
        fused = depth_mod.fuse_ranges([loose, tight])
        assert fused is not None
        assert abs(fused.range_m - 0.44) < abs(fused.range_m - 0.50)
        # Combining independent estimates must never lose precision.
        assert fused.sigma_m < tight.sigma_m

    def test_out_of_range_estimates_are_discarded(self):
        assert depth_mod.fuse_ranges([depth_mod.RangeEstimate(9.0, 0.1, "size")]) is None

    def test_no_estimates_returns_none(self):
        assert depth_mod.fuse_ranges([]) is None


class TestOrientation:
    @pytest.mark.parametrize("tilt_deg", [0, 20, 45, 60])
    def test_tilt_recovered_from_axis_ratio(self, tilt_deg):
        ellipse = Ellipse(320, 240, 100.0,
                          100.0 * math.cos(math.radians(tilt_deg)), 0.0)
        normal, confidence = orientation.normal_from_ellipse(ellipse)
        recovered = math.degrees(math.acos(min(1.0, abs(normal[2]))))
        assert recovered == pytest.approx(tilt_deg, abs=0.5)
        assert 0.0 <= confidence <= 1.0

    def test_normal_points_back_toward_the_camera(self):
        ellipse = Ellipse(320, 240, 100.0, 70.0, 33.0)
        normal, _ = orientation.normal_from_ellipse(ellipse)
        assert normal[2] < 0
        assert np.linalg.norm(normal) == pytest.approx(1.0, abs=1e-9)

    def test_probe_on_the_normal_sees_zero_incidence(self):
        ellipse = Ellipse(320, 240, 100.0, 70.0, 30.0)
        normal, _ = orientation.normal_from_ellipse(ellipse)
        intrinsics = CameraIntrinsics.from_fov(640, 480, 62.0)
        pose = Pose3D(position=intrinsics.backproject_range(320, 240, 0.5),
                      normal=normal)
        _, approach = transforms.approach_vector(pose, 0.04)
        assert orientation.incidence_angle(normal, approach) == pytest.approx(0.0, abs=1e-6)

    def test_incidence_along_optical_axis_equals_tilt(self):
        ellipse = Ellipse(320, 240, 100.0, 70.0, 30.0)
        normal, _ = orientation.normal_from_ellipse(ellipse)
        incidence = orientation.incidence_angle(normal, np.array([0.0, 0.0, 1.0]))
        assert incidence == pytest.approx(math.degrees(ellipse.tilt_rad), abs=0.5)

    def test_edge_on_flower_reports_low_confidence(self):
        nearly_edge_on = Ellipse(320, 240, 100.0, 8.0, 0.0)
        _, confidence = orientation.normal_from_ellipse(nearly_edge_on)
        assert confidence < 0.2


class TestExtrinsics:
    def _extrinsics(self) -> Extrinsics:
        return Extrinsics.from_config(Config({
            "pitch_deg": -20.0, "yaw_deg": 0.0, "roll_deg": 0.0,
            "offset_m": [0.35, 0.0, 0.30],
        }))

    def test_rotation_is_a_proper_rotation(self):
        extrinsics = self._extrinsics()
        assert np.allclose(extrinsics.rotation @ extrinsics.rotation.T, np.eye(3))
        assert np.linalg.det(extrinsics.rotation) == pytest.approx(1.0)

    def test_forward_point_maps_ahead_of_and_above_the_rover(self):
        extrinsics = self._extrinsics()
        pose = Pose3D(position=np.array([0.0, 0.0, 0.5]))
        rover = transforms.to_rover_frame(pose, extrinsics)
        assert rover.position[0] > 0.35          # ahead of the drive centre
        assert rover.position[2] > 0.0           # above the ground plane
        assert rover.position[1] == pytest.approx(0.0, abs=1e-9)

    def test_normals_are_rotated_but_not_translated(self):
        extrinsics = self._extrinsics()
        pose = Pose3D(position=np.array([0.0, 0.0, 0.5]),
                      normal=np.array([0.0, 0.0, -1.0]))
        rover = transforms.to_rover_frame(pose, extrinsics)
        assert np.linalg.norm(rover.normal) == pytest.approx(1.0, abs=1e-9)

    def test_probe_is_placed_off_the_flower_along_its_normal(self):
        pose = Pose3D(position=np.array([0.0, 0.0, 0.5]),
                      normal=np.array([0.0, 0.0, -1.0]))
        probe, approach = transforms.approach_vector(pose, 0.04)
        # The normal points back at the camera, so the probe sits nearer than
        # the flower and travels away from the camera to reach it.
        assert probe[2] == pytest.approx(0.46, abs=1e-9)
        assert approach[2] == pytest.approx(1.0, abs=1e-9)
