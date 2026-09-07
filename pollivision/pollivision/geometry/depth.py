"""Range estimation.

The report's baseline platform carries an RGB-D camera (Sec. III-B), and where
a depth channel exists this module simply uses it. The complication is the
stated deployment target: an ESP32-CAM is a single monocular sensor with no
depth channel at all, so range has to be recovered from the image alone.

Three estimators are provided and fused:

* **RGB-D passthrough** - a robust percentile of the true depth over the corolla
  mask. Percentile rather than mean because depth sensors produce dropouts and
  flying pixels at object boundaries, and a flower is nearly all boundary.
* **Size-prior ranging** - the pinhole relation ``d = f * D_real / D_pixels``
  using a species-specific corolla diameter. The important detail is *which*
  pixel measurement to use: perspective foreshortening compresses a tilted disc
  along one axis only, leaving the major axis of the fitted ellipse proportional
  to the true diameter. Using the major axis therefore makes this estimate
  tilt-invariant, unlike box-height or box-diagonal heuristics which shrink as
  the flower tips away from the camera and silently overestimate range.
* **Monocular depth network** - MiDaS produces relative inverse depth with an
  unknown scale and offset, which is useless on its own for a rover that must
  stop 4 cm from a flower. It becomes useful when anchored: fitting the affine
  scale/shift against the size-prior estimates in the same frame turns relative
  depth into a metric map that also covers the foliage between the rover and its
  target.

Every estimate carries a variance so the fusion is a proper inverse-variance
weighting rather than an unweighted average.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..logging_utils import get_logger
from ..types import BBox, Ellipse
from .camera import CameraIntrinsics

LOGGER = get_logger(__name__)


@dataclass
class RangeEstimate:
    """A range in metres with an uncertainty and a provenance tag."""

    range_m: float
    sigma_m: float
    source: str
    valid: bool = True

    @property
    def precision(self) -> float:
        """Inverse variance, the natural weight for fusing estimates."""
        return 1.0 / max(self.sigma_m ** 2, 1e-9)


def range_from_size(
    ellipse: Optional[Ellipse],
    box: BBox,
    intrinsics: CameraIntrinsics,
    corolla_diameter_m: float,
    sigma_fraction: float = 0.30,
    use_major_axis: bool = True,
) -> RangeEstimate:
    """Metric range from apparent corolla size.

    Uncertainty is dominated by the biological spread of corolla diameter within
    a species, not by pixel measurement noise, so ``sigma_fraction`` propagates
    directly into the range uncertainty.
    """
    if corolla_diameter_m <= 0:
        return RangeEstimate(0.0, 1e3, "size", valid=False)

    if ellipse is not None and use_major_axis and ellipse.major > 2.0:
        apparent_px = ellipse.major
    else:
        # Without an ellipse, the larger box side is the best available proxy.
        apparent_px = max(box.width, box.height)

    if apparent_px < 2.0:
        return RangeEstimate(0.0, 1e3, "size", valid=False)

    range_m = intrinsics.mean_focal * corolla_diameter_m / apparent_px
    # Relative error in range equals relative error in the assumed diameter,
    # plus a small pixel-measurement term that matters for tiny detections.
    relative = np.hypot(sigma_fraction, 2.0 / apparent_px)
    return RangeEstimate(float(range_m), float(range_m * relative), "size")


def range_from_depth(
    depth: np.ndarray,
    box: BBox,
    intrinsics: CameraIntrinsics,
    mask: Optional[np.ndarray] = None,
    percentile: float = 35.0,
    valid_range: tuple[float, float] = (0.05, 3.0),
) -> RangeEstimate:
    """Range from a metric depth map over the corolla.

    A low percentile is deliberate: the mask inevitably includes some background
    pixels around the petal margins, and those are *further* than the flower, so
    biasing toward the near side of the distribution tracks the corolla surface
    rather than the canopy behind it.
    """
    if depth is None:
        return RangeEstimate(0.0, 1e3, "depth", valid=False)

    x1, y1, x2, y2 = box.as_int()
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(depth.shape[1], max(x2, x1 + 1)), min(depth.shape[0], max(y2, y1 + 1))
    patch = depth[y1:y2, x1:x2]
    if patch.size == 0:
        return RangeEstimate(0.0, 1e3, "depth", valid=False)

    if mask is not None and mask.shape[:2] == depth.shape[:2]:
        local = mask[y1:y2, x1:x2]
        if local.shape == patch.shape and local.sum() > 16:
            patch = patch[local]

    values = patch[np.isfinite(patch)]
    values = values[(values >= valid_range[0]) & (values <= valid_range[1])]
    if values.size < 8:
        return RangeEstimate(0.0, 1e3, "depth", valid=False)

    depth_along_axis = float(np.percentile(values, percentile))
    # Convert axial depth (Z) to euclidean range along the pixel's ray.
    ray = intrinsics.pixel_to_ray(box.cx, box.cy)
    range_m = depth_along_axis / max(ray[2], 1e-6)

    spread = float(np.percentile(values, 75) - np.percentile(values, 25))
    sigma = max(0.005, 0.5 * spread)
    return RangeEstimate(range_m, sigma, "depth")


def fuse_ranges(estimates: list[RangeEstimate],
                valid_range: tuple[float, float] = (0.05, 3.0)) -> Optional[RangeEstimate]:
    """Inverse-variance fusion of independent range estimates."""
    usable = [
        e for e in estimates
        if e.valid and np.isfinite(e.range_m) and valid_range[0] <= e.range_m <= valid_range[1]
    ]
    if not usable:
        return None
    if len(usable) == 1:
        return usable[0]

    precisions = np.array([e.precision for e in usable])
    values = np.array([e.range_m for e in usable])
    total = precisions.sum()
    fused = float((values * precisions).sum() / total)
    sigma = float(np.sqrt(1.0 / total))
    return RangeEstimate(fused, sigma, "+".join(e.source for e in usable))


class MonocularDepth:
    """MiDaS relative-depth network with affine anchoring to metric scale.

    Kept optional and off by default: on a 4-core SBC it costs a large fraction
    of the frame budget, and size-prior ranging already answers the question the
    controller actually asks ("how far is *this flower*"). Its real value is the
    depth of everything *else* in the scene, which is what approach planning
    needs to avoid pushing through foliage.
    """

    def __init__(self, cfg, device: str = "cpu") -> None:
        import torch

        from ..zoo import resolve

        self.torch = torch
        self.device = device
        section = cfg.section("geometry.depth.midas")
        self.input_size = int(section.get("input_size", 256))

        weights = resolve(section.get("weights", "midas-small"))
        LOGGER.info("Loading monocular depth weights from %s", weights)
        # MiDaS ships its architectures in its own package; importing lazily
        # keeps it an optional dependency.
        from midas.model_loader import load_model  # type: ignore

        self.model, self.transform, _, _ = load_model(
            device, str(weights), "midas_v21_small_256", optimize=False,
            height=self.input_size, square=False,
        )
        self.model.eval()

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """Return a relative inverse-depth map at the frame's resolution."""
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        sample = self.transform({"image": rgb})["image"]
        with self.torch.no_grad():
            tensor = self.torch.from_numpy(sample).to(self.device).unsqueeze(0)
            prediction = self.model.forward(tensor)
            prediction = self.torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=frame.shape[:2],
                mode="bicubic", align_corners=False,
            ).squeeze()
        return prediction.cpu().numpy()

    @staticmethod
    def anchor_to_metric(
        inverse_depth: np.ndarray,
        anchors: list[tuple[float, float, float]],
        valid_range: tuple[float, float] = (0.05, 3.0),
    ) -> Optional[np.ndarray]:
        """Convert relative inverse depth to a metric depth map.

        MiDaS output is affine in *inverse* depth: ``1/d = a * v + b``. Fitting
        ``a`` and ``b`` needs at least two anchors of known range, which the
        size-prior estimator supplies for every detected flower in the frame.

        Args:
            inverse_depth: Raw network output.
            anchors: ``(u, v, range_m)`` triples of known metric range.
            valid_range: Output clamp.

        Returns:
            A metric depth map, or None if the anchors were insufficient or
            degenerate.
        """
        if inverse_depth is None or len(anchors) < 2:
            return None

        height, width = inverse_depth.shape[:2]
        samples, targets = [], []
        for u, v, range_m in anchors:
            ui, vi = int(round(u)), int(round(v))
            if not (0 <= ui < width and 0 <= vi < height) or range_m <= 1e-6:
                continue
            # Median over a small window: single-pixel reads are noisy.
            window = inverse_depth[max(0, vi - 2):vi + 3, max(0, ui - 2):ui + 3]
            if window.size == 0:
                continue
            samples.append(float(np.median(window)))
            targets.append(1.0 / range_m)

        if len(samples) < 2:
            return None

        samples_arr = np.asarray(samples, dtype=np.float64)
        if float(samples_arr.std()) < 1e-6:
            # All anchors landed at the same relative depth; the fit is
            # unconstrained and would produce an arbitrary scale.
            return None

        design = np.stack([samples_arr, np.ones_like(samples_arr)], axis=1)
        coefficients, *_ = np.linalg.lstsq(design, np.asarray(targets), rcond=None)
        scale, shift = float(coefficients[0]), float(coefficients[1])

        inverse_metric = scale * inverse_depth + shift
        with np.errstate(divide="ignore", invalid="ignore"):
            metric = np.where(inverse_metric > 1e-6, 1.0 / inverse_metric, np.nan)
        return np.clip(metric, valid_range[0], valid_range[1])
