"""Shared region extraction and image measurement utilities.

The perception heads all need the same handful of primitives: a corolla mask, an
ellipse fitted to it, the inner disc where the anther or stigma sits, the petal
annulus that provides a colour baseline, and a set of robust colour/texture
statistics. Centralising them here keeps the heads short and, more importantly,
guarantees they all measure the same regions the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..types import BBox, Ellipse


# --------------------------------------------------------------------------- #
# Cropping
# --------------------------------------------------------------------------- #


def crop(frame: np.ndarray, box: BBox, pad: int = 0) -> np.ndarray:
    """Extract a box from a frame, clamped to the frame bounds."""
    height, width = frame.shape[:2]
    x1 = int(max(0, np.floor(box.x1) - pad))
    y1 = int(max(0, np.floor(box.y1) - pad))
    x2 = int(min(width, np.ceil(box.x2) + pad))
    y2 = int(min(height, np.ceil(box.y2) + pad))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=frame.dtype)
    return frame[y1:y2, x1:x2]


def context_crop(frame: np.ndarray, box: BBox, scale: float = 1.6,
                 bias_down: float = 0.55) -> np.ndarray:
    """Crop with surrounding context, expanded preferentially downward.

    Sex classification depends on seeing what is *below* the corolla, because
    that is where a pistillate flower carries its inferior ovary. A tight crop
    on the flower alone removes the single most discriminative cue, so the crop
    is deliberately asymmetric.
    """
    height, width = frame.shape[:2]
    return crop(frame, box.scaled(scale, width, height, bias_down=bias_down))


# --------------------------------------------------------------------------- #
# Masks and shape
# --------------------------------------------------------------------------- #


def mask_in_box(mask: Optional[np.ndarray], box: BBox) -> Optional[np.ndarray]:
    """Slice a full-frame boolean mask down to a box."""
    if mask is None:
        return None
    x1, y1, x2, y2 = box.as_int()
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(mask.shape[1], max(x2, x1 + 1))
    y2 = min(mask.shape[0], max(y2, y1 + 1))
    return mask[y1:y2, x1:x2]


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component of a boolean mask.

    Instance masks routinely carry speckle from neighbouring petals; ellipse
    fitting and solidity are both badly distorted by it.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.sum() == 0:
        return binary.astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 2:
        return binary.astype(bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def fit_ellipse(mask: np.ndarray) -> Optional[Ellipse]:
    """Fit an ellipse to the outline of a boolean mask.

    The corolla of an open cucurbit flower is close to a planar disc, so the
    fitted ellipse carries real geometric information: its major axis is the
    true corolla diameter (unforeshortened regardless of tilt) and its axis
    ratio is the cosine of the tilt angle. Both are exploited downstream.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.sum() < 32:
        return None
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:  # cv2.fitEllipse needs at least five points
        return None
    (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
    major, minor = (max(axis_a, axis_b), min(axis_a, axis_b))
    if major < 1e-3:
        return None
    # fitEllipse reports the angle of the *first* returned axis; re-derive it
    # for the major axis so the convention is unambiguous downstream.
    if axis_a < axis_b:
        angle += 90.0
    return Ellipse(cx=float(cx), cy=float(cy), major=float(major),
                   minor=float(minor), angle_deg=float(angle % 180.0))


def ellipse_from_box(box: BBox) -> Ellipse:
    """Fallback ellipse when no mask is available: the box's inscribed ellipse."""
    major, minor = max(box.width, box.height), min(box.width, box.height)
    angle = 0.0 if box.width >= box.height else 90.0
    return Ellipse(cx=box.cx, cy=box.cy, major=float(major),
                   minor=float(minor), angle_deg=angle)


def solidity(mask: np.ndarray) -> float:
    """Area divided by convex-hull area; 1.0 for a convex blob.

    Immature cucurbit fruit are convex; leaves, stems and merged foliage are
    not. This is what separates a genuine ovary from a green background blob.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    return float(area / hull_area) if hull_area > 1e-6 else 0.0


def circularity(mask: np.ndarray) -> float:
    """Isoperimetric ratio 4*pi*A/P^2; 1.0 for a perfect circle.

    A fully anthetic corolla is round and smooth; a bud is small and a
    senescent flower is ragged and collapsed, both of which drop this value.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    if perimeter < 1e-6:
        return 0.0
    return float(np.clip(4.0 * np.pi * cv2.contourArea(contour) / (perimeter ** 2), 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Concentric regions of the corolla
# --------------------------------------------------------------------------- #


@dataclass
class CorollaRegions:
    """Inner disc and petal annulus of a corolla, as boolean masks.

    Both are expressed in the coordinate frame of the supplied crop. The inner
    disc is where the reproductive structures sit (anther on a staminate
    flower, stigma on a pistillate one); the petal ring supplies the local
    colour baseline those structures must be measured against.
    """

    inner: np.ndarray
    ring: np.ndarray
    centre: tuple[float, float]
    radius: float


def corolla_regions(shape: tuple[int, int], ellipse: Optional[Ellipse] = None,
                    inner_ratio: float = 0.45, ring_ratio: float = 0.80,
                    offset: tuple[float, float] = (0.0, 0.0),
                    mask: Optional[np.ndarray] = None) -> CorollaRegions:
    """Build concentric inner/ring masks for a corolla.

    Args:
        shape: ``(height, width)`` of the crop the masks must match.
        ellipse: Fitted corolla ellipse in *frame* coordinates, if available.
        inner_ratio: Inner disc radius as a fraction of the corolla radius.
        ring_ratio: Inner edge of the petal annulus, same units.
        offset: Frame coordinate of the crop's top-left corner.
        mask: Corolla mask in crop coordinates, used to exclude background.
    """
    height, width = shape[:2]
    if ellipse is not None:
        cx = ellipse.cx - offset[0]
        cy = ellipse.cy - offset[1]
        # Use the minor axis for the disc radius: it is the smaller apparent
        # extent, so a disc built on it stays inside the corolla under tilt.
        radius = max(ellipse.minor / 2.0, 2.0)
        angle = np.deg2rad(ellipse.angle_deg)
        ratio = max(ellipse.axis_ratio, 1e-3)
    else:
        cx, cy = width / 2.0, height / 2.0
        radius = max(min(width, height) / 2.0, 2.0)
        angle, ratio = 0.0, 1.0

    ys, xs = np.mgrid[0:height, 0:width]
    dx, dy = xs - cx, ys - cy
    # Rotate into the ellipse frame and rescale the major axis so that the
    # elliptical corolla maps to a unit circle; distances then compare directly.
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    u = (dx * cos_a + dy * sin_a) * ratio   # along the major axis
    v = -dx * sin_a + dy * cos_a            # along the minor axis
    distance = np.sqrt(u * u + v * v) / radius

    inner = distance <= inner_ratio
    ring = (distance > ring_ratio) & (distance <= 1.05)

    if mask is not None and mask.shape[:2] == (height, width):
        inner = inner & mask
        ring = ring & mask

    return CorollaRegions(inner=inner, ring=ring, centre=(float(cx), float(cy)),
                          radius=float(radius))


# --------------------------------------------------------------------------- #
# Colour and texture statistics
# --------------------------------------------------------------------------- #


def hsv_of(image: np.ndarray) -> np.ndarray:
    """BGR to HSV with OpenCV's 0-179 hue range."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def hue_mask(hsv: np.ndarray, hue_range: tuple[float, float],
             min_saturation: float = 0, min_value: float = 0) -> np.ndarray:
    """Boolean mask of pixels inside an HSV gate.

    Hue bounds are given in degrees (0-360) for readability in the config files
    and converted to OpenCV's 0-179 convention here. Wrap-around ranges (a low
    bound above the high bound) are handled, which matters for the red-orange
    end of the spectrum.
    """
    low = float(hue_range[0]) / 2.0
    high = float(hue_range[1]) / 2.0
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if low <= high:
        in_hue = (hue >= low) & (hue <= high)
    else:
        in_hue = (hue >= low) | (hue <= high)
    return in_hue & (saturation >= min_saturation) & (value >= min_value)


def robust_mean(values: np.ndarray, mask: Optional[np.ndarray] = None,
                percentile: float = 50.0) -> float:
    """Percentile of ``values`` over ``mask``; 0.0 when the mask is empty.

    A percentile rather than a mean because specular highlights on wet petals
    are common and would drag an arithmetic mean around badly.
    """
    if mask is not None:
        selected = values[mask]
    else:
        selected = values.reshape(-1)
    if selected.size == 0:
        return 0.0
    return float(np.percentile(selected, percentile))


def sharpness(image: np.ndarray, scale: float = 400.0) -> float:
    """Normalised Laplacian variance, a standard focus measure."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if gray.size == 0:
        return 0.0
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return float(np.clip(variance / scale, 0.0, 1.0))


def texture_energy(image: np.ndarray, mask: Optional[np.ndarray] = None,
                   scale: float = 400.0) -> float:
    """Local high-frequency energy inside a mask.

    Fresh pollen is granular: individual grains and their shadows produce
    high-frequency structure that a bare, spent anther simply does not have.
    Measured on the green channel, which for yellow-orange subjects carries
    good contrast without the noise of the blue channel.
    """
    if image.size == 0:
        return 0.0
    channel = image[..., 1] if image.ndim == 3 else image
    laplacian = cv2.Laplacian(channel.astype(np.float32), cv2.CV_32F)
    if mask is not None:
        if mask.shape != laplacian.shape or mask.sum() < 16:
            return 0.0
        values = laplacian[mask]
    else:
        values = laplacian.reshape(-1)
    if values.size == 0:
        return 0.0
    return float(np.clip(values.var() / scale, 0.0, 1.0))


def exposure_quality(image: np.ndarray, mask: Optional[np.ndarray] = None) -> tuple[float, float]:
    """Return ``(quality, clipped_fraction)`` for a region.

    Bright yellow petals in direct sun clip easily, and a clipped corolla
    destroys exactly the chroma and texture information the pollen head needs.
    """
    if image.size == 0:
        return 0.0, 1.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    values = gray[mask] if mask is not None and mask.shape == gray.shape else gray.reshape(-1)
    if values.size == 0:
        return 0.0, 1.0
    clipped = float(np.mean((values >= 250) | (values <= 5)))
    mean = float(values.mean()) / 255.0
    # Peaks at mid-grey and falls off toward either end of the range.
    quality = float(np.clip(1.0 - abs(mean - 0.5) * 2.0, 0.0, 1.0))
    return quality * (1.0 - clipped), clipped


def green_fraction(hsv: np.ndarray, mask: Optional[np.ndarray] = None,
                   hue_range: tuple[float, float] = (30, 95),
                   min_saturation: float = 45, min_value: float = 30) -> float:
    """Fraction of a region reading as vegetation green."""
    gate = hue_mask(hsv, hue_range, min_saturation, min_value)
    if mask is not None:
        if mask.shape != gate.shape or mask.sum() == 0:
            return 0.0
        return float(gate[mask].mean())
    return float(gate.mean()) if gate.size else 0.0
