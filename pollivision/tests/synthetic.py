"""Synthetic cucurbit scene generator for tests.

These renders are geometric caricatures, not photographs. They exist to exercise
logic that must hold regardless of photorealism - that the morphology cue fires
on an ovary and not on a bare pedicel, that ellipse fitting recovers a known
tilt, that the tracker keeps identity across motion - with a ground truth that
is known exactly because it was drawn.

They are explicitly *not* a substitute for real imagery when judging accuracy.
Nothing here says anything about how the vision-language cues behave on a real
flower; see ``docs/EVALUATION.md`` for how to measure that on your own captures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class SyntheticFlower:
    """Ground truth for one rendered flower."""

    cx: int
    cy: int
    radius: int
    sex: str                  # "male" or "female"
    stage: str = "receptive"
    pollen: float = 0.8       # anther load to render, 0-1
    tilt_deg: float = 0.0     # foreshortening applied to the corolla
    roll_deg: float = 0.0     # in-plane rotation of the corolla
    occluder: bool = False    # draw a leaf across the flower

    @property
    def box(self) -> tuple[int, int, int, int]:
        minor = int(self.radius * np.cos(np.deg2rad(self.tilt_deg)))
        return (self.cx - self.radius, self.cy - minor,
                self.cx + self.radius, self.cy + minor)


@dataclass
class SyntheticScene:
    """A rendered frame plus the ground truth used to draw it."""

    image: np.ndarray
    flowers: list[SyntheticFlower] = field(default_factory=list)
    depth: Optional[np.ndarray] = None


def render_scene(
    flowers: list[SyntheticFlower],
    width: int = 640,
    height: int = 480,
    seed: int = 0,
    blur: float = 0.0,
    brightness: float = 1.0,
    with_depth: bool = False,
    flower_range_m: float = 0.45,
    canopy_range_m: float = 0.75,
    depth_noise_m: float = 0.004,
) -> SyntheticScene:
    """Render foliage plus the requested flowers, optionally with a depth map.

    The depth map places each flower and the structure attached to it at
    ``flower_range_m`` and everything else at ``canopy_range_m``, which is the
    separation a real RGB-D head sees between a blossom on the near side of the
    canopy and the leaves behind it.
    """
    rng = np.random.default_rng(seed)
    image = _foliage(width, height, rng)
    depth = (np.full((height, width), canopy_range_m, dtype=np.float32)
             if with_depth else None)

    for flower in flowers:
        _draw_flower(image, flower, rng)
        if depth is not None:
            _draw_flower_depth(depth, flower, flower_range_m)

    if depth is not None and depth_noise_m > 0:
        depth += rng.normal(0, depth_noise_m, depth.shape).astype(np.float32)

    if blur > 0:
        kernel = max(3, int(blur) | 1)
        image = cv2.GaussianBlur(image, (kernel, kernel), 0)
    if brightness != 1.0:
        image = np.clip(image.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

    return SyntheticScene(image=image, flowers=list(flowers), depth=depth)


def _foliage(width: int, height: int, rng) -> np.ndarray:
    image = np.full((height, width, 3), (42, 78, 38), dtype=np.uint8)
    noise = rng.normal(0, 10, (height, width, 3))
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    for _ in range(30):
        centre = (int(rng.uniform(0, width)), int(rng.uniform(0, height)))
        axes = (int(rng.uniform(35, 100)), int(rng.uniform(22, 65)))
        colour = (int(rng.uniform(28, 58)), int(rng.uniform(75, 135)), int(rng.uniform(28, 58)))
        cv2.ellipse(image, centre, axes, float(rng.uniform(0, 180)), 0, 360, colour, -1)
    return image


def _draw_flower(image: np.ndarray, flower: SyntheticFlower, rng) -> None:
    """Draw one flower with the morphology its sex implies."""
    radius = flower.radius
    minor = max(3, int(radius * np.cos(np.deg2rad(flower.tilt_deg))))

    if flower.stage == "bud":
        # A bud is a small, tall, closed cone rather than an open disc.
        cv2.ellipse(image, (flower.cx, flower.cy),
                    (int(radius * 0.45), int(radius * 0.8)),
                    flower.roll_deg, 0, 360, (35, 175, 215), -1)
        _draw_pedicel(image, flower, length=int(radius * 1.4))
        return

    petal = (38, 195, 240) if flower.stage != "senescent" else (60, 130, 165)

    # The structure below the corolla is what distinguishes the sexes: a female
    # flower carries an inferior ovary, a male sits on a bare pedicel. Drawn
    # before the corolla so the corolla overlaps it naturally.
    if flower.sex == "female":
        ovary_h = int(radius * 0.85)
        ovary_w = int(radius * 0.55)
        cv2.ellipse(image, (flower.cx, flower.cy + minor + int(ovary_h * 0.55)),
                    (ovary_w, ovary_h), 0, 0, 360, (55, 145, 65), -1)
        cv2.ellipse(image, (flower.cx, flower.cy + minor + int(ovary_h * 0.55)),
                    (ovary_w, ovary_h), 0, 0, 360, (40, 110, 50), 2)
    else:
        _draw_pedicel(image, flower, length=int(radius * 2.0))

    cv2.ellipse(image, (flower.cx, flower.cy), (radius, minor),
                flower.roll_deg, 0, 360, petal, -1)
    # Petal lobes, so the outline is not a perfect circle.
    for angle in range(0, 360, 72):
        theta = np.deg2rad(angle + flower.roll_deg)
        px = int(flower.cx + radius * 0.92 * np.cos(theta))
        py = int(flower.cy + minor * 0.92 * np.sin(theta))
        cv2.circle(image, (px, py), int(radius * 0.22), petal, -1)

    if flower.stage == "senescent":
        return

    # Reproductive structure in the centre.
    inner_radius = max(2, int(radius * 0.28))
    if flower.sex == "male":
        load = float(np.clip(flower.pollen, 0.0, 1.0))
        # A loaded anther is a saturated orange; a spent one is pale and dull.
        colour = (int(20 + 30 * (1 - load)), int(120 + 40 * load), int(200 + 45 * load))
        cv2.circle(image, (flower.cx, flower.cy), inner_radius, colour, -1)
        # Granularity: individual pollen grains, in proportion to the load.
        for _ in range(int(180 * load)):
            angle = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(0, inner_radius)
            gx = int(flower.cx + r * np.cos(angle))
            gy = int(flower.cy + r * np.sin(angle) * (minor / max(radius, 1)))
            cv2.circle(image, (gx, gy), 1, (10, 205, 255), -1)
    else:
        # A three-lobed stigma.
        for angle in (90, 210, 330):
            theta = np.deg2rad(angle)
            sx = int(flower.cx + inner_radius * 0.6 * np.cos(theta))
            sy = int(flower.cy + inner_radius * 0.6 * np.sin(theta))
            cv2.circle(image, (sx, sy), max(2, int(inner_radius * 0.5)),
                       (60, 170, 220), -1)

    if flower.occluder:
        cv2.ellipse(image, (flower.cx + radius // 2, flower.cy),
                    (int(radius * 0.9), int(radius * 0.5)),
                    30, 0, 360, (40, 115, 45), -1)


def _draw_pedicel(image: np.ndarray, flower: SyntheticFlower, length: int) -> None:
    """A thin, straight stalk - the male morphology."""
    minor = max(3, int(flower.radius * np.cos(np.deg2rad(flower.tilt_deg))))
    thickness = max(2, int(flower.radius * 0.14))
    cv2.line(image, (flower.cx, flower.cy + minor),
             (flower.cx, flower.cy + minor + length), (50, 130, 60), thickness)


# --------------------------------------------------------------------------- #
# Convenience scenes
# --------------------------------------------------------------------------- #


def male_female_pair(width: int = 640, height: int = 480,
                     radius: int = 62, seed: int = 1) -> SyntheticScene:
    """One staminate and one pistillate flower, side by side."""
    return render_scene([
        SyntheticFlower(cx=int(width * 0.28), cy=int(height * 0.42),
                        radius=radius, sex="male", pollen=0.85),
        SyntheticFlower(cx=int(width * 0.72), cy=int(height * 0.42),
                        radius=radius, sex="female"),
    ], width, height, seed)


def moving_flower(steps: int = 10, width: int = 640, height: int = 480,
                  radius: int = 55) -> list[SyntheticScene]:
    """A single flower translating across frames, for tracker tests."""
    scenes = []
    for step in range(steps):
        cx = int(width * 0.2 + (width * 0.6) * step / max(steps - 1, 1))
        scenes.append(render_scene([
            SyntheticFlower(cx=cx, cy=int(height * 0.5), radius=radius,
                            sex="female")
        ], width, height, seed=2))
    return scenes


def _draw_flower_depth(depth: np.ndarray, flower: SyntheticFlower,
                       flower_range_m: float) -> None:
    """Stamp a flower and its attached structure into the depth map.

    Deliberately mirrors the geometry drawn into the colour image: the corolla,
    plus either an ovary (female) or a pedicel (male) at the same range. This is
    what lets a depth-window segmentation recover the width profile that the
    sex cue measures.
    """
    radius = flower.radius
    minor = max(3, int(radius * np.cos(np.deg2rad(flower.tilt_deg))))

    if flower.stage == "bud":
        cv2.ellipse(depth, (flower.cx, flower.cy),
                    (int(radius * 0.45), int(radius * 0.8)),
                    flower.roll_deg, 0, 360, float(flower_range_m), -1)
        return

    if flower.sex == "female":
        ovary_h = int(radius * 0.85)
        ovary_w = int(radius * 0.55)
        cv2.ellipse(depth, (flower.cx, flower.cy + minor + int(ovary_h * 0.55)),
                    (ovary_w, ovary_h), 0, 0, 360, float(flower_range_m), -1)
    else:
        thickness = max(2, int(radius * 0.14))
        cv2.line(depth, (flower.cx, flower.cy + minor),
                 (flower.cx, flower.cy + minor + int(radius * 2.0)),
                 float(flower_range_m), thickness)

    cv2.ellipse(depth, (flower.cx, flower.cy), (radius, minor),
                flower.roll_deg, 0, 360, float(flower_range_m), -1)
