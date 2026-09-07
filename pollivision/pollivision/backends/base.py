"""Detector backend interface.

A backend turns a BGR frame into a list of ``Detection`` objects tagged with the
canonical role (``flower``, ``bud``, ``ovary``, ``distractor``) that the prompt
bank assigned. Keeping the interface this narrow is what lets the ensemble mix
open-vocabulary and supervised detectors without special-casing either.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..types import Detection


class DetectorBackend(ABC):
    """Common surface for every detector in the ensemble."""

    #: Identifier recorded on each Detection, used for per-source diagnostics.
    name: str = "base"

    #: Relative trust placed in this backend during box fusion.
    weight: float = 1.0

    #: Whether this backend emits instance masks. Mask-bearing backends unlock
    #: the morphology, orientation and occlusion cues; box-only backends still
    #: contribute to localisation.
    provides_masks: bool = False

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on a single BGR frame."""

    def warmup(self, shape: tuple[int, int] = (480, 640)) -> None:
        """Run one throwaway inference so the first real frame is not slow."""
        dummy = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        try:
            self.detect(dummy)
        except Exception:  # noqa: BLE001 - warmup must never break startup
            pass

    def close(self) -> None:
        """Release any held resources."""


class PromptBank:
    """Maps free-text prompts onto canonical roles.

    Open-vocabulary detectors are prompted with natural language, but the rest
    of the pipeline reasons about roles. This class owns both directions of that
    mapping and keeps the prompt list stable and ordered so text embeddings can
    be computed once and cached.
    """

    ROLES = ("flower", "bud", "ovary", "distractor")

    def __init__(self, mapping: dict[str, list[str]]):
        self.mapping = {role: list(prompts) for role, prompts in mapping.items() if prompts}
        self.prompts: list[str] = []
        self.roles: list[str] = []
        for role in self.ROLES:
            for prompt in self.mapping.get(role, []):
                self.prompts.append(prompt)
                self.roles.append(role)
        # Any role beyond the canonical four is still usable, just appended.
        for role, prompts in self.mapping.items():
            if role in self.ROLES:
                continue
            for prompt in prompts:
                self.prompts.append(prompt)
                self.roles.append(role)

    def role_of(self, class_index: int) -> str:
        if 0 <= class_index < len(self.roles):
            return self.roles[class_index]
        return "distractor"

    def prompt_of(self, class_index: int) -> str:
        if 0 <= class_index < len(self.prompts):
            return self.prompts[class_index]
        return "?"

    def __len__(self) -> int:
        return len(self.prompts)

    @classmethod
    def from_config(cls, cfg, path: str = "detector.prompts") -> "PromptBank":
        raw = cfg.get(path, {}) or {}
        if not raw:
            raise ValueError(f"No prompt bank found at config path '{path}'")
        return cls(raw)


def masks_to_full_frame(masks: Optional[np.ndarray], height: int, width: int) -> Optional[list[np.ndarray]]:
    """Resize a stack of low-resolution instance masks to the frame size."""
    if masks is None or len(masks) == 0:
        return None
    import cv2

    out = []
    for mask in masks:
        arr = np.asarray(mask, dtype=np.float32)
        if arr.shape[:2] != (height, width):
            arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_LINEAR)
        out.append(arr > 0.5)
    return out
