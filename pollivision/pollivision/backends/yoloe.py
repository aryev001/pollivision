"""YOLOE open-vocabulary detection + instance segmentation backend.

This is the default detector, chosen for three reasons that matter to this
project specifically:

* **It needs no training.** Classes are set from natural-language prompts, so a
  cucurbit flower detector exists the moment the weights are downloaded. That
  removes the GPU requirement from the critical path entirely.
* **It emits instance masks.** The corolla mask is not a nicety here - it drives
  ellipse fitting (orientation and tilt-invariant ranging), the occlusion gate,
  and the anther/stigma region extraction. A box-only detector would force all
  three to fall back on cruder approximations.
* **Its text encoder is MobileCLIP, distributed on GitHub release assets.** The
  more common CLIP ViT-B/32 path pulls weights from hosts that a lot of
  institutional and agricultural networks block.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..logging_utils import get_logger
from ..types import BBox, Detection
from ..zoo import configure_ultralytics_cache, resolve
from .base import DetectorBackend, PromptBank, masks_to_full_frame

LOGGER = get_logger(__name__)


class YoloEBackend(DetectorBackend):
    """Text-prompted YOLOE detector/segmenter."""

    name = "yoloe"
    provides_masks = True

    def __init__(
        self,
        prompts: PromptBank,
        weights: str = "yoloe-11s-seg",
        conf: float = 0.10,
        iou: float = 0.60,
        imgsz: int = 640,
        max_det: int = 60,
        device: str = "cpu",
        weight: float = 1.0,
    ) -> None:
        from ultralytics import YOLOE

        configure_ultralytics_cache()
        self.prompts = prompts
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.max_det = int(max_det)
        self.device = device
        self.weight = float(weight)

        path = resolve(weights)
        LOGGER.info("Loading YOLOE weights from %s", path)
        self.model = YOLOE(str(path))

        # Text embeddings are computed once here. This is the expensive part of
        # an open-vocabulary detector (it loads a 570 MB CLIP bundle), and doing
        # it per-frame would dominate the loop.
        LOGGER.info("Encoding %d text prompts", len(self.prompts))
        names = list(self.prompts.prompts)
        self.model.set_classes(names, self.model.get_text_pe(names))
        self._release_text_encoder()

    def _release_text_encoder(self) -> None:
        """Drop the cached CLIP text tower once prompts are embedded.

        Prompts are fixed for the duration of a run, so holding the text encoder
        costs several hundred megabytes of RAM for nothing. On a 1-2 GB SBC that
        is the difference between running and being OOM-killed.
        """
        try:
            inner = getattr(self.model, "model", None)
            if inner is not None and getattr(inner, "clip_model", None) is not None:
                inner.clip_model = None
                LOGGER.debug("Released cached text encoder")
        except Exception as exc:  # noqa: BLE001 - purely an optimisation
            LOGGER.debug("Could not release text encoder: %s", exc)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        xyxy = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        full_masks: Optional[list[np.ndarray]] = None
        if result.masks is not None and result.masks.data is not None:
            full_masks = masks_to_full_frame(
                result.masks.data.cpu().numpy(), height, width
            )

        detections: list[Detection] = []
        for index, (box, score, cls) in enumerate(zip(xyxy, scores, classes)):
            mask = None
            if full_masks is not None and index < len(full_masks):
                mask = full_masks[index]
            detections.append(
                Detection(
                    box=BBox.from_xyxy(box).clip(width, height),
                    score=float(score),
                    label=self.prompts.role_of(int(cls)),
                    source=self.name,
                    mask=mask,
                    meta={"prompt": self.prompts.prompt_of(int(cls)),
                          "class_index": int(cls)},
                )
            )
        return detections
