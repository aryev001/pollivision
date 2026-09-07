"""Optional detector backends.

None of these are required. They exist so the ensemble can be strengthened when
the deployment environment allows it:

* ``YoloWorldBackend`` - a second open-vocabulary opinion. Its text encoder is
  OpenAI CLIP ViT-B/32, fetched from a host that some networks block, so it is
  disabled by default rather than made a hard dependency.
* ``Owlv2Backend`` - stronger zero-shot recall on small objects than either YOLO
  variant, at a large latency cost. Worth enabling for offline auto-labelling
  even when it is far too slow for the live loop.
* ``FineTunedBackend`` - a supervised detector trained on the user's own crop.
  Once trained this is the most accurate member and is given the highest fusion
  weight; the open-vocabulary backends then act as a recall safety net.
"""

from __future__ import annotations

import numpy as np

from ..logging_utils import get_logger
from ..types import BBox, Detection
from ..zoo import configure_ultralytics_cache, resolve
from .base import DetectorBackend, PromptBank, masks_to_full_frame

LOGGER = get_logger(__name__)


class YoloWorldBackend(DetectorBackend):
    """YOLO-World open-vocabulary detector (boxes only)."""

    name = "yolo_world"
    provides_masks = False

    def __init__(self, prompts: PromptBank, weights: str = "yolov8s-worldv2.pt",
                 conf: float = 0.10, iou: float = 0.6, imgsz: int = 640,
                 max_det: int = 60, device: str = "cpu", weight: float = 0.7) -> None:
        from ultralytics import YOLOWorld

        configure_ultralytics_cache()
        self.prompts = prompts
        self.conf, self.iou, self.imgsz = float(conf), float(iou), int(imgsz)
        self.max_det, self.device, self.weight = int(max_det), device, float(weight)
        self.model = YOLOWorld(str(weights))
        self.model.set_classes(list(prompts.prompts))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        results = self.model.predict(frame, conf=self.conf, iou=self.iou,
                                     imgsz=self.imgsz, max_det=self.max_det,
                                     device=self.device, verbose=False)
        if not results or results[0].boxes is None:
            return []
        boxes = results[0].boxes
        return [
            Detection(
                box=BBox.from_xyxy(xyxy).clip(width, height),
                score=float(score),
                label=self.prompts.role_of(int(cls)),
                source=self.name,
                meta={"prompt": self.prompts.prompt_of(int(cls))},
            )
            for xyxy, score, cls in zip(
                boxes.xyxy.cpu().numpy(),
                boxes.conf.cpu().numpy(),
                boxes.cls.cpu().numpy().astype(int),
            )
        ]


class Owlv2Backend(DetectorBackend):
    """OWLv2 open-vocabulary detector via the transformers library."""

    name = "owlv2"
    provides_masks = False

    def __init__(self, prompts: PromptBank,
                 weights: str = "google/owlv2-base-patch16-ensemble",
                 conf: float = 0.12, device: str = "cpu", weight: float = 0.8,
                 **_: object) -> None:
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.torch = torch
        self.prompts = prompts
        self.conf, self.device, self.weight = float(conf), device, float(weight)
        self.processor = Owlv2Processor.from_pretrained(weights)
        self.model = Owlv2ForObjectDetection.from_pretrained(weights).to(device).eval()
        self._queries = [list(prompts.prompts)]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        from PIL import Image

        height, width = frame.shape[:2]
        image = Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1]))
        inputs = self.processor(text=self._queries, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = self.torch.tensor([[height, width]], device=self.device)
        processed = self.processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=self.conf
        )[0]
        return [
            Detection(
                box=BBox.from_xyxy(box.tolist()).clip(width, height),
                score=float(score),
                label=self.prompts.role_of(int(label)),
                source=self.name,
                meta={"prompt": self.prompts.prompt_of(int(label))},
            )
            for box, score, label in zip(
                processed["boxes"].cpu(), processed["scores"].cpu(), processed["labels"].cpu()
            )
        ]


class FineTunedBackend(DetectorBackend):
    """A supervised YOLO detector trained on the operator's own imagery.

    Class names in the trained model are mapped onto canonical roles by
    substring match, so a model with classes like ``male_flower`` and
    ``female_flower`` slots in without configuration. When such a model also
    encodes sex directly, that prediction is preserved in ``meta['sex_hint']``
    and consumed by the sex head as an additional cue.
    """

    name = "finetuned"

    def __init__(self, weights: str, conf: float = 0.25, iou: float = 0.6,
                 imgsz: int = 640, max_det: int = 60, device: str = "cpu",
                 weight: float = 1.5, prompts: PromptBank | None = None) -> None:
        from ultralytics import YOLO

        configure_ultralytics_cache()
        if not weights:
            raise ValueError("FineTunedBackend requires a path to trained weights")
        path = resolve(weights)
        self.model = YOLO(str(path))
        self.conf, self.iou, self.imgsz = float(conf), float(iou), int(imgsz)
        self.max_det, self.device, self.weight = int(max_det), device, float(weight)
        self.names = dict(getattr(self.model, "names", {}) or {})
        self.provides_masks = str(getattr(self.model, "task", "")) == "segment"
        LOGGER.info("Fine-tuned backend classes: %s", self.names)

    @staticmethod
    def _role_and_sex(class_name: str) -> tuple[str, str | None]:
        lowered = class_name.lower()
        sex = None
        if "female" in lowered or "pistillate" in lowered:
            sex = "female"
        elif "male" in lowered or "staminate" in lowered:
            sex = "male"
        if "ovary" in lowered or "fruitlet" in lowered:
            return "ovary", sex
        if "bud" in lowered:
            return "bud", sex
        if "flower" in lowered or "blossom" in lowered or sex is not None:
            return "flower", sex
        return "distractor", sex

    def detect(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        results = self.model.predict(frame, conf=self.conf, iou=self.iou,
                                     imgsz=self.imgsz, max_det=self.max_det,
                                     device=self.device, verbose=False)
        if not results or results[0].boxes is None:
            return []
        result = results[0]
        full_masks = None
        if result.masks is not None and result.masks.data is not None:
            full_masks = masks_to_full_frame(result.masks.data.cpu().numpy(), height, width)

        detections = []
        for index, (xyxy, score, cls) in enumerate(zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy().astype(int),
        )):
            class_name = str(self.names.get(int(cls), cls))
            role, sex = self._role_and_sex(class_name)
            meta: dict[str, object] = {"class_name": class_name}
            if sex:
                meta["sex_hint"] = sex
            detections.append(Detection(
                box=BBox.from_xyxy(xyxy).clip(width, height),
                score=float(score),
                label=role,
                source=self.name,
                mask=full_masks[index] if full_masks and index < len(full_masks) else None,
                meta=meta,
            ))
        return detections
