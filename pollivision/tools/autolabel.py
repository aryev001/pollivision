#!/usr/bin/env python3
"""Auto-label field imagery with the open-vocabulary ensemble.

This is the piece that removes the GPU from the critical path of *training*, as
opposed to inference. Hand-labelling several thousand flower boxes is the real
cost of a supervised detector; running a large open-vocabulary teacher over the
same images and exporting YOLO-format labels turns that into a review-and-correct
task instead.

The workflow is:

1. Capture field images (the rover's own ESP32-CAM is fine).
2. Run this with the largest teacher your patience allows - ``yoloe-11l-seg`` is
   slow on a CPU but much more accurate, and this is an offline batch job where
   latency does not matter.
3. Review and correct the labels. This step is not optional: the teacher will
   make systematic mistakes, and a student trained on unreviewed labels inherits
   them with extra confidence.
4. Fine-tune a small detector on free cloud GPU (``notebooks/colab_finetune.ipynb``).

Labels are written in YOLO format alongside a dataset YAML, so the output drops
straight into standard training tooling.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pollivision.config import load_config, parse_overrides  # noqa: E402
from pollivision.perception.detector import EnsembleDetector, associate_ovaries, split_roles  # noqa: E402
from pollivision.perception.sex import SexClassifier  # noqa: E402
from pollivision.backends.clip_embed import get_vlm  # noqa: E402

# Class order in the exported dataset. Sex is baked into the flower classes so a
# student detector can learn it directly, which is much cheaper at inference
# time than running a separate classifier head.
CLASSES = ["flower_male", "flower_female", "flower_unknown", "bud", "ovary"]


def to_yolo(box, width: int, height: int) -> tuple[float, float, float, float]:
    """Convert a pixel box to normalised YOLO cx,cy,w,h."""
    cx = np.clip(box.cx / width, 0.0, 1.0)
    cy = np.clip(box.cy / height, 0.0, 1.0)
    bw = np.clip(box.width / width, 0.0, 1.0)
    bh = np.clip(box.height / height, 0.0, 1.0)
    return float(cx), float(cy), float(bw), float(bh)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", help="directory of field images")
    parser.add_argument("--output", default="dataset", help="dataset output directory")
    parser.add_argument("--config", default="default")
    parser.add_argument("--species", default=None)
    parser.add_argument("--teacher", default="yoloe-11l-seg",
                        help="open-vocabulary teacher weights (bigger is better here)")
    parser.add_argument("--conf", type=float, default=0.15,
                        help="detection threshold; lower catches more, needs more review")
    parser.add_argument("--min-sex-confidence", type=float, default=0.45,
                        help="below this a flower is labelled flower_unknown")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--preview", action="store_true",
                        help="also write annotated previews for review")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    overlays = [f"species/{args.species}"] if args.species else []
    cfg = load_config(args.config, overlays, parse_overrides(args.set) if args.set else None)
    # Point the ensemble at the heavyweight teacher.
    cfg.set("detector.backends", [{
        "name": "yoloe", "weights": args.teacher, "enabled": True,
        "weight": 1.0, "conf": args.conf, "iou": 0.6, "imgsz": 800, "max_det": 100,
    }])

    detector = EnsembleDetector(cfg)
    sex_classifier = SexClassifier(cfg, vlm=get_vlm(cfg))

    image_paths = sorted(
        p for p in Path(args.images).rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not image_paths:
        print(f"No images found under {args.images}", file=sys.stderr)
        return 2

    root = Path(args.output)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    if args.preview:
        (root / "preview").mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    manifest = []
    rng = np.random.default_rng(0)

    for index, path in enumerate(image_paths):
        image = cv2.imread(str(path))
        if image is None:
            print(f"  skip (unreadable): {path.name}")
            continue
        height, width = image.shape[:2]

        detections, _ = detector.detect(image)
        by_role = split_roles(detections)
        flowers = by_role.get("flower", [])
        buds = by_role.get("bud", [])
        ovaries = by_role.get("ovary", [])

        lines = []
        if flowers:
            estimates = sex_classifier.classify(
                image,
                [d.box for d in flowers],
                [d.mask for d in flowers],
                associate_ovaries(flowers, ovaries),
            )
            for detection, estimate in zip(flowers, estimates):
                if estimate.confidence < args.min_sex_confidence:
                    name = "flower_unknown"
                else:
                    name = f"flower_{estimate.sex.value}"
                    if name not in CLASSES:
                        name = "flower_unknown"
                class_id = CLASSES.index(name)
                counts[name] += 1
                lines.append((class_id, to_yolo(detection.box, width, height)))

        for detection in buds:
            counts["bud"] += 1
            lines.append((CLASSES.index("bud"), to_yolo(detection.box, width, height)))
        for detection in ovaries:
            counts["ovary"] += 1
            lines.append((CLASSES.index("ovary"), to_yolo(detection.box, width, height)))

        split = "val" if rng.random() < args.val_split else "train"
        cv2.imwrite(str(root / "images" / split / path.name), image)
        label_path = root / "labels" / split / f"{path.stem}.txt"
        label_path.write_text(
            "\n".join(f"{cid} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f} {c[3]:.6f}"
                      for cid, c in lines),
            encoding="utf-8",
        )

        if args.preview and lines:
            preview = image.copy()
            for class_id, (cx, cy, bw, bh) in lines:
                x1 = int((cx - bw / 2) * width)
                y1 = int((cy - bh / 2) * height)
                x2 = int((cx + bw / 2) * width)
                y2 = int((cy + bh / 2) * height)
                cv2.rectangle(preview, (x1, y1), (x2, y2), (60, 220, 90), 2)
                cv2.putText(preview, CLASSES[class_id], (x1, max(12, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 220, 90), 1)
            cv2.imwrite(str(root / "preview" / f"{path.stem}_labels.jpg"), preview)

        manifest.append({"image": path.name, "split": split, "boxes": len(lines)})
        print(f"[{index + 1}/{len(image_paths)}] {path.name}: {len(lines)} boxes -> {split}")

    (root / "data.yaml").write_text(
        "# Auto-generated by tools/autolabel.py.\n"
        "# REVIEW THESE LABELS BEFORE TRAINING. The teacher makes systematic\n"
        "# mistakes, and an unreviewed student inherits them with more confidence.\n"
        f"path: {root.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n",
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nWrote dataset to {root.resolve()}")
    print("Class counts:")
    for name in CLASSES:
        print(f"  {name:<16} {counts.get(name, 0)}")

    total_flowers = sum(counts.get(f"flower_{s}", 0) for s in ("male", "female", "unknown"))
    unknown = counts.get("flower_unknown", 0)
    if total_flowers and unknown / total_flowers > 0.4:
        print(f"\nNote: {unknown}/{total_flowers} flowers were left unlabelled for sex. "
              "That usually means no depth channel was available, which is the cue the "
              "sex head leans on most. Review those by hand, or capture with RGB-D.")
    print("\nNext: review the labels (see preview/), then train with "
          "notebooks/colab_finetune.ipynb")
    return 0


if __name__ == "__main__":
    sys.exit(main())
