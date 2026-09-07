#!/usr/bin/env python3
"""Score PolliVision output against hand-labelled ground truth.

Consumes the JSON that ``pollivision detect --json`` writes plus a directory of
YOLO-format label files, and reports the metrics that actually matter for a
pollination rover rather than only the ones that are conventional.

Two reporting choices are deliberate:

* **Sex is scored three ways** - accuracy on decided flowers, abstention rate,
  and error rate on decided flowers. A rover that abstains often but is almost
  never wrong is far more useful than one that always commits and is wrong one
  time in six, because a confident wrong answer sends the probe to the wrong
  flower. Collapsing this into a single accuracy number hides the distinction
  that matters.

* **Per-class recall is reported separately for the sexes.** A missed pistillate
  flower is a fruit that never sets; a missed staminate flower usually just means
  collecting from the next one. They are not equally costly and should not be
  averaged.

Ground-truth label format (YOLO, one line per object)::

    <class_id> <cx> <cy> <w> <h>

with the class names supplied by --names (default matches tools/autolabel.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DEFAULT_NAMES = ["flower_male", "flower_female", "flower_unknown", "bud", "ovary"]


def iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 1e-9 else 0.0


def load_truth(path: Path, names: list[str],
               width: int, height: int) -> list[dict]:
    """Read one YOLO label file into pixel-space boxes."""
    if not path.exists():
        return []
    objects = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        objects.append({
            "name": names[class_id] if class_id < len(names) else str(class_id),
            "box": ((cx - bw / 2) * width, (cy - bh / 2) * height,
                    (cx + bw / 2) * width, (cy + bh / 2) * height),
        })
    return objects


def match(predictions: list[dict], truth: list[dict],
          threshold: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy IoU matching. Returns (pairs, unmatched_pred, unmatched_truth)."""
    pairs: list[tuple[int, int]] = []
    used_truth: set[int] = set()
    order = sorted(range(len(predictions)),
                   key=lambda i: predictions[i].get("score", 0.0), reverse=True)

    for pred_index in order:
        best, best_iou = -1, threshold
        for truth_index, target in enumerate(truth):
            if truth_index in used_truth:
                continue
            value = iou(predictions[pred_index]["box"], target["box"])
            if value >= best_iou:
                best, best_iou = truth_index, value
        if best >= 0:
            pairs.append((pred_index, best))
            used_truth.add(best)

    matched_pred = {p for p, _ in pairs}
    return (pairs,
            [i for i in range(len(predictions)) if i not in matched_pred],
            [i for i in range(len(truth)) if i not in used_truth])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", help="JSON from `pollivision detect --json`")
    parser.add_argument("labels", help="directory of YOLO .txt ground-truth files")
    parser.add_argument("--names", nargs="*", default=DEFAULT_NAMES)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--report", default=None, help="write a Markdown report here")
    args = parser.parse_args()

    records = json.loads(Path(args.results).read_text(encoding="utf-8"))
    label_dir = Path(args.labels)

    detection = Counter()
    sex_counts = Counter()
    sex_confusion: dict[tuple[str, str], int] = defaultdict(int)
    stage_counts = Counter()
    recall_by_sex: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [found, total]
    ranges: list[float] = []

    for record in records:
        stem = Path(record["frame"]).stem
        flowers = record.get("flowers", [])
        if not flowers:
            # Without a stored frame size, an image with no detections carries no
            # usable geometry; count its ground truth as missed.
            truth = load_truth(label_dir / f"{stem}.txt", args.names, 1, 1)
            detection["fn"] += len([t for t in truth if t["name"].startswith("flower")])
            continue

        # Recover the frame size from the largest box seen; results carry pixel
        # boxes but not the image dimensions.
        max_x = max(f["box"][2] for f in flowers)
        max_y = max(f["box"][3] for f in flowers)
        width = int(record.get("width", max(max_x * 1.05, 640)))
        height = int(record.get("height", max(max_y * 1.05, 480)))

        truth = [t for t in load_truth(label_dir / f"{stem}.txt", args.names, width, height)
                 if t["name"].startswith("flower")]
        predictions = [{"box": tuple(f["box"]), "score": f.get("sex_confidence", 0.5),
                        "flower": f} for f in flowers]

        pairs, false_positives, false_negatives = match(predictions, truth, args.iou)
        detection["tp"] += len(pairs)
        detection["fp"] += len(false_positives)
        detection["fn"] += len(false_negatives)

        for target_index in false_negatives:
            true_sex = truth[target_index]["name"].replace("flower_", "")
            recall_by_sex[true_sex][1] += 1

        for pred_index, target_index in pairs:
            flower = predictions[pred_index]["flower"]
            true_sex = truth[target_index]["name"].replace("flower_", "")
            recall_by_sex[true_sex][0] += 1
            recall_by_sex[true_sex][1] += 1

            if true_sex not in ("male", "female"):
                continue  # ambiguous ground truth is excluded from the sex metric

            predicted = flower.get("sex", "unknown")
            sex_confusion[(true_sex, predicted)] += 1
            if predicted == "unknown":
                sex_counts["abstained"] += 1
            elif predicted == true_sex:
                sex_counts["correct"] += 1
            else:
                sex_counts["wrong"] += 1

            stage = flower.get("stage")
            if stage:
                stage_counts[stage] += 1
            if flower.get("range_m"):
                ranges.append(float(flower["range_m"]))

    lines = ["# PolliVision evaluation", ""]

    tp, fp, fn = detection["tp"], detection["fp"], detection["fn"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    lines += [
        f"## Detection (IoU >= {args.iou})", "",
        f"- Images: {len(records)}",
        f"- True positives: {tp}, false positives: {fp}, false negatives: {fn}",
        f"- **Precision {precision:.3f} | Recall {recall:.3f} | F1 {f1:.3f}**",
        "",
        "Recall by true sex (a missed pistillate flower is a fruit that never sets;",
        "a missed staminate flower usually just costs one visit):", "",
    ]
    for sex, (found, total) in sorted(recall_by_sex.items()):
        if total:
            lines.append(f"- `{sex}`: {found}/{total} = {found / total:.3f}")
    lines.append("")

    decided = sex_counts["correct"] + sex_counts["wrong"]
    total_sex = decided + sex_counts["abstained"]
    lines += ["## Sex classification", ""]
    if total_sex:
        lines += [
            f"- Flowers with unambiguous ground truth: {total_sex}",
            f"- Abstained: {sex_counts['abstained']} "
            f"({sex_counts['abstained'] / total_sex:.1%})",
            f"- **Accuracy on decided flowers: "
            f"{sex_counts['correct'] / max(decided, 1):.3f}** ({decided} decided)",
            f"- **Error rate on decided flowers: "
            f"{sex_counts['wrong'] / max(decided, 1):.3f}**  <- the number that matters",
            "",
            "A confident wrong answer sends the probe to the wrong flower; an",
            "abstention only costs a visit. Tune `perception.sex.min_confidence`",
            "to move along this trade-off deliberately.",
            "", "Confusion (true -> predicted):", "",
        ]
        for (true_sex, predicted), count in sorted(sex_confusion.items()):
            lines.append(f"- {true_sex} -> {predicted}: {count}")
    else:
        lines.append("- No unambiguous sex labels found.")
    lines.append("")

    if stage_counts:
        lines += ["## Anthesis stages predicted", ""]
        total_stages = sum(stage_counts.values())
        for stage, count in stage_counts.most_common():
            lines.append(f"- {stage}: {count} ({count / total_stages:.1%})")
        lines.append("")

    if ranges:
        values = np.array(ranges)
        lines += [
            "## Range estimates", "",
            f"- n={len(values)}, median {np.median(values) * 100:.1f} cm, "
            f"IQR {np.percentile(values, 25) * 100:.1f}-"
            f"{np.percentile(values, 75) * 100:.1f} cm",
            "",
            "Compare against tape-measured standoff. The size-prior estimator's",
            "error is multiplicative, so report mean absolute *percentage* error.",
            "",
        ]

    report = "\n".join(lines)
    print(report)
    if args.report:
        Path(args.report).write_text(report, encoding="utf-8")
        print(f"\nWrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
