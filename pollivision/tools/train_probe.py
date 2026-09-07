#!/usr/bin/env python3
"""Fit the optional sex linear probe from a handful of labelled crops.

This is the cheapest possible adaptation path. Point it at two directories of
cropped flower images - one male, one female, a few dozen each is enough - and
it embeds them with the vision-language model and fits a logistic regression
over those embeddings. Fitting takes under a second on a CPU.

The result slots in as a fourth cue in the sex head with the highest default
weight, because a probe trained on the operator's own cultivar and lighting
beats a zero-shot prompt at that specific task. The zero-shot cues keep working
unchanged if the probe is never trained, so this is strictly optional.

    python tools/train_probe.py --male crops/male --female crops/female \\
        --output sex_probe.npz

Then set ``perception.sex.probe_path: sex_probe.npz`` in your config.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pollivision.backends.clip_embed import get_vlm  # noqa: E402
from pollivision.config import load_config  # noqa: E402
from pollivision.fusion.calibration import LinearProbe  # noqa: E402

SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_crops(directory: Path) -> list[np.ndarray]:
    crops = []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in SUFFIXES:
            continue
        image = cv2.imread(str(path))
        if image is not None:
            crops.append(image)
    return crops


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--male", required=True, help="directory of staminate crops")
    parser.add_argument("--female", required=True, help="directory of pistillate crops")
    parser.add_argument("--output", default="sex_probe.npz")
    parser.add_argument("--config", default="default")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--val-split", type=float, default=0.25)
    args = parser.parse_args()

    cfg = load_config(args.config)
    vlm = get_vlm(cfg)
    if vlm is None:
        print("The vision-language model is required to fit a probe.", file=sys.stderr)
        return 2

    male = load_crops(Path(args.male))
    female = load_crops(Path(args.female))
    if len(male) < 10 or len(female) < 10:
        print(f"Need at least 10 crops per class; got {len(male)} male, "
              f"{len(female)} female.", file=sys.stderr)
        return 2
    print(f"Embedding {len(male)} male and {len(female)} female crops...")

    features = np.vstack([
        vlm.encode_images(male).cpu().numpy(),
        vlm.encode_images(female).cpu().numpy(),
    ]).astype(np.float32)
    # Label 1 = female, matching the p_female convention of the sex head.
    labels = np.hstack([np.zeros(len(male)), np.ones(len(female))]).astype(np.float32)

    rng = np.random.default_rng(0)
    order = rng.permutation(len(labels))
    features, labels = features[order], labels[order]
    split = int(len(labels) * (1.0 - args.val_split))
    if split < 8 or len(labels) - split < 4:
        print("Too few samples to hold out a validation split; training on everything.")
        split = len(labels)

    probe = LinearProbe.fit(features[:split], labels[:split], ["male", "female"],
                            epochs=args.epochs)

    train_accuracy = float(((probe.predict_proba(features[:split]) > 0.5)
                            == labels[:split].astype(bool)).mean())
    print(f"\nTrain accuracy: {train_accuracy:.3f}")
    if split < len(labels):
        held_out = features[split:], labels[split:]
        accuracy = float(((probe.predict_proba(held_out[0]) > 0.5)
                          == held_out[1].astype(bool)).mean())
        print(f"Held-out accuracy: {accuracy:.3f}  ({len(held_out[1])} samples)")
        if accuracy < 0.7:
            print("\nHeld-out accuracy is low. Usually this means the crops are "
                  "inconsistent (different framing between the two classes) or the "
                  "context below the corolla was cropped away - that is where the "
                  "ovary is, and without it the classes are genuinely hard to tell "
                  "apart.")

    probe.save(args.output)
    print(f"\nSaved probe to {args.output}")
    print(f"Enable it with:\n  perception:\n    sex:\n      probe_path: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
