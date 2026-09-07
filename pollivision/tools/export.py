#!/usr/bin/env python3
"""Export a trained detector for deployment on rover compute.

PyTorch on a CPU is the slowest reasonable way to run a detector. Once a student
model has been fine-tuned, exporting it buys a large speedup on the same
hardware with no accuracy change worth worrying about:

    onnx      - portable, works everywhere, roughly 1.5-2x faster than PyTorch
                on a CPU with onnxruntime. The safe default.
    openvino  - best option on Intel-based SBCs; often 2-4x.
    ncnn      - best option on ARM (Raspberry Pi, Orange Pi), which is what a
                low-cost ground rover is most likely to be carrying.
    tflite    - for Coral / mobile deployment.

Note this exports a *supervised* student. The open-vocabulary teacher cannot be
exported usefully, because its class set is defined by text embeddings computed
at load time - which is exactly the flexibility that makes it useful for
labelling and useless for a fixed-function deployment.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FORMATS = ("onnx", "openvino", "ncnn", "tflite", "torchscript")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("weights", help="path to the fine-tuned .pt model")
    parser.add_argument("--format", default="onnx", choices=FORMATS)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true",
                        help="FP16; ignored on most CPU targets")
    parser.add_argument("--int8", action="store_true",
                        help="INT8 quantisation, needs a calibration set")
    parser.add_argument("--benchmark", action="store_true",
                        help="time the exported model against the original")
    args = parser.parse_args()

    from ultralytics import YOLO

    from pollivision.zoo import configure_ultralytics_cache

    configure_ultralytics_cache()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"No such weights file: {weights}", file=sys.stderr)
        return 2

    model = YOLO(str(weights))
    print(f"Exporting {weights.name} to {args.format} at {args.imgsz}px...")
    exported = model.export(format=args.format, imgsz=args.imgsz,
                            half=args.half, int8=args.int8)
    print(f"Wrote {exported}")

    if args.benchmark:
        frame = (np.random.rand(args.imgsz, args.imgsz, 3) * 255).astype(np.uint8)
        print("\nBenchmarking (10 iterations after 3 warmup)...")
        results = {}
        for label, handle in (("pytorch", model), (args.format, YOLO(str(exported)))):
            for _ in range(3):
                handle.predict(frame, verbose=False, imgsz=args.imgsz)
            started = time.perf_counter()
            for _ in range(10):
                handle.predict(frame, verbose=False, imgsz=args.imgsz)
            results[label] = (time.perf_counter() - started) / 10 * 1000
            print(f"  {label:<12} {results[label]:7.1f} ms")
        if len(results) == 2:
            baseline, exported_ms = results["pytorch"], results[args.format]
            print(f"\nSpeedup: {baseline / max(exported_ms, 1e-6):.2f}x")

    print("\nPoint your config at the exported model:")
    print("  detector:\n    backends:\n      - name: finetuned\n"
          f"        weights: {exported}\n        enabled: true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
