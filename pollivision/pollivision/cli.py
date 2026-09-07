"""Command-line interface.

    pollivision fetch      - prime the weight cache for offline field use
    pollivision detect     - run perception over images/video and report
    pollivision run        - run the full closed-loop mission
    pollivision stream     - view a live ESP32-CAM feed with annotations
    pollivision calibrate  - solve camera intrinsics from a checkerboard
    pollivision bench      - measure per-stage latency on this machine
    pollivision selftest   - verify the install end to end
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .config import load_config, parse_overrides
from .logging_utils import get_logger

LOGGER = get_logger("pollivision.cli")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="default", help="base config name or path")
    parser.add_argument("--species", default=None,
                        help="species profile: pumpkin, cucumber, squash, watermelon, muskmelon")
    parser.add_argument("--esp32", action="store_true",
                        help="apply the ESP32-CAM tuning overlay")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="config overrides, e.g. --set electrostatic.reference.voltage_kv=7")
    parser.add_argument("--device", default=None, help="torch device (default: cpu)")


def _build_config(args):
    overlays = []
    if args.species:
        overlays.append(f"species/{args.species}")
    if getattr(args, "esp32", False):
        overlays.append("esp32cam")
    overrides = parse_overrides(args.set) if args.set else None
    cfg = load_config(args.config, overlays, overrides)
    if args.device:
        cfg.set("runtime.device", args.device)
    return cfg


# --------------------------------------------------------------------------- #


def cmd_fetch(args) -> int:
    from .zoo import REGISTRY, fetch_all

    names = args.model or ["yoloe-11s-seg", "mobileclip-blt"]
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        print(f"Unknown model(s): {unknown}\nAvailable: {sorted(REGISTRY)}", file=sys.stderr)
        return 2

    print(f"Fetching {len(names)} model(s) into the cache...")
    paths = fetch_all(names)
    for name, path in paths.items():
        size_mb = path.stat().st_size / (1 << 20)
        print(f"  {name:<20} {size_mb:8.1f} MB  {path}")
    print("\nCache primed. Set POLLIVISION_OFFLINE=1 to run without network access.")
    return 0


def cmd_detect(args) -> int:
    import cv2

    from .io.sources import open_source
    from .io.visualize import draw_frame
    from .runtime.pipeline import PerceptionPipeline

    cfg = _build_config(args)
    pipeline = PerceptionPipeline(cfg)

    output_dir = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    processed = 0
    with open_source(args.source, cfg, max_frames=args.max_frames) as source:
        for frame in source:
            result = pipeline.process(frame.image)
            processed += 1

            print(f"[{frame.index:5d}] {frame.name:<24} "
                  f"{len(result.flowers)} flowers "
                  f"(M{len(result.males)}/F{len(result.females)}), "
                  f"{len(result.rejected)} rejected, "
                  f"{result.latency_ms.get('total', 0):.0f} ms")
            for observation in result.flowers:
                print(f"          {_describe(observation)}")

            records.append(_record(frame.name, result))

            if output_dir:
                annotated = draw_frame(frame.image, result)
                cv2.imwrite(str(output_dir / f"{Path(frame.name).stem}_annotated.jpg"),
                            annotated)

            if args.max_frames and processed >= args.max_frames:
                break

    if args.json:
        Path(args.json).write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"\nWrote {len(records)} frame records to {args.json}")
    if output_dir:
        print(f"Wrote annotated frames to {output_dir}")
    return 0


def cmd_run(args) -> int:
    import cv2

    from .io.sources import open_source
    from .io.telemetry import build_telemetry
    from .io.visualize import draw_frame
    from .runtime.mission import MissionController, MissionState
    from .runtime.pipeline import PerceptionPipeline

    cfg = _build_config(args)
    pipeline = PerceptionPipeline(cfg)
    mission = MissionController(cfg)
    telemetry = build_telemetry(cfg)

    writer = None
    events = []

    try:
        with open_source(args.source, cfg, max_frames=args.max_frames) as source:
            for frame in source:
                result = pipeline.process(frame.image)
                environment = telemetry.read()

                # Without a real motion controller reporting probe pose, assume
                # the probe reaches its commanded pose. This is what makes the
                # loop runnable against recorded video; a real rover passes its
                # own readiness flag here instead.
                step = mission.step(
                    result, environment, frame=frame.image,
                    probe_at_pose=not args.require_pose_feedback,
                )

                line = (f"[{frame.index:5d}] {step.state.value:<9} {step.message}")
                if step.command is not None and step.command.voltage_kv > 0:
                    line += (f"  | {step.command.voltage_kv:.2f} kV "
                             f"{step.command.exposure_ms:.0f} ms")
                print(line)

                events.append({
                    "frame": frame.index,
                    "state": step.state.value,
                    "message": step.message,
                    "humidity": environment.humidity,
                    "command": step.command.to_wire() if step.command else None,
                    "verification": (
                        {"success": step.verification.success,
                         "note": step.verification.note}
                        if step.verification else None
                    ),
                })

                if args.video_out:
                    annotated = draw_frame(frame.image, result, step.command,
                                           step.state.value, step.message)
                    if writer is None:
                        height, width = annotated.shape[:2]
                        writer = cv2.VideoWriter(
                            args.video_out, cv2.VideoWriter_fourcc(*"mp4v"),
                            float(args.fps), (width, height))
                    writer.write(annotated)
    finally:
        if writer is not None:
            writer.release()
        telemetry.close()

    print("\n--- mission summary ---")
    for key, value in mission.ledger.stats().items():
        print(f"  {key:<18} {value}")
    if args.json:
        Path(args.json).write_text(json.dumps(events, indent=2), encoding="utf-8")
        print(f"  wrote event log to {args.json}")
    if args.video_out:
        print(f"  wrote annotated video to {args.video_out}")
    return 0


def cmd_stream(args) -> int:
    import cv2

    from .io.esp32 import Esp32CamStream
    from .io.visualize import draw_frame
    from .runtime.pipeline import PerceptionPipeline

    cfg = _build_config(args)
    if not args.no_model:
        pipeline = PerceptionPipeline(cfg)
    else:
        pipeline = None

    url = args.url or cfg.get("source.url", "http://192.168.4.1:81/stream")
    print(f"Connecting to {url} (ctrl-c to stop)")

    with Esp32CamStream(url, max_queue=int(cfg.get("source.max_queue", 2))) as stream:
        try:
            for frame in stream.frames():
                if pipeline is not None:
                    result = pipeline.process(frame.image)
                    canvas = draw_frame(frame.image, result)
                    status = (f"{len(result.flowers)} flowers, "
                              f"{result.latency_ms.get('total', 0):.0f} ms")
                else:
                    canvas = frame.image
                    status = "passthrough"

                print(f"\r[{frame.index:6d}] {status}, "
                      f"age {frame.age_s * 1000:.0f} ms, "
                      f"dropped {stream.dropped_frames}    ", end="", flush=True)

                if args.show:
                    cv2.imshow("PolliVision - ESP32-CAM", canvas)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if args.save_dir:
                    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(f"{args.save_dir}/frame_{frame.index:06d}.jpg", canvas)
        except KeyboardInterrupt:
            print("\nstopped")
        finally:
            if args.show:
                cv2.destroyAllWindows()
    return 0


def cmd_calibrate(args) -> int:
    import cv2

    from .geometry.camera import CameraIntrinsics

    pattern = tuple(int(v) for v in args.pattern.split("x"))
    square_m = float(args.square_size_mm) / 1000.0

    object_template = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    object_template[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    object_template *= square_m

    object_points, image_points = [], []
    image_size = None
    paths = sorted(Path(args.images).glob("*"))
    if not paths:
        print(f"No images found in {args.images}", file=sys.stderr)
        return 2

    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, pattern, None)
        if not found:
            print(f"  {path.name}: no checkerboard found")
            continue
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        object_points.append(object_template)
        image_points.append(corners)
        print(f"  {path.name}: ok")

    if len(object_points) < 5:
        print(f"Only {len(object_points)} usable views; need at least 5 "
              "(10-20 from varied angles is better).", file=sys.stderr)
        return 2

    rms, matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None)

    intrinsics = CameraIntrinsics(
        fx=float(matrix[0, 0]), fy=float(matrix[1, 1]),
        cx=float(matrix[0, 2]), cy=float(matrix[1, 2]),
        width=image_size[0], height=image_size[1],
        distortion=distortion.reshape(-1), name=args.name,
    )
    intrinsics.save(args.output)
    print(f"\nRMS reprojection error: {rms:.4f} px "
          f"({'good' if rms < 1.0 else 'high - recapture with sharper, more varied views'})")
    print(f"fx={intrinsics.fx:.2f} fy={intrinsics.fy:.2f} "
          f"cx={intrinsics.cx:.2f} cy={intrinsics.cy:.2f}")
    print(f"Horizontal FOV: {intrinsics.hfov_deg:.2f} deg")
    print(f"Saved to {args.output}")
    print("\nAdd to your config:\n"
          f"  camera:\n    fx: {intrinsics.fx:.4f}\n    fy: {intrinsics.fy:.4f}\n"
          f"    cx: {intrinsics.cx:.4f}\n    cy: {intrinsics.cy:.4f}")
    return 0


def cmd_bench(args) -> int:
    from .runtime.pipeline import PerceptionPipeline

    cfg = _build_config(args)
    pipeline = PerceptionPipeline(cfg)

    width = int(cfg.get("camera.width", 640))
    height = int(cfg.get("camera.height", 480))
    frame = _synthetic_scene(height, width)

    print(f"Warming up at {width}x{height}...")
    pipeline.warmup((height, width))

    totals: dict[str, list[float]] = {}
    for index in range(args.iterations):
        result = pipeline.process(frame)
        for key, value in result.latency_ms.items():
            totals.setdefault(key, []).append(value)
        print(f"  iteration {index + 1}/{args.iterations}: "
              f"{result.latency_ms.get('total', 0):.0f} ms")

    print(f"\nPer-stage latency over {args.iterations} iterations (ms):")
    print(f"  {'stage':<22} {'mean':>8} {'min':>8} {'max':>8}")
    for key, values in sorted(totals.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"  {key:<22} {np.mean(values):8.1f} {np.min(values):8.1f} "
              f"{np.max(values):8.1f}")
    total = float(np.mean(totals.get("total", [0.0])))
    if total > 0:
        print(f"\nSustained rate: {1000.0 / total:.2f} FPS "
              f"(detector runs every {pipeline.detect_every} frame(s), "
              f"so the effective loop rate is "
              f"{1000.0 / total * pipeline.detect_every:.2f} FPS)")
    return 0


def cmd_selftest(args) -> int:
    """Verify that the install works end to end on this machine."""
    from .runtime.mission import MissionController
    from .runtime.pipeline import PerceptionPipeline
    from .types import EnvironmentReading

    cfg = _build_config(args)
    failures = []

    print("1. config          ", end="", flush=True)
    print(f"ok (species={cfg.get('species.name')}, "
          f"{len(cfg.get('detector.backends', []))} backends declared)")

    print("2. weights         ", end="", flush=True)
    try:
        from .zoo import resolve

        path = resolve("yoloe-11s-seg")
        print(f"ok ({path.name}, {path.stat().st_size / (1 << 20):.0f} MB)")
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {exc}")
        failures.append("weights")

    print("3. pipeline build  ", end="", flush=True)
    try:
        pipeline = PerceptionPipeline(cfg)
        backends = [b.name for b in pipeline.detector.backends]
        vlm = "yes" if pipeline.vlm is not None else "no"
        print(f"ok (detectors={backends}, vlm={vlm})")
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {exc}")
        return 1

    print("4. inference       ", end="", flush=True)
    try:
        frame = _synthetic_scene(480, 640)
        started = time.perf_counter()
        result = pipeline.process(frame)
        elapsed = (time.perf_counter() - started) * 1000
        print(f"ok ({elapsed:.0f} ms, {len(result.flowers)} flowers, "
              f"{len(result.rejected)} rejected)")
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {exc}")
        failures.append("inference")
        return 1

    print("5. mission loop    ", end="", flush=True)
    try:
        mission = MissionController(cfg)
        step = mission.step(result, EnvironmentReading(humidity=60.0), frame=frame)
        print(f"ok (state={step.state.value})")
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {exc}")
        failures.append("mission")

    print("6. controller      ", end="", flush=True)
    try:
        from .control.electrostatic import AdaptiveElectrostaticController
        from .types import BBox, FlowerObservation, PollenEstimate, ProbeMode

        controller = AdaptiveElectrostaticController(cfg)
        probe = FlowerObservation(box=BBox(0, 0, 40, 40), score=0.9)
        probe.pollen = PollenEstimate(availability=0.7)
        probe.incidence_deg, probe.range_m = 20.0, 0.04
        command = controller.compute(probe, EnvironmentReading(humidity=70.0),
                                     ProbeMode.COLLECT)
        print(f"ok ({command.voltage_kv:.2f} kV, {command.exposure_ms:.0f} ms, "
              f"tip field {command.tip_field_kv_per_m:.0f} kV/m)")
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {exc}")
        failures.append("controller")

    if failures:
        print(f"\nSelf-test FAILED in: {', '.join(failures)}")
        return 1
    print("\nSelf-test passed. The stack is ready to run.")
    return 0


# --------------------------------------------------------------------------- #


def _describe(observation) -> str:
    parts = [
        f"#{observation.track_id}" if observation.track_id is not None else "-",
        f"{observation.sex.sex.value:<7}",
        f"p(F)={observation.sex.p_female:.2f}",
        f"conf={observation.sex.confidence:.2f}",
        f"{observation.anthesis.stage.value:<10}",
        f"pollen={observation.pollen.availability:.2f}",
    ]
    if observation.range_m is not None:
        parts.append(f"range={observation.range_m * 100:.0f}cm")
    if observation.incidence_deg is not None:
        parts.append(f"inc={observation.incidence_deg:.0f}deg")
    return "  ".join(parts)


def _record(name: str, result) -> dict:
    return {
        "frame": name,
        "width": result.width,
        "height": result.height,
        "latency_ms": {k: round(v, 2) for k, v in result.latency_ms.items()},
        "flowers": [
            {
                "track_id": o.track_id,
                "box": [round(v, 1) for v in o.box.as_array().tolist()],
                "sex": o.sex.sex.value,
                "p_female": round(o.sex.p_female, 4),
                "sex_confidence": round(o.sex.confidence, 4),
                "sex_cues": {k: round(v, 4) for k, v in o.sex.breakdown.cues.items()},
                "stage": o.anthesis.stage.value,
                "pollen": round(o.pollen.availability, 4),
                "range_m": round(o.range_m, 4) if o.range_m is not None else None,
                "incidence_deg": (round(o.incidence_deg, 2)
                                  if o.incidence_deg is not None else None),
                "rover_position_m": (
                    [round(float(v), 4) for v in o.rover_pose.position]
                    if o.rover_pose is not None else None
                ),
            }
            for o in result.flowers
        ],
        "rejected": [
            {"box": [round(v, 1) for v in o.box.as_array().tolist()],
             "reasons": [r.value for r in o.quality.reasons]}
            for o in result.rejected
        ],
    }


def _synthetic_scene(height: int, width: int) -> np.ndarray:
    """A crude foliage-and-blossom scene for benchmarking and self-test.

    This exercises the full code path at a realistic image size. It is *not* a
    substitute for real imagery when judging accuracy - see docs/EVALUATION.md.
    """
    import cv2

    rng = np.random.default_rng(0)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (45, 85, 40)
    noise = rng.normal(0, 12, (height, width, 3))
    frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    for _ in range(24):  # leaves
        centre = (int(rng.uniform(0, width)), int(rng.uniform(0, height)))
        axes = (int(rng.uniform(30, 90)), int(rng.uniform(20, 60)))
        cv2.ellipse(frame, centre, axes, float(rng.uniform(0, 180)), 0, 360,
                    (int(rng.uniform(30, 60)), int(rng.uniform(80, 130)),
                     int(rng.uniform(30, 60))), -1)

    for cx, cy, radius in ((int(width * 0.32), int(height * 0.55), int(height * 0.13)),
                           (int(width * 0.70), int(height * 0.45), int(height * 0.11))):
        cv2.circle(frame, (cx, cy), radius, (40, 200, 245), -1)
        cv2.circle(frame, (cx, cy), int(radius * 0.35), (30, 150, 230), -1)
    cv2.ellipse(frame, (int(width * 0.70), int(height * 0.45) + int(height * 0.15)),
                (int(height * 0.06), int(height * 0.08)), 0, 0, 360, (50, 150, 60), -1)
    return frame


# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pollivision",
        description="Perception and adaptive control for an electrostatic pollination rover.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download model weights into the cache")
    p.add_argument("--model", nargs="*", default=None)
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("detect", help="run perception over images or video")
    _add_common(p)
    p.add_argument("source", help="image, directory, video, camera index or stream URL")
    p.add_argument("--output", default=None, help="directory for annotated frames")
    p.add_argument("--json", default=None, help="write per-frame results to this JSON file")
    p.add_argument("--max-frames", type=int, default=None)
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("run", help="run the full closed-loop mission")
    _add_common(p)
    p.add_argument("source", help="image, directory, video, camera index or stream URL")
    p.add_argument("--json", default=None, help="write the event log here")
    p.add_argument("--video-out", default=None, help="write an annotated mp4 here")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--require-pose-feedback", action="store_true",
                   help="wait for real probe-pose feedback instead of assuming it")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("stream", help="view an annotated live ESP32-CAM feed")
    _add_common(p)
    p.add_argument("--url", default=None, help="MJPEG stream URL")
    p.add_argument("--show", action="store_true", help="open a display window")
    p.add_argument("--save-dir", default=None, help="save annotated frames here")
    p.add_argument("--no-model", action="store_true", help="passthrough, no inference")
    p.set_defaults(func=cmd_stream)

    p = sub.add_parser("calibrate", help="solve camera intrinsics from checkerboard images")
    p.add_argument("images", help="directory of checkerboard captures")
    p.add_argument("--pattern", default="9x6", help="inner corners, e.g. 9x6")
    p.add_argument("--square-size-mm", type=float, default=25.0)
    p.add_argument("--output", default="camera_intrinsics.json")
    p.add_argument("--name", default="esp32cam-ov2640")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("bench", help="measure per-stage latency on this machine")
    _add_common(p)
    p.add_argument("--iterations", type=int, default=5)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("selftest", help="verify the install end to end")
    _add_common(p)
    p.set_defaults(func=cmd_selftest)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
