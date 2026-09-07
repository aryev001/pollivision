"""Live camera session: capture, inference and display, decoupled.

Running the pipeline inside the display loop is the obvious way to do this and
it is wrong. The stack costs 350-850 ms per frame on a laptop CPU once the
vision-language cues fire, so a single-threaded loop shows the operator a 1-3
FPS slideshow, and - worse - every frame it displays was captured before the
last one was analysed, so the preview lags reality by however long the backlog
has grown.

This module separates the three rates that are genuinely independent:

* **Capture** runs in :class:`~pollivision.io.webcam.WebcamStream` at whatever
  the camera delivers, keeping only the newest frame.
* **Inference** runs on a worker thread over the newest frame available *at the
  moment it becomes free*, skipping everything captured while it was busy.
  Skipping is the correct behaviour: an intermediate frame that can no longer be
  acted on has no value to a control loop.
* **Display** runs on the main thread at camera rate, compositing the most
  recent perception result onto the *current* frame.

The consequence worth being explicit about: the overlay is up to one inference
period older than the video under it, so boxes lag a fast-moving flower. That is
a deliberate trade - a smooth, current preview with a slightly stale overlay
beats a stuttering one where both are stale - and the HUD reports the overlay's
actual age so the operator can see when it matters. ``--sync`` turns the trade
off and shows each analysed frame with its own overlay instead.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from ..io import visualize
from ..io.webcam import WebcamStream, gui_available, gui_hint
from ..logging_utils import get_logger
from ..types import EnvironmentReading, FrameResult

LOGGER = get_logger(__name__)

HELP_LINES = [
    "q / Esc   quit",
    "space     pause or resume inference",
    "s         save an annotated snapshot",
    "m         toggle corolla masks",
    "r         toggle rejected detections",
    "o         toggle the whole overlay",
    "f         toggle mirroring",
    "b         toggle the sex-cue breakdown",
    "h         hide this help",
]


@dataclass
class _Analysis:
    """One completed pipeline run, and the frame it ran on."""

    result: FrameResult
    frame: np.ndarray
    finished_at: float
    latency_ms: float
    state: str = ""
    message: str = ""
    command: object = None


@dataclass
class LiveStats:
    """Rolling rates, for the HUD and for the end-of-session summary."""

    capture_fps: float = 0.0
    display_fps: float = 0.0
    inference_fps: float = 0.0
    inference_ms: float = 0.0
    overlay_age_ms: float = 0.0
    frames_displayed: int = 0
    frames_analysed: int = 0
    frames_dropped: int = 0
    flowers: int = 0

    @property
    def frames_skipped(self) -> int:
        """Captured frames the pipeline never looked at.

        Not a fault: these are the frames that arrived while inference was
        busy. A control loop gains nothing from analysing a frame it can no
        longer act on, so they are dropped rather than queued.
        """
        return max(0, self.frames_displayed - self.frames_analysed)

    def summary(self) -> list[str]:
        return [
            f"frames displayed   {self.frames_displayed}",
            f"frames analysed    {self.frames_analysed}",
            f"frames skipped     {self.frames_skipped} (arrived mid-inference)",
            f"frames dropped     {self.frames_dropped} (never reached display)",
            f"display rate       {self.display_fps:.1f} FPS",
            f"inference rate     {self.inference_fps:.2f} FPS "
            f"({self.inference_ms:.0f} ms/frame)",
        ]


class _RateMeter:
    """Rate over a sliding window of event times."""

    def __init__(self, window: int = 45) -> None:
        self._times: deque[float] = deque(maxlen=window)

    def tick(self, now: Optional[float] = None) -> None:
        self._times.append(now if now is not None else time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0


class LiveSession:
    """Run the perception stack against a live camera with a preview window."""

    def __init__(
        self,
        cfg,
        pipeline,
        *,
        device: int | str = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
        mirror: bool = False,
        window: bool = True,
        window_name: str = "PolliVision - live",
        mission=None,
        telemetry=None,
        record: Optional[str] = None,
        snapshot_dir: Optional[str] = None,
        max_frames: Optional[int] = None,
        sync: bool = False,
        show_masks: bool = True,
        show_rejected: bool = True,
        on_result: Optional[Callable[[FrameResult], None]] = None,
        stream: Optional[WebcamStream] = None,
    ) -> None:
        self.cfg = cfg
        self.pipeline = pipeline
        self.mission = mission
        self.telemetry = telemetry
        self.mirror = bool(mirror)
        self.window = bool(window)
        self.window_name = window_name
        self.record_path = record
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        self.max_frames = max_frames
        self.sync = bool(sync)
        self.show_masks = bool(show_masks)
        self.show_rejected = bool(show_rejected)
        self.on_result = on_result

        self.stream = stream or WebcamStream(
            device=device,
            width=width,
            height=height,
            fps=fps,
            warmup_frames=int(cfg.get("source.warmup_frames", 5)),
        )

        self.stats = LiveStats()
        self._display_meter = _RateMeter()
        self._inference_meter = _RateMeter()

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_index = 0
        self._frame_lock = threading.Lock()
        self._analysis: Optional[_Analysis] = None
        self._analysis_lock = threading.Lock()
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._new_frame = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._writer: Optional[cv2.VideoWriter] = None
        self._show_help = False
        self._show_breakdown = False
        self._show_overlay = True
        self._snapshots = 0
        self._error: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def run(self) -> int:
        """Open the camera and run until the operator quits. Returns exit code."""
        if self.window and not gui_available():
            LOGGER.warning("No GUI backend in this OpenCV build; "
                           "falling back to --no-window.\n%s", gui_hint())
            self.window = False

        try:
            self.stream.start()
        except RuntimeError as exc:
            print(f"\n{exc}\n")
            return 2

        width, height = self.stream.size
        print(f"Camera open: {width}x{height} via {self.stream.backend_name}. "
              f"{'Press q in the window to stop.' if self.window else 'Ctrl-C to stop.'}")
        if self.window:
            print("Keys: q quit  space pause  s snapshot  m masks  "
                  "r rejected  o overlay  f mirror  h help")

        self._worker = threading.Thread(target=self._inference_loop,
                                        name="pollivision-inference", daemon=True)
        self._worker.start()

        try:
            self._display_loop()
        except KeyboardInterrupt:
            print("\nstopped")
        finally:
            self.close()

        self._print_summary()
        return 1 if self._error else 0

    def close(self) -> None:
        self._stop.set()
        self._new_frame.set()
        if self._worker is not None:
            self._worker.join(timeout=5.0)
            self._worker = None
        self.stream.stop()
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self.window:
            try:
                cv2.destroyWindow(self.window_name)
                cv2.waitKey(1)
            except Exception:  # noqa: BLE001 - closing a window must never fail a run
                pass

    # ------------------------------------------------------------------ #
    # Threads
    # ------------------------------------------------------------------ #

    def _display_loop(self) -> None:
        if self.window:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            width, height = self.stream.size
            if width and height:
                cv2.resizeWindow(self.window_name, width, height)

        while not self._stop.is_set():
            frame = self.stream.read(timeout_s=2.0)
            if frame is None:
                if not self.stream.connected:
                    break
                LOGGER.warning("Waiting for frames from the camera...")
                continue

            image = frame.image
            if self.mirror:
                image = cv2.flip(image, 1)

            with self._frame_lock:
                self._latest_frame = image
                self._latest_frame_index = frame.index
            self._new_frame.set()

            canvas = self._compose(image)

            self._display_meter.tick()
            self.stats.frames_displayed += 1
            self.stats.display_fps = self._display_meter.fps
            self.stats.capture_fps = self.stream.capture_fps
            self.stats.frames_dropped = self.stream.dropped_frames

            self._record(canvas)

            if self.window:
                cv2.imshow(self.window_name, canvas)
                if not self._handle_keys():
                    break
                # The operator clicking the title-bar close button should end
                # the session, not leave a headless loop spinning.
                try:
                    if cv2.getWindowProperty(self.window_name,
                                             cv2.WND_PROP_VISIBLE) < 1:
                        break
                except Exception:  # noqa: BLE001
                    pass
            elif self.stats.frames_displayed % 15 == 0:
                self._print_status()

            if self.max_frames and self.stats.frames_displayed >= self.max_frames:
                break

        self._stop.set()

    def _inference_loop(self) -> None:
        analysed_index = -1
        while not self._stop.is_set():
            if self._paused.is_set():
                time.sleep(0.05)
                continue

            self._new_frame.wait(timeout=0.5)
            self._new_frame.clear()
            if self._stop.is_set():
                return

            with self._frame_lock:
                frame = self._latest_frame
                index = self._latest_frame_index
            if frame is None or index == analysed_index:
                continue
            analysed_index = index
            # Some capture backends hand back a view onto a buffer they
            # reuse for the next frame, so the pipeline gets its own copy
            # rather than an array that may change under it mid-inference.
            frame = frame.copy()

            started = time.perf_counter()
            try:
                result = self.pipeline.process(frame)
            except Exception as exc:  # noqa: BLE001 - one bad frame must not end the run
                LOGGER.exception("Pipeline failed on frame %d: %s", index, exc)
                self._error = str(exc)
                time.sleep(0.2)
                continue
            latency_ms = (time.perf_counter() - started) * 1000.0

            analysis = _Analysis(result=result, frame=frame,
                                 finished_at=time.time(), latency_ms=latency_ms)
            if self.mission is not None:
                self._step_mission(analysis)

            with self._analysis_lock:
                self._analysis = analysis

            self._inference_meter.tick()
            self.stats.frames_analysed += 1
            self.stats.inference_fps = self._inference_meter.fps
            self.stats.inference_ms = latency_ms
            self.stats.flowers = len(result.flowers)

            if self.on_result is not None:
                try:
                    self.on_result(result)
                except Exception:  # noqa: BLE001 - a callback must not kill the loop
                    LOGGER.exception("on_result callback failed")

    def _step_mission(self, analysis: _Analysis) -> None:
        environment = (self.telemetry.read() if self.telemetry is not None
                       else EnvironmentReading())
        try:
            step = self.mission.step(analysis.result, environment,
                                     frame=analysis.frame, probe_at_pose=True)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Mission step failed: %s", exc)
            return
        analysis.state = step.state.value
        analysis.message = step.message
        analysis.command = step.command

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _compose(self, image: np.ndarray) -> np.ndarray:
        with self._analysis_lock:
            analysis = self._analysis

        if analysis is None:
            canvas = image.copy()
            visualize.draw_centered_notice(
                canvas, ["Loading models and analysing the first frame...",
                         "The preview is live; boxes appear shortly."])
            self.stats.overlay_age_ms = 0.0
        elif not self._show_overlay:
            canvas = image.copy()
            self.stats.overlay_age_ms = 0.0
        else:
            # In sync mode the overlay and the pixels under it are the same
            # frame, which is what you want for screenshots and for judging
            # detection quality; live mode favours a current, smooth preview.
            base = analysis.frame if self.sync else image
            if base.shape[:2] != image.shape[:2]:
                base = image
            canvas = visualize.draw_frame(
                base, analysis.result,
                command=analysis.command,
                state=analysis.state,
                message=analysis.message,
                show_rejected=self.show_rejected,
                show_masks=self.show_masks,
            )
            self.stats.overlay_age_ms = (time.time() - analysis.finished_at) * 1000.0
            if self._show_breakdown and analysis.result.flowers:
                visualize.draw_breakdown(canvas, analysis.result.flowers[0])

        visualize.draw_hud(canvas, self._hud_lines())
        if self._paused.is_set():
            visualize.draw_centered_notice(canvas, ["PAUSED - press space to resume"])
        if self._show_help:
            visualize.draw_centered_notice(canvas, HELP_LINES, scale=0.5)
        return canvas

    def _hud_lines(self) -> list[str]:
        line = (f"camera {self.stats.capture_fps:4.1f} FPS   "
                f"display {self.stats.display_fps:4.1f} FPS   "
                f"inference {self.stats.inference_fps:4.2f} FPS "
                f"({self.stats.inference_ms:.0f} ms)   "
                f"overlay +{self.stats.overlay_age_ms:.0f} ms")
        second = (f"analysed {self.stats.frames_analysed}   "
                  f"skipped {self.stats.frames_skipped}   "
                  f"{'sync' if self.sync else 'live'}   "
                  + ("h for help" if self.window and not self._show_help else ""))
        if self._error:
            second += f"   ERROR: {self._error[:60]}"
        return [line, second]

    def _record(self, canvas: np.ndarray) -> None:
        if not self.record_path:
            return
        if self._writer is None:
            height, width = canvas.shape[:2]
            fps = self.stats.display_fps or self.stream.capture_fps or 15.0
            self._writer = cv2.VideoWriter(
                self.record_path, cv2.VideoWriter_fourcc(*"mp4v"),
                float(max(1.0, min(60.0, fps))), (width, height))
            if not self._writer.isOpened():
                LOGGER.error("Could not open %s for writing; recording disabled",
                             self.record_path)
                self.record_path = None
                self._writer = None
                return
        self._writer.write(canvas)

    # ------------------------------------------------------------------ #
    # Interaction
    # ------------------------------------------------------------------ #

    def _handle_keys(self) -> bool:
        """Process one keypress. Returns False when the session should end."""
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return False
        if key == ord(" "):
            if self._paused.is_set():
                self._paused.clear()
            else:
                self._paused.set()
        elif key == ord("m"):
            self.show_masks = not self.show_masks
        elif key == ord("r"):
            self.show_rejected = not self.show_rejected
        elif key == ord("o"):
            self._show_overlay = not self._show_overlay
        elif key == ord("f"):
            self.mirror = not self.mirror
        elif key == ord("b"):
            self._show_breakdown = not self._show_breakdown
        elif key == ord("h"):
            self._show_help = not self._show_help
        elif key == ord("s"):
            self._save_snapshot()
        return True

    def _save_snapshot(self) -> None:
        directory = self.snapshot_dir or Path("snapshots")
        directory.mkdir(parents=True, exist_ok=True)
        with self._frame_lock:
            frame = self._latest_frame
        if frame is None:
            return
        showing_help, self._show_help = self._show_help, False
        try:
            canvas = self._compose(frame)
        finally:
            self._show_help = showing_help
        self._snapshots += 1
        path = directory / f"pollivision_{time.strftime('%Y%m%d_%H%M%S')}_{self._snapshots:03d}.jpg"
        cv2.imwrite(str(path), canvas)
        print(f"\nsaved {path}")

    # ------------------------------------------------------------------ #

    def _print_status(self) -> None:
        with self._analysis_lock:
            analysis = self._analysis
        flowers = len(analysis.result.flowers) if analysis else 0
        males = len(analysis.result.males) if analysis else 0
        females = len(analysis.result.females) if analysis else 0
        print(f"\r[{self.stats.frames_displayed:6d}] "
              f"{flowers} flowers (M{males}/F{females})  "
              f"cam {self.stats.capture_fps:4.1f} FPS  "
              f"inference {self.stats.inference_fps:4.2f} FPS "
              f"({self.stats.inference_ms:4.0f} ms)  "
              f"skipped {self.stats.frames_skipped}   ",
              end="", flush=True)

    def _print_summary(self) -> None:
        print("\n\n--- live session ---")
        for line in self.stats.summary():
            print(f"  {line}")
        if self.record_path:
            print(f"  recorded to       {self.record_path}")
        if self._snapshots:
            print(f"  snapshots saved   {self._snapshots}")
        if self.mission is not None:
            print("  mission ledger:")
            for key, value in self.mission.ledger.stats().items():
                print(f"    {key:<16} {value}")
        if self._error:
            print(f"  last error        {self._error}")
