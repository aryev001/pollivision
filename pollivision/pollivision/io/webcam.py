"""Laptop / USB webcam capture.

The ESP32-CAM client in :mod:`pollivision.io.esp32` solves this problem for a
network camera; this module solves it for the camera already attached to the
machine running the stack, which is what you have when you clone the repo and
want to watch the perception loop work before there is a rover.

The constraints are different from the network case but the conclusion is the
same, so the two classes deliberately expose the same surface:

* **A webcam has an internal queue too.** ``cv2.VideoCapture.read()`` returns the
  *oldest* undelivered frame, not the newest. A pipeline that takes 400 ms per
  frame against a 30 FPS camera therefore falls further behind every frame, and
  after a minute it is annotating a scene from ten seconds ago. The grab loop
  runs on its own thread and keeps only the newest frame, so latency is bounded
  by inference time instead of accumulating without limit.
* **Opening a camera is slow and platform-specific.** On Windows the default
  MSMF backend can take several seconds to open a device that DirectShow opens
  instantly; on macOS the AVFoundation backend must be used for the OS to raise
  its camera-permission prompt. Backend selection is therefore explicit, with an
  ordered per-platform fallback rather than a single blind ``VideoCapture(0)``.
* **Auto-exposure needs a moment.** The first handful of frames off a laptop
  camera are dark or white while AE and AWB settle, and feeding those to the
  quality gate produces a burst of spurious "bad exposure" rejections at every
  start-up. They are discarded during warm-up instead.
"""

from __future__ import annotations

import platform
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterator, Optional, Union

import cv2
import numpy as np

from ..logging_utils import get_logger

LOGGER = get_logger(__name__)

Device = Union[int, str]


# --------------------------------------------------------------------------- #
# Backend selection
# --------------------------------------------------------------------------- #

def _backend_candidates() -> list[tuple[str, int]]:
    """Ordered ``(name, cv2 flag)`` capture backends to try on this platform.

    ``CAP_ANY`` is always last so an unusual build still gets a chance.
    """
    system = platform.system()
    names: list[str]
    if system == "Windows":
        # DirectShow first: MSMF opens some integrated cameras an order of
        # magnitude slower and reports resolutions it will not actually deliver.
        names = ["CAP_DSHOW", "CAP_MSMF"]
    elif system == "Darwin":
        # AVFoundation is the backend that triggers the macOS privacy prompt.
        names = ["CAP_AVFOUNDATION"]
    else:
        names = ["CAP_V4L2"]

    candidates = []
    for name in names:
        flag = getattr(cv2, name, None)
        if flag is not None:
            candidates.append((name, int(flag)))
    candidates.append(("CAP_ANY", int(cv2.CAP_ANY)))
    return candidates


def permission_hint() -> str:
    """Platform-specific advice for the most common cause of a failed open."""
    system = platform.system()
    if system == "Darwin":
        return ("On macOS the camera prompt is raised for the *terminal or "
                "editor* that launched this process. Check System Settings > "
                "Privacy & Security > Camera and enable your terminal / VS Code, "
                "then restart it.")
    if system == "Windows":
        return ("On Windows check Settings > Privacy & security > Camera, and "
                "close any other application holding the camera (Teams, Zoom, "
                "the Camera app) - Windows will not share it.")
    return ("On Linux check that /dev/video* exists and that your user is in "
            "the 'video' group (`ls -l /dev/video*`, then "
            "`sudo usermod -aG video $USER` and log out and back in). Under WSL "
            "the host webcam is not passed through by default.")


def gui_available() -> bool:
    """Whether this OpenCV build can open a window.

    ``opencv-python-headless`` - which the project used to depend on, because
    the rover is headless - is built without any HighGUI backend, and
    ``cv2.imshow`` raises on it. Probing the build string once up front turns
    that into an actionable message instead of a stack trace thrown after the
    models have finished loading.
    """
    try:
        info = cv2.getBuildInformation()
    except Exception:  # noqa: BLE001
        return False

    lines = info.splitlines()
    for position, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("GUI:"):
            continue
        value = stripped[len("GUI:"):].strip()
        if value:
            # OpenCV >= 4.5.5 names the chosen backend here, or "NONE".
            return value.upper() != "NONE"
        # Older builds list each candidate backend on the following lines.
        for sub in lines[position + 1:position + 8]:
            substripped = sub.strip()
            if not substripped or not sub.startswith(" "):
                break
            name, _, state = substripped.partition(":")
            if name.strip().rstrip("+") in {"QT", "GTK", "GTK2", "GTK3",
                                            "Cocoa", "Win32 UI", "WinRT"}:
                if state.strip().upper().startswith("YES"):
                    return True
        return False
    return True  # unrecognised build string; let the caller find out by trying


def gui_hint() -> str:
    return ("This OpenCV build has no display backend, which means "
            "opencv-python-headless is installed. Run:\n"
            "    pip uninstall -y opencv-python-headless\n"
            "    pip install opencv-python\n"
            "or pass --no-window to run without a preview window.")


# --------------------------------------------------------------------------- #
# Enumeration
# --------------------------------------------------------------------------- #

@dataclass
class CameraInfo:
    """A camera index that actually delivered a frame when probed."""

    index: int
    width: int
    height: int
    fps: float
    backend: str

    def describe(self) -> str:
        fps = f"{self.fps:.0f} FPS" if self.fps > 0 else "unknown FPS"
        return (f"camera {self.index}: {self.width}x{self.height} @ {fps} "
                f"[{self.backend}]")


class _quiet_opencv:
    """Silence OpenCV's C++ logger for the duration of a block.

    Probing camera indices means deliberately opening devices that are not
    there, and OpenCV prints several lines of VIDEOIO warnings for each miss.
    Left alone, ``--list-cameras`` buries its own answer under a page of noise
    about failures that were expected. The logging entry point moved between
    OpenCV 4 and 5, so both spellings are tried.
    """

    def __init__(self) -> None:
        self._restore = None

    def __enter__(self) -> "_quiet_opencv":
        for getter, setter, silent in (
            (getattr(cv2, "getLogLevel", None), getattr(cv2, "setLogLevel", None), 0),
            *([(cv2.utils.logging.getLogLevel, cv2.utils.logging.setLogLevel,
                cv2.utils.logging.LOG_LEVEL_SILENT)]
              if hasattr(getattr(cv2, "utils", None), "logging") else []),
        ):
            if getter is None or setter is None:
                continue
            try:
                previous = getter()
                setter(silent)
                self._restore = (setter, previous)
                break
            except Exception:  # noqa: BLE001
                continue
        return self

    def __exit__(self, *exc) -> None:
        if self._restore is not None:
            setter, previous = self._restore
            try:
                setter(previous)
            except Exception:  # noqa: BLE001
                pass


def list_cameras(max_index: int = 6) -> list[CameraInfo]:
    """Probe camera indices and report the ones that yield a frame.

    Probing is done by actually reading a frame rather than by trusting
    ``isOpened()``: several drivers happily open a device node that then never
    produces an image, and reporting such a device as available only moves the
    failure to a more confusing place.
    """
    found: list[CameraInfo] = []
    with _quiet_opencv():
        for index in range(max(1, int(max_index))):
            for name, flag in _backend_candidates():
                capture = None
                try:
                    capture = cv2.VideoCapture(index, flag)
                    if not capture.isOpened():
                        continue
                    ok, image = capture.read()
                    if not ok or image is None:
                        continue
                    found.append(CameraInfo(
                        index=index,
                        width=int(image.shape[1]),
                        height=int(image.shape[0]),
                        fps=float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
                        backend=name,
                    ))
                    break  # first backend that works for this index wins
                except Exception:  # noqa: BLE001 - a bad index must not stop the scan
                    continue
                finally:
                    if capture is not None:
                        capture.release()
    return found


# --------------------------------------------------------------------------- #
# Capture
# --------------------------------------------------------------------------- #

@dataclass
class WebcamFrame:
    """One captured frame plus the metadata the pipeline uses to judge it."""

    image: np.ndarray
    timestamp: float
    index: int
    brightness: float = 0.0
    age_s: float = 0.0


class WebcamStream:
    """Threaded webcam reader that always yields the freshest available frame.

    Mirrors :class:`pollivision.io.esp32.Esp32CamStream` so that anything
    driving one can drive the other unchanged.
    """

    def __init__(
        self,
        device: Device = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
        warmup_frames: int = 5,
        fourcc: Optional[str] = "MJPG",
        backend: Optional[str] = None,
        open_timeout_s: float = 10.0,
    ) -> None:
        self.device = device
        self.requested_width = int(width) if width else None
        self.requested_height = int(height) if height else None
        self.requested_fps = float(fps) if fps else None
        self.warmup_frames = max(0, int(warmup_frames))
        # MJPG matters on USB 2.0: at 1280x720 the raw YUYV stream does not fit
        # in the bus budget and the driver silently drops to ~10 FPS.
        self.fourcc = fourcc
        self.backend = backend
        self.open_timeout_s = float(open_timeout_s)

        self._capture: Optional[cv2.VideoCapture] = None
        self._queue: deque[WebcamFrame] = deque(maxlen=1)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._index = 0
        self._dropped = 0
        self._error: Optional[str] = None
        self._backend_name = ""
        self._size: tuple[int, int] = (0, 0)
        self._capture_times: deque[float] = deque(maxlen=60)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def open(self) -> cv2.VideoCapture:
        """Open the device, trying each platform backend in turn."""
        device = self.device
        if isinstance(device, str) and device.isdigit():
            device = int(device)

        if self.backend:
            flag = getattr(cv2, self.backend, None)
            if flag is None:
                raise ValueError(f"Unknown OpenCV capture backend: {self.backend}")
            candidates = [(self.backend, int(flag))]
        else:
            candidates = _backend_candidates()

        errors: list[str] = []
        for name, flag in candidates:
            # Falling back from one backend to the next is routine (on Windows
            # MSMF often fails where DirectShow succeeds), so OpenCV's warnings
            # about the attempt are suppressed; what was tried is reported in
            # the error below if every backend fails.
            with _quiet_opencv():
                try:
                    capture = (cv2.VideoCapture(device, flag)
                               if isinstance(device, int)
                               else cv2.VideoCapture(str(device), flag))
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{name}: {exc}")
                    continue
                if not capture.isOpened():
                    capture.release()
                    errors.append(f"{name}: could not open device")
                    continue

                self._configure(capture)
                ok, image = capture.read()
                if not ok or image is None:
                    capture.release()
                    errors.append(f"{name}: opened but delivered no frames")
                    continue

            self._backend_name = name
            self._size = (int(image.shape[1]), int(image.shape[0]))
            LOGGER.info("Opened camera %s via %s at %dx%d",
                        device, name, self._size[0], self._size[1])
            if (self.requested_width and self.requested_height
                    and self._size != (self.requested_width, self.requested_height)):
                LOGGER.info("Camera delivered %dx%d rather than the requested "
                            "%dx%d; using what it gave.",
                            self._size[0], self._size[1],
                            self.requested_width, self.requested_height)
            return capture

        raise RuntimeError(
            f"Could not open camera {self.device!r}.\n"
            + "\n".join(f"  tried {e}" for e in errors)
            + "\n" + permission_hint()
            + "\nRun `pollivision webcam --list-cameras` to see what is available."
        )

    def _configure(self, capture: cv2.VideoCapture) -> None:
        if self.fourcc:
            try:
                capture.set(cv2.CAP_PROP_FOURCC,
                            cv2.VideoWriter_fourcc(*self.fourcc))
            except Exception:  # noqa: BLE001 - not every backend accepts it
                pass
        if self.requested_width:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        if self.requested_height:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
        if self.requested_fps:
            capture.set(cv2.CAP_PROP_FPS, self.requested_fps)
        # Ask the driver for the shallowest queue it will give us. Most ignore
        # it, which is exactly why the grab thread below exists as well.
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # noqa: BLE001
            pass

    def start(self) -> "WebcamStream":
        if self._thread is not None:
            return self
        self._capture = self.open()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="webcam", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=self.open_timeout_s):
            LOGGER.warning("Camera %s produced no frame within %.0fs",
                           self.device, self.open_timeout_s)
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "WebcamStream":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #

    @property
    def connected(self) -> bool:
        return self._capture is not None and not self._stop.is_set()

    @property
    def dropped_frames(self) -> int:
        """Frames grabbed but never consumed, i.e. discarded as stale.

        A large number is normal and desirable here: it is the count of frames
        the pipeline was too slow to look at, which were skipped rather than
        queued. It is a measure of how far behind real time the loop would
        otherwise have fallen.
        """
        return self._dropped

    @property
    def last_error(self) -> Optional[str]:
        return self._error

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    @property
    def capture_fps(self) -> float:
        """Measured grab rate, which is what the camera is really delivering."""
        with self._lock:
            times = list(self._capture_times)
        if len(times) < 2:
            return 0.0
        span = times[-1] - times[0]
        return (len(times) - 1) / span if span > 0 else 0.0

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #

    def read(self, timeout_s: float = 5.0) -> Optional[WebcamFrame]:
        """Return the newest frame, waiting up to ``timeout_s`` for one."""
        deadline = time.time() + timeout_s
        while True:
            with self._lock:
                if self._queue:
                    frame = self._queue[-1]
                    self._queue.clear()
                    frame.age_s = time.time() - frame.timestamp
                    return frame
            if self._stop.is_set() or time.time() >= deadline:
                return None
            time.sleep(0.002)

    def frames(self, timeout_s: float = 5.0) -> Iterator[WebcamFrame]:
        """Iterate over the freshest frames until the stream is stopped."""
        while not self._stop.is_set():
            frame = self.read(timeout_s)
            if frame is None:
                if self._stop.is_set():
                    return
                LOGGER.warning("No frame within %.1fs from camera %s",
                               timeout_s, self.device)
                continue
            yield frame

    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        capture = self._capture
        if capture is None:
            return
        warmed = 0
        failures = 0
        while not self._stop.is_set():
            ok, image = capture.read()
            if not ok or image is None:
                failures += 1
                self._error = "camera read failed"
                if failures > 60:
                    LOGGER.error("Camera %s stopped delivering frames", self.device)
                    self._stop.set()
                    return
                time.sleep(0.01)
                continue
            failures = 0
            self._error = None

            if warmed < self.warmup_frames:
                warmed += 1
                continue

            self._index += 1
            now = time.time()
            frame = WebcamFrame(
                image=image,
                timestamp=now,
                index=self._index,
                brightness=float(image.mean()) / 255.0,
            )
            with self._lock:
                if self._queue:
                    self._dropped += 1
                self._queue.append(frame)
                self._capture_times.append(now)
            self._ready.set()


def snapshot(device: Device = 0, warmup_frames: int = 8,
             width: Optional[int] = None,
             height: Optional[int] = None) -> Optional[np.ndarray]:
    """Grab a single settled still from a webcam.

    Used by ``pollivision calibrate --camera`` for checkerboard capture, where
    the warm-up matters more than the frame rate: a still taken before AE
    settles calibrates against a blurred, badly exposed board.
    """
    stream = WebcamStream(device, width=width, height=height,
                          warmup_frames=warmup_frames)
    try:
        stream.start()
        frame = stream.read(timeout_s=5.0)
        return frame.image if frame is not None else None
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Snapshot from camera %s failed: %s", device, exc)
        return None
    finally:
        stream.stop()
