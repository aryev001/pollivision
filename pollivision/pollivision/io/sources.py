"""Frame sources: files, directories, webcams, video and the ESP32-CAM.

One iterator interface over every input the system supports, so the pipeline,
the CLI and the tests never branch on where frames come from.

Live sources (a laptop or USB webcam, the ESP32-CAM) and recorded ones differ in
one way that matters: a recorded file must yield *every* frame in order, while a
live camera must yield the *newest* frame and discard whatever piled up while
the consumer was busy. Both live sources therefore run their reader on a thread
behind a depth-one queue, and only the file sources iterate exhaustively.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from ..logging_utils import get_logger

LOGGER = get_logger(__name__)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


@dataclass
class SourceFrame:
    """A frame plus where it came from."""

    image: np.ndarray
    index: int
    timestamp: float
    name: str = ""
    age_s: float = 0.0


class FrameSource:
    """Base iterator over frames."""

    def __iter__(self) -> Iterator[SourceFrame]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class ImageSource(FrameSource):
    """A single image or a directory of images, in sorted order."""

    def __init__(self, path: str | Path, loop: bool = False) -> None:
        path = Path(path)
        if path.is_dir():
            self.paths = sorted(
                p for p in path.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES
            )
            if not self.paths:
                raise FileNotFoundError(f"No images found in {path}")
        else:
            if not path.exists():
                raise FileNotFoundError(f"No such image: {path}")
            self.paths = [path]
        self.loop = loop

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self) -> Iterator[SourceFrame]:
        index = 0
        while True:
            for path in self.paths:
                image = cv2.imread(str(path))
                if image is None:
                    LOGGER.warning("Could not decode %s; skipping", path)
                    continue
                index += 1
                yield SourceFrame(image=image, index=index,
                                  timestamp=time.time(), name=path.name)
            if not self.loop:
                return


class VideoSource(FrameSource):
    """A video file or a camera index / device path."""

    def __init__(self, target: str | int, loop: bool = False,
                 max_frames: Optional[int] = None) -> None:
        self.capture = cv2.VideoCapture(target)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video source: {target}")
        self.target = target
        self.loop = loop
        self.max_frames = max_frames

    def __iter__(self) -> Iterator[SourceFrame]:
        index = 0
        while True:
            ok, image = self.capture.read()
            if not ok:
                if self.loop and isinstance(self.target, (str, Path)):
                    self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                return
            index += 1
            yield SourceFrame(image=image, index=index,
                              timestamp=time.time(), name=f"frame{index:06d}")
            if self.max_frames and index >= self.max_frames:
                return

    def close(self) -> None:
        self.capture.release()


class WebcamSource(FrameSource):
    """A laptop or USB webcam, read freshest-frame-first on its own thread.

    ``VideoSource`` can open a camera index too, but it reads the capture queue
    in order, so a consumer slower than the camera falls progressively further
    behind real time. Use this for anything live.
    """

    def __init__(self, device: str | int = 0, width: Optional[int] = None,
                 height: Optional[int] = None, fps: Optional[float] = None,
                 warmup_frames: int = 5, mirror: bool = False,
                 max_frames: Optional[int] = None) -> None:
        from .webcam import WebcamStream

        self.stream = WebcamStream(device=device, width=width, height=height,
                                   fps=fps, warmup_frames=warmup_frames)
        self.stream.start()
        self.mirror = bool(mirror)
        self.max_frames = max_frames

    def __iter__(self) -> Iterator[SourceFrame]:
        count = 0
        for frame in self.stream.frames():
            image = cv2.flip(frame.image, 1) if self.mirror else frame.image
            count += 1
            yield SourceFrame(image=image, index=frame.index,
                              timestamp=frame.timestamp,
                              name=f"cam-{frame.index:06d}", age_s=frame.age_s)
            if self.max_frames and count >= self.max_frames:
                return

    def close(self) -> None:
        self.stream.stop()


class Esp32Source(FrameSource):
    """Wraps the threaded ESP32-CAM MJPEG client in the common interface."""

    def __init__(self, url: str, reconnect_delay_s: float = 2.0,
                 max_queue: int = 2, timeout_s: float = 5.0) -> None:
        from .esp32 import Esp32CamStream

        self.stream = Esp32CamStream(url, reconnect_delay_s, max_queue, timeout_s)
        self.stream.start()

    def __iter__(self) -> Iterator[SourceFrame]:
        for frame in self.stream.frames():
            yield SourceFrame(image=frame.image, index=frame.index,
                              timestamp=frame.timestamp,
                              name=f"esp32-{frame.index:06d}", age_s=frame.age_s)

    def close(self) -> None:
        self.stream.stop()


def open_source(spec: str | int, cfg=None, loop: bool = False,
                max_frames: Optional[int] = None) -> FrameSource:
    """Build the right source for a spec.

    Accepts a camera index (``0``, ``"1"``, ``"webcam"``, ``"webcam:1"``), an
    ``http(s)://`` MJPEG stream URL, a video file, an image, or a directory of
    images. ``"config"`` reads the ``source`` section of ``cfg`` instead.
    """
    camera = cfg.section("camera") if cfg is not None else None
    source_cfg = cfg.section("source") if cfg is not None else None

    def _webcam(device: str | int) -> FrameSource:
        section = source_cfg
        width = height = fps = None
        warmup, mirror = 5, False
        if section is not None:
            width = section.get("width") or (camera.get("width") if camera else None)
            height = section.get("height") or (camera.get("height") if camera else None)
            fps = section.get("fps")
            warmup = int(section.get("warmup_frames", 5))
            mirror = bool(section.get("mirror", False))
        return WebcamSource(device=device, width=width, height=height, fps=fps,
                            warmup_frames=warmup, mirror=mirror,
                            max_frames=max_frames)

    if cfg is not None and isinstance(spec, str) and spec == "config":
        section = cfg.section("source")
        kind = str(section.get("kind", "esp32"))
        if kind == "esp32":
            return Esp32Source(
                url=str(section.get("url", "http://192.168.4.1:81/stream")),
                reconnect_delay_s=float(section.get("reconnect_delay_s", 2.0)),
                max_queue=int(section.get("max_queue", 2)),
            )
        if kind in {"webcam", "camera", "usb"}:
            return _webcam(section.get("device", 0))
        spec = str(section.get("url", 0))

    if isinstance(spec, int) or (isinstance(spec, str) and spec.isdigit()):
        # A bare index is a live camera, so it gets the freshest-frame reader
        # rather than the sequential one.
        return _webcam(int(spec))

    text = str(spec)
    lowered = text.lower()
    if lowered in {"webcam", "camera"}:
        return _webcam(0)
    if lowered.startswith(("webcam:", "camera:")):
        _, _, device = text.partition(":")
        return _webcam(int(device) if device.strip().isdigit() else device)
    if lowered.startswith("/dev/video"):
        return _webcam(text)

    if text.startswith(("http://", "https://")):
        return Esp32Source(url=text)

    path = Path(text)
    if path.is_dir() or path.suffix.lower() in _IMAGE_SUFFIXES:
        return ImageSource(path, loop=loop)
    if path.suffix.lower() in _VIDEO_SUFFIXES:
        return VideoSource(text, loop=loop, max_frames=max_frames)

    # Unknown suffix: let OpenCV decide, it handles more container formats than
    # any extension list would.
    return VideoSource(text, loop=loop, max_frames=max_frames)
