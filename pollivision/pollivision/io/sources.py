"""Frame sources: files, directories, webcams, video and the ESP32-CAM.

One iterator interface over every input the system supports, so the pipeline,
the CLI and the tests never branch on where frames come from.
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

    Accepts a camera index, an ``http(s)://`` stream URL, a video file, an
    image, or a directory of images. URL handling assumes an MJPEG endpoint,
    which is what the ESP32-CAM firmware serves.
    """
    if cfg is not None and isinstance(spec, str) and spec == "config":
        section = cfg.section("source")
        kind = str(section.get("kind", "esp32"))
        if kind == "esp32":
            return Esp32Source(
                url=str(section.get("url", "http://192.168.4.1:81/stream")),
                reconnect_delay_s=float(section.get("reconnect_delay_s", 2.0)),
                max_queue=int(section.get("max_queue", 2)),
            )
        spec = str(section.get("url", 0))

    if isinstance(spec, int) or (isinstance(spec, str) and spec.isdigit()):
        return VideoSource(int(spec), loop=loop, max_frames=max_frames)

    text = str(spec)
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
