"""ESP32-CAM MJPEG stream client.

The rover's eye is an AI-Thinker ESP32-CAM serving Motion JPEG over HTTP. Three
properties of that link shape this client:

* **It stalls.** Wi-Fi on an ESP32 in a field with metal and wet foliage drops
  out regularly. The reader therefore runs on its own thread and reconnects with
  backoff rather than letting a stall block the perception loop.
* **Latency compounds.** MJPEG has no frame dropping - if the consumer is slower
  than the producer, frames queue up and the rover ends up acting on what it saw
  seconds ago. A control loop driving a high-voltage probe toward a plant must
  never do that, so the queue is depth-limited and *stale frames are discarded*
  in favour of the newest one. Throughput is sacrificed for freshness
  deliberately.
* **Its auto-exposure hunts.** Pointing at bright yellow petals against dark
  foliage makes the OV2640's AEC oscillate, which shows up downstream as a
  flickering pollen estimate. Frames therefore carry their measured brightness
  so the pipeline can tell a genuine change from an exposure swing.

The matching firmware is in ``firmware/esp32cam_stream/``.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterator, Optional
from urllib.request import Request, urlopen

import cv2
import numpy as np

from ..logging_utils import get_logger

LOGGER = get_logger(__name__)

_JPEG_SOI = b"\xff\xd8"   # start of image
_JPEG_EOI = b"\xff\xd9"   # end of image


@dataclass
class StreamFrame:
    """One decoded frame plus the metadata the pipeline uses to judge it."""

    image: np.ndarray
    timestamp: float
    index: int
    brightness: float = 0.0
    age_s: float = 0.0


class Esp32CamStream:
    """Threaded MJPEG reader that always yields the freshest available frame."""

    def __init__(
        self,
        url: str,
        reconnect_delay_s: float = 2.0,
        max_queue: int = 2,
        timeout_s: float = 5.0,
        max_reconnects: int = 0,   # 0 = retry forever
    ) -> None:
        self.url = url
        self.reconnect_delay_s = float(reconnect_delay_s)
        self.timeout_s = float(timeout_s)
        self.max_reconnects = int(max_reconnects)

        self._queue: deque[StreamFrame] = deque(maxlen=max(1, int(max_queue)))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._index = 0
        self._connected = False
        self._dropped = 0
        self._error: Optional[str] = None

    # ------------------------------------------------------------------ #

    def start(self) -> "Esp32CamStream":
        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="esp32cam", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def __enter__(self) -> "Esp32CamStream":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ------------------------------------------------------------------ #

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def dropped_frames(self) -> int:
        """Frames discarded as stale. A large number means the loop is too slow
        for the stream, not that the link is bad."""
        return self._dropped

    @property
    def last_error(self) -> Optional[str]:
        return self._error

    def read(self, timeout_s: float = 5.0) -> Optional[StreamFrame]:
        """Return the newest frame, waiting up to ``timeout_s`` for one."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._lock:
                if self._queue:
                    frame = self._queue[-1]     # newest, not oldest
                    self._queue.clear()
                    frame.age_s = time.time() - frame.timestamp
                    return frame
            if self._stop.is_set():
                return None
            time.sleep(0.005)
        return None

    def frames(self, timeout_s: float = 5.0) -> Iterator[StreamFrame]:
        """Iterate over frames until the stream is stopped or times out."""
        while not self._stop.is_set():
            frame = self.read(timeout_s)
            if frame is None:
                if self._stop.is_set():
                    return
                LOGGER.warning("No frame within %.1fs from %s", timeout_s, self.url)
                continue
            yield frame

    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        attempts = 0
        while not self._stop.is_set():
            try:
                self._read_stream()
                attempts = 0
            except Exception as exc:  # noqa: BLE001 - reconnect on anything
                self._connected = False
                self._error = str(exc)
                attempts += 1
                if self.max_reconnects and attempts >= self.max_reconnects:
                    LOGGER.error("Giving up on %s after %d attempts: %s",
                                 self.url, attempts, exc)
                    return
                # Back off, but stay responsive enough to recover quickly once
                # the link returns.
                delay = min(self.reconnect_delay_s * min(attempts, 5), 15.0)
                LOGGER.warning("Stream error (%s); reconnecting in %.1fs", exc, delay)
                if self._stop.wait(delay):
                    return

    def _read_stream(self) -> None:
        request = Request(self.url, headers={"User-Agent": "PolliVision/1.0"})
        with urlopen(request, timeout=self.timeout_s) as response:
            self._connected = True
            self._error = None
            LOGGER.info("Connected to ESP32-CAM stream at %s", self.url)
            buffer = bytearray()
            while not self._stop.is_set():
                chunk = response.read(4096)
                if not chunk:
                    raise ConnectionError("stream ended")
                buffer.extend(chunk)

                # Parse JPEG frames out of the multipart body by scanning for
                # SOI/EOI markers, which is more tolerant of the ESP32's
                # slightly irregular boundary formatting than a strict
                # multipart parser.
                while True:
                    start = buffer.find(_JPEG_SOI)
                    if start < 0:
                        if len(buffer) > 1 << 20:
                            del buffer[:-2]   # nothing usable; avoid unbounded growth
                        break
                    end = buffer.find(_JPEG_EOI, start + 2)
                    if end < 0:
                        if start > 0:
                            del buffer[:start]
                        break
                    payload = bytes(buffer[start:end + 2])
                    del buffer[:end + 2]
                    self._emit(payload)

    def _emit(self, payload: bytes) -> None:
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return  # truncated frame; the next one will be along shortly
        self._index += 1
        frame = StreamFrame(
            image=image,
            timestamp=time.time(),
            index=self._index,
            brightness=float(image.mean()) / 255.0,
        )
        with self._lock:
            if len(self._queue) == self._queue.maxlen:
                self._dropped += 1
            self._queue.append(frame)


def snapshot(url: str, timeout_s: float = 5.0) -> Optional[np.ndarray]:
    """Grab a single still from an ESP32-CAM ``/capture`` endpoint.

    Used by ``pollivision calibrate`` for checkerboard capture, where a stream
    is unnecessary and a clean single exposure is preferable.
    """
    try:
        request = Request(url, headers={"User-Agent": "PolliVision/1.0"})
        with urlopen(request, timeout=timeout_s) as response:
            data = response.read()
        return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Snapshot from %s failed: %s", url, exc)
        return None
