"""Tests for the live-camera path.

No real camera exists in CI, so the capture layer is exercised through a fake
``VideoCapture`` and the session through a fake stream. What is worth asserting
here is not that OpenCV works, but the two behaviours the live path was written
for and which are easy to regress:

* a slow consumer must see the *newest* frame, never a queued backlog;
* a slow pipeline must not slow the display down, and the frames that arrive
  while it is busy must be skipped rather than queued.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from pollivision.io import webcam as webcam_module
from pollivision.io.webcam import WebcamFrame, WebcamStream
from pollivision.runtime.live import LiveSession
from pollivision.types import BBox, FlowerObservation, FrameResult


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #

class FakeCapture:
    """Minimal stand-in for cv2.VideoCapture that counts a frame number in."""

    def __init__(self, width: int = 64, height: int = 48, delay_s: float = 0.0):
        self.width, self.height, self.delay_s = width, height, delay_s
        self.frame_number = 0
        self.released = False
        self.properties: dict[int, float] = {}

    def isOpened(self) -> bool:  # noqa: N802 - mirrors the OpenCV API
        return not self.released

    def read(self):
        if self.delay_s:
            time.sleep(self.delay_s)
        self.frame_number += 1
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Encode the frame number in pixel 0 so a consumer can prove which
        # frame it received.
        image[0, 0] = (self.frame_number % 256, 0, 0)
        return True, image

    def set(self, prop, value):  # noqa: A003 - mirrors the OpenCV API
        self.properties[prop] = value
        return True

    def get(self, prop):
        return self.properties.get(prop, 0.0)

    def release(self):
        self.released = True


class FakeStream:
    """Stands in for WebcamStream: emits frames at a fixed rate, freshest-first."""

    def __init__(self, period_s: float = 0.005, size=(64, 48)):
        self.period_s = period_s
        self._size = size
        self._index = 0
        self.dropped_frames = 0
        self.backend_name = "FAKE"
        self.stopped = False

    @property
    def size(self):
        return self._size

    @property
    def capture_fps(self) -> float:
        return 1.0 / self.period_s

    @property
    def connected(self) -> bool:
        return not self.stopped

    def start(self):
        return self

    def stop(self):
        self.stopped = True

    def read(self, timeout_s: float = 5.0):
        if self.stopped:
            return None
        time.sleep(self.period_s)
        self._index += 1
        width, height = self._size
        return WebcamFrame(image=np.zeros((height, width, 3), dtype=np.uint8),
                           timestamp=time.time(), index=self._index)


class SlowPipeline:
    """A pipeline whose single frame of latency dwarfs the capture period."""

    def __init__(self, latency_s: float = 0.05):
        self.latency_s = latency_s
        self.calls = 0

    def process(self, frame):
        self.calls += 1
        time.sleep(self.latency_s)
        result = FrameResult(frame_index=self.calls,
                             width=frame.shape[1], height=frame.shape[0])
        result.flowers = [FlowerObservation(box=BBox(1, 1, 10, 10), score=0.9)]
        result.latency_ms["total"] = self.latency_s * 1000
        return result

    def warmup(self, shape=(48, 64)):
        pass


class Config(dict):
    """Just enough of pollivision.config.Config for the session."""

    def get(self, path, default=None):
        return super().get(path, default)


# --------------------------------------------------------------------------- #
# Capture
# --------------------------------------------------------------------------- #

def test_stream_yields_the_newest_frame_not_a_backlog(monkeypatch):
    """A consumer slower than the camera must not fall behind in time.

    This is the property that separates WebcamStream from a plain
    VideoCapture.read() loop, and the reason the class exists.
    """
    capture = FakeCapture(delay_s=0.001)
    monkeypatch.setattr(webcam_module.cv2, "VideoCapture",
                        lambda *a, **k: capture)

    stream = WebcamStream(device=0, warmup_frames=0)
    stream.start()
    try:
        first = stream.read(timeout_s=2.0)
        assert first is not None
        time.sleep(0.15)               # let a backlog build up
        second = stream.read(timeout_s=2.0)
        assert second is not None
        # The reader skipped everything captured during the sleep rather than
        # handing back the next one in line.
        assert second.index > first.index
        assert second.age_s < 0.1
        assert stream.dropped_frames > 0
    finally:
        stream.stop()
    assert capture.released


def test_stream_discards_warmup_frames(monkeypatch):
    capture = FakeCapture()
    monkeypatch.setattr(webcam_module.cv2, "VideoCapture", lambda *a, **k: capture)

    warmup = 5
    stream = WebcamStream(device=0, warmup_frames=warmup)
    stream.start()
    try:
        frame = stream.read(timeout_s=2.0)
        assert frame is not None
        # Delivered frames are numbered from 1, but the camera has produced at
        # least `warmup` more than that: the early ones were thrown away rather
        # than handed to the quality gate while auto-exposure was still hunting.
        assert capture.frame_number >= frame.index + warmup
    finally:
        stream.stop()


def test_open_reports_a_useful_error_when_no_backend_works(monkeypatch):
    class DeadCapture(FakeCapture):
        def isOpened(self):
            return False

    monkeypatch.setattr(webcam_module.cv2, "VideoCapture",
                        lambda *a, **k: DeadCapture())
    stream = WebcamStream(device=3)
    with pytest.raises(RuntimeError) as excinfo:
        stream.open()
    message = str(excinfo.value)
    assert "Could not open camera 3" in message
    assert "--list-cameras" in message      # tells the operator what to do next


def test_open_falls_back_to_the_next_backend(monkeypatch):
    """A backend that opens but delivers nothing must not be accepted."""
    attempts = []

    class SilentCapture(FakeCapture):
        def read(self):
            return False, None

    def factory(device, flag):
        attempts.append(flag)
        return SilentCapture() if len(attempts) == 1 else FakeCapture()

    monkeypatch.setattr(webcam_module.cv2, "VideoCapture", factory)
    stream = WebcamStream(device=0, warmup_frames=0)
    capture = stream.open()
    try:
        assert len(attempts) == 2
        assert stream.size == (64, 48)
    finally:
        capture.release()


def test_list_cameras_only_reports_devices_that_deliver(monkeypatch):
    class SilentCapture(FakeCapture):
        def read(self):
            return False, None

    def factory(index, flag):
        return FakeCapture() if index == 0 else SilentCapture()

    monkeypatch.setattr(webcam_module.cv2, "VideoCapture", factory)
    found = webcam_module.list_cameras(max_index=3)
    assert [info.index for info in found] == [0]
    assert found[0].width == 64 and found[0].height == 48


# --------------------------------------------------------------------------- #
# Live session
# --------------------------------------------------------------------------- #

def test_display_outruns_a_slow_pipeline():
    """The whole point of the threaded session: display is not gated on inference.

    With a 50 ms pipeline and a 5 ms camera, a single-threaded loop would show
    at most ~20 frames per second of inference. Here the display must run far
    ahead of it, and the frames captured meanwhile must be skipped rather than
    queued up to be shown late.
    """
    pipeline = SlowPipeline(latency_s=0.05)
    session = LiveSession(Config(), pipeline, window=False, max_frames=40,
                          stream=FakeStream(period_s=0.005))
    assert session.run() == 0

    assert session.stats.frames_displayed == 40
    assert pipeline.calls < session.stats.frames_displayed
    assert session.stats.frames_skipped > 0
    assert session.stats.frames_analysed == pipeline.calls


def test_session_survives_a_pipeline_that_raises():
    """One bad frame must not end a live session."""

    class ExplodingPipeline(SlowPipeline):
        def process(self, frame):
            self.calls += 1
            raise ValueError("boom")

    pipeline = ExplodingPipeline(latency_s=0.0)
    session = LiveSession(Config(), pipeline, window=False, max_frames=15,
                          stream=FakeStream(period_s=0.002))
    code = session.run()
    assert session.stats.frames_displayed == 15   # display kept going
    assert pipeline.calls > 0
    assert code == 1                              # but the failure is reported


def test_overlay_is_composited_onto_the_current_frame():
    """Live mode draws the last result over the newest pixels, not stale ones."""
    pipeline = SlowPipeline(latency_s=0.01)
    session = LiveSession(Config(), pipeline, window=False,
                          stream=FakeStream(period_s=0.002))
    session.stream.start()
    frame = session.stream.read()
    canvas = session._compose(frame.image)
    assert canvas.shape == frame.image.shape
    assert canvas is not frame.image        # never annotates the caller's buffer
    session.close()


def test_headless_falls_back_when_opencv_has_no_gui(monkeypatch):
    monkeypatch.setattr("pollivision.runtime.live.gui_available", lambda: False)
    session = LiveSession(Config(), SlowPipeline(latency_s=0.0), window=True,
                          max_frames=5, stream=FakeStream(period_s=0.002))
    assert session.run() == 0
    assert session.window is False          # degraded rather than crashed


# --------------------------------------------------------------------------- #
# Source routing
# --------------------------------------------------------------------------- #

@pytest.fixture
def captured_webcam(monkeypatch):
    """Record how open_source constructs a WebcamStream, without a camera."""
    calls: list[dict] = []

    class RecordingStream(FakeStream):
        def __init__(self, **kwargs):
            super().__init__()
            calls.append(kwargs)

        def frames(self, timeout_s: float = 5.0):
            yield self.read()

    monkeypatch.setattr(webcam_module, "WebcamStream", RecordingStream)
    return calls


@pytest.mark.parametrize("spec,expected_device", [
    (0, 0),
    ("0", 0),
    ("1", 1),
    ("webcam", 0),
    ("webcam:2", 2),
    ("camera:1", 1),
    ("/dev/video0", "/dev/video0"),
])
def test_open_source_routes_live_cameras_to_the_webcam_reader(
        spec, expected_device, captured_webcam):
    from pollivision.io.sources import WebcamSource, open_source

    source = open_source(spec)
    try:
        assert isinstance(source, WebcamSource)
        assert captured_webcam[0]["device"] == expected_device
    finally:
        source.close()


def test_open_source_applies_camera_settings_from_config(captured_webcam):
    from pollivision.config import load_config
    from pollivision.io.sources import open_source

    cfg = load_config("default", ["webcam"])
    source = open_source("config", cfg)
    try:
        assert captured_webcam[0]["width"] == 1280
        assert captured_webcam[0]["height"] == 720
        assert captured_webcam[0]["warmup_frames"] == 8
    finally:
        source.close()


def test_open_source_still_routes_files_and_urls(monkeypatch, tmp_path):
    import cv2

    from pollivision.io.sources import Esp32Source, ImageSource, open_source

    image = tmp_path / "frame.jpg"
    cv2.imwrite(str(image), np.zeros((8, 8, 3), dtype=np.uint8))
    assert isinstance(open_source(str(image)), ImageSource)

    monkeypatch.setattr("pollivision.io.esp32.Esp32CamStream.start",
                        lambda self: self)
    monkeypatch.setattr("pollivision.io.esp32.Esp32CamStream.stop",
                        lambda self: None)
    assert isinstance(open_source("http://192.168.4.1:81/stream"), Esp32Source)
