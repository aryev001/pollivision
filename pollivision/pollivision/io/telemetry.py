"""Environmental telemetry: humidity, temperature and time-of-flight range.

These are the sensor readings that parameterise the electrostatic controller
(report Sec. III-B and III-E). Three transports are supported - a mock for
offline runs, a serial link to the rover MCU, and MQTT - behind one interface,
so the control loop never learns which one is in use.

Readings are treated as perishable. A humidity value from thirty seconds ago is
not evidence about conditions now, and silently feeding a stale reading into a
kilovolt controller is exactly the kind of quiet failure this module exists to
prevent: past ``max_age_s`` a reading is marked invalid and the controller falls
back to its reference conditions.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Optional

from ..logging_utils import get_logger
from ..types import EnvironmentReading

LOGGER = get_logger(__name__)


class TelemetrySource:
    """Base interface for environmental sensing."""

    #: A reading older than this is no longer trusted.
    max_age_s: float = 10.0

    def read(self) -> EnvironmentReading:
        raise NotImplementedError

    def close(self) -> None:
        pass


class MockTelemetry(TelemetrySource):
    """Fixed readings, for offline runs and tests."""

    def __init__(self, humidity: float = 55.0, temperature_c: float = 26.0,
                 tof_distance_m: Optional[float] = None) -> None:
        self.humidity = float(humidity)
        self.temperature_c = float(temperature_c)
        self.tof_distance_m = tof_distance_m

    def read(self) -> EnvironmentReading:
        return EnvironmentReading(
            humidity=self.humidity,
            temperature_c=self.temperature_c,
            tof_distance_m=self.tof_distance_m,
            valid=True,
        )


class SerialTelemetry(TelemetrySource):
    """Reads newline-delimited JSON from the rover MCU over a serial link.

    Expected line format, which the sketch in ``firmware/`` emits::

        {"rh": 62.4, "t": 27.1, "tof_mm": 41}
    """

    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 115200,
                 max_age_s: float = 10.0) -> None:
        import serial  # pyserial, an optional dependency

        self.max_age_s = float(max_age_s)
        self.serial = serial.Serial(port, baud, timeout=0.2)
        self._latest = EnvironmentReading(valid=False)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="telemetry", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                line = self.serial.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial line or debug output; not worth logging
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Serial telemetry error: %s", exc)
                time.sleep(0.5)
                continue

            reading = EnvironmentReading(
                humidity=float(payload.get("rh", 55.0)),
                temperature_c=float(payload.get("t", 25.0)),
                tof_distance_m=(float(payload["tof_mm"]) / 1000.0
                                if payload.get("tof_mm") is not None else None),
                timestamp=time.time(),
                valid=True,
            )
            with self._lock:
                self._latest = reading

    def read(self) -> EnvironmentReading:
        with self._lock:
            reading = self._latest
        return _freshness_checked(reading, self.max_age_s)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            self.serial.close()
        except Exception:  # noqa: BLE001
            pass


class MqttTelemetry(TelemetrySource):
    """Subscribes to a JSON telemetry topic."""

    def __init__(self, host: str = "localhost", port: int = 1883,
                 topic: str = "rover/telemetry", max_age_s: float = 10.0) -> None:
        import paho.mqtt.client as mqtt  # optional dependency

        self.max_age_s = float(max_age_s)
        self._latest = EnvironmentReading(valid=False)
        self._lock = threading.Lock()

        self.client = mqtt.Client()
        self.client.on_message = self._on_message
        self.client.connect(host, port, keepalive=30)
        self.client.subscribe(topic)
        self.client.loop_start()

    def _on_message(self, _client, _userdata, message) -> None:
        try:
            payload = json.loads(message.payload.decode("utf-8"))
        except Exception:  # noqa: BLE001
            return
        reading = EnvironmentReading(
            humidity=float(payload.get("rh", 55.0)),
            temperature_c=float(payload.get("t", 25.0)),
            tof_distance_m=(float(payload["tof_mm"]) / 1000.0
                            if payload.get("tof_mm") is not None else None),
            timestamp=time.time(),
            valid=True,
        )
        with self._lock:
            self._latest = reading

    def read(self) -> EnvironmentReading:
        with self._lock:
            reading = self._latest
        return _freshness_checked(reading, self.max_age_s)

    def close(self) -> None:
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:  # noqa: BLE001
            pass


def _freshness_checked(reading: EnvironmentReading, max_age_s: float) -> EnvironmentReading:
    """Invalidate a reading that has gone stale."""
    if not reading.valid:
        return reading
    age = time.time() - reading.timestamp
    if age > max_age_s:
        LOGGER.warning("Telemetry is %.1fs stale; falling back to reference conditions", age)
        return EnvironmentReading(
            humidity=reading.humidity, temperature_c=reading.temperature_c,
            tof_distance_m=None, timestamp=reading.timestamp, valid=False,
        )
    return reading


def build_telemetry(cfg) -> TelemetrySource:
    """Construct the telemetry source named in the config, with a safe fallback."""
    kind = str(cfg.get("telemetry.source", "mock")).lower()
    try:
        if kind == "serial":
            section = cfg.section("telemetry.serial")
            return SerialTelemetry(str(section.get("port", "/dev/ttyUSB0")),
                                   int(section.get("baud", 115200)))
        if kind == "mqtt":
            section = cfg.section("telemetry.mqtt")
            return MqttTelemetry(str(section.get("host", "localhost")),
                                 int(section.get("port", 1883)),
                                 str(section.get("topic", "rover/telemetry")))
    except Exception as exc:  # noqa: BLE001
        # Losing telemetry must not stop the rover; the controller degrades to
        # its reference conditions, which is conservative rather than unsafe.
        LOGGER.warning("Telemetry source '%s' unavailable (%s); using mock readings",
                       kind, exc)

    section = cfg.section("telemetry.mock")
    return MockTelemetry(float(section.get("humidity", 55.0)),
                         float(section.get("temperature_c", 26.0)))
