# Rover firmware

Two nodes, deliberately kept separate.

## `esp32cam_stream/` — the camera node

An AI-Thinker ESP32-CAM serving MJPEG on port 81, which
`pollivision.io.esp32.Esp32CamStream` consumes.

**Flashing.** The module has no USB. Program it with an FTDI adapter at 3.3 V
logic (5 V to the module's 5V pin), IO0 tied to GND to enter bootloader, then
remove that link and reset to run. Board: *AI Thinker ESP32-CAM*. Partition
scheme: *Huge APP*, or the sketch will not fit.

**Network.** With `WIFI_SSID` left empty the node comes up as an access point
(`PolliVision-Cam` / `pollinate`) and the stream is at
`http://192.168.4.1:81/stream`. That is the arrangement to prefer in the field:
it needs no infrastructure and cannot be disrupted by a network the rover has
wandered out of range of.

**Before trusting any distance the rover reports**, do two things:

1. Fix the resolution and leave it fixed. The perception stack scales its camera
   intrinsics from the configured resolution, so changing frame size mid-run
   silently invalidates every range estimate. `/control` deliberately does not
   expose `framesize` for this reason.
2. Calibrate. The default 65° horizontal FOV in `configs/esp32cam.yaml` is an
   estimate, and range error is directly proportional to focal-length error:

   ```bash
   # grab ~20 checkerboard views at varied angles and distances
   for i in $(seq 1 20); do
     curl -s http://192.168.4.1/capture -o calib/shot_$i.jpg; sleep 2
   done
   pollivision calibrate calib/ --pattern 9x6 --square-size-mm 25
   ```

   Then copy the printed `fx/fy/cx/cy` into your config.

## Telemetry node

The perception stack expects humidity, temperature and time-of-flight range as
newline-delimited JSON, either over serial or MQTT (see
`pollivision/io/telemetry.py`):

```json
{"rh": 62.4, "t": 27.1, "tof_mm": 41}
```

Any MCU that can emit that will do — the rover's existing motion controller is
the natural place for it. A reading older than
`telemetry.max_age_s` is treated as invalid and the electrostatic controller
falls back to its reference conditions, so emit at ≥1 Hz.

The high-voltage driver is intentionally **not** specified here. It is the one
part of this system that can injure someone, and it should be designed against
your actual probe geometry with appropriate current limiting and interlocks. The
perception stack emits target parameters (`{"mode","kv","ms","delay","pol"}`) and
expects the hardware layer to enforce its own independent limits — never treat a
commanded voltage as pre-validated.
