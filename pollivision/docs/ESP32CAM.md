# ESP32-CAM as the rover's eye

The perception stack was built to run on a bare AI-Thinker ESP32-CAM (OV2640).
This page covers what that costs, how to set it up, and how to get the most out
of a sensor that is, frankly, marginal for the task — which is precisely why it
is worth being careful with.

---

## What the ESP32-CAM costs you

| | RGB-D head (paper baseline) | ESP32-CAM |
|---|---|---|
| Range | Direct measurement | Size prior from corolla diameter |
| Range error | ~1–2 cm | ~25–40 %, dominated by the biological spread of corolla size |
| Ovary morphology cue | Works (12/12 on synthetic ground truth) | **Abstains** |
| Sex decision rests on | Depth + vision-language + detector | Vision-language + detector |
| Frame rate | Sensor-limited | ~10–25 fps at SVGA, Wi-Fi dependent |

The honest summary: it works, and the system degrades explicitly rather than
silently. But **adding a depth channel is the single highest-value upgrade** to
this rover's perception, and the sex-classification table above is why.

---

## Setup

**1. Flash the firmware.** `firmware/esp32cam_stream/esp32cam_stream.ino`. The
module has no USB — program it via an FTDI adapter at 3.3 V logic (5 V to the
module's 5V pin) with IO0 tied to GND, then remove that link and reset. Board:
*AI Thinker ESP32-CAM*; partition scheme: *Huge APP*, or it will not fit.

**2. Connect.** With `WIFI_SSID` left empty it comes up as an access point
(`PolliVision-Cam` / `pollinate`), stream at `http://192.168.4.1:81/stream`.
Prefer this in the field: no infrastructure, and nothing to lose range to.

**3. Verify the stream** before involving the model:

```bash
pollivision stream --esp32 --no-model --url http://192.168.4.1:81/stream
```

Watch the reported frame age and dropped count. Frame age above ~200 ms means
the link is the bottleneck; a large dropped count means your perception loop is
slower than the stream, which is fine — the client deliberately discards stale
frames rather than building latency.

**4. Calibrate. Do not skip this.** The default 65° FOV is an estimate, and
**range error is directly proportional to focal-length error**. Every downstream
consumer — approach standoff, and through it the commanded voltage — inherits it.

```bash
mkdir calib
for i in $(seq 1 20); do
  curl -s http://192.168.4.1/capture -o calib/shot_$i.jpg
  sleep 2   # move the checkerboard between shots
done
pollivision calibrate calib/ --pattern 9x6 --square-size-mm 25
```

Vary angle and distance across the 20 views; 20 near-identical shots produce a
confident and wrong answer. An RMS reprojection error below 1.0 px is good.
Copy the printed `fx/fy/cx/cy` into your config.

**5. Run.**

```bash
pollivision run config --esp32 --species cucumber
```

---

## Tuning

`configs/esp32cam.yaml` already accounts for the sensor's quirks:

- **Sharpness gate loosened** (`min_sharpness: 0.07`). JPEG ringing depresses
  Laplacian energy, and the default gate throws away usable frames.
- **Anther hue gate widened**, with more weight on the relative petal-baseline
  comparison. The OV2640's auto white balance drifts, so absolute hue thresholds
  are unreliable; the in-frame relative comparison is not.
- **`imgsz: 512` and `detect_every_n_frames: 2`.** Keeps the loop responsive;
  the tracker bridges the skipped frames.
- **Size-prior ranging**, since there is no depth channel.

Runtime sensor adjustments via `/control?var=<name>&val=<n>`:

| Variable | Use |
|---|---|
| `led` | On-board illuminator. The fix for flowers rejected on exposure under a dense canopy. |
| `gainceiling` | Lower it if the pollen estimate flickers — that is usually AGC hunting, not real change. |
| `saturation` | The sex and pollen cues are chroma-based; a small lift helps. |
| `quality` | 10–15. Lower is better quality and bigger frames. |

**`framesize` is deliberately not exposed.** The stack scales camera intrinsics
from the configured resolution, so changing it mid-run silently invalidates every
range estimate — and therefore every standoff and every commanded voltage.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| Brownouts, resets under load | Underpowered supply. The OV2640 draws ~250 mA in bursts; a 5 V/1 A supply on short, thick leads. USB-serial adapter power is usually not enough. |
| Stream stalls after seconds | Overheating, or Wi-Fi power saving. Add a heatsink; keep the antenna clear of the chassis. |
| Frames decode but look torn | JPEG quality too aggressive for the link. Raise `quality` (a larger number = smaller frames). |
| Pollen estimate flickers frame to frame | AGC hunting on bright petals against dark foliage. Lower `gainceiling`. |
| Everything rejected as `exposure` | Deep canopy shade. Turn on the illuminator before loosening the gate — the gate is protecting the chroma cues that need the light. |
| Ranges are consistently wrong by a fixed ratio | Not calibrated, or the frame size changed after calibration. |
