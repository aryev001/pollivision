# Running live on a laptop webcam

The rover's eye is an ESP32-CAM, but nothing in the perception stack depends on
that. This page covers running the whole pipeline on the camera already attached
to the machine you cloned the repository onto — which is the fastest way to see
whether it works on your hardware, how fast it runs, and what it does with a
real scene rather than a synthetic one.

---

## Quick start

```bash
git clone <this repo>
cd pollivision/pollivision          # the importable package lives one level down

python -m pip install -r requirements.txt
python -m pip install -e .

pollivision webcam                  # first run downloads ~600 MB of weights
```

Or, without installing anything, from the repository root:

```bash
python run_webcam.py
```

**In VS Code:** open the repository folder, run *Terminal → Run Task →
`PolliVision: install`* once, then press <kbd>F5</kbd> and pick
**PolliVision: webcam (live)**. The other launch configurations cover the fast
mode, the mission loop and camera enumeration.

---

## What you should see

A window showing the live camera feed with the perception overlay composited on
top: a box and fitted corolla ellipse per flower, tinted by sex, the located
ovary, the aiming point, and per-flower sex posterior, stage, pollen estimate,
range and incidence. A panel at the top reports what the pipeline found; a strip
at the bottom reports what the loop is doing.

```
camera 30.0 FPS   display 29.6 FPS   inference 2.10 FPS (476 ms)   overlay +212 ms
analysed 63   skipped 812   live   h for help
```

Read that bottom line carefully the first time, because it explains the design:

| Field | Meaning |
|---|---|
| `camera` | What the sensor is delivering. |
| `display` | What the preview is running at. Should track `camera`. |
| `inference` | How often the pipeline completes a frame. **This is the real rate of the perception stack**, and on a laptop CPU it is 1–3 FPS with the vision-language cues on. |
| `overlay` | How old the boxes on screen are relative to the pixels under them. |
| `skipped` | Frames that arrived while inference was busy and were dropped rather than queued. A large number here is correct behaviour, not a fault. |

### Why the overlay lags

Capture, inference and display run at three genuinely different rates, on
separate threads. The preview shows the newest frame the camera produced with
the newest result the pipeline finished — which was computed on a slightly older
frame. So the boxes trail a fast-moving object by roughly one inference period.

The alternative — showing each analysed frame with its own overlay — is a
1–3 FPS slideshow whose every frame is already stale by the time it appears.
That trade is available with `--sync`, and it is the right choice when you are
judging detection quality or taking screenshots rather than aiming a camera.

---

## Keys

| Key | Action |
|---|---|
| <kbd>q</kbd> / <kbd>Esc</kbd> | quit |
| <kbd>space</kbd> | pause / resume inference (freezes the overlay, preview keeps running) |
| <kbd>s</kbd> | save an annotated still to `snapshots/` |
| <kbd>m</kbd> | toggle the corolla masks |
| <kbd>r</kbd> | toggle rejected detections and their rejection reasons |
| <kbd>o</kbd> | toggle the overlay entirely, to see the raw feed |
| <kbd>f</kbd> | toggle mirroring |
| <kbd>b</kbd> | toggle the per-cue sex evidence breakdown |
| <kbd>h</kbd> | help |

---

## Options worth knowing

```bash
pollivision webcam --list-cameras          # which indices actually work here
pollivision webcam --camera 1              # a USB camera rather than the built-in one
pollivision webcam --fast                  # much faster, less accurate (see below)
pollivision webcam --mirror                # easier to aim a handheld camera
pollivision webcam --width 640 --height 480
pollivision webcam --species cucumber      # crop priors: corolla size, prompts, standoff
pollivision webcam --mission               # also run the closed perceive-plan-act loop
pollivision webcam --record session.mp4    # write the annotated preview
pollivision webcam --json results.json     # per-frame structured output
pollivision webcam --no-window             # headless, prints status instead
pollivision webcam --sync                  # overlay and image always agree
```

### `--fast`

Two changes buy roughly an order of magnitude on a laptop CPU: the
vision-language cues are switched off (they are the dominant cost — several
hundred milliseconds per accepted flower) and the detector's input is shrunk to
448 px.

Accuracy falls, and specifically: with the vision-language cue gone and no depth
channel to drive the ovary-morphology cue, **the sex classifier is left with
almost nothing and will abstain on most flowers**. Use `--fast` to check that the
camera path works and to measure your frame rate, not to judge the stack.

### `--prompt`

The detector is open-vocabulary, so you can point it at anything:

```bash
pollivision webcam --fast --prompt "a person's face" --prompt "a coffee mug"
```

This replaces the crop prompt bank entirely and disables the receptivity gate.
It is the quickest way to confirm the model is genuinely running on your camera
before you have a flower to point it at — the downstream heads will still run,
and will report nonsense about the sex and pollen load of your mug, which is
expected.

### `--mission`

Runs the full closed loop against the live feed: target selection, the adaptive
electrostatic controller, verification and the pollination ledger. Nothing is
actuated — there is no hardware attached — and the humidity driving the voltage
calculation is the mock telemetry value from the config, so **the kilovolt
figures on screen are illustrative, not measurements**.

---

## Accuracy on a webcam, honestly

Two things about a laptop camera limit what these numbers mean:

1. **No depth channel.** The ovary-morphology sex cue needs depth and abstains
   without it, leaving the decision to the vision-language cue. The README
   quantifies the cost: 12/12 correct with depth, 6 correct and 6 abstentions
   without, on synthetic renders. Abstention is the intended failure mode, but
   it is still a failure to decide.
2. **Uncalibrated intrinsics.** Range comes from apparent corolla size against
   the species prior, using a focal length inferred from an assumed 65° field of
   view. That is good to maybe ±20 %. If you care about the range and incidence
   numbers, calibrate:

   ```bash
   # capture 10-20 checkerboard views with the 's' key, then:
   pollivision calibrate snapshots/ --pattern 9x6 --square-size-mm 25
   ```

   and put the resulting `fx/fy/cx/cy` into your config, as the command prints.

The species profile also matters more than it looks: `species.corolla_diameter_m`
is the scale reference the entire monocular range estimate rests on, so
`--species` should match what you are actually pointing at.

---

## Performance

Run `pollivision bench --webcam` for measured per-stage latency on your machine.
Typical numbers on four laptop CPU cores at 1280×720:

| Configuration | Inference rate |
|---|---|
| Default (vision-language cues on, several flowers in frame) | 1–3 FPS |
| Default, one flower in frame | 3–5 FPS |
| `--fast` | 10–20 FPS |

Cost scales with the number of flowers that **pass the quality gate**, not with
the number detected — the gate runs before any model inference precisely so that
occluded and blurred candidates cost nothing. The preview stays at camera rate
regardless.

Things that help, in order of effect:

- `--fast`, or `--set vlm.enabled=false` to drop only the vision-language cues.
- `--width 640 --height 480`: less to decode and resize every frame.
- `--set runtime.max_flowers_per_frame=4`: caps the worst case in a busy scene.
- `--set runtime.torch_threads=4`: PyTorch's default thread count is often wrong
  on laptops with efficiency cores.
- `--device mps` on Apple Silicon, `--device cuda` if you have an NVIDIA GPU.

---

## Troubleshooting

**"Could not open camera 0"** — run `pollivision webcam --list-cameras`. The
error message names each backend that was tried and why it failed, and then the
platform-specific cause, which is almost always one of:

- *macOS*: the camera permission belongs to the app that launched the process.
  System Settings → Privacy & Security → Camera, enable your terminal or VS
  Code, then **restart it** — the permission is only read at launch.
- *Windows*: another application is holding the camera. Windows does not share
  it. Close Teams / Zoom / the Camera app. Also check Settings → Privacy &
  security → Camera.
- *Linux*: `ls -l /dev/video*`; you probably need to be in the `video` group
  (`sudo usermod -aG video $USER`, then log out and back in). Under WSL the host
  webcam is not passed through at all.

**The window never opens, and the log says there is no GUI backend** —
`opencv-python-headless` is installed. It is built without HighGUI, so
`cv2.imshow` cannot work:

```bash
pip uninstall -y opencv-python-headless
pip install opencv-python
```

Never install both; they provide the same `cv2` module and which one wins is
undefined.

**It runs but detects nothing** — likeliest in order:

1. You are not pointing it at a cucurbit flower. The default prompt bank asks
   for large yellow squash blossoms. Try `--prompt` with something present.
2. The confidence floor is too high for your lighting: `--conf 0.05`.
3. Everything is being rejected by the quality gate rather than missed by the
   detector. Press <kbd>r</kbd> to show rejections with their reasons; if you
   see `occluded` or `blurred` on obviously clean flowers, the gate needs
   loosening — see `perception.quality` in `configs/default.yaml`.

**The first frame takes 30 seconds** — that is the one-time weight download.
Run `pollivision fetch` up front to get a progress report instead of a stall,
then `POLLIVISION_OFFLINE=1` guarantees it never reaches for the network again.

**Everything is rejected as badly exposed for the first second** — the warm-up
should absorb that; raise `source.warmup_frames` if your camera's auto-exposure
is slow to settle.
