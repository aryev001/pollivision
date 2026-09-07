# PolliVision

Perception and adaptive-control stack for the rover described in *"Adaptive
Vision-Guided Electrostatic Pollination Rover for Precision Pollination of
Ground-Level Cucurbit Crops"*.

It detects cucurbit flowers, tells staminate from pistillate, judges whether a
flower is actually receptive, estimates how much pollen is sitting on an anther,
localises the flower in the rover's frame, computes the electrostatic parameters
for the transfer, and verifies whether the transfer worked — the full
perceive → plan → act → verify loop of the paper's Fig. 1.

**It requires no training and no GPU.** The detector is open-vocabulary: its
classes are set from natural-language prompts at load time, so a flower detector
exists the moment the weights are downloaded. Everything runs inference-only on
a CPU.

---

## Quick start

```bash
pip install -r requirements.txt
pip install -e .

pollivision fetch                 # ~600 MB into ~/.cache/pollivision, once
pollivision selftest              # verify the install end to end

pollivision detect photos/ --species pumpkin --output annotated/
pollivision run video.mp4 --species pumpkin --video-out mission.mp4
```

With the ESP32-CAM as the rover's eye:

```bash
pollivision stream --esp32 --url http://192.168.4.1:81/stream --show
pollivision run config --esp32 --species cucumber
```

Everything after the first `fetch` works offline. Set `POLLIVISION_OFFLINE=1`
to make that guarantee explicit and fail loudly rather than stalling on a
download in the field.

---

## Why it is built this way

**No training in the critical path.** The obvious approach — collect a few
thousand cucurbit flower images, label them, fine-tune a detector — needs
labelled data and a GPU, and neither was available. Instead the primary detector
is [YOLOE](https://github.com/ultralytics/ultralytics), prompted with phrases
like *"a large yellow pumpkin blossom"*. Accuracy is lower than a well-trained
supervised model, so a distillation path is provided (below) for when field data
exists. But the rover works on day one.

**Weights come from GitHub, not a model hub.** Every default weight is mirrored
on GitHub release assets, including the vision-language model. Institutional and
agricultural networks block model hubs far more often than they block GitHub,
and a rover that cannot fetch its weights is not a rover.

**Multiple models, fused rather than stacked.** Detection merges any number of
backends with weighted box fusion; classification fuses independent cues in
log-odds space. Each cue carries its own weight and its contribution is recorded,
so a misclassification can be traced to the cue that caused it instead of being
an opaque score.

**Cues abstain rather than guess.** A cue that cannot make its measurement
returns exactly neutral. This matters more than it sounds: an early version had
the ovary-morphology cue return "probably male" whenever its segmentation
failed, and it confidently outvoted the cues that had actually worked. Absence
of evidence is not evidence of absence.

---

## Architecture

```
frame ──► detector ensemble ──► quality gate ──► depth ──► perception heads ──► localiser ──► tracker
          (open-vocab +          (occlusion,     (RGB-D    (sex, anthesis,     (range,       (identity,
           optional experts,      truncation,     or        pollen, stigma)     orientation,  smoothing)
           box fusion)            blur, exposure) monocular)                    rover frame)
                                                                                      │
                                       ┌──────────────────────────────────────────────┘
                                       ▼
                         target selection ──► adaptive electrostatic
                         (Sec. III-D)         controller (Sec. III-E/F)
                                       │              │
                                       │              ▼
                                       │      safety limits (corona ceiling,
                                       │      standoff floor) — independent
                                       ▼              │
                              verification ◄──────────┘
                              (Sec. III-G) ──► ledger ──► back to target selection
```

| Stage | Module | Paper |
|---|---|---|
| Open-vocabulary detection + masks | `backends/yoloe.py` | III-C |
| Multi-model box fusion | `fusion/wbf.py` | III-C |
| Quality / occlusion gate | `perception/occlusion.py` | III-C |
| Sex classification | `perception/sex.py` | III-C |
| Anthesis staging | `perception/anthesis.py` | III-C |
| Pollen availability | `perception/pollen.py` | III-E-iii |
| Range and orientation | `geometry/` | III-C, III-D |
| Target selection | `planning/target.py` | III-D |
| Adaptive electrostatic control | `control/electrostatic.py` | III-E, III-F |
| Safety limits | `control/safety.py` | III-E |
| Outcome verification | `control/verification.py` | III-G |
| Closed loop | `runtime/mission.py` | Fig. 1 |

Full mapping in [`docs/PAPER_MAPPING.md`](docs/PAPER_MAPPING.md).

---

## The models

| Role | Model | Size | Why |
|---|---|---|---|
| Detection + segmentation | YOLOE-11S-seg | 27 MB | Open-vocabulary, no training. Instance masks drive orientation, occlusion and anther analysis. |
| Sex / stage / pollen cues | MobileCLIP-B (LT) | 572 MB | Strong zero-shot image–text scoring on tight crops. |
| Monocular depth *(optional)* | MiDaS | 82 MB | Range recovery when there is no depth channel. |
| Ovary morphology | — | — | Closed-form geometry, no weights. |
| Supervised student *(optional)* | YOLO11s | 19 MB | Your own fine-tune; joins the ensemble as the highest-weighted member. |

Measured on four CPU cores at 640×480: **~80 ms** for detection and gating
alone, rising to **350–850 ms** when the vision-language cues run over several
accepted flowers. Cost scales with the number of flowers that pass the quality
gate, not with the number detected — which is why the gate runs first. Raise
`runtime.detect_every_n_frames` on slower hardware; the tracker bridges the
skipped frames. Run `pollivision bench` to measure your own hardware.

**A note on the vision-language model.** It ships as a TorchScript bundle that
carries both towers' weights but only exports the *text* tower's forward method,
so the image tower cannot be called directly. Rather than reaching for a model
hub, `backends/clip_embed.py` lifts the state dict out of the bundle and loads it
into the matching architecture from the open-source package — all 313 tensors
match exactly — recovering a working image encoder from a file that is already
reachable. A partial match raises rather than silently producing meaningless
embeddings.

---

## Adaptive electrostatic control

The paper's central contribution: charging parameters recomputed per attempt
from humidity, standoff, corolla orientation and pollen availability.

| Input | Effect | Reasoning |
|---|---|---|
| Standoff | V ∝ d^1.6 | Field falls as ~1/d²; hold the surface field roughly constant. |
| Humidity | logistic boost | Adsorbed water raises exine surface conductivity, so charge bleeds off — steeply past ~60–70 % RH. |
| Incidence | V ∝ 1/cos θ, floored | Only the field component normal to the corolla does useful work. |
| Pollen load | longer dwell when sparse | A depleted anther is yield-limited, not force-limited. |

Every command then passes an **independent** safety stage. The binding limit is
the field at the *probe tip*, which governs corona onset (via Peek's law), not
the much weaker field out at the flower — an early version limited the latter and
the ceiling never engaged, because it sits two orders of magnitude below the
former. Corona is not merely a hazard here: its ionic wind disperses the grains
the probe is trying to hold.

The controller also reports when a command is **safe but too weak to move
pollen**, rather than quietly raising the voltage to compensate — which would
defeat the ceiling that just clamped it.

```
RH 30 % → 5.67 kV   RH 55 % → 6.30 kV   RH 95 % → 8.39 kV
20 mm   → 2.08 kV   40 mm   → 6.30 kV   60 mm   → 12.0 kV (ceiling)
0°      → 6.30 kV   45°     → 8.39 kV   60°     → 11.3 kV
```

The shipped gains are **physically motivated starting points, not measured
constants**. Fit them to your own probe with
`tools/calibrate_electrostatic.py`, which is the empirical calibration Sec. III-F
describes.

---

## What is validated, and what is not

Being precise about this, because it determines what you can rely on.

**Verified here (113 passing tests, `pytest tests/`):**

- Ranging is exactly tilt-invariant. Using the fitted ellipse's major axis gives
  0.0 % error across tilt and roll, where a bounding-box-side estimator drifts to
  **+30 %** on a flower both tilted and rotated. This is the difference between
  stopping at the right standoff and driving the probe into a plant.
- Orientation recovery is exact on synthetic ellipses; incidence is 0° with the
  probe on the normal and equals the tilt along the optical axis.
- Extrinsics are proper rotations; normals rotate without translating.
- Electrostatic safety limits hold across a 360-point sweep of the full input
  space, including physically impossible sensor readings.
- The corona ceiling actually binds for a sharp tip, and weak commands are
  flagged.
- Mission logic: collect before deposit, never re-pollinate a serviced flower,
  re-acquire a flower by position after a track is lost, always verify.
- The open-vocabulary detector and vision-language model were confirmed working
  on real photographs.

**Sex classification, measured on synthetic renders with exact ground truth:**

| Configuration | Result |
|---|---|
| RGB-D (vision-language + depth morphology) | **12/12** |
| Monocular (vision-language only) | 6/12 correct, **6 abstained, 0 wrong** |

The monocular failures are *abstentions*, not misclassifications — the correct
failure mode for a rover that would otherwise pollinate the wrong flower.

**Not validated:** accuracy on real cucurbit imagery. No cucurbit photographs
were reachable from the build environment, so every accuracy number above comes
from geometric caricatures. They test logic, not photorealism. Real-world
detection and sex accuracy **must** be measured on your own captures before
field deployment — see [`docs/EVALUATION.md`](docs/EVALUATION.md), which includes
the harness to do it.

**Known limitation.** The ovary-morphology cue needs depth. An immature cucurbit
fruit is green against green foliage; colour thresholding and edge-growing
segmentation were both implemented and both merged the ovary into the background.
Rather than ship a cue that fires unreliably, it abstains without depth and the
decision rests on the vision-language cue and the detector's ovary class. **Adding
a depth channel is the single highest-value upgrade to sex accuracy** — the table
above quantifies it.

---

## Adapting it to your crop

Zero-effort, in increasing order of cost:

1. **Species profile** — `--species pumpkin|cucumber|squash|watermelon|muskmelon`.
   Sets corolla size priors, prompts and standoff windows.
2. **Prompts** — edit `detector.prompts` in your config. No code, no retraining.
3. **Linear probe** (~2 min, CPU) — `tools/train_probe.py` on a few dozen crops
   per class fits a logistic head over the vision-language embeddings.
4. **Full fine-tune** (~30 min, free Colab GPU) — auto-label with the heavy
   teacher, review, then `notebooks/colab_finetune.ipynb`:

```bash
python tools/autolabel.py field_images/ --output dataset --teacher yoloe-11l-seg --preview
# review dataset/preview/ — the teacher makes systematic mistakes
# then run the notebook, and export with tools/export.py
```

---

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how the stages fit together and why
- [`docs/PAPER_MAPPING.md`](docs/PAPER_MAPPING.md) — every paper claim → the code implementing it
- [`docs/ESP32CAM.md`](docs/ESP32CAM.md) — wiring, flashing, calibration, tuning
- [`docs/EVALUATION.md`](docs/EVALUATION.md) — how to measure real accuracy
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — rover integration, safety, offline operation
- [`firmware/README.md`](firmware/README.md) — camera and telemetry nodes

## Safety

This system commands kilovolts near living tissue and, potentially, near people.
The software limits are a backstop, not a substitute for hardware protection.
Give the high-voltage driver its own independent current limiting and interlocks,
and never treat a commanded voltage as pre-validated. See
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).
