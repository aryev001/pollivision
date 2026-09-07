# Paper → implementation mapping

Every substantive claim in *"Adaptive Vision-Guided Electrostatic Pollination
Rover for Precision Pollination of Ground-Level Cucurbit Crops"*, and where it
lives in this codebase. Where the implementation departs from the paper, or
where a claim is not yet supported by evidence, that is stated rather than
glossed.

---

## Sec. III-A — System overview

> "closed-loop perceive-plan-act-verify cycle, in which visual perception drives
> navigation, navigation triggers environmental sensing, sensing parameterizes
> the electrostatic controller, and the outcome of each pollination attempt is
> fed back"

`runtime/mission.py` — `MissionController` is that cycle, as an explicit state
machine over `SEARCH → APPROACH → ACTUATE → VERIFY → RECOVER`.

It emits commands and consumes outcomes but never touches motors or high
voltage itself. That separation is what makes the whole loop testable offline
against recorded video, which for a system whose actuator is a kilovolt probe
is worth the indirection.

Verified by `tests/test_integration.py::TestMissionLoop`, including a test that
humidity measured at the sensor actually reaches the fired command — that the
loop is closed in fact and not just in the diagram.

---

## Sec. III-B — Hardware architecture

| Paper component | Where it is handled |
|---|---|
| RGB-D camera on pan-tilt | `geometry/camera.py`, `geometry/depth.py` (RGB-D passthrough) |
| ToF probe-to-flower sensor | `io/telemetry.py` → `EnvironmentReading.tof_distance_m`; overrides the vision range in `control/electrostatic.py` |
| Humidity sensor | `io/telemetry.py` → drives the humidity gain |
| Variable-polarity HV probe | `control/electrostatic.py` emits `{mode, kv, ms, delay, polarity}` |
| Embedded compute | Whole stack is CPU-only; ~400 ms/frame on four cores |

**Departure.** The paper specifies an RGB-D head. The stated intent for this
build is an ESP32-CAM, which is monocular. `configs/esp32cam.yaml` handles that
by falling back to size-prior ranging, and `docs/EVALUATION.md` quantifies what
is lost. Telemetry readings older than `telemetry.max_age_s` are marked invalid
and the controller reverts to reference conditions, rather than feeding a stale
humidity value into a kilovolt decision.

---

## Sec. III-C — Vision-based flower detection and classification

> "a convolutional neural network trained to localize flowers within the camera
> frame and distinguish male from female cucurbit flowers"

`perception/detector.py` + `perception/sex.py`.

**Departure, and the reason for it.** The paper assumes a *trained* CNN. This
implementation uses an open-vocabulary detector prompted in natural language,
because no labelled cucurbit dataset and no GPU were available, and a rover that
cannot be built until a dataset exists is not useful. The supervised path is
provided (`tools/autolabel.py` → `notebooks/colab_finetune.ipynb`) and a trained
model slots into the ensemble as its highest-weighted member without displacing
anything. Accuracy is lower than a good supervised model would be; availability
is immediate.

> "discriminative morphological cues such as the presence of an inferior ovary
> in female flowers"

`perception/sex.py::_score_morphology`. Implemented as a **width-profile**
analysis of the structure attached to the corolla base: a pedicel is thin and
near-constant in width, an ovary bulges and then narrows.

**Honest limitation.** This cue requires depth. An immature cucurbit fruit is
green against green foliage; colour thresholding and edge-aware region growing
were both implemented and both merged the ovary into the background on scenes
with known ground truth. The cue therefore abstains (returns exactly neutral)
when no depth is available, rather than firing unreliably. Measured on synthetic
renders: **12/12 with depth**; without depth the decision falls to the
vision-language cue and the detector's ovary class, giving 6/12 correct with 6
abstentions and **zero misclassifications**.

> "filtered to remove partially occluded or low-confidence detections"

`perception/occlusion.py`. Runs *before* any vision-language inference, so
rejected candidates cost nothing. Gates on confidence, size, frame-edge
truncation, convexity deficit (a leaf across a corolla carves a bite out of the
mask), Laplacian sharpness and highlight clipping.

This matters beyond tidiness: a truncated mask yields a fitted ellipse implying a
tilt that is not there, which would feed a wrong voltage to the controller.

> "transformed from image space to the rover's local reference frame using the
> depth channel"

`geometry/localizer.py` + `geometry/transforms.py`.

**Extensions beyond the paper.** Two cues the paper does not describe but which
the stated goals require:

- **Anthesis staging** (`perception/anthesis.py`). Cucurbit flowers are anthetic
  for a single morning. The paper speaks of approaching "receptive flowers" but
  does not say how receptivity is determined; without staging, most of the
  rover's actuation budget goes to buds and spent flowers.
- **Stigma localisation** (`perception/stigma.py`). The field falls steeply with
  distance, so aiming at the corolla centroid rather than the stigma wastes much
  of the dose on petal tissue.

---

## Sec. III-D — Autonomous navigation and approach planning

> "computes a short-range approach trajectory that positions the electrostatic
> probe within effective pollination range"

`planning/target.py`. Produces the *goal* — a flower, a probe pose in the rover
frame, and the reason it was chosen. Path planning and obstacle avoidance belong
to the rover's own navigation stack and are deliberately out of scope.

Selection is a weighted score, not nearest-first, because throughput here is
limited by pollen logistics: the rover must collect from a staminate flower
before any pistillate flower is worth visiting. The `pollen_need` term expresses
both "go find pollen" and "go spend it" in one scoring function.

> "the ToF sensor continuously refines the probe-to-flower distance"

`control/electrostatic.py::_resolve_standoff` prefers the ToF reading over the
vision estimate. Tested in `test_control.py::test_tof_reading_overrides_the_vision_range`.

**Extension.** `tracking/ledger.py` records what has been serviced and enforces a
cooldown and attempt cap. Re-acquisition is by position as well as track ID, so
a dropped track does not defeat the memory. Without this the rover
re-pollinates whatever re-enters frame.

---

## Sec. III-E — Adaptive electrostatic pollen transfer

> "dynamically adjusts probe voltage, exposure duration, and discharge timing"

`control/electrostatic.py::compute`.

### (i) Humidity

> "affects pollen grain cohesion and charge retention"

Logistic in RH, normalised to unity at the reference humidity. A logistic rather
than a linear correction because adsorbed water raises exine surface conductivity
gently at moderate RH and steeply past ~60–70 %; a linear term would either
under-compensate when humid or over-drive when dry.

### (ii) Distance and orientation

> "determine the effective field strength at the flower surface"

Voltage scales as `d^1.6` — between the ideal plane (1.0) and sphere (2.0) cases,
tunable because a real probe is neither. Orientation enters as `1/cos(incidence)`,
floored: past a certain obliquity the right response is to reposition the rover,
not to keep raising the voltage.

Incidence comes from `geometry/orientation.py`, which recovers the corolla normal
from the projected ellipse — a circle viewed obliquely projects to an ellipse
whose axis ratio is the cosine of the tilt.

### (iii) Pollen availability

> "an estimated measure of pollen availability derived from the visual appearance
> of the male flower's anther"

`perception/pollen.py`. Four measurements: anther-coloured area fraction within
the inner corolla disc, granular texture energy (fresh pollen is a powder of
discrete grains), chroma excess, and a zero-shot loaded-vs-depleted judgement.

The key detail is that area and chroma are measured **relative to the petal ring
in the same frame**, not against fixed thresholds. Anther and petals are both
yellow and both shift with sunlight and white balance; the relative comparison is
what survives an ESP32-CAM's drifting auto white balance.

### Safety

Not in the paper, but a system driving kilovolts near living tissue needs it.
`control/safety.py` applies limits **after** the controller, independently.

The binding limit is the field at the **probe tip** (corona onset, via Peek's
law), not the field at the flower. An earlier version limited the latter and the
ceiling never engaged — the tip field is two orders of magnitude larger, so
limiting the target field is limiting the wrong quantity. Corona is not merely a
hazard here: its ionic wind disperses the grains the probe is trying to hold.

A command that is safe but **too weak to move pollen** is flagged rather than
silently boosted — boosting would defeat the ceiling that just clamped it.

Verified across a 360-point sweep of the full input space, including impossible
sensor readings, in `test_control.py::TestSafetyInvariants`.

---

## Sec. III-F — Adaptive control algorithm

> "a rule-based mapping refined through empirical calibration, with parameters
> constrained to keep the effective electrostatic field within a safe and
> efficient operating range"

`ControllerGains` holds the coefficients; `configs/default.yaml` documents each
one's physical meaning; `tools/calibrate_electrostatic.py` fits them to measured
transfer-efficiency trials.

**The shipped gains are physically motivated starting points, not measured
constants.** The calibration tool exists precisely because they should be
replaced with values fitted to a specific probe and crop, and it warns when the
supplied trials do not span enough of the operating envelope to constrain the
fit.

> "run in real time on the embedded compute unit without requiring extensive
> onboard processing resources"

The controller is closed-form: microseconds per call, no allocation, no model
inference. `test_control.py` exercises 360 evaluations in well under a second.

---

## Sec. III-G — Verification and feedback

> "a verification step confirms the outcome and feeds the result back into the
> navigation module to select the next target flower"

`control/verification.py`.

Verification is **differential**: absolute appearance is far too sensitive to
viewpoint and exposure to support a threshold, but the change across a single
attempt — same flower, same camera, seconds apart — cancels most of that.

- After collection, the anther should read measurably emptier.
- After deposition, the stigma should read measurably more loaded.

Two guards that matter. A settling delay, because firing the probe disturbs the
flower and measuring immediately reads motion blur as a change in pollen load.
And an exposure-shift check, because both measurements are chroma-based: if the
camera's auto-exposure moved between the two frames the result is reported as
*inconclusive* rather than as a failure. Punishing a flower for a bad frame would
make the rover abandon perfectly good targets.

---

## Table I — Positioning against prior work

The paper's Table I contrasts this work with prior systems. Where those contrasts
are load-bearing:

| Claim | Implementation |
|---|---|
| Ground-level cucurbit crops, not orchard/greenhouse | Species profiles for pumpkin, cucumber, squash, watermelon, muskmelon; camera extrinsics for a low, down-tilted mount |
| Selective transfer, not indiscriminate spraying | Sex classification gates every attempt; the ledger prevents repeats |
| Adaptive parameters, not fixed | Sec. III-E/F above; every command carries the gains that produced it |
| Low-cost, not an expensive manipulator | CPU-only inference; ESP32-CAM supported as the primary sensor |
| Attention to pollen transfer, not perception alone | `pollen.py`, `electrostatic.py`, `verification.py` |

---

## What this codebase does *not* establish

- **Real-world accuracy.** No cucurbit photographs were reachable from the build
  environment. All accuracy figures come from synthetic renders that test logic,
  not photorealism. `docs/EVALUATION.md` provides the harness to measure the real
  numbers, and they must be measured before any claim is published.
- **Pollination efficacy.** No transfer efficiency, fruit set or yield data
  exists. The controller implements the paper's *proposed* mapping; whether that
  mapping improves transfer is exactly what
  `tools/calibrate_electrostatic.py`'s trial data is meant to establish.
- **Navigation under canopy.** Out of scope here; the paper cites Sivakumar et
  al. [13] for this, and this stack emits goals for such a planner to consume.
