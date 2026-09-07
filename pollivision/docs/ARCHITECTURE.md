# Architecture

How the stages fit together, and the reasoning behind the arrangement. For the
mapping to the paper see [`PAPER_MAPPING.md`](PAPER_MAPPING.md).

---

## Per-frame flow

`runtime/pipeline.py::PerceptionPipeline.process` runs these in order. The order
is not arbitrary — cheap filtering happens before expensive inference, and
everything that later stages depend on is computed before them.

```
1. Detect        detector ensemble → weighted box fusion       ~250 ms
2. Split roles   flower / bud / ovary / distractor              <1 ms
3. Quality gate  occlusion, truncation, blur, exposure           ~5 ms
4. Shape         corolla mask → fitted ellipse                   ~2 ms
5. Depth         RGB-D passthrough, or monocular + anchoring     0–200 ms
6. Sex           morphology + vision-language + detector         ~60 ms
7. Anthesis      vision-language + geometry                      ~40 ms
8. Pollen        colour, texture, chroma, vision-language        ~40 ms
9. Localise      range → pose → rover frame → incidence          ~3 ms
10. Track        identity, temporal smoothing                    ~1 ms
```

Three ordering decisions carry real weight:

**The quality gate runs before any vision-language inference.** Occluded,
blurred and truncated candidates are discarded before they cost a model forward
pass. On a four-core SBC this reordering alone is the difference between a usable
frame rate and a slideshow.

**Depth is resolved before the perception heads, not during localisation.** The
sex head's morphology cue needs depth to separate an inferior ovary from the
canopy behind it. Resolving depth later — the more natural-looking arrangement,
since depth is "geometry" — would silently disable the cue.

**Ellipse fitting precedes both depth and localisation.** The fitted ellipse
supplies the tilt-invariant size prior that anchors monocular depth to metric
scale, and the corolla normal used for incidence.

---

## Fusion, in two places

**Box fusion** (`fusion/wbf.py`) merges detectors. Non-maximum suppression picks
one winner per cluster and discards the rest, which is wrong when merging
heterogeneous detectors: two models that localise the same flower slightly
differently carry complementary information, and averaging their boxes is more
accurate than picking either. Members are weighted by (backend trust × score),
and the fused score is discounted by how much of the available agreement the
cluster actually attracted.

Fusion happens *within a role*. A corolla and the ovary beneath it overlap
heavily by construction and must never merge.

**Cue fusion** (`fusion/calibration.py`) merges classification evidence. Cue
scores arrive on incompatible scales — a zero-shot softmax is overconfident, a
geometric heuristic is bounded, a trained probe is roughly calibrated — so
averaging probabilities lets the overconfident cue dominate. Instead each cue is
converted to log-odds, weighted, and summed.

Two properties this buys:

- **Neutral cues are genuinely neutral.** Each cue contributes only its
  *departure* from the prior, so a cue returning 0.5 moves nothing regardless of
  how many cues are present.
- **Disagreement is visible.** `agreement_confidence` reports whether the cues
  pointed the same way. A posterior of 0.9 built from two cues that nearly
  cancelled is reported as low-confidence, where a naive average would hide it.

---

## Abstention as a design principle

A cue that cannot make its measurement returns exactly 0.5.

This sounds like a detail and is not. An early version of the ovary-morphology
cue returned "probably male" whenever its segmentation failed. Segmentation
failed on essentially every frame, so the cue emitted a constant confident −1.32
log-odds that outvoted the vision-language cue on every single flower. Fused
accuracy was 6/12, and every error was a *confident wrong answer* rather than an
abstention. Making failure neutral was the fix.

The principle generalises: **absence of evidence is not evidence of absence**,
and a fusion architecture cannot distinguish the two unless the cues do it
themselves. Where absence genuinely *is* informative — a clearly-segmented thin
pedicel really is evidence of a staminate flower — the cue says so, and scales
its confidence by how much of the search band it actually managed to observe.

---

## Degradation

Every optional component fails soft, because a rover that stops is worse than one
that runs on fewer cues:

| Missing | Effect |
|---|---|
| Vision-language model | Geometric cues only; sex accuracy drops substantially |
| Depth | Morphology cue abstains; size-prior ranging takes over |
| A detector backend | Remaining backends carry on; fusion renormalises |
| Telemetry | Reference humidity used; stale readings marked invalid |
| Instance masks | Box-inscribed ellipse; orientation and occlusion degrade |

The one thing that is *not* soft is safety: `control/safety.py` runs
unconditionally and its limits are enforced regardless of what else is missing.

---

## Coordinate frames

Getting this wrong produces plausible-looking numbers that are silently wrong, so
it is stated explicitly.

- **Optical frame** — +x right, +y down, +z forward. All pixel and depth work.
- **Rover body frame** — +x forward, +y left, +z up, origin at the drive centre
  on the ground plane. All planning and actuation.

`Extrinsics.from_config` folds the axis permutation into the mount rotation.
Points are rotated and translated; **normals are rotated only** — a translated
normal still looks like a unit vector, which is why this is tested explicitly.

The corolla normal points *back toward the camera*, so the probe is placed by
moving *along* it from the flower, and the approach direction is its negation.

---

## Temporal smoothing

`tracking/tracker.py` earns its place three times over:

1. **Identity**, so the ledger can prevent repeat pollination.
2. **Accuracy.** Sex, stage and pollen load are constant properties estimated
   from a moving platform with a rolling-shutter sensor. Per-frame noise averages
   out over a track, and each frame's contribution is weighted by that frame's
   own confidence, so a blurred view barely moves the running estimate.
3. **Throughput.** With tracking, the detector can run every Nth frame
   (`runtime.detect_every_n_frames`) and the tracker bridges the gap.

Smoothing only applies to confirmed tracks. A brand-new track has no history
worth trusting, so its single-frame estimate is left alone rather than being
"smoothed" against nothing.

---

## Extension points

- **New species** — add `configs/species/<name>.yaml`. Corolla diameter is the
  important one; it sets the ranging scale.
- **New detector** — subclass `DetectorBackend`, return `Detection` objects with
  canonical role labels, register it in `EnsembleDetector._make_backend`.
- **New sex cue** — add an entry to the `evidence` dict in
  `SexClassifier.classify` with a weight. Log-odds fusion handles the rest.
- **Different actuator** — `ElectrostaticCommand` is the interface. A mechanical
  brush or air-jet end-effector would replace `control/electrostatic.py` and
  leave every other stage untouched.
