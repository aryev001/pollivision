# Measuring real accuracy

Every accuracy figure in this repository comes from synthetic renders. They
verify logic — that the ovary cue fires on an ovary and not a pedicel, that
ranging is tilt-invariant, that safety limits hold — but they say nothing about
photorealism. **Before you publish a number or deploy to a field, measure on your
own imagery.** This page is the procedure.

---

## 1. Collect a validation set

Aim for **200–400 images**, and bias the collection toward what will actually
break the system rather than toward good photographs:

- Both sexes, and at least 60 pistillate flowers. Cucurbits produce far more
  staminate flowers, so an unstratified sample will barely contain the class you
  most need to get right.
- All four anthesis stages, including senescent — a rover that pollinates spent
  flowers wastes most of its budget.
- Early morning *and* midday light. Cucurbit flowers are anthetic in the morning,
  which is also when the light is least like a dataset photograph.
- Flowers half-buried in foliage. These are the normal case under a vine canopy
  and the hardest case for every stage.
- Shot from the rover's actual camera at its actual mounting height and angle.
  A validation set shot standing up with a phone will flatter the system and
  tell you nothing about how it will behave at 25 cm off the ground.

Capture straight from the rover's camera:

```bash
pollivision stream --esp32 --no-model --save-dir validation/
```

Or, if the rover is not built yet, from the machine's own camera — press
<kbd>s</kbd> to save each still, and prefer `--sync` so the saved overlay
matches the frame it was computed on:

```bash
pollivision webcam --sync --snapshot-dir validation/
```

Be aware of what changes when you do: a laptop camera is a better sensor held at
a different height, so a set captured this way will flatter the system relative
to what the rover's OV2640 will see at 25 cm off the ground. It is useful for
building and debugging the labelling pipeline, not for the number you quote.

## 2. Label it

Ground truth per flower: a box, sex (`male`/`female`), anthesis stage, and
(optionally) a subjective pollen load in {none, light, heavy}. Any labelling tool
that exports YOLO format works. Two rules that matter more than the tool:

- **Label from the image, not from memory.** If you cannot tell the sex from the
  image alone, neither can the model, and marking it `female` because you
  remember that plant makes your ground truth worse than useless.
- **Mark ambiguous flowers as ambiguous** and exclude them from the sex metric
  rather than forcing a call. Otherwise you are measuring your own guessing.

## 3. Run and score

```bash
pollivision detect validation/ --species pumpkin --json results.json
python tools/evaluate.py results.json labels/ --report report.md
```

## 4. Read the results properly

**Detection.** mAP@50 is the headline, but per-class recall is what matters
operationally. A missed pistillate flower is a fruit that never sets; a missed
staminate flower usually just means collecting from the next one. Weight them
accordingly.

**Sex.** Report three numbers, not one:

| | Meaning |
|---|---|
| Accuracy on decided flowers | How often it is right when it commits |
| Abstention rate | How often it declines to decide |
| **Error rate on decided flowers** | The number that actually matters |

A system that abstains on 40 % and is right on 99 % of the rest is far more
useful here than one that always decides and is right on 85 %, because a
confident wrong answer sends the probe to the wrong flower. Tune
`perception.sex.min_confidence` to move along this trade-off deliberately.

**Anthesis.** Score `receptive` vs everything else — the four-way confusion
matrix is interesting but the binary is what gates actuation.

**Pollen.** With only ordinal ground truth, use Spearman rank correlation
against your {none, light, heavy} labels. The controller uses this as a
continuous gain, so monotonicity matters more than absolute calibration.

**Range.** If you can measure true standoff with a tape or a ToF sensor, report
the mean absolute *percentage* error, not the absolute error — the size-prior
estimator's error is multiplicative.

---

## Expected weak points

Where to look first when the numbers disappoint, based on how the stack is
built:

1. **Sex, without depth.** The morphology cue abstains entirely, so the decision
   rests on the vision-language cue alone. This is the single largest expected
   gap, and the fix is a depth channel rather than better prompts.
2. **Flowers in deep shade under the canopy.** The quality gate rejects them on
   exposure. That is the right behaviour, but if it rejects too much, the fix is
   the on-board illuminator (`/control?var=led&val=1`), not a looser gate.
3. **Small flowers at range.** Cucumber and watermelon corollas are 2–3 cm; at
   1 m they occupy few pixels and every chroma-based cue degrades. Lower
   `perception.quality.min_box_px` only if you have verified the cues still work
   at that size.
4. **Pollen estimates under blown highlights.** Direct sun on bright yellow
   petals clips the sensor and destroys the chroma the estimator needs. The
   estimate's confidence already drops; check that you are propagating it.
5. **Prompt sensitivity.** Open-vocabulary detection is genuinely sensitive to
   phrasing. Try several variants in `detector.prompts` before concluding the
   model cannot see your crop — this is often the cheapest large win.

---

## If accuracy is not good enough

In increasing order of cost:

1. **Retune prompts.** Free, and frequently the biggest single improvement.
2. **Add a depth sensor.** The largest structural gain for sex accuracy; on
   synthetic ground truth it took the cue from abstaining to 12/12.
3. **Fit a linear probe** (`tools/train_probe.py`). A few dozen crops per class,
   under a second of CPU. Adapts the sex head to your cultivar and lighting.
4. **Fine-tune a detector** (`tools/autolabel.py` → `notebooks/colab_finetune.ipynb`).
   Half an hour on free cloud GPU, and the largest gain available for detection.
5. **Enable a second detector backend** and set `fusion.require_votes: 2` for
   precision at the cost of recall.

---

## Reporting honestly

If these results go into the paper, report the conditions with the numbers:
which species, how many images, what lighting, whether depth was available, and
the abstention rate alongside the accuracy. A sex-classification accuracy quoted
without its abstention rate is not a meaningful number, and a reviewer who
notices that will rightly distrust the rest.
