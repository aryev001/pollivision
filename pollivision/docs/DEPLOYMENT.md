# Deployment

Integrating the stack with a physical rover, and the safety obligations that
come with it.

---

## Safety first

**This system commands kilovolts near living tissue and potentially near people.
The software limits are a backstop, not a substitute for hardware protection.**

`control/safety.py` enforces a corona-onset ceiling on the tip field, a minimum
standoff, and absolute voltage and exposure bounds, verified across a 360-point
sweep of the input space including impossible sensor readings. That is worth
having, and it is not enough. It cannot protect against a driver that ignores
its input, a probe that has drifted from its assumed geometry, or a person
reaching into the workspace.

The high-voltage driver **must** have:

- **Independent current limiting** sized so a direct short cannot deliver a
  dangerous charge, set in hardware and not in firmware.
- **A hardware interlock** — an enable line the compute unit must actively hold,
  which fails safe on crash, watchdog reset or power loss.
- **Its own limits.** Never treat a commanded voltage as pre-validated. The
  perception stack's output is a *request*.
- **Bleed resistors** across any storage capacitance, so the probe is not live
  after power-down.
- **Physical guarding** and clear marking of the probe's working volume.

Verify the interlock trips before the first energised test, and treat every
"it should be off" as "it might be on".

---

## Integration

The stack is a decision-maker, not a driver. It emits goals and commands and
consumes outcomes; motion and high-voltage hardware stay behind your own layer.

```python
from pollivision import load_config
from pollivision.runtime.pipeline import PerceptionPipeline
from pollivision.runtime.mission import MissionController
from pollivision.io.telemetry import build_telemetry

cfg = load_config("default", ["species/pumpkin"])
pipeline = PerceptionPipeline(cfg)
mission = MissionController(cfg)
telemetry = build_telemetry(cfg)

def fire(command) -> bool:
    """Actuate the probe. Return whether it actually fired.

    Enforce your own limits here regardless of what was requested.
    """
    return hv_driver.pulse(kv=command.voltage_kv,
                           ms=command.exposure_ms,
                           polarity=command.polarity)

for frame, depth in camera.frames():
    result = pipeline.process(frame, depth)
    step = mission.step(
        result,
        telemetry.read(),
        frame=frame,
        probe_at_pose=motion.probe_settled(),   # your motion controller says
        fire=fire,
    )
    if step.plan is not None:
        motion.goto(step.plan.probe_position, step.plan.approach_direction)
```

Two contract points worth stating:

**`probe_at_pose` must come from your motion controller.** The vision stack
cannot observe its own end-effector. Passing `True` unconditionally is what the
CLI does for offline replay against recorded video, and is wrong on hardware.

**`fire` returning `False` routes the mission to `RECOVER`.** Use it — report
actuation failures rather than swallowing them, or the ledger will record
attempts that never happened.

---

## Wire format

`MissionStep.to_wire()` produces a compact dict for a serial or MQTT link:

```json
{
  "state": "verify",
  "goal": {"track": 7, "sex": "male", "mode": "collect",
           "pos_m": [0.31, 0.02, 0.27], "dir": [-1.0, 0.0, 0.0],
           "standoff_m": 0.04},
  "hv": {"mode": "collect", "kv": 6.3, "ms": 413, "delay": 0, "pol": -1}
}
```

Telemetry comes back as newline-delimited JSON at ≥1 Hz:

```json
{"rh": 62.4, "t": 27.1, "tof_mm": 41}
```

A reading older than `telemetry.max_age_s` (default 10 s) is marked invalid and
the controller reverts to reference conditions. Do not work around this by
resending a cached value — a stale humidity reading feeding a kilovolt decision
is exactly the failure it prevents.

---

## Offline operation

Prime the weight cache before going out, then make offline operation explicit:

```bash
pollivision fetch --model yoloe-11s-seg mobileclip-blt
export POLLIVISION_OFFLINE=1
```

With `POLLIVISION_OFFLINE=1`, a missing weight raises immediately with the
command to fix it, instead of stalling on a download attempt in a field with no
signal. Set `POLLIVISION_HOME` to relocate the cache (a read-only or
network-mounted cache shared across several rovers works fine).

---

## Performance

Measured on four CPU cores at 640×480: ~80 ms for detection and gating alone,
350–850 ms once the vision-language cues run over several accepted flowers. The
variable cost is per *accepted* flower, so a frame full of rejected candidates is
cheap. To go faster, in order of effect:

1. **`runtime.detect_every_n_frames: 2` or `3`.** The tracker bridges the gap.
   Nearly free at rover approach speeds.
2. **Lower `detector.backends[].imgsz`** to 512 or 448.
3. **Fine-tune and export a student** (`tools/export.py`). ONNX gives ~1.5–2×
   on CPU; NCNN more on ARM.
4. **`vlm.enabled: false`.** Large saving, large accuracy cost — the sex and
   pollen cues lean on it heavily. Only worth it with a trained probe in place.
5. **Fewer prompts.** Text embedding is one-time, but detection cost scales with
   the class count.

`pollivision bench` reports per-stage latency on your actual hardware; measure
before optimising.

---

## Pre-flight checklist

- [ ] `pollivision selftest` passes on the rover's own compute
- [ ] Camera calibrated; `fx/fy/cx/cy` in the config match the running resolution
- [ ] Frame size fixed and not changed after calibration
- [ ] Extrinsics measured — mount height, offset and pitch actually match the config
- [ ] Species profile matches the crop
- [ ] Telemetry arriving at ≥1 Hz; readings plausible
- [ ] HV interlock verified to trip; current limit verified in hardware
- [ ] Controller gains calibrated (`tools/calibrate_electrostatic.py`) or the
      shipped starting points knowingly accepted
- [ ] Weight cache primed and `POLLIVISION_OFFLINE=1` set
- [ ] Accuracy measured on real imagery (`docs/EVALUATION.md`) — not assumed
