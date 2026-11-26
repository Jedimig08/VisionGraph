# VisionGraph Simulation & Motor-Control Dev Plan

Status: design only — no implementation code has been written.

## 1. Where the project is today

- Perception-only pipeline: `cv_stream.py:569-577` runs
  `ArucoMarkerDetector.detect` → `MarkerTracker.update` → `PoseEstimator.update`.
- Output is `PoseEstimate`: `x_mm`, `y_mm`, `angle_deg`, `reliability`,
  `predicted` (`pose_estimator.py:170-190`). It is drawn as a HUD only and
  never sent to a motor.
- Motors: two DC motors on an H-bridge (pins 32/33 and 25/26) driven by raw
  strings `P <pin> <0-255>` via `POST /bluetooth-send`
  (`commandhub_manual_control.py:45-70`).
- No encoders, no motor feedback, no simulator, no ML model, no inverse
  kinematics, no closed loop. `README.md:10-11` confirms kinematics and
  motor integration are not implemented.
- Dependencies: `numpy`, `opencv-contrib-python`, `zeroconf`
  (`requirements.txt`). Tests use `unittest` (`README.md:119`).

## 2. Goal

A lightweight, code-only simulator used to find the best motor control
strategy. Two toolpath modes:

- Mode A — "make a drawing": image/shape → strokes → optimize stroke order,
  simplification, and feedrate for fast, clean output.
- Mode B — "G-code": parse and follow G-code as accurately as possible.

Both produce the same segment representation and feed the same controller.
The controller is tuned/trained in simulation. Compare classical control
against RL. Observations are pose state only (no rendered pixels).
Dependencies stay lightweight: numpy + existing deps, optionally
scipy/gymnasium/stable-baselines3. No MuJoCo/Gazebo/ROS.

The simulator must **build itself**: it starts from a crude model, tests it
against the physical machine, and automatically keeps improving the model
until it behaves like the machine within a measured tolerance.

## 3. What you provide vs what the system discovers

- **You provide only the drawing-area width and height** (already a startup
  input, `README.md:38-39`). No motor specs, spool radius, mass, friction,
  gains, or hand-tuned constants.
- **The system discovers:** anchor geometry and cord lengths from camera
  pose; motor→cord polarity; motor gain, deadzone, saturation, lag and rate
  limits; latency; mass, damping, swing and drag; gravity/friction asymmetry;
  region-dependent nonlinearity; noise/dropout statistics; and the mapping
  from swing state to the camera's `angle_deg`.

Everything is learned by auto-driving the machine and logging pose through
the existing camera pipeline. No new sensors are required.

## 4. Core idea: one interface, two implementations

Create `interfaces.py` with a single env contract:

    reset() -> obs
    step(pwm_commands) -> (obs, reward, done, info)

`pwm_commands` are the exact strings from `commandhub_manual_control.py:54-70`,
so the same controller code runs in sim and on hardware.

- `SimEnv` — numpy model of the machine.
- `RealEnv` — wraps `CommandHub` + the existing CV pipeline.

Observations mirror `PoseEstimate` fields (`x_mm`, `y_mm`, `angle_deg`,
`reliability`, `predicted`), so the controller cannot tell which env it is in.
The `AutoModeler` also uses `RealEnv` to drive and observe the machine.

## 5. "Physics" = pure math, not an engine

The simulator is difference equations in numpy, not a physics library.
It has two optional stages behind a `dynamics` flag.

### Stage 1 — kinematic (quasi-static)

With anchors at the drawing area's top corners `(0, 0)` and `(W, 0)`, the
camera pose directly determines cord lengths — no spool or encoder constants
needed:

    L1 = hypot(x, y)
    L2 = hypot(x - W, y)
    x  = (L1^2 - L2^2 + W^2) / (2W)
    y  = sqrt(L1^2 - x^2)

Useful for: checking IK, toolpath generation, stroke ordering, feedrate
planning, and a trivially simple baseline controller. It describes an
impossible machine (zero mass, no gravity, no lag), so it cannot teach swing
suppression or settling behaviour.

### Stage 2 — dynamic (the same math, with time)

Cord tension and Newton's law, integrated at fixed substeps (~1 ms):

    T_i = k * max(0, L_i - dist_i) + c * (Ldot_i - radial_velocity_i)
    a   = (T_1 + T_2 + gravity + drag) / m
    v  += a * dt
    p  += v * dt

Motor model per cord: signed command `u_i = pwm_up_i - pwm_down_i` →
commanded length rate with fitted gain, deadzone, saturation, rate limit,
first-order lag and backlash.

Observation model: Gaussian noise, camera-rate sampling (~15-30 fps), latency
queue, dropouts with the `predicted` flag, and an `angle_deg` signal regressed
from swing state against the real camera's measurement.

`PhysicalParams` holds current values plus randomization ranges for domain
randomization. All its values are fitted by the AutoModeler, never entered by
hand.

## 6. AutoModeler — the self-building loop

Unattended, deterministic, and bounded. It drives the machine, validates the
sim on held-out runs, and rebuilds the model when the sim disagrees.

    M = kinematic_model(W, H)
    D_fit, D_val = [], []
    while budget remains:
        D_fit += run_experiment(M.uncertainty)
        fit_params(M, D_fit)
        err = held_out_error(M, D_val)        # replay recorded commands open-loop
        if err <= tolerance and stable for 2 runs:
            accept; break
        signature = classify_residuals(M, D_val)
        if no rule matches or err regressed:
            revert to best snapshot; try next action
        M = repair(M, signature)              # deterministic table
    report best model + unresolved signature if budget exhausted

Tolerance defaults to ~2–3× the measured camera noise floor, so it requires no
extra input from you.

### 6.1 Experiments (fully automatic, no user input)

- **A — sanity:** require locked calibration and valid pose; measure the
  reachable workspace from observed poses.
- **B — polarity probe:** short low-speed steps on each motor; observe which
  cord length changes and its sign → derive motor→cord mapping.
- **C — step sweep:** each motor at 20/40/60/80% both directions, hold then
  rest → gain, deadzone, saturation, lag, overshoot.
- **D — PRBS/chirp:** 30–60 s of pseudo-random direction flips → broadband
  dynamics.
- **E — validation moves:** random safe point-to-point targets, held out from
  fitting.
- Later rounds target the regions/conditions where residuals are largest.

All moves are pen-up, speed-capped, and clipped to an inset safety envelope;
abort on invalid pose or loss of tracking.

### 6.2 Staged fitting

1. Polarity from command↔cord-length correlation (rule-based).
2. Gain/deadzone/saturation from steady-state cord speed vs PWM (regression).
3. Lag from 63% rise time; damping/inertia from overshoot and swing decay.
4. Remaining dynamics by nonlinear least squares: replay recorded commands
   into the sim and minimize multi-step pose prediction error over 0.5–2 s
   windows.
5. Latency by grid search maximizing fit; noise/dropout from residual
   statistics; angle mapping by regressing swing state → `angle_deg`.

### 6.3 Repair table (the "keep building" logic)

| Residual signature | Likely cause | Repair |
|---|---|---|
| Constant x/y offset | anchor/calibration bias | add bias term |
| Error scales with command | motor gain/deadzone | refit gain/deadzone |
| Phase lag, right amplitude | latency / lag | add delay + first-order lag |
| Overshoot then oscillation | mass/damping/swing | add second-order swing state |
| Worse near edges | cord-wrap/sag nonlinearity | add region-dependent term |
| Up ≠ down | gravity/Coulomb friction | separate direction params |
| `angle_deg` unexplained | angle model too simple | refit angle regression |
| Random scatter | noise/dropout | fit noise model + `predicted` handling |
| Fast moves only | saturation/rate limit | add command limits |

### 6.4 Model ladder

Grow only when residuals are systematic:
kinematics → +motor → +lag/latency → +swing → +friction/gravity asymmetry →
+region correction / residual model.

### 6.5 Stopping and guardrails

- Stop when held-out RMS error is below tolerance across the operating
  envelope for two consecutive runs.
- Hard iteration and time budget; if exhausted, report the best model and the
  unresolved error signature rather than claiming success.
- Keep a best-snapshot rollback; revert any repair that worsens held-out
  error.
- Validation data is never used for fitting; penalize complexity so structure
  is only added for systematic residuals.
- Pen-up, speed/workspace caps, estop, and abort on poor pose quality.

## 7. Toolpath layer (`control/toolpath.py`)

Common segment representation: start/end in mm, feedrate, pen-up/down.

- G-code parser: G0/G1/G2/G3, F feedrate, pen M-codes.
- Image/shape input: OpenCV threshold/contours/skeleton → ordered strokes;
  stroke reordering, Douglas–Peucker simplification, per-segment feedrate.
- "Best drawing" = choose the plan minimizing predicted draw time plus
  tracking error in sim, scored by the same controller used for G-code.

## 8. Classical controller (`control/classical.py`)

- Inverse kinematics maps toolpath point → cord-length targets.
- Trapezoidal velocity profile with accel/jerk limits.
- Feedforward cord rate + PID on position error.
- ZV/ZVD input shaper to cancel swing, using `angle_deg` as swing feedback.
- Tune gains and profile limits by lightweight random or CMA search in sim.

## 9. RL controller (`sim/env.py`, `control/rl.py`)

- Gym-style `PolargraphEnv`.
- Reward = path progress − position error − swing penalty − effort/jerk.
- Compact PPO in numpy (~250 lines), or optional stable-baselines3.
- Domain-randomize around fitted params and their uncertainty.
- Curriculum: stabilize → point-to-point → full paths.

## 10. Evaluation (`control/evaluate.py`)

One benchmark suite for both controllers and both modes. Metrics:

- RMS and max path error (mm)
- settling time
- swing RMS (`angle_deg`)
- overshoot
- control effort / jerk
- pen-up time

Pick a winner per mode (drawing optimization vs G-code fidelity).

## 11. Sim-to-real loop

Deploy best controller via `RealEnv` → log → re-run AutoModeler → retune →
repeat. Re-calibration is automatic after each real deployment.

## 12. Proposed file layout under `VisionGraph/`

    interfaces.py
    sim/polargraph.py
    sim/parameters.py
    sim/env.py
    sim/automodeler.py
    sim/identification.py
    control/toolpath.py
    control/classical.py
    control/rl.py
    control/evaluate.py
    tools/auto_calibrate.py
    tests/test_sim_*.py

## 13. Build order

1. Shared interface + safety wrapper
2. Kinematic sim from W×H
3. Auto-experiment runner + logging
4. Staged fitting + held-out validation + residual classifier
5. Repair table + model ladder + rollback/budget loop  ← self-building core
6. Toolpaths + evaluation suite
7. Classical controller + tuning
8. RL + comparison
9. Sim-to-real iteration

Rationale: prove the self-building sim and its metrics before adding
controllers or dynamics.

## 14. Tests

Follow the existing `unittest` style (`README.md:119`). Add:

- forward/inverse kinematics round-trip
- integrator stability at fixed dt
- observation latency and dropout behaviour
- command-format parity between `SimEnv` and `RealEnv`
- toolpath parser and stroke-ordering correctness
- residual classifier → repair action mapping
- held-out validation and rollback with a synthetic "machine"

## 15. Risks / honest caveats

- "Exactly like the physical machine" is unattainable from code alone;
  auto-calibration plus repeated real validation is what closes the gap.
- No encoders means control is camera-in-the-loop, so modeling latency,
  noise, and dropouts matters more than perfect motor internals.
- The simulated `angle_deg` must be validated against the real camera's
  measurement; otherwise swing feedback is misleading.
- A kinematic-only sim cannot teach swing suppression; keep the `dynamics`
  flag so the two can be compared directly.
- The self-building loop must stay bounded: without a budget and held-out
  validation it would overfit or run forever.
