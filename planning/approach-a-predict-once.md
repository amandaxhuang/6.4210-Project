# Approach A Spec & Plan — Predict Once, Then Intercept

**Project:** Effect of Continual Planning on Robotic Arm Catching Robustness (6.4210, Team 4.4)
**Weeks:** 2–3 (of 2026-11-02 and 2026-11-09), toward the final deadline of 2026-12-02
**Source:** `6.4210 project planning.pdf`, §4 "Approach A (predict once, then intercept)" and
"Reference baseline"
**Builds on:** `planning/week1-environment-and-constraints.md`, `results/catch_feasibility.md`
**Companion:** `planning/approach-b-closed-loop-replanning.md`
**Status:** Draft for team review

**Updated 2026-11-04:** all toss numbers now follow the 2.72 m/s toss in
`results/catch_feasibility.md`, released at 0.77 m. Three things changed enough to need a team
decision: how toss noise is applied (§12), the nominal σ_p (§14), and gripper timing, which is now
tighter than the pad height (§10).

**Implementation notes, 2026-11-04.** Steps 1 to 9 of §16 are built on branch `approach-a`. Measuring them
changed a few things below, which are left as written for the record. The evidence is in
`results/catch_timing.md`, `results/approach_a.md` and `planning/assumptions.md`.

- **The close trigger replaces §10's arithmetic.** `T_GRIPPER` is gone. The trigger looks up the gripper's
  response by command phase and aims the pads 3 ms before the ball (`catch.close_command_time`).
- **A's prediction error at 0.10 s is 2.87 cm, not 2.5 cm**, with a measured 1σ timing error of about 11 ms.
  §14 excludes the sample at release, but `prediction_error()` counts it. A1 is tested on the same 10 samples.
  §14's error budget grows accordingly.
- **The ball's real tolerances differ from §2's.** Sideways tolerance depends on the close timing, and the
  timing window is about 22 ms rather than 7 ms, but only 3 ms of that holds the ball between the pads; the rest
  perches it on their edges. §13.3 stays the headline, with a 1 N grip threshold and perches counted, and every table
  reports on-pad catches beside it. See `results/catch_timing.md`.
- **"Bit-identical" (A4, A5) isn't achievable.** Drake repeats to about 1e-12 m, and the solver can turn that into about
  2 mm. Over 600 repeated trials no outcome changed, so outcomes compare exactly and measurements within a tolerance:
  `planning/assumptions.md`.
- **Q2 is answered:** the driver stays `position_only`.
- **Q8 is answered:** toss noise goes onto the ball's velocity 8 ms after release, so §12's discards and
  `toss_infeasible` don't arise. A4 passes.
- **At the nominal cell of §13.4, the baseline beats A**, 80 to 16 of 100, because the one-shot fit's error exceeds the
  toss noise. σ_lat = 0.08 m/s doesn't defeat the arm as §12 expected: with a perfect sensor A catches 40 of 60. See
  `results/approach_a.md`.
- **Catch motions are planned at 0.85 of rated speed** (`catch.PLAN_SPEED_LIMIT`), beyond what §9 asks for. At the toss's 0.986
  they relied on speeds the arm isn't rated for, because Drake doesn't enforce the rating: at one cell, 61 of 68 catches went more than
  5% over. This applies to every approach. The shortest motion to the nominal catch is then 0.436 s rather than §14's 0.410 s.
- **§14's expectation about 0.15 s holds at that limit.** 0.15 s loses to 0.10 s in all six cells compared. Q3's cell is still to be
  chosen: `results/approach_a.md` has the grid and recommends σ_p = 2 mm, σ_lat = 0.04 m/s, 0.10 s.
- **Q5 and Q7 are answered:** the tilt limit stays at 25°, and 1 ms is small enough. See `results/catch_timing.md`.
- **Q6: single catches only.** This departs from the proposal, so the team should agree to it, and the paper should say so.

---

## 1. Purpose and Scope

The proposal defines Approach A in two sentences:

> "Fit a ballistic trajectory by least squares over an early window of noisy measurements,
> compute one intercept pose and time, and execute one planned motion. Provides a reference
> for the closed-loop replanning."

This spec covers that, plus the open-loop **reference baseline** from the same section, because
the two share every line of code except one number (the length of the observation window).

**A also owns the shared layer.** Approach A is built first, so everything both approaches need
— the noisy ball sensor, the estimator, the catch-motion planner, the gripper close trigger, the
trial harness, and the metrics — is specified and built here. Approach B then replaces exactly
one object (the policy, §5.3) and nothing else. That is what makes the comparison controlled, so
the division is deliberate and should not drift: **if Approach B needs a change outside
`replan.py`, that change belongs in this spec and applies to A and the baseline too.**

### In scope

- Noisy observation of the ball (`sensing.py`)
- One-shot batch least-squares ballistic fit (`sensing.py`)
- Intercept point/time selection and the catch motion (`catch.py`)
- Gripper close trigger with modeled latency (`catch.py`)
- The three-way policy interface: baseline, A, B (`catch.py`)
- Per-trial toss variability (`catch.py` or `toss.py` extension — see §12)
- The trial harness, metrics, and seeding (`trials.py`)

### Out of scope

- Approach B's recursive estimator and replanning loop — `planning/approach-b-closed-loop-replanning.md`
- Repeated toss–catch cycles ("consecutive successful cycles"). See §17, Q6: this needs a
  second toss from the catch pose, which is a separate release-pose problem. **Stretch goal.**
- The paper, the video, and the final sweeps (Week 4+)

---

## 2. Where This Sits: What Already Exists

Unlike Week 1, most of the physics is done. Read this before writing code, because the spec
below assumes it.

| File | What it gives us | Status |
|---|---|---|
| `environment.py` | iiwa14 + WSG 50 + floor + ball scene; `make_scenario()`, `place_ball_in_gripper()`; the `PINCH` point and finger constants | Working |
| `toss.py` | `plan_toss()` → a `TossPlan` (joint trajectory, finger schedule, release state); `ballistic_position()`, `time_to_height()`, `catch_point()`, `catch_pose()` (IK), `ball_contacts()`, `arm_slice()`, `gripper_frame()`; `simulate()` → a `TossLog` with measured joint speeds and torques, `free_flight()`, `arm_limits()` | Working at 76.5° |
| `feasibility.py` | `Kinematics`, `Motion`, `plan_reach()` (trajectory optimization to a catch pose), `min_time()`, `prediction_error()`, `accel_limits()` | Working, offline only |
| `release_search.py` | The search that found the release pose; `results/release_search.md` has its tables | Offline only |
| `results/catch_feasibility.md` | The timing and error budget this spec is built on | Current |

**Note the layout divergence.** The Week 1 spec proposed a `tosscatch/` package with a
`scripts/` directory. The repo instead grew flat top-level modules (`environment.py`, `toss.py`,
`feasibility.py`) that import each other. The code is the source of truth; **this spec follows
the flat convention.** If we ever want the package layout, it is a single mechanical move and
should be done in one commit, not drifted into.

### The key numbers, from `results/catch_feasibility.md`

All at the current operating point: `TOSS_SPEED = 2.72` m/s, `TOSS_ANGLE = 76.5°`,
`TOSS_HEADING = -123.74°`, `CATCH_DROP = 0.1` m, `SWING = 0.5` s, `BRAKE = 0.1` s,
`VELOCITY_MARGIN = 0.986`.

| Quantity | Value |
|---|---|
| Release (pinch) point | `[0.800, 0, 0.774]` m |
| Release velocity | `[-0.353, -0.528, 2.645]` m/s |
| Apex above release | 0.357 m (peak 1.130 m planned, 1.128 m simulated) |
| Catch point (nominal) | `[0.597, -0.304, 0.674]` m |
| Ball reaches catch plane | **0.575 s** after release |
| Ball speed at catch plane | 2.99 m/s, downward |
| Brake ends | 0.100 s after release |
| Earliest the arm can arrive at rest | **0.510 s** after release (brake-level acceleration) |
| Slack | **0.065 s** |
| Gripper: command → pads at ball width | 20–28 ms (3 ms to start moving; 8 ms spread from the WSG controller's 50 ms update) |
| Lateral catch tolerance | **±1.6 cm** — pad faces open to 0.112 m (`FINGER_OPEN` 0.107 + 0.005 pad offset) against an 0.08 m ball |
| Vertical catch tolerance | ~2 cm (pad height) |
| Correction envelope in the tightest direction (−y) at 0.10 s | 6.3 cm (4.5 cm at 0.25 s, 4.2 cm at 0.30 s, 4.0 cm at 0.40 s, 1.2 cm at 0.50 s) |
| Simulated catch point vs planned | 1.4 cm, repeatable, mostly along +x |

### Two things that must be recorded in the paper

1. **The toss runs at the joint velocity limits.** Measured joint speeds peak at 0.999 × limit, with
   the commanded cap at 0.986. That is what puts the flight inside the proposal's window: 0.575 s to
   the catch plane against 0.5–0.65 s, from a 2.65 m/s vertical release (Week 1 derived 2.45–3.19 m/s).
   Two costs: no headroom for a faster toss, which is why toss noise is a problem (§12), and a toss
   13.5° off vertical that drifts about 0.37 m back toward the base before the catch.
2. **The catch relies on accelerations the model does not limit.** `results/catch_feasibility.md`
   finding 6: under the swing's acceleration (≈2·v_max/s) the arm arrives at 0.887 s, far too late;
   under the brake's (≈10·v_max/s, 13–23 rad/s²) it arrives at 0.510 s. The iiwa14 SDF has no
   acceleration limits and Drake does not enforce any. Torque is different: the simulator clips it
   at the effort limits, and a 0.05 s brake hit them, which is why `BRAKE` is 0.1 s. **Every catch
   motion this spec plans must carry the brake acceleration limit explicitly**
   (`feasibility.accel_limits()["brake"]`), and the paper must state it as an assumption.

---

## 3. The Approach in One Timeline

Times are seconds after release (release is at t = 1.00 s of simulation time: `SETTLE` 0.5 +
`SWING` 0.5).

```
 t=0.000  release; fingers open; ball in free flight
 t=0.000  estimator starts collecting observations at 100 Hz
 t=0.100  brake ends; arm at rest at q_stop
 t=0.100  DECISION INSTANT (t_window)
          - batch least-squares fit of (p0, v0) from the 10 observations so far
          - intersect the fitted parabola with the catch plane -> (t_catch, p_catch)
          - solve ONE trajectory optimization: rest at q_stop -> rest at p_catch at t_catch
          - freeze the close-command time t_close = t_catch - (close time + latency)
 t=0.100  arm begins the catch motion; NOTHING is recomputed after this instant
 ~0.52    gripper close command issued at t_close
 t=0.575  ball crosses the catch plane
 +0.5 s   catch is scored (ball still held, non-zero grip force)
```

The baseline is the same picture, also deciding at t = 0.100, with **no observations
used at all**: it goes to the nominal catch point computed from the nominal toss plan.

---

## 4. Deliverables

| ID | Deliverable | Acceptance criterion |
|----|-------------|----------------------|
| **A1** | `sensing.py`: `BallSensor` (noisy, rate-limited) and `BallisticFit` (batch LS) | Unit test: with σ = 0, the fit recovers the true release state to < 1 mm / < 1 mm·s⁻¹; with σ_p = 5 mm and a 0.10 s window, the empirical 1σ catch-point error over 1000 draws matches `feasibility.prediction_error()` to within 10% |
| **A2** | `catch.py`: `CatchController` with the three-policy interface, `OpenLoopPolicy`, `PredictOncePolicy` | A single toss-and-catch runs end to end in Meshcat; `HEADLESS=1` run prints a per-trial record |
| **A3** | `catch.py`: gripper close trigger with configurable latency | Sweep of the trigger lead shows a success peak, and the peak sits within 10 ms of the §10 prediction |
| **A4** | Toss variability (§12) | With `sigma_toss = 0`, two runs at different seeds are bit-identical; with σ > 0, the landing point spread matches the commanded σ within 15% |
| **A5** | `trials.py`: one-trial runner, metrics record, seeded sweep driver | 100 trials run headless and write `results/approach_a.csv` plus a summary `results/approach_a.md`; re-running the same seeds reproduces the CSV exactly |
| **A6** | Nominal-condition result for A and the baseline | ≥ 50% catch rate and ≥ 90% contact rate for A at σ_p = 0, σ_toss = 0 (the proposal's baseline goal) |

---

## 5. Architecture

### 5.1 New modules

```
sensing.py    BallSensor, Observation, BallisticFit (batch LS), RecursiveFit (stub for B)
catch.py      CatchPlan, CatchController, OpenLoopPolicy, PredictOncePolicy,
              plan_catch_motion(), close_command_time()
replan.py     [Approach B] ReplanPolicy -- the ONLY file B adds
trials.py     TrialConfig, TrialResult, run_trial(), sweep(), summarize()
```

### 5.2 The diagram

Built in `catch.py`, extending `toss.build_toss_system()` rather than copying it:

```
                        +-----------------+
   station.body_poses --|                 |
station.body_spatial_   |   BallSensor    |-- obs (abstract: Observation) --+
     velocities --------|   (100 Hz)      |                                |
                        +-----------------+                                |
                                                                           v
                                              +--------------------------------------+
                                              |          CatchController             |
                                              |  (200 Hz discrete update)            |
                                              |   state: estimator, CatchPlan,       |
                                              |          t_close                     |
                                              |   policy: baseline | A | B           |
                                              +--------------------------------------+
                                                   |                      |
                                      iiwa.position|          wsg.position|
                                                   v                      v
                                              station input ports
```

**Both plant output ports are available.** `MakeHardwareStation` exports every
`MultibodyPlant` output port (verified in the installed `manipulation/station.py`, line ~1081),
so `station.GetOutputPort("body_poses")`, `"body_spatial_velocities"` and `"contact_results"`
all exist on the station. No modification to the station is needed.

### 5.3 The policy interface — the heart of the controlled comparison

`CatchController` contains **all** the machinery: it holds the toss plan, follows
`plan.q_traj` until the hand-off, runs the estimator, drives the fingers, and applies the
latency. It delegates exactly one decision:

```python
class CatchPolicy(Protocol):
    def update(self, t_rel: float, est: Estimate | None,
               q_cmd: np.ndarray, qdot_cmd: np.ndarray,
               plan: CatchPlan | None) -> CatchPlan | None:
        """Return a new CatchPlan to adopt, or None to keep the current one.

        t_rel      seconds since release
        est        the estimator's current state, or None if it has no fit yet
        q_cmd      the position currently being commanded (NOT measured -- see B's spec §8)
        qdot_cmd   its derivative
        plan       the plan being followed, or None before the first decision
        """
```

- `OpenLoopPolicy` returns a plan once, at `t_rel >= BRAKE`, ignoring `est` entirely and using
  the nominal catch point from `toss.catch_point(toss_plan)`.
- `PredictOncePolicy` returns a plan once, at `t_rel >= T_WINDOW`, from `est`, then `None` forever.
- `ReplanPolicy` (Approach B) returns a plan on every tick until its deadline.

Three policies, one controller, one simulation path. A difference in the results can then only
come from the policy.

**Guard this in code, not just in prose:** `CatchController` asserts that the policy returned a
plan by `t_rel = DECISION_DEADLINE` and records `policy_calls` and `plans_adopted` in the trial
record, so an accidental asymmetry shows up in the data.

### 5.4 What `CatchController` does on each 200 Hz tick

1. Read the latest `Observation` (the sensor publishes at 100 Hz, so every other tick repeats).
2. Feed it to the estimator **only if the ball is in free flight** (§7.1).
3. Call `policy.update(...)`.
4. If a plan came back, stash it in abstract state, together with the time it may first be
   *acted on* (`t + compute_latency`, §B9 — zero for A and the baseline by construction, since
   A's single solve happens at the decision instant and is then static).
5. Evaluate the current command:
   - before the hand-off: `toss_plan.q_traj.value(t)`
   - after: `plan.q_traj.value(t)`, or hold `q_stop` if no plan exists yet
6. Evaluate the finger command: `FINGER_CLOSED` → `FINGER_OPEN` at release → `FINGER_CLOSED`
   once `t >= t_close + latency` (§10).

---

## 6. The Observation Model (`sensing.py`)

### 6.1 `BallSensor`

A `DTSystem` (from `utils/dtsystems.py`) at `OBS_RATE = 100` Hz:

```python
@dataclass(frozen=True)
class Observation:
    t: float                  # simulation time of the sample
    p: np.ndarray             # noisy ball position, world frame (3,)
    v: np.ndarray | None      # noisy ball velocity, or None in POSITION_ONLY mode
```

- Input: the station's `body_poses` and `body_spatial_velocities` ports (abstract-valued).
- Output: an abstract `Observation`.
- Noise: `p = p_true + N(0, σ_p² I)`, `v = v_true + N(0, σ_v² I)`, independent per axis and per
  sample, drawn from a `np.random.Generator` seeded per trial (§13.2).

Two rules that matter more than they look:

- **σ_v is a sensor specification, not a finite difference of σ_p.** Week 1 §8 derived that
  differencing two 5 mm readings at 100 Hz gives σ_v ≈ 0.71 m/s, a quarter of the release speed
  — not what a tracker delivers. Keep them independent.
- **Sample the noise in the sensor, once per sample, not at read time.** The controller reads
  the sensor port at 200 Hz and must see the *same* 100 Hz observation twice, not two draws.

### 6.2 Observation mode: position-only is the headline

`results/catch_feasibility.md` finding 8 is the reason this matters:

> A prediction from the first 100 ms is off by 2.5 cm (1σ per axis), more than the gripper's
> ±1.6 cm. [...] With velocity readings the 100 ms prediction is already 0.8 cm.

Giving every approach a clean velocity channel measures mostly the sensor, not the strategy.
Week 1 §8 left this open; the feasibility numbers close it:

> **Decision: `POSITION_ONLY` is the default observation mode and the headline configuration.**
> `POSITION_AND_VELOCITY` stays available behind the same flag and is run as a secondary
> condition if the sweep budget allows, to show that the A–B gap shrinks when the sensor does
> the estimator's work. Both go in the paper; position-only is the one the claim rests on.

This is a change from Week 1's provisional default and should be noted as such in
`planning/assumptions.md`.

### 6.3 Noise sweep

| Parameter | Nominal | Sweep |
|---|---|---|
| σ_p | 5 mm | {0, 2, 5, 10, 20} mm |
| σ_v | n/a (position-only) | {0.02, 0.05, 0.1, 0.2} m/s, only in the secondary mode |
| Observation rate | 100 Hz | fixed |

σ_p = 5 mm was chosen to put A at the pad clearance. At this toss its 100 ms prediction error is
2.5 cm, past the ±1.6 cm clearance, so A should struggle there and B should have room; §14 asks
whether to lower it. σ_p = 0
is the control that the proposal's baseline goal is stated against; σ_p = 20 mm should break A
outright.

---

## 7. The One-Shot Estimator (`sensing.py`)

### 7.1 When flight starts

The estimator must know what `t = 0` means. Three options, in order of preference:

1. **Use the commanded release instant** (`plan.t_release`). The robot opened its own fingers;
   knowing when it did is not cheating. **Default.**
2. Detect it from the observations (vertical acceleration settles to −g). Needs ~3 samples =
   30 ms, which is 30 ms off the front of a 65 ms budget. Keep as a flag for the paper's
   "what if the catcher does not know the thrower" discussion.
3. Detect it from contact results. Not a sensor a real catcher has. Do not use.

The actual release lags the commanded one by a few ms. This does not bias the fit: least squares
estimates `(p0, v0)` *at the anchor time*, whatever that is, and the extrapolation to `t_catch`
is unaffected as long as the anchor is consistent. State this in the paper rather than
pretending the anchor is exact.

### 7.2 The model

Gravity is known, so after subtracting it the whole fit is **linear in 6 parameters**:

```
p(t) = p0 + v0 * t + 0.5 * g * t^2 ,   g = [0, 0, -9.81]

per axis:   [p_obs(t_i) - 0.5 g_axis t_i^2]  =  [1  t_i] [p0_axis; v0_axis]
```

Stack the rows, weight by 1/σ_p (and the velocity rows by 1/σ_v when they exist), and solve the
2×2 normal equations per axis in closed form. This is exactly the estimator
`feasibility.prediction_error()` already models, so the analytic error numbers in
`results/catch_feasibility.md` apply to the implementation without reinterpretation. **That
correspondence is an acceptance criterion (A1), not a coincidence to be assumed.**

```python
@dataclass(frozen=True)
class Estimate:
    p0: np.ndarray        # (3,) fitted release position
    v0: np.ndarray        # (3,) fitted release velocity
    cov: np.ndarray       # (6,6) parameter covariance -- carried for B and for diagnostics
    n_obs: int
    t_anchor: float
```

Carry `cov` from the start even though A does not use it: it costs nothing, it is what lets the
paper plot prediction uncertainty against time, and B's deadline rule (§B7) may want it.

### 7.3 The window

`T_WINDOW = 0.10` s → 10 observations (release sample excluded; see §14 for why the first sample
is at t = 0.01 s). Chosen from the timing budget in §14, not by taste.

**Degenerate cases.** With fewer than 2 position samples per axis the fit is rank-deficient
(`feasibility.prediction_error()` returns `inf`). The policy must then fall back to the nominal
catch point and record `fit_rank_deficient = True` in the trial record. This can only happen if
`T_WINDOW` is misconfigured, but a silent `LinAlgError` 80 trials into a sweep is worth one
`if`.

---

## 8. Intercept Selection (`catch.py`)

Given an `Estimate`:

```python
t_catch = time_to_height(est.p0, est.v0, Z_CATCH)     # the later root; already in toss.py
p_catch = ballistic_position(est.p0, est.v0, t_catch)
```

- **The catch plane is fixed, shared, and nominal.** `Z_CATCH = p_release_nominal[2] - CATCH_DROP
  = 0.674` m, computed once from the nominal toss plan and held constant across every trial and
  every approach. It must *not* be recomputed from the estimate, or the three approaches would
  be aiming at different planes and the comparison would stop being controlled.
- **Reject unreachable intercepts.** If `time_to_height` has no real root (the fitted `v_z` is
  so wrong the parabola never descends to the plane) or `t_catch` is outside
  `[t_now + T_MIN_MOTION, 1.0]`, fall back to the nominal catch point and set
  `intercept_rejected = True`. At σ_p = 20 mm this will fire; it should be counted, not crashed
  on.
- Clamp `p_catch` to the reachable set only through the planner's own failure (§9), not with an
  ad-hoc clamp here — an unreachable target is a *result*, and silently moving it would hide
  exactly the failure mode we are measuring.

---

## 9. The Catch Motion (`catch.py`)

Reuse `feasibility.plan_reach()`. It already does precisely what is needed: a trajectory
optimization over `N_KNOTS` knots from `(q0, v0)` at `t0` to rest at `t0 + T` with the pinch
point within 3 mm of a target and the gripper within `CATCH_TILT = 25°` of up, holding the pinch
point at or below the ball throughout, under velocity, position, and acceleration limits.

```python
motion = plan_reach(kin, toss_plan, q0=q_cmd, v0=qdot_cmd, t0=t_rel,
                    T=t_catch - t_rel, p_target=p_catch,
                    a_max=accel_limits(kin)["brake"])
```

Four things to get right:

1. **`plan_reach` lives in `feasibility.py`, which is a report generator.** Move `Kinematics`,
   `Motion`, `plan_reach`, `initial_guess`, and `accel_limits` into a new `planning_core.py` (or
   into `catch.py`) and have `feasibility.py` import them. Do this *before* writing A, not after
   B needs it too. `feasibility.py`'s report output must be unchanged by the move — rerun it and
   diff against the committed `results/catch_feasibility.md`.
2. **The acceleration limit is mandatory** (§2, item 2). `a_max=None` produces motions the
   simulated arm cannot track.
3. **Convert the `Motion` knots to a `PiecewisePolynomial`** for the controller to evaluate.
   `Motion` has piecewise-linear velocity and quadratic position, which is exactly
   `PiecewisePolynomial.CubicHermite(t, q.T, v.T)` restricted to those knots — the Hermite cubic
   reproduces it with a zero cubic term. Verify numerically against `Motion.state()` at 100
   random times (tolerance 1e-9); do not assume.
4. **Arrive at rest.** `plan_reach` already constrains `v[n] = 0`. Catching with a moving hand
   is a different (harder) problem and is not what the feasibility study measured.

```python
@dataclass
class CatchPlan:
    q_traj: PiecewisePolynomial   # absolute simulation time
    t_catch: float                # absolute sim time the ball reaches the catch plane
    p_catch: np.ndarray
    est: Estimate | None          # the estimate it was planned from (None for baseline)
    solve_time: float             # wall-clock seconds -- B needs this; A records it too
    solve_ok: bool
```

### Solve failure

`plan_reach` returns `None` when the solver fails. For A this is terminal for the trial: record
`solve_failed = True`, hold `q_stop`, and let the trial be scored a miss. **Do not retry with a
different seed or a looser tolerance** — a retry that A gets and B does not (or vice versa)
breaks the comparison. If failures are common enough to matter (> 5% of trials), that is a
finding, and the fix is a better initial guess applied identically to all three policies.

---

## 10. Gripper Close Trigger and Latency (`catch.py`)

### The budget

From `results/catch_feasibility.md` §1 and the Week 1 spec's latency parameter:

```
t_close_command = t_catch - (T_GRIPPER + T_LATENCY)

T_GRIPPER = 0.025 s   measured: 20-28 ms from command to pads at ball width
T_LATENCY = 0.030 s   the proposal's "small, fixed delta", decision -> action
```

so the command goes out about 55 ms before the ball reaches the pinch point.

### Why timing is a first-order failure mode, not a detail

The ball crosses the catch plane at **2.99 m/s downward**, and the pads are about **2 cm** tall.
A timing error of 7 ms is 2 cm of ball travel — a whole pad height. Two consequences:

- The 8 ms spread in the gripper's own response (the WSG controller's 50 ms update period means
  the command lands at an arbitrary phase) is now 2.4 cm of ball travel, **more than a pad height
  on its own**. It goes in the error budget and the paper unless A3 can shrink it: the response
  time repeats by command phase (`results/catch_feasibility.md` §1), so a lead looked up from the
  phase may work.
- **An error in the estimated `v_z` costs time, not just position.** `dt_catch/dv_z ≈ 0.192` s
  per m/s at this operating point, so a 0.05 m/s error in fitted vertical velocity moves the
  arrival by 9.6 ms = 2.9 cm of ball travel. For Approach A that error is frozen at the decision
  instant. This is a large part of what B should recover, and it is **why the trial record must
  separate horizontal miss from timing error** (§13.1) — otherwise the paper cannot say which
  one A is losing to.

### Implementation

- `t_close_command` is computed **once per plan**, from that plan's `t_catch`. For A it is frozen
  at the decision instant along with everything else.
- `T_LATENCY` is applied by the controller as a pure delay on the finger command, not folded into
  the trigger arithmetic. Keeping it a property of the plumbing means every policy pays it
  identically and a sweep over it is one configuration change.
- Sweep `T_LATENCY ∈ {0, 20, 30, 50}` ms (Week 1 proposed {0, 20, 50}; 30 added because it is the
  nominal).
- A3 asks for a sweep over the trigger lead in isolation, to confirm the success peak lands where
  this arithmetic says. If it does not, the arithmetic is wrong and everything downstream inherits
  the error.

---

## 11. The Open-Loop Reference Baseline

The proposal:

> "An open-loop catcher that uses no observations and goes to where the nominal toss should land,
> to show how much each approach gains from sensing."

`OpenLoopPolicy` is `PredictOncePolicy` with the estimate replaced by the nominal toss plan:

```python
t_catch, p_catch = toss.catch_point(toss_plan)   # from the NOMINAL plan, no observations
```

and the decision taken at `t_rel = BRAKE` (0.100 s), the earliest moment the arm is free. It
therefore gets as much motion time as any of the three — which is the point: whatever it loses,
it loses to not looking, not to being rushed.

It also gets the best possible nominal: the toss plan is what the robot *commanded*, so the
baseline is not merely open-loop, it is open-loop with perfect knowledge of the intended toss.
The real toss lands a repeatable 1.4 cm from that plan, most of the ±1.6 cm tolerance, so calibrate
that offset out first. Then with `sigma_toss = 0` the baseline should catch nearly every time — and
that is a **useful self-test of the whole pipeline**, not a boring result. Run it first: if the
baseline cannot catch a perfectly nominal toss, nothing downstream means anything.

---

## 12. Toss Variability

The proposal: "the toss is mostly upward but includes small lateral velocity variation, so the
ball does not always come back to the hand."

### Parameterization

Perturb the **commanded release velocity** before `toss.plan_toss()` solves for the release joint
velocity, and decompose it so the two failure modes stay separable:

```python
v_release = release_velocity(TOSS_SPEED, TOSS_ANGLE)
v_release[0] += rng.normal(0, sigma_lat)    # +x: toward/away from the robot
v_release[1] += rng.normal(0, sigma_lat)    # +y: sideways
v_release[2] += rng.normal(0, sigma_vz)     # vertical: shifts ARRIVAL TIME
```

Perturbing the *velocity* rather than the pose keeps the release pose, and therefore the
Jacobian, the catch-plane geometry, and the reachability picture, identical across trials. It
also means `release_joint_velocity()`'s QP re-solves each trial, which is both realistic and a
free check that the perturbed toss is within the velocity limits — it raises if not, and the
trial is discarded and recorded as `toss_infeasible` (do not silently resample: a resampled
distribution is not the distribution the paper reports).

> **Needs a decision: this does not work at the current toss.** `TOSS_SPEED` is the most this pose
> can do, so most perturbed commands are out of reach and would be discarded (§17, Q8). Number
> of 300 draws out of reach, with the same σ on all three axes:
>
> | Toss speed | Planned peak | Flight | σ = 0.01 | σ = 0.02 | σ = 0.04 | σ = 0.08 |
> |---|---|---|---|---|---|---|
> | 2.72 m/s (current) | 1.130 m | 0.575 s | 143 | 162 | 186 | 198 |
> | 2.64 m/s | 1.109 m | 0.560 s | 0 | 24 | 67 | 125 |
> | 2.55 m/s | 1.087 m | 0.543 s | 0 | 0 | 18 | 71 |
> | 2.40 m/s | 1.051 m | 0.515 s | 0 | 0 | 1 | 24 |
>
> The two options are to add the noise to the ball's velocity right after release, which keeps the
> toss at 2.72 m/s and loses nothing to discards, or to toss slower with the method above. 2.55 m/s
> covers the nominal 0.02 m/s, but the larger σ_lat levels would still lose tosses.

### Sizing, from the correction envelope

`results/catch_feasibility.md` finding 7 and §3: at 76.5°, the −y and −x envelopes are about 4–7 cm
for most of the flight, while +x and +y are 8–50 cm. **−y is the binding direction and the sweep
must be sized against it.**

- A landing displacement of δ needs δv = δ / 0.575 s.
- The envelope at the decision instant (t = 0.10 s) is 6.3 cm → δv of 0.11 m/s is the hard wall.
- Choose σ_lat so that 2σ of landing displacement stays inside it: σ_lat = 0.02 m/s gives 1.15 cm
  of 1σ landing shift, 2.3 cm at 2σ, comfortably inside 6.3 cm but comparable to the ±1.6 cm pad
  clearance — i.e. big enough to matter, small enough to be correctable. **Nominal: 0.02 m/s.**

| Parameter | Nominal | Sweep | 1σ effect at the catch plane |
|---|---|---|---|
| σ_lat | 0.02 m/s | {0, 0.01, 0.02, 0.04, 0.08} m/s | 0.57 / 1.15 / 2.3 / 4.6 cm of landing shift |
| σ_vz | 0.02 m/s | {0, 0.02, 0.05} m/s | 3.8 / 9.6 ms of arrival-time shift = 1.15 / 2.9 cm of ball travel |

σ_lat = 0.08 m/s should defeat everything including B (2σ = 9.2 cm > the 6.3 cm envelope), which
is worth having in the sweep: a condition where B stops helping is evidence that the gap we
measure elsewhere is real and bounded, not an artifact.

---

## 13. Trial Harness, Metrics, and Seeds (`trials.py`)

### 13.1 The trial record

One row per trial. Everything here is cheap to record and expensive to re-run for.

| Field | Why |
|---|---|
| `trial`, `seed`, `approach`, `sigma_p`, `sigma_v`, `obs_mode`, `sigma_lat`, `sigma_vz`, `latency` | The experimental cell |
| `caught` (bool) | Primary metric (§13.3) |
| `contact` (bool) | Secondary metric |
| `miss_horizontal` (m) | ‖(p_ball − p_pinch)_xy‖ at the moment the ball crosses `Z_CATCH` |
| `miss_vertical` (m) | pinch-point height error at the same instant |
| `timing_error` (s) | actual arrival at `Z_CATCH` − planned `t_catch` |
| `predicted_miss` (m) | ‖p_catch_estimated − p_catch_true‖ — the estimator's error alone |
| `arm_arrival_error` (m) | ‖pinch at `t_catch` − p_catch_planned‖ — the tracker's error alone |
| `solve_ok`, `solve_time`, `n_solves`, `fit_rank_deficient`, `intercept_rejected`, `toss_infeasible` | Diagnostics and fairness audit |
| `v_release_true`, `p_release_true` | What the toss actually did, for attributing failures |

`predicted_miss` and `arm_arrival_error` decompose the total miss into *estimation* error and
*execution* error. Without that split, "A missed by 2 cm" cannot be attributed, and the paper's
central claim is about estimation. Record both from trial 1.

### 13.2 Seeding — the thing that makes the comparison paired

```python
rng_toss = np.random.default_rng([SEED_ROOT, trial, 0])   # toss perturbation
rng_obs  = np.random.default_rng([SEED_ROOT, trial, 1])   # observation noise
```

Two independent streams, both derived from the trial index only — **not from the approach**. So
trial 37 is the same toss with the same noise realization for the baseline, A, and B. That buys
three things:

- Variance reduction: the approaches are compared on identical tosses, as the proposal requires.
- Paired statistics: 100 paired binary outcomes support McNemar's test, which is far stronger
  than comparing two independent proportions and costs nothing extra.
- Debuggability: "trial 37 fails under A and succeeds under B" is one Meshcat run away.

**Separate streams matter.** One shared stream would make the observation noise depend on how
many draws the toss consumed, so changing the toss model would silently change every noise
realization and invalidate comparisons with earlier runs.

### 13.3 Success definitions (fixed now, per Week 1 §8)

- **Caught (primary):** the ball is in contact with both finger pads and the measured grip force
  is non-zero, continuously for 0.5 s starting from the moment the fingers stop closing. Hold
  duration is what distinguishes a catch from a deflection. `toss.ball_contacts()` gives the
  contact side; `wsg.force_measured` gives the force.
- **Contact (secondary):** any ball↔finger contact at any point in the trial.
- **Consecutive cycles:** deferred (§17 Q6).

`AFTER = 1.5` s of post-brake simulation already covers the 0.5 s hold. Confirm it still does
once the catch motion is longer than the brake.

### 13.4 Sweep structure and runtime

A full cross of 3 approaches × 5 σ_p × 5 σ_lat × 4 latency × 100 trials is 30,000 trials. Do not.

**One-factor-at-a-time around a nominal cell** (σ_p = 5 mm, σ_lat = 0.02 m/s, σ_vz = 0.02 m/s,
latency = 30 ms, position-only):

| Block | Cells | Trials |
|---|---|---|
| Nominal | 1 | 300 |
| σ_p sweep | 4 more levels | 1200 |
| σ_lat sweep | 4 more levels | 1200 |
| latency sweep | 3 more levels | 900 |
| σ_vz sweep | 2 more levels | 600 |
| Secondary mode (position+velocity), nominal | 1 | 300 |
| **Total** | | **4500** |

**Measure the per-trial wall time in Week 3 and put it in `results/`.** The sim is ~2.6 s of
simulated time at a 1 ms step; headless that is seconds, and B adds solver time on top. Trials
are independent, so `multiprocessing.Pool` over seeds is the obvious lever — but Drake solvers
are themselves threaded, so set `OMP_NUM_THREADS=1` in the workers or the pool will fight itself.
If 4500 trials does not fit the Week 4 budget, cut *trials per cell* before cutting cells, and
report the resulting confidence intervals honestly.

---

## 14. Timing and Error Budget

### Timing: why `T_WINDOW = 0.10` s

The constraint is `T_WINDOW + T_motion(T_WINDOW) + margin ≤ t_catch`.

| Term | Value | Source |
|---|---|---|
| `t_catch` | 0.575 s | §2 |
| Minimum motion time from rest | 0.410 s | `results/catch_feasibility.md` §2: arrival 0.510 s from a 0.100 s hand-off |
| Hard upper bound on `T_WINDOW` | ≈ 0.165 s | 0.575 − 0.410 |
| **Chosen `T_WINDOW`** | **0.10 s** | leaves 0.065 s of margin for solver time, tracking error, and the fact that minimum-time motions saturate the velocity limits |

Longer windows buy accuracy (1σ falls from 2.5 cm at 0.10 s to 1.4 cm at 0.15 s, position-only)
but spend the margin that lets the motion be tracked rather than merely planned. A window shorter
than 0.10 s gains nothing, because the arm is braking until then. **Run
`T_WINDOW ∈ {0.10, 0.125, 0.15}` as a one-off sensitivity check in Week 3**, expecting failures at
0.15 s with only 15 ms of margin left, and put the three numbers in the paper — "we chose the window with evidence" is worth two sentences.

Note that the first usable sample is at t = 0.01 s, not t = 0: the observation at the release
instant is the ball still in the fingers. Start the window at the first sample strictly after
release, and verify the count is 10, not 11.

### Error budget at the nominal cell (σ_p = 5 mm, position-only, T_WINDOW = 0.10 s)

| Source | 1σ, per horizontal axis | Note |
|---|---|---|
| Estimator (A, frozen at 0.10 s) | **2.5 cm** | `feasibility.prediction_error()` |
| Toss variability, σ_lat = 0.02 m/s | 1.15 cm | §12 |
| Arrival timing from σ_vz = 0.02 m/s | 1.15 cm equivalent | 3.8 ms × 2.99 m/s |
| Gripper response spread | 1.2 cm equivalent | half of 8 ms × 2.99 m/s, unless A3 shrinks it (§10) |
| Tracking error | **unmeasured** | Week 1 §7.1–7.2; must be measured in Week 3 |
| **Tolerance to beat** | **±1.6 cm** | pad clearance |

Summed in quadrature (excluding tracking): ≈ 3.2 cm 1σ against a ±1.6 cm tolerance. **A should land
well below a coin flip at the nominal cell**, while a coin flip is the operating point that makes
the A-vs-B comparison most informative. If A comes out at 90% or at 5%, the nominal cell is in the
wrong place and should be moved (by σ_p) before the Week 4 runs, not after. Expect to lower it: at
σ_p = 3 mm the estimator term falls to 1.5 cm and the total to about 2.5 cm.

### The unmeasured term

Week 1 §7.1 and §7.2 flagged two sources of tracking error that nobody has measured: the iiwa
controller does not model the held ball, and `control_mode: position_only` derives its velocity
command from the position command. During the catch the arm is *not* holding the ball, so §7.1
mostly affects the toss (and shows up as release-velocity error, which the estimator sees and
corrects for). §7.2 affects the catch directly, since the catch motion saturates joint velocities.
The toss itself now has numbers: 11 mrad of tracking error at release, and joint speeds overshooting
by about 1.4% just after it. The catch motion is still unmeasured.

> **Required in Week 3, before any sweep:** log commanded vs. measured joint positions through
> one catch motion and report the peak pinch-point error. If it is a significant fraction of
> 1.6 cm, switch to `position_and_torque` or `InverseDynamicsDriver` (Week 1 §7.2), record the
> change in `planning/assumptions.md`, and **re-run the baseline self-test (§11) afterwards**.

---

## 15. Risks and Failure Modes

| Risk | Signal | Response |
|---|---|---|
| Tracking error eats the whole tolerance | `arm_arrival_error` ≫ `predicted_miss` | Change the driver (§14); it is a one-line scenario change by design |
| `plan_reach` fails often at perturbed targets | `solve_ok` false > 5% | Better warm start, applied identically to all policies; never a per-policy retry |
| The catch motion collides with the descending ball | Ball deflects before the catch plane | `plan_reach` already constrains the pinch point to stay at or below the ball; verify it is active for the *whole* motion when `t0 > BRAKE` |
| The arm reaches the catch point and the ball lands on the closed fingers | `contact` true, `caught` false, miss ≈ 0 | Trigger timing (§10); this is the failure A3 exists to find |
| A is too good or too bad at the nominal cell | Catch rate > 85% or < 15% | Move the nominal σ_p before Week 4, not after |
| Solver nondeterminism breaks reproducibility | Same seed, different CSV | Pin the solver and its options; if SNOPT is nondeterministic across runs, record it as a limitation and report mean ± CI rather than exact reproduction |
| The catch motion hits the torque limits | `arm_arrival_error` grows; `TossLog.arm_effort` reaches `arm_limits()` | The simulator clips torque silently, as the 0.05 s brake showed. Lower the catch's acceleration limit for all three policies, which spends the 65 ms margin |
| Gripper timing decides the catch | `contact` true, `caught` false, timing error within the 8 ms spread | Look the lead up by command phase (§10); the ball now covers a pad height in 7 ms |
| Trials too slow for 4500 runs | Measured per-trial time × 4500 > budget | Parallelize with `OMP_NUM_THREADS=1`; then cut trials per cell, not cells |

---

## 16. Work Plan

**Week 2 remainder (by 2026-11-08)**

1. Move `Kinematics` / `Motion` / `plan_reach` / `accel_limits` out of `feasibility.py`; confirm
   `feasibility.py` still reproduces `results/catch_feasibility.md` byte for byte. *(half day)*
2. `sensing.py`: `BallSensor` + `BallisticFit`, with the A1 statistical test. *(1 day)*
3. `catch.py`: `CatchController`, the policy interface, `OpenLoopPolicy`. Get the **baseline
   self-test** (§11) catching a nominal toss in Meshcat. *(1–2 days)*
4. `PredictOncePolicy` on top. *(half day)*

**Week 3 (by 2026-11-15)**

5. Gripper trigger + latency, and the A3 lead sweep. *(half day)*
6. **Tracking-error measurement (§14)** and the driver decision. *(half day — do this before 7)*
7. Toss variability (§12) and its A4 check, once §17 Q8 is decided. *(half day)*
8. `trials.py`: trial record, metrics, seeding, single-cell runner. *(1 day)*
9. Nominal-cell run, 100 trials, baseline + A. Check A against the §14 prediction; move the
   nominal cell if needed. *(half day)*
10. Hand the shared layer to Approach B — frozen from this point except for bugs. *(—)*

**Week 4 (from 2026-11-16):** full sweep, shared with B.

The ordering constraint that matters: **3 before 4, and 6 before 9.** A baseline that cannot
catch a nominal toss means the bug is in the shared layer, and finding that out after A, B, and
the harness are written costs days.

---

## 17. Open Questions

1. **Does the observation window start at release or at first detection?** Default is release
   (§7.1). Decide whether the "catcher does not know the thrower" variant is worth a paper
   paragraph. *Owner: TBD. Needed by: Week 3.*
2. **Does the driver change to `position_and_torque` / `InverseDynamicsDriver`?** Answered by the
   §14 measurement. *Needed before: the Week 4 sweep.*
3. **Is the nominal cell at the right σ_p?** §14 predicts A well below a coin flip at 5 mm. Answered
   by step 9 of §16. *Needed before: Week 4.*
4. **Do we run the secondary `POSITION_AND_VELOCITY` sweep?** Budget-dependent; the position-only
   decision in §6.2 means the paper's claim does not require it. *Needed by: Week 4 planning.*
5. **Is `catch_pose`'s `CATCH_TILT = 25°` still right once the ball may come down off-centre?**
   A tilted gripper has a smaller effective aperture in the direction of tilt. Check whether
   tightening it to 15° costs reachability. *Owner: TBD. Needed by: Week 3.*
6. **Consecutive cycles.** The proposal promises "consecutive successful cycles", which needs a
   second toss from wherever the catch ended — a different release pose, re-solved per trial.
   Currently unplanned and unbudgeted. **Decide in Week 3 whether to attempt it or to state in
   the paper that single-cycle results stand in for it.** *Needed by: Week 3.*
7. **Is 1 ms a small enough plant time step for the catch contact?** Week 1 §9 Q6 left this open
   and there is now a real catch to test it on: re-run one successful trial at 5e-4 and confirm
   the outcome does not change. *Needed by: Week 4.*
8. **How is toss noise applied?** Velocity noise on the command no longer works at the arm's limit
   (§12). Add it to the ball's velocity right after release, or toss slower, to 2.55 m/s and a
   1.087 m peak for the nominal 0.02 m/s? *Owner: TBD. Needed before: step 7 of §16.*

---

## 18. Acceptance Checklist

- [x] `plan_reach` and friends moved out of `feasibility.py`; `feasibility.py` output unchanged
- [x] `sensing.BallSensor` publishes at 100 Hz, one noise draw per sample, seeded per trial
- [x] `BallisticFit` matches `feasibility.prediction_error()` within 10% over 1000 draws (A1)
- [ ] `CatchController` runs all three policies through one code path; `policy_calls` recorded
- [x] Baseline catches a nominal, noiseless toss in Meshcat (§11 self-test)
- [x] `PredictOncePolicy` catches a nominal, noiseless toss
- [x] Close-trigger lead sweep peaks within 10 ms of the §10 arithmetic (A3): a plateau rather than a peak, which
      passes at the 1 N grip threshold but not at 0.1 N (`results/catch_timing.md`)
- [x] Tracking error through a catch motion measured and written up; driver decision made (§14)
- [x] Toss variability verified against commanded σ (A4)
- [x] `trials.py` runs 100 seeded trials headless and writes `results/approach_a.csv`
- [x] Re-running the same seeds reproduces the CSV: outcomes exactly, measurements within a tolerance by column
      (`planning/assumptions.md`)
- [x] Trial record contains `predicted_miss` and `arm_arrival_error` separately
- [x] Nominal cell: A ≥ 50% catch and ≥ 90% contact at σ_p = 0, σ_toss = 0 (proposal goal, A6): 100 of 100 for both
- [x] Per-trial wall time measured and the Week 4 sweep budget written down (`results/approach_a.md`)
- [x] `planning/assumptions.md` updated: position-only default, the toss at the joint limits with a
      0.575 s flight, the acceleration assumption, the driver decision
- [ ] Shared layer frozen and handed to Approach B
