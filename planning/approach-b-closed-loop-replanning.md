# Approach B Spec & Plan — Closed-Loop Replanning

**Project:** Effect of Continual Planning on Robotic Arm Catching Robustness (6.4210, Team 4.4)
**Weeks:** 2–3 (of 2026-11-02 and 2026-11-09), toward the final deadline of 2026-12-02
**Source:** `6.4210 project planning.pdf`, §4 "Approach B (closed-loop replanning)"
**Builds on:** `planning/approach-a-predict-once.md` (the shared layer),
`planning/week1-environment-and-constraints.md`, `results/catch_feasibility.md`
**Status:** Built through step 9 of §14 on branch `approach-a`. Results are in `results/approach_b.md` and `results/replan_timing.md`. The robustness sweep is run, for the
moving-start versions: `results/robustness.md`.

**Result, 2026-11-05.** As written here, at the nominal cell B catches 90 of 100 against A's 68 and the baseline's 44, with 22 only B and none only A (p = 5e-7), and 91 against 70 on
tosses the cell wasn't chosen from. The gap is 0, 22 and 56 tosses at σ_p = 0, 2 and 5 mm. Measuring it changed the design in the ways below, and most of the
sections that follow are left as written for the record.

**The headline is now a moving start, chosen 2026-11-05 after seeing it, so the team should agree to it.** The arm starts on the baseline's plan, which needs no measurements, so
it could really be computed before the toss, and no solve after that is free (`replan.MovingStartPolicy`). The matching A follows the same plan and makes one charged correction from
the same data (`replan.OneCorrectionPolicy`). B catches 87 to A's 62 at the nominal cell, with 25 only B and none only A (p = 6e-8), and 83 to 24 at 5 mm. This replaces §2's "first
solve is A's" and §3's timeline: with replanning off, B is now the baseline rather than A, and the equivalence test checks that. `results/approach_b.md` compares the versions, and
`results/robustness.md` has how the headline holds up.

- **Replanning is at about 22 Hz with a 45 ms delay, not 50 Hz.** Warm-started replans take longer than §9 expected. So they use 12 knots (§11.2, step 3), and
  one solver works at a time, so the period equals the delay: a 20 ms period with a 45 ms delay isn't something a real system could do. `replan.py` and
  `results/replan_timing.md` have the details.
- **§8 rule 1 is wrong when there's a delay, and rule 2 doesn't follow from it.** A plan seeded at *t* from the commanded state and adopted at *t* + delay disagrees with the
  old plan, which has moved on by then. Each replan instead starts from where the current plan will be at adoption, and a plan takes over when its trajectory starts. With no delay
  this is rule 1.
- **In the version as written here, the first solve is free, as A's is**, not adopted at 0.100 + `tau_compute` as §3 has it. With a first solve charged even 45 ms, A falls from 68 to 15, so charging only B's would
  end the comparison. This is the biggest assumption behind both approaches, and the paper has to say so.
- **The deadline is on when a plan takes over**, which is what §7.3's `t_c` measures, rather than when its solve starts.
- **§2's "exactly one file" can't hold literally.** B needs hooks in the shared layer that A and the baseline ignore, recorded in A's spec as §2 asks. `replan_timing.py` and
  `replan_trials.py` are analysis scripts, as `catch_timing.py` and `trials.py` are for A.
- **A single replan gets most of the gain.** At 2 mm one replan catches as many as any number of them, so the result supports looking again more than looking
  continually. This bears on Q7 and on how the paper describes B.

**Updated 2026-11-04:** all toss numbers now follow the 2.72 m/s toss in
`results/catch_feasibility.md`. The replan deadline moved to 0.470 s and the longer window nearly
doubles the solves per trial. Decisions in `approach-a-predict-once.md` (toss noise, nominal σ_p)
apply here too.

**Implementation notes, 2026-11-04.** From building the shared layer, see `approach-a-predict-once.md`. §10's
`T_GRIPPER` is gone: use `catch.close_command_time`, which looks up the gripper's response by command phase. §6.3's,
§7.3's and §12's accuracy figures are now at the new σ_p, and leave out the sample at release, which the fit excludes.
B3's "identical" has to mean within a tolerance, because
Drake repeats to about 1e-12 m rather than bit for bit, and the solver can turn that into about 2 mm: use
`trials.csv_differences`, and see `planning/assumptions.md`. A policy gets the commanded state as the spec says, and
`CatchController` already takes `compute_latency`. Toss noise is `simulate_catch(sigma_lat=, sigma_vz=)`. §12's predicted
collapse at σ_lat = 0.08 m/s may need a larger level: with a perfect sensor Approach A still catches 40 of 60 there.

Four results from step 9 of A's plan bear on this spec (`results/approach_a.md`):

- **The hypothesis's cell has moved**, to σ_p = 2 mm, σ_lat = 0.04 m/s, σ_vz = 0.02 m/s, with a 0.10 s window. §1, §6.3, §7.3 and §12 are
  updated. A catches 68 of 100 there, the baseline 44 and a perfect sensor 91. At the old cell A caught 16 and the baseline 80.
- **Catch motions are planned at 0.85 of rated speed**, for every approach (`catch.PLAN_SPEED_LIMIT`). §7.3's correction envelope was
  worked out at the toss's limit, so expect it to be smaller. It hasn't been re-measured.
- **`T_FIRST_SOLVE` doesn't change**, since the window stays at 0.10 s. An earlier version of this note argued against importing it, for a
  window of 0.15 s, but 0.15 s no longer works.
- **Cold solves are slow against `tau_compute` at the measured p95 (§9).** At the nominal cell A's solves took a median of 0.37 s and a
  p95 of 0.72 s (7 workers in parallel on a laptop), with 0.475 s between 0.10 s and the catch. B6 has to show that warm starts
  change this a great deal, or the latency model needs another look.

---

## 1. Purpose and Scope

The proposal defines Approach B in one sentence:

> "Refit the trajectory estimate at every control step (recursive least squares or a Kalman
> filter) and replan toward the updated intercept as data arrives, in the spirit of [1] but
> scaled down."

This is the project's thesis. Everything else — the environment, the toss, Approach A, the
baseline, the metrics — exists to put a number on whether it is true.

### The hypothesis, stated so it can fail

> At the nominal operating cell (σ_p = 2 mm, position-only observations, σ_lat = 0.04 m/s,
> σ_vz = 0.02 m/s, 30 ms latency), **Approach B achieves a higher catch rate than Approach A on
> the same 100 tosses**, and the gap is larger at larger σ_p. The gap closes toward zero as
> σ_p → 0 (nothing to correct) and as σ_lat grows past the arm's correction envelope (nothing
> *can* be corrected).

This was first written at σ_p = 5 mm and σ_lat = 0.02 m/s. The cell moved on 2026-11-05, after seeing A's and the baseline's results and
before B ran, so B's result played no part, but the paper has to say so. The grid chose it on trials 0 to 99, so both approaches also ran on
trials 100 to 199, to keep the comparison apart from the tosses it was chosen on.

That last clause matters: a result where B beats A everywhere by the same margin would suggest
the two differ for a reason other than information. **The predicted shape of the curve is part
of the claim** (§12), and `results/catch_feasibility.md` already gives us enough to predict it.

### In scope

- `RecursiveFit`: an incremental ballistic estimator (`sensing.py`)
- `replan.py`: `ReplanPolicy` — the replanning loop, warm starts, deadline, fallback
- The compute-latency model that keeps the comparison honest (§9)
- B's diagnostics: replan count, per-solve wall time, plan drift
- B's half of the Week 4 sweep

### Out of scope

- **Everything in `planning/approach-a-predict-once.md` §§5–13.** The sensor, the catch-plane
  choice, the trajectory optimizer, the gripper trigger, the toss variability, the trial harness,
  the metrics, and the seeding are shared and already specified. If B seems to need a change to
  any of them, that change goes in A's spec and applies to A and the baseline too (§2).

---

## 2. The Controlled-Comparison Contract

The proposal's contribution is "a controlled comparison of catching strategies". That is a claim
about the *experiment*, and it is cheap to break by accident. B inherits, unchanged and
bit-identical:

| Shared | Why it must not be B-specific |
|---|---|
| The scene, the toss, the release pose | Different tosses would confound the strategy with the throw |
| The per-trial seeds (`[SEED_ROOT, trial, 0]` and `[..., 1]`) | Trial *n* must be the same toss and the same noise realization for all three approaches — this is what makes the comparison paired |
| `BallSensor`, σ_p, σ_v, 100 Hz, position-only default | A sensing difference would be read as a planning difference |
| The catch plane `Z_CATCH = 0.674` m | Three approaches aiming at three planes is three experiments |
| `plan_reach`, `N_KNOTS`, `POSITION_TOL`, `CATCH_TILT`, the brake acceleration limit | The motion planner must be the same planner |
| `T_GRIPPER`, `T_LATENCY`, the close-trigger arithmetic | §10 |
| The trial record, the success definitions, the sweep cells | Same metrics or no comparison |

**B adds exactly one file (`replan.py`) and one class in `sensing.py`.** If a diff for Approach B
touches `catch.py`, `trials.py`, `environment.py`, or `toss.py` other than to register the new
policy, stop and move the change into the shared layer.

Two mechanical guards, both cheap:

- `trials.py` records `policy_calls`, `plans_adopted`, and `n_solves` for every approach, so an
  asymmetry is visible in the data rather than hidden in the code.
- A regression test runs one trial under `ReplanPolicy` with the replan deadline set to
  `T_WINDOW` and replanning disabled after the first solve, and asserts the trial record is
  **identical to Approach A's** on the same seed. If B-with-replanning-off is not A, then B and A
  differ by something other than replanning, and every number in the paper is suspect. This test
  is the single most valuable thing in this spec.

---

## 3. The Approach in One Timeline

Times are seconds after release. Compare with A's timeline (`approach-a-predict-once.md` §3).

```
 t=0.000  release; fingers open; ball in free flight
 t=0.000  recursive estimator starts; 100 Hz observations
 t=0.100  brake ends; arm at rest at q_stop
 t=0.100  FIRST SOLVE (identical to Approach A's, from identical data)
          -> plan adopted at t=0.100+tau_compute
 t=0.120  replan: refit, re-intersect, re-solve from the CURRENT COMMANDED state,
 t=0.140  replan: ... warm-started from the previous solution's tail
 t=0.160  replan: ...                 (every REPLAN_PERIOD = 0.02 s)
   ...    each adopted plan updates t_catch, p_catch, and the close-command time
 t=0.470  REPLAN DEADLINE -- the envelope is down to ~2 cm; commit to the last plan
 ~0.52    gripper close command, from the LAST estimate (not the first)
 t=0.575  ball crosses the catch plane
 +0.5 s   catch is scored
```

The first solve is deliberately identical to A's: same estimator state, same data, same target,
same optimizer. **B starts where A ends.** Everything B gains, it gains after t = 0.100 s, which
is why the diagnostics in §11.2 can attribute the gain directly.

---

## 4. Deliverables

| ID | Deliverable | Acceptance criterion |
|----|-------------|----------------------|
| **B1** | `sensing.RecursiveFit` | On the same observation stream, `RecursiveFit` after *n* samples equals `BallisticFit` over those *n* samples to within 1e-9 in `p0`, `v0`, and `cov`. Not "close" — algebraically the same estimator, so a mismatch is a bug |
| **B2** | `replan.ReplanPolicy` | A single toss-and-catch runs end to end in Meshcat with the plan visibly updating; `n_solves` ≥ 5 at nominal settings |
| **B3** | Equivalence regression test (§2) | B with replanning disabled produces a trial record identical to A's on the same seed |
| **B4** | Command continuity (§8) | Over a full catch with replanning on, the commanded joint position is continuous and the commanded velocity has no step larger than `a_max · dt`; measured peak tracking error is no worse than A's |
| **B5** | Compute-latency model (§9) | `tau_compute` is configurable, defaults to the measured p95 solve time, and a sweep over it is one configuration change |
| **B6** | Solver budget report | `results/replan_timing.md`: measured solve wall time (median/p95) vs. `N_KNOTS` and warm-start strategy, with the chosen `REPLAN_PERIOD` justified against it |
| **B7** | Nominal-cell result | 100 paired trials, B vs. A vs. baseline, with McNemar's test on the A–B pairing |

---

## 5. Architecture Delta

Only the policy changes. From `approach-a-predict-once.md` §5.3:

```python
class ReplanPolicy:
    def update(self, t_rel, est, q_cmd, qdot_cmd, plan) -> CatchPlan | None:
        if t_rel < T_FIRST_SOLVE:        # 0.100 s -- same as Approach A
            return None
        if t_rel > REPLAN_DEADLINE:      # 0.470 s -- see §7.3
            return None
        if plan is not None and t_rel - self.t_last_solve < REPLAN_PERIOD:
            return None
        return self._solve(t_rel, est, q_cmd, qdot_cmd, warm_start=plan)
```

`CatchController` already handles everything else — evaluating the plan, applying latency, the
finger command, the diagnostics. B does not get its own controller, its own simulation loop, or
its own metrics.

### Configuration

| Name | Nominal | Note |
|---|---|---|
| `T_FIRST_SOLVE` | 0.100 s | Identical to A's `T_WINDOW`, by definition, not by coincidence — import it |
| `REPLAN_PERIOD` | 0.045 s (about 22 Hz) | Set by the solver budget, §11. It equals `tau_compute`, with one solver at a time |
| `REPLAN_DEADLINE` | 0.470 s | Set by the correction envelope, §7.3 |
| `tau_compute` | 0.045 s, the measured p95 rounded up | §9, and `results/replan_timing.md` |
| `WARM_START` | `"tail"` | §7.2 |

**`REPLAN_PERIOD` is not 200 Hz.** The proposal says "every control step", and Week 1 put the
control rate at 200 Hz. A full 24-knot nonlinear trajectory optimization at 200 Hz is not
something this project can run 4500 times, and pretending otherwise in simulation — where compute
is free — would make B look better than any real system could be. §11 turns this into a measured
decision with a fallback ladder, and §9 makes the cost visible in the results rather than hidden.
The honest framing for the paper is **"replanning at about 22 Hz with a modeled 45 ms compute delay"**, not
"replanning every control step".

---

## 6. The Recursive Estimator (`sensing.RecursiveFit`)

### 6.1 Why RLS and not a Kalman filter

The proposal offers either. Choose **recursive least squares over the same linear model A uses**,
because it makes the comparison clean:

- The horizontal axes are `p0 + v0·t` and the vertical is `p0 + v0·t − ½g t²` with *g known*, so
  after moving the gravity term to the left-hand side the model is linear in the same 6
  parameters as A's (`approach-a-predict-once.md` §7.2).
- The ball is genuinely ballistic — no drag, no spin, no process noise (Week 1 §8). A Kalman
  filter with zero process noise *is* recursive least squares. Using a KF with invented process
  noise would add a tuning knob that A does not have, and any gain from tuning it would be
  indistinguishable from a gain from replanning.
- RLS after *n* samples is algebraically identical to batch LS over those *n* samples. So at
  t = 0.100 s, **B's estimate equals A's exactly** (B1), and every subsequent difference is
  attributable to data arriving after A stopped listening. That is the cleanest possible
  attribution and it is worth protecting.

Implementation: information-form updates, one rank-1 update per axis per observation.

```
Lambda <- Lambda + a a^T / sigma_p^2        a = [1, t]
eta    <- eta    + a (y - 0.5 g t^2) / sigma_p^2
(p0, v0) = Lambda^-1 eta ,  cov = Lambda^-1
```

2×2 per axis — invert in closed form, not with `np.linalg.solve`, and this costs nothing at any
rate we might want. Keep a sample count and refuse to produce an estimate below rank.

### 6.2 What a Kalman filter would buy, and when to switch

Only if the ballistic model stops being exact: if we ever add drag, model the bounce off a
fingertip, or hit observation outliers. None is in scope. If Week 3 reveals the free-flight
residuals are not white (check this — it is three lines and it validates the whole model), that
is a finding worth a paragraph, and *then* a filter with process noise is justified. Record the
residual check in `results/`.

### 6.3 Expected accuracy over time

From `feasibility.prediction_error()`, position-only, σ_p = 2 mm, 100 Hz, on the samples after release, 1σ per
horizontal axis at the catch point. These scale with σ_p, so multiply by 2.5 for 5 mm:

| Data up to | A (frozen at 0.10 s) | B (refit) |
|---|---|---|
| 0.05 s | — | 3.45 cm |
| 0.10 s | **1.15 cm** | 1.15 cm |
| 0.15 s | 1.15 cm | 0.59 cm |
| 0.20 s | 1.15 cm | 0.37 cm |
| 0.25 s | 1.15 cm | 0.25 cm |
| 0.30 s | 1.15 cm | 0.18 cm |
| 0.40 s | 1.15 cm | 0.11 cm |
| 0.45 s | 1.15 cm | 0.08 cm |

**This table is the project's hypothesis in numbers, and at the new cell it is a closer call.** At 5 mm, A's 2.87 cm sat past the ±1.6 cm
pad clearance for the whole flight. At 2 mm its 1.15 cm is inside it on each axis, though the two axes and the close timing (4.3 ms 1σ, or
1.3 cm of ball travel) combine. Measured, A catches 68 of 100, and 22 of its 32 misses have a motion, where a perfect sensor has none.
That is the room B has. Its refit halves A's error by 0.15 s — *provided* the arm can still act on what B learns, which is §7.3.

---

## 7. The Replanning Loop (`replan.py`)

### 7.1 Each replan

1. Take the current `Estimate` from `RecursiveFit`.
2. Recompute the intercept: `t_catch = time_to_height(est.p0, est.v0, Z_CATCH)`,
   `p_catch = ballistic_position(...)`. **Both change**, not just `p_catch` — see §10.
3. Re-solve from the **currently commanded** state:
   ```python
   motion = plan_reach(kin, toss_plan, q0=q_cmd, v0=qdot_cmd, t0=t_rel,
                       T=t_catch - t_rel, p_target=p_catch,
                       a_max=accel_limits(kin)["brake"],
                       guess=previous.window(t_rel, t_catch))
   ```
4. If the solve succeeds, adopt the new plan at `t_rel + tau_compute` (§9).
5. If it fails, **keep the current plan** and increment `solve_failures`. Never fall back to the
   nominal, never stop replanning — a transient solver failure should cost one cycle, not the
   trial. (Contrast with A, where a failed solve is terminal because there is no previous plan.)

### 7.2 Warm starting is what makes this affordable

`feasibility.Motion.window(t0, t1)` already exists for exactly this: it resamples the part of a
motion from `t0` to `t1` onto its own knots, as an initial guess. Since consecutive replans
differ by millimetres of target and 20 ms of horizon, the previous solution's tail is an
excellent guess and the solve should converge in a few iterations.

Measure all three and report in B6:

| Strategy | Expected |
|---|---|
| Cold (`initial_guess`, as A uses) | Slowest; the honest upper bound |
| `"tail"` — previous solution windowed to the remaining horizon | Default; expected several times faster |
| `"tail+shift"` — tail, with the terminal pose nudged by `J⁺ δp` toward the new target | Marginal gain; only if `"tail"` is not fast enough |

A warm start changes *speed*, not the problem. Verify that on a handful of trials the warm-started
and cold solutions agree to within `POSITION_TOL` — if the optimizer is landing in different local
minima depending on its guess, the warm start is silently changing B's behaviour and that belongs
in the paper.

### 7.3 The deadline: when replanning stops paying

From `results/catch_feasibility.md` §3, 76.5°, post-brake hand-off — how far the catch point can
still be moved toward −y (the binding direction, finding 7) if the new target is learned at `t_c`:

| `t_c` (s) | −y envelope | B's 1σ estimate error, σ_p = 2 mm | Verdict |
|---|---|---|---|
| 0.10 | 6.3 cm | 1.15 cm | Correction is worth far more than the error |
| 0.15 | 5.6 cm | 0.59 cm | Still paying |
| 0.20 | 5.0 cm | 0.37 cm | Still paying |
| 0.25 | 4.5 cm | 0.25 cm | Still paying |
| 0.30 | 4.2 cm | 0.18 cm | Still paying |
| 0.35 | 4.2 cm | 0.14 cm | Still paying |
| 0.40 | 4.0 cm | 0.11 cm | Still paying |
| 0.45 | 2.7 cm | 0.08 cm | Marginal |
| 0.50 | 1.2 cm | 0.07 cm | Below the pad clearance; not worth the disturbance |

> **`REPLAN_DEADLINE = 0.470 s`**, with the last useful correction around 2 cm — comparable to
> the ±1.6 cm tolerance, and past it the envelope collapses faster than the estimate improves.

Two refinements worth having:

- **Stop early when the correction is below the noise.** If `‖p_catch_new − p_catch_current‖` is
  under ~2 mm, skip the solve. At late times the estimate barely moves, and a solve that changes
  nothing still costs compute and still perturbs the command. Record `solves_skipped`.
- **Do not confuse the deadline with the horizon.** Replanning stops at 0.470 s; the *plan* still
  runs to `t_catch ≈ 0.575 s`. The arm keeps moving after the last replan.

Keep the deadline a configuration value and sweep it ({0.35, 0.47, 0.55} s) as a Week 3
sensitivity check: "replanning past 0.47 s does not help, and here is the measurement" is a
stronger paper sentence than a chosen constant.

---

## 8. Command Continuity — the Failure Mode That Will Bite

A new plan adopted mid-motion must not step the position command. The iiwa is in
`control_mode: position_only`, which **derives its velocity command from the position command**
(Week 1 §7.2). A discontinuity in commanded position is therefore an impulse in commanded
velocity, and the tracking error that follows will swamp the millimetres B is trying to win.

Three rules:

1. **Seed the solve from the commanded state, not the measured state.**
   `plan_reach` constrains `q[0] = q0` and `v[0] = v0`, so passing the current *commanded*
   `(q_cmd, qdot_cmd)` makes the new trajectory start exactly where the old one is, with the same
   velocity: C¹ continuous by construction. Passing the *measured* state instead would inject the
   tracking error into the command every 20 ms and turn the plant's own lag into a positive
   feedback loop. This is the single most likely way to make B worse than A.
2. **Evaluate the old plan right up to the adoption instant, then the new one.** With rule 1 and
   `tau_compute` handled as in §9, the two agree in value and slope at the switch.
3. **Check it, do not assume it.** B4 asks for a log of the commanded position and its numerical
   derivative across every adoption instant in a full trial. A step shows up immediately.

The acceleration *limit* is only enforced within each solve, so consecutive plans can still
produce a corner in commanded acceleration. That is acceptable (the arm has no acceleration limit
in the model anyway — `approach-a-predict-once.md` §2, item 2), but it is another reason the §7.3 "skip tiny corrections" rule is worth
having.

---

## 9. Compute Latency — the Honesty Requirement

In simulation, `plan_reach` returns instantly from the simulator's point of view: the clock does
not advance while the solver runs. A naive Approach B therefore replans with **zero delay**, which
no real system can do, and the A-vs-B gap we report would be partly an artifact of free compute.

> **Required:** a plan computed from data available at time *t* may not be acted on before
> `t + tau_compute`.

Implementation: `CatchController` already stashes an adoption time with each plan
(`approach-a-predict-once.md` §5.4, step 4). B sets it to `t + tau_compute`; A and the baseline
set it to `t` — correctly, because A's single solve happens at its decision instant and A's budget
analysis (`approach-a-predict-once.md` §14) already left 65 ms of margin for it. For strictness, run A with the same
`tau_compute` applied to its one solve as a Week 3 check; it should change nothing, and confirming
that is one line in the paper.

- **Default: the measured p95 solve wall time** from B6, rounded up to a multiple of the 5 ms
  control period. If warm-started solves come in at ~15 ms, `tau_compute = 20 ms` = one replan
  period, which is both realistic and tidy.
- **Sweep it:** {0, measured, 2×measured}. `tau_compute = 0` is the "infinitely fast computer"
  bound and is worth reporting *alongside* the realistic number — the gap between them is how much
  of B's advantage depends on compute speed, which is a genuinely interesting result for a paper
  about whether continual planning is worth it.

This is also the right place to be careful in the writing: our solve times come from a laptop
running a general-purpose nonlinear solver, not from an embedded real-time controller. Report them
as what they are.

---

## 10. Retiming and the Gripper Trigger

### The arrival time moves too

`approach-a-predict-once.md` §10 established that `dt_catch/dv_z ≈ 0.192` s per m/s, and that the
ball crosses the catch plane at 2.99 m/s against a 2 cm pad height — so **7 ms of timing error is
a whole pad height.** A freezes its `t_catch` at 0.100 s and inherits that error for the rest of
the flight. B re-estimates it on every replan, so:

- The horizon `T = t_catch − t_rel` changes between replans. `plan_reach` takes `T` as an argument,
  so this is free — but it means consecutive plans end at *different times*, and the controller
  must take `t_catch` from the currently adopted plan, not from the first one.
- The close-command time is recomputed on every adoption:
  `t_close = t_catch − (T_GRIPPER + T_LATENCY)`, with `T_GRIPPER = 0.025` s and
  `T_LATENCY = 0.030` s exactly as for A.
- **Once the close command has gone out, it is irrevocable.** The fingers are already moving. So
  the trigger must latch: after `t_close` has passed, later replans may still adjust the arm's
  path but must not retract the command. Record `t_close_final − t_close_first` per trial — the
  amount B's retiming moved the trigger is a direct, isolated measurement of the timing half of
  B's advantage.

**Do not let the deadline hide the trigger.** `REPLAN_DEADLINE = 0.470 s` and the nominal
`t_close ≈ 0.520 s` (0.575 − 0.055), so the last replan before the deadline sets the final trigger
time with 50 ms to spare. Check this ordering holds across the sweep; if a faster toss ever puts
`t_close` before the deadline, the latch rule above is what keeps it sane.

---

## 11. Computational Budget and the Fallback Ladder

### 11.1 The arithmetic

| | |
|---|---|
| Replan window | 0.100 → 0.470 s = 0.37 s |
| At 50 Hz (`REPLAN_PERIOD` = 20 ms) | 19 solves per trial |
| At 200 Hz | 75 solves per trial |
| Trials in the full sweep (B's share) | ~1500 (`approach-a-predict-once.md` §13.4) |
| At 50 Hz and 15 ms/solve | 285 ms of solver per trial, ~7 min total — fine |
| At 50 Hz and 200 ms/solve | 3.8 s of solver per trial, ~95 min — tolerable |
| At 200 Hz and 200 ms/solve | 15 s per trial, ~6 h of solver alone — not fine |

So the decision depends entirely on a number nobody has measured yet. **B6 measures it first**;
`REPLAN_PERIOD` is chosen after, not before.

### 11.2 Fallback ladder, in order

1. **Warm start from the previous tail** (§7.2). Expected to be enough.
2. **Reduce `REPLAN_PERIOD` to 20–40 ms.** 9–19 corrections across a window where the estimate
   improves from 1.15 cm to 0.08 cm is plenty; the information does not arrive fast enough to
   justify 200 Hz anyway. Note that at 100 Hz observations, replanning faster than 100 Hz cannot
   see new data at all — **the sensor rate is a hard ceiling on useful replanning**, and saying so
   in the paper is better than implying we chose 50 Hz for convenience.
3. **Reduce `N_KNOTS`** from 24 for the replans (keep 24 for the first solve). Check the solution
   quality does not degrade: compare terminal pinch-point error and peak velocity against the
   24-knot solve on a sample of trials.
4. **First-order correction instead of a full re-solve.** The catch point moves by centimetres,
   so the terminal pose can be corrected with `δq = J⁺ δp` at the terminal configuration and the
   trajectory's tail re-blended to meet it. Milliseconds instead of hundreds. This changes what B
   *is* — a local corrector rather than a replanner — so it must be described as such in the
   paper. Keep it as plan D, and if it is used, run a small comparison against full re-solves so
   the paper can say what the approximation cost.

### 11.3 Diagnostics that make the result explainable

Beyond the shared trial record (`approach-a-predict-once.md` §13.1), B records:

| Field | What it answers |
|---|---|
| `n_solves`, `solves_skipped`, `solve_failures` | Did B actually replan? |
| `solve_time_median`, `solve_time_p95` | §9 and §11 |
| `plan_drift` = ‖p_catch_final − p_catch_first‖ | **How far B moved the target after A stopped looking.** This is the causal link between replanning and the result |
| `trigger_shift` = `t_close_final − t_close_first` | The timing half of the same story (§10) |
| `adoption_times` | Continuity auditing (§8) |

If B beats A but `plan_drift` is near zero, B is winning for some other reason and the result is
not what we think it is. **Plot catch success against `plan_drift`** — that figure, more than the
headline bar chart, is what shows the mechanism rather than just the outcome.

---

## 12. Predicted Results, and What Would Falsify the Hypothesis

### Predicted shape (from §6.3 against the ±1.6 cm tolerance and §7.3's envelope)

| σ_p | A's frozen 1σ | B's final 1σ | A caught of 100, measured | Prediction |
|---|---|---|---|---|
| 0 mm | 0 | 0 | 91 | **No gap**: nothing to correct. The honest control. A's 9 misses have no motion. *Measured: B 91, no disagreements* |
| 2 mm (nominal) | 1.15 cm | 0.08 cm | 68 | **A moderate gap.** A's 22 misses with a motion are what B could recover. This row said "small gap" before A was measured, because A's 1σ is inside the clearance. *Measured: B 90, recovering all 22* |
| 5 mm | 2.87 cm | 0.21 cm | 11 | **Large gap** — A is past the tolerance edge, B is well inside. *Measured: B 67* |
| 10 mm | 5.73 cm | 0.42 cm | not run | **Likely the largest gap**; A mostly fails, B is still well inside. *Measured, moving start: A 31, B 84, 53 only B and 0 only A. The gap was largest at 5 mm (59 / 0), not here. A's correction fails to solve on 60 tosses, which stay on the baseline's plan and give 30 of its 31 catches* |
| 20 mm | 11.5 cm | 0.85 cm | not run | A fails; B limited by the 4–6 cm correction envelope, so the gap narrows again. *Measured, moving start: A 37, B 75, 40 / 2. The gap narrows, but not for the predicted reason: B stays on the baseline's plan as often as at 2 mm (13 tosses), and loses catches to its estimate, with a median predicted miss of 1.3 cm* |

A's measured column is at σ_lat = 0.04 m/s, σ_vz = 0.02 m/s and a 0.10 s window, from `results/approach_a.md`. Only the 2 mm and 0 mm
predictions were rewritten after seeing it, and all of them before B ran. The *Measured* notes were added after, from `results/approach_b.md` for the
first three rows and from `results/robustness.md` for 10 and 20 mm. Those two are the headline's moving-start versions, so A's 2 mm and 5 mm there are 62
and 24, not 68 and 11. The shape holds, a gap that opens with sensor noise and narrows again, but it peaks earlier than predicted.

And against toss variability: the gap should grow with σ_lat up to the point where the required
correction exceeds the envelope (σ_lat ≈ 0.08 m/s, where 2σ of landing shift is 9.2 cm against a
6.3 cm envelope), then collapse.

### What would falsify it, and what each outcome would mean

- **B ≈ A at every σ_p.** Either B is not actually replanning (check `plan_drift`, `n_solves`), or
  the arm cannot act on the correction (check `arm_arrival_error` against `predicted_miss`), or
  tracking error dominates both (check `approach-a-predict-once.md` §14's tracking measurement). All three are diagnosable from the
  recorded fields, which is why they are recorded.
- **B < A.** Most likely command discontinuity (§8) or a `tau_compute` large enough that B acts on
  stale plans. Both are measurable, and a genuine "replanning hurt because of latency" result is a
  *publishable* finding for a paper about real-world robustness — it should be reported, not
  engineered away.
- **B ≫ A even at σ_p = 0.** Something is not shared. Run the §2 equivalence test.

---

## 13. Risks

| Risk | Signal | Response |
|---|---|---|
| Command discontinuity at adoption | Tracking error spikes at adoption instants; B worse than A | §8 rule 1 — seed from the commanded state |
| Solver too slow for the sweep | B6 p95 > ~200 ms | §11.2 ladder |
| Solver lands in different local minima between replans | Large jumps in the planned terminal pose with a tiny target change | Warm start always; compare warm vs. cold (§7.2); if it persists, add a regularizer toward the previous terminal pose and apply the same cost to A's single solve |
| Replanning chases noise rather than signal | `plan_drift` large but success no better | The §7.3 "skip corrections under 2 mm" rule; consider a covariance-gated update using `est.cov` |
| B's advantage is an artifact of free compute | Gap collapses when `tau_compute` > 0 | That *is* the result (§9); report both |
| The deadline is in the wrong place | Deadline sweep shows success still rising at 0.47 s | Move it; it is configuration, and the sweep exists to catch this |
| Shared layer drifts | B's diff touches `catch.py` / `trials.py` | §2; the equivalence test catches the rest |

---

## 14. Work Plan

Approach B's build starts once A's shared layer is frozen (`approach-a-predict-once.md` §16,
step 10). The estimator does not depend on that, so it can start sooner.

**Week 2 remainder (by 2026-11-08)**

1. `sensing.RecursiveFit` with the B1 equivalence test against `BallisticFit`. Independent of the
   rest of A; start now. *(half day)*
2. Residual whiteness check on one noiseless free flight (§6.2) — validates the model for both
   approaches. *(1 hour)*

**Week 3 (by 2026-11-15)**

3. **B6 first:** measure `plan_reach` solve time, cold vs. `"tail"` warm start, at `N_KNOTS` ∈
   {12, 24}. Write `results/replan_timing.md`. Choose `REPLAN_PERIOD` and `tau_compute` from it.
   *(half day — do this before 4; it sets two constants 4 depends on)*
4. `replan.ReplanPolicy`: loop, warm start, deadline, failure fallback. *(1–2 days)*
5. **B3 equivalence test** (replanning off ⇒ identical to A). *(2 hours — before any results)*
6. B4 continuity audit across adoption instants. *(2 hours)*
7. `tau_compute` plumbed and swept at the nominal cell. *(2 hours)*
8. Nominal-cell run: 100 paired trials, baseline/A/B, with McNemar on the A–B pairing. Check the
   result against §12's prediction. *(half day)*
9. Deadline and `REPLAN_PERIOD` sensitivity checks. *(half day)*

**Week 4 (from 2026-11-16):** full sweep shared with A; the `plan_drift` figure (§11.3); drafting.

Ordering that matters: **3 before 4** (it sets `REPLAN_PERIOD` and `tau_compute`), and **5 and 6
before 8** — a continuity bug or a hidden asymmetry invalidates every number produced after it,
and both tests take an afternoon.

---

## 15. Open Questions

1. **What is the measured solve time, and therefore `REPLAN_PERIOD`?** Answered by B6.
   *Needed before: step 4.* **A warm 12-knot replan has a median of 22 ms and a p95 of 42 ms, so both are 45 ms.**
2. **Is the first-order corrector (§11.2 step 4) needed?** Depends on B6. If it is, the paper's
   description of Approach B changes and the proposal's "replan at every control step" becomes
   "correct at every control step, replan at 50 Hz". *Needed by: Week 3.* **No: 12 knots was enough.**
3. **Should the replan be covariance-gated** — only adopt a new target when the change exceeds the
   estimate's own uncertainty? Principled, and `est.cov` is already carried. May be
   over-engineering for a 0.37 s window. *Decide in Week 3 from whether `plan_drift` tracks the
   true error or the noise.* **Not needed: B lost no toss to A, so nothing suggests it is chasing noise.**
4. **Do we report `tau_compute = 0` as a headline or an ablation?** §9 argues ablation, with the
   measured value as the headline. *Needed by: Week 4.* **`results/approach_b.md` does that.**
5. **Does B need a different `CATCH_TILT` or terminal-pose regularizer** to keep consecutive
   solutions in the same IK branch? Only if §13's local-minimum risk materializes. *Needed by:
   Week 3.* **It didn't: warm and cold solutions agree to within 4.5 mrad.**
6. **Consecutive cycles** (proposal: "consecutive successful cycles"). Inherited open question
   from `approach-a-predict-once.md` §17 Q6, and it is *more* interesting for B, since a continual
   planner is what makes cycle *n*+1 recoverable from a sloppy catch at cycle *n*. Still
   unbudgeted. *Decide in Week 3.* **Single catches only, for now, so the team should agree to it, and the paper should say so.**
7. **Is 50 Hz replanning on a 100 Hz sensor defensible as "continual"?** §11.2 argues the sensor
   rate is the real ceiling. Make sure the paper says this plainly rather than leaving a reader
   to notice it. *Needed by: Week 4 drafting.* **It is 22 Hz, so the solver is the ceiling here rather than the sensor. And one replan gets most of the gain: `results/approach_b.md`, finding 6.**

---

## 16. Acceptance Checklist

- [x] `RecursiveFit` matches `BallisticFit` to 1e-9 on the same data (B1): bit for bit in practice
- [x] Free-flight residuals checked for whiteness; result recorded (`results/replan_timing.md`)
- [x] `results/replan_timing.md` written; `REPLAN_PERIOD` and `tau_compute` chosen from measurement (B6)
- [ ] `replan.py` is the only new file; no changes to `catch.py` / `trials.py` beyond registering the policy: not achievable, so the hooks B needs are recorded in A's spec, and A's records don't change
- [x] B catches a noiseless toss headless, and a noisy nominal toss with Meshcat running (B2). On a noiseless toss every replan is skipped because there's nothing to correct, so `n_solves` is 1 there; at the nominal cell its median is 6. No browser was connected, so nobody has watched the plan update yet: `python replan.py`
- [x] **Equivalence test passes: B with replanning off reproduces A's trial record, within `trials.csv_differences`' tolerances (B3)**
- [x] Commanded position continuous across every adoption instant; no velocity step > `a_max·dt` (B4), over all 1979 motions
- [x] Warm-started and cold solutions agree within `POSITION_TOL` on a sample of trials: at most 4.5 mrad, ending the same distance from the target
- [x] `tau_compute` configurable, defaulted to measured p95, sweepable (B5)
- [x] Close-command trigger latches once issued; `trigger_shift` recorded (§10)
- [x] `plan_drift`, `trigger_shift`, `n_solves`, `solve_failures`, solve times in every B trial record
- [x] Replan deadline sensitivity run at {0.35, 0.47, 0.55} s
- [x] Nominal cell: 100 paired trials, B vs. A vs. baseline, with McNemar's test (B7)
- [x] Result checked against §12's predicted shape, at the three levels of σ_p run; the rest waits for the Week 4 sweep
- [x] Success-vs-`plan_drift` figure produced (§11.3): `results/figures/caught_by_target_moved.png`, for the moving-start versions. B caught every toss it moved its target on, and A's catches don't fall off with how far B moved, as §11.3 expected: 14, 14, 16 and 18 of about 22 per group, with no significant trend. A's misses come from its estimate rather than from large moves: `results/robustness.md`
- [x] `planning/assumptions.md` updated: RLS over KF and why, `REPLAN_PERIOD`, `tau_compute`,
      the replan deadline, and the sensor-rate ceiling on useful replanning
