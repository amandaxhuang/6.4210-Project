# Assumptions and Constraints

What the simulation assumes, why, and where the evidence is. This is the document the paper's methods
section gets written from. Week 1 §8 listed the provisional values; this records what they became.

**Status:** started 2026-11-04 with Approach A's shared layer. Rows marked *open* need a team decision.

## The scene

| Quantity | Value | Basis |
|---|---|---|
| Arm | iiwa 14 | Proposal |
| Gripper | Schunk WSG 50, hydroelastic, pads 2 cm square | `manipulation` package |
| Ball | Radius 0.04 m, mass 0.01 kg | `environment.py`. Week 1 proposed a tennis ball at 0.035 m and 0.06 kg. Mass barely changes the toss: `results/catch_feasibility.md` |
| Drag and spin | Neglected | Proposal |
| Plant time step | 1 ms | Week 1 §8. Checked against 0.5 ms (spec Q7): 0 of 30 noisy outcomes changed, and the ball ended a median of 0.15 cm away. `results/catch_timing.md` |
| Arm collision geometry | None | The ball can pass through the arm. Checked separately for the parked arm only: `results/catch_feasibility.md` |

## The toss

| Quantity | Value | Basis |
|---|---|---|
| Speed, direction | 2.72 m/s, 76.5° up, heading −123.74° | `results/catch_feasibility.md` |
| Flight to the catch plane | 0.575 s, inside the proposal's 0.5–0.65 s | Same |
| Joint speeds | Commanded at 0.986 × limit, measured peak 0.999 × limit | The toss runs at the arm's limit, so it has no headroom for a faster throw or for noise on the commanded velocity |
| Toss noise | Added to the ball's velocity 8 ms after release, per trial: σ on each horizontal axis and on vertical | Spec §17 Q8. The commanded release can't take noise because the toss is at the joint limits. **State in the paper that the simulator adds it.** With the limits relaxed, the arm delivered commanded noise within 3% (8 tosses each at σ = 0.02 and 0.04 m/s), so the ball ends up with nearly the same velocity either way. The arm's own motion is the same every trial, a small difference whose effect isn't measured |

## The catch

| Quantity | Value | Basis |
|---|---|---|
| Acceleration limit on catch motions | Brake level, 13–23 rad/s² | The model has no acceleration limits and Drake enforces none. **State this in the paper as an assumption.** At swing-level acceleration the arm arrives far too late: `results/catch_feasibility.md` |
| Joint torque during the catch | Joint 6 reaches its limit for about 40 ms early in the motion | Drake clips torque silently. It recovers well before arrival: `results/catch_timing.md` |
| Joint speed during the catch | Measured peak 1.02 × limit on joint 2 | Drake doesn't enforce the velocity limit either. **State this in the paper.** Plans stay within 0.986 × limit; the overshoot is tracking |
| Catch tilt | Gripper within 25° of up | The nominal catch uses 23.8°, so the ball comes in about 36° off the finger axis. Spec Q5: tighter limits lose reach on noisy tosses. At 15° there is no motion for 15 of 60 tosses, and it catches 27 against 42 at 25°. `results/catch_timing.md` |
| Close latency | 30 ms, decision to command | Proposal's "small, fixed delta" |
| Close timing | Pads aimed 3 ms before the ball, through the gripper's measured response by command phase | Replaces the spec's fixed 25 ms gripper time. The pads can't arrive 20 to 29 ms after a WSG controller update: `results/catch_timing.md` |
| Catch definition | Both fingers touching the ball and grip force above 1 N, continuously for 0.5 s from when the fingers stop | Spec §13.3, and the headline metric. **This counts a ball perched on the pads' edges**, which includes the default noiseless catch, so every table also reports on-pad catches, where the ball's center ends within 1 cm of the pad's center along both axes. State both in the paper. The spec says only "non-zero" force; 1 N is this code's choice, and it sets the early edge of the timing window: `results/catch_timing.md` |
| Single catches | Approaches are compared on single catches | **A departure from the proposal's consecutive cycles, so the team should agree to it and the paper should say so.** A multi-cycle demo for the video stays possible, but whether a perched ball survives the arm moving hasn't been tested |
| Solve time | Approach A and the baseline are charged none: a plan takes effect at the tick it is decided | Spec §5.4, and §9 of B's spec. Cold solves took a median of 0.42 s and a p95 of 0.64 s of wall clock (7 workers in parallel on a laptop). **State this in the paper.** With solves that slow, a real system couldn't plan this way at either window |

## Sensing

| Quantity | Value | Basis |
|---|---|---|
| Observation mode | Position only is the headline | Spec §6.2. Changed from Week 1's position-and-velocity default, because a clean velocity channel measures the sensor rather than the strategy |
| Rate, noise | 100 Hz, σ_p swept | Spec §6.3 has 5 mm as nominal. *Open:* the nominal cell, from `results/approach_a.md` |
| Fit window | 10 samples, 0.01 to 0.10 s after release, or 15 samples to 0.15 s | Spec §14 has 0.10 s. *Open:* 0.15 s catches more in 8 of the 9 cells compared and ties the other: `results/approach_a.md`. The sample at release is excluded because the ball is still in the fingers |
| Prediction error from that window | **2.87 cm**, 1σ per axis | Measured against the formula on the same samples. The 2.5 cm in the feasibility report and the spec counts 11 samples, including the one at release |
| When flight starts | The commanded release time | Spec §7.1. The robot knows when it opened its fingers |
| Controller rate | 200 Hz, ticking 1 ms after each 5 ms mark | So a tick sees the sensor sample from the same millisecond; on the same tick it would see the previous one |

## The driver

**Decision: keep `control_mode: position_only`.** Spec §14 asks for a change if tracking error is a
significant fraction of the tolerance. It is 0.56 cm at arrival, against measured sideways tolerances of
about ±2 cm and an estimator error of 2.87 cm at the nominal noise. It builds up while the plan decelerates into
the catch and takes about 200 ms to decay afterward, and the noiseless catch works with it. A new driver would
also change how the toss tracks, which would then need re-checking against the joint limits, so the other drivers
weren't tried. Evidence: `results/catch_timing.md`.

## Reproducibility

**Outcomes repeat. Measurements repeat to within a tolerance, not bit for bit.** There are two causes:

- **Drake.** Running the same simulation twice gives one of two results at random, even within one process and with
  no controller or solver, differing by up to about 1e-12 m. The source is inside Drake's simulation and wasn't
  pinned down further: it isn't the floating-point mode, and it doesn't need ball contact.
- **The solver magnifies that.** SNOPT gives exactly the same answer for exactly the same inputs. But moving the
  target by as little as 1e-14 s can send it to a different solution, 1 to 5 mrad away along the path, though the end of
  the motion stays pinned. The fit repeats only to about 1e-13, so this happens on some re-runs, even with no noise at
  all. The baseline has no fit, and a noiseless baseline repeated exactly in 5 re-runs.

Six cells of the grid were run twice, 600 trials in all:

| What | Largest difference |
|---|---|
| Any outcome (caught, on pad, contact, solve) | None changed, and every cell's counts matched |
| What the fit saw (predicted miss, timing error) | 1e-13 |
| Where the arm and the ball met (miss, arrival error) | 0.21 mm |
| Where a held ball ended | 0.29 mm |
| Where a dropped ball came to rest | 1.24 cm |

So `trials.csv_differences` compares outcomes exactly, what the fit saw within 1e-9, what follows the solve within
1 mm, and doesn't compare where a dropped ball rests. `python trials.py repeat` runs the check. Those tolerances were set
after seeing this data, so the 600 trials show what the spread is rather than testing them independently.

A trial sitting right on the edge of an outcome could still flip. None did in 600, which bounds that at about 0.5% of
trials at 95% confidence, small against the roughly ±10 points of a 100-trial cell. Rounding the target before the solve
might remove most of the spread, but it wasn't tried.

- Tests compare within 1e-9 up to the solve, and with the tolerances above after it. An earlier version compared
  Approach A's whole run at 1e-9 and passed or failed at random.
- A4's "bit-identical" holds within 1e-9 for the flight and for the whole baseline run. It can't hold for Approach A's
  arm after the solve.
- A5's "reproduces the CSV exactly" can't either; the table above is what it means here.
- The spec's risk table anticipated this: report means with confidence intervals, which the results do.
