# Assumptions and Constraints

What the simulation assumes, why, and where the evidence is. This is the document the paper's methods
section gets written from. Week 1 §8 listed the provisional values; this records what they became.

**Status:** started 2026-11-04 with Approach A's shared layer, with Approach B added 2026-11-05. Rows marked *open* need a team decision.

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
| Toss noise | Added to the ball's velocity 8 ms after release, per trial: σ on each horizontal axis and on vertical. Nominal 0.04 and 0.02 m/s | Spec §17 Q8. The commanded release can't take noise because the toss is at the joint limits. **State in the paper that the simulator adds it.** With the limits relaxed, the arm delivered commanded noise within 3% (8 tosses each at σ = 0.02 and 0.04 m/s), so the ball ends up with nearly the same velocity either way. The arm's own motion is the same every trial, a small difference whose effect isn't measured |

## The catch

| Quantity | Value | Basis |
|---|---|---|
| Acceleration limit on catch motions | Brake level, 13–23 rad/s² | The model has no acceleration limits and Drake enforces none. **State this in the paper as an assumption.** At swing-level acceleration the arm arrives far too late: `results/catch_feasibility.md` |
| Joint torque during the catch | Joint 6 reaches its limit for about 70 ms early in a noiseless motion, and for up to 163 ms on noisy tosses at the nominal cell | Drake clips torque silently. On a noiseless catch it recovers well before arrival, and noisy arrivals are as good, 0.49 cm at the median: `results/catch_timing.md` |
| Where a plan stops | 5.2 mm from its target, at (+3, +3, +3) mm, in every plan of every approach | That is the corner of `POSITION_TOL`'s +-3 mm box: the smoothness cost pushes the end as far as the constraint allows. With about 0.7 cm of tracking error, the reported arrival error is 0.3 to 0.8 cm. A 1 mm tolerance made every approach worse, so it stays: `results/approach_b.md` |
| Joint speed during the catch | Planned at 0.85 × rated (`catch.PLAN_SPEED_LIMIT`). Measured peak 0.96 × rated on a noiseless catch, and up to 1.11 × on noisy tosses at 0.10 s | Drake doesn't enforce the rating either. **State this in the paper.** At the toss's 0.986 × rated, 61 of 68 catches at one cell went more than 5% over, so the grid was re-run at 0.85. The limit is on the plan; the rest is tracking overshoot, which nothing stops: `results/approach_a.md` |
| Catch tilt | Gripper within 25° of up | The nominal catch uses 23.8°, so the ball comes in about 36° off the finger axis. Spec Q5: tighter limits lose reach. At 15° the noiseless catch has no motion, and noisy tosses catch 5 of 60 against 42 at 25°. `results/catch_timing.md` |
| Close latency | 30 ms, decision to command | Proposal's "small, fixed delta". **The catch depends on knowing it.** Swept with the trigger still assuming 30 ms: with the real latency at 0 ms nothing is caught, and B keeps 65 of its 87 catches at 20 ms and 1 at 50 ms. 10 ms late wasn't run: `results/robustness.md` |
| Close timing | Pads aimed 3 ms before the ball, through the gripper's measured response by command phase | Replaces the spec's fixed 25 ms gripper time. The pads can't arrive 20 to 29 ms after a WSG controller update: `results/catch_timing.md` |
| Catch definition | Both fingers touching the ball and grip force above 1 N, continuously for 0.5 s from when the fingers stop | Spec §13.3, and the headline metric. **This counts a ball perched on the pads' edges**, which includes the default noiseless catch, so every table also reports on-pad catches, where the ball's center ends within 1 cm of the pad's center along both axes. State both in the paper. The spec says only "non-zero" force; 1 N is this code's choice, and it sets the early edge of the timing window: `results/catch_timing.md`. **The on-pad count is fragile:** a noiseless catch ends the ball 1.47 cm along the fingers, and B's catches end at a median of 0.95 cm, right at the 1 cm edge, so it swings from 1 to 62 of B's catches with vertical toss noise alone. Report where the ball ends beside it: `planning/metrics.md` |
| Single catches | Approaches are compared on single catches | **A departure from the proposal's consecutive cycles, so the team should agree to it and the paper should say so.** A multi-cycle demo for the video stays possible, but whether a perched ball survives the arm moving hasn't been tested |
| Solve time | Approach A and the baseline are charged none: a plan takes effect at the tick it is decided | Spec §5.4, and §9 of B's spec. At 2 mm, 0.04 m/s, 0.10 s, cold solves took a median of 0.37 s and a p95 of 0.72 s of wall clock (7 workers in parallel on a laptop). **State this in the paper.** With solves that slow, a real system couldn't plan this way at either window |

## Sensing

| Quantity | Value | Basis |
|---|---|---|
| Observation mode | Position only is the headline | Spec §6.2. Changed from Week 1's position-and-velocity default, because a clean velocity channel measures the sensor rather than the strategy |
| Rate, noise | 100 Hz, σ_p = 2 mm nominal, then swept | The spec's first nominal was 5 mm, where Approach A caught 16 of 100 against the baseline's 80. **State in the paper that the cell was chosen after seeing A's and the baseline's results, before B was run, and that 2 mm is a good sensor:** it is the only nonzero σ_p tried where one look clearly beats none. `results/approach_a.md` |
| Fit window | 10 samples, 0.01 to 0.10 s after release | Spec §14 has 0.10 s, and it stays: 0.15 s loses to 0.10 s in all six cells compared, and 0.125 s depends on the sensor. The sample at release is excluded because the ball is still in the fingers. `results/approach_a.md` |
| Prediction error from that window | **1.15 cm**, 1σ per axis | From the formula on those samples, and 2.87 cm at σ_p = 5 mm, where it was checked by simulation (spec A1). It scales with σ_p. The 2.5 cm in the feasibility report counts 11 samples, including the one at release |
| When flight starts | The commanded release time | Spec §7.1. The robot knows when it opened its fingers |
| Controller rate | 200 Hz, ticking 1 ms after each 5 ms mark | So a tick sees the sensor sample from the same millisecond; on the same tick it would see the previous one |

## Approach B

| Quantity | Value | Basis |
|---|---|---|
| Estimator | Recursive least squares, not a Kalman filter | Spec §6.1. The flight is ballistic, with 2.1e-15 m of residual and white noise, so there's no process noise to model, and a tuning knob would be a difference from A with no basis. It matches A's fit bit for bit at the decision: `results/replan_timing.md` |
| First motion | **The headline A and B start on the baseline's plan**, and every solve after it is charged | That plan needs no measurements, so it can be computed before the toss. **Chosen after seeing the results, so the team should agree and the paper should say so.** The specs' versions instead make a first solve that takes no time. A cold one really takes a median of about 140 ms and a p95 of about 320 ms on a laptop, and charged only 45 ms, that A falls from 68 to 15 of 100, because the arm has about 40 ms of margin at 0.10 s: `results/approach_b.md` |
| Replans | Warm-started from the current plan's tail, 12 knots. A failed one keeps the current plan | Spec §7.2 and §11.2. Warm and cold solutions agree. 12 knots catches the same as 24 on the same schedule: `results/approach_b.md` |
| `tau_compute` | 45 ms, the p95 of a warm 12-knot replan run one at a time, rounded up | Spec §9. Parallel workers double it. A laptop and a general-purpose solver, so **report it as what it is.** 0 and 90 ms are reported beside it |
| Replan period | 45 ms, about 22 Hz | One solver at a time: a replan has to start from the plan the last one produced, so it can't start before that one has finished. **Say "about 22 Hz with a 45 ms delay" in the paper**, not "every control step" or the spec's 50 Hz. Here the solver limits the rate; the 100 Hz sensor would limit it next |
| Replan deadline | A new plan takes over by 0.47 s after release | Spec §7.3. At 5 mm a deadline of 0.35, 0.47 and 0.55 s catches 66, 67 and 67 |
| Small corrections | Skipped under 2 mm and 0.67 ms | Spec §7.3, with timing added: 1 ms of arrival is 3 mm of ball travel |
| After a failed first solve | Retried cold, charged 3 × `tau_compute` | A cold solve has about 2.3 times the p95 of a warm one. This rarely helps: the arm has already lost that time |

## The robustness sweep

| Quantity | Value | Basis |
|---|---|---|
| Versions | The moving-start A and B, the headline | Chosen after seeing results, above. The specs' versions are in `results/approach_b.md` at the levels already run |
| Design | One factor at a time around the nominal cell, plus toss noise crossed with sensor noise, on the same 100 tosses | A's spec §13.4. Every comparison is paired, with McNemar's exact test |
| Levels | σ_p 0 to 20 mm, σ lateral 0 to 0.12 m/s, σ vertical 0 to 0.05 m/s, real latency 0 to 50 ms | **0.12 m/s was added after seeing 0.08**, which didn't show the collapse B's spec predicted |
| Latency | The trigger keeps assuming 30 ms while the real latency moves | A trigger told each value would change nothing |
| Multiple comparisons | Not corrected | The sweep makes dozens of tests. The primary one is B against A at the nominal cell; **treat a p near 0.05 elsewhere as a hint, and say so in the paper** |
| Speed and height of the toss | Not swept | The toss is at the joint limits, and slower tosses mostly measure reach. Catches are reported against the movement each toss needs instead: `planning/metrics.md` |

## The driver

**Decision: keep `control_mode: position_only`.** Spec §14 asks for a change if tracking error is a
significant fraction of the tolerance. The arm is 0.71 cm from its plan on arrival, against measured sideways tolerances of
about ±2 cm and an estimator error of 1.15 cm at the nominal σ_p = 2 mm. It builds up while the plan decelerates into
the catch and takes about 200 ms to decay afterward, and the noiseless catch works with it. At the nominal toss noise a
perfect sensor catches all 91 tosses it has a motion for, so tracking alone isn't losing catches. A new driver would
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

Six cells of the grid at the speed limit were run twice, 600 trials in all:

| What | Largest difference |
|---|---|
| Any outcome (caught, on pad, contact, solve) | None changed, and every cell's counts matched |
| What the fit saw (predicted miss, timing error) | Under 1e-9, except 2e-6 on the one trial whose ball touched the hand before the catch plane, since its crossing is extrapolated from where it touched |
| Where the arm and the ball met (miss, arrival error) | 0.88 mm |
| Peak speed | 0.3% of rated speed |
| How long the ball was held | 1 ms |
| Where a held ball ended | 1.9 mm |
| Where a dropped ball came to rest | 12.6 cm |

So `trials.csv_differences` compares outcomes exactly, what the fit saw within 1e-9 (1e-5 for those two columns), what follows
the solve within 5e-3, meaning 5 mm, 5 ms or 0.5% of rated speed, and doesn't compare where a dropped ball rests. `python trials.py repeat`
runs the check. Those tolerances were set after seeing this data, and widened once after seeing it at the speed limit, so these trials
show what the spread is rather than testing them independently.

A trial sitting right on the edge of an outcome could still flip. None did in these 600, or in 600 at the earlier limit, which bounds
that at about 0.25% of trials at 95% confidence, small against the roughly ±10 points of a 100-trial cell. Rounding the target before the solve
might remove most of the spread, but it wasn't tried.

- Tests compare within 1e-9 up to the solve, and with the tolerances above after it. An earlier version compared
  Approach A's whole run at 1e-9 and passed or failed at random.
- A4's "bit-identical" holds within 1e-9 for the flight and for the whole baseline run. It can't hold for Approach A's
  arm after the solve.
- A5's "reproduces the CSV exactly" can't either; the table above is what it means here.
- B's nominal cell, 100 trials run twice, changed no outcome, solve count or adoption time. Nothing passed the tolerances above except the largest command step on one trial,
  by 0.0085 of what the plan allows, so that column compares within 0.02. That tolerance was set after seeing this, too. Each of B's solves starts from the last, but that
  didn't let the solver's spread grow past what A's has.
- The robustness sweep's nominal A and B, 200 trials run again with `python robustness.py repeat`, passed the tolerances above with nothing
  changed. Its 1000 trials that repeat cells of `results/approach_b.csv` matched in every outcome and measure compared, except where 7 dropped
  balls came to rest, by under 1e-8 m.
- The spec's risk table anticipated this: report means with confidence intervals, which the results do.
