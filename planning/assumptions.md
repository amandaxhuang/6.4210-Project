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
| Plant time step | 1 ms | Week 1 §8; not yet checked against 0.5 ms (spec Q7) |
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
| Catch tilt | Gripper within 25° of up | The nominal catch uses 23.8°, so the ball comes in about 36° off the finger axis. Whether a tighter limit is reachable is spec Q5 |
| Close latency | 30 ms, decision to command | Proposal's "small, fixed delta" |
| Close timing | Pads aimed 3 ms before the ball, through the gripper's measured response by command phase | Replaces the spec's fixed 25 ms gripper time. The pads can't arrive 20 to 29 ms after a WSG controller update: `results/catch_timing.md` |
| Catch definition | Both fingers touching the ball and grip force above 1 N, continuously for 0.5 s from when the fingers stop | Spec §13.3. *Open:* **this counts a ball perched on the pads' edges**, which includes the default noiseless catch. Trials record `end_in_hand`, so either definition can be applied later. The spec says only "non-zero" force; 1 N is this code's choice, and it sets the early edge of the timing window: `results/catch_timing.md` |

## Sensing

| Quantity | Value | Basis |
|---|---|---|
| Observation mode | Position only is the headline | Spec §6.2. Changed from Week 1's position-and-velocity default, because a clean velocity channel measures the sensor rather than the strategy |
| Rate, noise | 100 Hz, σ_p = 5 mm per axis (swept) | Spec §6.3 |
| Fit window | 10 samples, 0.01 to 0.10 s after release | Spec §14. The sample at release is excluded because the ball is still in the fingers |
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

**Simulations are repeatable to about 1e-12 m, not bit for bit.** Running the same simulation twice gives one
of two results at random, even within one process and with no controller or solver, differing by up to about
1e-12 m. All metrics agreed to 12 decimal places across repeated runs. The source is inside Drake's simulation
and wasn't pinned down further: it isn't the floating-point mode, and it doesn't need ball contact. So:

- Tests compare within 1e-9, not for equality.
- A4's "bit-identical" can't be met as worded; "identical within 1e-9" can.
- A5's CSV comparison should use a numeric tolerance. Rounding alone isn't enough, because a value near a
  rounding boundary can still flip its last digit.
