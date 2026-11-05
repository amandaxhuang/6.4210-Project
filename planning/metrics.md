# Metrics

The list the paper reports, what each number means, and where it comes from. Every metric is a column of `results/robustness.csv`, one row per trial, unless it
says otherwise. `results/robustness.md` shows all of them at the nominal cell.

**Status:** proposed final list, 2026-11-05, tested on every cell of the robustness sweep. Team to confirm.

## What to report

| Role | Metric | Column | Definition |
|---|---|---|---|
| **Primary** | Caught | `caught` | Both fingers touch the ball with at least 1 N of grip, continuously for 0.5 s from when the fingers stop closing (A's spec, §13.3). Counts a ball perched on the pads' edges |
| Secondary | Contact | `contact` | The ball touches the gripper at any point after leaving it. The proposal asks for this beside the catch rate |
| Stricter count | Caught on the pad | `caught_on_pad` | Caught, and the ball's center ends within 1 cm of the pad's center along both axes. **A fragile count:** B's caught balls end a median of 0.95 cm along the fingers, right at that edge, so it swings from 1 to 62 of B's catches with vertical toss noise alone. Report the next row beside it |
| Stricter count | Where the ball ends | `end_fingers` | Median distance along the fingers from the pinch point to the caught ball's center. 1.47 cm on a noiseless catch, which is why none of those is on the pad. Mostly set by when the fingers close: closing 10 ms earlier moves it about 1 cm |
| Stricter count | Caught within rating | `peak_speed` | Caught, and no joint passes 1.05x of its rated speed. Drake doesn't enforce the rating |
| Where a miss came from | Predicted miss | `predicted_miss` | Distance from the last plan's target to where the ball really crosses the catch plane: the estimator alone |
| Where a miss came from | Timing error | `timing_error` | When the ball really crosses the plane, minus when the last plan said. Reported as 1 sigma |
| Where a miss came from | Plan stop error | `plan_stop_error` | Distance from where the last plan stops to its target: 5.2 mm in every plan, from `POSITION_TOL` |
| Where a miss came from | Tracking error | `tracking_error` | Distance from the pinch point at the catch time to where the plan stops: the tracker alone |
| Where a miss came from | Stayed on the baseline's plan | `plans_adopted` | At most one plan adopted, so every solve failed or nothing needed correcting |
| How B works | Target moved | `plan_drift` | Distance from the first plan's target to the last's |
| How B works | Close decision moved | `trigger_shift` | The last close decision's time, minus the first's |
| How B works | Solves | `n_solves`, `solves_skipped`, `solve_failures` | How many ran, how many were skipped as too small to matter, and how many failed |
| Cost | Solve time | `results/replan_timing.md` | Median and p95, run one trial at a time. The trial record's `solve_time` comes from parallel workers, which about double it, so don't report that |
| Reach | Movement needed | Worked out from `p_release_true_*`, `v_release_true_*` | Distance from where the brake leaves the pinch point to where this toss really comes down. The proposal's "movement needed from ball release to catch" |

`arm_arrival_error` stays in the record because the first results use it, but it combines plan stop and tracking error, which partly cancel. Report those two instead.

## How to report it

- **Catch rates are counts of 100 with a 95% Wilson interval.** A 100-trial cell is good to about ±10 points, so a difference of a few catches between unpaired rows means little.
- **Compare approaches on the same tosses, with McNemar's exact test.** Trial n is the same toss and sensor noise for every approach, so report how many only one approach caught, and p.
- **The primary comparison is B against A at the nominal cell,** on trials 0 to 99, with trials 100 to 199 as a check on tosses the cell wasn't chosen from. Everything else is secondary. The
  sweep makes dozens of comparisons with no correction for that, so treat a p near 0.05 elsewhere as a hint, and say so.
- **State what was chosen after seeing results:** the nominal cell, and the moving start.

## The proposal's goal

At least 50% caught and 90% contact on a nominal catch with no toss noise. That's the "No noise at all" row of `results/robustness.md`.

## What the proposal lists that isn't here

| Proposal | Why not | Instead |
|---|---|---|
| Consecutive successful cycles | Single catches only, a decision the team should agree to. It would need a second toss from wherever the catch ended, and whether a perched ball survives the arm moving hasn't been tested | State it in the paper |
| Initial speed, toss height | The toss is already at the arm's joint limits, so it can't go faster. Slower tosses have shorter flights, 0.543 s at 2.55 m/s and 0.515 s at 2.40 m/s, against about 0.54 s the arm needs for the nominal catch at 0.85 of rated speed. So they would mostly measure the arm's reach, and they haven't been run | Report against movement needed, and say why in the paper |
