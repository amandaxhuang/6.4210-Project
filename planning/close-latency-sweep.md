# Close-Latency Sweep and Latency Noise — Implementation Plan

**Project:** Effect of Continual Planning on Robotic Arm Catching Robustness (6.4210, Team 4.4)
**Builds on:** `planning/approach-a-predict-once.md` §10, `planning/metrics.md`, `results/robustness.md` finding 5
**Status:** Approved 2026-11-06, in implementation.

## Why

`results/robustness.md` finding 5 says the gripper's timing matters more than any noise in the
study: the real latency sweep over {0, 20, 30, 50} ms takes B from 87 catches to 0, 65 and 1, and
it is the one cell where B does clearly worse than A. That claim rests on four points, three of
them 20 ms apart, so nobody can tell whether B's collapse between 30 and 50 ms is a cliff or a
slope. And `planning/assumptions.md` records close latency as the proposal's "small, fixed delta" —
a constant the controller is told exactly, which no real gripper is.

This work replaces that with a resolved curve and a distribution:

1. **Eleven constant levels**, 0 and then 0.005 to 0.050 s in 0.005 s steps, so the shape between
   30 and 50 ms is measured rather than interpolated. 0 ms is kept because finding 5 cites it.
2. **Noise cells**, where each toss draws its own delay from a Gaussian: the gripper decides to
   close, then waits a random short time before the fingers actually start moving.
3. **Every metric on `planning/metrics.md`'s final list, and plots, for the baseline, A and B at
   every cell.**

### The two sigmas

The request specified the Gaussian two ways, and they are very different experiments:

| Source | Mean | σ | 5th–95th percentile |
|---|---|---|---|
| "0.005 at the lower 5% end and 0.050 at the upper 95% end" | 27.5 ms | **13.7 ms** | 5 to 50 ms |
| "mean 27.5 ms and standard deviation 2.5 ms" | 27.5 ms | **2.5 ms** | 23.4 to 31.6 ms |

σ = 2.5 ms is mild jitter that stays inside the ~22 ms timing window; σ = 13.7 ms spans the whole
constant sweep. **Both are run** — a cell is 300 trials, about 6 minutes — and the sampler takes σ
as a parameter, so further cells cost one list entry each. The paper has to say which one it
quotes as the headline.

## Decisions

- Extend `robustness.py`, rather than adding a separate script. `results/robustness.csv` and
  `results/robustness.md` are regenerated.
- 100 trials per cell, matching every other cell in the paper.
- The trigger keeps assuming 30 ms at every level, constant or noisy, as the existing sweep does
  (`robustness.ASSUMED_LATENCY`). A trigger told each real latency would change nothing, and in the
  noise cells it could not know the draw anyway.

## Step 0 — unblock the environment ✅

`catch.py` called `plant.SetFreeBodySpatialVelocity(context, body, V)`; drake 1.45.0 binds only
`(body, V_PB, context)`, so every test that simulated a catch raised `TypeError` — 21 of 118. The
argument order is swapped; the ball is a free body whose parent frame is the world, so `V_PB` is
`V_WB` and the semantics are unchanged.

Gate, both before anything else:

- `HEADLESS=1 python -m unittest discover tests` → 180 pass.
- `HEADLESS=1 python robustness.py repeat` → no differences beyond `planning/assumptions.md`'s
  tolerances, which proves the fix reproduces the committed 6600-trial sweep. Run this *before*
  adding config fields: `trials.repeat_check` reads every `ReplanConfig` field name out of the
  saved CSV, so a new field makes the old CSV unreadable to it.

## Step 1 — the latency sampler (`catch.py`)

One function beside `toss_velocity_noise`, which it deliberately mirrors:

```python
def close_latency(trial: int, latency: float = T_LATENCY, sigma_latency: float = 0.0) -> float:
    if sigma_latency <= 0:
        return latency
    return max(0.0, latency + sigma_latency * np.random.default_rng([SEED_ROOT, trial, 2]).normal())
```

Three properties the experiment depends on:

- **Stream 2 is new.** The toss uses `[SEED_ROOT, trial, 0]` and the sensor `[SEED_ROOT, trial, 1]`,
  so every saved toss and sensor realization is untouched and the old rows still reproduce.
- **The short circuit at σ = 0 is not an optimization.** It makes the constant cells bit-identical
  to what the existing code produces, which is what makes the repeat gate meaningful.
- **A trial draws one latency, shared by the baseline, A and B.** Trial *n* is the same toss, the
  same sensor noise *and* the same gripper for all three, so the noise cells stay paired and
  McNemar's test still applies. Without it they could only be compared unpaired, at ±10 points.

One draw per trial is the whole story because `CatchController._t_close` latches: once the decision
to close is taken it never moves, so there is only one delay to draw.

It is clipped at 0 because the fingers cannot start moving before the decision. At σ = 13.7 ms,
Φ(−27.5/13.7) ≈ 2% of draws fall below zero, so about 2 trials per 100 are clipped and the report
says so; at σ = 2.5 ms none are.

Threading it through, with `sigma_latency=0.0` defaults so no existing caller changes:
`simulate_catch` computes the realized value and passes *that* to `build_catch_system`, so
`CatchController.latency` is what the gripper really did. `_calc_fingers` and `hold_after_closing`
already key off `controller.t_close + controller.latency`, so both follow. `assumed_latency` is
untouched: it stays what the trigger *believes*, which is the point of the experiment.

## Step 2 — carry it through the record (`trials.py`, `replan_trials.py`)

| Where | Field | Why |
|---|---|---|
| `TrialConfig` | `sigma_latency: float = 0.0` | Identifies the cell. `latency` keeps meaning the **mean**, so `select(..., latency=…)` and `robustness._settings` keep working |
| `TrialResult` | `sigma_latency`, `latency_drawn` | The cell, and what this trial's gripper really waited |
| `trials.simulate` | pass `sigma_latency=config.sigma_latency` | — |
| `trials.record` | `latency_drawn=run.controller.latency` | The realized value, read back off the controller |

Both new fields have defaults and `write_csv`/`read_csv` are name-based, so column order is free.
`ReplanResult` inherits them, and `read_results` skips columns a saved CSV lacks, so existing rows
load with `sigma_latency = 0.0` — which is how they were run. That is what lets
`robustness.missing()` recognize them as already done and reuse them.

## Step 3 — the cells (`robustness.py`)

```python
LATENCIES = (0.0, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050)  # s
LATENCY_MEAN = 0.0275  # s, the midpoint of 0.005 and 0.050
LATENCY_NOISE = ((LATENCY_MEAN, 0.0025), (LATENCY_MEAN, SIGMA_FOR_5_TO_50))  # (mean, sigma)
```

`grid()` keeps its constant loop, which already applies `_latency_settings`, and adds one cell per
`LATENCY_NOISE` entry for the baseline, A and B.

**`select()` needs a `sigma_latency` default.** It filters only on settings it names, so without
`"sigma_latency": 0.0` in its defaults a constant-cell lookup would also match noise rows whose
mean equals that constant. Nothing collides at 27.5 ms, but a later cell centred on a constant
would quietly double a table's row count.

**Run with `extend`, not `run`.** It reuses every saved row whose full config matches and runs only
what is new. The 0, 20, 30 and 50 ms constants are already saved (1200 rows), so this is **2700 new
trials, about 52 minutes** on 7 workers at the recorded 8.1 s per trial, against 2.2 hours for a
full re-run. The CSV's row order changes because `grid()` grew; the saved rows' contents are
written back verbatim.

## Step 4 — report every metric at every cell (`robustness.py`)

1. `metric_table(results, **settings)` — generalized to take cell settings and pass them to
   `select`. No arguments still means the nominal cell, so the existing section is unchanged.
2. **"Every metric at each close latency"**: one `metric_table` per cell under its own heading, the
   full 14-metric list for the baseline, A and B. Long, so it goes at the end of the report.
3. The existing compact latency table (`factor_table` over `LATENCIES`) needs no change and becomes
   the headline: 11 rows with caught, the pairing and p, on-pad, where the ball ends, contact,
   stayed-on-the-baseline's-plan, failed solves, predicted miss and timing error.
4. **A noise table** via `factor_table`'s `settings_of` hook, plus the columns only noise cells
   have: median, σ and the 5th and 95th percentiles of `latency_drawn` — the audit that the sampler
   matches the spec — and how many draws were clipped at 0.

## Step 5 — plots (`figures.py`)

Reusing `_style`, `_series`, `_save`, `_legend`, `_end_label` and `COLORS`, so each approach keeps
its fixed color and the Wilson wash.

| Figure | Shows |
|---|---|
| `caught_by_close_latency.png` | The resolved constant curve, 11 levels, three approaches. `by_factor` unchanged — it just gets the longer `LATENCIES` |
| `caught_by_latency_noise.png` | Each noise cell against that curve: the constants as the line, each noise cell as a labelled band per approach |
| `caught_by_realized_latency.png` | **The mechanism figure.** Catches binned by `latency_drawn`, overlaid on the constant curve. If a jitter cell is just the constant curve averaged over the distribution, the two coincide; if it is worse, the interaction is real |
| `latency_metrics_panels.png` | Small multiples against latency: timing error 1σ, where the ball ends along the fingers, on-pad count, contact |
| `latency_draws.png` | ECDF of the drawn latencies per noise cell, with the 5th and 95th marks and the clipping at 0 |

`_series` spaces the x axis by index, which is right here: the constants are evenly spaced in time.

## Step 6 — tests

- `tests/test_catch.py`: `close_latency` returns the fixed delay exactly at σ = 0; the same trial
  gives the same delay under every policy; draws have the right mean and σ, and at σ = 13.679 ms
  the 5th and 95th percentiles land on 5 and 50 ms; negative draws are clipped; and the fingers
  start moving at `t_close + latency_drawn` in a simulated run.
- `tests/test_robustness.py`: the grid covers all 11 constants and every noise cell at 100 trials
  (`test_every_cell_has_100_trials` keys on config fields, so `sigma_latency` joins its key or the
  noise cells collapse into the constants); the trigger assumption holds for noise cells too; the
  figure count rises.
- `tests/test_trials.py`: a row round-trips the new columns, and a CSV without them still loads.

## Verification

1. `HEADLESS=1 python -m unittest discover tests`.
2. `HEADLESS=1 python robustness.py repeat` — step 0's gate, before adding fields.
3. `close_latency` against a hand-computed draw, then one `simulate_catch(sigma_latency=…)` whose
   `controller.latency` equals it.
4. `HEADLESS=1 python robustness.py extend` → "2700 trials run, 9300 written".
5. `HEADLESS=1 python robustness.py repeat` again: the nominal cell still matches, so adding cells
   changed nothing already measured.
6. `python robustness.py report && python figures.py`, then check every number in a figure appears
   in a table.
7. The constant curve passes through the committed 0/20/30/50 ms values exactly, because those are
   the same rows; the σ = 2.5 ms cell sits near the 25–30 ms constants; `latency_drawn`'s
   percentiles match the spec.

## What to write up afterwards

`results/robustness.md` finding 5 and `planning/assumptions.md`'s "Close latency" and "Levels" rows
state the four-point sweep and the fixed-delta assumption, and both go stale the moment this runs.
`planning/metrics.md` gains `latency_drawn`. The report should say which σ the paper treats as the
headline, and that a real gripper's latency *distribution* would have to be measured rather than
assumed — the existing text says that about the constant, and it matters more for the spread.
