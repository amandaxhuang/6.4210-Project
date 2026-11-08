# 6.4210 Robotic Manipulation Final Project

[progress report](https://claude.ai/artifact/SaA4gStUfhaddejy6g8dFJ)

A Drake simulation in which a KUKA iiwa 14 with a Schunk WSG 50 gripper tosses a ball upward and
catches it with the same gripper. The question is how the arm should use noisy observations of the
ball to guide the catch, so three strategies are compared on identical tosses, under increasing
sensor noise, toss noise, and gripper closing delay.

## The approaches

All three run through the same controller (`catch.CatchController`), which follows the toss, feeds
sensor readings to an estimator, follows the current catch plan, and closes the fingers. An approach
is only a *policy* that decides which plan to follow and when, so any difference in results comes
from the policy.

| Approach | Idea | Class |
|---|---|---|
| Baseline | Ignore the ball; go where a perfect toss would land | `catch.OpenLoopPolicy` |
| Approach A | Fit the ball's path once from an early window of observations, plan one correction, then commit | `replan.OneCorrectionPolicy` |
| Approach B | Keep refitting (recursive least squares) and replanning as observations arrive, up to a deadline | `replan.MovingStartPolicy` |

A and B both start on the baseline's plan, which needs no observations and can be computed before
the toss, and every solve after that is charged a realistic compute delay. These "moving start"
versions are the headline. The original spec versions, where the first solve takes no time, are
`catch.PredictOncePolicy` and `replan.ReplanPolicy`; `results/approach_b.md` compares the two.

Simplifications: single catches (no repeated toss–catch cycles), no drag or spin, no arm collision
geometry. `planning/assumptions.md` lists every assumption with its evidence.

## Setup

Python 3.13 and Drake. Keep the virtual environment on a path with **no spaces**, or Drake's model
download fails with `returncode == 512` (see `utils/drake_models.py`).

```sh
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Watching a toss and catch

Run without `HEADLESS`. Each script prints a Meshcat URL (usually http://localhost:7000), waits a
few seconds for a browser, runs, and keeps the scene open until you press Enter. Use Meshcat's
Animations panel to replay.

```sh
python toss.py                    # the toss alone
python catch.py open_loop         # baseline, on a noise-free toss
python catch.py predict_once      # Approach A (spec version), on a noise-free toss
python replan.py 3                # Approach B (headline), nominal noise, trial 3, targets drawn
```

`python catch.py` with no argument runs the baseline. To run the headline A or the baseline on the
same noisy trial as `replan.py`:

```sh
python -c "import catch, replan; from utils.viz import keep_meshcat_open; \
r = catch.simulate_catch(replan.OneCorrectionPolicy, sigma_lat=0.04, sigma_vz=0.02, trial=3); \
catch.report(r); keep_meshcat_open()"
```

Swap in `catch.OpenLoopPolicy` for the baseline or `replan.MovingStartPolicy` for B. Trial *n* is
the same toss and sensor noise under every approach.

## Layout

| Path | Contents |
|---|---|
| `environment.py` | Scene: arm, gripper, floor, ball held in the fingers |
| `toss.py` | The toss and projectile helpers |
| `planning_core.py` | Trajectory optimization for catch motions |
| `sensing.py` | Noisy ball sensor, batch fit (A), recursive fit (B) |
| `catch.py` | Catch controller, baseline and spec-A policies, close trigger, scoring |
| `replan.py` | Approach B and the moving-start policies |
| `trials.py` | Seeded trial records, parallel sweeps, statistics (Wilson, McNemar) |
| `robustness.py`, `replan_trials.py` | Experiment sweeps and their reports |
| `figures.py`, `tables.py` | Figures and table images for the paper |
| `catch_timing.py`, `feasibility.py`, `release_search.py`, `replan_timing.py` | Studies that set the constants |
| `planning/` | Specs, `assumptions.md` (read first), `metrics.md` |
| `results/` | Reports, CSVs, `figures/`, `tables/` |
| `tests/` | Unit tests |
| `utils/` | Shared course helpers (Meshcat, plotting backend) |
| `references/` | Earlier iiwa pick-and-place code kept for reference |