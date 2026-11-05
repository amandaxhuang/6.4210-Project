"""What sets Approach B's constants: is the ball's flight really ballistic, how long do
replans take, and does a warm start or a coarser plan change the answer?

    HEADLESS=1 python replan_timing.py [residuals] [solve_times]

prints the sections of results/replan_timing.md.  Solve times run one trial at a
time, because parallel workers slow each other down by about a factor of two.
No section looks at whether a ball was caught, so the delay and the replan period
can't be chosen with an eye on B's result.
"""

import multiprocessing
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("HEADLESS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import catch
import environment as env
import planning_core
import replan
import sensing
import toss

N_WORKERS = 7
SIGMA_P, SIGMA_LAT, SIGMA_VZ = 0.002, 0.04, 0.02
NOMINAL = dict(sigma_p=SIGMA_P, sigma_lat=SIGMA_LAT, sigma_vz=SIGMA_VZ)
N_RESIDUAL_TRIALS = 100
RESIDUAL_LAGS = (1, 2, 3, 5)
N_TIMING_TRIALS = 20
PROBE_PERIODS = (0.05, 0.10)  # s between solves while measuring
TIME_BINS = ((0.10, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.48))


def init_worker():
    env.ASSET_DIR = Path(tempfile.mkdtemp(prefix="replan_timing_"))


def flight_residuals(run: catch.CatchRun, sigma_p: float) -> np.ndarray:
    """Each sample of the free flight, until the ball touches anything or reaches
    the catch plane, minus the fit over all of them, (n, 3)."""
    fit = sensing.BallisticFit(run.toss_plan.t_release, sigma_p)
    t_end = toss.free_flight(run.toss_plan, run.log).t_down
    if run.outcome.t_first_contact is not None:
        t_end = min(t_end, run.outcome.t_first_contact)
    samples = [o for o in run.controller.fit.observations if o.t < t_end - 1e-3]
    for o in samples:
        fit.add(o)
    est = fit.estimate()
    return np.array([o.p - toss.ballistic_position(est.p0, est.v0, o.t - est.t_anchor) for o in samples])


def run_residuals(args):
    trial, sigma_p = args
    run = catch.simulate_catch(catch.OpenLoopPolicy, sigma_p=sigma_p, sigma_lat=SIGMA_LAT, sigma_vz=SIGMA_VZ, trial=trial)
    return flight_residuals(run, sigma_p)


def report_residuals(pool):
    print("## 1. Is the flight ballistic?\n")
    quiet = run_residuals((0, 0.0))
    print(f"With a perfect sensor, the fit to one flight's {len(quiet)} samples leaves at most {np.abs(quiet).max():.1e} m.\n")
    noisy = pool.map(run_residuals, [(t, SIGMA_P) for t in range(N_RESIDUAL_TRIALS)], chunksize=1)
    n = sum(len(r) for r in noisy)
    print(f"With sigma_p = {1e3 * SIGMA_P:.0f} mm, over {N_RESIDUAL_TRIALS} flights and {n} samples, per axis.  "
          f"White noise would give autocorrelations near zero, within about {1.96 / np.sqrt(n):.3f} at 95%, plus a small "
          "negative bias from the fit itself:\n")
    print("| axis | residual, 1 sigma (mm) | " + " | ".join(f"lag {lag}" for lag in RESIDUAL_LAGS) + " |")
    print("|---|---|" + "---|" * len(RESIDUAL_LAGS))
    for axis, name in enumerate("xyz"):
        energy = sum((r[:, axis] ** 2).sum() for r in noisy)
        spread = np.sqrt(energy / n)
        rows = [sum((r[:-lag, axis] * r[lag:, axis]).sum() for r in noisy) / energy for lag in RESIDUAL_LAGS]
        print(f"| {name} | {1e3 * spread:.3f} | " + " | ".join(f"{rho:+.3f}" for rho in rows) + " |")
    print()


class TimedPolicy(replan.ReplanPolicy):
    """Follows the warm-started plans at this many knots, and times the alternatives on the same inputs."""

    def __init__(self, planner, period, knots):
        super().__init__(planner, tau_compute=0.0, period=period, n_knots=knots)
        self.rows = []
        self.first_solve = None

    def update(self, t_rel, est, q_cmd, qdot_cmd, plan):
        decided = self.first.decided
        new = super().update(t_rel, est, q_cmd, qdot_cmd, plan)
        if not decided and new is not None:
            self.first_solve = dict(time=new.solve_time, ok=new.solve_ok)
        return new

    def _replan(self, t_start, est, q_cmd, qdot_cmd, plan):
        target = None if est is None else catch.intercept(est, self.planner.z_catch, t_start, replan.T_MIN_HORIZON)
        if (target is None or plan is None
                or (np.linalg.norm(target[1] - plan.p_catch) < replan.SKIP_DISTANCE
                    and abs(self.planner.toss_plan.t_release + target[0] - plan.t_catch) < replan.SKIP_TIME)):
            return super()._replan(t_start, est, q_cmd, qdot_cmd, plan)
        t_catch, p_catch = target
        row = dict(t=t_start, moved=float(np.linalg.norm(p_catch - plan.p_catch)))
        solved = {}
        for name, knots, warm in (("warm_12", 12, True), ("cold_12", 12, False), ("warm_24", 24, True), ("cold_24", 24, False)):
            guess = plan.motion.window(t_start, t_catch, knots) if warm else None
            solved[name] = self.planner.plan(t_start, q_cmd, qdot_cmd, t_catch, p_catch, est=est, guess=guess, n_knots=knots)
            row[name] = solved[name].solve_time
            row[name + "_ok"] = solved[name].solve_ok
        self.rows.append(row | compare(self.planner, solved, p_catch))
        new = solved[f"warm_{self.n_knots}"]
        self.solve_times.append(new.solve_time)
        self.solve_failures += not new.solve_ok
        return new if new.solve_ok else None


def compare(planner, solved, p_catch) -> dict:
    """How far apart the solutions are, and how close each gets to the target."""
    pinch, v_max = planner.kin.pinch, planner.kin.v_max
    out = {}
    if solved["warm_12"].solve_ok and solved["cold_12"].solve_ok:
        out["warm_cold_12"] = float(np.abs(solved["warm_12"].motion.q - solved["cold_12"].motion.q).max())
    if solved["warm_24"].solve_ok and solved["cold_24"].solve_ok:
        out["warm_cold_24"] = float(np.abs(solved["warm_24"].motion.q - solved["cold_24"].motion.q).max())
    for name in ("warm_12", "warm_24"):
        plan = solved[name]
        if plan.solve_ok:
            out[name + "_miss"] = float(np.linalg.norm(pinch(plan.motion.q[-1]) - p_catch))
            out[name + "_peak"] = float((np.abs(plan.motion.v) / v_max).max())
    if solved["warm_12"].solve_ok and solved["warm_24"].solve_ok:
        m, n = solved["warm_12"].motion, solved["warm_24"].motion
        coarse = np.array([m.state(t)[0] for t in n.t])
        out["knots_q"] = float(np.abs(coarse - n.q).max())
    return out


def run_timing(args):
    trial, period, knots = args
    run = catch.simulate_catch(lambda planner: TimedPolicy(planner, period, knots), **NOMINAL, trial=trial)
    policy = run.controller.policy
    return dict(first=policy.first_solve, rows=policy.rows)


def stats(values) -> str:
    values = 1e3 * np.array(values)
    return f"{np.median(values):.0f} / {np.percentile(values, 95):.0f} / {values.max():.0f}" if len(values) else "-"


def report_solve_times(pool):
    print("## 2. How long a solve takes\n")
    print(f"Along real replanning runs at sigma_p = {1e3 * SIGMA_P:.0f} mm, sigma lateral = {SIGMA_LAT} and sigma vertical = "
          f"{SIGMA_VZ} m/s, {N_TIMING_TRIALS} trials one at a time.  Every line solves the same problems, from the same state to the "
          "same target.  Median / p95 / worst, in ms:\n")
    print("| replans every (ms) | solves | first solve, cold, 24 knots | warm, 12 knots | cold, 12 knots | warm, 24 knots | cold, 24 knots |")
    print("|---|---|---|---|---|---|---|")
    results = {}
    for period in PROBE_PERIODS:
        out = [run_timing((t, period, 12)) for t in range(N_TIMING_TRIALS)]
        rows = [r for o in out for r in o["rows"]]
        results[period] = rows
        first = [o["first"]["time"] for o in out if o["first"]]
        print(f"| {1e3 * period:.0f} | {len(rows)} | {stats(first)} | {stats([r['warm_12'] for r in rows])} | "
              f"{stats([r['cold_12'] for r in rows])} | {stats([r['warm_24'] for r in rows])} | {stats([r['cold_24'] for r in rows])} |")
    rows = results[PROBE_PERIODS[0]]
    print(f"\nBy when in the flight, at {1e3 * PROBE_PERIODS[0]:.0f} ms between solves, median / p95 / worst in ms:\n")
    print("| takes over (s after release) | solves | warm, 12 knots | cold, 12 knots | warm, 24 knots | cold, 24 knots |")
    print("|---|---|---|---|---|---|")
    for low, high in TIME_BINS:
        part = [r for r in rows if low <= r["t"] < high]
        print(f"| {low:.2f}-{high:.2f} | {len(part)} | {stats([r['warm_12'] for r in part])} | {stats([r['cold_12'] for r in part])} | "
              f"{stats([r['warm_24'] for r in part])} | {stats([r['cold_24'] for r in part])} |")
    print("\nDoes the warm start or the coarser plan change the answer?\n")
    print("| replans every (ms) | solves | failed: warm 12 / cold 12 / warm 24 / cold 24 | largest joint difference, warm against cold, 12 knots, "
          "median / worst (mrad) | same at 24 knots | target miss, median / worst (mm): 12 knots | 24 knots | planned peak speed, worst: 12 knots | "
          "24 knots | 12 against 24 knots, median / worst (mrad) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for period, rows in results.items():
        failed = " / ".join(str(sum(not r[name + "_ok"] for r in rows)) for name in ("warm_12", "cold_12", "warm_24", "cold_24"))

        def spread(key, scale=1e3):
            values = scale * np.array([r[key] for r in rows if key in r])
            return f"{np.median(values):.2f} / {values.max():.2f}"

        def worst(key):
            return f"{max(r[key] for r in rows if key in r):.3f}"

        print(f"| {1e3 * period:.0f} | {len(rows)} | {failed} | {spread('warm_cold_12')} | {spread('warm_cold_24')} | "
              f"{spread('warm_12_miss')} | {spread('warm_24_miss')} | {worst('warm_12_peak')} | {worst('warm_24_peak')} | {spread('knots_q')} |")
    print()


def run(sections):
    with multiprocessing.get_context("spawn").Pool(N_WORKERS, initializer=init_worker) as pool:
        for name, section in (("residuals", report_residuals), ("solve_times", report_solve_times)):
            if name in sections:
                section(pool)


if __name__ == "__main__":
    run(sys.argv[1:] or ["residuals", "solve_times"])
