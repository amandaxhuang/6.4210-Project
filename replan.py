"""Approach B: refit as data arrives, and replan toward the updated intercept.

The first solve is Approach A's, from the same estimate at the same instant, so B
starts where A ends.  After that one solver works at a time.  A solve starts from
everything seen so far, takes tau_compute, and only then can the arm act on it, so
it plans from where the current plan will be by then.  Consecutive plans therefore
agree in position and velocity where one takes over from the other.  When no
solve is running, the next one starts.

Replans are warm-started from the current plan's tail, on fewer knots than the
first solve.  planning/approach-b-closed-loop-replanning.md has the design, and
results/replan_timing.md the measurements that set the constants below.

    python replan.py [trial]

runs one toss at the nominal cell in Meshcat, with a sphere on the current target.
"""

import sys

import numpy as np
from pydrake.all import RigidTransform, Rgba, Sphere

import catch
from sensing import RecursiveFit
import toss
from utils.viz import current_meshcat, keep_meshcat_open

T_FIRST_SOLVE = catch.T_WINDOW  # the same instant as Approach A's only solve, by definition
REPLAN_KNOTS = 12
TAU_COMPUTE = 0.045  # s; the p95 of a warm-started replan, rounded up to a control period
REPLAN_DEADLINE = 0.470  # s after release; the last time a new plan may take over
T_MIN_HORIZON = 0.02  # s; a moving arm can't be told anything useful closer to the catch than this

# With no plan to warm-start from, a solve takes this many times as long, by p95, rounded up.
COLD_DELAY_FACTOR = 3

# Corrections smaller than these aren't worth a solve, or the disturbance of a new plan.
SKIP_DISTANCE = 0.002  # m
BALL_SPEED_AT_CATCH = 2.99  # m/s
SKIP_TIME = SKIP_DISTANCE / BALL_SPEED_AT_CATCH  # s


class ReplanPolicy:
    """Approach B.  A deadline at the observation window leaves only the first solve,
    which makes this Approach A.

    tau_compute  how long after its data a plan takes over
    period       how often a solve starts, at least tau_compute: one solver at a time
    deadline     the last time a new plan may take over (s after release)
    n_knots      how finely replans are discretized
    warm         whether replans start from the current plan's tail
    """

    fit_class = RecursiveFit

    def __init__(self, planner: catch.CatchPlanner, tau_compute: float = TAU_COMPUTE, period: float | None = None,
                 deadline: float = REPLAN_DEADLINE, n_knots: int = REPLAN_KNOTS, warm: bool = True):
        period = tau_compute if period is None else period
        if period < tau_compute:
            raise ValueError(f"a solve can't start every {period} s if each takes {tau_compute} s")
        self.planner = planner
        self.first = catch.PredictOncePolicy(planner, T_FIRST_SOLVE)
        self.tau_compute = tau_compute
        self.period = period
        self.deadline = deadline
        self.n_knots = n_knots
        self.warm = warm
        self.t_next = np.inf  # when the next solve is due, s after release
        self.solve_times: list[float] = []
        self.solve_failures = 0
        self.solves_skipped = 0
        self.intercepts_rejected = 0

    @property
    def n_solves(self) -> int:
        return len(self.solve_times)

    def update(self, t_rel, est, q_cmd, qdot_cmd, plan):
        if not self.first.decided:
            new = self.first.update(t_rel, est, q_cmd, qdot_cmd, plan)
            if new is None:
                return None
            self.t_next = t_rel + self.period
            return self._counted(new)
        if t_rel < self.t_next - 1e-9:
            return None

        # A plan that failed left the arm holding still, with nothing to warm-start from.
        moving = plan is not None and plan.solve_ok
        delay = self.tau_compute if moving else self.tau_compute * COLD_DELAY_FACTOR
        t_start = t_rel + delay
        if t_start > self.deadline + 1e-9:
            return None
        self.t_next = t_rel + max(self.period, delay)
        return self._replan(t_start, est, q_cmd, qdot_cmd, plan if moving else None)

    def _replan(self, t_start, est, q_cmd, qdot_cmd, plan):
        """A new plan to take over at t_start (s after release), or None to keep the current one."""
        target = None if est is None else catch.intercept(est, self.planner.z_catch, t_start, T_MIN_HORIZON)
        if target is None:
            self.intercepts_rejected += 1
            return None
        t_catch, p_catch = target
        t_release = self.planner.toss_plan.t_release
        guess = None
        if plan is not None:
            if (np.linalg.norm(p_catch - plan.p_catch) < SKIP_DISTANCE
                    and abs(t_release + t_catch - plan.t_catch) < SKIP_TIME):
                self.solves_skipped += 1
                return None
            if self.warm:
                guess = plan.motion.window(t_start, t_catch, self.n_knots)
            if self.tau_compute > 0:
                t = t_release + t_start
                q_cmd, qdot_cmd = plan.q_traj.value(t).ravel(), plan.q_traj.EvalDerivative(t, 1).ravel()
        new = self.planner.plan(t_start, q_cmd, qdot_cmd, t_catch, p_catch, est=est, guess=guess, n_knots=self.n_knots)
        # A failed solve costs one cycle, not the plan already being followed.
        counted = self._counted(new)
        return counted if counted.solve_ok else None

    def _counted(self, plan):
        self.solve_times.append(plan.solve_time)
        self.solve_failures += not plan.solve_ok
        return plan


class MovingStartPolicy(ReplanPolicy):
    """Approach B with no free solve.  The arm starts on the baseline's plan, which
    uses no measurements, so it could really be computed before the toss.  The
    solver is then free at once, and every solve after that is charged."""

    def __init__(self, planner: catch.CatchPlanner, **settings):
        super().__init__(planner, **settings)
        self.first = catch.OpenLoopPolicy(planner)

    def update(self, t_rel, est, q_cmd, qdot_cmd, plan):
        decided = self.first.decided
        new = super().update(t_rel, est, q_cmd, qdot_cmd, plan)
        if new is not None and not decided:
            self.t_next = t_rel
        return new


class OneCorrectionPolicy(MovingStartPolicy):
    """Approach A on the same footing as MovingStartPolicy: the same start, and one
    charged solve from the same data as Approach A's, kept to whatever comes of it."""

    def __init__(self, planner: catch.CatchPlanner, **settings):
        super().__init__(planner, **settings)
        self.corrected = False

    def _replan(self, t_start, est, q_cmd, qdot_cmd, plan):
        if self.corrected:
            return None
        self.corrected = True
        return super()._replan(t_start, est, q_cmd, qdot_cmd, plan)


def command_steps(run: catch.CatchRun) -> tuple[float, float]:
    """How close the arm command came to breaking continuity, over the whole catch
    motion (spec deliverable B4), as fractions of what the plans allow in one row: the
    largest position step against the rated speed, and the largest step in its
    numerical derivative against the plan's acceleration limit."""
    log, plans = run.log, [p for p in run.plans if p.solve_ok]
    if not plans:
        return 0.0, 0.0
    during = np.flatnonzero((log.t >= plans[0].q_traj.start_time() - 0.002) & (log.t <= plans[-1].t_catch))
    t, q = log.t[during], log.q_cmd[during]
    new_row = np.concatenate([[True], np.diff(t) > 1e-9])  # the log can repeat a time to within rounding
    t, q = t[new_row], q[new_row]
    dt = np.diff(t)[:, np.newaxis]
    v = np.diff(q, axis=0) / dt
    v_max, _ = toss.arm_limits(run.plant)
    a_max = run.controller.policy.planner.a_max
    between = (dt[:-1] + dt[1:]) / 2  # near the catch the simulator's steps aren't all 1 ms
    return float((np.abs(v) / v_max).max()), float((np.abs(np.diff(v, axis=0)) / (a_max * between)).max())


def draw_targets(run: catch.CatchRun) -> None:
    """Add to Meshcat's animation where the arm is aiming, from when each plan takes over."""
    meshcat = current_meshcat()
    if meshcat is None:
        return
    path = "replan/target"
    meshcat.SetObject(path, Sphere(0.012), Rgba(1.0, 0.45, 0.1, 0.9))
    recording = meshcat.get_mutable_recording()
    target = None
    for plan, t in zip(run.plans, run.controller.adoption_times):
        frame = recording.frame(t)
        if target is not None:  # hold the last target until this frame, rather than sliding to the next
            recording.SetTransform(frame - 1, f"/drake/{path}", target)
        target = RigidTransform(plan.p_catch)
        recording.SetTransform(frame, f"/drake/{path}", target)
    meshcat.PublishRecording()


catch.POLICIES["replan"] = ReplanPolicy
catch.POLICIES["replan_moving_start"] = MovingStartPolicy
catch.POLICIES["one_correction"] = OneCorrectionPolicy


if __name__ == "__main__":
    run = catch.simulate_catch(ReplanPolicy, sigma_lat=0.04, sigma_vz=0.02, trial=int(sys.argv[1]) if sys.argv[1:] else 0)
    catch.report(run)
    print(f"plans take over at {np.round(np.array(run.controller.adoption_times) - run.toss_plan.t_release, 3)} s after release")
    draw_targets(run)
    keep_meshcat_open()
