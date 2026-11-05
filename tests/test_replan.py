"""Approach B's policy, and the guards that keep it comparable with Approach A
(planning/approach-b-closed-loop-replanning.md, section 2)."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

import catch
import planning_core
import replan
import sensing
import toss
import trials

P0, V0 = np.array([0.8, 0.0, 0.774]), np.array([-0.353, -0.528, 2.645])
T_RELEASE = 1.0
Q, V = np.full(7, 0.25), np.full(7, 0.5)
FIRST = catch.T_WINDOW + 0.001
TAU = 0.05


def estimate(p0=P0, v0=V0):
    return sensing.Estimate(np.array(p0, dtype=float), np.array(v0, dtype=float), np.eye(6), 10, T_RELEASE)


class Line:
    """A trajectory whose position is its time and whose velocity is twice that."""

    def value(self, t):
        return np.full((7, 1), t)

    def EvalDerivative(self, t, order):
        return np.full((7, 1), 2 * t)


class Tail:
    def window(self, t0, t1, n):
        return "tail", t0, t1, n


class RecordingPlanner:
    """Stands in for the trajectory optimizer, so the policy's logic can be tested alone."""

    z_catch = 0.674
    nominal = catch.Nominal(0.575, np.array([0.6, -0.3, 0.674]))
    toss_plan = SimpleNamespace(t_release=T_RELEASE)

    def __init__(self, fail=()):
        self.calls = []
        self.fail = fail  # which calls fail, counting from 1

    def plan(self, t_rel, q0, v0, t_catch, p_catch, est=None, guess=None, n_knots=planning_core.N_KNOTS, **flags):
        self.calls.append(dict(t_rel=t_rel, q0=q0, v0=v0, t_catch=t_catch, p_catch=p_catch, est=est, guess=guess,
                               n_knots=n_knots, **flags))
        ok = len(self.calls) not in self.fail
        return SimpleNamespace(solve_ok=ok, q_traj=Line() if ok else None, motion=Tail() if ok else None,
                               t_catch=T_RELEASE + t_catch, p_catch=p_catch, solve_time=0.01, est=est)


def moved(by=0.05):
    """An estimate whose catch point has moved sideways."""
    return estimate(P0 + np.array([by, 0.0, 0.0]))


class PolicyTest(unittest.TestCase):
    def started(self, fail=(), **kwargs):
        """A policy that has made its first solve, and the plan that gave."""
        self.planner = RecordingPlanner(fail)
        policy = replan.ReplanPolicy(self.planner, **{"tau_compute": TAU, **kwargs})
        return policy, policy.update(FIRST, estimate(), Q, V, None)

    def test_the_first_solve_is_approach_as(self):
        a, b = RecordingPlanner(), RecordingPlanner()
        est = estimate()
        self.assertIsNone(replan.ReplanPolicy(b).update(catch.T_WINDOW - 0.001, est, Q, V, None))
        catch.PredictOncePolicy(a).update(FIRST, est, Q, V, None)
        replan.ReplanPolicy(b).update(FIRST, est, Q, V, None)
        self.assertEqual(a.calls[0].keys(), b.calls[0].keys())
        for name, value in a.calls[0].items():
            np.testing.assert_array_equal(b.calls[0][name], value, name)

    def test_the_estimator_is_the_recursive_one(self):
        self.assertIs(replan.ReplanPolicy.fit_class, sensing.RecursiveFit)

    def test_nothing_is_replanned_until_the_last_plan_could_be_finished(self):
        policy, plan = self.started()
        self.assertIsNone(policy.update(FIRST + TAU - 0.005, moved(), Q, V, plan))
        self.assertIsNotNone(policy.update(FIRST + TAU, moved(), Q, V, plan))

    def test_a_period_shorter_than_the_delay_is_refused(self):
        """With one solver at a time, the next solve can't start until the last has finished."""
        with self.assertRaises(ValueError):
            replan.ReplanPolicy(RecordingPlanner(), tau_compute=0.05, period=0.02)

    def test_a_replan_starts_where_the_plan_will_be_when_it_takes_over(self):
        policy, plan = self.started()
        now = FIRST + TAU
        policy.update(now, moved(), Q, V, plan)
        call = self.planner.calls[-1]
        self.assertAlmostEqual(call["t_rel"], now + TAU)
        np.testing.assert_allclose(call["q0"], np.full(7, T_RELEASE + now + TAU))
        np.testing.assert_allclose(call["v0"], np.full(7, 2 * (T_RELEASE + now + TAU)))

    def test_with_no_delay_a_replan_starts_from_the_commanded_state(self):
        policy, plan = self.started(tau_compute=0.0, period=0.02)
        policy.update(FIRST + 0.02, moved(), Q, V, plan)
        call = self.planner.calls[-1]
        self.assertAlmostEqual(call["t_rel"], FIRST + 0.02)
        np.testing.assert_array_equal(call["q0"], Q)
        np.testing.assert_array_equal(call["v0"], V)

    def test_a_replan_is_warm_started_from_the_tail_on_its_own_knots(self):
        policy, plan = self.started(n_knots=12)
        now = FIRST + TAU
        policy.update(now, moved(), Q, V, plan)
        call = self.planner.calls[-1]
        self.assertEqual(call["n_knots"], 12)
        self.assertEqual(call["guess"], ("tail", call["t_rel"], call["t_catch"], 12))

    def test_a_cold_policy_passes_no_guess(self):
        policy, plan = self.started(warm=False)
        policy.update(FIRST + TAU, moved(), Q, V, plan)
        self.assertIsNone(self.planner.calls[-1]["guess"])

    def test_a_replan_aims_at_the_latest_estimate(self):
        policy, plan = self.started()
        est = moved()
        policy.update(FIRST + TAU, est, Q, V, plan)
        call = self.planner.calls[-1]
        self.assertIs(call["est"], est)
        t, p = catch.intercept(est, self.planner.z_catch, call["t_rel"], replan.T_MIN_HORIZON)
        self.assertEqual(call["t_catch"], t)
        np.testing.assert_array_equal(call["p_catch"], p)

    def test_the_deadline_is_on_when_the_plan_takes_over(self):
        policy, plan = self.started(deadline=0.47)
        self.assertIsNone(policy.update(0.425, moved(), Q, V, plan))
        self.assertIsNotNone(policy.update(0.415, moved(), Q, V, plan))

    def test_a_deadline_at_the_window_leaves_only_the_first_solve(self):
        """This is the configuration the equivalence with Approach A is tested in."""
        policy, plan = self.started(deadline=catch.T_WINDOW)
        for t in FIRST + np.arange(0.005, 0.4, 0.005):
            self.assertIsNone(policy.update(t, moved(), Q, V, plan))
        self.assertEqual(len(self.planner.calls), 1)

    def test_a_failed_replan_keeps_the_current_plan_and_costs_one_cycle(self):
        policy, plan = self.started(fail=(2,))
        self.assertIsNone(policy.update(FIRST + TAU, moved(), Q, V, plan))
        self.assertEqual(policy.solve_failures, 1)
        self.assertIsNone(policy.update(FIRST + TAU + 0.005, moved(), Q, V, plan))
        self.assertIsNotNone(policy.update(FIRST + 2 * TAU, moved(), Q, V, plan))

    def test_a_failed_first_solve_is_reported_then_retried_cold(self):
        policy, plan = self.started(fail=(1,))
        self.assertFalse(plan.solve_ok)
        self.assertEqual(policy.solve_failures, 1)
        delay = TAU * replan.COLD_DELAY_FACTOR
        now = FIRST + TAU
        new = policy.update(now, estimate(), Q, V, plan)
        self.assertTrue(new.solve_ok)
        call = self.planner.calls[-1]
        self.assertIsNone(call["guess"])
        self.assertAlmostEqual(call["t_rel"], now + delay)
        np.testing.assert_array_equal(call["q0"], Q)
        np.testing.assert_array_equal(call["v0"], V)
        self.assertIsNone(policy.update(now + delay - 0.005, estimate(), Q, V, new))

    def test_a_correction_too_small_to_matter_is_skipped(self):
        policy, plan = self.started()
        self.assertIsNone(policy.update(FIRST + TAU, moved(0.001), Q, V, plan))
        self.assertEqual(policy.solves_skipped, 1)
        self.assertEqual(len(self.planner.calls), 1)

    def test_a_timing_correction_is_not_skipped(self):
        """A millisecond of arrival is 3 mm of ball travel, even if the catch point barely moves."""
        policy, plan = self.started()
        later = estimate(P0, V0 + np.array([0.0, 0.0, 0.01]))
        self.assertIsNotNone(policy.update(FIRST + TAU, later, Q, V, plan))
        self.assertEqual(policy.solves_skipped, 0)

    def test_a_rejected_intercept_keeps_the_plan_without_solving(self):
        policy, plan = self.started()
        self.assertIsNone(policy.update(FIRST + TAU, estimate([0, 0, 0.5], [0, 0, 0.1]), Q, V, plan))
        self.assertEqual(policy.intercepts_rejected, 1)
        self.assertEqual(len(self.planner.calls), 1)

    def test_every_solve_is_counted_and_timed(self):
        policy, plan = self.started(fail=(2,))
        policy.update(FIRST + TAU, moved(), Q, V, plan)
        policy.update(FIRST + 2 * TAU, moved(), Q, V, plan)
        self.assertEqual(policy.n_solves, 3)
        self.assertEqual(policy.solve_times, [0.01] * 3)


class InterceptForAMovingArmTest(unittest.TestCase):
    def test_the_time_to_reach_it_can_be_shorter_once_the_arm_is_moving(self):
        self.assertIsNone(catch.intercept(estimate(), 0.674, 0.3))
        self.assertIsNotNone(catch.intercept(estimate(), 0.674, 0.3, replan.T_MIN_HORIZON))


class EquivalenceTest(unittest.TestCase):
    """Spec deliverable B3: with replanning off, B is A, so anything else that differs
    between them is replanning and nothing else."""

    def assert_records_match(self, trial):
        noise = dict(sigma_p=0.002, sigma_lat=0.04, sigma_vz=0.02, trial=trial)
        config = trials.TrialConfig("predict_once", **noise)
        a = catch.simulate_catch(catch.PredictOncePolicy, **noise)
        b = catch.simulate_catch(lambda planner: replan.ReplanPolicy(planner, deadline=catch.T_WINDOW), **noise)
        with tempfile.TemporaryDirectory() as folder:
            path_a, path_b = Path(folder) / "a.csv", Path(folder) / "b.csv"
            trials.write_csv([trials.record(config, a, 0.0)], path_a)
            trials.write_csv([trials.record(config, b, 0.0)], path_b)
            self.assertEqual(trials.csv_differences(path_a, path_b), [])
        return b

    def test_a_toss_with_a_motion(self):
        run = self.assert_records_match(0)
        self.assertTrue(run.plans[-1].solve_ok)
        self.assertEqual(run.controller.policy.n_solves, 1)

    def test_a_toss_without_a_motion(self):
        """Trial 15 has no motion at the nominal cell, so a failed solve has to match as well."""
        run = self.assert_records_match(15)
        self.assertFalse(run.plans[-1].solve_ok)


class ReplanningRunTest(unittest.TestCase):
    """One noisy toss at the nominal cell, with replanning on."""

    @classmethod
    def setUpClass(cls):
        cls.b = catch.simulate_catch(lambda planner: replan.ReplanPolicy(planner, tau_compute=TAU),
                                     sigma_p=0.002, sigma_lat=0.04, sigma_vz=0.02, trial=0)
        cls.controller = cls.b.controller
        cls.policy = cls.controller.policy

    def test_it_replans_at_least_five_times(self):
        self.assertGreaterEqual(self.policy.n_solves + self.policy.solves_skipped, 5)
        self.assertGreater(self.controller.plans_adopted, 1)

    def test_each_plan_takes_over_at_its_own_start(self):
        starts = [p.q_traj.start_time() for p in self.b.plans]
        np.testing.assert_allclose(self.controller.adoption_times, starts, atol=1e-3)

    def test_a_replan_stays_pending_for_its_compute_delay(self):
        for plan in self.b.plans[1:]:
            self.assertAlmostEqual(plan.est.n_obs, round((plan.q_traj.start_time() - TAU - self.b.toss_plan.t_release) * 100),
                                   delta=1)

    def test_the_command_stays_continuous_across_every_adoption(self):
        """Spec deliverable B4."""
        position, velocity = replan.command_steps(self.b)
        self.assertLess(position, 1.0)
        self.assertLess(velocity, 1.0 + 1e-6)

    def test_the_close_decision_comes_from_the_last_plan_before_it_was_taken(self):
        t_close = catch.close_command_time(self.b.plans[-1].t_catch)
        self.assertAlmostEqual(self.controller.t_close, t_close, places=6)

    def test_the_first_close_decision_is_kept_for_the_record(self):
        t_close = catch.close_command_time(self.b.plans[0].t_catch)
        self.assertAlmostEqual(self.controller.t_close_first, t_close, places=6)


class CloseLatchTest(unittest.TestCase):
    def setUp(self):
        self.plan = toss.plan_toss(catch.shared_kinematics().plant)
        policy, fit = catch.OpenLoopPolicy(None), sensing.BallisticFit(self.plan.t_release)
        self.controller = catch.CatchController(self.plan, policy, fit)
        self.t_catch = self.plan.t_release + 0.575
        self.solved = SimpleNamespace(solve_ok=True, t_catch=self.t_catch)

    def test_a_later_plan_moves_a_decision_not_yet_taken(self):
        t_close = self.controller._t_close(self.t_catch - 0.2, self.solved, np.inf)
        moved = SimpleNamespace(solve_ok=True, t_catch=self.t_catch + 0.01)
        self.assertGreater(self.controller._t_close(self.t_catch - 0.1, moved, t_close), t_close)

    def test_a_decision_once_taken_stands(self):
        t_close = self.controller._t_close(self.t_catch - 0.2, self.solved, np.inf)
        moved = SimpleNamespace(solve_ok=True, t_catch=self.t_catch + 0.01)
        self.assertEqual(self.controller._t_close(t_close + 0.001, moved, t_close), t_close)


if __name__ == "__main__":
    unittest.main()
