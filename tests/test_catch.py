"""The catch controller, its policies, and the end-to-end self-test (spec section 11)."""

from types import SimpleNamespace
import unittest

import numpy as np

import catch
import environment as env
import feasibility
import planning_core
import sensing
import toss
import trials
from utils.viz import HEADLESS


# Simulating the same thing twice can differ by up to about 1e-12, at random, even
# in one process: results/catch_timing.md.  So "the same" means this close.
SAME = 1e-9


def estimate(p0, v0, n_obs=10):
    return sensing.Estimate(np.array(p0, dtype=float), np.array(v0, dtype=float), np.eye(6), n_obs, 1.0)


class RecordingPlanner:
    """Stands in for the trajectory optimizer, so policy logic can be tested alone."""

    z_catch = 0.674
    nominal = catch.Nominal(0.575, np.array([0.6, -0.3, 0.674]))

    def __init__(self):
        self.calls = []

    def plan(self, t_rel, q0, v0, t_catch, p_catch, est=None, **flags):
        self.calls.append(dict(t_rel=t_rel, t_catch=t_catch, p_catch=p_catch, est=est, **flags))
        return f"plan {len(self.calls)}"


class InterceptTest(unittest.TestCase):
    P0, V0 = [0.8, 0.0, 0.774], [-0.353, -0.528, 2.645]

    def test_finds_the_later_crossing_of_the_plane(self):
        t, p = catch.intercept(estimate(self.P0, self.V0), 0.674, 0.1)
        self.assertAlmostEqual(t, 0.5747, places=3)
        self.assertAlmostEqual(p[2], 0.674, places=9)
        np.testing.assert_allclose(p, toss.ballistic_position(np.array(self.P0), np.array(self.V0), t))

    def test_rejects_a_parabola_that_never_reaches_the_plane(self):
        self.assertIsNone(catch.intercept(estimate([0, 0, 0.5], [0, 0, 0.1]), 0.674, 0.1))

    def test_rejects_a_crossing_too_soon_to_reach(self):
        self.assertIsNone(catch.intercept(estimate(self.P0, [0, 0, 0.5]), 0.674, 0.1))

    def test_rejects_a_crossing_after_the_limit(self):
        self.assertIsNone(catch.intercept(estimate(self.P0, [0, 0, 6.0]), 0.674, 0.1))


def pads_arrive(t_command):
    """When the pads reach ball width, from the measured response."""
    k = round(t_command * 1000 + 0.5)  # the first simulator step at or after the command
    return (k + catch.GRIPPER_RESPONSE_MS[k % 50]) / 1000


class CloseTriggerTest(unittest.TestCase):
    def test_an_explicit_lead_is_used_as_given(self):
        self.assertAlmostEqual(catch.close_command_time(1.575, lead=0.04), 1.535)

    def test_the_pads_arrive_at_the_target_time(self):
        """Away from the gap in the response, the lookup lands within a step of the target."""
        for ms_after_update in (30, 35, 40, 45, 50, 60, 69):
            target = 1.5 + ms_after_update / 1000
            t_catch = target + catch.PAD_LEAD
            arrival = pads_arrive(catch.finger_command_time(t_catch))
            self.assertLessEqual(abs(arrival - target), 0.001, f"{ms_after_update} ms after an update")

    def test_targets_in_the_gap_go_to_the_nearer_side(self):
        """Pads can't arrive 20 to 29 ms after an update: a command just before one
        arrives at 20 ms and one just after it at 29 ms."""
        for ms_after_update, expected in ((21, 20), (23, 20), (26, 29), (28, 29)):
            target = 1.5 + ms_after_update / 1000
            arrival = pads_arrive(catch.finger_command_time(target + catch.PAD_LEAD))
            self.assertAlmostEqual(arrival, 1.5 + expected / 1000, delta=0.0011)

    def test_arrival_never_runs_backward(self):
        arrivals = [pads_arrive(catch.finger_command_time(t))
                    for t in np.arange(1.40, 1.70, 0.001)]
        self.assertGreaterEqual(np.diff(arrivals).min(), -1e-9)

    def test_the_decision_comes_a_latency_before_the_fingers(self):
        t_catch = 1.575
        self.assertAlmostEqual(catch.close_command_time(t_catch, latency=0.02),
                               catch.finger_command_time(t_catch) - 0.02)

    def test_the_response_table_is_current(self):
        """If this fails the gripper changed: re-measure with catch_timing.py."""
        for k in (0, 10, 30, 40, 49):
            measured = feasibility.measure_close(0.300 + k / 1000)["contact"]
            self.assertAlmostEqual(measured * 1000, catch.GRIPPER_RESPONSE_MS[k], delta=1.0, msg=f"phase {k}")


class PolicyTest(unittest.TestCase):
    Q, V = np.zeros(7), np.zeros(7)

    def test_open_loop_plans_once_at_the_end_of_the_brake_and_ignores_the_estimate(self):
        planner = RecordingPlanner()
        policy = catch.OpenLoopPolicy(planner)
        wild = estimate([9, 9, 9], [9, 9, 9])
        self.assertIsNone(policy.update(toss.BRAKE - 0.001, wild, self.Q, self.V, None))
        plan = policy.update(toss.BRAKE + 0.001, wild, self.Q, self.V, None)
        self.assertEqual(plan, "plan 1")
        self.assertIsNone(policy.update(toss.BRAKE + 0.006, wild, self.Q, self.V, plan))
        self.assertEqual(len(planner.calls), 1)
        self.assertEqual(planner.calls[0]["t_catch"], planner.nominal.t_catch)
        np.testing.assert_array_equal(planner.calls[0]["p_catch"], planner.nominal.p_catch)
        self.assertIsNone(planner.calls[0]["est"])

    def test_predict_once_plans_once_from_the_estimate_at_the_window(self):
        planner = RecordingPlanner()
        policy = catch.PredictOncePolicy(planner)
        est = estimate([0.8, 0.0, 0.774], [-0.353, -0.528, 2.645])
        self.assertIsNone(policy.update(catch.T_WINDOW - 0.001, est, self.Q, self.V, None))
        plan = policy.update(catch.T_WINDOW + 0.001, est, self.Q, self.V, None)
        self.assertEqual(plan, "plan 1")
        self.assertIsNone(policy.update(catch.T_WINDOW + 0.006, est, self.Q, self.V, plan))
        self.assertEqual(len(planner.calls), 1)
        self.assertIs(planner.calls[0]["est"], est)
        self.assertAlmostEqual(planner.calls[0]["t_catch"], 0.5747, places=3)

    def test_predict_once_falls_back_to_nominal_without_a_fit(self):
        planner = RecordingPlanner()
        catch.PredictOncePolicy(planner).update(catch.T_WINDOW + 0.001, None, self.Q, self.V, None)
        call = planner.calls[0]
        self.assertTrue(call["fit_rank_deficient"])
        self.assertEqual(call["t_catch"], planner.nominal.t_catch)

    def test_predict_once_falls_back_to_nominal_on_an_unreachable_intercept(self):
        planner = RecordingPlanner()
        est = estimate([0, 0, 0.5], [0, 0, 0.1])
        catch.PredictOncePolicy(planner).update(catch.T_WINDOW + 0.001, est, self.Q, self.V, None)
        call = planner.calls[0]
        self.assertTrue(call["intercept_rejected"])
        self.assertEqual(call["t_catch"], planner.nominal.t_catch)


class PlannerTest(unittest.TestCase):
    def test_trajectory_reproduces_the_motion(self):
        kin = planning_core.Kinematics()
        plan = toss.plan_toss(kin.plant)
        t_catch, p_catch = toss.catch_point(plan)
        q_stop = plan.q_traj.value(plan.t_release + toss.BRAKE).ravel()
        a_max = planning_core.accel_limits(kin)["brake"]
        motion = planning_core.plan_reach(kin, plan, q_stop, np.zeros(7), toss.BRAKE,
                                          t_catch - toss.BRAKE, p_catch, a_max)
        trajectory = catch.motion_to_trajectory(motion, plan.t_release)
        rng = np.random.default_rng(0)
        for t in rng.uniform(motion.t[0], motion.t[-1], 100):
            q, v = motion.state(t)
            np.testing.assert_allclose(trajectory.value(t + plan.t_release).ravel(), q, atol=1e-9)
            np.testing.assert_allclose(trajectory.EvalDerivative(t + plan.t_release, 1).ravel(), v, atol=1e-9)

    def test_a_failed_solve_is_recorded_rather_than_raised(self):
        kin = planning_core.Kinematics()
        plan = toss.plan_toss(kin.plant)
        planner = catch.CatchPlanner(kin, plan)
        q_stop = plan.q_traj.value(plan.t_release + toss.BRAKE).ravel()
        result = planner.plan(toss.BRAKE, q_stop, np.zeros(7), toss.BRAKE + 0.05,
                              np.array([0.6, -0.3, 0.674]))
        self.assertFalse(result.solve_ok)
        self.assertIsNone(result.q_traj)


class ControllerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_a = catch.simulate_catch(catch.PredictOncePolicy, sigma_p=0.0)

    def test_the_toss_is_unchanged_by_the_controller(self):
        """Up to the decision, the arm and the ball must match toss.py."""
        plan, log, _ = toss.simulate()
        run = self.run_a
        t_decision = plan.t_release + toss.BRAKE
        n = np.searchsorted(log.t, t_decision)
        np.testing.assert_allclose(run.log.t[:n], log.t[:n], atol=SAME)
        np.testing.assert_allclose(run.log.arm_q[:n], log.arm_q[:n], atol=SAME)
        np.testing.assert_allclose(run.log.ball_p[:n], log.ball_p[:n], atol=SAME)

    def test_the_fit_uses_ten_samples_after_release(self):
        est = self.run_a.plans[0].est
        self.assertEqual(est.n_obs, 10)
        first, last = (o.t - self.run_a.toss_plan.t_release
                       for o in (self.run_a.controller.fit.observations[0],
                                 self.run_a.controller.fit.observations[9]))
        self.assertAlmostEqual(first, 0.01, places=9)
        self.assertAlmostEqual(last, 0.10, places=9)

    def test_a_noiseless_fit_predicts_the_real_flight(self):
        run = self.run_a
        fly = toss.free_flight(run.toss_plan, toss_only_log())
        plan = run.plans[0]
        self.assertLess(abs(plan.t_catch - fly.t_down), 0.002)
        self.assertLess(np.linalg.norm(plan.p_catch - fly.p_down), 0.002)

    def test_one_policy_decision_is_recorded(self):
        self.assertEqual(self.run_a.controller.plans_adopted, 1)
        self.assertGreater(self.run_a.controller.policy_calls, 100)

    def test_a_policy_that_never_decides_is_caught(self):
        class Never:
            def __init__(self, planner):
                pass

            def update(self, *args):
                return None

        with self.assertRaises(RuntimeError):
            catch.simulate_catch(Never, sigma_p=0.0)

    def test_compute_latency_delays_adoption(self):
        run = catch.simulate_catch(catch.PredictOncePolicy, sigma_p=0.0, compute_latency=0.02)
        t_decision = run.toss_plan.t_release + catch.T_WINDOW
        # Still holding at q_stop until the plan is due, then moving.
        rows = run.log.t < t_decision + 0.02
        q_stop = run.toss_plan.q_traj.value(t_decision).ravel()
        after_brake = rows & (run.log.t > t_decision)
        np.testing.assert_allclose(run.log.q_cmd[after_brake], np.tile(q_stop, (after_brake.sum(), 1)))


_toss_log = None


def toss_only_log():
    global _toss_log
    if _toss_log is None:
        _toss_log = toss.simulate()[1]
    return _toss_log


class CalibrationTest(unittest.TestCase):
    def test_nominal_offset_is_current(self):
        """If this fails the toss changed: re-measure with catch.measure_nominal_offset()."""
        kin = planning_core.Kinematics()
        plan = toss.plan_toss(kin.plant)
        fly = toss.free_flight(plan, toss_only_log())
        nominal = catch.nominal_catch(plan)
        self.assertAlmostEqual(nominal.t_catch, fly.t_down - plan.t_release, places=4)
        np.testing.assert_allclose(nominal.p_catch, fly.p_down, atol=1e-4)


class FingersStoppedTest(unittest.TestCase):
    def test_a_brief_pause_is_not_stopping(self):
        rate = np.concatenate([np.full(5, -1.0), np.zeros(3), np.full(5, -1.0), np.zeros(30)])
        self.assertEqual(catch.first_settled(rate), 13)

    def test_never_settling(self):
        self.assertIsNone(catch.first_settled(np.full(50, -1.0)))


class TossNoiseDrawTest(unittest.TestCase):
    def test_no_sigma_means_no_noise(self):
        np.testing.assert_array_equal(catch.toss_velocity_noise(3, 0.0, 0.0), np.zeros(3))

    def test_the_noise_depends_only_on_the_trial(self):
        np.testing.assert_array_equal(catch.toss_velocity_noise(3, 0.02, 0.02), catch.toss_velocity_noise(3, 0.02, 0.02))
        self.assertFalse(np.array_equal(catch.toss_velocity_noise(3, 0.02, 0.02), catch.toss_velocity_noise(4, 0.02, 0.02)))

    def test_lateral_and_vertical_are_separate(self):
        lateral = catch.toss_velocity_noise(3, 0.02, 0.0)
        vertical = catch.toss_velocity_noise(3, 0.0, 0.02)
        self.assertEqual(lateral[2], 0.0)
        np.testing.assert_array_equal(vertical[:2], np.zeros(2))
        self.assertNotEqual(vertical[2], 0.0)

    def test_a_larger_sigma_scales_the_same_draw(self):
        np.testing.assert_allclose(catch.toss_velocity_noise(3, 0.04, 0.08), 2 * catch.toss_velocity_noise(3, 0.02, 0.04))

    def test_the_toss_and_observation_streams_are_independent(self):
        """Trial n uses its own stream for each, as spec section 13.2 asks."""
        expected = np.random.default_rng([catch.SEED_ROOT, 3, 0]).normal(size=3)
        np.testing.assert_allclose(catch.toss_velocity_noise(3, 1.0, 1.0), expected)


class CloseLatencyDrawTest(unittest.TestCase):
    """The gripper's delay, drawn per trial."""

    WIDE = 0.013679028718014808  # puts the 5th percentile on 5 ms and the 95th on 50 ms

    def test_no_sigma_means_the_fixed_delay(self):
        self.assertEqual(catch.close_latency(3, 0.03, 0.0), 0.03)
        self.assertEqual(catch.close_latency(3, 0.05), 0.05)

    def test_the_delay_depends_only_on_the_trial(self):
        """So a trial is the same gripper under every policy, and the pairing holds."""
        self.assertEqual(catch.close_latency(3, 0.0275, 0.0025), catch.close_latency(3, 0.0275, 0.0025))
        self.assertNotEqual(catch.close_latency(3, 0.0275, 0.0025), catch.close_latency(4, 0.0275, 0.0025))

    def test_it_draws_from_its_own_stream(self):
        """Not the toss's or the sensor's, so adding it leaves both unchanged."""
        expected = 0.0275 + 0.0025 * np.random.default_rng([catch.SEED_ROOT, 3, 2]).normal()
        self.assertAlmostEqual(catch.close_latency(3, 0.0275, 0.0025), expected, places=15)

    def test_the_draws_have_the_mean_and_sigma_asked_for(self):
        drawn = np.array([catch.close_latency(t, 0.0275, 0.0025) for t in range(20000)])
        self.assertAlmostEqual(drawn.mean(), 0.0275, delta=5e-5)
        self.assertAlmostEqual(drawn.std(ddof=1), 0.0025, delta=5e-5)

    def test_the_wide_sigma_spans_5_to_50_ms(self):
        """The task's other reading: 0.005 at the lower 5% end and 0.050 at the upper 95%."""
        drawn = np.array([catch.close_latency(t, 0.0275, self.WIDE) for t in range(20000)])
        self.assertAlmostEqual(np.percentile(drawn, 5), 0.005, delta=5e-4)
        self.assertAlmostEqual(np.percentile(drawn, 95), 0.050, delta=5e-4)

    def test_a_negative_draw_is_clipped_at_zero(self):
        """The fingers cannot start moving before the decision to close."""
        drawn = [catch.close_latency(t, 0.0275, self.WIDE) for t in range(2000)]
        self.assertTrue(all(d >= 0.0 for d in drawn))
        self.assertGreater(sum(d == 0.0 for d in drawn), 0)  # at this sigma about 2% clip
        self.assertEqual(catch.close_latency(0, 0.0, 0.01), max(0.0, catch.close_latency(0, 0.0, 0.01)))


class CloseLatencyRunTest(unittest.TestCase):
    """A drawn delay, through a whole run."""

    TRIAL = 7
    SIGMA = 0.013679028718014808

    @classmethod
    def setUpClass(cls):
        cls.drawn = catch.close_latency(cls.TRIAL, 0.0275, cls.SIGMA)
        cls.catch_run = catch.simulate_catch(catch.PredictOncePolicy, sigma_p=0.0, trial=cls.TRIAL,
                                       latency=0.0275, sigma_latency=cls.SIGMA, assumed_latency=catch.T_LATENCY)

    def test_the_controller_waits_the_delay_this_trial_drew(self):
        self.assertAlmostEqual(self.catch_run.controller.latency, self.drawn, places=15)
        self.assertNotAlmostEqual(self.drawn, 0.0275)  # this trial really drew something else

    def test_the_trigger_is_not_told_the_draw(self):
        """It allows for what it was told, so the fingers are early or late by the error."""
        self.assertEqual(self.catch_run.controller.assumed_latency, catch.T_LATENCY)

    def test_the_fingers_close_after_the_delay_this_trial_drew(self):
        """The scorer measures the close from the drawn delay, not the mean: the fingers
        settle after it, and the catch is held from there.  (Inward motion on its own is
        not the close -- the fingers ring as the toss releases the ball.)"""
        controller, log = self.catch_run.controller, self.catch_run.log
        held, t_stopped = catch.hold_after_closing(log, controller)
        self.assertIsNotNone(t_stopped)
        self.assertGreater(t_stopped, controller.t_close + controller.latency)
        # Whether it is caught is a result, not a property of the plumbing: this trial drew
        # 40.9 ms against the 30 ms the trigger assumed, so the fingers are 11 ms late.
        self.assertGreater(self.drawn, catch.T_LATENCY)
        self.assertEqual(held, 0.0)

    def test_a_drawn_delay_defaults_to_a_trigger_that_knows_only_the_mean(self):
        """Left to itself the trigger would assume whatever the controller was handed,
        which with a draw would make it clairvoyant."""
        run = catch.simulate_catch(catch.PredictOncePolicy, sigma_p=0.0, trial=self.TRIAL,
                                   latency=0.0275, sigma_latency=self.SIGMA)
        self.assertEqual(run.controller.assumed_latency, 0.0275)
        self.assertAlmostEqual(run.controller.latency, self.drawn, places=15)


class TossNoiseTest(unittest.TestCase):
    """A perfect sensor, so that the toss noise is the only error."""

    TRIAL = 5

    @classmethod
    def setUpClass(cls):
        noise = dict(sigma_lat=0.02, sigma_vz=0.02, trial=cls.TRIAL)
        cls.quiet = catch.simulate_catch(catch.PredictOncePolicy, sigma_p=0.0, trial=cls.TRIAL)
        cls.a = catch.simulate_catch(catch.PredictOncePolicy, sigma_p=0.0, **noise)
        cls.baseline = catch.simulate_catch(catch.OpenLoopPolicy, sigma_p=0.0, **noise)
        cls.noise = catch.toss_velocity_noise(cls.TRIAL, 0.02, 0.02)
        cls.t_noise = cls.a.toss_plan.t_release + catch.T_TOSS_NOISE

    def row(self, run, t):
        return np.searchsorted(run.log.t, t)

    def test_the_run_records_the_noise(self):
        np.testing.assert_array_equal(self.a.toss_noise, self.noise)
        np.testing.assert_array_equal(self.quiet.toss_noise, np.zeros(3))

    def test_nothing_changes_before_the_noise_goes_in(self):
        n = self.row(self.a, self.t_noise)
        np.testing.assert_allclose(self.a.log.ball_p[:n], self.quiet.log.ball_p[:n], atol=SAME)
        np.testing.assert_allclose(self.a.log.arm_q[:n], self.quiet.log.arm_q[:n], atol=SAME)

    def test_the_ball_is_free_when_the_noise_goes_in(self):
        self.assertEqual(self.a.log.touching[self.row(self.a, self.t_noise)], frozenset())

    def test_the_noise_goes_in_before_the_first_observation(self):
        self.assertLess(catch.T_TOSS_NOISE, 1 / sensing.OBS_RATE)

    def test_the_ball_velocity_changes_by_exactly_the_noise(self):
        t = self.t_noise + 0.005
        difference = self.a.log.ball_v[self.row(self.a, t)] - self.quiet.log.ball_v[self.row(self.quiet, t)]
        np.testing.assert_allclose(difference, self.noise, atol=1e-6)

    def test_a_perfect_fit_predicts_the_perturbed_flight(self):
        fly = toss.free_flight(self.a.toss_plan, self.a.log)
        plan = self.a.plans[0]
        self.assertLess(abs(plan.t_catch - fly.t_down), 0.002)
        self.assertLess(np.linalg.norm(plan.p_catch - fly.p_down), 0.002)

    def test_the_baseline_ignores_it_and_aims_where_it_did(self):
        nominal = catch.CatchPlanner(catch.shared_kinematics(), self.baseline.toss_plan).nominal
        np.testing.assert_array_equal(self.baseline.plans[0].p_catch, nominal.p_catch)
        fly = toss.free_flight(self.baseline.toss_plan, self.baseline.log)
        self.assertAlmostEqual(self.baseline.outcome.predicted_miss, np.linalg.norm(nominal.p_catch - fly.p_down))

    def test_a_trial_gets_the_same_toss_under_every_policy(self):
        t = self.t_noise + 0.09  # up to the decision, before either policy has moved the arm
        n = self.row(self.a, t)
        np.testing.assert_allclose(self.a.log.ball_p[:n], self.baseline.log.ball_p[:n], atol=SAME)

    def test_without_noise_different_trials_are_the_same(self):
        """Spec deliverable A4.  The flight repeats within SAME, but what follows the
        solve repeats less closely, since a change that small can move the solver's answer."""
        a = catch.simulate_catch(catch.PredictOncePolicy, sigma_p=0.0, trial=1)
        b = catch.simulate_catch(catch.PredictOncePolicy, sigma_p=0.0, trial=2)
        decision = self.row(a, self.t_noise + 0.09)
        before_the_catch = self.row(a, a.toss_plan.t_release + 0.5)
        np.testing.assert_allclose(a.log.ball_p[:before_the_catch], b.log.ball_p[:before_the_catch], atol=SAME)
        np.testing.assert_allclose(a.log.arm_q[:decision], b.log.arm_q[:decision], atol=SAME)
        self.assertEqual(a.outcome.caught, b.outcome.caught)
        self.assertAlmostEqual(a.outcome.miss_horizontal, b.outcome.miss_horizontal, delta=trials.SAME_AFTER_SOLVE)

    def test_without_noise_the_baseline_repeats_all_the_way(self):
        """It has no fit to feed the solver a slightly different target."""
        a = catch.simulate_catch(catch.OpenLoopPolicy, sigma_p=0.0, trial=1)
        b = catch.simulate_catch(catch.OpenLoopPolicy, sigma_p=0.0, trial=2)
        np.testing.assert_allclose(a.log.ball_p, b.log.ball_p, atol=SAME)
        np.testing.assert_allclose(a.log.arm_q, b.log.arm_q, atol=SAME)

    def test_toss_noise_leaves_the_observation_noise_alone(self):
        def sensor_noise(run):
            return np.array([o.p - catch._at(o.t, run.log.t, run.log.ball_p)
                             for o in run.controller.fit.observations[:10]])

        quiet = catch.simulate_catch(catch.PredictOncePolicy, sigma_p=0.005, trial=self.TRIAL)
        noisy = catch.simulate_catch(catch.PredictOncePolicy, sigma_p=0.005, sigma_lat=0.02, sigma_vz=0.02,
                                     trial=self.TRIAL)
        np.testing.assert_allclose(sensor_noise(quiet), sensor_noise(noisy), atol=1e-6)


class HoldTimeTest(unittest.TestCase):
    BOTH = frozenset({"wsg::left_finger", "wsg::right_finger"})

    def scenario(self, touching=None):
        """Fingers close from 0.1 s to 0.12 s; the grip force falls from 2 N to 0.5 N at 0.6 s."""
        t = np.arange(0.0, 1.0, 0.001)
        rate = np.where((t >= 0.1) & (t < 0.12), -1.0, 0.0)
        force = np.where(t < 0.6, 2.0, 0.5)
        log = SimpleNamespace(t=t, finger_rate=rate, grip_force=force,
                              touching=touching if touching is not None else [self.BOTH] * len(t))
        return log, SimpleNamespace(t_close=0.07, latency=0.03)

    def test_the_hold_ends_when_the_force_falls_below_the_threshold(self):
        log, controller = self.scenario()
        held, t_stop = catch.hold_after_closing(log, controller, grip_force_min=1.0)
        self.assertAlmostEqual(t_stop, 0.12, delta=0.002)
        self.assertAlmostEqual(held, 0.6 - t_stop, delta=0.002)

    def test_a_lower_threshold_holds_longer(self):
        log, controller = self.scenario()
        held, t_stop = catch.hold_after_closing(log, controller, grip_force_min=0.1)
        self.assertAlmostEqual(held, 0.999 - t_stop, delta=0.002)

    def test_the_hold_ends_when_a_finger_lets_go(self):
        log, controller = self.scenario()
        log.touching = [self.BOTH if t < 0.3 else frozenset({"wsg::left_finger"}) for t in log.t]
        held, t_stop = catch.hold_after_closing(log, controller, grip_force_min=0.1)
        self.assertAlmostEqual(held, 0.3 - t_stop, delta=0.002)

    def test_no_contact_is_no_hold(self):
        log, controller = self.scenario(touching=[frozenset()] * 1000)
        self.assertEqual(catch.hold_after_closing(log, controller)[0], 0.0)

    def test_fingers_that_never_moved_give_no_hold(self):
        log, controller = self.scenario()
        log.finger_rate = np.zeros(len(log.t))
        self.assertEqual(catch.hold_after_closing(log, controller), (0.0, None))


class SelfTest(unittest.TestCase):
    """Spec section 11: if these can't catch a perfect toss, nothing downstream means anything."""

    def test_baseline_catches_a_noiseless_toss(self):
        run = catch.simulate_catch(catch.OpenLoopPolicy, sigma_p=0.0)
        self.assertTrue(run.outcome.caught, run.outcome)
        self.assertTrue(run.outcome.contact, run.outcome)

    def test_predict_once_catches_a_noiseless_toss(self):
        run = catch.simulate_catch(catch.PredictOncePolicy, sigma_p=0.0)
        self.assertTrue(run.outcome.caught, run.outcome)
        self.assertTrue(run.outcome.contact, run.outcome)

    def test_a_catch_motion_stays_within_the_plan_speed_limit(self):
        plan = catch.simulate_catch(catch.OpenLoopPolicy, sigma_p=0.0).plans[-1]
        v_max = catch.shared_kinematics().v_max
        speeds = [np.abs(plan.q_traj.EvalDerivative(t, 1).ravel()) / v_max
                  for t in np.linspace(plan.q_traj.start_time(), plan.t_catch, 400)]
        self.assertLessEqual(np.max(speeds), catch.PLAN_SPEED_LIMIT + 1e-6)

    def test_the_arm_stays_within_its_rating_on_a_noiseless_catch(self):
        run = catch.simulate_catch(catch.OpenLoopPolicy, sigma_p=0.0)
        self.assertLess(run.peak_speed, 1.0)
        self.assertGreater(run.peak_speed, catch.PLAN_SPEED_LIMIT)  # it does overshoot the plan

    def test_a_run_with_no_motion_has_no_peak_speed(self):
        run = catch.simulate_catch(lambda planner: catch.PredictOncePolicy(planner, 0.15), sigma_p=0.0)
        self.assertFalse(run.plans[-1].solve_ok)  # 0.15 s leaves too little time at this limit
        self.assertTrue(np.isnan(run.peak_speed))

    def test_identical_runs_give_identical_records(self):
        a = catch.simulate_catch(catch.PredictOncePolicy, sigma_p=0.005, trial=11)
        b = catch.simulate_catch(catch.PredictOncePolicy, sigma_p=0.005, trial=11)
        decision = np.searchsorted(a.log.t, a.toss_plan.t_release + catch.T_WINDOW)
        np.testing.assert_allclose(a.log.ball_p[:decision], b.log.ball_p[:decision], atol=SAME)
        np.testing.assert_allclose(a.log.arm_q[:decision], b.log.arm_q[:decision], atol=SAME)
        self.assertEqual(a.outcome.caught, b.outcome.caught)
        self.assertAlmostEqual(a.outcome.predicted_miss, b.outcome.predicted_miss, delta=SAME)
        self.assertAlmostEqual(a.outcome.miss_horizontal, b.outcome.miss_horizontal, delta=trials.SAME_AFTER_SOLVE)

    def test_different_trials_see_different_noise(self):
        a = catch.simulate_catch(catch.PredictOncePolicy, sigma_p=0.005, trial=11)
        b = catch.simulate_catch(catch.PredictOncePolicy, sigma_p=0.005, trial=12)
        self.assertGreater(np.linalg.norm(a.plans[0].p_catch - b.plans[0].p_catch), 1e-4)

    def test_a_trial_sees_the_same_noise_under_every_policy(self):
        a = catch.simulate_catch(catch.PredictOncePolicy, sigma_p=0.005, trial=11)
        b = catch.simulate_catch(catch.OpenLoopPolicy, sigma_p=0.005, trial=11)
        np.testing.assert_allclose(
            [o.p for o in a.controller.fit.observations[:10]],
            [o.p for o in b.controller.fit.observations[:10]], atol=SAME)


if __name__ == "__main__":
    unittest.main()
