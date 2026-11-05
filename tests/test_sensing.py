"""The ball sensor and the one-shot fit (spec deliverable A1)."""

import unittest

import numpy as np
from pydrake.all import RigidTransform, Simulator, SpatialVelocity

import feasibility
import sensing
import toss

P0 = np.array([0.8, 0.0, 0.774])
V0 = np.array([-0.353, -0.528, 2.645])
T_CATCH = 0.575
WINDOW_TIMES = 0.01 * np.arange(1, 11)  # section 14: 10 samples, the release sample excluded


def observations(times, rng=None, sigma_p=0.0, sigma_v=None):
    """Observations of the flight from (P0, V0) at `times`, with t_anchor = 0."""
    rng = rng or np.random.default_rng(0)
    out = []
    for t in times:
        p = toss.ballistic_position(P0, V0, t) + sigma_p * rng.normal(size=3)
        v = None
        if sigma_v is not None:
            v = V0 + np.array([0, 0, -toss.G_ACCEL]) * t + sigma_v * rng.normal(size=3)
        out.append(sensing.Observation(float(t), p, v))
    return out


def fit(obs, **kwargs):
    ballistic = sensing.BallisticFit(t_anchor=0.0, **kwargs)
    for o in obs:
        ballistic.add(o)
    return ballistic.estimate()


class BallisticFitTest(unittest.TestCase):
    def test_recovers_release_state_without_noise(self):
        est = fit(observations(WINDOW_TIMES), sigma_p=0.0)
        self.assertLess(np.abs(est.p0 - P0).max(), 1e-3)
        self.assertLess(np.abs(est.v0 - V0).max(), 1e-3)
        self.assertEqual(est.n_obs, 10)
        self.assertEqual(est.t_anchor, 0.0)
        self.assertEqual(est.cov.shape, (6, 6))

    def test_catch_point_error_matches_the_analytic_prediction(self):
        rng = np.random.default_rng(1)
        expected = feasibility.prediction_error(0.10, T_CATCH, False, t_first=0.01)
        truth = toss.ballistic_position(P0, V0, T_CATCH)
        errors = []
        for _ in range(1000):
            est = fit(observations(WINDOW_TIMES, rng, sigma_p=0.005), sigma_p=0.005)
            errors.append(toss.ballistic_position(est.p0, est.v0, T_CATCH) - truth)
        spread = np.std(errors, axis=0)
        self.assertAlmostEqual(expected, 0.0287, delta=0.0005)
        for axis in range(3):
            self.assertLess(abs(spread[axis] / expected - 1), 0.10, f"axis {axis}: {spread}")

    def test_velocity_channel_matches_the_analytic_prediction(self):
        rng = np.random.default_rng(2)
        expected = feasibility.prediction_error(0.10, T_CATCH, True, t_first=0.01)
        truth = toss.ballistic_position(P0, V0, T_CATCH)
        errors = []
        for _ in range(1000):
            obs = observations(WINDOW_TIMES, rng, sigma_p=0.005, sigma_v=0.05)
            est = fit(obs, sigma_p=0.005, sigma_v=0.05)
            errors.append(toss.ballistic_position(est.p0, est.v0, T_CATCH) - truth)
        spread = np.std(errors, axis=0)
        for axis in range(3):
            self.assertLess(abs(spread[axis] / expected - 1), 0.10, f"axis {axis}: {spread}")

    def test_covariance_predicts_the_catch_point_error(self):
        est = fit(observations(WINDOW_TIMES), sigma_p=0.005)
        x = np.array([1.0, T_CATCH])
        block = est.cov[np.ix_([0, 3], [0, 3])]
        self.assertAlmostEqual(np.sqrt(x @ block @ x), 0.0287, delta=0.0005)

    def test_too_few_samples_give_no_estimate(self):
        self.assertIsNone(fit([], sigma_p=0.005))
        self.assertIsNone(fit(observations([0.01]), sigma_p=0.005))
        self.assertIsNotNone(fit(observations([0.01, 0.02]), sigma_p=0.005))

    def test_observation_times_are_measured_from_the_anchor(self):
        shifted = [sensing.Observation(o.t + 1.0, o.p, o.v) for o in observations(WINDOW_TIMES)]
        ballistic = sensing.BallisticFit(t_anchor=1.0, sigma_p=0.0)
        for o in shifted:
            ballistic.add(o)
        est = ballistic.estimate()
        self.assertLess(np.abs(est.p0 - P0).max(), 1e-3)
        self.assertLess(np.abs(est.v0 - V0).max(), 1e-3)
        self.assertEqual(est.t_anchor, 1.0)


class RecursiveFitTest(unittest.TestCase):
    """Spec deliverable B1: algebraically the same estimator as the batch fit, so a mismatch is a bug."""

    FLIGHT_TIMES = 0.01 * np.arange(1, 48)

    def assert_matches_the_batch_fit(self, obs, t_anchor=0.0, **kwargs):
        batch = sensing.BallisticFit(t_anchor, **kwargs)
        recursive = sensing.RecursiveFit(t_anchor, **kwargs)
        for o in obs:
            batch.add(o)
            recursive.add(o)
            a, b = batch.estimate(), recursive.estimate()
            if a is None:
                self.assertIsNone(b)
                continue
            np.testing.assert_allclose(b.p0, a.p0, rtol=0, atol=1e-9)
            np.testing.assert_allclose(b.v0, a.v0, rtol=0, atol=1e-9)
            np.testing.assert_allclose(b.cov, a.cov, rtol=0, atol=1e-9)
            self.assertEqual((b.n_obs, b.t_anchor), (a.n_obs, a.t_anchor))

    def test_matches_the_batch_fit_after_every_sample(self):
        rng = np.random.default_rng(3)
        self.assert_matches_the_batch_fit(observations(self.FLIGHT_TIMES, rng, sigma_p=0.002), sigma_p=0.002)

    def test_matches_it_with_a_velocity_channel(self):
        rng = np.random.default_rng(4)
        obs = observations(self.FLIGHT_TIMES, rng, sigma_p=0.005, sigma_v=0.05)
        self.assert_matches_the_batch_fit(obs, sigma_p=0.005, sigma_v=0.05)

    def test_matches_it_with_a_perfect_sensor(self):
        self.assert_matches_the_batch_fit(observations(self.FLIGHT_TIMES), sigma_p=0.0)

    def test_matches_it_from_a_later_anchor(self):
        shifted = [sensing.Observation(o.t + 1.0, o.p, o.v) for o in observations(self.FLIGHT_TIMES)]
        self.assert_matches_the_batch_fit(shifted, t_anchor=1.0, sigma_p=0.002)

    def test_too_few_samples_give_no_estimate(self):
        recursive = sensing.RecursiveFit(0.0, sigma_p=0.002)
        self.assertIsNone(recursive.estimate())
        recursive.add(observations([0.01])[0])
        self.assertIsNone(recursive.estimate())
        recursive.add(observations([0.02])[0])
        self.assertIsNotNone(recursive.estimate())


class BallSensorTest(unittest.TestCase):
    BALL = 2
    TRUE_P = np.array([0.1, 0.2, 0.3])
    TRUE_V = np.array([0.4, 0.5, 0.6])

    def run_sensor(self, seed, duration=0.05, read_every=0.005, **kwargs):
        """Run a sensor against a fixed ball; return its output at each read."""
        sensor = sensing.BallSensor(self.BALL, np.random.default_rng(seed), **kwargs)
        simulator = Simulator(sensor)
        context = simulator.get_mutable_context()
        poses = [RigidTransform()] * 3
        poses[self.BALL] = RigidTransform(self.TRUE_P)
        velocities = [SpatialVelocity(np.zeros(3), np.zeros(3))] * 3
        velocities[self.BALL] = SpatialVelocity(np.zeros(3), self.TRUE_V)
        sensor.GetInputPort("body_poses").FixValue(context, poses)
        sensor.GetInputPort("body_spatial_velocities").FixValue(context, velocities)
        reads = []
        for t in np.arange(read_every, duration + 1e-9, read_every):
            simulator.AdvanceTo(t)
            reads.append(sensor.GetOutputPort("observation").Eval(context))
        return reads

    def test_publishes_at_100_hz_with_one_draw_per_sample(self):
        reads = self.run_sensor(seed=3, sigma_p=0.005)
        stamps = [o.t for o in reads]
        self.assertEqual(len(set(np.round(stamps, 6))), len(reads) // 2)  # each sample read twice
        for a, b in zip(reads, reads[1:]):
            if a.t == b.t:
                np.testing.assert_array_equal(a.p, b.p)
            else:
                self.assertAlmostEqual(b.t - a.t, 0.01, places=9)
                self.assertFalse(np.array_equal(a.p, b.p))

    def test_position_only_has_no_velocity(self):
        self.assertTrue(all(o.v is None for o in self.run_sensor(seed=4, sigma_p=0.005)))

    def test_noiseless_reads_the_truth(self):
        for o in self.run_sensor(seed=5, sigma_p=0.0, sigma_v=0.0, mode=sensing.POSITION_AND_VELOCITY):
            np.testing.assert_array_equal(o.p, self.TRUE_P)
            np.testing.assert_array_equal(o.v, self.TRUE_V)

    def test_same_seed_same_noise(self):
        a, b = self.run_sensor(seed=6), self.run_sensor(seed=6)
        c = self.run_sensor(seed=7)
        np.testing.assert_array_equal([o.p for o in a], [o.p for o in b])
        self.assertFalse(np.array_equal([o.p for o in a], [o.p for o in c]))

    def test_position_noise_does_not_depend_on_the_mode(self):
        """Switching on the velocity channel must not change the position noise."""
        a = self.run_sensor(seed=8)
        b = self.run_sensor(seed=8, mode=sensing.POSITION_AND_VELOCITY)
        np.testing.assert_array_equal([o.p for o in a], [o.p for o in b])


if __name__ == "__main__":
    unittest.main()
