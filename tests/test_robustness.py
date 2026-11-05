"""The robustness sweep's cells, and how its tables pick them out."""

import unittest

import numpy as np

import catch
import robustness
from robustness import A, B, BASELINE
import sensing
from tests.test_replan import row as result_row


def has(configs, approach, **settings):
    wanted = {**robustness.NOMINAL, "obs_mode": sensing.POSITION_ONLY, "latency": catch.T_LATENCY, **settings}
    if approach == BASELINE:
        wanted["sigma_p"] = 0.0
    return any(c.approach == approach and all(getattr(c, name) == value for name, value in wanted.items()) for c in configs)


class GridTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs = robustness.grid()

    def test_no_trial_runs_twice(self):
        self.assertEqual(len(set(self.configs)), len(self.configs))

    def test_every_cell_has_100_trials(self):
        counts = {}
        for c in self.configs:
            key = (c.approach, c.sigma_p, c.sigma_lat, c.sigma_vz, c.latency, c.obs_mode, c.trial // 100)
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(set(counts.values()), {100})

    def test_every_level_of_every_factor_is_there(self):
        for approach in (A, B):
            for sigma_p in robustness.SIGMAS_P:
                self.assertTrue(has(self.configs, approach, sigma_p=sigma_p), sigma_p)
            self.assertTrue(has(self.configs, approach, obs_mode=sensing.POSITION_AND_VELOCITY))
        for approach in (BASELINE, A, B):
            for sigma_lat in robustness.SIGMAS_LAT:
                self.assertTrue(has(self.configs, approach, sigma_lat=sigma_lat), sigma_lat)
            for sigma_vz in robustness.SIGMAS_VZ:
                self.assertTrue(has(self.configs, approach, sigma_vz=sigma_vz), sigma_vz)
            for latency in robustness.LATENCIES:
                self.assertTrue(has(self.configs, approach, latency=latency), latency)
            self.assertTrue(has(self.configs, approach, sigma_p=0.0, sigma_lat=0.0, sigma_vz=0.0))

    def test_toss_noise_and_sensor_noise_also_move_together(self):
        for sigma_lat in robustness.CROSS_SIGMAS_LAT:
            self.assertTrue(has(self.configs, BASELINE, sigma_lat=sigma_lat))
            for sigma_p in robustness.CROSS_SIGMAS_P:
                for approach in (A, B):
                    self.assertTrue(has(self.configs, approach, sigma_p=sigma_p, sigma_lat=sigma_lat), (sigma_p, sigma_lat))

    def test_the_nominal_cell_repeats_on_tosses_it_wasnt_chosen_from(self):
        for approach in (BASELINE, A, B):
            self.assertEqual(sum(c.approach == approach and c.trial >= 100 for c in self.configs), 100)

    def test_a_latency_sweep_keeps_the_trigger_at_the_nominal_latency(self):
        """If the trigger were told the real latency, sweeping it would change nothing."""
        for c in self.configs:
            if c.latency != catch.T_LATENCY:
                self.assertEqual(c.assumed_latency, robustness.ASSUMED_LATENCY)
            else:
                self.assertIsNone(c.assumed_latency)

    def test_the_baseline_ignores_the_sensor_so_it_runs_once(self):
        self.assertTrue(all(c.sigma_p == 0.0 for c in self.configs if c.approach == BASELINE))
        self.assertFalse(any(c.approach == BASELINE and c.obs_mode != sensing.POSITION_ONLY for c in self.configs))


def rows(approach, caught, **settings):
    return [result_row(approach=approach, trial=t, caught=c, **settings) for t, c in enumerate(caught)]


class TablesTest(unittest.TestCase):
    def test_a_cell_is_picked_out_by_its_settings(self):
        results = rows(A, [True]) + rows(A, [False], sigma_p=0.005) + rows(A, [False], latency=0.05)
        self.assertEqual(robustness.select(results, A).caught, 1)
        self.assertEqual(len(robustness.select(results, A).results), 1)
        self.assertEqual(len(robustness.select(results, A, sigma_p=0.005).results), 1)
        self.assertEqual(len(robustness.select(results, A, latency=0.05).results), 1)

    def test_fresh_tosses_are_kept_apart(self):
        results = rows(A, [True]) + [result_row(approach=A, trial=100, caught=False)]
        self.assertEqual(robustness.select(results, A).caught, 1)
        self.assertEqual(robustness.select(results, A, trial_set=1).caught, 0)

    def test_the_baseline_is_found_whatever_the_sensor_level(self):
        results = rows(BASELINE, [True], sigma_p=0.0)
        self.assertEqual(robustness.select(results, BASELINE, sigma_p=0.010).caught, 1)

    def test_a_row_reports_the_pairing_against_a(self):
        a = rows(A, [True, False, False, True])
        b = rows(B, [True, True, True, True])
        base = rows(BASELINE, [False, False, False, False], sigma_p=0.0)
        table = robustness.factor_table(a + b + base, "sigma_p", (0.002,), "sigma_p", str)
        self.assertIn("2 / 0, p = 0.5", table[2])

    def test_movement_needed_is_how_far_the_ball_lands_from_where_the_brake_ends(self):
        plan = robustness.toss.plan_toss(catch.shared_kinematics().plant)
        t, p = robustness.toss.catch_point(plan)
        r = result_row(p_release_true_x=plan.p_release[0], p_release_true_y=plan.p_release[1], p_release_true_z=plan.p_release[2],
                        v_release_true_x=plan.v_release[0], v_release_true_y=plan.v_release[1], v_release_true_z=plan.v_release[2])
        start = catch.shared_kinematics().pinch(plan.q_traj.value(plan.t_release + robustness.toss.BRAKE).ravel())
        self.assertAlmostEqual(robustness.movement_needed([r])[0], np.linalg.norm(p - start), places=6)


if __name__ == "__main__":
    unittest.main()
