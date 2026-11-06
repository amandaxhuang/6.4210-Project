"""The robustness sweep's cells, and how its tables pick them out."""

from dataclasses import fields
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import catch
import figures
import robustness
from robustness import A, B, BASELINE
import sensing
from tests.test_replan import row as result_row


def has(configs, approach, **settings):
    wanted = {**robustness.NOMINAL, "obs_mode": sensing.POSITION_ONLY, "latency": catch.T_LATENCY,
              "sigma_latency": 0.0, **settings}
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
            key = (c.approach, c.sigma_p, c.sigma_lat, c.sigma_vz, c.latency, c.sigma_latency, c.obs_mode, c.trial // 100)
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
            for sigma_latency in robustness.LATENCY_SIGMAS:
                self.assertTrue(has(self.configs, approach, latency=robustness.LATENCY_MEAN,
                                    sigma_latency=sigma_latency), sigma_latency)
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
        """If the trigger were told the real latency, sweeping it would change nothing.
        A cell that draws its delay cannot tell the trigger either, even at the nominal."""
        for c in self.configs:
            if c.latency != catch.T_LATENCY or c.sigma_latency:
                self.assertEqual(c.assumed_latency, robustness.ASSUMED_LATENCY)
            else:
                self.assertIsNone(c.assumed_latency)

    def test_every_drawn_latency_cell_shares_the_same_mean(self):
        """So the spread is the only thing that moves between them."""
        drawn = [c for c in self.configs if c.sigma_latency]
        self.assertTrue(drawn)
        self.assertEqual({c.latency for c in drawn}, {robustness.LATENCY_MEAN})
        self.assertEqual({c.sigma_latency for c in drawn}, set(robustness.LATENCY_SIGMAS))

    def test_the_constant_levels_cover_5_to_50_ms_in_5_ms_steps(self):
        self.assertEqual(robustness.LATENCIES, (0.0,) + tuple(round(0.005 * k, 3) for k in range(1, 11)))

    def test_every_setting_survives_the_csv_round_trip(self):
        """A cell is selected out of the saved rows by comparing its settings exactly, so a
        setting that write_csv rounds is a cell that can be run but never read back again."""
        for config in self.configs:
            for field in fields(config):
                value = getattr(config, field.name)
                if isinstance(value, float):
                    self.assertEqual(float(f"{value:.12g}"), value, f"{field.name} = {value!r}")

    def test_the_wider_sigma_still_spans_5_to_50_ms(self):
        """Rounding it must not move the percentiles it was chosen for."""
        drawn = np.array([catch.close_latency(t, robustness.LATENCY_MEAN, robustness.LATENCY_SIGMAS[1])
                          for t in range(20000)])
        self.assertAlmostEqual(np.percentile(drawn, 5), 0.005, delta=5e-4)
        self.assertAlmostEqual(np.percentile(drawn, 95), 0.050, delta=5e-4)

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

    def test_a_fixed_cell_does_not_pick_up_rows_that_drew_their_delay(self):
        """They can share a mean, so select has to filter on the spread as well."""
        results = rows(A, [True], latency=0.0275) + rows(A, [False], latency=0.0275, sigma_latency=0.0025)
        self.assertEqual(len(robustness.select(results, A, latency=0.0275).results), 1)
        self.assertEqual(robustness.select(results, A, latency=0.0275).caught, 1)
        drawn = robustness.select(results, A, latency=0.0275, sigma_latency=0.0025)
        self.assertEqual(len(drawn.results), 1)
        self.assertEqual(drawn.caught, 0)

    def test_the_draw_table_reports_what_each_cell_asked_for(self):
        results = []
        for approach in (BASELINE, A, B):
            for trial in range(20):
                drawn = catch.close_latency(trial, robustness.LATENCY_MEAN, robustness.LATENCY_SIGMAS[0])
                results.append(result_row(approach=approach, trial=trial, sigma_p=0.0 if approach == BASELINE else robustness.NOMINAL["sigma_p"],
                                          latency=robustness.LATENCY_MEAN, sigma_latency=robustness.LATENCY_SIGMAS[0],
                                          assumed_latency=robustness.ASSUMED_LATENCY, latency_drawn=drawn))
        table = robustness.latency_draw_table(results)
        self.assertEqual(len(table), 3)  # a header, its rule, and one cell
        self.assertIn("| yes |", table[2])  # every approach drew the same delay

    def test_the_baseline_is_found_whatever_the_sensor_level(self):
        results = rows(BASELINE, [True], sigma_p=0.0)
        self.assertEqual(robustness.select(results, BASELINE, sigma_p=0.010).caught, 1)

    def test_a_row_reports_the_pairing_against_a(self):
        a = rows(A, [True, False, False, True])
        b = rows(B, [True, True, True, True])
        base = rows(BASELINE, [False, False, False, False], sigma_p=0.0)
        table = robustness.factor_table(a + b + base, "sigma_p", (0.002,), "sigma_p", str)
        self.assertIn("2 / 0, p = 0.5", table[2])

    def test_where_the_ball_ends_counts_only_caught_balls(self):
        a = [result_row(approach=A, trial=t, caught=c, end_fingers=e) for t, (c, e) in enumerate(((True, 0.012), (True, 0.004), (True, 0.006), (False, -0.5)))]
        b = rows(B, [False, False, False, False])
        table = robustness.factor_table(a + b, "sigma_p", (0.002,), "sigma_p", str, approaches=(A, B))
        self.assertIn("| 0.60 / - |", table[2])

    def test_movement_needed_is_how_far_the_ball_lands_from_where_the_brake_ends(self):
        plan = robustness.toss.plan_toss(catch.shared_kinematics().plant)
        t, p = robustness.toss.catch_point(plan)
        r = result_row(p_release_true_x=plan.p_release[0], p_release_true_y=plan.p_release[1], p_release_true_z=plan.p_release[2],
                        v_release_true_x=plan.v_release[0], v_release_true_y=plan.v_release[1], v_release_true_z=plan.v_release[2])
        start = catch.shared_kinematics().pinch(plan.q_traj.value(plan.t_release + robustness.toss.BRAKE).ravel())
        self.assertAlmostEqual(robustness.movement_needed([r])[0], np.linalg.norm(p - start), places=6)


class ExtendTest(unittest.TestCase):
    def test_only_the_trials_not_saved_are_run(self):
        configs = robustness.cells((A, B), range(2)) + robustness.cells((A, B), range(2), sigma_lat=0.12)
        saved = [result_row(**{f.name: getattr(c, f.name) for f in fields(c)}) for c in configs[:4]]
        self.assertEqual(robustness.missing(configs, saved), configs[4:])

    def test_a_saved_row_with_any_setting_different_is_not_the_same_trial(self):
        config = robustness.cells((B,), range(1))[0]
        saved = result_row(**{**{f.name: getattr(config, f.name) for f in fields(config)}, "deadline": 0.35})
        self.assertEqual(robustness.missing([config], [saved]), [config])


class FiguresTest(unittest.TestCase):
    def test_every_figure_is_written(self):
        results = [result_row(approach=c.approach, trial=c.trial, sigma_p=c.sigma_p, sigma_lat=c.sigma_lat, sigma_vz=c.sigma_vz, latency=c.latency,
                              obs_mode=c.obs_mode, caught=c.trial % 3 == 0, plans_adopted=3, plan_drift=0.001 * (c.trial % 40),
                              sigma_latency=c.sigma_latency, assumed_latency=c.assumed_latency,
                              end_fingers=0.009 + 0.0001 * (c.trial % 5),
                              latency_drawn=catch.close_latency(c.trial, c.latency, c.sigma_latency))
                   for c in robustness.grid() if c.trial % 100 < 8]
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(figures, "OUT", Path(folder)):
            paths = figures.make_all(results)
            self.assertEqual(len(paths), 10)
            for path in paths:
                self.assertGreater(path.stat().st_size, 5000)


if __name__ == "__main__":
    unittest.main()
