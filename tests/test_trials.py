"""The trial harness: the trial record, the CSV, and the summary (spec deliverable A5)."""

import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import catch
import toss
import trials


class WilsonTest(unittest.TestCase):
    def test_contains_the_observed_rate(self):
        low, high = trials.wilson(50, 100)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)
        self.assertAlmostEqual(low, 0.404, delta=0.002)
        self.assertAlmostEqual(high, 0.596, delta=0.002)

    def test_stays_inside_zero_and_one(self):
        self.assertEqual(trials.wilson(0, 100)[0], 0.0)
        self.assertEqual(trials.wilson(100, 100)[1], 1.0)

    def test_no_trials_says_nothing(self):
        self.assertEqual(trials.wilson(0, 0), (0.0, 1.0))


class McNemarTest(unittest.TestCase):
    def test_identical_outcomes_show_no_difference(self):
        self.assertEqual(trials.mcnemar([True, False, True], [True, False, True]), (0, 0, 1.0))

    def test_all_disagreements_one_way(self):
        only_a, only_b, p = trials.mcnemar([True] * 10, [False] * 10)
        self.assertEqual((only_a, only_b), (10, 0))
        self.assertAlmostEqual(p, 2 * 0.5**10)

    def test_balanced_disagreements_show_no_difference(self):
        self.assertEqual(trials.mcnemar([True, False], [False, True])[2], 1.0)


class RunTrialTest(unittest.TestCase):
    """A perfect sensor, so the toss noise is the only error."""

    @classmethod
    def setUpClass(cls):
        cls.noisy = trials.TrialConfig("predict_once", trial=5, sigma_p=0.0, sigma_lat=0.02, sigma_vz=0.02)
        cls.a = trials.run_trial(cls.noisy)
        cls.quiet = trials.run_trial(trials.TrialConfig("predict_once", trial=5, sigma_p=0.0, sigma_lat=0.0, sigma_vz=0.0))
        cls.baseline = trials.run_trial(trials.TrialConfig("open_loop", trial=5, sigma_p=0.0, sigma_lat=0.02, sigma_vz=0.02))

    def test_the_record_names_its_cell(self):
        self.assertEqual(self.a.approach, "predict_once")
        self.assertEqual(self.a.trial, 5)
        self.assertEqual(self.a.seed, catch.SEED_ROOT)
        self.assertEqual(self.a.sigma_lat, 0.02)
        self.assertEqual(self.a.obs_mode, "position_only")
        self.assertEqual(self.a.latency, catch.T_LATENCY)

    def test_the_record_separates_estimation_from_tracking(self):
        self.assertLess(self.a.predicted_miss, 0.002)  # a perfect sensor predicts it
        self.assertGreater(self.a.arm_arrival_error, 0.002)  # the arm still lags
        self.assertGreater(self.baseline.predicted_miss, 0.002)  # not looking misses the toss noise

    def test_the_record_measures_what_the_toss_really_did(self):
        moved = np.array([self.a.v_release_true_x - self.quiet.v_release_true_x,
                          self.a.v_release_true_y - self.quiet.v_release_true_y,
                          self.a.v_release_true_z - self.quiet.v_release_true_z])
        np.testing.assert_allclose(moved, catch.toss_velocity_noise(5, 0.02, 0.02), atol=1e-6)
        self.assertAlmostEqual(self.quiet.p_release_true_z, 0.771, delta=0.005)

    def test_the_fairness_audit_is_recorded(self):
        self.assertEqual(self.a.n_solves, 1)
        self.assertEqual(self.a.plans_adopted, 1)
        self.assertGreater(self.a.policy_calls, 100)
        self.assertTrue(self.a.solve_ok)
        self.assertFalse(self.a.fit_rank_deficient)
        self.assertFalse(self.a.intercept_rejected)
        self.assertGreater(self.a.solve_time, 0.0)
        self.assertGreater(self.a.wall_time, self.a.solve_time)

    def test_both_catch_definitions_are_recorded(self):
        self.assertEqual(self.a.caught_on_pad, self.a.caught and catch.ball_on_pad(
            np.array([0.0, self.a.end_fingers, self.a.end_across])))
        self.assertEqual(self.quiet.caught_on_pad, False)  # the default noiseless catch is a perch
        self.assertTrue(self.quiet.caught)


class WindowTest(unittest.TestCase):
    def test_approach_a_observes_for_the_window_it_is_given(self):
        short = trials.run_trial(trials.TrialConfig("predict_once", trial=1, sigma_p=0.0, sigma_lat=0.0, sigma_vz=0.0))
        long = trials.run_trial(trials.TrialConfig("predict_once", trial=1, sigma_p=0.0, sigma_lat=0.0, sigma_vz=0.0,
                                                   t_window=0.15))
        self.assertEqual((short.t_window, short.n_obs), (catch.T_WINDOW, 10))
        self.assertEqual((long.t_window, long.n_obs), (0.15, 15))

    def test_a_trial_records_how_fast_the_arm_went(self):
        moved = trials.run_trial(trials.TrialConfig("open_loop", trial=1, sigma_p=0.0, sigma_lat=0.0, sigma_vz=0.0))
        stayed = trials.run_trial(trials.TrialConfig("predict_once", trial=1, sigma_p=0.0, sigma_lat=0.0, sigma_vz=0.0,
                                                     t_window=0.15))
        self.assertGreater(moved.peak_speed, catch.PLAN_SPEED_LIMIT)
        self.assertTrue(math.isnan(stayed.peak_speed))

    def test_cells_count_the_catches_within_the_speed_tolerance(self):
        rows = [record(trial=0, caught=True, peak_speed=1.0), record(trial=1, caught=True, peak_speed=1.06),
                record(trial=2, caught=False, peak_speed=1.0), record(trial=3, caught=False, solve_ok=False, peak_speed=math.nan)]
        cell = trials.summarize(rows)[0]
        self.assertEqual(cell.caught_within_speed, 1)
        self.assertAlmostEqual(cell.peak_median, 1.0)
        self.assertAlmostEqual(cell.peak_worst, 1.06)

    def test_a_cell_with_no_motions_has_no_peak_speed(self):
        cell = trials.summarize([record(caught=False, solve_ok=False, peak_speed=math.nan)])[0]
        self.assertTrue(math.isnan(cell.peak_median))
        self.assertTrue(math.isnan(cell.peak_worst))

    def test_the_baseline_uses_no_observations(self):
        run = trials.run_trial(trials.TrialConfig("open_loop", trial=1, sigma_p=0.0, sigma_lat=0.0, sigma_vz=0.0))
        self.assertEqual(run.n_obs, 0)

    def test_windows_are_separate_cells_paired_with_the_same_baseline(self):
        baseline = [record(approach="open_loop", sigma_p=0.0, trial=t, caught=False) for t in range(4)]
        short = [record(trial=t, t_window=0.10, caught=t < 1) for t in range(4)]
        long = [record(trial=t, t_window=0.15, caught=t < 3) for t in range(4)]
        cells = {(c.approach, c.t_window): c for c in trials.summarize(baseline + short + long)}
        self.assertEqual(cells[("predict_once", 0.10)].caught, 1)
        self.assertEqual(cells[("predict_once", 0.15)].caught, 3)
        self.assertEqual(cells[("predict_once", 0.15)].only_this, 3)

    def test_the_grid_has_the_nominal_cell_so_it_can_be_repeated(self):
        self.assertIn(trials.TrialConfig("predict_once", trial=7), trials.grid())

    def test_the_grid_has_a_perfect_sensor_at_each_window_with_and_without_toss_noise(self):
        configs = trials.grid()
        for t_window in trials.GRID_WINDOWS:
            for sigma_lat, sigma_vz in [(0.0, 0.0)] + [(s, trials.GRID_SIGMA_VZ) for s in trials.GRID_SIGMA_LAT]:
                self.assertIn(trials.TrialConfig("predict_once", sigma_p=0.0, sigma_lat=sigma_lat, sigma_vz=sigma_vz,
                                                 t_window=t_window), configs)

    def test_the_grid_has_both_windows_and_every_toss_noise_has_a_baseline(self):
        configs = trials.grid()
        windows = {c.t_window for c in configs if c.approach == "predict_once"}
        self.assertEqual(windows, set(trials.GRID_WINDOWS))
        for sigma_lat in trials.GRID_SIGMA_LAT:
            self.assertTrue(any(c.approach == "open_loop" and c.sigma_lat == sigma_lat for c in configs))
        self.assertTrue(any(c.approach == "predict_once" and c.sigma_p == 0 and c.sigma_lat == 0 for c in configs))


class SweepTest(unittest.TestCase):
    def test_parallel_trials_match_and_keep_their_order(self):
        configs = [trials.TrialConfig("predict_once", trial=t, sigma_p=0.002) for t in (3, 1)]
        swept = trials.sweep(configs, n_workers=2)
        self.assertEqual([r.trial for r in swept], [3, 1])
        alone = trials.run_trial(configs[0])
        self.assertAlmostEqual(swept[0].miss_horizontal, alone.miss_horizontal, delta=trials.SAME_AFTER_SOLVE)
        self.assertEqual(swept[0].caught, alone.caught)


def record(**fields):
    defaults = dict(approach="predict_once", trial=0, seed=catch.SEED_ROOT, t_window=0.10, n_obs=10, sigma_p=0.005, sigma_v=0.05,
                    obs_mode="position_only", sigma_lat=0.02, sigma_vz=0.02, latency=0.03, caught=True,
                    caught_on_pad=False, contact=True, hold_time=1.0, miss_horizontal=0.01, miss_vertical=0.0,
                    miss_closing=0.0, miss_fingers=0.0, miss_across=0.0, end_fingers=0.015, end_across=0.0,
                    timing_error=0.001, predicted_miss=0.02, arm_arrival_error=0.006, peak_speed=0.96, solve_ok=True, solve_time=0.3,
                    wall_time=3.0, n_solves=1, policy_calls=320, plans_adopted=1, fit_rank_deficient=False,
                    intercept_rejected=False, p_release_true_x=0.8, p_release_true_y=0.0, p_release_true_z=0.77,
                    v_release_true_x=-0.33, v_release_true_y=-0.53, v_release_true_z=2.65)
    return trials.TrialResult(**{**defaults, **fields})


class CsvTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "a.csv"
        self.other = Path(self.dir.name) / "b.csv"

    def test_a_record_survives_the_round_trip(self):
        trials.write_csv([record(miss_vertical=math.nan)], self.path)
        row = trials.read_csv(self.path)[0]
        self.assertEqual(row["approach"], "predict_once")
        self.assertEqual(row["caught"], "True")
        self.assertEqual(float(row["predicted_miss"]), 0.02)
        self.assertTrue(math.isnan(float(row["miss_vertical"])))

    def test_the_same_results_compare_equal(self):
        trials.write_csv([record(), record(trial=1)], self.path)
        trials.write_csv([record(), record(trial=1)], self.other)
        self.assertEqual(trials.csv_differences(self.path, self.other), [])

    def test_differences_below_the_tolerance_are_ignored(self):
        trials.write_csv([record(predicted_miss=0.02)], self.path)
        trials.write_csv([record(predicted_miss=0.02 + 1e-12)], self.other)
        self.assertEqual(trials.csv_differences(self.path, self.other), [])

    def test_differences_above_the_tolerance_are_reported(self):
        trials.write_csv([record(predicted_miss=0.02)], self.path)
        trials.write_csv([record(predicted_miss=0.021)], self.other)
        self.assertEqual(len(trials.csv_differences(self.path, self.other)), 1)

    def test_what_the_fit_saw_compares_tightly(self):
        trials.write_csv([record(predicted_miss=0.02)], self.path)
        trials.write_csv([record(predicted_miss=0.02 + 1e-4)], self.other)
        self.assertEqual(len(trials.csv_differences(self.path, self.other)), 1)

    def test_a_crossing_extrapolated_from_an_early_touch_may_differ_a_little(self):
        trials.write_csv([record(predicted_miss=0.02, timing_error=0.001)], self.path)
        trials.write_csv([record(predicted_miss=0.02 + 2e-6, timing_error=0.001 + 2e-6)], self.other)
        self.assertEqual(trials.csv_differences(self.path, self.other), [])

    def test_the_arm_may_differ_a_little_after_the_solve(self):
        trials.write_csv([record(miss_fingers=0.01, arm_arrival_error=0.006)], self.path)
        trials.write_csv([record(miss_fingers=0.01 + 2e-4, arm_arrival_error=0.006 + 2e-4)], self.other)
        self.assertEqual(trials.csv_differences(self.path, self.other), [])

    def test_the_peak_speed_may_differ_with_the_plan(self):
        trials.write_csv([record(peak_speed=0.96)], self.path)
        trials.write_csv([record(peak_speed=0.963)], self.other)
        self.assertEqual(trials.csv_differences(self.path, self.other), [])

    def test_the_hold_time_may_differ_by_a_time_step(self):
        trials.write_csv([record(hold_time=0.96)], self.path)
        trials.write_csv([record(hold_time=0.961)], self.other)
        self.assertEqual(trials.csv_differences(self.path, self.other), [])

    def test_the_arm_differing_by_more_than_that_is_reported(self):
        trials.write_csv([record(miss_fingers=0.01)], self.path)
        trials.write_csv([record(miss_fingers=0.01 + 6e-3)], self.other)
        self.assertEqual(len(trials.csv_differences(self.path, self.other)), 1)

    def test_a_dropped_ball_may_come_to_rest_elsewhere(self):
        trials.write_csv([record(caught=False, end_across=-1.0)], self.path)
        trials.write_csv([record(caught=False, end_across=-1.012)], self.other)
        self.assertEqual(trials.csv_differences(self.path, self.other), [])

    def test_a_held_ball_has_to_end_in_the_same_place(self):
        trials.write_csv([record(caught=True, end_across=0.01)], self.path)
        trials.write_csv([record(caught=True, end_across=0.016)], self.other)
        self.assertEqual(len(trials.csv_differences(self.path, self.other)), 1)

    def test_a_held_ball_may_end_a_little_elsewhere(self):
        trials.write_csv([record(caught=True, end_fingers=0.01)], self.path)
        trials.write_csv([record(caught=True, end_fingers=0.012)], self.other)
        self.assertEqual(trials.csv_differences(self.path, self.other), [])

    def test_a_changed_outcome_is_reported(self):
        trials.write_csv([record(caught=True)], self.path)
        trials.write_csv([record(caught=False)], self.other)
        self.assertEqual(len(trials.csv_differences(self.path, self.other)), 1)

    def test_wall_clock_columns_are_ignored(self):
        trials.write_csv([record(solve_time=0.3, wall_time=3.0)], self.path)
        trials.write_csv([record(solve_time=0.5, wall_time=4.0)], self.other)
        self.assertEqual(trials.csv_differences(self.path, self.other), [])

    def test_a_missing_row_is_reported(self):
        trials.write_csv([record(), record(trial=1)], self.path)
        trials.write_csv([record()], self.other)
        self.assertTrue(trials.csv_differences(self.path, self.other))


class RepeatCheckTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.saved = Path(self.dir.name) / "saved.csv"
        self.config = trials.TrialConfig("predict_once", trial=4, sigma_p=0.003, sigma_v=0.05, t_window=0.15)

    def rerun(self, **fields):
        return mock.patch.object(trials, "sweep", return_value=[record(trial=4, sigma_p=0.003, t_window=0.15, **fields)])

    def test_only_the_rows_it_ran_are_compared(self):
        other_cell = record(trial=4, sigma_p=0.005, t_window=0.15, caught=False)
        trials.write_csv([other_cell, record(trial=4, sigma_p=0.003, t_window=0.15)], self.saved)
        with self.rerun():
            self.assertEqual(trials.repeat_check([self.config], self.saved), [])

    def test_a_changed_outcome_is_reported(self):
        trials.write_csv([record(trial=4, sigma_p=0.003, t_window=0.15)], self.saved)
        with self.rerun(caught=False):
            self.assertEqual(len(trials.repeat_check([self.config], self.saved)), 1)

    def test_a_row_missing_from_the_saved_file_is_reported(self):
        trials.write_csv([record(trial=5, sigma_p=0.003, t_window=0.15)], self.saved)
        with self.rerun():
            self.assertEqual(trials.repeat_check([self.config], self.saved), ["0 rows against 1"])


class SummaryTest(unittest.TestCase):
    def results(self):
        a = [record(trial=t, caught=t < 6, caught_on_pad=t < 2, predicted_miss=0.03) for t in range(10)]
        baseline = [record(approach="open_loop", sigma_p=0.0, trial=t, caught=t < 3, predicted_miss=0.01) for t in range(10)]
        return a + baseline

    def test_cells_count_their_outcomes(self):
        cells = {(c.approach, c.sigma_p): c for c in trials.summarize(self.results())}
        cell = cells[("predict_once", 0.005)]
        self.assertEqual((cell.n, cell.caught, cell.on_pad, cell.contact, cell.no_plan), (10, 6, 2, 10, 0))
        self.assertAlmostEqual(cell.predicted_median, 0.03)

    def test_a_cell_is_paired_with_the_baseline_at_its_toss_noise(self):
        cells = {(c.approach, c.sigma_p): c for c in trials.summarize(self.results())}
        cell = cells[("predict_once", 0.005)]
        self.assertEqual((cell.only_this, cell.only_baseline), (3, 0))
        self.assertAlmostEqual(cell.paired_p, 0.25)
        self.assertIsNone(cells[("open_loop", 0.0)].paired_p)

    def test_the_generated_part_replaces_only_what_is_below_the_marker(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "report.md"
            trials.write_report(self.results(), path)
            self.assertIn(trials.GENERATED_MARKER, path.read_text())
            path.write_text("interpretation\n\n" + path.read_text())
            trials.write_report(self.results()[:10], path)
            text = path.read_text()
            self.assertTrue(text.startswith("interpretation"))
            self.assertEqual(text.count(trials.GENERATED_MARKER), 1)
            self.assertNotIn("open_loop", text.split(trials.GENERATED_MARKER)[1].replace("baseline", ""))


class CatchHelpersTest(unittest.TestCase):
    def test_on_the_pad_means_within_the_square_both_ways(self):
        self.assertTrue(catch.ball_on_pad(np.array([0.03, 0.005, -0.005])))
        self.assertFalse(catch.ball_on_pad(np.array([0.0, 0.015, 0.0])))
        self.assertFalse(catch.ball_on_pad(np.array([0.0, 0.0, -0.015])))

    def test_the_baseline_does_not_depend_on_the_sensor(self):
        a = catch.simulate_catch(catch.OpenLoopPolicy, sigma_p=0.0, trial=3)
        b = catch.simulate_catch(catch.OpenLoopPolicy, sigma_p=0.02, trial=3)
        np.testing.assert_allclose(a.log.ball_p, b.log.ball_p, atol=1e-9)

    def test_latency_is_compensated_unless_the_trigger_assumes_another(self):
        nominal = catch.simulate_catch(catch.PredictOncePolicy, sigma_p=0.0, latency=0.03)
        known = catch.simulate_catch(catch.PredictOncePolicy, sigma_p=0.0, latency=0.05)
        unknown = catch.simulate_catch(catch.PredictOncePolicy, sigma_p=0.0, latency=0.05, assumed_latency=0.03)
        fingers = nominal.controller.t_close + 0.03
        self.assertAlmostEqual(known.controller.t_close + 0.05, fingers)  # still moves at the same time
        self.assertAlmostEqual(unknown.controller.t_close + 0.05, fingers + 0.02)  # 20 ms late


if __name__ == "__main__":
    unittest.main()
