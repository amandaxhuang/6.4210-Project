"""The friction sweep: the ball's coefficient, and the sweep's cells.  NEW FILE (friction sweep)."""

import unittest

import environment as env
import environment_friction as ef
import robustness_friction as rf
from robustness import A, B, BASELINE


class BallFrictionTest(unittest.TestCase):
    def tearDown(self):
        ef.reset()

    def test_every_material_combines_back_to_its_value(self):
        for name, target in ef.MATERIALS.items():
            self.assertAlmostEqual(ef.combined(ef.ball_mu(target)), target, msg=name)

    def test_rubber_is_the_existing_ball(self):
        self.assertEqual(ef.ball_mu(ef.MATERIALS["rubber"]), 1.0)

    def test_out_of_reach_friction_is_refused(self):
        for target in (0.0, 2.0, 3.0):
            with self.assertRaises(ValueError):
                ef.set_friction(target)
        self.assertIs(env.make_scenario, ef._original_make_scenario)

    def test_ball_sdf_carries_the_friction(self):
        ef.set_friction(0.25)
        ef.ball_directive()
        text = (env.ASSET_DIR / "ball.sdf").read_text()
        self.assertIn(f"<drake:mu_dynamic>{ef.ball_mu(0.25):.12g}</drake:mu_dynamic>", text)
        self.assertIn(f"<drake:mu_static>{ef.ball_mu(0.25):.12g}</drake:mu_static>", text)

    def test_reset_restores_environment(self):
        ef.set_material("sticky")
        self.assertIs(env.make_scenario, ef.make_scenario)
        ef.reset()
        self.assertIs(env.make_scenario, ef._original_make_scenario)
        self.assertEqual(ef.current(), ef.DEFAULT)


class GridTest(unittest.TestCase):
    def test_every_material_has_100_tosses_per_approach(self):
        configs = [c for c in rf.grid() if c.sigma_lat > 0]
        for mu in rf.FRICTIONS:
            for approach in (BASELINE, A, B):
                trials_ = [c.trial for c in configs if c.friction == mu and c.approach == approach]
                self.assertEqual(trials_, list(range(100)), msg=(mu, approach))

    def test_no_trial_runs_twice(self):
        configs = rf.grid()
        self.assertEqual(len(set(configs)), len(configs))

    def test_one_noise_free_toss_per_material_and_approach(self):
        self.assertEqual(len(rf.pilot_grid()), 3 * len(rf.FRICTIONS))
        self.assertTrue(all(c.sigma_p == c.sigma_lat == c.sigma_vz == 0 for c in rf.pilot_grid()))

    def test_baseline_ignores_the_sensor(self):
        self.assertTrue(all(c.sigma_p == 0 for c in rf.grid() if c.approach == BASELINE))


if __name__ == "__main__":
    unittest.main()
