"""Scientific display contracts for censored and early-solved runs."""
import unittest
from tools.multi_target_campaign import trajectory, plot_horizon, BACKENDS


class CampaignTrajectories(unittest.TestCase):
    def test_crop_requires_all_thirty_population_solves(self):
        runs = [dict(cell={'backend': b}, index=i, population_solved=True,
                     history=[dict(absolute_generation=199, population_mean_raw=.9),
                              dict(absolute_generation=200, population_mean_raw=1.0),
                              dict(absolute_generation=201, population_mean_raw=1.0)])
                for b in BACKENDS for i in range(10)]
        self.assertEqual(plot_horizon(runs), 200)
        self.assertEqual(plot_horizon(runs[:-1]), 600)
        runs[-1]['population_solved'] = False
        self.assertEqual(plot_horizon(runs), 600)
        runs[-1]['population_solved'] = True
        runs[-1]['history'][-1]['population_mean_raw'] = .9999999
        self.assertEqual(plot_horizon(runs), 600)

    def test_only_population_perfection_can_extend_a_run(self):
        run = {'population_solved': True, 'history': [
            {'absolute_generation': 0, 'population_mean': .1},
            {'absolute_generation': 1, 'population_mean': 1.0}]}
        values = trajectory(run)
        self.assertEqual(len(values), 601)
        self.assertEqual(values[:2], [.1, 1.0])
        self.assertTrue(all(x == 1.0 for x in values[2:]))

    def test_champion_solve_and_timeout_do_not_fill_population_tail(self):
        run = {'trained': True, 'timed_out': True, 'history': [
            {'absolute_generation': 0, 'population_mean': .2},
            {'absolute_generation': 1, 'population_mean': .6}]}
        self.assertEqual(trajectory(run), [.2, .6])

    def test_missing_observations_are_not_interpolated(self):
        self.assertEqual(trajectory({'history': []}), [])
        self.assertEqual(trajectory({'history': [
            {'absolute_generation': 0, 'population_mean': .2},
            {'absolute_generation': 2, 'population_mean': .6}]}), [.2])


if __name__ == '__main__':
    unittest.main()
