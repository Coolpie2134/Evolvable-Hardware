import math
import random
import unittest

import nv_evo.pulse as pulse
from nv_async_ui import NervousPlayer, pulses_from_trial, toggle_pulse
from nv_evo.oracle import echo_oracle, pair_oracle, sr_latch_oracle
from nv_evo.targets import TEMPORAL_TARGETS, Trial, oscillator, spike_target
from nv_evo.temporal import (
    TemporalTraces,
    _best_event_shift,
    _event_counts,
    cadence_score,
    event_score,
    run_nervous_events,
    sampled_events,
    temporal_report,
    trial_input_summary,
    expected_window_summary,
)
from lut_evo.ga import lut_report
from snn_evo.targets import TARGETS as LOGIC_TARGETS
from target_ui import target_category


class RawEventSimulationTests(unittest.TestCase):
    def setUp(self):
        self.physics = pulse.DELAY, pulse.WIDTH, pulse.COINC

    def tearDown(self):
        pulse.DELAY, pulse.WIDTH, pulse.COINC = self.physics

    @staticmethod
    def line_net():
        grid = {(0, 0): 1, (1, 0): 2}
        routing = {
            (0, 0): (None, None, None, 'and'),
            (1, 0): ('R', 'R', None, 'and'),
        }
        return grid, routing

    def test_subtick_edge_survives_when_midtick_sample_misses_it(self):
        pulse.DELAY = 0.25
        pulse.WIDTH = 0.10
        grid, routing = self.line_net()
        states, traces, events, overflow = run_nervous_events(
            grid, routing, [(0, 0)], {'Q': (1, 0)}, [(1,)], 1,
            prune=False, max_events=32)
        self.assertFalse(overflow)
        self.assertEqual(traces['Q'], [0])
        self.assertEqual(events[(1, 0)], [0.25])

    def test_runaway_loop_hits_deterministic_event_cap(self):
        grid = {(0, 0): 2, (1, 0): 2}
        routing = {
            (0, 0): ('R', 'R', None, 'and'),
            (1, 0): ('R', 'R', None, 'and'),
        }
        _, _, _, overflow = run_nervous_events(
            grid, routing, [(0, 0)], {}, [(1,)] + [(0,)] * 19, 20,
            prune=False, max_events=3)
        self.assertTrue(overflow)

    def test_unsampled_fast_path_preserves_sampled_run_edges(self):
        grid, routing = self.line_net()
        streams = [(1,), (1,), (0,), (1,), (0,)]
        sampled = run_nervous_events(
            grid, routing, [(0, 0)], {'Q': (1, 0)}, streams, 5,
            prune=False, max_events=32, sample=True)
        sparse = run_nervous_events(
            grid, routing, [(0, 0)], {'Q': (1, 0)}, streams, 5,
            prune=False, max_events=32, sample=False)
        self.assertEqual(sampled[2], sparse[2])
        self.assertEqual(sampled[3], sparse[3])

    def test_unsampled_path_matches_random_streams_and_physics(self):
        grid, routing = self.line_net()
        rng = random.Random(20260710)
        for delay, width in ((1.0, 1.0), (0.25, 0.1), (0.75, 2.5)):
            pulse.DELAY, pulse.WIDTH = delay, width
            for _ in range(30):
                streams = [(rng.randrange(2),) for _ in range(12)]
                sampled = run_nervous_events(
                    grid, routing, [(0, 0)], {}, streams, 12,
                    prune=False, max_events=128, sample=True)
                sparse = run_nervous_events(
                    grid, routing, [(0, 0)], {}, streams, 12,
                    prune=False, max_events=128, sample=False)
                self.assertEqual(sampled[2], sparse[2])
                self.assertEqual(sampled[3], sparse[3])


class SparseEventScoringTests(unittest.TestCase):
    def make_target(self):
        target = spike_target(
            'events',
            [({0: [1]}, [3, 7]), ({0: [2]}, [4, 9])],
            T=12, n_inputs=1, latency=1)
        target.event_tolerance = 0.05
        return target

    def test_one_continuous_latency_is_shared_across_trials(self):
        target = self.make_target()
        traces = TemporalTraces(
            {'Q': [[0] * 24, [0] * 24]},
            events={'Q': [[4.25, 8.25], [5.25, 10.25]]})
        shift, score = _best_event_shift(traces, target)
        self.assertAlmostEqual(shift, 1.20)
        self.assertEqual(score, 1.0)

        inconsistent = TemporalTraces(
            {'Q': [[0] * 24, [0] * 24]},
            events={'Q': [[4.25, 8.25], [6.0, 11.0]]})
        self.assertLess(event_score(inconsistent, target), 1.0)

    def test_extra_event_lowers_precision(self):
        target = self.make_target()
        clean = TemporalTraces(
            {'Q': [[0] * 24, [0] * 24]},
            events={'Q': [[4.0, 8.0], [5.0, 10.0]]})
        noisy = TemporalTraces(
            {'Q': [[0] * 24, [0] * 24]},
            events={'Q': [[4.0, 6.0, 8.0], [5.0, 10.0]]})
        self.assertEqual(event_score(clean, target), 1.0)
        self.assertLess(event_score(noisy, target), 1.0)

    def test_clocked_adapter_collapses_held_high_to_one_event(self):
        self.assertEqual(sampled_events([0, 1, 1, 0, 1]), [1.0, 4.0])

    def test_optimized_matcher_equals_reference_scan(self):
        def reference(actual, expected, exp, shift, tolerance):
            mapped = sorted(
                event - shift for event in actual
                if (0 <= math.floor(event - shift + 1e-9) < len(exp)
                    and exp[math.floor(event - shift + 1e-9)] is not None))
            expected = sorted(float(event) for event in expected)
            i = j = matches = 0
            while i < len(mapped) and j < len(expected):
                delta = mapped[i] - expected[j]
                if abs(delta) <= tolerance + 1e-9:
                    matches += 1; i += 1; j += 1
                elif delta < -tolerance:
                    i += 1
                else:
                    j += 1
            return matches, len(expected), matches, len(mapped)

        rng = random.Random(20260710)
        for _ in range(1000):
            actual = sorted(round(rng.random() * 30, 3)
                            for _ in range(rng.randrange(15)))
            expected = sorted(round(rng.random() * 15, 3)
                              for _ in range(rng.randrange(10)))
            exp = [None if rng.random() < 0.2 else rng.randrange(2)
                   for _ in range(rng.randrange(1, 20))]
            shift = round(rng.uniform(-5, 5), 3)
            tolerance = rng.choice((0.0, 0.1, 0.5, 1.0))
            self.assertEqual(
                _event_counts(actual, expected, exp, shift, tolerance),
                reference(actual, expected, exp, shift, tolerance))


class SemanticTargetTests(unittest.TestCase):
    def test_target_categories_follow_behavior_not_one_flat_registry(self):
        self.assertEqual(target_category('AND', LOGIC_TARGETS['AND']), 'Logic gates')
        self.assertEqual(target_category('Half adder', LOGIC_TARGETS['Half adder']),
                         'Arithmetic')
        self.assertEqual(target_category('2:1 MUX', LOGIC_TARGETS['2:1 MUX']),
                         'Routing & decisions')
        expected = {
            'events': 'Timed events',
            'trace': 'Memory & state',
            'cadence': 'Cadence & patterns',
            'period_stepper': 'Cadence & patterns',
        }
        for name, target in TEMPORAL_TARGETS.items():
            with self.subTest(target=name):
                self.assertEqual(target_category(name, target), expected[target.score_mode])

    def test_all_target_descriptions_use_the_same_three_sections(self):
        for name, target in TEMPORAL_TARGETS.items():
            with self.subTest(target=name):
                self.assertIn('Goal:', target.description)
                self.assertIn('Scoring:', target.description)
                self.assertIn('Tests:', target.description)
                self.assertNotIn(' ms', target.description)

    def test_reports_use_test_language_and_shared_descriptions(self):
        for name, target in TEMPORAL_TARGETS.items():
            with self.subTest(target=name):
                for report in (temporal_report(target), lut_report(target)):
                    self.assertIn('Goal:', report)
                    self.assertIn('Scoring:', report)
                    self.assertIn('Tests:', report)
                    self.assertNotIn('Trial ', report)

    def test_report_helpers_collapse_held_inputs_and_dense_windows(self):
        trial = Trial([(1, 0), (1, 1), (0, 1), (1, 0)], {'Q': []})
        self.assertEqual(trial_input_summary(trial, 2), 'A@[0, 3]  B@[1]')
        self.assertEqual(
            expected_window_summary([None, None, 0, 0, 1, 1, 0]),
            'ignore[0:2), quiet[2:4), active[4:6), quiet[6:7)')

    def test_cadence_requires_sustained_coverage(self):
        target = oscillator()
        perfect, burst = [], []
        for trial in target.trials:
            kick = next(t for t, row in enumerate(trial.streams) if row[0])
            start = int(kick + 2 + target.cadence_settle)
            perfect.append([float(t) for t in range(start, target.T + 2, 2)])
            burst.append([float(t) for t in range(start, start + 8, 2)])
        dense = {'Q': [[0] * (2 * target.T) for _ in target.trials]}
        good = TemporalTraces(dense, events={'Q': perfect})
        short = TemporalTraces(dense, events={'Q': burst})
        self.assertEqual(cadence_score(good, target)[0], 1.0)
        self.assertLess(cadence_score(short, target)[0], 1.0)

    def test_event_oracles_and_latch_persistence_guards(self):
        self.assertEqual(echo_oracle().score_mode, 'events')
        self.assertEqual(pair_oracle().score_mode, 'events')
        latch = sr_latch_oracle()
        self.assertEqual(len(latch.trials), 14)
        self.assertTrue(all(bit == (0, 0) for bit in latch.trials[-2].streams))
        self.assertIn((1, 0), latch.trials[-1].streams)
        self.assertIn((0, 1), latch.trials[-1].streams)


class AsyncPlaybackTests(unittest.TestCase):
    def setUp(self):
        self.physics = pulse.DELAY, pulse.WIDTH, pulse.COINC

    def tearDown(self):
        pulse.DELAY, pulse.WIDTH, pulse.COINC = self.physics

    def test_player_accepts_and_reports_subtick_edges(self):
        pulse.DELAY = 0.25
        pulse.WIDTH = 0.10
        grid, routing = RawEventSimulationTests.line_net()
        player = NervousPlayer(grid, routing, horizon=2.0, dt=0.25,
                               pulse_width=0.1)
        player.set_schedule({(0, 0): [0.1]})
        player.step()
        player.step()
        self.assertEqual(player.events_upto((1, 0)), [0.35])
        player.reset()
        self.assertEqual(player.cursor, 0.0)
        self.assertEqual(player.events_upto((1, 0)), [])

    def test_timeline_helpers_share_edge_semantics(self):
        target = oscillator()
        pulses = pulses_from_trial(target, 1)
        self.assertEqual(pulses[0], [0.0])
        row = [1.0]
        self.assertEqual(toggle_pulse(row, 1.1, 0.5), [])
        self.assertEqual(toggle_pulse(row, 2.0, 0.5), [2.0])


if __name__ == '__main__':
    unittest.main()
