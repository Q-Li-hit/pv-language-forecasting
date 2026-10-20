"""Finite-sample CQR and chronological calibration availability tests."""

import unittest

import numpy as np

from pvforecast.calibration import (
    FixedCalibrator,
    arrived_label_mask,
    calibrated_interval,
    conformal_quantile,
    conformity_scores,
    fit_fixed_calibrator,
    rearrange_quantiles,
    recent_adjustments,
)


class ConformalQuantileTest(unittest.TestCase):
    def test_finite_order_is_not_interpolated(self):
        self.assertEqual(conformal_quantile(np.arange(10), alpha=.2), 8)

    def test_insufficient_sample_yields_infinite_expansion(self):
        self.assertEqual(conformal_quantile([1, 2, 3], alpha=.1), float("inf"))
        self.assertEqual(conformal_quantile([], alpha=.1), float("inf"))
        lower, upper = calibrated_interval([2], [4], float("inf"))
        np.testing.assert_array_equal(lower, [0])
        np.testing.assert_array_equal(upper, [10])

    def test_signed_scores_but_nonnegative_expansion(self):
        scores = conformity_scores([2, 3], [0, 1], [4, 5])
        np.testing.assert_array_equal(scores, [-2, -2])
        self.assertEqual(conformal_quantile(scores, alpha=.5), 0)
        self.assertEqual(conformal_quantile(scores, alpha=.5, nonnegative=False), -2)

    def test_invalid_quantile_inputs(self):
        for scores, alpha in (([1], 0), ([1], 1), ([np.nan], .1), ([[1]], .1)):
            with self.subTest(scores=scores, alpha=alpha), self.assertRaises(ValueError):
                conformal_quantile(scores, alpha=alpha)


class EndpointsTest(unittest.TestCase):
    def test_quantile_rearrangement_is_sorted_and_clipped(self):
        np.testing.assert_array_equal(rearrange_quantiles([[12, -1, 4], [3, 2, 1]]), [[0, 4, 10], [1, 2, 3]])

    def test_nested_interval_family_enlarges_outer_intervals(self):
        lower, upper = calibrated_interval([[3, 4, 2]], [[5, 4.5, 6]], [[0, 0, 1]], nested=True)
        np.testing.assert_array_equal(lower, [[3, 3, 1]])
        np.testing.assert_array_equal(upper, [[5, 5, 7]])

    def test_crossing_and_negative_adjustments_rejected(self):
        for lower, upper, adjustment in (([2], [1], 0), ([1], [2], -.1), ([1], [2], np.nan)):
            with self.subTest(adjustment=adjustment), self.assertRaises(ValueError):
                calibrated_interval(lower, upper, adjustment)


class FixedCalibrationTest(unittest.TestCase):
    def test_per_horizon_threshold_and_pooled_fallback(self):
        y = np.zeros(30)
        lower = np.r_[np.arange(20), np.arange(10) + 100]
        upper = lower + 2
        calibrator = fit_fixed_calibrator(y, lower, upper, [1] * 20 + [2] * 10)
        self.assertEqual(calibrator.horizon_thresholds[1], (18.,))
        self.assertEqual(calibrator.horizon_thresholds[2], calibrator.pooled_thresholds)
        self.assertEqual(calibrator.horizon_counts, {1: 20, 2: 10})
        self.assertEqual(calibrator.thresholds_for([3])[0, 0], calibrator.pooled_thresholds[0])

    def test_multilevel_scores_have_corresponding_levels(self):
        calibrator = fit_fixed_calibrator(np.zeros(30), np.column_stack((np.arange(30), np.arange(30) - 5)), np.column_stack((np.arange(30) + 2, np.arange(30) + 5)), np.ones(30), levels=(.5, .9))
        self.assertEqual(calibrator.thresholds_for([1]).shape, (1, 2))
        with self.assertRaises(ValueError):
            fit_fixed_calibrator([0], [1], [2], [1], levels=(.9, .5))


class ArrivedLabelTest(unittest.TestCase):
    def test_five_minute_delay_and_exact_window_boundaries(self):
        ends = ["2025-01-01T23:55Z", "2025-03-02T23:54Z", "2025-03-02T23:55Z", "2025-03-02T23:56Z"]
        selected = arrived_label_mask(ends, "2025-03-03T12:00Z")
        # First arrival is exactly cutoff - 60 days, so the window excludes it.
        np.testing.assert_array_equal(selected, [False, True, True, False])

    def test_utc_date_update_distinct_from_per_issuance(self):
        end = ["2025-01-02T00:01Z"]
        self.assertFalse(arrived_label_mask(end, "2025-01-02T00:06Z")[0])
        self.assertTrue(arrived_label_mask(end, "2025-01-02T00:06Z", update_at_day_start=False)[0])

    def test_measured_label_arrival_replaces_assumed_delay(self):
        selected = arrived_label_mask(["2025-01-01T23:00Z"], "2025-01-02T00:10Z", label_available_at=["2025-01-02T00:01Z"])
        self.assertFalse(selected[0])
        with self.assertRaises(ValueError):
            arrived_label_mask(["2025-01-02T00:00Z"], "2025-01-03", label_available_at=["2025-01-01T23:00Z"])


class RecentCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.fixed = FixedCalibrator((.5,), (4.,), {1: (5.,), 2: (6.,), 3: (7.,)}, {1: 30, 2: 30, 3: 30})

    def test_horizon_pooled_and_frozen_fallback(self):
        result = recent_adjustments([1, 2, 3, 10], [1, 1, 1, 2], ["2025-01-01"] * 4, "2025-01-02", [1, 2], self.fixed, minimum_per_horizon=3, minimum_pooled=4)
        self.assertEqual(result.sources, ("recent_horizon", "recent_pooled"))
        np.testing.assert_array_equal(result.thresholds, [[2], [3]])
        result = recent_adjustments([1], [1], ["2025-01-01"], "2025-01-02", [2], self.fixed)
        self.assertEqual(result.sources, ("fixed",))
        np.testing.assert_array_equal(result.thresholds, [[6]])

    def test_future_label_perturbations_cannot_change_threshold(self):
        common = dict(horizons=[1, 1, 1], target_end=["2025-01-01", "2025-01-02", "2025-01-04"], issuance="2025-01-03", requested_horizons=[1], fixed=self.fixed, minimum_per_horizon=2)
        before = recent_adjustments([1, 2, 3], **common)
        after = recent_adjustments([1, 2, 1000000], **common)
        np.testing.assert_array_equal(before.thresholds, after.thresholds)
        self.assertEqual(before.used_counts, (2,))


if __name__ == "__main__":
    unittest.main()
