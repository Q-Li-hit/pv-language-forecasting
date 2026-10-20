"""Checks of the evaluation identities and paired calendar-date sampling."""

import unittest

import numpy as np

from pvforecast.evaluation import (
    fit_simplex_weights,
    interval_metrics,
    metrics,
    mse_average_decomposition,
    paired_date_block_bootstrap,
)


class PointMetricsTest(unittest.TestCase):
    def test_known_errors_and_capacity_scaling(self):
        result = metrics([1, 2, 4], [2, 0, 4], capacity_mw=10, reference=[3, 4, 6])
        self.assertAlmostEqual(result["MSE_MW2"], 5 / 3)
        self.assertAlmostEqual(result["MAE_MW"], 1)
        self.assertAlmostEqual(result["MBE_MW"], -1 / 3)
        self.assertAlmostEqual(result["nRMSE_percent"], 10 * np.sqrt(5 / 3))
        self.assertAlmostEqual(result["Skill_pooled"], 1 - np.sqrt(5 / 3) / 2)

    def test_constant_target_and_perfect_reference(self):
        result = metrics([1, 1], [1, 1], reference=[1, 1])
        self.assertTrue(np.isnan(result["R2"]))
        self.assertTrue(np.isnan(result["Skill_pooled"]))

    def test_rejects_implicit_mask_and_broadcasting(self):
        for y, prediction in (([], []), ([1, 2], [1]), ([1, np.nan], [1, 2]), ([[1]], [[1]])):
            with self.subTest(y=y), self.assertRaises(ValueError):
                metrics(y, prediction)


class PairedBootstrapTest(unittest.TestCase):
    def test_reproduces_direct_calendar_block_calculation(self):
        dates = np.array(["2025-01-01", "2025-01-01", "2025-01-03", "2025-01-08"], dtype="datetime64[D]")
        y = np.array([0., 1., 2., 3.])
        a = np.array([1., 3., 2.5, 2.])
        b = np.array([0.5, 2., 4., 6.])
        repetitions, block_days, seed = 75, 3, 71
        result = paired_date_block_bootstrap(y, a, b, dates, repetitions=repetitions, block_days=block_days, seed=seed)
        rng = np.random.default_rng(seed)
        starts = rng.integers(0, 8, size=(repetitions, 3))
        indices = ((starts[..., None] + np.arange(block_days)) % 8).reshape(repetitions, -1)[:, :8]
        days = (dates - dates.min()).astype(int)
        rmse, mae = [], []
        for sample in indices:
            selected = np.concatenate([np.flatnonzero(days == day) for day in sample])
            if selected.size:
                error_a, error_b = a[selected] - y[selected], b[selected] - y[selected]
                rmse.append(np.sqrt(np.mean(error_a**2)) - np.sqrt(np.mean(error_b**2)))
                mae.append(np.mean(np.abs(error_a)) - np.mean(np.abs(error_b)))
        self.assertEqual(result["calendar_days"], 8)
        self.assertEqual(result["valid_repetitions"], len(rmse))
        self.assertAlmostEqual(result["RMSE_CI_low_MW"], np.quantile(rmse, .025))
        self.assertAlmostEqual(result["RMSE_CI_high_MW"], np.quantile(rmse, .975))
        self.assertAlmostEqual(result["MAE_CI_low_MW"], np.quantile(mae, .025))
        self.assertAlmostEqual(result["MAE_CI_high_MW"], np.quantile(mae, .975))

    def test_identical_methods_have_zero_paired_difference(self):
        result = paired_date_block_bootstrap([1, 2], [3, 5], [3, 5], ["2025-01-01", "2025-01-02"], repetitions=30)
        for key in ("delta_RMSE_MW", "RMSE_CI_low_MW", "RMSE_CI_high_MW", "delta_MAE_MW", "MAE_CI_low_MW", "MAE_CI_high_MW"):
            self.assertEqual(result[key], 0)
        self.assertFalse(result["seed_uncertainty_included"])

    def test_timezone_aware_timestamps_group_by_utc_issue_date(self):
        result = paired_date_block_bootstrap([0, 0, 0], [1, 2, 3], [2, 3, 4], ["2025-01-02T00:30:00+02:00", "2025-01-01T23:00:00Z", "2025-01-02T23:00:00Z"], repetitions=10)
        self.assertEqual(result["issue_dates"], 2)
        self.assertEqual(result["calendar_days"], 2)

    def test_invalid_parameters_and_missing_dates(self):
        inputs = ([0, 0], [1, 2], [2, 3], ["2025-01-01", "2025-01-02"])
        for options in ({"confidence": 1}, {"confidence": 0}, {"repetitions": 0}, {"block_days": 1.5}, {"block_days": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                paired_date_block_bootstrap(*inputs, **options)
        with self.assertRaises(ValueError):
            paired_date_block_bootstrap(*inputs[:3], ["2025-01-01", "2025-01-01"])
        with self.assertRaises(ValueError):
            paired_date_block_bootstrap(*inputs[:3], np.array(["2025-01-01", "NaT"], dtype="datetime64[D]"))


class AverageDecompositionTest(unittest.TestCase):
    def test_opposite_errors_cancel_exactly(self):
        y = np.array([1., 2., 3.])
        result = mse_average_decomposition(np.vstack((y - 2, y, y + 2)), y)
        self.assertAlmostEqual(result["mean_single_MSE_MW2"], 8 / 3)
        self.assertEqual(result["mean_prediction_MSE_MW2"], 0)
        self.assertAlmostEqual(result["averaging_MSE_reduction_MW2"], 8 / 3)
        self.assertAlmostEqual(result["identity_residual_MW2"], 0)

    def test_identical_forecasts_have_no_averaging_gain(self):
        result = mse_average_decomposition([[2, 3], [2, 3]], [1, 2])
        self.assertEqual(result["averaging_MSE_reduction_MW2"], 0)
        self.assertEqual(result["mean_prediction_MSE_MW2"], 1)

    def test_wrong_orientation_rejected(self):
        with self.assertRaises(ValueError):
            mse_average_decomposition(np.zeros((3, 2)), np.zeros(3))


class SimplexWeightsTest(unittest.TestCase):
    def test_known_interior_optimum(self):
        predictions = np.array([[0., 2.], [2., 0.], [1., 3.]])
        target = predictions @ np.array([.75, .25])
        weights = fit_simplex_weights(predictions, target)
        np.testing.assert_allclose(weights, [.75, .25], atol=1e-7)
        self.assertAlmostEqual(weights.sum(), 1)
        self.assertTrue(np.all(weights >= 0))

    def test_optimum_can_exclude_a_member(self):
        weights = fit_simplex_weights([[0., 1.], [1., 2.], [2., 3.]], [0., 1., 2.])
        np.testing.assert_allclose(weights, [1, 0], atol=1e-7)

    def test_regularization_matches_analytic_solution(self):
        # One target: (2w_2 - .5)^2 + 2*((w_1-.5)^2+(w_2-.5)^2).
        weights = fit_simplex_weights([[0., 2.]], [.5], penalty=2)
        np.testing.assert_allclose(weights, [.625, .375], atol=1e-7)

    def test_bad_prior_or_penalty_rejected(self):
        for options in ({"prior": [.1, .2]}, {"prior": [1.1, -.1]}, {"penalty": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                fit_simplex_weights([[0., 1.]], [0.], **options)


class IntervalMetricsTest(unittest.TestCase):
    def test_interval_score_and_three_coverage_objects(self):
        # All four targets have width 2; two fall one MW outside the interval.
        result = interval_metrics([0, 2, 4, 6], [-1, 0, 1, 3], [1, 2, 3, 5], alpha=.1, capacity_mw=10, trajectory_ids=[0, 0, 1, 1])
        self.assertEqual(result["coverage"], .5)
        self.assertEqual(result["mean_width_MW"], 2)
        self.assertEqual(result["interval_score_MW"], 12)
        self.assertEqual(result["normalized_mean_width"], .2)
        self.assertEqual(result["trajectory_coverage"], .5)
        self.assertEqual(result["trajectories"], 2)
        conditional = interval_metrics([4, 6], [1, 3], [3, 5])
        self.assertEqual(conditional["coverage"], 0)

    def test_invalid_bounds_and_missing_trajectory_rejected(self):
        with self.assertRaises(ValueError):
            interval_metrics([0], [1], [0])
        with self.assertRaises(ValueError):
            interval_metrics([0], [-1], [1], trajectory_ids=[None])
        with self.assertRaises(ValueError):
            interval_metrics([0], [-1], [1], alpha=0)


if __name__ == "__main__":
    unittest.main()
