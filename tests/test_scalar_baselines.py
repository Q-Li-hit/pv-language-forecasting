"""Scalar numerical baselines and local-only contextual model preparation."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pvforecast.data import ScalarData
from pvforecast.scalar_baselines import (
    context_indices, fit_scalar_tree, fit_tabpfn, load_tabpfn, predict_scalar_tree, predict_tabpfn,
)


class ScalarBaselineTests(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(14)
        n = 600
        state = rng.normal(size=(n, 12)).astype(np.float32)
        baseline = (3 + 0.5 * state[:, 1]).astype(np.float32)
        target = np.clip(baseline + np.sin(state[:, 0]) + 0.1 * state[:, 2], 0, 10)
        return ScalarData(
            rng.uniform(size=(n, 6)).astype(np.float32), state,
            np.zeros((n, 33), np.float32), target, baseline,
            np.array(["train"] * 400 + ["model_select"] * 100 + ["test"] * 100),
            np.repeat("2025-01-01", n), np.tile(["bon", "dra"], n // 2),
            {"kind": "weather", "preprocessing_id": "fixed-test"},
        )

    def test_context_selector_matches_balanced_original(self):
        import pandas as pd
        stations = np.repeat(["bon", "dra"], 30)
        horizons = np.tile(np.repeat([1, 2, 3], 10), 2)
        training = pd.DataFrame({"station_id": stations, "horizon": horizons})
        budget, seed = 23, 43
        rng = np.random.default_rng(seed)
        groups = [np.asarray(v, dtype=int) for v in training.groupby(
            ["station_id", "horizon"], observed=True).indices.values()]
        initial = np.concatenate([rng.choice(g, min(budget // len(groups), len(g)), replace=False)
                                  for g in groups])
        unused = np.setdiff1d(np.arange(len(training)), initial)
        expected = np.sort(np.r_[initial, rng.choice(unused, budget - len(initial), replace=False)])
        chosen = context_indices(stations, horizons, budget, seed)
        np.testing.assert_array_equal(chosen, expected)
        self.assertEqual(len(np.unique(chosen)), budget)
        self.assertEqual(len(set(zip(stations[chosen], horizons[chosen]))), 6)

    def test_context_selector_rejects_invalid_budgets(self):
        for budget in (0, 4, True):
            with self.assertRaises(ValueError):
                context_indices(["bon"] * 3, [1, 2, 3], budget, 42)

    @unittest.skipUnless(importlib.util.find_spec("lightgbm"), "LightGBM is optional")
    def test_scalar_tree_fit_predict_and_test_labels(self):
        data = self.data()
        for objective in ("direct", "residual"):
            with self.subTest(objective=objective), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp) / "fit"
                report = fit_scalar_tree(data, directory, objective=objective, threads=1,
                                         maximum_estimators=30, early_stopping_rounds=5)
                self.assertEqual(report["train_rows"], 400)
                self.assertEqual(report["selection_rows"], 100)
                prediction = predict_scalar_tree(data, directory, threads=1, batch_size=71)
                self.assertEqual(prediction.shape, (600,))
                self.assertTrue(np.isfinite(prediction).all())
                self.assertTrue(((prediction >= 0) & (prediction <= 10)).all())
                changed = self.data()
                changed.target[changed.roles == "test"] = 9.9
                repeated = predict_scalar_tree(changed, directory, threads=1)
                np.testing.assert_array_equal(prediction, repeated)
                mse = np.mean((prediction[-100:] - data.target[-100:]) ** 2)
                persistence_mse = np.mean((data.baseline[-100:] - data.target[-100:]) ** 2)
                self.assertLess(mse, persistence_mse)

    def test_tabpfn_requires_local_weights_before_import(self):
        data = self.data()
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(FileNotFoundError):
            fit_tabpfn(data, Path(tmp) / "absent.ckpt", horizons=np.ones(600), budget=16)

    def test_tabpfn_requires_explicit_rawscale_inputs(self):
        data = self.data()
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp) / "local.ckpt"
            weights.write_bytes(b"validation-only")
            with self.assertRaisesRegex(ValueError, "raw_state"):
                fit_tabpfn(data, weights, horizons=np.ones(600), budget=16)

    def test_tabpfn_context_is_training_only_and_can_be_restored(self):
        class LocalEstimator:
            def fit(self, x, y):
                self.x, self.y = x.copy(), y.copy()

            def predict(self, x, output_type="mean"):
                return x[:, 0] + 3

        data = self.data()
        data.raw_state = data.state.copy()
        horizons = np.tile([1, 2, 3], 200)
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp) / "weights.ckpt"
            weights.write_bytes(b"local-weights-identity")
            directory = Path(tmp) / "context"
            with patch("pvforecast.scalar_baselines._tabpfn_estimator", side_effect=lambda *a: LocalEstimator()), patch(
                    "pvforecast.scalar_baselines.importlib.metadata.version", return_value="test"):
                predictor, report = fit_tabpfn(data, weights, horizons=horizons, budget=29,
                                               seed=43, output=directory)
                indices = np.asarray(report["context_training_indices"])
                self.assertTrue((data.roles[indices] == "train").all())
                np.testing.assert_array_equal(predictor.estimator.x, data.raw_state[indices])
                np.testing.assert_array_equal(predictor.estimator.y, data.target[indices])
                first = predict_tabpfn(predictor, data, batch_size=37)
                restored = load_tabpfn(directory, weights)
                np.testing.assert_array_equal(predict_tabpfn(restored, data), first)
                self.assertEqual({path.name for path in directory.iterdir()}, {"fit.json", "context.npz"})
                weights.write_bytes(b"different-model")
                with self.assertRaisesRegex(ValueError, "weights"):
                    load_tabpfn(directory, weights)


if __name__ == "__main__":
    unittest.main()
