"""Weather representations, training-only transforms and model interfaces."""

import unittest
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch

from pvforecast.weather import (
    CATEGORY_SIZES, STATE_FIELDS, CapacityMatchedControl, ResidualTransformer,
    TransformerConfig, WeatherPreprocessor, WeightedPCA, category_context,
    context_codes_key, context_fields, numeric_projection, text_from_codes,
)
from pvforecast.data import ScalarData
from pvforecast.preparation import prepare_weather


class WeatherTests(unittest.TestCase):
    def frame(self):
        frame = pd.DataFrame({name: [0.0, 50.0, np.nan] for name in STATE_FIELDS})
        frame["hist_0"] = [1.0, 2.0, np.nan]
        frame["hist_1"] = [0.5, 1.5, np.nan]
        return frame

    def test_state_thresholds_and_template(self):
        frame = pd.DataFrame({
            "cloud_low": [19.0, 20.0, np.nan], "cloud_mid": [59.0, 60.0, np.nan],
            "cloud_high": [89.0, 90.0, np.nan], "cloud_total": [0.0, 99.0, np.nan],
            "nwp_csi": [0.34, 0.7, np.nan], "recent_ramp": [-0.6, 0.5, np.nan],
            "observed_residual": [-0.5, 0.49, np.nan],
        })
        codes, names = context_fields(frame)
        np.testing.assert_array_equal(codes[0], [0, 1, 2, 0, 0, 0, 1])
        np.testing.assert_array_equal(codes[1], [1, 2, 3, 3, 2, 2, 1])
        np.testing.assert_array_equal(codes[2], np.array(CATEGORY_SIZES) - 1)
        self.assertEqual(names, list(STATE_FIELDS))
        self.assertIn("low cloud clear", text_from_codes(codes)[0])
        onehot = category_context(codes)
        self.assertEqual(onehot.shape, (3, 33))
        np.testing.assert_array_equal(onehot.sum(axis=1), [7, 7, 7])

    def test_training_only_preprocessing(self):
        frame = self.frame()
        names = ["hist_0", "hist_1", *STATE_FIELDS]
        preprocessor = WeatherPreprocessor.fit(frame, names)
        before = preprocessor.scaler.mean_.copy()
        unseen = frame.copy()
        unseen.loc[0, "hist_0"] = 1e6
        history, state = preprocessor.transform(unseen)
        np.testing.assert_array_equal(before, preprocessor.scaler.mean_)
        self.assertTrue(np.isfinite(state).all())
        self.assertEqual(history[2, 0], 0)
        context = preprocessor.numeric_context(state)
        self.assertEqual(context.shape, (3, 33))
        self.assertLessEqual(np.linalg.matrix_rank(numeric_projection()), 7)

    def test_weighted_pca_uses_training_frequencies(self):
        codes = np.array([[0] * 7, [1] * 7, [2] * 7], dtype=np.int8)
        keys = context_codes_key(codes)
        bank = np.array([[0.0, 0.0], [2.0, 0.0], [2000.0, 3000.0]])
        training = codes[[0, 0, 1]]
        projection = WeightedPCA.fit(bank, keys, training, n_components=2)
        np.testing.assert_allclose(projection.mean, [2 / 3, 0])
        self.assertEqual(projection.transform(bank, add_constant=True).shape, (3, 3))
        changed = bank.copy()
        changed[2] *= 10
        alternate = WeightedPCA.fit(changed, keys, training, n_components=2)
        np.testing.assert_allclose(projection.mean, alternate.mean)
        np.testing.assert_allclose(projection.components, alternate.components)

    def test_capacity_matched_shapes_and_masked_context(self):
        config = TransformerConfig(d_model=8, heads=2, layers=1, dropout=0)
        torch.manual_seed(42)
        model = CapacityMatchedControl(2, 9, config).eval()
        self.assertEqual(model.adapter.weight.numel(), 1056)
        history, state = torch.rand(3, 2), torch.rand(3, 9)
        with torch.inference_mode():
            self.assertEqual(model(history, state, torch.rand(3, 33)).shape, (3,))
            numeric = ResidualTransformer(2, 9, "numeric", config).eval()
            first = numeric(history, state, torch.zeros(3, 32))
            second = numeric(history, state, torch.full((3, 32), 100.0))
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_invalid_contexts_and_config(self):
        with self.assertRaises(ValueError):
            category_context(np.full((1, 7), -1))
        with self.assertRaises(ValueError):
            TransformerConfig(d_model=7, heads=2)

    def aligned_frame(self):
        n = 6
        frame = pd.DataFrame({name: np.linspace(0.05, 0.95, n) for name in STATE_FIELDS})
        for name in STATE_FIELDS[:4]:
            frame[name] = np.linspace(10, 98, n)
        frame["hist_0"] = np.linspace(0.1, 5, n)
        frame["hist_1"] = np.linspace(0.1, 4, n)
        for name in ("forecast_age_h", "forecast_lead_h", "forecast_elapsed_h", "horizon",
                     "station_code", "sin_hour", "cos_hour", "sin_doy", "cos_doy",
                     "csky_power", "persistence_mw"):
            frame[name] = np.linspace(0.1, 1, n)
        frame["target_mw"] = np.linspace(0.3, 5.3, n)
        frame["base_mw"] = np.linspace(0.1, 5, n)
        frame["role"] = ["train", "train", "train", "model_select", "model_select", "test"]
        frame["issuance_utc"] = pd.date_range("2024-01-01", periods=n, tz="UTC").astype(str)
        frame["station_id"] = "bon"
        return frame

    def test_prepare_weather_roundtrip_and_inference_only_rows(self):
        frame = self.aligned_frame()
        for representation in ("numeric", "category"):
            with self.subTest(representation=representation), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "all.npz"
                original = prepare_weather(frame, path, representation=representation)
                loaded = ScalarData.load(path)
                for field in ("history", "state", "context", "target", "baseline"):
                    np.testing.assert_array_equal(getattr(original, field), getattr(loaded, field))
                self.assertEqual(loaded.context.shape, (6, 33))
                saved = path.with_suffix(".preprocessing.npz")
                repeated = prepare_weather(frame, Path(tmp) / "repeat.npz",
                                           representation=representation, preprocessor=saved)
                np.testing.assert_array_equal(original.state, repeated.state)
                np.testing.assert_array_equal(original.context, repeated.context)
                inference = frame.loc[frame.role.eq("test")].copy()
                transferred = prepare_weather(inference, Path(tmp) / "test.npz",
                                              representation=representation, preprocessor=saved)
                np.testing.assert_array_equal(original.state[-1:], transferred.state)
                np.testing.assert_array_equal(original.context[-1:], transferred.context)

    def test_selection_values_do_not_fit_weather_preprocessing(self):
        frame = self.aligned_frame()
        changed = frame.copy()
        changed.loc[~changed.role.eq("train"), "hist_0"] = 9000.0
        changed.loc[~changed.role.eq("train"), "cloud_low"] = 99.0
        changed.loc[~changed.role.eq("train"), "target_mw"] = 999.0
        with tempfile.TemporaryDirectory() as tmp:
            first_path, second_path = Path(tmp) / "first.npz", Path(tmp) / "second.npz"
            first = prepare_weather(frame, first_path, representation="numeric")
            second = prepare_weather(changed, second_path, representation="numeric")
            for field in ("state", "context", "history"):
                np.testing.assert_array_equal(getattr(first, field)[:3], getattr(second, field)[:3])
            with np.load(first_path.with_suffix(".preprocessing.npz")) as a, np.load(
                    second_path.with_suffix(".preprocessing.npz")) as b:
                for name in ("fill", "mean", "scale", "var", "numeric_projection"):
                    np.testing.assert_array_equal(a[name], b[name])

    def embedding_bank(self, frame, path, representation="text", revision="test-revision"):
        keys = np.unique(context_codes_key(context_fields(frame)[0]))
        bank = np.random.default_rng(9).normal(size=(len(keys), 32)).astype(np.float32)
        metadata = {"representation": representation, "model_id": "test-encoder",
                    "revision": revision, "random_seed": None}
        np.savez_compressed(path, keys=keys, embedding=bank, metadata=np.asarray(json.dumps(metadata)))

    def test_text_preprocessing_roundtrip_and_encoder_identity(self):
        frame = self.aligned_frame()
        with tempfile.TemporaryDirectory() as tmp:
            bank = Path(tmp) / "bank.npz"
            self.embedding_bank(frame, bank)
            output = Path(tmp) / "text.npz"
            first = prepare_weather(frame, output, representation="text", embedding_bank=bank)
            self.assertEqual(first.context.shape, (6, 33))
            np.testing.assert_array_equal(first.context[:, -1], np.ones(6))
            repeated = prepare_weather(frame, Path(tmp) / "again.npz", representation="text",
                                       preprocessor=output.with_suffix(".preprocessing.npz"),
                                       embedding_bank=bank)
            np.testing.assert_allclose(first.context, repeated.context, rtol=1e-6, atol=1e-6)
            changed = Path(tmp) / "changed.npz"
            self.embedding_bank(frame, changed, revision="different-revision")
            with self.assertRaises(ValueError):
                prepare_weather(frame, Path(tmp) / "wrong.npz", representation="text",
                                preprocessor=output.with_suffix(".preprocessing.npz"),
                                embedding_bank=changed)
            with self.assertRaises(ValueError):
                prepare_weather(frame, Path(tmp) / "wrong-representation.npz",
                                representation="random", embedding_bank=bank)

    def test_preparation_requires_common_state_variables(self):
        frame = self.aligned_frame().drop(columns=[STATE_FIELDS[-1]])
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            prepare_weather(frame, Path(tmp) / "missing.npz", representation="numeric")


if __name__ == "__main__":
    unittest.main()
