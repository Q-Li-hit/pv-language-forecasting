"""Evaluation masks and fitted preprocessing must survive later preparation."""
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from pvforecast.cli import _save_predictions, evaluate
from pvforecast.data import ScalarData
from pvforecast.preparation import prepare_weather
from pvforecast.utils import model_digest
from pvforecast.weather import STATE_FIELDS, context_codes_key, context_fields


class IntegrityTests(unittest.TestCase):
    def frame(self):
        n = 8
        frame = pd.DataFrame({name: np.linspace(.1, .9, n) for name in STATE_FIELDS})
        for name in STATE_FIELDS[:4]:
            frame[name] = np.linspace(10, 98, n)
        frame["hist_0"] = np.linspace(.2, 5., n)
        frame["hist_1"] = np.linspace(.1, 4., n)
        for name in (
            "forecast_age_h", "forecast_lead_h", "forecast_elapsed_h", "station_code",
            "sin_hour", "cos_hour", "sin_doy", "cos_doy", "csky_power", "persistence_mw",
        ):
            frame[name] = np.linspace(.1, 1., n)
        frame["horizon"] = 1
        frame["target_mw"] = np.linspace(.3, 5.3, n)
        frame["base_mw"] = np.linspace(.1, 5., n)
        frame["role"] = ["train"] * 3 + ["model_select"] * 2 + ["test"] * 3
        frame["issuance_utc"] = pd.date_range("2024-01-01", periods=n, tz="UTC").astype(str)
        frame["station_id"] = "bon"
        frame["daylight"] = [False, True, False, True, False, True, False, True]
        return frame

    def bank(self, frame, destination):
        keys = np.unique(context_codes_key(context_fields(frame)[0]))
        values = np.random.default_rng(45).normal(size=(len(keys), 32)).astype(np.float32)
        np.savez_compressed(
            destination, keys=keys, embedding=values,
            metadata=np.asarray(json.dumps({
                "representation": "text", "declared_model_id": "local-test-module",
                "declared_revision": "fixture", "snapshot_sha256": "fixture",
                "random_seed": None, "pooling": "last-valid-token-l2", "max_length": 128,
            })),
        )

    def archive_arrays(self, path):
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}

    def test_daylight_only_changes_evaluation_and_survives_archive(self):
        frame = self.frame()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weather.npz"
            data = prepare_weather(frame, path, representation="category")
            restored = ScalarData.load(path)
            np.testing.assert_array_equal(restored.evaluation_mask, [False] * 5 + [True, False, True])
            np.testing.assert_array_equal(restored.roles, frame.role)
            np.testing.assert_array_equal(data.sample_keys, restored.sample_keys)
            self.assertEqual(len(np.unique(restored.sample_keys)), len(frame))
            predictions = restored.target + 1.
            predictions[6] += 1000.
            saved = Path(directory) / "predictions.npz"
            _save_predictions(saved, restored, predictions)
            report = Path(directory) / "metrics.json"
            with redirect_stdout(StringIO()):
                evaluate(Namespace(
                    predictions=saved, reference=None, output=report,
                    capacity=10., repetitions=20, confidence=.95,
                ))
            result = json.loads(report.read_text(encoding="utf-8"))
            self.assertAlmostEqual(result["RMSE_MW"], 1.)

    def test_daylight_values_are_validated(self):
        frame = self.frame()
        frame["daylight"] = "day"
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            prepare_weather(frame, Path(directory) / "invalid.npz", representation="category")

    def test_changed_preprocessing_arrays_are_rejected(self):
        frame = self.frame()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prepared.npz"
            prepare_weather(frame, output, representation="numeric")
            preprocessor = output.with_suffix(".preprocessing.npz")
            arrays = self.archive_arrays(preprocessor)
            arrays["fill"][0] += .25
            changed = Path(directory) / "changed.preprocessing.npz"
            np.savez_compressed(changed, **arrays)
            with self.assertRaisesRegex(ValueError, "identity"):
                prepare_weather(
                    frame.loc[frame.role.eq("test")], Path(directory) / "later.npz",
                    representation="numeric", preprocessor=changed,
                )

    def test_changed_embedding_values_are_rejected(self):
        frame = self.frame()
        with tempfile.TemporaryDirectory() as directory:
            bank = Path(directory) / "bank.npz"
            self.bank(frame, bank)
            prepared = Path(directory) / "text.npz"
            prepare_weather(frame, prepared, representation="text", embedding_bank=bank)
            arrays = self.archive_arrays(bank)
            arrays["embedding"][0, 0] += .01
            changed = Path(directory) / "changed-bank.npz"
            np.savez_compressed(changed, **arrays)
            with self.assertRaisesRegex(ValueError, "identity"):
                prepare_weather(
                    frame.loc[frame.role.eq("test")], Path(directory) / "later.npz",
                    representation="text", embedding_bank=changed,
                    preprocessor=prepared.with_suffix(".preprocessing.npz"),
                )

    def test_model_snapshot_hash_depends_on_contents_and_relative_names(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first", Path(directory) / "second"
            for path in (first, second):
                path.mkdir()
                (path / "config.json").write_text('{"model_type":"fixture"}', encoding="utf-8")
                (path / "model.safetensors").write_bytes(b"fixture weights")
            expected = model_digest(first)
            self.assertEqual(expected, model_digest(second))
            (second / "model.safetensors").write_bytes(b"different weights")
            self.assertNotEqual(expected, model_digest(second))


if __name__ == "__main__":
    unittest.main()
