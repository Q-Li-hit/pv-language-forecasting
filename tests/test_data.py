"""Portable archive validation and equivalent-power calculations."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pvforecast.data import (
    ScalarData, SequenceData, equivalent_power, prompt_statistics, tree_features,
)


def sequence_fixture():
    generator = np.random.default_rng(17)
    n, horizons = 8, 3
    train = np.zeros((n, horizons), bool)
    selection = np.zeros_like(train)
    evaluation = np.zeros_like(train)
    train[:4] = [[True, False, True], [True, False, False], [True, True, True], [False, True, False]]
    selection[4:6] = [[True, True, True], [True, False, True]]
    evaluation[6:] = True
    return SequenceData(
        generator.normal(size=(n, 12)).astype(np.float32),
        generator.normal(size=(n, horizons, 4)).astype(np.float32),
        generator.uniform(1, 4, size=(n, horizons)).astype(np.float32),
        train, selection, evaluation,
        np.asarray([f"2025-01-{i+1:02d}" for i in range(n)]),
        np.asarray(["BON"] * n), {"capacity_mw": 10., "preprocessor": "fixture"},
    )


def scalar_fixture():
    generator = np.random.default_rng(26)
    n = 9
    return ScalarData(
        generator.normal(size=(n, 6)).astype(np.float32),
        generator.normal(size=(n, 5)).astype(np.float32),
        generator.normal(size=(n, 33)).astype(np.float32),
        generator.uniform(1, 4, n).astype(np.float32),
        np.ones(n, np.float32),
        np.asarray(["train"] * 4 + ["model_select"] * 3 + ["test"] * 2),
        np.asarray([f"2025-02-{i+1:02d}" for i in range(n)]),
        np.asarray(["BON"] * n), {"capacity_mw": 10., "preprocessor": "fixture"},
    )


class DataTests(unittest.TestCase):
    def test_sequence_archive_roundtrip_without_pickle(self):
        data = sequence_fixture()
        data.ids = np.tile(np.asarray([1, 2, 0], np.int32), (len(data.history), 1))
        data.attention = data.ids != 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sequence.npz"
            data.save(path)
            with np.load(path, allow_pickle=False) as archive:
                self.assertTrue(all(archive[name].dtype.kind != "O" for name in archive.files))
            restored = SequenceData.load(path)
        self.assertEqual(restored.metadata, data.metadata)
        for name in ("history", "future", "target", "train_mask", "selection_mask", "evaluation_mask", "ids", "attention"):
            np.testing.assert_array_equal(getattr(restored, name), getattr(data, name))

    def test_scalar_archive_roundtrip_without_pickle(self):
        data = scalar_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scalar.npz"
            data.save(path)
            restored = ScalarData.load(path)
            with np.load(path, allow_pickle=False) as archive:
                self.assertTrue(all(archive[name].dtype.kind != "O" for name in archive.files))
        self.assertEqual(restored.metadata, data.metadata)
        np.testing.assert_array_equal(restored.roles, data.roles)
        np.testing.assert_array_equal(restored.context, data.context)

    def test_object_archive_is_rejected(self):
        data = sequence_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "object.npz"
            data.save(path)
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            arrays["metadata"] = np.asarray({"unexpected": "pickle"}, dtype=object)
            np.savez_compressed(path, **arrays)
            with self.assertRaises(ValueError):
                SequenceData.load(path)

    def test_sequence_shapes_masks_and_tokens_are_checked(self):
        malformed = [
            {"history": np.ones((8, 0))},
            {"future": np.ones((8, 2, 4))},
            {"target": np.full((8, 3), np.nan)},
            {"train_mask": np.ones((8, 3), np.int32)},
            {"selection_mask": sequence_fixture().train_mask.copy()},
            {"ids": np.ones((8, 2), np.int64)},
            {"ids": np.ones((8, 2)), "attention": np.ones((8, 2), np.int8)},
            {"ids": np.ones((8, 2), np.int64), "attention": np.zeros((8, 2), np.int8)},
            {"issue_dates": np.asarray(["NaT"] * 8)},
            {"metadata": []},
        ]
        for changes in malformed:
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                replace(sequence_fixture(), **changes).validate()

    def test_scalar_shapes_roles_and_dates_are_checked(self):
        malformed = [
            {"context": np.ones((9, 32))},
            {"baseline": np.ones((9, 1))},
            {"target": np.full(9, np.inf)},
            {"roles": np.asarray(["invalid"] * 9)},
            {"station_ids": np.asarray([""] * 9)},
            {"issue_dates": np.asarray(["NaT"] * 9)},
        ]
        for changes in malformed:
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                replace(scalar_fixture(), **changes).validate()

    def test_equivalent_power_units_clipping_and_missing_values(self):
        irradiance = np.asarray([-100, 0, 500, 1000, 2000, np.nan])
        temperature = np.asarray([25, 25, 25, 50, -50, 25])
        actual = equivalent_power(irradiance, temperature, capacity_mw=10)
        np.testing.assert_allclose(actual[:5], [0, 0, 5, 9, 10])
        self.assertTrue(np.isnan(actual[5]))
        for capacity in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                equivalent_power([500], [25], capacity_mw=capacity)

    def test_tree_features_retain_history_target_features_and_prompt_statistics(self):
        data = sequence_fixture().validate()
        groups, steps = np.asarray([6, 1, 6]), np.asarray([2, 0, 1])
        actual = tree_features(data, groups, steps)
        self.assertEqual(actual.shape, (3, 12 + 4 + 9))
        np.testing.assert_array_equal(actual[:, :12], data.history[groups])
        np.testing.assert_array_equal(actual[:, 12:16], data.future[groups, steps])
        np.testing.assert_array_equal(actual[:, 16:], prompt_statistics(data.history)[groups])
        self.assertTrue(actual.flags.c_contiguous)
        self.assertEqual(actual.dtype, np.float32)
        with self.assertRaises(ValueError):
            prompt_statistics(np.ones((2, 4)))


if __name__ == "__main__":
    unittest.main()
