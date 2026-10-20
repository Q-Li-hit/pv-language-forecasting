"""Small complete fits covering loss masks, checkpointing, and prediction."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from test_data import scalar_fixture, sequence_fixture


@unittest.skipIf(torch is None, "Install the sequence extra to test training.")
class TrainingTests(unittest.TestCase):
    def setUp(self):
        from pvforecast.training import TrainingConfig

        self.settings = TrainingConfig(
            learning_rate=.001, batch_size=4, microbatch=4,
            minimum_epochs=1, maximum_epochs=2, patience=2, threads=1,
        )

    def sequence_model(self):
        from pvforecast.sequence import ForecastModel, SequenceConfig

        torch.manual_seed(54)
        return ForecastModel(
            "dense_numeric", 12, 4, 3,
            config=SequenceConfig(d_model=8, heads=2, layers=1, dropout=0.),
        )

    def scalar_model(self):
        from pvforecast.weather import CapacityMatchedControl, TransformerConfig

        torch.manual_seed(63)
        return CapacityMatchedControl(6, 5, TransformerConfig(d_model=8, heads=2, layers=1, dropout=0.))

    def test_sequence_fit_counts_only_masked_targets_and_reloads(self):
        from pvforecast.training import fit, load_task_parameters, predict

        data = sequence_fixture()
        model = self.sequence_model()
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "fit"
            report = fit(model, data, run, config=self.settings, seed=54)
            self.assertTrue(all(row["training_targets"] == 7 for row in report["history"]))
            self.assertTrue(all(row["selection_targets"] == 5 for row in report["history"]))
            self.assertEqual(report["best_selection_loss"], min(row["selection_loss"] for row in report["history"]))
            with (run / "fit.json").open(encoding="utf-8") as stream:
                self.assertEqual(json.load(stream)["checkpoint_sha256"], report["checkpoint_sha256"])
            state = torch.load(run / "parameters.pt", map_location="cpu", weights_only=True)
            self.assertTrue(all(isinstance(value, torch.Tensor) for value in state.values()))
            self.assertEqual(set(state), set(model.trainable_state()))
            expected = predict(model, data, batch_size=3)
            restored = self.sequence_model()
            load_task_parameters(restored, run / "parameters.pt")
            actual = predict(restored, data, batch_size=3)
            np.testing.assert_array_equal(expected, actual)
            self.assertEqual(actual.shape, data.target.shape)
            self.assertTrue(((actual >= 0) & (actual <= 10)).all())

    def test_evaluation_labels_do_not_change_training_or_selection(self):
        from pvforecast.training import fit

        data = sequence_fixture()
        changed = sequence_fixture()
        changed.target[changed.evaluation_mask] += 1000
        left, right = self.sequence_model(), self.sequence_model()
        with tempfile.TemporaryDirectory() as directory:
            first = fit(left, data, Path(directory) / "original", config=self.settings, seed=54)
            second = fit(right, changed, Path(directory) / "changed", config=self.settings, seed=54)
        self.assertEqual(first["history"], second["history"])
        for key, value in left.trainable_state().items():
            self.assertTrue(torch.equal(value, right.trainable_state()[key]))

    def test_microbatching_preserves_effective_batch_loss_and_update(self):
        from pvforecast.training import fit

        class TrajectoryHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Linear(4, 1)

            def forward(self, history, future):
                return self.head(future).squeeze(-1)

        torch.manual_seed(54)
        left = TrajectoryHead()
        torch.manual_seed(54)
        right = TrajectoryHead()
        data = sequence_fixture()
        with tempfile.TemporaryDirectory() as directory:
            whole = fit(left, data, Path(directory) / "whole", config=self.settings, seed=54)
            split = fit(right, data, Path(directory) / "micro", config=replace(self.settings, microbatch=1), seed=54)
        for key, value in left.state_dict().items():
            torch.testing.assert_close(value, right.state_dict()[key], atol=1e-7, rtol=1e-6)
        for a, b in zip(whole["history"], split["history"]):
            self.assertAlmostEqual(a["training_loss"], b["training_loss"], places=5)
            self.assertAlmostEqual(a["selection_loss"], b["selection_loss"], places=5)

    def test_scalar_residual_fit_prediction_and_checkpoint(self):
        from pvforecast.training import fit, load_task_parameters, predict

        data = scalar_fixture()
        model = self.scalar_model()
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "scalar"
            report = fit(model, data, run, config=self.settings, seed=63)
            self.assertTrue(all(row["training_targets"] == 4 for row in report["history"]))
            self.assertTrue(all(row["selection_targets"] == 3 for row in report["history"]))
            result = predict(model, data, batch_size=3)
            clone = self.scalar_model()
            load_task_parameters(clone, run / "parameters.pt")
            np.testing.assert_array_equal(result, predict(clone, data, batch_size=3))
        with torch.no_grad():
            raw = model(
                torch.as_tensor(data.history[:3]), torch.as_tensor(data.state[:3]),
                torch.as_tensor(data.context[:3]),
            ).numpy()
        np.testing.assert_allclose(result[:3], np.clip(raw + data.baseline[:3], 0, 10), atol=1e-6)

    def test_masked_loss_ignores_unselected_target_values(self):
        from pvforecast.training import _loss_sum

        data = sequence_fixture()
        output = torch.tensor([[1., 100., 3.]])
        target = torch.tensor([[0., -1000., 1.]])
        mask = torch.tensor([[True, False, True]])
        result = _loss_sum(output, target, mask, data, self.settings)
        self.assertAlmostEqual(result.item(), (1 - .175) + (2 - .175), places=6)

    def test_missing_selection_or_wrong_checkpoint_is_rejected(self):
        from pvforecast.training import fit, load_task_parameters

        model = self.sequence_model()
        data = sequence_fixture()
        data.selection_mask[:] = False
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                fit(model, data, Path(directory) / "missing", config=self.settings)
            path = Path(directory) / "wrong.pt"
            torch.save({"unrelated": torch.tensor(1)}, path)
            with self.assertRaises(ValueError):
                load_task_parameters(model, path)


if __name__ == "__main__":
    unittest.main()
