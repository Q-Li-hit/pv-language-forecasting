import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pvforecast.benchmark import benchmark_inference
from pvforecast.cli import main
from pvforecast.sequence import ForecastModel, SequenceConfig


class CommandTests(unittest.TestCase):
    def test_resident_benchmark_reports_output_scope(self):
        model = ForecastModel("dense_numeric", 8, 3, 2,
                              config=SequenceConfig(d_model=8, heads=2, layers=1, dropout=0))
        inputs = (torch.ones(2, 8), torch.ones(2, 2, 3))
        result = benchmark_inference(model, inputs, warmup=1, repetitions=3)
        self.assertEqual(result["output_count"], 4)
        self.assertGreaterEqual(result["P95_ms"], result["P50_ms"])

    def test_evaluate_rejects_different_sample_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            arrays = dict(target=np.array([1., 1.]), prediction=np.array([1.1, 1.2]),
                          mask=np.array([True, True]), issue_dates=np.array(["2025-01-01"] * 2),
                          station_ids=np.array(["a", "a"]), sample_keys=np.array(["a|00", "a|01"]))
            np.savez_compressed(tmp / "a.npz", **arrays)
            arrays["sample_keys"] = arrays["sample_keys"][::-1]
            np.savez_compressed(tmp / "b.npz", **arrays)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(["evaluate", "--predictions", str(tmp / "a.npz"), "--reference", str(tmp / "b.npz"),
                      "--output", str(tmp / "result.json")])
            self.assertEqual(error.exception.code, 2)
            self.assertFalse((tmp / "result.json").exists())

    def test_demo_then_reload_and_evaluate(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            tmp = Path(tmp)
            main(["demo", "--output", str(tmp / "demo")])
            main(["predict", "--data", str(tmp / "demo/input.npz"), "--model", str(tmp / "demo/model"),
                  "--output", str(tmp / "reloaded.npz")])
            with np.load(tmp / "demo/predictions.npz") as a, np.load(tmp / "reloaded.npz") as b:
                np.testing.assert_array_equal(a["prediction"], b["prediction"])
            main(["evaluate", "--predictions", str(tmp / "reloaded.npz"), "--output", str(tmp / "metrics.json")])
            self.assertTrue((tmp / "metrics.json").is_file())


if __name__ == "__main__":
    unittest.main()
