"""A small synthetic example for checking the installed forecasting pipeline."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .data import SequenceData
from .evaluation import metrics
from .sequence import ForecastModel, SequenceConfig
from .training import fit, predict, TrainingConfig, write_json


def run_demo(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(17)
    n, length, horizons, fields = 72, 48, 12, 5
    phase = rng.uniform(-1, 1, size=(n, 1))
    history = np.clip(4 + 2 * np.sin(np.linspace(0, 3, length)[None, :] + phase)
                      + rng.normal(0, .05, (n, length)), 0, 10).astype(np.float32)
    future = rng.normal(size=(n, horizons, fields)).astype(np.float32)
    target = np.clip(history[:, -1:] + .2 * future[:, :, 0], 0, 10).astype(np.float32)
    train = np.broadcast_to((np.arange(n) < 40)[:, None], target.shape).copy()
    selection = np.broadcast_to(((np.arange(n) >= 40) & (np.arange(n) < 56))[:, None], target.shape).copy()
    evaluation = np.broadcast_to((np.arange(n) >= 56)[:, None], target.shape).copy()
    data = SequenceData(history, future, target, train, selection, evaluation,
                        np.datetime_as_string(np.datetime64("2025-01-01") + np.arange(n)),
                        np.full(n, "synthetic"),
                        {"kind": "sequence", "synthetic": True, "capacity_mw": 10.0})
    data.save(output / "input.npz")
    cfg = SequenceConfig(d_model=16, heads=2, layers=1, dropout=0.)
    torch.manual_seed(42)
    model = ForecastModel("dense_numeric", length, fields, horizons, config=cfg)
    spec = {"family": "sequence", "variant": "dense_numeric", "seed": 42, "random": False,
            "history_length": length, "features": fields, "horizons": horizons,
            "architecture": asdict(cfg), "revision": None}
    training = TrainingConfig(minimum_epochs=2, maximum_epochs=2, batch_size=16, microbatch=8)
    fit(model, data, output / "model", config=training, model_spec=spec)
    prediction = predict(model, data)
    np.savez_compressed(output / "predictions.npz", prediction=prediction, target=target, mask=evaluation,
                        issue_dates=data.issue_dates, station_ids=data.station_ids, sample_keys=data.sample_keys)
    report = metrics(target[evaluation], prediction[evaluation])
    report["synthetic"] = True
    write_json(output / "metrics.json", report)
    print(f"Synthetic example complete: {output}")
