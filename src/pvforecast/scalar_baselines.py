"""Numerical baselines for the scalar weather-feature experiments."""

from dataclasses import dataclass
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np

from .utils import file_sha256, write_json


SCALAR_TREE_CANDIDATES = ((31, 50), (63, 80))


def _capacity(value):
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Capacity must be finite and positive.")
    return float(value)


def fit_scalar_tree(data, output, *, objective="direct", threads=4, capacity_mw=10.0,
                    maximum_estimators=500, early_stopping_rounds=40):
    """Select a direct or residual LightGBM using the model-selection rows.

    The numerical inputs are the common training-standardized state matrix.
    The two leaf settings, squared-error selection and regression defaults
    follow the scalar baseline experiment.
    """
    import lightgbm as lgb

    data.validate()
    capacity_mw = _capacity(capacity_mw)
    if objective not in ("direct", "residual"):
        raise ValueError("Objective must be direct or residual.")
    if min(threads, maximum_estimators, early_stopping_rounds) < 1:
        raise ValueError("Thread, estimator and stopping budgets must be positive.")
    training, selection = data.roles == "train", data.roles == "model_select"
    if not training.any() or not selection.any():
        raise ValueError("Training and model-selection rows are required.")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    train_y = data.target[training].astype(float)
    select_y = data.target[selection].astype(float)
    if objective == "residual":
        train_y -= data.baseline[training]
        select_y -= data.baseline[selection]
    trials = []
    best = float("inf")
    for leaves, minimum in SCALAR_TREE_CANDIDATES:
        started = time.perf_counter()
        estimator = lgb.LGBMRegressor(
            n_estimators=maximum_estimators, learning_rate=0.04, reg_lambda=1.0,
            colsample_bytree=0.9, num_leaves=leaves, min_child_samples=minimum,
            random_state=42, n_jobs=threads, verbosity=-1,
        )
        estimator.fit(
            data.state[training], train_y, eval_set=[(data.state[selection], select_y)],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        prediction = estimator.booster_.predict(data.state[selection], num_threads=threads)
        if objective == "residual":
            prediction += data.baseline[selection]
        score = float(np.mean((np.clip(prediction, 0, capacity_mw) - data.target[selection]) ** 2))
        trial = {"num_leaves": leaves, "min_child_samples": minimum,
                 "selection_mse_mw2": score, "best_iteration": estimator.best_iteration_,
                 "fit_seconds": time.perf_counter() - started}
        trials.append(trial)
        if score < best:
            best, selected = score, trial
            (destination / "model.txt").write_text(
                estimator.booster_.model_to_string(num_iteration=estimator.best_iteration_),
                encoding="utf-8",
            )
    report = {"family": "scalar_lightgbm", "objective": objective,
              "capacity_mw": capacity_mw, "features": data.state.shape[1],
              "train_rows": int(training.sum()), "selection_rows": int(selection.sum()),
              "selected": selected, "trials": trials, "input_metadata": data.metadata,
              "lightgbm": lgb.__version__, "model_sha256": file_sha256(destination / "model.txt")}
    write_json(destination / "fit.json", report)
    return report


def predict_scalar_tree(data, model_path, *, threads=4, batch_size=4096):
    """Predict from a fitted scalar-baseline directory with fixed preprocessing."""
    import lightgbm as lgb

    data.validate()
    if threads < 1 or batch_size < 1:
        raise ValueError("Thread and batch sizes must be positive.")
    directory = Path(model_path)
    report = json.loads((directory / "fit.json").read_text(encoding="utf-8"))
    if report["input_metadata"] != data.metadata:
        raise ValueError("Input preprocessing metadata differs from the fitted model.")
    if file_sha256(directory / "model.txt") != report["model_sha256"]:
        raise ValueError("Saved LightGBM model differs from its fitted record.")
    model = lgb.Booster(model_str=(directory / "model.txt").read_text(encoding="utf-8"),
                        params={"num_threads": threads})
    if model.num_feature() != data.state.shape[1]:
        raise ValueError("Scalar input dimensions differ from the fitted model.")
    result = []
    for first in range(0, len(data.target), batch_size):
        last = first + batch_size
        values = model.predict(data.state[first:last], num_threads=threads)
        if report["objective"] == "residual":
            values += data.baseline[first:last]
        result.append(np.clip(values, 0, report["capacity_mw"]).astype(np.float32))
    return np.concatenate(result)


def context_indices(station_ids, horizons, budget, seed):
    """Sample within training rows by station and horizon, then fill the budget.

    Indices refer to the supplied training arrays. Sampling follows the original
    station/horizon-balanced selector, including its uniform remainder draw.
    """
    import pandas as pd

    stations, horizons = np.asarray(station_ids), np.asarray(horizons)
    if (stations.ndim != 1 or horizons.shape != stations.shape or not len(stations)
            or not np.isfinite(horizons.astype(float)).all()):
        raise ValueError("Aligned nonempty station and finite horizon vectors are required.")
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)) or not 0 < budget <= len(stations):
        raise ValueError("Context budget must be between one and the training row count.")
    groups = [np.asarray(v, dtype=int) for v in pd.DataFrame({
        "station_id": stations, "horizon": horizons,
    }).groupby(["station_id", "horizon"], observed=True).indices.values()]
    rng = np.random.default_rng(seed)
    each = budget // len(groups)
    initial = np.concatenate([rng.choice(group, min(each, len(group)), replace=False) for group in groups])
    unused = np.setdiff1d(np.arange(len(stations)), initial)
    return np.sort(np.r_[initial, rng.choice(unused, budget - len(initial), replace=False)])


def _local_weights(weights):
    path = Path(weights).expanduser()
    if not path.is_file():
        raise FileNotFoundError("TabPFN requires an existing local model-weight file.")
    return path


def _tabpfn_estimator(weights, seed, device):
    weights = _local_weights(weights)
    # The estimator receives an explicit local file and runs with network access disabled.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
    os.environ["TABPFN_NO_BROWSER"] = "1"
    from tabpfn import TabPFNRegressor
    return TabPFNRegressor(model_path=str(weights), device=device, n_estimators=1,
                           random_state=seed, fit_mode="fit_with_cache", n_preprocessing_jobs=4)


@dataclass
class TabPFNPredictor:
    estimator: object
    input_metadata: dict
    capacity_mw: float
    features: int


def fit_tabpfn(data, weights, *, horizons, budget=512, seed=42, device="cpu",
               capacity_mw=10.0, output=None):
    """Fit a local contextual predictor using labels from training rows only.

    Context-budget selection is the caller's development-stage decision. Raw
    numerical inputs must be supplied in ``data.raw_state``; their missing
    values have already been filled using the training preprocessing.
    """
    data.validate()
    weights = _local_weights(weights)
    raw = getattr(data, "raw_state", None)
    if raw is None:
        raise ValueError("TabPFN requires unstandardized raw_state inputs.")
    raw = np.asarray(raw, np.float32)
    if raw.shape != data.state.shape or not np.isfinite(raw).all():
        raise ValueError("Raw state must match the finite numerical-state matrix.")
    horizons = np.asarray(horizons)
    if horizons.shape != data.target.shape:
        raise ValueError("Supply one horizon value per scalar target.")
    training_rows = np.flatnonzero(data.roles == "train")
    chosen = context_indices(data.station_ids[training_rows], horizons[training_rows], budget, seed)
    indices = training_rows[chosen]
    capacity_mw = _capacity(capacity_mw)
    estimator = _tabpfn_estimator(weights, seed, device)
    started = time.perf_counter()
    estimator.fit(raw[indices], data.target[indices])
    report = {"family": "scalar_tabpfn", "package_version": importlib.metadata.version("tabpfn"),
              "budget": int(budget), "seed": int(seed), "device": device,
              "capacity_mw": capacity_mw, "features": raw.shape[1], "n_estimators": 1,
              "context_training_indices": indices.tolist(), "context_roles": "train",
              "input_metadata": data.metadata, "weights_sha256": file_sha256(weights),
              "fit_seconds": time.perf_counter() - started,
              "context_budget_selection": "supplied by caller; use model-selection data only",
              "inputs": "training-imputed, unstandardized common numerical features and state codes"}
    if output is not None:
        destination = Path(output)
        destination.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(destination / "context.npz", features=raw[indices],
                            target=data.target[indices], training_indices=indices)
        report["context_sha256"] = file_sha256(destination / "context.npz")
        write_json(destination / "fit.json", report)
    return TabPFNPredictor(estimator, dict(data.metadata), capacity_mw, raw.shape[1]), report


def load_tabpfn(directory, weights, *, device="cpu"):
    """Restore a contextual predictor from numeric context arrays and local weights."""
    directory = Path(directory)
    report = json.loads((directory / "fit.json").read_text(encoding="utf-8"))
    weights = _local_weights(weights)
    if file_sha256(weights) != report["weights_sha256"]:
        raise ValueError("Local TabPFN weights differ from the saved context configuration.")
    if file_sha256(directory / "context.npz") != report["context_sha256"]:
        raise ValueError("Saved TabPFN context differs from its fitted record.")
    with np.load(directory / "context.npz", allow_pickle=False) as archive:
        features, target = archive["features"], archive["target"]
    estimator = _tabpfn_estimator(weights, report["seed"], device)
    estimator.fit(features, target)
    return TabPFNPredictor(estimator, report["input_metadata"], report["capacity_mw"], report["features"])


def predict_tabpfn(predictor, data, *, batch_size=4096):
    data.validate()
    if batch_size < 1:
        raise ValueError("Prediction batch size must be positive.")
    if data.metadata != predictor.input_metadata:
        raise ValueError("Input preprocessing metadata differs from the fitted context.")
    raw = getattr(data, "raw_state", None)
    if raw is None or raw.shape != (len(data.target), predictor.features):
        raise ValueError("Matching unstandardized raw_state inputs are required.")
    result = []
    for first in range(0, len(raw), batch_size):
        values = predictor.estimator.predict(raw[first:first + batch_size], output_type="mean")
        result.append(np.clip(values, 0, predictor.capacity_mw).astype(np.float32))
    return np.concatenate(result)
