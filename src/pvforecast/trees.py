"""LightGBM baseline using the complete sequence inputs and prompt statistics."""

from pathlib import Path
import time

import numpy as np

from .data import tree_features
from .utils import write_json


TREE_CANDIDATES = ((31, 50), (63, 80), (127, 100))


def smooth_l1(y, prediction):
    error = abs(np.asarray(prediction, float) - np.asarray(y, float))
    return "selection_smooth_l1", float(np.where(error < .35, .5 * error**2 / .35,
                                                 error - .175).mean()), False


def fit_tree(data, output, *, threads=4):
    import lightgbm as lgb

    data.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    train_groups, train_steps = np.nonzero(data.train_mask)
    select_groups, select_steps = np.nonzero(data.selection_mask)
    if not len(train_groups) or not len(select_groups):
        raise ValueError("Training and selection masks must be nonempty.")
    train_x = tree_features(data, train_groups, train_steps)
    select_x = tree_features(data, select_groups, select_steps)
    train_y, select_y = data.target[train_groups, train_steps], data.target[select_groups, select_steps]
    trials = []
    best = float("inf")
    for leaves, minimum in TREE_CANDIDATES:
        started = time.perf_counter()
        model = lgb.LGBMRegressor(
            objective="regression", metric="None", num_leaves=leaves, min_child_samples=minimum,
            learning_rate=.04, reg_lambda=1., n_estimators=500, n_jobs=threads,
            random_state=20261014, verbosity=-1, deterministic=True,
            force_col_wise=True, feature_pre_filter=False,
        )
        model.fit(train_x, train_y, eval_set=[(select_x, select_y)], eval_metric=smooth_l1,
                  callbacks=[lgb.early_stopping(40, first_metric_only=True, verbose=False)])
        prediction = model.booster_.predict(select_x, num_iteration=model.best_iteration_)
        loss = smooth_l1(select_y, prediction)[1]
        row = {"num_leaves": leaves, "min_child_samples": minimum,
               "best_iteration": model.best_iteration_, "selection_loss": loss,
               "fit_seconds": time.perf_counter() - started}
        trials.append(row)
        # Resolve ties by the original string configuration identifier.
        key = (loss, f"leaves{leaves}_min{minimum}")
        if loss < best or (loss == best and key < best_key):
            best, best_key = loss, key
            text = model.booster_.model_to_string(num_iteration=model.best_iteration_)
            (output / "model.txt").write_text(text, encoding="utf-8")
            selected = row
    report = {"selected": selected, "trials": trials, "input_metadata": data.metadata,
              "history_length": data.history.shape[1], "features": data.future.shape[-1],
              "horizons": data.future.shape[1], "lightgbm": lgb.__version__}
    write_json(output / "fit.json", report)
    return report


def predict_tree(data, model_path, *, threads=4, capacity_mw=10., batch_size=4096):
    import lightgbm as lgb

    data.validate()
    model = lgb.Booster(model_str=Path(model_path).read_text(encoding="utf-8"),
                        params={"num_threads": threads})
    result = np.empty_like(data.target)
    stats_width = 9
    expected = data.history.shape[1] + data.future.shape[-1] + stats_width
    if model.num_feature() != expected:
        raise ValueError("Tree input dimensions differ from the archive.")
    for first in range(0, len(data.history), batch_size):
        groups = np.repeat(np.arange(first, min(first + batch_size, len(data.history))), data.target.shape[1])
        steps = np.tile(np.arange(data.target.shape[1]), len(np.unique(groups)))
        prediction = model.predict(tree_features(data, groups, steps), num_threads=threads)
        result[np.unique(groups)] = np.clip(prediction.reshape(-1, data.target.shape[1]), 0, capacity_mw)
    return result
