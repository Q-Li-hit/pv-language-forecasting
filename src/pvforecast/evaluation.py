"""Point, ensemble and interval evaluation on a shared set of PV targets.

Power inputs are in MW. Error differences are evaluated method minus reference;
negative values favour the evaluated method. Functions require finite inputs so
the caller must establish one common evaluation mask before comparing methods.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from numbers import Integral

import numpy as np


def _vector(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _matching_vectors(**values):
    arrays = {name: _vector(value, name) for name, value in values.items()}
    if len({array.shape for array in arrays.values()}) != 1:
        raise ValueError("All target and forecast arrays must have the same shape")
    return tuple(arrays.values())


def _capacity(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("capacity_mw must be finite and positive")
    return value


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def metrics(y, prediction, *, capacity_mw=None, reference=None):
    """Compute pooled point errors, optionally capacity-normalized and skill.

    ``R2`` is undefined for a constant target; ``Skill_pooled`` is undefined
    when the reference RMSE is zero. These two cases return ``nan``.
    """
    y, prediction = _matching_vectors(y=y, prediction=prediction)
    error = prediction - y
    mse = float(np.mean(error**2))
    rmse = float(np.sqrt(mse))
    total = float(np.sum((y - y.mean()) ** 2))
    result = {
        "samples": int(y.size),
        "MSE_MW2": mse,
        "RMSE_MW": rmse,
        "MAE_MW": float(np.mean(np.abs(error))),
        "MBE_MW": float(np.mean(error)),
        "AE95_MW": float(np.quantile(np.abs(error), 0.95)),
        "R2": float(1 - np.sum(error**2) / total) if total else float("nan"),
    }
    if capacity_mw is not None:
        capacity = _capacity(capacity_mw)
        result["nRMSE_percent"] = 100 * rmse / capacity
        result["nMAE_percent"] = 100 * result["MAE_MW"] / capacity
    if reference is not None:
        _, reference = _matching_vectors(y=y, reference=reference)
        reference_rmse = float(np.sqrt(np.mean((reference - y) ** 2)))
        result["Skill_pooled"] = (
            1 - rmse / reference_rmse if reference_rmse else float("nan")
        )
    return result


def _utc_days(issue_dates, length):
    values = np.asarray(issue_dates)
    if values.ndim != 1 or values.size != length:
        raise ValueError("issue_dates must have one date or timestamp per target")
    if np.issubdtype(values.dtype, np.datetime64):
        days = values.astype("datetime64[D]")
    else:
        converted = []
        for value in values:
            if isinstance(value, np.datetime64):
                converted.append(value.astype("datetime64[D]"))
                continue
            if isinstance(value, str):
                try:
                    value = datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError as exc:
                    raise ValueError("issue_dates must contain ISO dates or timestamps") from exc
            if isinstance(value, datetime):
                if value.tzinfo is not None:
                    value = value.astimezone(timezone.utc)
                value = value.date()
            if not isinstance(value, date):
                raise ValueError("issue_dates must contain dates or timestamps")
            converted.append(np.datetime64(value, "D"))
        days = np.asarray(converted, dtype="datetime64[D]")
    if np.isnat(days).any():
        raise ValueError("issue_dates cannot contain missing dates")
    return days


def paired_date_block_bootstrap(
    y,
    evaluated,
    reference,
    issue_dates,
    *,
    block_days=7,
    repetitions=2000,
    confidence=0.95,
    seed=20261008,
):
    """Percentile confidence intervals for paired RMSE and MAE differences.

    Circular moving blocks sample consecutive calendar dates, including dates
    without targets. Each replicate contains the original calendar-span length.
    All stations and horizons on a sampled UTC issue date travel together; the
    same date draw is used for both forecasts. Timezone-naive dates/timestamps
    are interpreted as UTC. Draws containing no targets are discarded.

    The interval describes date variation of these fixed predictions. It does
    not include retraining randomness, multiplicity correction, or a guarantee
    of coverage for an individual station or weather condition.
    """
    y, evaluated, reference = _matching_vectors(
        y=y, evaluated=evaluated, reference=reference
    )
    block_days = _positive_integer(block_days, "block_days")
    repetitions = _positive_integer(repetitions, "repetitions")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    days = _utc_days(issue_dates, y.size)
    unique, inverse = np.unique(days, return_inverse=True)
    if unique.size < 2:
        raise ValueError("At least two issue dates are required")

    errors_a, errors_b = evaluated - y, reference - y
    statistics = np.column_stack(
        (np.ones(y.size), errors_a**2, errors_b**2, np.abs(errors_a), np.abs(errors_b))
    )
    calendar_length = int((unique[-1] - unique[0]).astype(int)) + 1
    daily = np.zeros((calendar_length, statistics.shape[1]), dtype=np.float64)
    occupied_indices = (unique - unique[0]).astype(int)
    for column in range(statistics.shape[1]):
        daily[occupied_indices, column] = np.bincount(
            inverse, weights=statistics[:, column], minlength=unique.size
        )

    rng = np.random.default_rng(seed)
    blocks_per_draw = (calendar_length + block_days - 1) // block_days
    offsets = np.arange(block_days)
    differences = []
    # Bound temporary storage when many dates or replicates are requested.
    batch_size = max(1, min(repetitions, 1_000_000 // max(calendar_length, 1)))
    for start in range(0, repetitions, batch_size):
        size = min(batch_size, repetitions - start)
        starts = rng.integers(0, calendar_length, size=(size, blocks_per_draw))
        indices = ((starts[..., None] + offsets) % calendar_length).reshape(size, -1)
        totals = daily[indices[:, :calendar_length]].sum(axis=1)
        totals = totals[totals[:, 0] > 0]
        differences.append(
            np.column_stack(
                (
                    np.sqrt(totals[:, 1] / totals[:, 0])
                    - np.sqrt(totals[:, 2] / totals[:, 0]),
                    (totals[:, 3] - totals[:, 4]) / totals[:, 0],
                )
            )
        )
    draws = np.concatenate(differences, axis=0)
    if draws.size == 0:
        raise ValueError("No resampled block contains an evaluation target")
    alpha = 1 - float(confidence)
    lower, upper = np.quantile(draws, [alpha / 2, 1 - alpha / 2], axis=0)
    full = statistics.sum(axis=0)
    return {
        "delta_RMSE_MW": float(np.sqrt(full[1] / full[0]) - np.sqrt(full[2] / full[0])),
        "RMSE_CI_low_MW": float(lower[0]),
        "RMSE_CI_high_MW": float(upper[0]),
        "delta_MAE_MW": float((full[3] - full[4]) / full[0]),
        "MAE_CI_low_MW": float(lower[1]),
        "MAE_CI_high_MW": float(upper[1]),
        "samples": int(y.size),
        "issue_dates": int(unique.size),
        "calendar_days": calendar_length,
        "block_days": block_days,
        "repetitions": repetitions,
        "valid_repetitions": int(draws.shape[0]),
        "confidence": float(confidence),
        "seed_uncertainty_included": False,
    }


def mse_average_decomposition(predictions, y):
    """Separate mean single-model MSE from the MSE reduction by averaging.

    ``predictions`` has shape (models, targets). All calculations use float64;
    the arithmetic mean can therefore differ slightly from a saved float32
    average. The exact identity is mean single-model MSE = mean-prediction MSE
    + mean squared deviation of the individual predictions from their mean.
    """
    y = _vector(y, "y")
    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != y.size:
        raise ValueError("predictions must have shape (models, targets)")
    if not np.isfinite(values).all():
        raise ValueError("predictions must contain only finite values")
    mean_prediction = values.mean(axis=0, dtype=np.float64)
    single_mse = np.mean((values - y) ** 2, axis=1)
    mean_mse = float(np.mean((mean_prediction - y) ** 2))
    reduction = float(np.mean((values - mean_prediction) ** 2))
    mean_single = float(single_mse.mean())
    return {
        "models": int(values.shape[0]),
        "samples": int(y.size),
        "mean_single_MSE_MW2": mean_single,
        "mean_prediction_MSE_MW2": mean_mse,
        "averaging_MSE_reduction_MW2": reduction,
        "identity_residual_MW2": mean_single - mean_mse - reduction,
        "mean_single_RMSE_MW": float(np.mean(np.sqrt(single_mse))),
        "RMS_of_single_MSE_MW": float(np.sqrt(mean_single)),
        "mean_prediction_RMSE_MW": float(np.sqrt(mean_mse)),
        "mean_prediction_MAE_MW": float(np.mean(np.abs(mean_prediction - y))),
        "averaging_MSE_reduction_fraction": reduction / mean_single if mean_single else float("nan"),
    }


def fit_simplex_weights(predictions, y, *, prior=None, penalty=0.0):
    """Fit nonnegative MSE weights summing to one on development data only.

    Rows are targets and columns are ensemble members. The objective is mean
    squared error plus ``penalty * sum((weights - prior)**2)``. The default
    prior is uniform. This is the study's SLSQP optimization and tolerances.
    Removing a member and refitting differs from skipping an exactly zero-
    weight member while retaining the original fitted weights.
    """
    from scipy.optimize import minimize

    y = _vector(y, "y")
    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != y.size or values.shape[1] == 0:
        raise ValueError("predictions must have shape (targets, members)")
    if not np.isfinite(values).all():
        raise ValueError("predictions must contain only finite values")
    penalty = float(penalty)
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("penalty must be finite and nonnegative")
    members = values.shape[1]
    pi = np.full(members, 1 / members) if prior is None else _vector(prior, "prior")
    if pi.shape != (members,) or np.any(pi < 0) or not np.isclose(pi.sum(), 1):
        raise ValueError("prior must have one nonnegative normalized weight per member")
    if members == 1:
        return np.ones(1, dtype=np.float64)

    def objective(weights):
        residual = values @ weights - y
        return np.mean(residual**2) + penalty * np.sum((weights - pi) ** 2)

    def gradient(weights):
        return 2 * values.T @ (values @ weights - y) / y.size + 2 * penalty * (weights - pi)

    solution = minimize(
        objective,
        pi,
        jac=gradient,
        method="SLSQP",
        bounds=[(0, 1)] * members,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1,
                     "jac": lambda w: np.ones(members)},
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not solution.success:
        raise RuntimeError(f"Simplex optimization failed: {solution.message}")
    weights = np.maximum(solution.x, 0)
    return weights / weights.sum()


def interval_metrics(y, lower, upper, *, alpha=0.1, capacity_mw=None, trajectory_ids=None):
    """Evaluate central prediction intervals and optional complete trajectories.

    Coverage is a fraction; interval width and interval score retain MW units.
    Each trajectory receives equal weight in simultaneous trajectory coverage.
    The caller must supply every required step of each complete trajectory;
    filtering out missing steps changes the coverage object. These empirical
    scores do not constitute a conditional coverage guarantee.
    """
    y, lower, upper = _matching_vectors(y=y, lower=lower, upper=upper)
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if np.any(lower > upper):
        raise ValueError("lower bounds must not exceed upper bounds")
    covered = (y >= lower) & (y <= upper)
    width = upper - lower
    score = width + 2 / alpha * np.maximum(lower - y, 0) + 2 / alpha * np.maximum(y - upper, 0)
    result = {
        "samples": int(y.size),
        "nominal_coverage": 1 - float(alpha),
        "coverage": float(np.mean(covered)),
        "mean_width_MW": float(np.mean(width)),
        "interval_score_MW": float(np.mean(score)),
    }
    if capacity_mw is not None:
        result["normalized_mean_width"] = result["mean_width_MW"] / _capacity(capacity_mw)
    if trajectory_ids is not None:
        identifiers = np.asarray(trajectory_ids)
        if identifiers.ndim != 1 or identifiers.size != y.size:
            raise ValueError("trajectory_ids must have one identifier per target")
        if any(identifier is None or identifier != identifier for identifier in identifiers):
            raise ValueError("trajectory_ids cannot contain missing identifiers")
        _, inverse = np.unique(identifiers, return_inverse=True)
        trajectories = int(inverse.max()) + 1
        misses = np.bincount(inverse, weights=~covered, minlength=trajectories)
        result["trajectories"] = trajectories
        result["trajectory_coverage"] = float(np.mean(misses == 0))
    return result
