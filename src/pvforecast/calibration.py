"""Fixed and arrived-label calibration for independent quantile forecasts.

These utilities adjust an already fitted quantile predictor. Labels used to
update calibration do not update predictor parameters. Power is expressed in MW.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd


def _finite_array(values, name, *, ndim=None, nonempty=True):
    array = np.asarray(values, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if nonempty and array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _levels(values):
    levels = _finite_array(values, "levels", ndim=1)
    if np.any((levels <= 0) | (levels >= 1)) or np.any(np.diff(levels) <= 0):
        raise ValueError("levels must be strictly increasing and between zero and one")
    return levels


def _horizons(values, size=None):
    values = _finite_array(values, "horizons", ndim=1)
    if size is not None and values.size != size:
        raise ValueError("horizons must have one value per target")
    if np.any(values < 1) or np.any(values != np.floor(values)):
        raise ValueError("horizons must contain positive integers")
    return values.astype(np.int64)


def conformal_quantile(scores, *, alpha=0.1, nonnegative=True):
    """Return the finite-sample CQR threshold at coverage ``1 - alpha``.

    The order is ceil((n + 1) * (1 - alpha)). If that order exceeds n, the
    result is positive infinity, as in the study implementation. An empty
    calibration set has the same result. Signed conformity scores are retained;
    the study truncates only the final threshold at zero to avoid shrinking an
    interval. Set ``nonnegative=False`` to inspect its signed order statistic.
    """
    values = _finite_array(scores, "scores", ndim=1, nonempty=False)
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    order = int(np.ceil((values.size + 1) * (1 - alpha)))
    if order > values.size:
        return float("inf")
    threshold = float(np.partition(values, order - 1)[order - 1])
    return max(0.0, threshold) if nonnegative else threshold


def conformity_scores(y, lower, upper):
    """Compute signed CQR scores max(lower - y, y - upper).

    Endpoint arrays may have shape (targets,) or (targets, coverage levels).
    Scores inside a raw interval can be negative. All inputs must share the
    same finite target set, and each raw lower endpoint must not exceed upper.
    """
    y = _finite_array(y, "y", ndim=1)
    lower = _finite_array(lower, "lower")
    upper = _finite_array(upper, "upper")
    if lower.ndim not in (1, 2) or lower.shape != upper.shape or lower.shape[0] != y.size:
        raise ValueError("Endpoints must have matching target or target-by-level shapes")
    if np.any(lower > upper):
        raise ValueError("Raw lower endpoints must not exceed upper endpoints")
    target = y if lower.ndim == 1 else y[:, None]
    return np.maximum(lower - target, target - upper)


def rearrange_quantiles(predictions, *, capacity_mw=10.0):
    """Sort a quantile grid in increasing probability order and clip to capacity.

    Columns must correspond to increasing quantile probabilities. Sorting is
    label-independent monotone rearrangement, not an empirical recalibration.
    """
    predictions = _finite_array(predictions, "predictions", ndim=2)
    capacity = float(capacity_mw)
    if not np.isfinite(capacity) or capacity <= 0:
        raise ValueError("capacity_mw must be finite and positive")
    return np.clip(np.sort(predictions, axis=1), 0, capacity)


def calibrated_interval(lower, upper, adjustment, *, capacity_mw=10.0, nested=False):
    """Expand, physically clip and optionally nest a prediction-interval family.

    For a family, columns must be ordered by increasing coverage level.
    Cumulative minima of lower and maxima of upper endpoints enlarge outer
    intervals for nesting. Positive-infinite adjustments yield the full physical
    range. Negative adjustments are rejected because this study uses expansion.
    """
    lower = _finite_array(lower, "lower")
    upper = _finite_array(upper, "upper")
    if lower.ndim not in (1, 2) or lower.shape != upper.shape:
        raise ValueError("Endpoints must have matching one- or two-dimensional shapes")
    if np.any(lower > upper):
        raise ValueError("Raw lower endpoints must not exceed upper endpoints")
    capacity = float(capacity_mw)
    if not np.isfinite(capacity) or capacity <= 0:
        raise ValueError("capacity_mw must be finite and positive")
    adjustment = np.asarray(adjustment, dtype=np.float64)
    if np.isnan(adjustment).any() or np.any(adjustment < 0):
        raise ValueError("adjustment must be nonnegative and cannot contain NaN")
    try:
        adjustment = np.broadcast_to(adjustment, lower.shape)
    except ValueError as exc:
        raise ValueError("adjustment must broadcast to the endpoint shape") from exc
    lo = np.clip(np.clip(lower, 0, capacity) - adjustment, 0, capacity)
    hi = np.clip(np.clip(upper, 0, capacity) + adjustment, 0, capacity)
    if nested:
        if lower.ndim != 2:
            raise ValueError("nested=True requires target-by-level endpoint arrays")
        lo = np.minimum.accumulate(lo, axis=1)
        hi = np.maximum.accumulate(hi, axis=1)
    return lo, hi


@dataclass(frozen=True)
class FixedCalibrator:
    """Thresholds fitted on a designated calibration set, separate from selection."""

    levels: tuple[float, ...]
    pooled_thresholds: tuple[float, ...]
    horizon_thresholds: dict[int, tuple[float, ...]]
    horizon_counts: dict[int, int]

    def thresholds_for(self, horizons):
        """Return target-by-level thresholds; unseen horizons use the pooled set."""
        horizons = _horizons(horizons)
        return np.asarray(
            [self.horizon_thresholds.get(int(horizon), self.pooled_thresholds) for horizon in horizons],
            dtype=np.float64,
        )


def fit_fixed_calibrator(y, lower, upper, horizons, *, levels=(0.9,), minimum_per_horizon=20):
    """Fit per-horizon CQR thresholds with the study's pooled fallback.

    Supply only the chosen calibration cohort (daylight targets in the study).
    Model fitting and hyperparameter selection must use their separate cohorts.
    One-dimensional endpoints represent one coverage level.
    """
    levels = _levels(levels)
    scores = conformity_scores(y, lower, upper)
    scores = scores[:, None] if scores.ndim == 1 else scores
    if scores.shape[1] != levels.size:
        raise ValueError("One endpoint column is required per coverage level")
    horizons = _horizons(horizons, scores.shape[0])
    minimum = _positive_integer(minimum_per_horizon, "minimum_per_horizon")

    def thresholds(values):
        return tuple(
            conformal_quantile(values[:, column], alpha=1 - level)
            for column, level in enumerate(levels)
        )

    pooled = thresholds(scores)
    by_horizon, counts = {}, {}
    for horizon in np.unique(horizons):
        selected = scores[horizons == horizon]
        counts[int(horizon)] = len(selected)
        by_horizon[int(horizon)] = thresholds(selected) if len(selected) >= minimum else pooled
    return FixedCalibrator(tuple(levels), pooled, by_horizon, counts)


def _timestamps(values, name):
    try:
        timestamps = pd.DatetimeIndex(pd.to_datetime(values, utc=True, format="mixed"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain valid timestamps") from exc
    if timestamps.isna().any():
        raise ValueError(f"{name} cannot contain missing timestamps")
    return timestamps


def _cutoff(issuance, update_at_day_start):
    timestamp = _timestamps([issuance], "issuance")[0]
    return timestamp.floor("D") if update_at_day_start else timestamp


def arrived_label_mask(
    target_end,
    issuance,
    *,
    label_delay_minutes=5,
    window_days=60,
    update_at_day_start=True,
    label_available_at=None,
):
    """Select labels arriving in (update cutoff - window, update cutoff].

    The default updates at the start of the UTC issuance date and assumes a
    calibration-label delay of five minutes after target end. This is distinct
    from when historical observations become forecast inputs. If measured label
    availability timestamps are supplied, they replace the assumed delay.
    """
    ends = _timestamps(target_end, "target_end")
    window_days = float(window_days)
    delay = float(label_delay_minutes)
    if not np.isfinite(window_days) or window_days <= 0:
        raise ValueError("window_days must be finite and positive")
    if not np.isfinite(delay) or delay < 0:
        raise ValueError("label_delay_minutes must be finite and nonnegative")
    arrivals = ends + pd.Timedelta(minutes=delay) if label_available_at is None else _timestamps(label_available_at, "label_available_at")
    if len(arrivals) != len(ends):
        raise ValueError("label_available_at must have one timestamp per target")
    if (arrivals < ends).any():
        raise ValueError("A target label cannot arrive before its target window ends")
    cutoff = _cutoff(issuance, update_at_day_start)
    return np.asarray((arrivals > cutoff - pd.Timedelta(days=window_days)) & (arrivals <= cutoff))


@dataclass(frozen=True)
class RecentCalibration:
    """Thresholds and fallback choices for one chronological update."""

    thresholds: np.ndarray
    sources: tuple[str, ...]
    horizon_counts: tuple[int, ...]
    used_counts: tuple[int, ...]
    cutoff_utc: str


def recent_adjustments(
    scores,
    horizons,
    target_end,
    issuance,
    requested_horizons,
    fixed,
    *,
    minimum_per_horizon=50,
    minimum_pooled=50,
    window_days=60,
    label_delay_minutes=5,
    update_at_day_start=True,
    label_available_at=None,
):
    """Use recent per-horizon scores, recent pooled scores, or fixed thresholds.

    ``scores`` may include multiple stations but must already contain only the
    calibration-eligible target type (daylight in the study). Future-label rows
    may be present; only labels selected by ``arrived_label_mask`` are used.
    Supply the same frozen calibrator throughout a chronological evaluation.
    """
    if not isinstance(fixed, FixedCalibrator):
        raise ValueError("fixed must be a FixedCalibrator")
    scores = _finite_array(scores, "scores")
    scores = scores[:, None] if scores.ndim == 1 else scores
    if scores.ndim != 2 or scores.shape[1] != len(fixed.levels):
        raise ValueError("scores must have one column per fixed coverage level")
    horizons = _horizons(horizons, scores.shape[0])
    requested = _horizons(requested_horizons)
    minimum_per_horizon = _positive_integer(minimum_per_horizon, "minimum_per_horizon")
    minimum_pooled = _positive_integer(minimum_pooled, "minimum_pooled")
    arrived = arrived_label_mask(
        target_end, issuance, window_days=window_days,
        label_delay_minutes=label_delay_minutes, update_at_day_start=update_at_day_start,
        label_available_at=label_available_at,
    )
    if arrived.size != scores.shape[0]:
        raise ValueError("target_end must have one timestamp per score")
    recent_scores, recent_horizons = scores[arrived], horizons[arrived]
    thresholds, sources, counts, used = [], [], [], []
    for horizon in requested:
        chosen = recent_scores[recent_horizons == horizon]
        counts.append(len(chosen))
        if len(chosen) >= minimum_per_horizon:
            source = "recent_horizon"
        elif len(recent_scores) >= minimum_pooled:
            chosen, source = recent_scores, "recent_pooled"
        else:
            chosen, source = None, "fixed"
        if chosen is None:
            value = fixed.thresholds_for([horizon])[0]
        else:
            value = [
                conformal_quantile(chosen[:, column], alpha=1 - level)
                for column, level in enumerate(fixed.levels)
            ]
        thresholds.append(value)
        sources.append(source)
        used.append(0 if chosen is None else len(chosen))
    return RecentCalibration(
        np.asarray(thresholds, dtype=np.float64), tuple(sources), tuple(counts),
        tuple(used), _cutoff(issuance, update_at_day_start).isoformat(),
    )
