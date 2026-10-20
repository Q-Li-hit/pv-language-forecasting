"""Portable array archives for scalar and complete-sequence experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


def equivalent_power(irradiance, temperature, capacity_mw=10.0):
    """Apply the study's common irradiance/ambient-temperature power mapping.

    Irradiance is in W/m² and ambient temperature in °C. This constructs an
    equivalent comparison target rather than a measured AC power estimate.
    Missing meteorological inputs remain missing.
    """
    if not np.isfinite(capacity_mw) or capacity_mw <= 0:
        raise ValueError("Capacity must be finite and positive.")
    irradiance = np.clip(np.asarray(irradiance, float), 0, 1300)
    temperature = np.asarray(temperature, float)
    correction = np.clip(1 - .004 * (temperature - 25), .75, 1.10)
    return np.clip(capacity_mw * irradiance / 1000 * correction, 0, capacity_mw)


def _finite_matrix(array, name, ndim):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != ndim or not all(array.shape) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite, nonempty {ndim}D array.")
    return np.ascontiguousarray(array)


def _metadata(value):
    if not isinstance(value, dict):
        raise ValueError("Archive metadata must be a JSON object.")
    return value


def _mask(value, shape, name):
    value = np.asarray(value)
    if value.dtype.kind != "b" or value.shape != shape:
        raise ValueError(f"{name} must be a boolean array of shape {shape}.")
    return value


def _sample_keys(value, size):
    keys = np.asarray([str(i) for i in range(size)] if value is None else value, dtype=str)
    if keys.shape != (size,) or (keys == "").any() or len(np.unique(keys)) != size:
        raise ValueError("Sample keys must be unique, nonempty strings with one key per row.")
    return keys


@dataclass
class SequenceData:
    history: np.ndarray
    future: np.ndarray
    target: np.ndarray
    train_mask: np.ndarray
    selection_mask: np.ndarray
    evaluation_mask: np.ndarray
    issue_dates: np.ndarray
    station_ids: np.ndarray
    metadata: dict
    ids: np.ndarray | None = None
    attention: np.ndarray | None = None
    sample_keys: np.ndarray | None = None

    def validate(self):
        self.history = _finite_matrix(self.history, "history", 2)
        self.future = _finite_matrix(self.future, "future", 3)
        self.target = _finite_matrix(self.target, "target", 2)
        n, horizons = self.target.shape
        if self.future.shape[:2] != (n, horizons) or len(self.history) != n:
            raise ValueError("History, future inputs and targets have incompatible shapes.")
        masks = []
        for name in ("train_mask", "selection_mask", "evaluation_mask"):
            value = _mask(getattr(self, name), (n, horizons), name)
            setattr(self, name, value)
            masks.append(value)
        if any((a.any(1) & b.any(1)).any() for i, a in enumerate(masks) for b in masks[i+1:]):
            raise ValueError("Training, selection and evaluation must use separate issuances.")
        for name in ("issue_dates", "station_ids"):
            values = np.asarray(getattr(self, name), dtype=str)
            if values.shape != (n,) or (values == "").any():
                raise ValueError(f"{name} must have one nonempty value per issuance.")
            setattr(self, name, values)
        if np.isnat(self.issue_dates.astype("datetime64[D]")).any():
            raise ValueError("Issue dates cannot be missing.")
        if (self.ids is None) != (self.attention is None):
            raise ValueError("Token IDs and attention masks must be supplied together.")
        if self.ids is not None:
            ids, attention = np.asarray(self.ids), np.asarray(self.attention)
            if (ids.ndim != 2 or ids.shape != attention.shape or len(ids) != n
                    or ids.dtype.kind not in "iu" or (ids < 0).any()
                    or not np.isin(attention, [0, 1]).all() or not attention.any(1).all()):
                raise ValueError("Invalid prompt token IDs or attention mask.")
            self.ids = ids.astype(np.int64)
            self.attention = attention.astype(np.int8)
        _metadata(self.metadata)
        self.sample_keys = _sample_keys(self.sample_keys, n)
        return self

    def save(self, path):
        self.validate()
        arrays = {name: getattr(self, name) for name in (
            "history", "future", "target", "train_mask", "selection_mask", "evaluation_mask",
            "issue_dates", "station_ids",
            "sample_keys",
        )}
        if self.ids is not None:
            arrays.update(ids=self.ids, attention=self.attention)
        arrays["metadata"] = np.asarray(json.dumps(self.metadata))
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            fields = {name: archive[name] for name in (
                "history", "future", "target", "train_mask", "selection_mask", "evaluation_mask",
                "issue_dates", "station_ids",
            )}
            fields["metadata"] = json.loads(str(archive["metadata"]))
            fields["ids"] = archive["ids"] if "ids" in archive else None
            fields["attention"] = archive["attention"] if "attention" in archive else None
            fields["sample_keys"] = archive["sample_keys"] if "sample_keys" in archive else None
        return cls(**fields).validate()


@dataclass
class ScalarData:
    history: np.ndarray
    state: np.ndarray
    context: np.ndarray
    target: np.ndarray
    baseline: np.ndarray
    roles: np.ndarray
    issue_dates: np.ndarray
    station_ids: np.ndarray
    metadata: dict
    evaluation_mask: np.ndarray | None = None
    sample_keys: np.ndarray | None = None
    raw_state: np.ndarray | None = None

    def validate(self):
        for name in ("history", "state", "context"):
            setattr(self, name, _finite_matrix(getattr(self, name), name, 2))
        n = len(self.history)
        if self.context.shape != (n, 33) or len(self.state) != n:
            raise ValueError("Matched weather controls require a 33-coordinate context.")
        for name in ("target", "baseline"):
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.shape != (n,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain one finite value per target.")
            setattr(self, name, value)
        for name in ("roles", "issue_dates", "station_ids"):
            value = np.asarray(getattr(self, name), dtype=str)
            if value.shape != (n,) or (value == "").any():
                raise ValueError(f"{name} must have one value per target.")
            setattr(self, name, value)
        if np.isnat(self.issue_dates.astype("datetime64[D]")).any():
            raise ValueError("Issue dates cannot be missing.")
        if not np.isin(self.roles, ["train", "model_select", "fusion_fit", "fusion_select",
                                     "calibration", "test", "unused", "embargo"]).all():
            raise ValueError("Unknown temporal role.")
        _metadata(self.metadata)
        if self.evaluation_mask is None:
            self.evaluation_mask = self.roles == "test"
        self.evaluation_mask = _mask(self.evaluation_mask, (n,), "evaluation_mask")
        if (self.evaluation_mask & (self.roles != "test")).any():
            raise ValueError("Scalar evaluation targets must belong to the test role.")
        self.sample_keys = _sample_keys(self.sample_keys, n)
        if self.raw_state is not None:
            self.raw_state = _finite_matrix(self.raw_state, "raw_state", 2)
            if self.raw_state.shape != self.state.shape:
                raise ValueError("Raw and standardized state matrices must have the same shape.")
        return self

    def save(self, path):
        self.validate()
        arrays = {name: getattr(self, name) for name in (
            "history", "state", "context", "target", "baseline", "roles", "issue_dates", "station_ids",
            "evaluation_mask", "sample_keys",
        )}
        arrays["metadata"] = np.asarray(json.dumps(self.metadata))
        if self.raw_state is not None:
            arrays["raw_state"] = self.raw_state
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            fields = {name: archive[name] for name in (
                "history", "state", "context", "target", "baseline", "roles", "issue_dates", "station_ids",
            )}
            fields["metadata"] = json.loads(str(archive["metadata"]))
            fields["evaluation_mask"] = archive["evaluation_mask"] if "evaluation_mask" in archive else None
            fields["sample_keys"] = archive["sample_keys"] if "sample_keys" in archive else None
            fields["raw_state"] = archive["raw_state"] if "raw_state" in archive else None
        return cls(**fields).validate()


def prompt_statistics(history):
    """Nine history statistics used in the matched-information tree baseline."""
    x = _finite_matrix(history, "history", 2)
    if x.shape[1] < 5:
        raise ValueError("At least five history points are required.")
    centered = x - x.mean(axis=1, keepdims=True)
    correlation = np.fft.irfft(abs(np.fft.rfft(centered, axis=1))**2, n=x.shape[1], axis=1)
    lags = np.argsort(correlation, axis=1)[:, -5:][:, ::-1]
    rounded = lambda values: np.asarray([float(f"{v:.1f}") for v in values], np.float32)
    return np.column_stack([
        rounded(x.min(1)), rounded(x.max(1)), rounded(np.median(x, axis=1)),
        (x[:, -1] > x[:, 0]).astype(np.float32), lags,
    ]).astype(np.float32)


def tree_features(data, groups, steps):
    stats = prompt_statistics(data.history)
    return np.ascontiguousarray(np.column_stack([
        data.history[groups], data.future[groups, steps], stats[groups],
    ]), dtype=np.float32)
