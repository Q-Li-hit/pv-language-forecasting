"""Weather-state representations and scalar residual forecasting models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn


STATE_FIELDS = (
    "cloud_low", "cloud_mid", "cloud_high", "cloud_total",
    "nwp_csi", "recent_ramp", "observed_residual",
)
CATEGORY_SIZES = (5, 5, 5, 5, 5, 4, 4)
CONTEXT_LABELS = (
    ("clear", "broken", "cloudy", "overcast", "unknown"),
    ("clear", "broken", "cloudy", "overcast", "unknown"),
    ("clear", "broken", "cloudy", "overcast", "unknown"),
    ("clear", "broken", "cloudy", "overcast", "unknown"),
    ("low", "medium", "high", "extreme", "unknown"),
    ("decreasing", "stable", "increasing", "unknown"),
    ("negative", "near zero", "positive", "unknown"),
)


def context_fields(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Discretize the seven existing weather and power-state variables."""
    bounds = ([20, 60, 90],) * 4 + ([0.35, 0.7, 1.2], [-0.5, 0.5], [-0.5, 0.5])
    columns = []
    for name, limits in zip(STATE_FIELDS, bounds):
        values = frame[name].to_numpy(float)
        codes = np.digitize(values, limits)
        codes[~np.isfinite(values)] = len(limits) + 1
        columns.append(codes)
    return np.stack(columns, axis=1).astype(np.int8), list(STATE_FIELDS)


def _validate_codes(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes)
    if codes.ndim != 2 or codes.shape[1] != len(STATE_FIELDS):
        raise ValueError("State codes must have shape (samples, 7).")
    if not np.isfinite(codes).all() or not np.equal(codes, np.floor(codes)).all():
        raise ValueError("State codes must be finite integers.")
    if ((codes < 0) | (codes >= np.asarray(CATEGORY_SIZES))).any():
        raise ValueError("State code is outside its field's category range.")
    return codes.astype(np.int8)


def text_from_codes(codes: np.ndarray) -> list[str]:
    """Apply the fixed text template without adding weather observations."""
    result = []
    for row in _validate_codes(codes):
        values = [CONTEXT_LABELS[i][int(code)] for i, code in enumerate(row)]
        result.append(
            f"Photovoltaic weather state: low cloud {values[0]}; middle cloud {values[1]}; "
            f"high cloud {values[2]}; total cloud {values[3]}; forecast clear sky ratio "
            f"{values[4]}; recent power trend {values[5]}; latest available forecast "
            f"residual {values[6]}."
        )
    return result


def context_codes_key(codes: np.ndarray) -> np.ndarray:
    return np.asarray([",".join(map(str, row)) for row in _validate_codes(codes)], dtype=str)


def category_context(codes: np.ndarray) -> np.ndarray:
    """Return the 33-coordinate one-hot interface used by matched controls."""
    codes = _validate_codes(codes)
    return np.concatenate(
        [np.eye(size, dtype=np.float32)[codes[:, i]] for i, size in enumerate(CATEGORY_SIZES)],
        axis=1,
    )


def numeric_projection(seed: int = 20261008) -> np.ndarray:
    """Create the fixed, redundant seven-to-33 numerical expansion."""
    return np.random.default_rng(seed).normal(size=(7, 33)).astype(np.float32) / np.sqrt(7)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    names = [
        name for name in frame.columns
        if name.startswith(("hist_", "phase_", "nwp_", "cloud_")) and name != "nwp_target_hour"
    ]
    names += [
        "forecast_age_h", "forecast_lead_h", "forecast_elapsed_h", "horizon", "station_code",
        "sin_hour", "cos_hour", "sin_doy", "cos_doy", "csky_power", "persistence_mw",
        "recent_ramp", "observed_residual",
    ]
    return list(dict.fromkeys(names))


def raw_features(frame: pd.DataFrame, names: Sequence[str]) -> np.ndarray:
    numeric = frame[list(names)].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    codes, _ = context_fields(frame)
    return np.column_stack([numeric, codes.astype(np.float32)])


@dataclass
class WeatherPreprocessor:
    """Training-fitted imputation and scaling for the common numerical branch."""

    features: tuple[str, ...]
    lag_names: tuple[str, ...]
    fill: np.ndarray
    scaler: StandardScaler
    capacity_mw: float = 10.0

    @classmethod
    def fit(cls, training: pd.DataFrame, features: Sequence[str] | None = None,
            capacity_mw: float = 10.0) -> "WeatherPreprocessor":
        if len(training) == 0 or not np.isfinite(capacity_mw) or capacity_mw <= 0:
            raise ValueError("Nonempty training data and positive capacity are required.")
        names = tuple(feature_columns(training) if features is None else features)
        lags = tuple(name for name in names if name.startswith("hist_"))
        if not lags:
            raise ValueError("At least one history feature is required.")
        x = raw_features(training, names)
        x[~np.isfinite(x)] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fill = np.nanmedian(x, axis=0)
        fill = np.nan_to_num(fill)
        x = np.where(np.isfinite(x), x, fill)
        return cls(names, lags, fill, StandardScaler().fit(x), capacity_mw)

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = raw_features(frame, self.features)
        state = self.scaler.transform(np.where(np.isfinite(x), x, self.fill)).astype(np.float32)
        history = frame[list(self.lag_names)].to_numpy(np.float32) / self.capacity_mw
        history = np.nan_to_num(history)
        return np.ascontiguousarray(history), np.ascontiguousarray(state)

    def numeric_context(self, state: np.ndarray, projection: np.ndarray | None = None) -> np.ndarray:
        indices = [self.features.index(name) for name in STATE_FIELDS]
        matrix = numeric_projection() if projection is None else np.asarray(projection, np.float32)
        if matrix.shape != (7, 33):
            raise ValueError("Numerical context projection must have shape (7, 33).")
        return np.ascontiguousarray(state[:, indices] @ matrix, dtype=np.float32)


@dataclass
class WeightedPCA:
    """Context PCA fitted using training-state frequencies only."""

    mean: np.ndarray
    components: np.ndarray

    @classmethod
    def fit(cls, embeddings: np.ndarray, keys: np.ndarray, training_codes: np.ndarray,
            n_components: int = 32) -> "WeightedPCA":
        bank = np.asarray(embeddings, np.float64)
        keys = np.asarray(keys, dtype=str)
        if bank.ndim != 2 or bank.shape[0] != len(keys) or not np.isfinite(bank).all():
            raise ValueError("Embedding rows must match finite context-bank keys.")
        if keys.ndim != 1 or len(keys) == 0 or not np.array_equal(keys, np.unique(keys)):
            raise ValueError("Context-bank keys must be unique and sorted.")
        if not 0 < n_components <= bank.shape[1]:
            raise ValueError("Invalid PCA component count.")
        requested = context_codes_key(training_codes)
        if len(requested) == 0:
            raise ValueError("PCA fitting requires training states.")
        indices = np.searchsorted(keys, requested)
        if (indices >= len(keys)).any() or not np.array_equal(keys[np.minimum(indices, len(keys)-1)], requested):
            raise ValueError("A training state is absent from the context bank.")
        weights = np.bincount(indices, minlength=len(keys)).astype(float) / len(indices)
        mean = np.sum(bank * weights[:, None], axis=0)
        weighted = (bank - mean) * np.sqrt(weights)[:, None]
        _, vectors = np.linalg.eigh(weighted.T @ weighted)
        components = vectors[:, -n_components:][:, ::-1].T
        pivots = components[np.arange(n_components), np.argmax(abs(components), axis=1)]
        components *= np.where(pivots >= 0, 1.0, -1.0)[:, None]
        return cls(mean, components)

    def transform(self, embeddings: np.ndarray, add_constant: bool = False) -> np.ndarray:
        result = ((np.asarray(embeddings) - self.mean) @ self.components.T).astype(np.float32)
        if add_constant:
            result = np.column_stack([result, np.ones(len(result), np.float32)])
        return np.ascontiguousarray(result)


def frozen_embeddings(texts: Sequence[str], model_path: str, random_seed: int | None = None,
                      batch_size: int = 8, max_length: int = 128,
                      device: str = "cpu") -> np.ndarray:
    """Encode templates with a local pretrained model or one fixed random instance.

    The tokenizer must use right padding, matching the last-valid-token pooling
    used by the weather experiments. Model weights are never downloaded here.
    """
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    if not texts or batch_size < 1 or max_length < 1:
        raise ValueError("Nonempty texts and positive batch/token limits are required.")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.padding_side != "right":
        raise ValueError("Weather encoding requires right-padded tokenizer inputs.")
    if random_seed is None:
        model = AutoModel.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float32)
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(random_seed)
            model = AutoModel.from_config(AutoConfig.from_pretrained(model_path, local_files_only=True))
    model = model.float().to(device).eval()
    model.requires_grad_(False)
    results = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            inputs = tokenizer(list(texts[start:start + batch_size]), padding=True, truncation=True,
                               max_length=max_length, return_tensors="pt").to(device)
            hidden = model(**inputs).last_hidden_state
            last = inputs["attention_mask"].sum(1) - 1
            pooled = hidden[torch.arange(len(last), device=last.device), last]
            results.append(nn.functional.normalize(pooled, dim=1).cpu().numpy().astype(np.float32))
    return np.concatenate(results)


@dataclass(frozen=True)
class TransformerConfig:
    d_model: int = 64
    heads: int = 4
    layers: int = 2
    dropout: float = 0.08

    def __post_init__(self) -> None:
        if self.d_model < 1 or self.heads < 1 or self.layers < 1 or self.d_model % self.heads:
            raise ValueError("Positive model dimensions and divisible attention heads are required.")
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must lie in [0, 1).")


class ResidualTransformer(nn.Module):
    """Common history, numerical-state and context-token prediction structure."""

    def __init__(self, lag_count: int, features: int, variant: str,
                 config: TransformerConfig | None = None):
        super().__init__()
        cfg = TransformerConfig() if config is None else config
        d = cfg.d_model
        self.value = nn.Linear(1, d)
        self.state = nn.Sequential(nn.Linear(features, d), nn.GELU(), nn.LayerNorm(d))
        self.context = nn.Sequential(nn.Linear(32, d), nn.GELU(), nn.LayerNorm(d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.position = nn.Parameter(torch.zeros(1, lag_count + 3, d))
        self.kind = nn.Parameter(torch.zeros(1, 3, d))
        layer = nn.TransformerEncoderLayer(d, cfg.heads, 4 * d, cfg.dropout,
                                          batch_first=True, norm_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                  nn.Dropout(cfg.dropout), nn.Linear(d, 1))
        self.variant = variant
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.kind, std=0.02)

    def forward(self, seq: torch.Tensor, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        history = self.value(seq.unsqueeze(-1)) + self.kind[:, 0:1]
        state = self.state(x).unsqueeze(1) + self.kind[:, 1:2]
        ctx = self.context(context).unsqueeze(1) + self.kind[:, 2:3]
        tokens = torch.cat([self.cls.expand(len(seq), -1, -1), history, state, ctx], dim=1)
        padding = torch.zeros((len(seq), tokens.shape[1]), dtype=torch.bool, device=tokens.device)
        if self.variant in ("numeric", "no_history", "no_physics", "direct_target"):
            padding[:, -1] = True
        if self.variant == "no_history":
            padding[:, 1:1 + seq.shape[1]] = True
        output = self.transformer(tokens + self.position, src_key_padding_mask=padding)
        return self.head(output[:, 0]).squeeze(-1)


class CapacityMatchedControl(ResidualTransformer):
    """Identical 33-to-32 trainable interface for numerical, category and text inputs."""

    def __init__(self, lag_count: int, features: int, config: TransformerConfig | None = None):
        super().__init__(lag_count, features, "capacity_matched", config)
        self.adapter = nn.Linear(33, 32, bias=False)

    def forward(self, sequence: torch.Tensor, state: torch.Tensor,
                context: torch.Tensor) -> torch.Tensor:
        return super().forward(sequence, state, self.adapter(context))
