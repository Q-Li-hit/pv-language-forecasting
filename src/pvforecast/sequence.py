"""Sequence forecasting with numerical, patch, and frozen language models.

The history branch predicts a complete trajectory. A separate numerical branch
adds the contribution of the available features for each target window.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


DOMAIN_PREFIX = (
    "Context: equivalent photovoltaic power in MW. Completed power history describes recent generation. "
    "A separate numerical branch supplies target-weather forecasts and available historical, time, solar and station fields. "
    "Daytime generation depends on solar irradiance; nighttime generation is usually small. "
    "The history branch models temporal power variation and the numerical branch accounts for available conditions. "
)


@dataclass(frozen=True)
class SequenceConfig:
    """Architecture settings shared by the three sequence models."""

    d_model: int = 64
    heads: int = 4
    layers: int = 2
    dropout: float = 0.08
    patch_length: int = 8
    patch_stride: int = 4
    time_llm_prototypes: int = 128
    full_backbone_layers: int = 28
    prompt_max_length: int = 128
    domain_prompt_max_length: int = 192

    def __post_init__(self) -> None:
        dimensions = (
            self.d_model, self.heads, self.layers, self.patch_length,
            self.patch_stride, self.time_llm_prototypes,
            self.full_backbone_layers, self.prompt_max_length,
            self.domain_prompt_max_length,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("Sequence dimensions must be positive.")
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")


def build_statistical_prompts(
    sequence: np.ndarray | Sequence[Sequence[float]],
    task: str,
    *,
    domain: bool = False,
) -> list[str]:
    """Create the original statistical prompt, optionally with domain context.

    Power values are in MW. Autocorrelation lags retain the original circular
    FFT calculation and include the zero lag.
    """
    if task not in {"day_ahead", "rolling"}:
        raise ValueError("task must be 'day_ahead' or 'rolling'.")
    values = np.asarray(sequence)
    if values.ndim != 2 or values.shape[1] < 5 or not np.isfinite(values).all():
        raise ValueError("sequence must be a finite two-dimensional array with at least five history points.")
    windows = "24 target UTC hourly windows" if task == "day_ahead" else "12 future five-minute windows"
    prompts = []
    for row in values:
        centered = row - row.mean()
        correlation = np.fft.irfft(abs(np.fft.rfft(centered)) ** 2, n=len(row))
        lags = np.argsort(correlation)[-5:][::-1].tolist()
        text = (
            f"Dataset: equivalent photovoltaic power. Forecast {windows} from completed history. "
            f"Input statistics: min {row.min():.1f}, max {row.max():.1f}, median {np.median(row):.1f} MW; "
            f'trend {"up" if row[-1] > row[0] else "down"}; top five autocorrelation lags {lags}.'
        )
        prompts.append((DOMAIN_PREFIX if domain else "") + text)
    return prompts


def tokenize_prompts(
    sequence: np.ndarray,
    task: str,
    tokenizer,
    *,
    config: SequenceConfig | None = None,
    domain: bool = False,
    max_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize statistical prompts without discarding their input statistics."""
    cfg = config or SequenceConfig()
    length = max_length if max_length is not None else (
        cfg.domain_prompt_max_length if domain else cfg.prompt_max_length
    )
    if length <= 0:
        raise ValueError("max_length must be positive.")
    prompts = build_statistical_prompts(sequence, task, domain=domain)
    if any(len(tokenizer.encode(text)) > length for text in prompts):
        raise ValueError("The prompt exceeds max_length; increase the budget rather than truncate statistics.")
    encoded = tokenizer(
        prompts, padding="max_length", truncation=False,
        max_length=length, return_tensors="np",
    )
    return (
        np.asarray(encoded["input_ids"], dtype=np.int32),
        np.asarray(encoded["attention_mask"], dtype=np.int8),
    )


class ForecastModel(nn.Module):
    """Forecast all target windows from history and per-window features.

    ``variant`` is ``dense_numeric``, ``patchtst``, or ``time_llm_qwen``.
    ``backbone`` can supply an already loaded language model; otherwise the
    language variant requires ``model_path``. Numerical models do not import
    Transformers or access model files.

    The language model is frozen, while gradients pass through its operations
    to the task modules. Random controls replace its parameters after task
    initialization, inside an isolated random-number state.
    """

    def __init__(
        self,
        variant: str,
        history_length: int,
        features: int,
        horizons: int,
        *,
        config: SequenceConfig | None = None,
        model_path: str | None = None,
        revision: str | None = None,
        local_files_only: bool = True,
        backbone: nn.Module | None = None,
        random_backbone: bool = False,
        head_seed: int = 42,
        model_dtype: torch.dtype | None = None,
        random_backbone_factory: Callable | None = None,
    ) -> None:
        super().__init__()
        self.config = config or SequenceConfig()
        self.variant = variant
        self.history_length = history_length
        self.horizons = horizons
        self.features = features
        self._projection_cache = None
        if min(history_length, features, horizons) <= 0:
            raise ValueError("History length, feature count, and horizon count must be positive.")
        if variant not in {"dense_numeric", "patchtst", "time_llm_qwen"}:
            raise ValueError(f"Unknown sequence model: {variant}")
        if variant != "time_llm_qwen" and (backbone is not None or random_backbone):
            raise ValueError("Language-model parameters apply only to time_llm_qwen.")
        cfg = self.config
        d = cfg.d_model
        self.future = nn.Sequential(
            nn.Linear(features, d), nn.GELU(), nn.LayerNorm(d), nn.Linear(d, 1),
        )
        if variant == "dense_numeric":
            self.embed = nn.Linear(1, d)
            self.position = nn.Parameter(torch.zeros(1, history_length, d))
            self.encoder = self._encoder()
            self.head = nn.Linear(history_length * d, horizons)
        else:
            if history_length < cfg.patch_length:
                raise ValueError("history_length must be at least patch_length.")
            self.patch_length = cfg.patch_length
            self.patch_stride = cfg.patch_stride
            self.patches = (history_length - cfg.patch_length) // cfg.patch_stride + 2
            self.patch_embed = nn.Linear(cfg.patch_length, d, bias=False)
            if variant == "patchtst":
                self.position = nn.Parameter(torch.zeros(1, self.patches, d))
                self.encoder = self._encoder()
                self.head = nn.Linear(self.patches * d, horizons)
            else:
                if backbone is None:
                    if model_path is None:
                        raise ValueError("time_llm_qwen requires a model_path or an injected backbone.")
                    from transformers import AutoModel

                    dtype = model_dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
                    kwargs = dict(local_files_only=local_files_only, torch_dtype=dtype, attn_implementation="sdpa")
                    if revision is not None:
                        kwargs["revision"] = revision
                    backbone = AutoModel.from_pretrained(str(model_path), **kwargs)
                self.backbone = backbone
                self._freeze_language_model()
                vocab, llm_dim = self.backbone.get_input_embeddings().weight.shape
                if llm_dim < d:
                    raise ValueError("The language model hidden size must be at least d_model.")
                self.source_mapping = nn.Linear(vocab, cfg.time_llm_prototypes)
                self.query = nn.Linear(d, d)
                self.key = nn.Linear(llm_dim, d)
                self.value = nn.Linear(llm_dim, d)
                self.reprogram_output = nn.Linear(d, llm_dim)
                self.head = nn.Linear(self.patches * d, horizons)
                self.attn_dropout = nn.Dropout(cfg.dropout)
                if random_backbone:
                    self._replace_random_language_model(head_seed, model_dtype, random_backbone_factory)
        self.dropout = nn.Dropout(cfg.dropout)
        if hasattr(self, "position"):
            nn.init.trunc_normal_(self.position, std=0.02)

    def _encoder(self) -> nn.TransformerEncoder:
        cfg = self.config
        layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.heads, 4 * cfg.d_model, cfg.dropout,
            batch_first=True, norm_first=True, activation="gelu",
        )
        return nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)

    def _freeze_language_model(self) -> None:
        actual = self.backbone.config.num_hidden_layers
        if actual != self.config.full_backbone_layers:
            raise ValueError(f"Expected {self.config.full_backbone_layers} language layers, found {actual}.")
        self.backbone.config.use_cache = False
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    def _replace_random_language_model(self, head_seed, dtype, factory) -> None:
        configuration = copy.deepcopy(self.backbone.config)
        original_dtype = self.backbone.get_input_embeddings().weight.dtype
        original_device = self.backbone.get_input_embeddings().weight.device
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(20261100 + head_seed)
            if factory is None:
                from transformers import AutoModel

                replacement = AutoModel.from_config(
                    configuration, torch_dtype=dtype or original_dtype,
                    attn_implementation="sdpa",
                )
            else:
                replacement = factory(configuration)
        self.backbone = replacement.to(device=original_device, dtype=dtype or original_dtype)
        self._freeze_language_model()

    def train(self, mode: bool = True):
        self.clear_inference_cache()
        super().train(mode)
        if hasattr(self, "backbone"):
            self.backbone.eval()
        return self

    def _apply(self, function, recurse=True):
        self.clear_inference_cache()
        return super()._apply(function, recurse=recurse)

    def load_state_dict(self, *args, **kwargs):
        self.clear_inference_cache()
        return super().load_state_dict(*args, **kwargs)

    def clear_inference_cache(self) -> None:
        """Discard fixed projections after a state, device, or mode change."""
        self._projection_cache = None

    def _cache_signature(self):
        device = self.backbone.get_input_embeddings().weight.device
        enabled = torch.is_autocast_enabled(device.type)
        autocast_dtype = torch.get_autocast_dtype(device.type) if enabled else None
        parameters = [self.backbone.get_input_embeddings().weight]
        for module in (self.source_mapping, self.key, self.value):
            parameters.extend(module.parameters())
        return (
            device, enabled, autocast_dtype,
            tuple((id(value), value.dtype, value._version) for value in parameters),
        )

    def _project_prototypes(self):
        words = self.backbone.get_input_embeddings().weight
        prototypes = self.source_mapping(words.T).T
        return self.key(prototypes), self.value(prototypes)

    @contextmanager
    def inference_cache(self) -> Iterator["ForecastModel"]:
        """Reuse fixed vocabulary, key, and value projections during inference.

        Enter this context inside the same autocast context used for prediction.
        It requires evaluation mode and gradient-free forward calls. Changing
        parameters invalidates the cache; leaving the context releases it.
        """
        if self.variant != "time_llm_qwen":
            raise ValueError("Projection caching applies only to time_llm_qwen.")
        if self.training:
            raise RuntimeError("Call eval() before building an inference cache.")
        if self._projection_cache is not None:
            raise RuntimeError("Inference cache contexts cannot be nested.")
        try:
            with torch.inference_mode():
                key, value = self._project_prototypes()
            self._projection_cache = (key, value, self._cache_signature())
            yield self
        finally:
            self.clear_inference_cache()

    def forward(self, sequence, future, ids=None, prompt_mask=None):
        if sequence.ndim != 2 or sequence.shape[1] != self.history_length:
            raise ValueError("sequence must have shape (batch, history_length).")
        if future.shape != (sequence.shape[0], self.horizons, self.features):
            raise ValueError("future must have shape (batch, horizons, features).")
        cfg = self.config
        mean = sequence.mean(dim=1, keepdim=True)
        std = torch.sqrt(sequence.var(dim=1, keepdim=True, unbiased=False) + 1e-4)
        x = (sequence - mean) / std
        if self.variant == "dense_numeric":
            tokens = self.embed(x.unsqueeze(-1)) + self.position
            hidden = self.encoder(tokens)
        else:
            x = nn.functional.pad(x, (0, self.patch_stride), mode="replicate")
            tokens = self.patch_embed(x.unfold(-1, self.patch_length, self.patch_stride))
            if self.variant == "patchtst":
                hidden = self.encoder(tokens + self.position)
            else:
                if ids is None or prompt_mask is None or ids.shape != prompt_mask.shape or ids.shape[0] != sequence.shape[0]:
                    raise ValueError("Language forecasts require aligned prompt ids and attention masks.")
                if self._projection_cache is not None:
                    if torch.is_grad_enabled() or self.training:
                        raise RuntimeError("Cached projections require gradient-free evaluation.")
                    k, v, signature = self._projection_cache
                    if signature != self._cache_signature():
                        self.clear_inference_cache()
                        raise RuntimeError("Model state or autocast settings changed; rebuild the inference cache.")
                else:
                    k, v = self._project_prototypes()
                batch, patches, _ = tokens.shape
                width = cfg.d_model // cfg.heads
                q = self.query(tokens).reshape(batch, patches, cfg.heads, width)
                k = k.reshape(-1, cfg.heads, width)
                v = v.reshape(-1, cfg.heads, width)
                scores = torch.einsum("blhe,she->bhls", q, k) / np.sqrt(width)
                attention = self.attn_dropout(torch.softmax(scores, dim=-1))
                combined = torch.einsum("bhls,she->blhe", attention, v).reshape(batch, patches, -1)
                patches_in = self.reprogram_output(combined)
                prefix = self.backbone.get_input_embeddings()(ids)
                mask = torch.cat([
                    prompt_mask,
                    torch.ones((batch, patches), device=prompt_mask.device, dtype=prompt_mask.dtype),
                ], dim=1)
                positions = (mask.cumsum(-1) - 1).clamp(min=0)
                embeddings = torch.cat([prefix, patches_in.to(prefix.dtype)], dim=1)
                output = self.backbone(
                    inputs_embeds=embeddings, attention_mask=mask,
                    position_ids=positions, use_cache=False, return_dict=True,
                )
                hidden = output.last_hidden_state[:, -patches:, :cfg.d_model].float()
        normalized = self.head(self.dropout(hidden.flatten(1)))
        return normalized * std + mean + self.future(future).squeeze(-1)

    def trainable_state(self) -> dict[str, Tensor]:
        """Return task parameters without copying frozen language weights."""
        return {
            name: value.detach().cpu().clone()
            for name, value in self.named_parameters() if value.requires_grad
        }

    def load_trainable_state(self, state: Mapping[str, Tensor]) -> None:
        """Load a task checkpoint and reject missing or unexpected task keys."""
        expected = {name for name, value in self.named_parameters() if value.requires_grad}
        if set(state) != expected:
            raise ValueError("Checkpoint task parameter names do not match the model.")
        self.load_state_dict(state, strict=False)
