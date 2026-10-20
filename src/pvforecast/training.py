"""Training and prediction with explicit temporal masks and selection rules."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .data import ScalarData, SequenceData
from .utils import file_sha256, write_json


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = .0004
    weight_decay: float = .0001
    batch_size: int = 128
    microbatch: int = 128
    minimum_epochs: int = 8
    maximum_epochs: int = 240
    patience: int = 5
    relative_delta: float = .0001
    absolute_delta: float = .000001
    gradient_clip: float = 1.0
    beta_mw: float = .35
    precision: str = "float32"
    threads: int = 4

    def __post_init__(self):
        if (not np.isfinite(self.learning_rate) or self.learning_rate <= 0
                or self.weight_decay < 0 or not np.isfinite(self.weight_decay)):
            raise ValueError("Learning rate must be positive and weight decay nonnegative.")
        for name in ("batch_size", "microbatch", "minimum_epochs", "maximum_epochs", "patience", "threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.maximum_epochs < self.minimum_epochs:
            raise ValueError("Maximum epochs must be at least minimum epochs.")
        if (not all(np.isfinite(v) for v in (self.relative_delta, self.absolute_delta,
                                            self.gradient_clip, self.beta_mw))
                or min(self.relative_delta, self.absolute_delta) < 0 or self.gradient_clip <= 0
                or self.beta_mw <= 0):
            raise ValueError("Invalid early-stopping or loss configuration.")
        if self.precision not in ("float32", "bfloat16"):
            raise ValueError("Precision must be float32 or bfloat16.")


def configure_runtime(device, config):
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    if config.precision == "bfloat16" and device.type != "cuda":
        raise ValueError("This runner uses bfloat16 autocast only on CUDA.")
    torch.set_num_threads(config.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    return device


def _execution(device, precision):
    return torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bfloat16" else nullcontext()


def _attention():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel(SDPBackend.MATH)


def _inputs(data, indices, device):
    tensor = lambda array: torch.as_tensor(array[indices], device=device)
    if isinstance(data, ScalarData):
        return tensor(data.history), tensor(data.state), tensor(data.context)
    if data.ids is None:
        return tensor(data.history), tensor(data.future)
    mask = data.attention[indices]
    end = np.flatnonzero(mask.any(0))[-1] + 1
    return (tensor(data.history), tensor(data.future),
            torch.as_tensor(data.ids[indices, :end], device=device),
            torch.as_tensor(mask[:, :end], device=device))


def _loss_sum(output, target, mask, data, config):
    if isinstance(data, ScalarData):
        return nn.functional.mse_loss(output.float(), target, reduction="sum")
    values = nn.functional.smooth_l1_loss(output.float(), target, reduction="none", beta=config.beta_mw)
    return values[mask].sum()


def fit(model, data, output, *, config=None, seed=42, device="cpu", model_spec=None):
    """Fit one initialized model; choose its checkpoint using selection labels.

    Sequence losses are averaged over valid scalar targets in each effective
    batch. Scalar controls predict residual power and use squared error. The
    lowest absolute selection loss is saved; patience uses a separate reference
    and the configured minimum improvement.
    """
    config = TrainingConfig() if config is None else config
    data.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    device = configure_runtime(device, config)
    model.to(device)
    if isinstance(data, ScalarData):
        train_mask = data.roles == "train"
        selection_mask = data.roles == "model_select"
        target = data.target - data.baseline
    else:
        train_mask, selection_mask = data.train_mask, data.selection_mask
        target = data.target
    rows = lambda mask: np.flatnonzero(mask if mask.ndim == 1 else mask.any(1))
    train_rows, selection_rows = rows(train_mask), rows(selection_mask)
    if len(train_rows) == 0 or len(selection_rows) == 0:
        raise ValueError("Both training and selection targets are required.")
    if np.intersect1d(train_rows, selection_rows).size:
        raise ValueError("Training and selection cannot share issuances.")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    best = reference = float("inf")
    stale = best_epoch = 0
    history = []
    started = time.perf_counter()
    stop_reason = "maximum_epoch_budget"
    for epoch in range(1, config.maximum_epochs + 1):
        model.train()
        order = np.random.default_rng(seed + epoch - 1).permutation(train_rows)
        training_sum = 0.
        training_count = 0
        with _attention():
            for first in range(0, len(order), config.batch_size):
                batch = order[first:first + config.batch_size]
                count = int(train_mask[batch].sum())
                optimizer.zero_grad(set_to_none=True)
                for micro in range(0, len(batch), config.microbatch):
                    indices = batch[micro:micro + config.microbatch]
                    with _execution(device, config.precision):
                        prediction = model(*_inputs(data, indices, device))
                    raw = _loss_sum(prediction, torch.as_tensor(target[indices], device=device),
                                    torch.as_tensor(train_mask[indices], device=device), data, config)
                    (raw / count).backward()
                    training_sum += float(raw.detach())
                nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
                optimizer.step()
                training_count += count
        model.eval()
        selection_sum = selection_count = 0
        with torch.inference_mode(), _attention():
            for first in range(0, len(selection_rows), config.microbatch):
                indices = selection_rows[first:first + config.microbatch]
                with _execution(device, config.precision):
                    prediction = model(*_inputs(data, indices, device))
                selection_sum += float(_loss_sum(
                    prediction, torch.as_tensor(target[indices], device=device),
                    torch.as_tensor(selection_mask[indices], device=device), data, config,
                ))
                selection_count += int(selection_mask[indices].sum())
        score = selection_sum / selection_count
        if not np.isfinite(score) or not np.isfinite(training_sum):
            raise FloatingPointError("Nonfinite training or selection loss.")
        history.append({"epoch": epoch, "training_loss": training_sum / training_count,
                        "selection_loss": score, "training_targets": training_count,
                        "selection_targets": selection_count})
        if score < best:
            best, best_epoch = score, epoch
            state = {name: parameter.detach().cpu() for name, parameter in model.named_parameters()
                     if parameter.requires_grad}
            torch.save(state, output / "parameters.pt")
        threshold = max(config.absolute_delta, config.relative_delta * reference)
        if not np.isfinite(reference) or reference - score > threshold:
            reference, stale = score, 0
        else:
            stale += 1
        print(f"epoch {epoch}: train={history[-1]['training_loss']:.6f}, selection={score:.6f}", flush=True)
        if epoch >= config.minimum_epochs and stale >= config.patience:
            stop_reason = "predefined_patience"
            break
    report = {
        "seed": seed, "model": model_spec or {}, "training": asdict(config),
        "best_epoch": best_epoch, "best_selection_loss": best,
        "stop_reason": stop_reason, "history": history,
        "seconds": time.perf_counter() - started,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_sha256": file_sha256(output / "parameters.pt"),
        "input_metadata": data.metadata,
        "environment": {"torch": torch.__version__, "numpy": np.__version__, "device": str(device)},
    }
    write_json(output / "fit.json", report)
    load_task_parameters(model, output / "parameters.pt")
    return report


def load_task_parameters(model, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    if set(state) != expected:
        raise ValueError("Checkpoint keys differ from this model's task parameters.")
    model.load_state_dict(state, strict=False)


def predict(model, data, *, device="cpu", batch_size=128, precision="float32", capacity_mw=10.):
    if batch_size < 1 or not np.isfinite(capacity_mw) or capacity_mw <= 0:
        raise ValueError("Positive batch size and capacity are required.")
    data.validate()
    device = configure_runtime(device, TrainingConfig(precision=precision))
    model.to(device).eval()
    result = []
    with torch.inference_mode(), _attention():
        for first in range(0, len(data.history), batch_size):
            indices = np.arange(first, min(first + batch_size, len(data.history)))
            with _execution(device, precision):
                prediction = model(*_inputs(data, indices, device)).float().cpu().numpy()
            if isinstance(data, ScalarData):
                prediction += data.baseline[indices]
            result.append(np.clip(prediction, 0, capacity_mw))
    return np.concatenate(result)
