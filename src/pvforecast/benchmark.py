"""Resident point-forecast timing for an already loaded model and fixed inputs."""

from contextlib import nullcontext
import time

import numpy as np
import torch


def benchmark_inference(model, inputs, *, warmup=10, repetitions=200, cache=False):
    """Measure raw forward calls, excluding data preparation and model loading.

    Inputs are already on the intended device. The caller controls precision and
    attention execution. Cache construction and the first forward are reported
    separately from warmed repeated calls. This function does not train a model.
    """
    if warmup < 0 or repetitions < 2:
        raise ValueError("Nonnegative warmup and at least two repetitions are required.")
    device = inputs[0].device
    if any(value.device != device for value in inputs):
        raise ValueError("All inputs must occupy the same device.")
    synchronize = lambda: torch.cuda.synchronize(device) if device.type == "cuda" else None
    model.eval()
    with torch.inference_mode():
        synchronize()
        before = time.perf_counter()
        context = model.inference_cache() if cache else nullcontext()
        with context:
            synchronize()
            setup_seconds = time.perf_counter() - before
            synchronize()
            before = time.perf_counter()
            output = model(*inputs)
            synchronize()
            first_ms = (time.perf_counter() - before) * 1000
            for _ in range(warmup):
                model(*inputs)
            synchronize()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            elapsed = []
            for _ in range(repetitions):
                synchronize()
                before = time.perf_counter()
                model(*inputs)
                synchronize()
                elapsed.append((time.perf_counter() - before) * 1000)
            memory = {
                "allocated_bytes": torch.cuda.memory_allocated(device),
                "reserved_bytes": torch.cuda.memory_reserved(device),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            } if device.type == "cuda" else {}
    return {"cache_setup_seconds": setup_seconds, "first_request_ms": first_ms,
            "P50_ms": float(np.percentile(elapsed, 50)), "P95_ms": float(np.percentile(elapsed, 95)),
            "repetitions": repetitions, "warmup": warmup, "output_count": output.numel(),
            "device": str(device), "prototype_cache": cache, **memory}
