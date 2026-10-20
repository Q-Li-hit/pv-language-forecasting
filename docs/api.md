# Additional Python interfaces

## Combination weights and prediction averaging

```python
from pvforecast.evaluation import fit_simplex_weights, mse_average_decomposition

# Fit only on designated combination-fit targets. Rows must be common targets.
weights = fit_simplex_weights(member_predictions, targets)
combined = member_predictions @ weights

# Each row is one independent training instance on the same targets.
decomposition = mse_average_decomposition(seed_predictions, targets)
```

For a member-deletion comparison, remove that column and refit on the same
combination-fit rows. Prediction averaging across seeds and fitting weights across
different models are separate operations.

## Fixed and arrived-label calibration

`pvforecast.calibration` provides `fit_fixed_calibrator`, `recent_adjustments`,
`arrived_label_mask` and `calibrated_interval`. The input quantiles come from an
already fitted quantile predictor. Threshold fitting does not change predictor
parameters. `recent_adjustments` applies the configured label delay and recent
window before choosing per-horizon, pooled or fixed thresholds. Inspect the
function signatures for coverage levels, minimum sample counts and update time.
`pvforecast.evaluation.interval_metrics` evaluates the resulting endpoints and
can also accept trajectory identifiers for simultaneous coverage.

## Local TabPFN reference

```python
from pvforecast.data import ScalarData
from pvforecast.scalar_baselines import fit_tabpfn, predict_tabpfn

data = ScalarData.load("data/weather-category.npz")
horizons = data.raw_state[:, data.metadata["features"].index("horizon")]
predictor, report = fit_tabpfn(
    data, "models/tabpfn.ckpt", horizons=horizons, budget=1024,
    seed=42, device="cpu", output="runs/tabpfn",
)
predictions = predict_tabpfn(predictor, data)
```

The context file stores numeric features, labels and training indices. Loading
it with `load_tabpfn` recreates the estimator from that same context and checks the
local weight identity. The package does not distribute weights or download them
on demand. Actual pretrained TabPFN inference was not part of release checking;
tests exercise training-only context selection, persistence and loader checks.

## Resident inference timing

```python
import torch
from pvforecast.benchmark import benchmark_inference

model.eval()
# inputs are the model's already prepared tensor arguments on the same device.
with torch.inference_mode():
    result = benchmark_inference(model, inputs, warmup=10, repetitions=200)
```

For a frozen language sequence model, `cache=True` measures its inference-only
prototype cache. Use the same autocast context for cached and uncached calls.
The function reports cache setup, first forward, warmed P50/P95 and CUDA memory
where available. These are raw model forwards; loading, feature construction,
power clipping, interval construction and communication are outside the timed
region. Output equivalence is tested independently from timing.
