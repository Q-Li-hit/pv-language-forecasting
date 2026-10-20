# Experiment rules

## Temporal roles

| Period | Role |
| --- | --- |
| 2020–2023 | Training and fitted preprocessing |
| January–June 2024 | Architecture/learning-rate choice and early stopping |
| July–September 2024 | Scalar combination-weight fitting |
| October 2024 | Combination configuration selection |
| November–December 2024 | Fixed interval calibration |
| 2025 | Previously examined retrospective evaluation |

The original data split excludes 48-hour transition gaps. Raw numerical weather
availability follows an assumed two-hour delay after initialization; this is an
archive convention rather than measured file-arrival time. Completed observation
windows can enter the model at issuance. A calibration label has an additional
five-minute delay. These two availability rules describe different uses of data.

The arrays supplied to the package must already apply the target grid, release
grid, quality control and availability convention. The package does not infer
those rules from unannotated raw weather files.

## Weather representations

The active matched control uses a common history/state Transformer and a
bias-free 33-to-32 context adapter. Its numerical interface is a fixed seven-to-33
linear expansion. Categories use 33 one-hot coordinates; text uses 32
training-fitted principal components plus a constant. Matching the interface and
trainable parameter count does not equate representation rank or inductive bias.
The common numerical branch also retains the underlying state information.

`configs/weather_training.json` describes the early-stopped matched-MSE panel.
Fixed-budget variants can be specified by setting minimum and maximum epochs to
the same value: 20 for day-ahead and 10 for rolling, with learning rate 0.0004.
Report the chosen loss and budget with each comparison rather than pooling these
settings into one result. Encoder instances 20261007, 20261008 and 20261009 are
separate from downstream initialization seeds 42, 43 and 44.

## Complete-sequence comparisons

The architecture settings are in `configs/sequence_architecture.json`. The
sequence adaptation follows Time-LLM's patch reprogramming approach with the
complete frozen Qwen3 embedding model. It uses 128 learned vocabulary prototypes
and the first 64 output components. A separate numerical MLP processes each
target's available feature vector. All language-model layers remain present;
freezing them still permits gradients to reach the task mapping. The task modules
include the large vocabulary projection, so freezing should not be equated with
a one-percent adaptation budget.

For the final early-stopped sequence comparison:

- Test learning rates 0.0001, 0.0004 and 0.0016 using seed 42 and selection labels.
- Select the lowest complete selection SmoothL1 loss independently for each
  task/model condition; break ties by the smaller learning rate.
- At that rate, fit independent seeds 43 and 44. Retain seed 42's selected run.
- Use masked SmoothL1 loss with beta 0.35 MW, effective batch 128, AdamW weight
  decay 0.0001 and gradient clipping at 1.
- Run at least eight epochs, at most 240, with patience five. Patience resets on
  an improvement exceeding `max(1e-6,1e-4 × reference_loss)`; the best absolute
  selection score is checkpointed separately.

The CLI trains one configuration per invocation. A maximum-epoch stop is recorded
as a budget stop rather than convergence. The supplied CUDA training configuration
uses bfloat16; the default command without that file uses CPU float32 for local
checks. Execution uses explicit mathematical scaled-dot-product attention and
disables Transformer evaluation fast paths and reduced-precision float32 matmul.
Different microbatches, hardware or floating-point backends can change training
trajectories, especially with dropout; they require an explicit comparison before
being treated as interchangeable.

Random language-model parameters use seed `20261100 + task_seed` in an isolated
random-number state. The task initial tensors and surrounding random state match
the pretrained construction. The pretrained and random configurations can select
different learning rates under the same rule; their difference is therefore a
configuration comparison under that selection procedure.

Report each single model and the equal-weight mean of seeds 42–44 separately.
Average individually clipped predictions. `mse_average_decomposition` identifies
the mean individual MSE, averaged-prediction MSE and reduction due to averaging;
this algebra does not establish a new physical mechanism.

## Numerical reference

The tree baseline concatenates the entire main history, the original available
vector for each horizon, and the nine deterministic statistics used in the prompt.
Its three candidates have `(num_leaves,min_child_samples)` of `(31,50)`, `(63,80)`
and `(127,100)`. Training uses squared error, learning rate 0.04, L2 penalty 1,
at most 500 trees and early-stopping patience 40. Selection uses unclipped
SmoothL1 loss on the same selection targets. The models have the same raw
information, but their numerical organization and training objectives differ.

## Combination, intervals and uncertainty

`fit_simplex_weights` estimates one nonnegative normalized vector on designated
combination-fit targets. A deletion/refit comparison removes a member and fits
the remaining weights again. Skipping a zero-weight member preserves the original
weights and is a different operation. Weather and horizon groups reuse the
task-level weights unless an explicitly separate experiment specifies otherwise.

The evaluation module reports overall point errors, signed paired differences,
90% prediction-interval coverage/width/score and simultaneous trajectory coverage.
The calibration module implements nonnegative signed conformal adjustments and
the arrived-label recent-window policy. Calibration and bootstrap intervals are
different objects: prediction intervals concern future power; bootstrap confidence
intervals concern error differences of fixed forecast arrays.

Bootstrap defaults are 2,000 circular seven-calendar-day block draws with seed
20261008. Draws are synchronized across stations and horizons. The default is a
pointwise 95% confidence interval. For a prespecified family of `m` comparisons,
set confidence to `1 - 0.05/m` for the study's Bonferroni interval. This does not
account for training randomness or automatically guarantee conditional coverage.

## Resource measurements

The sequence model exposes an evaluation-only prototype cache. Build and use it
inside `torch.inference_mode()` with the model in evaluation mode. It caches the
vocabulary projection and its key/value projections, not the prediction itself.
Training, loading task parameters or moving devices invalidates the cache.
This repository keeps ordinary gradient computation during training.

Time loading, cache construction, first request and repeated resident predictions
separately. Synchronize the GPU around timed regions. Record output count, batch,
precision, warm-up and repetitions. One cold request is a single observation;
P50/P95 summaries require repeated requests. Model timings exclude acquisition,
communication and forecast-product arrival delays. Shared frozen parameters can
reduce resident memory while retaining multiple task heads and forward passes.
