# Language pretraining for photovoltaic power forecasting

This repository contains the forecasting and evaluation code for controlled
comparisons of language pretraining in equivalent photovoltaic power forecasting.
It separates two uses of a frozen language model: encoding existing weather and
power-state features, and adapting a complete historical power sequence. The
sequence models form a separate experiment from the scalar weighted combination.

The study uses meteorological observations to construct a common 10 MW equivalent
power target at seven stations. The tasks are day-ahead forecasting and real-time
rolling forecasting. Equivalent power is a comparison target; it is not measured
power from an operating photovoltaic plant.

## Installation

Use Python 3.10 or later in a virtual environment. From the repository directory:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[all]"
pvforecast doctor
```

For point-error, ensemble and interval calculations alone, `pip install -e .`
is sufficient. The `neural` and `trees` extras install the corresponding model
dependencies. Choose a PyTorch build appropriate for the machine before installing
the neural extras. The local validation versions are listed in
[requirements-tested.txt](requirements-tested.txt).

## Quick check

```bash
python -m unittest discover -s tests -v
pvforecast demo --output runs/synthetic
pvforecast evaluate --predictions runs/synthetic/predictions.npz --output runs/synthetic/evaluation.json
```

The example creates synthetic inputs, fits a small numerical Transformer, reloads
its selected task parameters, and evaluates a separate set of targets. It runs on
a CPU and downloads no models. Its numbers are software checks, not paper results.

## Experiments

| Component | Implementation |
| --- | --- |
| Weather/state representations | Seven-variable discretization, fixed text template, numerical and category controls, frozen pretrained/random embeddings, training-fitted projection |
| Scalar forecast | Common residual Transformer and matched 33-to-32 trainable interface |
| Scalar numerical references | Direct/residual LightGBM and a local-weight TabPFN adapter with training-only labelled context |
| Complete-sequence forecast | Numerical Transformer, patch Transformer, and frozen Qwen sequence adaptation with a separate available-feature branch |
| Numerical reference | LightGBM using the same complete history, available feature vector, and history-derived prompt statistics |
| Combination and attribution | Nonnegative normalized weights, paired date-block bootstrap, decomposition of individual and averaged forecast error |
| Prediction intervals | Finite-sample conformal adjustment, fixed and arrived-label recent calibration, interval and trajectory coverage |

The package begins with aligned input tables or complete-sequence arrays. Raw
meteorological archives, trained checkpoints and tokenizer/model weights are
external inputs. The original archive download, manuscript editing and plotting
history are outside this repository. Older scalar-sampled sequence caches are
not accepted by the native-cache exporter.

### Complete-sequence models

Prepare an archive using the contract in [docs/data.md](docs/data.md). A trusted
native cache from the study can be converted without changing its numerical
inputs, masks or targets:

```bash
python scripts/export_native_cache.py data/rolling.joblib --task rolling --scalar-inputs data/scalar-inputs --output data/rolling.npz
```

Fit one numerical model:

```bash
pvforecast train-sequence --data data/rolling.npz --variant dense_numeric --seed 42 --learning-rate 0.0004 --output runs/numeric-42
pvforecast predict --data data/rolling.npz --model runs/numeric-42 --output runs/numeric-42/predictions.npz
```

For the language model, obtain a local snapshot of
[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), including
its tokenizer. The study snapshot is revision
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`. Model loading is local-only.
Archives exported from the native cache already contain the basic prompt tokens.
For a newly prepared archive:

```bash
pvforecast tokenize --data data/rolling.npz --task rolling --model-path models/qwen3 --output data/rolling-tokenized.npz
pvforecast train-sequence --data data/rolling-tokenized.npz --variant time_llm_qwen --model-path models/qwen3 --device cuda --training configs/sequence_training.json --seed 42 --output runs/pretrained-42
pvforecast predict --data data/rolling-tokenized.npz --model runs/pretrained-42 --model-path models/qwen3 --device cuda --output runs/pretrained-42/predictions.npz
```

Use `--random` to construct the same frozen architecture with random parameters.
Use `tokenize --domain` to add the static photovoltaic-domain description. Each
run fits one model. Learning-rate selection and independent seeds are separate
runs; the rules in [docs/experiments.md](docs/experiments.md) describe their use.

### Weather encoding

The aligned CSV must include the numerical fields and temporal roles specified
in [docs/data.md](docs/data.md). Numerical and category controls need no language
model:

```bash
pvforecast prepare-weather --csv data/rolling.csv --representation category --output data/weather-category.npz
pvforecast train-weather --data data/weather-category.npz --training configs/weather_training.json --seed 42 --output runs/weather-category-42
pvforecast predict --data data/weather-category.npz --model runs/weather-category-42 --output runs/weather-category-42/predictions.npz
```

For text encoding, first encode the distinct input states, then fit preprocessing
using the training rows only:

```bash
pvforecast encode-weather --csv data/rolling.csv --model-path models/qwen3 --output data/text-bank.npz
pvforecast prepare-weather --csv data/rolling.csv --representation text --embedding-bank data/text-bank.npz --output data/weather-text.npz
```

`encode-weather --random-seed 20261007` produces a fixed random-parameter encoder
bank. Repeat with the prescribed encoder seeds to examine encoder variation.
Preparing an evaluation table with `--preprocessor` reuses the saved imputation,
scaling and projection parameters. It also checks the embedding bank identity.

The scalar LightGBM references use the same prepared numerical state:

```bash
pvforecast scalar-tree fit --data data/weather-category.npz --objective direct --output runs/direct-tree
pvforecast scalar-tree fit --data data/weather-category.npz --objective residual --output runs/residual-tree
pvforecast scalar-tree predict --data data/weather-category.npz --model runs/direct-tree --output runs/direct-tree/predictions.npz
```

The optional TabPFN adapter is available through the Python API. Install
`pip install -e ".[tabular]"` separately and supply locally obtained weights.
Its labelled context is selected exclusively from training rows. Context-size
selection belongs to the development stage; the adapter does not choose it using
test errors. See [docs/api.md](docs/api.md) for the interface.

### Matched-information tree and paired evaluation

```bash
pvforecast tree fit --data data/rolling.npz --output runs/tree
pvforecast tree predict --data data/rolling.npz --model runs/tree --output runs/tree/predictions.npz
pvforecast evaluate --predictions runs/pretrained-42/predictions.npz --reference runs/tree/predictions.npz --output runs/paired.json
```

Evaluation requires identical target ordering, station identifiers, issuance
dates and masks. Differences use **evaluated method minus reference**. Negative
error differences favour the evaluated method. The bootstrap samples seven-day
blocks of UTC issuance dates together across all stations and horizons.

## Repository layout

```text
src/pvforecast/    models, data preparation, training, evaluation and command line
configs/          architecture and training settings
scripts/          local archive conversion
tests/            numerical and end-to-end checks
docs/             input contracts, experiment rules and method references
```

Use `pvforecast --help` or `python -m pvforecast --help` for command details.
Run directories are created afresh to prevent accidental replacement of a fitted
model. Generated data, checkpoints and environments are excluded by `.gitignore`.

## Interpretation and scope

Weather text describes existing variables; it adds no independent weather
observation. The common numerical branch retains these variables and their state
codes. The random-parameter comparisons examine the specified encoders and model
configurations. Individual-model error, equal-weight prediction averaging and
heterogeneous weighted combination are evaluated separately.

The main evaluation periods have already been examined and are retrospective.
Date-block confidence intervals describe variation of fixed forecasts across
dates, not uncertainty from repeating the entire training process. Model rankings
depend on the task, error metric and averaging configuration. Computational
measurements refer to model calculations and exclude data acquisition and
communication.

## References and licensing

Method references, external data links and model attribution are in
[docs/references.md](docs/references.md). No repository-wide software licence has
been assigned. Model weights, datasets and dependencies retain their own terms;
none are redistributed in this package.
