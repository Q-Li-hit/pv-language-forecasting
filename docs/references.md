# Methods, data and source lineage

## Method references

- **Time-LLM:** Jin et al., *Time-LLM: Time Series Forecasting by Reprogramming
  Large Language Models*, ICLR 2024.
  [Paper](https://arxiv.org/abs/2310.01728),
  [authors' implementation](https://github.com/KimMeen/Time-LLM).
  The sequence module follows patch reprogramming and statistical prompting,
  with the study's Qwen model and separate numerical forecast branch. It is a
  study-specific adaptation rather than the original Time-LLM package.
- **PatchTST:** Nie et al., *A Time Series is Worth 64 Words: Long-term Forecasting
  with Transformers*, ICLR 2023.
  [Paper](https://arxiv.org/abs/2211.14730).
  The patch reference is the univariate study implementation with a shared
  numerical branch, rather than a claim to reproduce every original configuration.
- **Qwen3 embeddings:**
  [model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B),
  [technical report](https://arxiv.org/abs/2506.05176).
  The model card specifies Apache-2.0 for the external weights. Download them
  separately and retain the model's notices when redistributing them.
- **Conformalized quantile regression:** Romano, Patterson and Candès,
  [*Conformalized Quantile Regression*](https://arxiv.org/abs/1905.03222),
  NeurIPS 2019. Empirical temporal/conditional coverage is reported separately
  from the assumptions behind conformal coverage statements.

## Data sources

The study's public source archives are
[NOAA SURFRAD](https://gml.noaa.gov/grad/surfrad/) radiation and meteorological
observations and [HRRR](https://rapidrefresh.noaa.gov/hrrr/) numerical weather
predictions. Stations are BON, DRA, FPK, GWN, PSU, SXF and TBL. Source archives
and transformed study arrays are not bundled. Their actual coverage, quality
control and release-time assumptions must accompany a scientific comparison.

## Code organization

The release consolidates the study's existing implementations into importable
modules with explicit configurations and paths:

| Module | Study implementation retained |
| --- | --- |
| `weather.py` | State categorization/template, common residual Transformer, capacity-matched context interface, weighted context projection |
| `sequence.py` | Complete-history normalization, patch mapping, prototype reprogramming, frozen full-layer sequence model and numerical branch |
| `data.py`, `trees.py` | Equivalent-power mapping and matched-information tree features |
| `evaluation.py` | Common-target errors, normalized nonnegative weighting, paired date-block statistics and prediction-average MSE identity |
| `calibration.py` | Finite-order conformal correction and arrived-label recent calibration |

The portable runners use aligned public-data-derived inputs. Import-time reads
from historical experiment folders have been replaced with explicit arguments.
Archived download, retry, manuscript-editing and workstation setup scripts are
not required by the public API. No third-party source repository or model weight
is vendored. Dependencies are installed from their own distributions and retain
their respective licences.

The repository has no assigned software licence. This is distinct from the
licences of referenced methods, dependencies and external model weights.
