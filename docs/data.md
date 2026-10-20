# Data contracts

All power arrays use MW unless stated otherwise. Input construction must respect
the information available at forecast issuance. Transformations estimated from
data use training rows only. Evaluation labels may occupy the same archive, but
training and selection use explicit disjoint masks.

## Complete-sequence NPZ

`SequenceData.save` writes the following arrays with no Python object payloads.
`SequenceData.load` uses `allow_pickle=False`.

| Field | Shape | Meaning |
| --- | --- | --- |
| `history` | `N × L` | Completed power windows, oldest first, after training-fitted imputation |
| `future` | `N × H × F` | Available numerical inputs for every target horizon, already training-standardized |
| `target` | `N × H` | Equivalent power; unscored placeholders have a false mask |
| `train_mask` | `N × H`, boolean | Effective training targets |
| `selection_mask` | `N × H`, boolean | Effective model-selection targets |
| `evaluation_mask` | `N × H`, boolean | Effective retrospective evaluation targets |
| `issue_dates` | `N`, string | UTC issuance dates in `YYYY-MM-DD` format |
| `station_ids` | `N`, string | Station identifier per issuance |
| `sample_keys` | `N`, string | Unique station/UTC-issuance identifiers, preserving row order |
| `ids` / `attention` | `N × T`, optional | Prompt token IDs and valid-token mask |
| `metadata` | scalar JSON string | Task, capacity, feature order, fitted-preprocessing identity and prompt information |

For day-ahead forecasting, `L=24` hourly history points and `H=24` hourly target
windows. For rolling forecasting, `L=48` five-minute history points and `H=12`
future five-minute windows. The original numerical vectors have `F=51` and
`F=47`, respectively, including seven state codes. They retain auxiliary power
lags, available weather variables, time and solar features, and the station code.
The numerical vector is therefore broader than a weather-only vector.

Every future horizon needs its own available inputs, including horizons whose
labels are missing. A target mask does not justify filling an entire horizon's
inputs with zero. Training, selection and evaluation do not share issuance rows.
The exporter retains the original masks and array order. For the primary daytime
evaluation, `--scalar-inputs` supplies the corresponding original scalar tables
with their daylight flags. The exporter verifies their targets and ordering
against the native maps before constructing the daytime mask. `--all-hours`
explicitly selects day and night; its metrics differ from the primary comparison.

The native-cache converter uses joblib only to read trusted study files. Loading
pickle or joblib from an unknown source can execute code. Public array archives
should use the NPZ contract above.

## Aligned weather/state CSV

Each row is one scalar target. Required identifiers are `station_id`,
`issuance_utc`, `role`, `target_mw` and `base_mw`. Temporal roles are `train`,
`model_select`, `fusion_fit`, `fusion_select`, `calibration` and `test`;
`unused` and `embargo` rows can also be retained. All scored targets and residual
baselines must be finite before preparation.
An optional boolean `daylight` column restricts evaluation to daytime test rows;
without it, all rows with role `test` are evaluated. This choice does not change
the training or selection rows. The study's primary daytime comparison requires
the original daylight field.

The seven state fields are:

| Field | Definition | Boundaries used for categorization |
| --- | --- | --- |
| `cloud_low` | Forecast low cloud cover (%) | 20, 60, 90 |
| `cloud_mid` | Forecast middle cloud cover (%) | 20, 60, 90 |
| `cloud_high` | Forecast high cloud cover (%) | 20, 60, 90 |
| `cloud_total` | Forecast total cloud cover (%) | 20, 60, 90 |
| `nwp_csi` | Forecast irradiance/clear-sky irradiance ratio | 0.35, 0.7, 1.2 |
| `recent_ramp` | Recent completed power change (MW) | −0.5, 0.5 |
| `observed_residual` | Latest completed power minus the then-available numerical forecast conversion (MW) | −0.5, 0.5 |

Each field has an explicit unknown category. The first five fields have five
categories each; the final two have four. The category interface therefore has
33 coordinates. These are weather **and power-state** fields, rather than seven
independent meteorological measurements.

Common numerical features include every column prefixed `hist_`, `phase_`,
`nwp_` or `cloud_`, except `nwp_target_hour`, followed by:

```text
forecast_age_h forecast_lead_h forecast_elapsed_h horizon station_code
sin_hour cos_hour sin_doy cos_doy csky_power persistence_mw
recent_ramp observed_residual
```

These columns must be numerical. Do not include timestamp or target-label columns
under the reserved prefixes. The common numerical branch appends the seven
category codes. History-token columns are the `hist_` fields in the original
table order, normalized by nominal capacity; their missing values become zero.
The common numerical branch uses coordinate-wise training medians and a
training-fitted standard scaler.

`prepare-weather` writes both the portable scalar archive and a
`.preprocessing.npz` file. The scalar archive contains `history`, `state`,
`context`, `target`, `baseline`, `roles`, `issue_dates`, `station_ids` and JSON
metadata. The numerical expansion, one-hot categories and projected text each
produce a 33-coordinate context. For text, the 32 principal components are fitted
with training-state frequencies; a constant coordinate completes the interface.
The optional `raw_state` matrix retains the same training-imputed numerical
variables before standardization for scalar tree and TabPFN references.

For later data, pass the existing `--preprocessor`. The fitted arrays have a
content fingerprint. Text preprocessing also checks the supplied embedding bank
against the bank used at fitting; extending the state bank requires an explicit
new preparation rather than silently replacing an encoder.

## Forecast files

`predict` writes `target`, `prediction`, `mask`, `issue_dates`, `station_ids` and
`sample_keys`.
Prediction shape matches the target. Power is clipped to `[0,10] MW` after adding
the residual baseline for scalar models. Two forecast files must have identical
keys and masks before paired evaluation. Keep full timestamps and horizon keys
in the upstream aligned tables when joining predictions from separate sources;
an issuance date alone is not a unique target identifier.
The native exporter retains station/issuance keys; scalar preparation retains
station/issuance/horizon keys. Direct array users should supply equivalent unique
keys. If omitted, the low-level data class uses positional row keys, which are
suitable only when all methods consume the same archive without reordering.
