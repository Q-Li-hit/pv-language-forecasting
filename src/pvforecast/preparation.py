"""Prepare matched weather inputs from an already aligned scalar-target table."""

import json
from pathlib import Path

import numpy as np

from .data import ScalarData
from .utils import array_digest


def prepare_weather(frame, output, *, representation, preprocessor=None, embedding_bank=None,
                    capacity_mw=10.0, projection_seed=20261008):
    from sklearn.preprocessing import StandardScaler
    from .weather import (WeatherPreprocessor, WeightedPCA, STATE_FIELDS, context_fields,
                          context_codes_key, category_context, numeric_projection, feature_columns, raw_features)

    required = {"target_mw", "base_mw", "role", "issuance_utc", "station_id", *STATE_FIELDS}
    if required - set(frame):
        raise ValueError("Missing columns: " + ", ".join(sorted(required - set(frame))))
    missing_features = set(feature_columns(frame)) - set(frame)
    if missing_features:
        raise ValueError("Missing numerical features: " + ", ".join(sorted(missing_features)))
    if representation not in ("numeric", "category", "text", "random"):
        raise ValueError("Unknown weather representation.")
    training = frame.role.eq("train").to_numpy()
    codes, _ = context_fields(frame)
    bank = bank_keys = pca = None
    if representation in ("text", "random"):
        if embedding_bank is None:
            raise ValueError("Text and random representations require a frozen embedding bank.")
        with np.load(embedding_bank, allow_pickle=False) as archive:
            bank_keys, bank = archive["keys"], archive["embedding"]
            bank_metadata = json.loads(str(archive["metadata"]))
        if bank_metadata["representation"] != representation:
            raise ValueError("Embedding bank belongs to a different representation.")
        bank_identity = array_digest({"keys": bank_keys, "embedding": bank})
    if preprocessor is None:
        fitted = WeatherPreprocessor.fit(frame.loc[training], capacity_mw=capacity_mw)
        projection = numeric_projection(projection_seed)
        if bank is not None:
            pca = WeightedPCA.fit(bank, bank_keys, codes[training])
        metadata = {"kind": "weather", "representation": representation,
                    "capacity_mw": capacity_mw, "features": list(fitted.features),
                    "lag_names": list(fitted.lag_names), "projection_seed": projection_seed}
        if bank is not None:
            metadata["encoder"] = bank_metadata
            metadata["embedding_bank_sha256"] = bank_identity
        destination = Path(output).with_suffix(".preprocessing.npz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(fill=fitted.fill, mean=fitted.scaler.mean_, scale=fitted.scaler.scale_,
                      var=fitted.scaler.var_, numeric_projection=projection,
                      metadata=np.asarray(json.dumps(metadata)))
        if pca is not None:
            arrays.update(pca_mean=pca.mean, pca_components=pca.components)
        metadata["preprocessing_sha256"] = array_digest({k: v for k, v in arrays.items() if k != "metadata"})
        arrays["metadata"] = np.asarray(json.dumps(metadata))
        np.savez_compressed(destination, **arrays)
    else:
        with np.load(preprocessor, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            if metadata["representation"] != representation or metadata["capacity_mw"] != capacity_mw:
                raise ValueError("Preprocessor representation or capacity differs.")
            identity = array_digest({k: saved[k] for k in saved.files if k != "metadata"})
            if identity != metadata["preprocessing_sha256"]:
                raise ValueError("Preprocessing parameters differ from their recorded identity.")
            scaler = StandardScaler()
            scaler.mean_, scaler.scale_, scaler.var_ = saved["mean"], saved["scale"], saved["var"]
            scaler.n_features_in_ = len(scaler.mean_)
            fitted = WeatherPreprocessor(tuple(metadata["features"]), tuple(metadata["lag_names"]),
                                         saved["fill"], scaler, capacity_mw)
            projection = saved["numeric_projection"]
            if bank is not None:
                if bank_metadata != metadata["encoder"] or bank_identity != metadata["embedding_bank_sha256"]:
                    raise ValueError("Frozen encoder identity differs from the fitted preprocessing.")
                pca = WeightedPCA(saved["pca_mean"], saved["pca_components"])
    history, state = fitted.transform(frame)
    if representation == "numeric":
        context = fitted.numeric_context(state, projection)
    elif representation == "category":
        context = category_context(codes)
    else:
        wanted = context_codes_key(codes)
        indices = np.searchsorted(bank_keys, wanted)
        if (indices >= len(bank_keys)).any() or not np.array_equal(bank_keys[np.minimum(indices, len(bank_keys)-1)], wanted):
            raise ValueError("Embedding bank lacks an input state; encode the new states first.")
        context = pca.transform(bank[indices], add_constant=True)
    import pandas as pd
    dates = pd.to_datetime(frame.issuance_utc, utc=True).dt.strftime("%Y-%m-%d").to_numpy()
    mask = frame.role.eq("test").to_numpy()
    if "daylight" in frame:
        if not frame.daylight.isin([True, False, 0, 1]).all():
            raise ValueError("The daylight column must contain boolean or 0/1 values.")
        mask &= frame.daylight.to_numpy(bool)
    timestamp = pd.to_datetime(frame.issuance_utc, utc=True).astype(str)
    keys = (frame.station_id.astype(str) + "|" + timestamp + "|" + frame.horizon.astype(str)).to_numpy()
    data = ScalarData(history, state, context, frame.target_mw.to_numpy(), frame.base_mw.to_numpy(),
                      frame.role.to_numpy(), dates, frame.station_id.to_numpy(), metadata, mask, keys)
    raw = raw_features(frame, fitted.features)
    data.raw_state = np.where(np.isfinite(raw), raw, fitted.fill).astype(np.float32)
    data.save(output)
    return data
