"""Convert a trusted local complete-sequence joblib cache to portable NPZ."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pvforecast.data import SequenceData


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path, help="Trusted native complete-sequence cache")
    parser.add_argument("--task", choices=["day_ahead", "rolling"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--scalar-inputs", type=Path, help="Original scalar input folder, retaining the daylight flags")
    scope.add_argument("--all-hours", action="store_true", help="Explicitly evaluate day and night instead of the primary daytime targets")
    args = parser.parse_args()
    # Joblib is used only at this migration boundary. Never deserialize an
    # untrusted pickle or joblib file; the published NPZ loader disables pickle.
    original = joblib.load(args.cache)
    horizon = 24 if args.task == "day_ahead" else 12
    history = 24 if args.task == "day_ahead" else 48
    if original["sequence"].shape[1] != history or original["future"].shape[1] != horizon:
        raise ValueError("Cache dimensions differ from the selected task.")
    evaluation = np.zeros_like(original["target"], dtype=bool)
    for name, (groups, steps) in original["maps"].items():
        if name == "development":
            raise ValueError("Use the native complete-sequence cache, not the earlier scalar-sampled cache.")
        if args.all_hours:
            daylight = np.ones(len(groups), dtype=bool)
        else:
            filename = "day_ahead.pkl.gz" if args.task == "day_ahead" else f"rolling_test_{name}.pkl.gz"
            frame = pd.read_pickle(args.scalar_inputs / filename)
            if args.task == "day_ahead":
                frame = frame.loc[frame.role.eq("test")].reset_index(drop=True)
            if len(frame) != len(groups):
                raise ValueError("Scalar target count differs from the native target map.")
            expected_keys = original["keys"].iloc[groups]
            if (not np.array_equal(frame.station_id.to_numpy(), expected_keys.station_id.to_numpy())
                    or not np.array_equal(pd.to_datetime(frame.issuance_utc, utc=True).to_numpy(),
                                          pd.to_datetime(expected_keys.issuance_utc, utc=True).to_numpy())
                    or not np.array_equal(frame.horizon.to_numpy() - (args.task == "rolling"), steps)
                    or not np.array_equal(frame.target_mw.to_numpy(np.float32), original["target"][groups, steps])):
                raise ValueError("Scalar targets or ordering differ from the native target map.")
            daylight = frame.daylight.to_numpy(bool)
        evaluation[groups, steps] = daylight
    keys = original["keys"]
    metadata = {"kind": "sequence", "task": args.task, "capacity_mw": 10.0,
                "features": list(original["features"]), "state_codes": 7,
                "history_imputation_mw": float(original["imputation"]),
                "preprocessing": "original training-fitted values preserved",
                "evaluation_scope": "all-hours" if args.all_hours else "daytime",
                "prompt": "basic", "tokenizer_revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"}
    result = SequenceData(
        original["sequence"], original["future"], original["target"],
        original["masks"]["train"], original["masks"]["model_select"], evaluation,
        pd.to_datetime(keys.issuance_utc, utc=True).dt.strftime("%Y-%m-%d").to_numpy(),
        keys.station_id.to_numpy(), metadata, original["ids"], original["attention"],
        (keys.station_id.astype(str) + "|" + pd.to_datetime(keys.issuance_utc, utc=True).astype(str)).to_numpy(),
    )
    result.save(args.output)
    print(f"Exported {len(result.history)} issuances to {args.output}")


if __name__ == "__main__":
    main()
