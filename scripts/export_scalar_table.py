"""Export trusted study scalar-target tables as one aligned CSV."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Trusted .pkl.gz tables")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frames = []
    for path in args.inputs:
        # Pandas pickle is accepted only for migration of trusted local files.
        frame = pd.read_pickle(path)
        if "role" not in frame:
            raise ValueError("Input tables must retain their temporal role labels.")
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    columns = ["station_id", "issuance_utc", "horizon"]
    if frame.duplicated(columns).any():
        raise ValueError("Input tables contain duplicate station/issuance/horizon targets.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Exported {len(frame)} scalar targets to {args.output}")


if __name__ == "__main__":
    main()
