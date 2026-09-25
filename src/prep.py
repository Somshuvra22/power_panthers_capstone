"""Validate and chronologically split the synthetic time-series data."""

import argparse
from pathlib import Path

import pandas as pd

from tab_utils import find_file, load_config


def validate_frame(frame: pd.DataFrame, config: dict) -> None:
    required = [config["timestamp_column"], *config["approved_features"], config["target"]]
    missing_columns = sorted(set(required) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if frame[required].isna().any().any():
        raise ValueError("Required columns contain missing values")
    if frame[config["target"]].nunique() < 2:
        raise ValueError("Target must contain both classes")
    timestamps = pd.to_datetime(frame[config["timestamp_column"]], utc=True, errors="coerce")
    if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("Timestamp must be valid and chronologically sorted")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", required=True)
    parser.add_argument("--train_out", required=True)
    parser.add_argument("--test_out", required=True)
    parser.add_argument("--test_ratio", type=float, required=True)
    args = parser.parse_args()
    if not 0 < args.test_ratio < 1:
        raise ValueError("--test_ratio must be between 0 and 1")
    config = load_config()
    frame = pd.read_csv(find_file(args.raw_data))
    validate_frame(frame, config)
    split = int(len(frame) * (1 - args.test_ratio))
    if split <= 0 or split >= len(frame):
        raise ValueError("Chronological split produced an empty partition")
    Path(args.train_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.test_out).parent.mkdir(parents=True, exist_ok=True)
    frame.iloc[:split].to_csv(args.train_out, index=False)
    frame.iloc[split:].to_csv(args.test_out, index=False)


if __name__ == "__main__":
    main()