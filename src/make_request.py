"""Create a JSON scoring request from one local CSV file or folder."""

import argparse
import json
from pathlib import Path

from tab_utils import find_file, load_config
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="CSV file or folder containing one CSV")
    parser.add_argument("--out", default="request.json")
    args = parser.parse_args()
    config = load_config()
    frame = pd.read_csv(find_file(args.input))
    missing = sorted(set(config["approved_features"]) - set(frame.columns))
    if missing:
        raise ValueError(f"Input is missing approved features: {missing}")
    payload = {"data": frame[config["approved_features"]].head(5).to_dict(orient="records")}
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()