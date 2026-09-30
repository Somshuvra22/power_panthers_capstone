"""Local utilities shared by the table/process pipeline."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the local project configuration."""
    config_path = Path(path) if path else ROOT / "src" / "config.json"
    with config_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def find_file(path: str | Path, pattern: str = "*.csv") -> Path:
    """Resolve a file or require exactly one matching file in a directory."""
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if not candidate.is_dir():
        raise FileNotFoundError(f"No file or directory found at {candidate}")
    matches = sorted(candidate.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {pattern} file in {candidate}; found {len(matches)}")
    return matches[0]


def read_single_csv(path: str | Path) -> pd.DataFrame:
    """Read one CSV from a file or a folder."""
    return pd.read_csv(find_file(path))


def ensure_model_exists(model_dir: str | Path | None = None) -> Path:
    """Create the local trained model if it has not been generated yet."""
    directory = Path(model_dir) if model_dir else ROOT / "models"
    model_path = directory / "model.pkl"
    if model_path.exists():
        return model_path

    directory.mkdir(parents=True, exist_ok=True)
    raw_data = ROOT / "data" / "process" / "synthetic_asset_windows.csv"
    if not raw_data.exists():
        subprocess.run([sys.executable, str(ROOT / "src" / "generate_data.py")], check=True)

    train_out = ROOT / "split" / "train.csv"
    test_out = ROOT / "split" / "test.csv"
    if not train_out.exists() or not test_out.exists():
        subprocess.run([
            sys.executable,
            str(ROOT / "src" / "prep.py"),
            "--raw_data",
            str(ROOT / "data" / "process"),
            "--train_out",
            str(train_out),
            "--test_out",
            str(test_out),
            "--test_ratio",
            "0.2",
        ], check=True)

    subprocess.run([
        sys.executable,
        str(ROOT / "src" / "train.py"),
        "--train_data",
        str(train_out),
        "--model_dir",
        str(directory),
    ], check=True)
    return model_path


def load_model(model_dir: str | Path | None = None) -> Any:
    """Load the locally persisted complete model pipeline."""
    directory = Path(model_dir) if model_dir else ROOT / "models"
    model_path = ensure_model_exists(directory)
    return joblib.load(model_path)


def log_metrics(metrics: dict[str, Any], path: str | Path = ROOT / "metrics.log.jsonl") -> None:
    """Append local-only metric logging; no cloud tracking is used."""
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, default=str) + "\n")