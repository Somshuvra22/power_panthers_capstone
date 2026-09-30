import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tab_utils import ensure_model_exists  # noqa: E402


def _config():
    return json.loads((ROOT / "src" / "config.json").read_text(encoding="utf-8"))


def _data():
    files = sorted((ROOT / "data" / "process").glob("*.csv"))
    assert len(files) == 1, "Expected exactly one generated process CSV"
    return pd.read_csv(files[0])


def test_required_columns_exist():
    frame = _data()
    config = _config()
    assert set([config["timestamp_column"], config["target"], *config["approved_features"]]).issubset(frame.columns)


def test_dataset_contains_at_least_200_rows():
    assert len(_data()) >= 200


def test_target_is_not_empty():
    target = _data()[_config()["target"]]
    assert not target.empty
    assert target.notna().all()


def test_approved_features_are_present():
    frame = _data()
    assert set(_config()["approved_features"]).issubset(frame.columns)


def test_model_is_created_when_missing(tmp_path):
    model_dir = tmp_path / "models"
    model_path = ensure_model_exists(model_dir)
    assert model_path.exists()
    assert model_path.name == "model.pkl"
