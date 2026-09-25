"""Local scoring interface for the persisted pipeline."""

from pathlib import Path

import pandas as pd

from tab_utils import load_model


MODEL = None


def init() -> None:
    global MODEL
    MODEL = load_model(Path(__file__).resolve().parents[1] / "models")


def run(data: dict) -> dict:
    if MODEL is None:
        init()
    records = data.get("data")
    if not isinstance(records, list) or not records:
        raise ValueError("data must contain a non-empty list under 'data'")
    frame = pd.DataFrame(records)
    predictions = MODEL.predict(frame).astype(int).tolist()
    probabilities = MODEL.predict_proba(frame)[:, 1].tolist()
    return {"predictions": predictions, "probabilities": probabilities}