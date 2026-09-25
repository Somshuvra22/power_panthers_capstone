"""Train local baseline and neural-network pipelines."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tab_utils import find_file, load_config


def make_models(seed: int) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))]),
        "mlp_classifier": Pipeline([("scaler", StandardScaler()), ("model", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=seed))]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--model_dir", required=True)
    args = parser.parse_args()
    config = load_config()
    frame = pd.read_csv(find_file(args.train_data))
    timestamp = pd.to_datetime(frame[config["timestamp_column"]], utc=True)
    frame = frame.assign(_timestamp=timestamp).sort_values("_timestamp").drop(columns="_timestamp")
    split = int(len(frame) * 0.8)
    train, validation = frame.iloc[:split], frame.iloc[split:]
    X_train, y_train = train[config["approved_features"]], train[config["target"]]
    X_valid, y_valid = validation[config["approved_features"]], validation[config["target"]]
    results = {}
    fitted = {}
    for name, model in make_models(config["random_seed"]).items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_valid)
        results[name] = {"precision": precision_score(y_valid, prediction, zero_division=0), "recall": recall_score(y_valid, prediction, zero_division=0), "f1": f1_score(y_valid, prediction, zero_division=0)}
        fitted[name] = model
    selected = max(results, key=lambda name: (results[name]["recall"], results[name]["f1"]))
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[selected], model_dir / "model.pkl")
    comparison = {"validation_split": "chronological 80/20", "models": results, "selected_model": selected, "selection_reason": "Selected by highest recall, with F1 as the tie-breaker because recall is the prioritised approved metric."}
    (model_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()