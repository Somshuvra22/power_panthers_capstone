"""Evaluate the persisted local pipeline on a chronological test holdout."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from tab_utils import find_file, load_config, load_model, log_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--metrics_out", required=True)
    parser.add_argument("--min_f1", type=float, required=True)
    args = parser.parse_args()
    config = load_config()
    frame = pd.read_csv(find_file(args.test_data))
    model = load_model(args.model_dir)
    y_true = frame[config["target"]]
    probabilities = model.predict_proba(frame[config["approved_features"]])[:, 1]
    predictions = (probabilities >= config["decision_threshold"]).astype(int)
    metrics = {"dataset_status": "SYNTHETIC", "threshold": config["decision_threshold"], "accuracy": accuracy_score(y_true, predictions), "precision": precision_score(y_true, predictions, zero_division=0), "recall": recall_score(y_true, predictions, zero_division=0), "f1": f1_score(y_true, predictions, zero_division=0), "positive_window_count": int(y_true.sum()), "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist()}
    # Audit diagnostics use only stop timestamps retained outside the feature set.
    alerts = pd.to_datetime(frame.loc[predictions == 1, "timestamp"], utc=True)
    stops = pd.to_datetime(frame.loc[frame["stop_timestamp"].fillna("") != "", "stop_timestamp"], utc=True).drop_duplicates()
    lead_times = [(stop - alert).total_seconds() / 3600 for stop in stops for alert in alerts if pd.Timedelta(0) < stop - alert <= pd.Timedelta(hours=2)]
    metrics["event_level_recall"] = float(sum(any(pd.Timedelta(0) < stop - alert <= pd.Timedelta(hours=2) for alert in alerts) for stop in stops) / len(stops)) if len(stops) else None
    metrics["alert_lead_time_hours"] = {"count": len(lead_times), "values": lead_times, "mean": sum(lead_times) / len(lead_times) if lead_times else None}
    output = Path(args.metrics_out)
    if output.suffix.lower() != ".json":
        output.mkdir(parents=True, exist_ok=True)
        output = output / "metrics.json"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log_metrics(metrics, output.parent / "metrics.log.jsonl")
    raise SystemExit(0 if metrics["f1"] >= args.min_f1 else 1)


if __name__ == "__main__":
    main()