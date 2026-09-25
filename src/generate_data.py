"""Generate the deterministic, synthetic chronological asset dataset."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "process" / "synthetic_asset_windows.csv"
SEED = 42


def generate_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    periods = 60 * 24
    timestamps = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")
    # Autocorrelated normal operation with correlated load/current/temperature/vibration.
    load = np.empty(periods)
    load[0] = 60.0
    for index in range(1, periods):
        load[index] = 0.96 * load[index - 1] + 0.04 * (60 + 12 * np.sin(index / 24)) + rng.normal(0, 1.4)
    load = np.clip(load, 20, 100)
    current = np.clip(20 + 0.85 * load + rng.normal(0, 3, periods), 20, 120)
    temperature = np.clip(48 + 0.30 * load + rng.normal(0, 1.5, periods), 40, 95)
    vibration = np.clip(0.8 + 0.045 * load + rng.normal(0, 0.18, periods), 0.5, 12)

    frame = pd.DataFrame({
        "timestamp": timestamps,
        "rolling_vibration_trend": vibration,
        "rolling_temperature_trend": temperature,
        "rolling_current_trend": current,
        "rolling_load_trend": load,
        "abnormal_window": 0,
        "event_id": "",
        "stop_timestamp": "",
        "precursor_detectable": False,
    })

    # 30 events in 60 days = the current scenario rate of approximately 15/month.
    event_starts = np.linspace(70, periods - 3, 30, dtype=int)
    # 23/30 = 76.7%, the closest whole-event approximation to three quarters.
    detectable_count = int(np.ceil(len(event_starts) * 0.75))
    detectable_events = set(rng.permutation(len(event_starts))[:detectable_count])
    for event_number, stop_index in enumerate(event_starts, start=1):
        event_id = f"STOP_{event_number:03d}"
        stop_time = timestamps[stop_index]
        frame.loc[stop_index, ["event_id", "stop_timestamp"]] = [event_id, stop_time.isoformat()]
        is_detectable = event_number - 1 in detectable_events
        frame.loc[stop_index, "precursor_detectable"] = is_detectable
        if is_detectable:
            # The two preceding hourly windows are the abnormal two-hour lead period.
            for offset, strength in ((2, 0.55), (1, 1.0)):
                index = stop_index - offset
                frame.loc[index, "abnormal_window"] = 1
                frame.loc[index, "event_id"] = event_id
                frame.loc[index, "stop_timestamp"] = stop_time.isoformat()
                frame.loc[index, "precursor_detectable"] = True
                frame.loc[index, "rolling_vibration_trend"] = np.clip(vibration[index] + 2.2 * strength, 0.5, 12)
                frame.loc[index, "rolling_temperature_trend"] = np.clip(temperature[index] + 10 * strength, 40, 95)
                frame.loc[index, "rolling_current_trend"] = np.clip(current[index] + 18 * strength, 20, 120)
                frame.loc[index, "rolling_load_trend"] = np.clip(load[index] + 16 * strength, 20, 100)
    return frame


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_dataset()
    frame.to_csv(OUT, index=False)
    metadata = {
        "dataset_status": "SYNTHETIC",
        "random_seed": SEED,
        "sampling_interval": "1 hour",
        "time_span": "60 days",
        "rows": len(frame),
        "stoppage_events": int(frame["event_id"].replace("", np.nan).nunique()),
        "detectable_events": int(frame.loc[frame["precursor_detectable"], "event_id"].replace("", np.nan).nunique()),
        "precursor_event_proportion": int(frame.loc[frame["precursor_detectable"], "event_id"].replace("", np.nan).nunique()) / int(frame.loc[frame["event_id"].replace("", np.nan).notna(), "event_id"].nunique()),
        "feature_units": {"vibration": "mm/s", "temperature": "°C", "current": "A", "load": "% rated load"},
        "missing_value_handling": "none generated; prep rejects missing values",
        "clipping": "values clipped to specification simulation ranges",
    }
    OUT.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {OUT} with shape {frame.shape}")


if __name__ == "__main__":
    main()