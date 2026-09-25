"""Power Panthers local Streamlit application."""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tab_utils import load_config, load_model  # noqa: E402


CONFIG = load_config()
FEATURES = CONFIG["approved_features"]
MODEL_PATH = ROOT / "models" / "model.pkl"


@st.cache_resource
def get_model():
    """Load the persisted local model once per Streamlit process."""
    if os.getenv("MODE", "local").lower() != "local":
        raise RuntimeError("Only MODE=local is supported; Azure/cloud modes are not available.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Local model was not found at {MODEL_PATH}")
    return load_model(ROOT / "models")


def validate_features(frame: pd.DataFrame) -> list[str]:
    missing = [feature for feature in FEATURES if feature not in frame.columns]
    if missing:
        return missing
    if frame[FEATURES].isna().any().any():
        return ["missing values in one or more approved feature columns"]
    for feature in FEATURES:
        try:
            pd.to_numeric(frame[feature], errors="raise")
        except (TypeError, ValueError):
            return [f"{feature} must contain numeric values"]
    return []


def prediction_frame(model, frame: pd.DataFrame) -> pd.DataFrame:
    values = frame[FEATURES].apply(pd.to_numeric)
    predictions = model.predict(values).astype(int)
    probabilities = model.predict_proba(values)[:, 1] if hasattr(model, "predict_proba") else [None] * len(frame)
    result = frame.copy()
    result["prediction"] = ["Abnormal" if value else "Normal" for value in predictions]
    result["abnormal_probability"] = probabilities
    return result


def show_probability_chart(probability: float) -> None:
    figure, axis = plt.subplots(figsize=(7, 1.5))
    axis.barh(["Abnormal window probability"], [probability], color="#d9534f" if probability >= CONFIG["decision_threshold"] else "#2e8b57")
    axis.set_xlim(0, 1)
    axis.set_xlabel("Probability")
    axis.axvline(CONFIG["decision_threshold"], color="black", linestyle="--", label="Decision threshold")
    axis.legend(loc="lower right")
    st.pyplot(figure, clear_figure=True)


st.set_page_config(page_title="Power Panthers Stoppage Predictor", page_icon="⚡", layout="wide")
st.title("Power Panthers Stoppage Predictor")
st.caption("Local MODE=local inference for abnormal operating windows")
st.info("This project uses SYNTHETIC data for development and evaluation. Results are not production evidence.")

try:
    model = get_model()
except Exception as error:
    st.error(f"Unable to load the local model: {error}")
    st.stop()

single_tab, batch_tab = st.tabs(["Single operating window", "Batch CSV predictions"])

with single_tab:
    st.subheader("Enter approved sensor trends")
    with st.form("single_prediction"):
        columns = st.columns(2)
        single_values = {}
        defaults = {"rolling_vibration_trend": 2.5, "rolling_temperature_trend": 65.0, "rolling_current_trend": 70.0, "rolling_load_trend": 60.0}
        for index, feature in enumerate(FEATURES):
            single_values[feature] = columns[index % 2].number_input(feature, value=float(defaults.get(feature, 0.0)), format="%.4f")
        submitted = st.form_submit_button("Predict operating window")
    if submitted:
        try:
            result = prediction_frame(model, pd.DataFrame([single_values]))
            abnormal = result.loc[0, "prediction"] == "Abnormal"
            probability = float(result.loc[0, "abnormal_probability"])
            if abnormal:
                st.error("Predicted status: ABNORMAL operating window")
            else:
                st.success("Predicted status: NORMAL operating window")
            st.metric("Abnormal-window probability", f"{probability:.1%}")
            show_probability_chart(probability)
        except Exception as error:
            st.error(f"Could not score this operating window. Check that all inputs are valid numbers. Details: {error}")

with batch_tab:
    st.subheader("Upload a CSV for batch predictions")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
            errors = validate_features(batch)
            if errors:
                st.error("Invalid CSV: " + "; ".join(errors) + ". Required columns are: " + ", ".join(FEATURES))
            else:
                result = prediction_frame(model, batch)
                abnormal_count = int((result["prediction"] == "Abnormal").sum())
                st.success(f"Scored {len(result)} rows successfully. Abnormal predictions: {abnormal_count}.")
                st.dataframe(result, use_container_width=True)
                chart_data = result["prediction"].value_counts().reindex(["Normal", "Abnormal"], fill_value=0)
                figure, axis = plt.subplots(figsize=(6, 3))
                chart_data.plot.bar(ax=axis, color=["#2e8b57", "#d9534f"], rot=0)
                axis.set_ylabel("Number of windows")
                axis.set_title("Batch prediction status")
                st.pyplot(figure, clear_figure=True)
        except Exception as error:
            st.error(f"Could not read or score the uploaded CSV. Please upload a valid CSV. Details: {error}")