# Power Panthers Stoppage Predictor

## Project status

This is a local **Table (Process)** machine-learning project for detecting abnormal operating windows before an unplanned equipment stoppage.

**All data in this repository is SYNTHETIC.** It is generated locally for development, testing, and end-to-end workflow validation. It is not production sensor data and the results do not prove business impact.

The project is intentionally local-only. It does not require or use Azure, Azure ML, or cloud tracking.

## Business problem

Unplanned stoppages increased from 7 to 15 per month while availability fell from 94.0% to 90.7%. Most stops are assumed to have a detectable sensor precursor, creating an opportunity for earlier maintenance intervention.

The business direction is to reduce monthly unplanned stops and recover availability toward the earlier approximately 94% baseline. The exact numeric business target remains to be agreed with operations; this project does not invent one.

## ML problem

This is a time-windowed binary classification/anomaly-detection problem. The model predicts whether an operating window is an abnormal precursor window within the two hours before an unplanned stoppage.

Recall is prioritised because missing a real precursor is more costly than raising a false alert. Evaluation uses chronological validation and test splits, rather than random splits.

## Features and target

The model uses only these four approved rolling trends over the prior two hours:

| Feature | Unit / simulation range |
|---|---|
| `rolling_vibration_trend` | RMS velocity, mm/s; approximately 0.5–12 |
| `rolling_temperature_trend` | °C; approximately 40–95 |
| `rolling_current_trend` | A; approximately 20–120 |
| `rolling_load_trend` | Percent of rated load; approximately 20–100% |

Target: `abnormal_window`. A positive label means an abnormal precursor operating window within the two-hour lead-time interval before a stop. `timestamp`, `event_id`, `stop_timestamp`, and `precursor_detectable` are retained for chronology and audit diagnostics, but are not model input features.

## Synthetic data configuration

- Seed: `42`; one representative asset; hourly sampling; 60 days
- Dataset shape: 1,440 rows × 9 columns
- Stoppage events: 30, approximately 15 per month
- Detectable precursor events: 23/30, or 76.7%
- No-precursor events: 7/30, or 23.3%
- Target windows: 46 positive and 1,394 negative

The closest whole-event approximation to the specified 75% / 25% concept is used because 30 events cannot be divided into exact quarters. Values are clipped to the stated simulation ranges. No missing values are generated; preparation rejects missing required values.

## Project structure

```text
app/app.py                                  Local Streamlit application
data/process/synthetic_asset_windows.csv   Generated synthetic dataset
data/process/synthetic_asset_windows.metadata.json  Dataset metadata
models/model.pkl                            Selected scaler + model pipeline
models/comparison.json                      Validation model comparison
metrics/metrics.json                        Test metrics and diagnostics
specs/problem_brief.md                      Approved problem brief
specs/capstone_spec.md                     Approved capstone specification
specs/validation_report.md                  Phase 5 acceptance report
src/config.json                             Task, target, and features
src/generate_data.py                        Deterministic data generator
src/prep.py                                 Validation and chronological split
src/train.py                                Model training and selection
src/evaluate.py                             Holdout evaluation and quality gate
src/score.py                                Local scoring interface
src/make_request.py                         Sample request generator
src/tab_utils.py                            Local file/model utilities
tests/test_data.py                          Dataset acceptance tests
```

## Installation

Use Python 3.11 or a compatible Python 3 environment on the local Linux VM.

```bash
cd /root/power_panthers_capstone
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Data generation, training, and evaluation

```bash
python3 src/generate_data.py
pytest
python3 src/prep.py --raw_data data/process --train_out split/train --test_out split/test --test_ratio 0.2
python3 src/train.py --train_data split/train --model_dir models
python3 src/evaluate.py --model_dir models --test_data split/test --metrics_out metrics --min_f1 0.60
```

Training compares Logistic Regression with an MLPClassifier using StandardScaler; the MLP hidden layers are `(64, 32)`. The complete selected pipeline is saved to `models/model.pkl`, comparison results to `models/comparison.json`, and evaluation results to `metrics/metrics.json`.

## Local scoring and Streamlit

Create a sample request:

```bash
python3 src/make_request.py data/process --out request.json
```

Run the local application:

```bash
MODE=local python3 -m streamlit run app/app.py
```

The application is named **Power Panthers Stoppage Predictor**. It supports single-window inputs, CSV batch upload, validation, status, probabilities, and visualizations. It clearly identifies the data as SYNTHETIC.

## Results from the approved run

| Model | Validation recall | Validation F1 |
|---|---:|---:|
| Logistic Regression | 1.0000 | 0.9231 |
| MLPClassifier | 0.0000 | 0.0000 |

Logistic Regression was selected because recall is prioritised, with F1 as the tie-breaker.

Test holdout results:

| Metric | Result |
|---|---:|
| Accuracy | 0.9896 |
| Precision | 0.8000 |
| Recall | 1.0000 |
| F1 | 0.8889 |
| Event-level recall | 0.8571 |
| Mean alert lead time | 1.5 hours |

Confusion matrix, label order `[0, 1]`:

```text
[[273, 3],
 [  0, 12]]
```

The F1 quality gate of `0.60` **PASSED** (`0.8889 >= 0.60`). These strong results are synthetic workflow validation only, not evidence of production performance.

## Limitations

- The data is entirely SYNTHETIC and represents one simulated asset.
- Operations must confirm units, ranges, sampling, and safe operating limits before real-world use.
- The 30-event scenario approximates, but cannot exactly equal, a 75% / 25% event split.
- Deliberately distinguishable synthetic precursor values may make performance optimistic.
- The evaluation is one chronological holdout, not production monitoring or deployment validation.
- Business targets and availability calculations remain to be agreed with operations.
- The `score.py` smoke test produced valid predictions and probabilities, but its Python process did not terminate before the shell timeout on this VM. This is an execution-environment issue, not invalid scoring output.