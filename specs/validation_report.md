# Power Panthers Validation Report

## Scope and status

This report records evidence from Phases 1–4 against the acceptance criteria in `specs/capstone_spec.md`.

**Data status: PASS — SYNTHETIC.** The dataset is generated locally with seed 42. It is not production data and results must not be interpreted as proof of business impact.

**Execution status: PASS — local only.** The pipeline, tests, model, metrics, and Streamlit application run locally on the Linux VM. No Azure, Azure ML, or cloud tracking dependency is used.

## Evidence summary

- Dataset: 1,440 rows × 9 columns; hourly; 60 days; seed 42.
- Stoppage events: 30, approximately 15 per month.
- Detectable events: 23/30 = 76.7%; no-precursor events: 7/30 = 23.3%.
- Target distribution: 1,394 negative windows and 46 positive windows.
- `pytest`: **4 passed**.
- Chronological split: 1,152 training rows and 288 test rows.

Validation comparison:

| Model | Recall | F1 | Result |
|---|---:|---:|---|
| Logistic Regression | 1.0000 | 0.9231 | Selected |
| MLPClassifier `(64, 32)` | 0.0000 | 0.0000 | Not selected |

The selected Logistic Regression pipeline includes `StandardScaler` and was saved to `models/model.pkl`. Selection prioritised recall and used F1 as the tie-breaker.

Chronological test results:

| Metric | Result |
|---|---:|
| Accuracy | 0.9896 |
| Precision | 0.8000 |
| Recall | 1.0000 |
| F1 | 0.8889 |
| Positive-window count | 12 |
| Event-level recall | 0.8571 |
| Mean alert lead time | 1.5 hours |

Confusion matrix, label order `[0, 1]`:

```text
[[273, 3],
 [  0, 12]]
```

Quality gate: **PASS**. Required F1 was 0.60; observed F1 was 0.8889. Evaluation with `--min_f1 0.60` exited successfully with code 0.

## Acceptance criteria

### Criterion 1 — Synthetic data and local implementation

**PASS**

Evidence: `src/generate_data.py` uses seed 42 and writes locally; metadata identifies `SYNTHETIC`; `src/tab_utils.py`, training, evaluation, scoring, and Streamlit use local files; no Azure, Azure ML, or cloud tracking dependency is present.

### Criterion 2 — Four approved features, units, ranges, timestamps, and labels

**PASS**

Evidence: `src/config.json` defines exactly the four approved features. The CSV contains timestamps, those features, and `abnormal_window`. The metadata sidecar records units, sampling interval, time span, clipping, and missing-value handling. Audit columns are excluded from the model feature list. `tests/test_data.py` verifies required and approved columns.

### Criterion 3 — Correct two-hour target and no future-data leakage

**PASS**

Evidence: the generator labels the two preceding hourly windows for detectable events; configuration defines the two-hour abnormal precursor target; stop and audit fields are excluded from `approved_features`; `prep.py` validates chronological ordering and performs a chronological split; training and evaluation use chronological validation and test holdouts.

### Criterion 4 — Day-1 evidence and detectable/no-precursor scenario

**PASS**

Evidence: 30 events in 60 days represent approximately 15 per month; detectable events are 23/30 and no-precursor events are 7/30, approximating three quarters and one quarter. The approved specifications record the 7-to-15 stoppage context and availability change from 94.0% to 90.7%. The implementation does not claim synthetic results prove those outcomes.

### Criterion 5 — Chronological evaluation, recall/F1, and event diagnostics

**PASS**

Evidence: training and test data are chronological; validation reports recall and F1 for both models; test evaluation reports recall, F1, precision, accuracy, positive-window count, and confusion matrix; event-level recall is 0.8571; lead-time diagnostics contain 12 valid alerts at 1 or 2 hours with a 1.5-hour mean; threshold is 0.5.

### Criterion 6 — Business metrics treated as monitoring outcomes

**PASS**

Evidence: README and this report describe reducing stoppages and recovering availability toward approximately 94% as business direction, without inventing an exact target. Model metrics are explicitly synthetic workflow validation and not proof of business impact.

### Criterion 7 — Phase-gated implementation delivery

**PASS**

Evidence: training code, generated data, and Streamlit were delivered only after Phase 1 approval. Phases 2–4 delivered the approved generator, pipeline, model artifacts, tests, and local application. Phase 5 adds documentation only and no cloud deployment.

## Phase 4 local application evidence

**PASS:** `app/app.py` is titled **Power Panthers Stoppage Predictor**, loads `models/model.pkl` under `MODE=local`, provides all four feature inputs, supports CSV batch scoring, validates columns and values, shows status/probability, includes visualizations, and identifies the data as SYNTHETIC. Headless startup succeeded on `127.0.0.1:8501`; `/_stcore/health` returned `ok`.

**PASS — make_request.py:** Generated `/tmp/phase4_request.json` with five records containing all four approved features.

**FUNCTIONAL OUTPUT PASS; PROCESS TERMINATION CAVEAT — score.py:** Loaded the model and returned five valid predictions and five probabilities, all in `[0, 1]`:

```text
predictions = [0, 0, 0, 0, 0]
probabilities = [0.3334, 0.000417, 0.001771, 0.000095, 0.004545]
```

The Python process did not terminate before the shell timeout on this VM and the wrapper reported exit code 124 after valid output was produced. This is recorded honestly as an environment/process-teardown issue, not as invalid scoring output or a clean process-level pass.

## Suspicious-performance assessment

Performance is strong but not perfect: there were three false positives and event-level recall was 85.71%. However, the generator deliberately injects precursor signals distinguishable from normal values, so performance may be optimistic and must not be generalized to production.

## Limitations and next steps

- Confirm sensor units, ranges, sampling, and safe limits with operations.
- Replace synthetic data with representative governed production data before operational use.
- Evaluate across multiple chronological periods and assets.
- Define the agreed business target and measure monthly stoppages and availability separately from model metrics.
- Investigate the `score.py` process teardown behavior before using it as a production integration interface.