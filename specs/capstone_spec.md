# Power Panthers Version 2 Capstone Specification

**Status:** Phase 1 approved-spec implementation draft  
**Data status:** **SYNTHETIC**. No production or Azure data is used or required.

## 1. Business goal

Unplanned stoppages have increased from **7 to 15 per month**, while
availability has fallen from **94.0% to 90.7%**. Most stops show a detectable
sensor precursor, creating an opportunity for earlier maintenance intervention.

The business goal is to reduce the monthly count of unplanned stoppages from
the current **15/month** and recover availability toward the earlier
**approximately 94% baseline**. The exact numeric business target is to be
agreed with operations; this specification does not invent that target.

## 2. ML goal

Using rolling two-hour vibration, temperature, current, and load trends, detect
an abnormal operating window ahead of a likely stoppage so that maintenance
can be alerted with roughly **two hours of lead time**.

The model is expected to catch the majority, but not all, of the unplanned
stops because approximately **one quarter show no prior signal change**. Recall
is prioritised because missing a real precursor is the costlier error.

## 3. ML problem type

This is an **anomaly-detection** problem on time-windowed sensor data. The
prediction unit is an operating window. A window is positive when it represents
an abnormal operating condition within the two-hour lead-time period before a
stop; otherwise it is negative for evaluation purposes.

The model must operate without using information from after the prediction
window or from the stoppage itself. The stoppage event is used for retrospective
labelling and evaluation, not as an input feature.

## 4. Target

**Target:** Abnormal operating window within a **2-hour lead time before a
stop**.

For a time-indexed evaluation record, the target is positive when the record's
operating window is an abnormal precursor window occurring in the two hours
before an unplanned stoppage. It is negative for normal operating windows and
for windows outside the defined precursor interval. The label definition must
be applied chronologically so that future sensor values and future stop
information cannot leak into model inputs.

## 5. Features

The approved input features are rolling trends over the **prior two hours**:

| Feature | Definition | Simulation unit and realistic range |
|---|---|---|
| Rolling vibration trend | Change and/or fitted trend of vibration over the prior two hours | Vibration represented as RMS velocity in **mm/s**; simulated operating values approximately **0.5–12 mm/s** |
| Rolling temperature trend | Change and/or fitted trend of equipment temperature over the prior two hours | **°C**; simulated operating values approximately **40–95 °C** |
| Rolling current trend | Change and/or fitted trend of electrical current over the prior two hours | **A**; simulated operating values approximately **20–120 A** |
| Rolling load trend | Change and/or fitted trend of equipment load over the prior two hours | **% of rated load**; simulated operating values approximately **20–100%** |

The ranges above are simulation assumptions for a plausible industrial asset,
not claims about the asset's actual instrumentation. Before operational use,
operations must confirm sensor units, normal ranges, sampling frequency, and
safe operating limits. The model feature set must remain limited to the four
approved sensor trends; no unapproved business, maintenance, or post-event
features may be added.

## 6. Synthetic-data-generation methodology

The dataset will be explicitly labelled **SYNTHETIC** and generated locally
for development and evaluation. It will be designed to match the approved
Day-1 evidence rather than to represent measured production data.

The generator will:

1. Create a chronological time series of operating windows for a single
   representative asset (or clearly identified repeated asset simulations),
   with the sampling interval and time span recorded in the dataset metadata.
2. Simulate normal operation using plausible values within the feature ranges,
   with realistic autocorrelation, gradual load variation, sensor noise, and
   correlated changes between load, current, temperature, and vibration.
3. Inject unplanned stoppage events at a rate consistent with the evidence of
   approximately **15 per month** in the current condition. The event rate is
   a scenario parameter, not a newly approved business target.
4. Generate precursor scenarios for approximately **three quarters** of
   stoppages. In those scenarios, one or more approved sensor trends gradually
   move away from the normal pattern during the two hours before the stop.
5. Generate approximately **one quarter** of stoppages without a prior signal
   change, so that the data reflects the stated limitation that not every stop
   is detectable from these features.
6. Mark the abnormal precursor operating windows as the positive target and
   retain the stop timestamp separately for auditability and lead-time
   evaluation. The stop timestamp must not be included as a model feature.
7. Include enough normal windows to reflect an imbalanced anomaly-detection
   setting. The generation configuration, random seed, feature units, event
   counts, precursor proportion, and any clipping or missing-value handling
   must be recorded with the dataset.

The synthetic generator must not imply that the resulting data proves the
business case. Synthetic results are for validating the end-to-end method and
acceptance checks only.

## 7. Relevant Day-1 evidence

The specification is grounded in the following evidence from the approved
problem brief:

- Unplanned stoppages increased from **7/month to 15/month**.
- Availability decreased from **94.0% to 90.7%**.
- Most stops show a detectable sensor precursor.
- Approximately **25%** of stops show no prior signal change.
- The requested intervention horizon is roughly **two hours**.
- The business success direction is fewer monthly unplanned stops and
  availability recovering toward approximately **94%**.
- The model success measures are **recall and F1**, with recall prioritised.
- The approved track is **Table (Process)**.
- The data used for this work is **SYNTHETIC**, generated to match the Day-1
  evidence.

## 8. Evaluation metrics

### Model metrics

Report the following on a strictly chronological holdout that is not used for
model fitting or threshold selection:

- **Recall** for abnormal precursor windows: the primary model metric because
  missed real precursors are the costlier error.
- **F1 score** for abnormal precursor windows: the required balanced metric
  across precision and recall.
- Precision, confusion matrix, and the positive-window count as supporting
  diagnostics.
- **Event-level recall**: the fraction of stoppage events with at least one
  valid alert in the preceding two-hour window.
- Alert lead time, including the distribution and whether alerts occur before
  the stop, to verify the roughly two-hour operational horizon.

Metrics must be reported with the decision threshold and the synthetic-data
configuration used. Window-level metrics must not be presented as proof of
business impact.

### Business metrics

Track monthly unplanned-stop count and availability. The intended direction is
to reduce stoppages from the current **15/month** and recover availability
toward the earlier **approximately 94% baseline**. The exact numeric business
success target remains to be agreed with operations.

## 9. Acceptance criteria

The Phase 1 specification is accepted when:

1. The implementation identifies the dataset as **SYNTHETIC** and uses no
   Azure or Azure ML dependency.
2. The generated data contains only the four approved two-hour rolling sensor
   trends as model inputs, with units, ranges, timestamps, and target labels
   documented.
3. The target is defined exactly as an abnormal operating window within a
   two-hour lead time before an unplanned stop, with no future-data leakage.
4. The synthetic scenario documents and reflects the Day-1 evidence: the
   current 15/month stop context, the 94.0% to 90.7% availability change,
   detectable precursors for most events, and approximately 25% undetectable
   events.
5. Evaluation uses a chronological holdout and reports recall and F1 as the
   required model metrics, with recall treated as the priority metric and
   event-level lead-time diagnostics included.
6. Business metrics are reported as monitoring outcomes, while the exact
   numeric business target is left for agreement with operations.
7. No training code, generated dataset, or Streamlit application is part of
   this Phase 1 deliverable. Those artifacts require review and approval of
   this specification first.
