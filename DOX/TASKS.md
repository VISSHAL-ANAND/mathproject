# Tasks

## Implemented
- [x] Define a human-centered campus support use case and white/blue dashboard.
- [x] Add regression models and a training-mean baseline.
- [x] Select the highlighted model on validation MAE, not test MAE.
- [x] Separate training, validation, calibration, and test data.
- [x] Use chronological evaluation when valid `opened_at` timestamps exist.
- [x] Normalize common CSV aliases and disclose rows omitted for missing/invalid targets.
- [x] Add descriptive statistics, a bootstrap confidence interval, prediction intervals, and deadline-risk estimates.
- [x] Validate minimum calibration sample sizes for high-coverage intervals.
- [x] Add priority/category breakdowns, timestamp trend, and a cautious backlog what-if scenario.
- [x] Document the optional synthetic CC0 sample and its limitations.
- [x] Add a Streamlit app smoke test to CI.

## Remaining
- [ ] Confirm the latest GitHub Actions run.
- [ ] Browser-test the dashboard and prediction flow.
- [ ] Validate on authorized, representative real campus/help-desk data.
- [ ] Review errors by priority/category and evaluate on a later time period.
- [ ] Investigate unanswered tickets explicitly before claiming overall waiting-time performance.
- [ ] Prepare the final academic report and viva notes using actual, documented results.
