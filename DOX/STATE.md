# Project State

## Current product
CampusHelp Response Intelligence predicts first-response time and demonstrates regression, descriptive statistics, bootstrap confidence intervals, split-conformal prediction intervals, and empirical deadline-risk estimates.

## Implemented
- White-and-blue Streamlit dashboard and prediction form.
- Linear Regression, Ridge Regression, Random Forest, and training-mean baseline.
- Validation-based model selection, with the final test set kept out of model selection.
- Approximate 60/10/10/20 train/validation/calibration/test split.
- Chronological split when valid `opened_at` timestamps exist; reproducible random split otherwise.
- CSV aliases for `first_response_min`, `queue`, and `opened_at`.
- Disclosure of rows excluded for missing/invalid response targets.
- Descriptive statistics, bootstrap CI for the mean, prediction intervals, and empirical deadline-risk estimates.
- More explicit errors when requested conformal coverage needs more calibration rows.
- README, DOX documentation, CI syntax check, and automated tests.

## Data integrity
The built-in dataset and optional public sample are synthetic. They are useful for development and demonstration only. A validated real campus-support dataset has not been obtained; do not claim real-world performance.

## Verification
Run GitHub Actions on the current main-branch commit and record the final result. The automated suite does not replace a browser-based end-to-end review or real-data validation.

## Next steps
1. Confirm CI on the latest commit and fix any failures.
2. Launch and test the Streamlit UI end-to-end.
3. Validate on authorized, appropriately documented real data when available.
4. Review errors by priority/category and assess performance on a later time period.
5. Add queue-capacity what-if simulation only with clear non-causal disclaimers.
