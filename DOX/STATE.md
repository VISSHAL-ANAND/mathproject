# Project State

## Current product
CampusHelp Response Intelligence predicts first-response time and combines regression, descriptive statistics, uncertainty estimates, and deadline-risk analysis in a white-and-blue Streamlit dashboard.

## Implemented
- Linear Regression, Ridge Regression, Random Forest, and training-mean baseline.
- Select the highlighted model using validation MAE, keeping the final test set out of model selection.
- Approximate 60/10/10/20 train/validation/calibration/test split.
- Chronological split for timestamped ticket data; reproducible random split otherwise.
- CSV aliases for `first_response_min`, `queue`, and `opened_at`.
- Disclosure of rows excluded for missing/invalid response targets.
- Mean, median, standard deviation, 90th percentile, and bootstrap confidence interval for the mean.
- Calibration-based prediction intervals and empirical deadline-risk estimates.
- Explicit handling when calibration data is insufficient for requested interval coverage.
- Priority/category breakdowns, timestamp trend, backlog what-if scenario, and human-oversight language.
- README, engineering documentation, Python syntax CI check, core tests, and Streamlit smoke test.

## Data integrity
The built-in and optional public sample datasets are synthetic. They are useful for development and demonstrations only. A validated real campus-support dataset has not been obtained; do not claim real-world performance.

## Verification
Treat the latest GitHub Actions run for the current `main` commit as the source of truth for test status. CI passing does not replace visual browser testing or real-data validation.

## Next steps
1. Confirm CI on the current commit.
2. Run the dashboard in a browser and review the end-to-end prediction flow.
3. Validate on authorized, representative campus/help-desk data when available.
4. Check errors by priority/category and evaluate on a later time period.
5. Explore survival methods for tickets that have not yet received a response.
