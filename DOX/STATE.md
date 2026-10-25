# Project State

## Current direction
CampusHelp Response Intelligence is a human-centered support-ticket project. It predicts time to first response and demonstrates regression, descriptive statistics, bootstrap confidence intervals, split-conformal prediction intervals, and empirical SLA-breach probability.

## Implemented
- Streamlit dashboard and interactive prediction form.
- Linear Regression, Ridge Regression, and Random Forest.
- MAE, RMSE, R², residual and actual-vs-predicted plots.
- Mean, median, standard deviation, 90th percentile, and bootstrap 95% confidence interval for the mean.
- Separate 60/20/20 train/calibration/test design.
- Calibration-residual prediction intervals and estimated service-deadline breach probability.
- Automated tests and GitHub Actions workflow.
- README and DOX documentation updated to describe limitations and data integrity.

## Data integrity
The built-in dataset remains synthetic. It does not establish real-world predictive accuracy. A suitable, license-checked dataset with a defensible first-response target has not yet been selected.

## Verification status
GitHub Actions runs were still in progress at the last check. Local test execution was previously blocked because this environment could not resolve GitHub. Do not claim the tests pass until the latest workflow run completes successfully.

## Next steps
1. Confirm the latest CI run and fix any failures.
2. Test the app and training CLI in a working Python environment.
3. Find and verify a suitable dataset, then document its source, license, and target semantics.
4. Run a chronological or otherwise leakage-aware real-data experiment and report honest results.
5. Add subgroup error checks and a final college report.
