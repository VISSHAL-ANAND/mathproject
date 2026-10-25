# Product Requirements

## Product vision
CampusHelp Response Intelligence helps students and campus support teams understand likely first-response waiting time, uncertainty, and potential service-deadline risk. It supports human decisions and does not automatically deny or deprioritize service.

## Human problem
People reporting Wi-Fi, account, library, hostel-maintenance, or other support problems often do not know how long they may wait or whether a response deadline may be missed.

## Functional requirements
- Generate a reproducible synthetic demonstration dataset or accept a user-supplied ticket CSV.
- Normalize supported aliases: `first_response_min` for the target, `queue` for category, and `opened_at` for hour/day features and chronological evaluation.
- Count and disclose rows excluded for missing or invalid targets.
- Compare Linear Regression, Ridge Regression, and Random Forest against a training-mean baseline.
- Select the highlighted model with validation MAE, not the final test score.
- Show test MAE, RMSE, and R² for final evaluation.
- Show mean, median, standard deviation, 90th percentile, and a bootstrap 95% confidence interval for the dataset mean.
- Use a separate calibration split for split-conformal prediction intervals and empirical deadline-risk estimates.
- Explain when the requested prediction-interval coverage needs more calibration rows.
- Visualize model errors and support an interactive single-request prediction.
- Clearly separate synthetic demo metrics from real-world evidence.

## Statistical design
- Default split is approximately 60% training, 10% validation, 10% calibration, and 20% test.
- Use a chronological split when every usable row has a valid `opened_at` timestamp; otherwise use a reproducible random split.
- Select a model with validation MAE, then refit candidate models on training + validation data.
- Use calibration residuals only for interval/risk estimation and keep test rows out of model selection and calibration.
- Include a training-mean baseline so complex models must demonstrate measurable value.
- Explain that prediction intervals and risk estimates rely on calibration data representing future cases; time drift can weaken these assumptions.
- Require enough data for the chosen interval coverage; a finite 95% split-conformal interval needs at least 19 calibration residuals.

## Data and safety
- Use only predictors known when the prediction is made.
- If no first-response time exists, a row is not a zero-minute response; it is excluded from regression, and this exclusion must be disclosed.
- Avoid uploading personal or confidential data.
- Review error by priority and issue category before any operational use.
- Never claim real-world accuracy using synthetic data.

## Acceptance criteria
- Automated tests cover data generation, CSV aliases, target validation, baseline/model selection, split strategy, statistics, prediction intervals, and deadline-risk estimates.
- CI checks Python syntax and runs tests on supported Python versions.
- UI displays data-quality exclusions, split method, model metrics, interval limitations, and synthetic-data warnings.
