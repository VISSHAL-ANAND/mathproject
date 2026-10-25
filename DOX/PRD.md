# Product Requirements

## Product vision
CampusHelp Response Intelligence helps students and support teams understand likely first-response waiting time, prediction uncertainty, and possible service-deadline risk. It supports human decisions; it does not automatically deny or deprioritize service.

## Human problem
People reporting Wi-Fi, account, library, hostel-maintenance, or other support issues often have no useful estimate of when help will arrive or whether the request may miss its service deadline.

## Functional requirements
- Generate reproducible synthetic demo data or accept a support-ticket CSV.
- Normalize supported aliases: `first_response_min` for the target, `queue` for category, and `opened_at` for derived hour/day features and chronological splitting.
- Disclose rows excluded due to missing or invalid response targets.
- Compare Linear Regression, Ridge Regression, and Random Forest against a training-mean baseline.
- Select the highlighted model using validation MAE, never the final test score.
- Report held-out test MAE, RMSE, and R².
- Show mean, median, standard deviation, 90th percentile, and a bootstrap 95% confidence interval for the dataset mean.
- Use a separate calibration split for prediction intervals and empirical deadline-risk estimates.
- Show when the requested interval coverage cannot be supported by the available calibration sample.
- Show wait-time breakdowns by priority/category and a trend for timestamped data.
- Provide an interactive prediction form and a cautious backlog what-if scenario.
- Clearly distinguish synthetic demo results from real-world evidence.

## Statistical design
- Default split is approximately 60% train, 10% validation, 10% calibration, and 20% test.
- Use a chronological split when all usable rows have valid `opened_at`; otherwise use a reproducible random split.
- Fit initial models on training rows; select using validation MAE; refit candidates on train + validation.
- Use calibration residuals for uncertainty estimates. Reserve the final test set for final metrics and plots.
- Include a training-mean baseline so model complexity must demonstrate added value.
- A finite split-conformal interval at 95% coverage requires at least 19 calibration residuals.
- Explain that time drift and non-representative calibration data can weaken coverage and probability estimates.

## Data and safety
- Use only predictors known at the time of prediction.
- A missing response time is not a zero-minute response. If missing values represent unanswered tickets, disclose that excluding them can bias results toward answered tickets.
- Avoid uploading personal or confidential data.
- Audit errors by priority/category before operational use.
- Treat backlog what-if results as model-based scenarios, not causal estimates.
- Never claim real-world accuracy using synthetic data.

## Acceptance criteria
- Automated tests cover data generation, validation, CSV aliases, model selection, baseline, split strategy, uncertainty, and deadline-risk functions.
- CI checks syntax and runs tests on supported Python versions.
- The UI shows data-quality exclusions, split method, model metrics, uncertainty limitations, and synthetic-data warnings.
