# Product Requirements

## Product vision
CampusHelp Response Intelligence helps students and campus support teams make better decisions when a request is waiting for a first response. The system estimates waiting time and deadline risk; it does not automatically deny service or make final priority decisions.

## Human problem
Students reporting Wi-Fi, account, hostel-maintenance, or facility issues may have no useful estimate of when someone will respond. Support teams also need a transparent way to identify requests that may miss a service deadline.

## Functional requirements
- Load a synthetic demonstration dataset or a user-supplied CSV.
- Validate the target and supported predictor columns.
- Train and compare Linear Regression, Ridge Regression, and Random Forest.
- Report held-out MAE, RMSE, and R².
- Show mean, median, standard deviation, 90th percentile, and a bootstrap 95% confidence interval for the mean.
- Estimate an individual prediction interval from a separate calibration split.
- Estimate the empirical probability of exceeding a user-selected SLA deadline.
- Visualize actual-vs-predicted values and residuals.
- Provide an interactive prediction form and a CLI training report.
- Clearly distinguish synthetic demo results from real evidence.

## Statistical design
- Split data into approximately 60% train, 20% calibration, and 20% test.
- Fit preprocessing and models on training data only.
- Use calibration residuals for split-conformal interval width and empirical SLA breach probability.
- Keep the final test set separate from calibration.
- Explain that interval coverage and probability reliability depend on exchangeability, sufficient representative data, and stable future conditions.

## Safety and fairness
- A risk score is decision support, not an automatic service-denial rule.
- Avoid sensitive personal attributes.
- Before deployment, check performance across relevant request categories and priority groups, and investigate systematic underprediction for urgent requests.
- Never claim real-world accuracy from synthetic data.

## Acceptance criteria
- The app runs from documented setup instructions.
- Automated tests cover generation, validation, evaluation, statistics, intervals, and SLA probability.
- The app shows model metrics, descriptive statistics, interval estimates, and deadline-risk estimates.
- Research claims are limited to evidence from a documented, suitable dataset.
