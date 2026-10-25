# Research Notes and Methodology

## Research question
Can features known when a support request is submitted predict time to first response, and can calibration residuals provide useful prediction intervals and estimates of service-deadline risk?

## Human use case
CampusHelp targets students reporting Wi-Fi, account, library, or hostel-maintenance issues. The intended value is to estimate the likely wait, show uncertainty, and help a human support team notice possible deadline risk without automating service priority decisions.

## Target definition
`response_time_minutes` is elapsed time from submission to first human response; it is not resolution time. The app also recognizes `first_response_min`. Missing target rows are excluded from regression. If they mean tickets never answered, the evaluation describes answered tickets only and may have survivorship bias.

## Optional sample dataset
The [Datanemics Customer Support Tickets CSV](https://datanemics.com/datasets/data/support-tickets.csv) is described by its publisher as a 4,000-row synthetic dataset under CC0 public-domain terms. It includes `first_response_min`, `opened_at`, `queue`, `priority`, and `channel`; it is useful for testing CSV import and chronological splitting, not for proving real-world performance. The publisher documents missing first-response times for never-answered tickets: https://datanemics.com/datasets/support-tickets/

## Evaluation protocol
At the default 20% test size, the approximate split is 60% train, 10% validation, 10% calibration, and 20% test.
1. Fit candidate models on training rows.
2. Select the highlighted model using validation MAE.
3. Refit candidates using training + validation rows.
4. Use calibration residuals for conformal interval widths and empirical deadline-risk estimates.
5. Use the held-out test set for final test metrics and error plots only.

If all usable rows have valid `opened_at`, split chronologically so the latest records form the test period. Otherwise, use a reproducible random split. Chronological evaluation is more representative of forecasting future tickets, but distribution shifts can weaken uncertainty calibration.

## Methods
- Descriptive statistics: mean, median, sample standard deviation, and 90th percentile.
- Bootstrap percentile confidence interval for the dataset mean; not an individual prediction interval.
- Linear Regression, Ridge Regression, Random Forest Regression, and a training-mean baseline.
- MAE, RMSE, and R² for held-out test evaluation.
- Split-conformal prediction interval from absolute calibration residuals.
- Empirical deadline-risk estimate from the calibration residual distribution.
- Priority/category breakdowns to help investigate where observed waits are higher.

## Limitations
A finite split-conformal interval at 90% coverage needs at least 9 calibration points; this app uses a minimum of 10. A finite interval at 95% needs at least 19. If the sample is too small, the app explains that the interval is unavailable rather than showing a false finite range. The SLA estimate is empirical, not a validated real-world probability. It relies on calibration errors representing future cases.

Unanswered tickets are not zero-minute waits. Removing them can bias a model toward tickets that receive a response. Production use may require time-to-event/survival methods and a carefully defined observation window.

## Reference sources
1. Scikit-learn linear models: https://scikit-learn.org/stable/modules/linear_model.html
2. Scikit-learn model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html
3. Scikit-learn preprocessing and pipelines: https://scikit-learn.org/stable/modules/compose.html
4. Scikit-learn prediction interval example: https://scikit-learn.org/stable/auto_examples/inspection/plot_prediction_intervals.html
5. An Introduction to Statistical Learning: https://www.statlearning.com/
6. Datanemics synthetic CC0 ticket dataset: https://datanemics.com/datasets/support-tickets/

## Integrity note
The built-in and optional sample datasets are synthetic. Their results demonstrate the workflow, not real-world predictive value. Real operational use requires authorized data, a documented target, time-aware evaluation, privacy review, and subgroup error analysis.
