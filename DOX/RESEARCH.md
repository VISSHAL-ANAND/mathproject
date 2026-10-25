# Research Notes and Methodology

## Research question
Can support-ticket attributes predict time to first response, and can a separate calibration sample provide useful prediction intervals and estimates of service-deadline risk?

## Human use case
Students reporting Wi-Fi, account, library, or hostel-maintenance problems often lack an understandable estimate of when help will arrive. The dashboard aims to provide a predicted wait, communicate uncertainty, and help staff notice possible deadline risk without automating service priority decisions.

## Target
`response_time_minutes` is elapsed time from ticket submission to first human response. It is not ticket-resolution time. The app also recognizes `first_response_min` as a target alias. Missing first-response values are excluded from regression; if missing means never answered, the resulting dataset only describes answered tickets and may have survivorship bias.

## Data source for import testing
The [Datanemics Customer Support Tickets CSV](https://datanemics.com/datasets/data/support-tickets.csv) is described by its publisher as a 4,000-row synthetic dataset released under CC0 public-domain terms. It has `first_response_min`, `opened_at`, `queue`, `priority`, and `channel` fields. It is suitable for testing CSV import and chronological splitting, not for establishing real-world performance. The publisher documents blank first-response times for tickets that were never answered: https://datanemics.com/datasets/support-tickets/

## Evaluation protocol
For the default 20% test size, the approximate split is 60% train, 10% validation, 10% calibration, 20% test.
1. Fit preprocessing and candidate models on training rows.
2. Select the highlighted model using validation MAE.
3. Refit models on training + validation rows.
4. Use calibration residuals for uncertainty intervals and empirical deadline-risk estimates.
5. Use the final test set only for reported test metrics and error plots.

If every usable row has a valid `opened_at`, use a chronological split: earlier data for training, then validation and calibration, and the latest period for final testing. Otherwise use reproducible random splitting. Chronological evaluation better matches forecasting future tickets, but temporal drift can reduce the validity of exchangeability-based uncertainty guarantees.

## Probability and statistics concepts
- Mean, median, sample standard deviation, and 90th percentile.
- Bootstrap percentile confidence interval for the dataset mean; this is not a per-ticket prediction interval.
- Linear Regression, Ridge Regression, Random Forest Regression.
- Training-mean baseline to test whether model complexity adds value.
- MAE, RMSE, and R² on held-out test rows.
- Split-conformal prediction interval from absolute calibration residuals.
- Empirical deadline-risk estimate from the calibration residual distribution.

## Finite-sample and data-quality limits
A finite split-conformal interval at 90% coverage requires at least 9 calibration residuals; this project uses a more conservative minimum of 10. At 95% coverage, at least 19 are required. If there are too few, the app explains that the requested interval is unavailable rather than presenting a misleading finite range. Risk estimates are still empirical and require representative future data to assess calibration.

Unanswered tickets are not zero-minute waits. If the selected source has missing response time for unanswered tickets, dropping those rows can bias model evaluation toward answered tickets. A production version should handle unanswered tickets explicitly, potentially using survival/time-to-event methods, before claiming total queue performance.

## Reference sources
1. Scikit-learn linear models: https://scikit-learn.org/stable/modules/linear_model.html
2. Scikit-learn model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html
3. Scikit-learn preprocessing and pipelines: https://scikit-learn.org/stable/modules/compose.html
4. Scikit-learn prediction interval example: https://scikit-learn.org/stable/auto_examples/inspection/plot_prediction_intervals.html
5. James, Witten, Hastie, Tibshirani, Taylor, *An Introduction to Statistical Learning*: https://www.statlearning.com/
6. Datanemics, Customer Support Tickets (synthetic, CC0): https://datanemics.com/datasets/support-tickets/

## Integrity note
Built-in and optional public sample data are synthetic. Their scores demonstrate implementation behavior, not real-world predictive value. Real operational use requires approved data, a documented target, temporal evaluation, privacy review, and error analysis by important groups.
