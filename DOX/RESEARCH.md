# Research Notes and Reference Strategy

## Research question
Can support-ticket attributes predict time to first response, and can regression residuals provide useful uncertainty intervals and estimates of service-deadline breach risk?

## Human use case
Campus support is the initial framing: students reporting Wi-Fi, account, hostel maintenance, or facility issues need an understandable estimate of when help may arrive. The same design can be adapted to other help desks if the target and data are suitable.

## Target definition
`response_time_minutes` is elapsed minutes from submission to first human response. It is not ticket-resolution time. A real dataset must document timestamps, time zones, business-hour treatment, unanswered tickets, duplicates, and censoring. If no reliable first-response target exists, do not relabel resolution time as first response.

## Probability and statistics concepts demonstrated
- Descriptive statistics: mean, median, sample standard deviation, and 90th percentile.
- Bootstrap confidence interval: resample observed durations to estimate uncertainty in the population mean.
- Regression: Linear Regression, Ridge Regression, and Random Forest.
- Error analysis: MAE, RMSE, R², residual distribution, and actual-versus-predicted plot.
- Split-conformal prediction interval: use absolute residuals from a calibration split to estimate an interval around a new prediction.
- Empirical SLA-breach probability: apply the calibration residual distribution to the point estimate and calculate the fraction that exceeds the selected deadline.

## Evaluation protocol
The current random split is approximately 60% train, 20% calibration, and 20% test. Fit all preprocessing on training rows only; use calibration residuals only for uncertainty and SLA calculations; use test rows only for model metrics. For real chronological ticket logs, use a time-based evaluation to better represent future predictions. If multiple rows come from the same requester or repeated incident, consider grouped splitting to avoid leakage.

## Important limitations
The split-conformal interval relies on assumptions such as exchangeability between calibration and future cases; it may under-cover when operations change. The SLA estimate is an empirical residual-based estimate and should not be called a calibrated probability until validated on later real data. The bootstrap interval for the mean is not an individual prediction interval. Synthetic demonstration data can only test the software flow and concepts, not prove predictive value.

## Reference sources
1. Scikit-learn linear models: https://scikit-learn.org/stable/modules/linear_model.html
2. Scikit-learn model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html
3. Scikit-learn preprocessing and pipelines: https://scikit-learn.org/stable/modules/compose.html
4. Scikit-learn prediction-interval example: https://scikit-learn.org/stable/auto_examples/inspection/plot_prediction_intervals.html
5. James, Witten, Hastie, Tibshirani, Taylor, *An Introduction to Statistical Learning*: https://www.statlearning.com/
6. UCI Machine Learning Repository: https://archive.ics.uci.edu/

## Real-data checklist (not yet completed)
- Find a documented dataset with a valid first-response duration or timestamps.
- Verify license and redistribution conditions.
- Exclude personal, confidential, and proprietary information.
- Confirm every feature is known at the time of prediction.
- Document missing/unanswered tickets and service-hour rules.
- Evaluate on a later time period, compare to a simple baseline, and check underprediction for urgent tickets.

## Integrity note
The built-in data generator creates artificial values with chosen effects and random noise. Its metrics must never be reported as real-world evidence. A suitable real dataset has not yet been selected or validated.
