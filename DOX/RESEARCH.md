# Research Notes and Reference Strategy

## Research question
Can support-request attributes predict the elapsed time between submission and first response, and how do linear regression, regularized linear regression, and a tree-ensemble regressor compare?

## Target definition
response_time_minutes is elapsed minutes until the first response. It is not ticket resolution time. Dataset documentation must confirm that the source field really represents first response and explain how timestamps, business hours, and missing responses are handled.

## Model concepts
- Simple least-squares regression: choose intercept and slope to minimize the sum of squared residuals.
- Multiple linear regression: estimates a linear combination of input features.
- Ridge regression: adds an L2 penalty to discourage overly large coefficients.
- Random Forest regression: averages predictions from an ensemble of decision trees and can capture nonlinear relationships.

## Evaluation plan
Use the same held-out test rows for each model. Report MAE in minutes, RMSE in minutes, and R-squared. Document split strategy, random seed, dataset size, target distribution, feature processing, and package versions. Compare against a simple baseline such as predicting the training-set mean. For chronological ticket logs, prefer a time-based split over a random split.

## Initial public references
1. Scikit-learn linear models: https://scikit-learn.org/stable/modules/linear_model.html
2. Scikit-learn model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html
3. Scikit-learn preprocessing and pipelines: https://scikit-learn.org/stable/modules/compose.html
4. James, Witten, Hastie, Tibshirani, Taylor, *An Introduction to Statistical Learning*: https://www.statlearning.com/
5. UCI Machine Learning Repository: https://archive.ics.uci.edu/

## Dataset research checklist
- Is there a clear first-response timestamp or duration?
- Is redistribution permitted by the dataset license?
- Are there personal, confidential, or proprietary records?
- Do the predictors exist at the moment a prediction would be made?
- Are missing or unanswered tickets handled consistently?
- Are rows time-dependent, requiring chronological evaluation?

## Integrity note
The included demo generator creates artificial values using chosen effects and random noise. Its scores only test software behavior; they do not establish that the predictors work on real support operations. The real-data experiment remains a task.
