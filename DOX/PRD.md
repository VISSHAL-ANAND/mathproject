# Product Requirements
## Goals
- Demonstrate regression mathematics with runnable code.
- Predict first response time in minutes.
- Compare models on the same held-out test split.
- Explain prediction errors and limitations.

## Functional requirements
- Load synthetic demo data or a CSV upload.
- Validate target and supported feature columns.
- Train Linear Regression, Ridge, and Random Forest pipelines.
- Report MAE, RMSE, and R-squared.
- Offer an interactive prediction form.
- Show actual-vs-predicted and residual plots.
- Provide CLI training and automated tests.

## Acceptance criteria
- Setup instructions work in supported Python versions.
- Tests cover data generation, validation, and model evaluation.
- UI shows metrics and predictions.
- Documentation labels synthetic data and makes no unsupported accuracy claims.
