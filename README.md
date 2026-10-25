# CampusHelp Response Intelligence

A statistics and machine-learning college project aimed at a real human problem: students report campus support issues but often do not know how long help will take or whether the service deadline may be missed.

The app predicts **time to first response**, estimates a **prediction interval**, estimates the **probability of missing a service deadline**, and compares regression models. It can be adapted for campus IT, hostel maintenance, library support, or general help-desk tickets.

**Important data honesty:** the built-in dataset is synthetic. It is useful for demonstrating the software and mathematics, not for proving real-world accuracy. Use approved, anonymized real data before making real operational claims.

## Quick start

Python 3.10–3.13 recommended.

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    streamlit run app.py

Windows PowerShell activation: .venv\\Scripts\\Activate.ps1

Run automated tests with `pytest -q`; generate local demo reports with `python -m src.train`.

## What makes it more than a basic regression project?

- **Regression:** Linear Regression, Ridge Regression, and Random Forest predict response time.
- **Descriptive statistics:** mean, median, sample standard deviation, and 90th percentile.
- **Bootstrap confidence interval:** estimates uncertainty in the dataset's mean response time.
- **Prediction uncertainty:** a split-conformal interval is built from calibration-set residuals, not the final test set.
- **Probability:** estimates the empirical chance that the predicted response exceeds a user-selected service deadline.
- **Model evaluation:** held-out MAE, RMSE, and R² plus actual-vs-predicted and residual plots.
- **Decision support:** flags estimated deadline risk so a human operator can decide whether to prioritize a ticket.

## Data split and interpretation

The default evaluation uses approximately 60% training data, 20% calibration data, and 20% test data. Calibration residuals are kept separate from final test evaluation. The prediction interval's nominal coverage is not a guarantee—especially if future ticket patterns differ from the calibration data. The SLA breach probability is an empirical estimate, not a calibrated real-world probability until tested on representative data.

## CSV format

Required target: `response_time_minutes`, a numeric, non-negative duration until the **first response** (not resolution). Supported predictors: `priority`, `category`, `channel`, `hour_submitted`, `day_of_week`, `backlog`, `agent_experience_years`. At least 30 usable rows are required; 50+ is preferable for the app, and more representative data is better for research. Avoid personal information and do not upload confidential records.

## Research references

- Scikit-learn linear models: https://scikit-learn.org/stable/modules/linear_model.html
- Scikit-learn model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html
- Scikit-learn pipelines and preprocessing: https://scikit-learn.org/stable/modules/compose.html
- Conformal prediction overview: https://scikit-learn.org/stable/auto_examples/inspection/plot_prediction_intervals.html
- Introduction to Statistical Learning: https://www.statlearning.com/
- UCI Machine Learning Repository: https://archive.ics.uci.edu/

See `DOX/` for requirements, architecture, methodology, and a checklist for verifying a real dataset.
