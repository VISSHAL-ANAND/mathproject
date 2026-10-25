# Response Time Prediction Using Regression Models

A college mathematics project predicting the time from a support request's submission to its first response, measured in minutes.

**Data honesty:** The built-in dataset is synthetic and for demonstration only. It is not real customer-service data. Do not report demo results as evidence of real-world performance.

## Quick start
Python 3.10–3.13 recommended.

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    streamlit run app.py

Windows PowerShell activation: .venv\\Scripts\\Activate.ps1

Run tests with `pytest -q`; train demo models with `python -m src.train`.

## Models and mathematics
- Simple linear regression: y-hat = b0 + b1 x.
- Multiple linear regression: y-hat = beta0 + sum(beta_j x_j).
- Ridge regression: linear regression with L2 regularization.
- Random Forest regression: a nonlinear ensemble baseline.

Compare held-out MAE, RMSE, and R-squared. MAE is average absolute error; RMSE penalizes large errors more strongly; R-squared compares the model against predicting the test-set mean. Negative R-squared is possible.

## Input data
A CSV needs a numeric, non-negative response_time_minutes column and at least one supported predictor: priority, category, channel, hour_submitted, day_of_week, backlog, agent_experience_years. Extra columns are ignored. Do not upload private customer data to a public repository.

## Research references
- Scikit-learn linear models: https://scikit-learn.org/stable/modules/linear_model.html
- Model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html
- Pipelines and preprocessing: https://scikit-learn.org/stable/modules/compose.html
- Introduction to Statistical Learning: https://www.statlearning.com/
- UCI Machine Learning Repository: https://archive.ics.uci.edu/

For methodology, decisions, architecture, and setup, see the DOX directory. Model quality must be measured on a documented, suitable dataset before making research claims.
