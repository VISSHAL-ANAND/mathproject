# Technical Requirements
Python 3.10–3.13; pandas, NumPy, scikit-learn, Streamlit, Matplotlib, joblib, pytest.

Target: response_time_minutes, numeric and non-negative. Supported features: priority, category, channel, hour_submitted, day_of_week, backlog, agent_experience_years.

Use a scikit-learn Pipeline and ColumnTransformer. Numeric values use median imputation and scaling for linear models; categorical values use most-frequent imputation and one-hot encoding with unknown-category handling. Fit preprocessing on training data only.

Evaluate held-out data using MAE, RMSE, and R-squared. Consider chronological evaluation for real time-ordered logs. Generated artifacts stay local and are ignored by Git.
