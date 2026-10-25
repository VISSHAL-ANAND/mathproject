# Feature 001: Response-Time Prediction
Input features: priority, category, channel, hour_submitted, day_of_week, backlog, agent_experience_years.

Output: estimated first-response time in minutes.

Reject datasets without the target, without supported features, or with fewer than 30 usable target rows. Drop rows with missing or invalid target values. Impute missing predictors inside the training pipeline. Ignore unknown extra columns. Clearly warn when synthetic demo data is used.
