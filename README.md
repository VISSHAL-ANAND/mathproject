# Response Time Prediction Using Regression Models
### CampusHelp Response Intelligence

<p align="center">
  <strong>Predict the wait. Explain the uncertainty. Estimate deadline risk.</strong>
</p>

<p align="center">
  <a href="https://github.com/VISSHAL-ANAND/mathproject/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/VISSHAL-ANAND/mathproject/actions/workflows/tests.yml/badge.svg"></a>
  <img alt="Python 3.10–3.13" src="https://img.shields.io/badge/Python-3.10%E2%80%933.13-2563EB">
  <img alt="Streamlit" src="https://img.shields.io/badge/UI-Streamlit-2563EB">
</p>

CampusHelp Response Intelligence is a statistics and machine-learning project built around a practical human problem: **when someone reports an issue, they usually do not know how long they will wait for a response.**

The application predicts the time until a support request receives its first response, estimates a range for that prediction, and calculates an empirical estimate of the chance of missing a selected service deadline. The initial use case is campus support, such as Wi-Fi, account, library, or hostel-maintenance requests.

> **Important:** The built-in dataset is synthetic. It demonstrates the application and its mathematics; it does not prove real-world accuracy. A suitable real dataset must be validated before operational claims can be made.

## The idea

Many simple prediction tools give one number and stop. Our project brings three pieces of information together so a student or support team can make a more informed decision.

| Output | The question it answers |
|---|---|
| **Predicted response time** | How long might this request wait? |
| **Prediction interval** | How uncertain is that estimate? |
| **Deadline-risk estimate** | How likely is the response time to exceed the chosen deadline? |

### What makes our approach different?

The innovation is the **decision-focused combination of regression, uncertainty estimation, and deadline-risk analysis** in one clear dashboard. Instead of presenting a prediction as a guaranteed fact, the app also shows uncertainty and the estimated chance of a late response.

The project uses established statistical methods; it does not claim to invent a new regression algorithm. Its goal is to make those methods transparent and useful for a real human problem. The result supports human judgment—it does not automatically reject requests or decide who deserves help first.

## Features

- Compare **Linear Regression**, **Ridge Regression**, and **Random Forest Regression**.
- Report **MAE**, **RMSE**, and **R²** on held-out test data.
- Summarize response times using the mean, median, standard deviation, and 90th percentile.
- Calculate a bootstrap 95% confidence interval for the dataset mean.
- Estimate a prediction interval from a separate calibration split.
- Estimate the empirical chance of exceeding a user-selected response deadline.
- Explore actual-versus-predicted and prediction-error charts.
- Upload a CSV or use reproducible synthetic demonstration data.
- Run automated tests through GitHub Actions.

## Statistics and mathematics

### Regression

The models estimate response time from features known when the request is submitted, such as priority, category, channel, hour, backlog, and agent experience.

A multiple linear regression model can be written as:

$$
\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p
$$

Here, \(\hat{y}\) is the predicted response time, and the \(x\) values represent input features.

- **Linear Regression** models a linear relationship.
- **Ridge Regression** adds L2 regularization to discourage overly large coefficients.
- **Random Forest Regression** combines decision trees to capture nonlinear patterns.

### Descriptive statistics and confidence intervals

The dashboard reports the mean, median, sample standard deviation, and 90th percentile. A bootstrap procedure repeatedly resamples the observed response times to estimate a **95% confidence interval for the dataset mean**. This is not an interval for an individual request.

### Prediction uncertainty and deadline probability

The default evaluation split is approximately **60% training, 20% calibration, and 20% testing**.

- A **split-conformal prediction interval** uses absolute errors from the calibration data to estimate a range around a prediction.
- The **deadline-risk estimate** applies the calibration error distribution to the prediction and measures the proportion of estimated outcomes that exceed the chosen deadline.

These estimates depend on the calibration data representing future cases. A nominal 90% prediction interval is not a guarantee for every request, and the deadline-risk estimate must not be treated as a validated real-world probability until tested on appropriate data.

### Model evaluation

| Metric | Meaning | Preferred direction |
|---|---|---|
| **MAE** | Average absolute error, in minutes | Lower |
| **RMSE** | Penalizes large errors more strongly, in minutes | Lower |
| **R²** | Comparison with predicting the test-set mean | Generally higher |

A negative R² is possible and can indicate that the model performs worse than that simple baseline on the test set.

## How it works

1. **Load data** from the built-in demo dataset or a support-ticket CSV.
2. **Validate** the target and supported input columns.
3. **Split** the data into training, calibration, and test sets.
4. **Train and compare** the three regression models.
5. **Evaluate** model errors on the held-out test set.
6. **Estimate uncertainty and deadline risk** using calibration errors, kept separate from the final test set.
7. **Explore a request** in the interactive form and review the prediction before making a human decision.

## Run locally

**Requirements:** Python 3.10–3.13 and pip.

### Ubuntu / Linux / macOS

```bash
git clone https://github.com/VISSHAL-ANAND/mathproject.git
cd mathproject

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

### Windows PowerShell

```powershell
git clone https://github.com/VISSHAL-ANAND/mathproject.git
cd mathproject

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local address to open in your browser.

### Useful commands

Run the automated tests:

```bash
pytest -q
```

Train the demonstration models and export local metrics and model files:

```bash
python -m src.train
```

Generated files are saved under `artifacts/` and are excluded from Git.

## Use your own CSV

The required target column is **`response_time_minutes`**. It must contain a numeric, non-negative duration from submission to **first response**—not ticket resolution time. The current workflow requires at least 50 usable rows.

| Column | Type | Description |
|---|---|---|
| `response_time_minutes` | Numeric target | Minutes until first response |
| `priority` | Category | For example: low, medium, high, urgent |
| `category` | Category | For example: account, billing, technical, general |
| `channel` | Category | For example: email, chat, web |
| `hour_submitted` | Numeric | Hour of day, 0–23 |
| `day_of_week` | Numeric | Day index, 0–6; use one consistent mapping |
| `backlog` | Numeric | Queue size when the request is submitted |
| `agent_experience_years` | Numeric | Agent experience in years |

Include only predictors that are available at the time a request is submitted. Unsupported extra columns are ignored. More rows help only when the data is relevant, correctly labelled, and representative of the situation you want to predict.

**Privacy:** upload only data you are authorized to use. Remove names, student IDs, email addresses, ticket text, and other unnecessary personal or confidential information first.

## Project structure

```text
mathproject/
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Python dependencies
├── pytest.ini              # Test discovery configuration
├── src/
│   ├── data.py             # Demo data and validation
│   ├── models.py           # Regression models and preprocessing
│   ├── evaluate.py         # Data splits, metrics, calibration residuals
│   ├── statistics.py       # Descriptive statistics and risk estimates
│   └── train.py            # CLI training and local artifact export
├── tests/
│   └── test_project.py     # Automated tests
├── data/                   # Guidance for local datasets
└── DOX/                    # Requirements, research, and engineering docs
```

## Testing

GitHub Actions checks Python syntax and runs the automated tests on Python 3.10, 3.11, 3.12, and 3.13. Use the badge at the top of this page to see the workflow status.

Passing tests verifies the tested software behavior; it does **not** prove real-world prediction accuracy.

## Limitations and responsible use

- **Synthetic demo data:** it cannot establish whether the approach works on real campus support requests.
- **Real-data validation is still needed:** verify the dataset's source, usage terms, and first-response definition before reporting experimental results.
- **Changing conditions:** prediction intervals and deadline-risk estimates may become unreliable when future ticket patterns differ from calibration data.
- **Fairness:** review errors for urgent requests and relevant issue categories before any operational use.
- **Human oversight:** use the output as decision support, not as an automatic rule for denying, delaying, or deprioritizing support.

## Next steps

These are planned improvements, not features claimed as complete:

- Validate the models on a suitable, approved, anonymized real dataset.
- Use time-based evaluation for historical ticket logs.
- Compare against a simple baseline, such as predicting the training-set mean.
- Check model errors by priority and issue category.
- Add a queue/backlog what-if simulation and monitor model reliability over time.

## References

- [Scikit-learn: Linear models](https://scikit-learn.org/stable/modules/linear_model.html)
- [Scikit-learn: Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Scikit-learn: Pipelines and preprocessing](https://scikit-learn.org/stable/modules/compose.html)
- [Scikit-learn: Prediction interval example](https://scikit-learn.org/stable/auto_examples/inspection/plot_prediction_intervals.html)
- [An Introduction to Statistical Learning](https://www.statlearning.com/)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/)

For detailed requirements, architecture, research notes, and operational guidance, see the `DOX/` directory.
