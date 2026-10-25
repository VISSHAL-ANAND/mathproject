# Response Time Prediction Using Regression Models
### CampusHelp Response Intelligence — Regression, Probability & Statistical Decision Support

<p align="center">
  <strong>Know the likely wait. Understand the uncertainty. Spot deadline risk earlier.</strong>
</p>

<p align="center">
  <a href="https://github.com/VISSHAL-ANAND/mathproject/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/VISSHAL-ANAND/mathproject/actions/workflows/tests.yml/badge.svg"></a>
  <img alt="Python 3.10–3.13" src="https://img.shields.io/badge/Python-3.10%E2%80%933.13-2563EB">
  <img alt="Streamlit" src="https://img.shields.io/badge/UI-Streamlit-2563EB">
  <img alt="Focus" src="https://img.shields.io/badge/Focus-Statistics%20%2B%20ML-1D4ED8">
</p>

**CampusHelp Response Intelligence** is a college project designed around a real, relatable problem: when a student reports a campus issue—such as Wi-Fi failure, an account problem, or a hostel maintenance request—they often have no useful estimate of when someone will respond.

The application predicts **time until first response**, shows how uncertain the estimate is, and estimates the **chance of missing a chosen service deadline**. It combines regression models with probability and statistical analysis in one understandable dashboard.

> **Data honesty:** the included data is synthetic and only demonstrates how the application works. It is not real campus or customer-support data, and its scores must not be presented as proof of real-world accuracy.

## Contents
- [The human problem](#the-human-problem)
- [Our idea and what makes it different](#our-idea-and-what-makes-it-different)
- [What the application does](#what-the-application-does)
- [Statistics and mathematics used](#statistics-and-mathematics-used)
- [How the system works](#how-the-system-works)
- [Run it locally](#run-it-locally)
- [Use your own CSV](#use-your-own-csv)
- [Understand the results](#understand-the-results)
- [Project structure](#project-structure)
- [Testing](#testing)
- [Limitations and responsible use](#limitations-and-responsible-use)
- [Future improvements](#future-improvements)
- [References](#references)

## The human problem

A ticketing system may record that a request exists, but the student still wants to know:
1. **How long might I wait for the first response?**
2. **Could this request miss the service deadline?**
3. **How uncertain is the estimate?**
4. **Should a support team review this request sooner?**

A single predicted number cannot answer all four questions. The project therefore reports a point estimate together with uncertainty and deadline-risk information.

## Our idea and what makes it different

The idea is to make prediction **useful for a human decision**, not merely produce a model score.

For each support request, CampusHelp brings three views together:

| View | What it tells the user | How it is produced |
|---|---|---|
| **Likely wait** | Estimated minutes until first response | A selected regression model |
| **Prediction range** | A range that reflects typical errors seen on calibration data | Split-conformal interval using held-out calibration residuals |
| **Deadline risk** | Estimated chance the response time exceeds the chosen service deadline | Empirical proportion of calibrated residual outcomes over the deadline |

The dashboard also shows average, median, variability, the 90th percentile, model comparison metrics, and prediction-error plots.

### What is innovative about our approach?

The innovation is the **decision-focused integration** of standard statistical methods: point prediction, uncertainty, and deadline risk appear together, in plain language, to support a real help-desk workflow. A student or staff member can see not just what the model predicts, but also how uncertain it is and whether a deadline may be missed.

This project does **not** claim to invent a new regression algorithm. Its value is in combining established methods into a transparent, human-centered tool and making the uncertainty visible instead of hiding it behind one number. Human staff remain responsible for prioritization; the application does not automatically reject or deprioritize anyone.

## What the application does

- Loads reproducible synthetic demonstration data or a user-supplied CSV.
- Validates the target and supported features.
- Compares **Linear Regression**, **Ridge Regression**, and **Random Forest Regression**.
- Reports **MAE**, **RMSE**, and **R²** on held-out test data.
- Calculates mean, median, sample standard deviation, and 90th percentile.
- Estimates a bootstrap 95% confidence interval for the dataset mean.
- Estimates a prediction interval using a separate calibration split.
- Estimates empirical risk of exceeding a user-selected response deadline.
- Shows actual-versus-predicted and residual-distribution charts.
- Provides an interactive form to predict a support request.
- Provides a command-line training script and automated tests.

## Statistics and mathematics used

### 1. Regression

The model estimates response time from the information available when the ticket is submitted, such as priority, issue category, channel, hour, backlog, and agent experience.

A simple linear model can be written as:

`\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p`

where `\hat{y}` is the predicted response time and the `x` values are the available features.

- **Linear Regression** models a linear relationship.
- **Ridge Regression** adds L2 regularization to discourage overly large coefficients.
- **Random Forest Regression** combines decision trees to capture nonlinear relationships.

### 2. Descriptive statistics

The dashboard reports:
- **Mean:** arithmetic average response time.
- **Median:** middle observed response time, less affected by extreme values.
- **Sample standard deviation:** variation in response times.
- **90th percentile:** the value at or below which 90% of observed response times fall.

### 3. Bootstrap confidence interval for the mean

The data is repeatedly resampled with replacement, and a mean is calculated for each resample. The 2.5th and 97.5th percentiles of those bootstrap means form the displayed 95% interval.

This describes uncertainty around the **dataset's mean response time**. It is not the interval for a single student's ticket.

### 4. Prediction interval

The data is split into training, calibration, and test portions. The model's errors on the separate calibration portion are used to estimate a split-conformal prediction interval for a new estimate.

The displayed coverage level (such as 90%) is a target under the method's assumptions, **not a guarantee** that every interval will contain the future response time.

### 5. Probability of missing a service deadline

For a chosen deadline, the app combines the predicted time with the empirical error distribution from the calibration data and calculates the proportion of outcomes that exceed the deadline.

This is an **empirical risk estimate**, not a fully calibrated real-world probability. Its reliability must be checked using representative historical data and later unseen cases.

### 6. Model evaluation

- **MAE:** average absolute prediction error, in minutes. Lower is better.
- **RMSE:** error metric that penalizes larger mistakes more strongly, in minutes. Lower is better.
- **R²:** compares model performance against predicting the test-set mean. Higher is generally better; a negative value is possible.

## How the system works

1. **Load data:** use the demo dataset or upload an appropriate CSV.
2. **Validate:** clean invalid target values and retain supported features.
3. **Split:** use approximately 60% for training, 20% for calibration, and 20% for final testing.
4. **Train:** fit three regression pipelines using training data.
5. **Compare:** score the models on the same held-out test rows.
6. **Calibrate:** use calibration errors for prediction intervals and deadline-risk estimates—not the final test rows.
7. **Predict:** enter a request's features, select a model, and choose a service deadline.
8. **Interpret:** review the estimated wait, prediction interval, deadline risk, and model limitations before a human decides what to do.

## Run it locally

**Requirements:** Python 3.10–3.13, pip, and a terminal.

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

Streamlit will print a local URL to open in your browser.

### Windows PowerShell

```powershell
git clone https://github.com/VISSHAL-ANAND/mathproject.git
cd mathproject

python -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

streamlit run app.py
```

### Other useful commands

Run the test suite:

```bash
pytest -q
```

Train the demonstration models and save a local report and model files:

```bash
python -m src.train
```

The CLI creates local files in `artifacts/`, which are excluded from Git.

## Use your own CSV

The app expects a numeric, non-negative target column named **`response_time_minutes`**. This must represent elapsed minutes until the **first response**, not the time until a ticket is resolved.

| Column | Type | Meaning |
|---|---|---|
| `response_time_minutes` | Numeric target | Minutes from ticket submission to first response |
| `priority` | Category | For example: low, medium, high, urgent |
| `category` | Category | For example: account, billing, technical, general |
| `channel` | Category | For example: email, chat, web |
| `hour_submitted` | Numeric | Hour of day, from 0 to 23 |
| `day_of_week` | Numeric | Day index, 0 to 6 |
| `backlog` | Numeric | Queue size at submission time |
| `agent_experience_years` | Numeric | Experience value, in years |

At least **50 usable rows** are required for the current calibration workflow. More data is generally better, but representativeness and quality matter more than row count alone. Supported predictors may be included as appropriate; unsupported extra columns are ignored.

Before uploading real data, remove names, email addresses, ticket text, student IDs, and other unnecessary personal or confidential details. Use only data you are authorized to process.

## Understand the results

- A lower MAE means smaller average absolute errors on the test split.
- A wider prediction interval reflects more observed uncertainty in the calibration errors.
- The estimated deadline risk can rise when the predicted wait is closer to or above the chosen deadline.
- These are not guarantees. If the real support process changes, old patterns may no longer apply.
- Always inspect errors for urgent and unusual tickets; average metrics can hide failures for important subgroups.

## Project structure

```text
mathproject/
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Python dependencies
├── pytest.ini              # Test discovery configuration
├── src/
│   ├── data.py             # Demo data and input validation
│   ├── models.py           # Regression models and preprocessing
│   ├── evaluate.py         # Data splits, metrics, calibration residuals
│   ├── statistics.py       # Confidence intervals and risk estimates
│   └── train.py            # CLI training and local artifacts
├── tests/
│   └── test_project.py     # Automated tests
├── data/                   # Guidance for local datasets
└── DOX/                    # Requirements, research, architecture, and runbook
```

## Testing

GitHub Actions runs the automated test suite on Python 3.10, 3.11, 3.12, and 3.13. The workflow badge at the top links to the current status. Passing automated tests checks the coded behavior; it does not prove that the model predicts real-world response times accurately.

## Limitations and responsible use

1. **Synthetic data:** the built-in dataset is generated by code and is only for demonstrations and testing.
2. **No real-world claim yet:** a suitable real dataset with a verified first-response target and redistribution/usage terms has not yet been validated.
3. **Calibration assumptions:** prediction intervals and empirical deadline risk can become unreliable if future data differs from calibration data.
4. **Possible bias:** group-specific prediction errors must be checked before operational use.
5. **Human oversight:** use the output to inform support staff, not as an automatic rule for denying help or lowering a person's priority.
6. **Timing definition:** document business hours, time zones, unanswered tickets, missing timestamps, and duplicate records before a real experiment.

## Future improvements

These are planned additions, **not features claimed as complete**:
- Validate on an appropriately licensed, anonymized real support-ticket dataset.
- Use chronological evaluation when predicting future tickets from historical logs.
- Compare against simple baselines such as the training-set mean.
- Audit prediction error for urgent tickets and relevant issue categories.
- Add queue-capacity / backlog what-if simulation.
- Add model monitoring for changing patterns and interval coverage.
- Deploy only after privacy, reliability, and operational safeguards are reviewed.

## References

- [Scikit-learn: Linear models](https://scikit-learn.org/stable/modules/linear_model.html)
- [Scikit-learn: Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Scikit-learn: Pipelines and preprocessing](https://scikit-learn.org/stable/modules/compose.html)
- [Scikit-learn: Prediction interval example](https://scikit-learn.org/stable/auto_examples/inspection/plot_prediction_intervals.html)
- [An Introduction to Statistical Learning](https://www.statlearning.com/)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/)

For detailed decisions, requirements, research methodology, and operational notes, see the `DOX/` directory.
