# Architecture

- `app.py`: Streamlit UI, CSV upload and exclusions, descriptive statistics, priority/category breakdowns, time trend, model comparison, intervals, deadline risk, backlog what-if, and prediction form.
- `src/data.py`: reproducible synthetic data, CSV alias normalization, target validation, and optional opening-timestamp metadata.
- `src/models.py`: preprocessing pipelines, training-mean baseline, and regression estimators.
- `src/evaluate.py`: chronological or random split; validation-based model selection; refit on training + validation; calibration residuals; held-out test metrics.
- `src/statistics.py`: descriptive summaries, bootstrap confidence interval, split-conformal prediction interval, and empirical deadline-risk estimate.
- `src/train.py`: CLI demo training and local report/model export.
- `tests/`: checks for data validation, alias normalization, split strategy, model selection, statistical functions, and dashboard smoke testing.
- `data/`: data instructions and optional sample-source notes; no private data is committed.
- `artifacts/`: generated local outputs, excluded from Git.
- `DOX/`: project requirements, architecture, research, tasks, and operating notes.

The optional `__opened_at` column is metadata for chronological splitting, not a predictor. Preprocessing is fit in model pipelines. Model selection uses validation MAE; calibration residuals estimate uncertainty; the test set is held out for final metrics.
