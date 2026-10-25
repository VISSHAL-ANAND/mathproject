# Architecture

- `app.py`: Streamlit UI, CSV upload, data-exclusion disclosures, statistics, model comparison, plots, interval/risk display, and prediction form.
- `src/data.py`: reproducible synthetic data, CSV alias normalization, target validation, and optional opened-at metadata.
- `src/models.py`: preprocessing pipelines, training-mean baseline, and regression estimators.
- `src/evaluate.py`: random or chronological split; validation-based model selection; refit on train + validation; calibration residuals; final test metrics.
- `src/statistics.py`: descriptive summaries, bootstrap confidence interval, split-conformal prediction interval, and empirical deadline-risk estimate.
- `src/train.py`: CLI demo training and local report/model export.
- `tests/`: tests for data, validation, aliases, splits, model selection, and statistical methods.
- `data/`: notes and instructions; external sample data is downloaded separately and no private data is committed.
- `artifacts/`: local generated model/report files, excluded from Git.
- `DOX/`: project requirements, architecture, research, tasks, and operational notes.

The application keeps feature columns separate from the `__opened_at` metadata used for chronological splitting. Preprocessing is fit within training pipelines. Model selection uses validation MAE, uncertainty uses calibration residuals, and the test set remains separate for final evaluation.
