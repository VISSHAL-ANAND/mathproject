# Architecture
- app.py: Streamlit UI, dataset upload, statistics, model comparison, uncertainty interval, and SLA-risk display.
- src/data.py: reproducible synthetic demo data and input schema validation.
- src/models.py: preprocessing pipelines and regression estimators.
- src/evaluate.py: approximately 60% train / 20% calibration / 20% test split; test metrics and calibration residuals are kept separate.
- src/statistics.py: descriptive statistics, bootstrap confidence interval, split-conformal prediction interval, and empirical SLA-breach probability.
- src/train.py: CLI demo training and local artifact/report export.
- tests/: automated checks for data, metrics, and statistical functions.
- data/: instructions for local datasets; no private dataset is committed.
- artifacts/: local generated model files, excluded from Git.
- DOX/: product, engineering, research, and operational documentation.

The UI and CLI depend on core src modules; core modules do not depend on Streamlit. The test set is reserved for evaluation; uncertainty estimates use a separate calibration split.
