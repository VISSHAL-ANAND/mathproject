# Architecture
- app.py: Streamlit UI, CSV upload, metrics, plots, and prediction form.
- src/data.py: synthetic data and schema validation.
- src/models.py: preprocessing pipelines and regression estimators.
- src/evaluate.py: split, fit, predictions, and metrics.
- src/train.py: command-line demo training and local artifact export.
- tests/: automated checks.
- data/: instructions for local datasets; no private dataset is committed.
- artifacts/: local generated model files, excluded from Git.
- DOX/: project and engineering documentation.

UI and CLI depend on core src modules; core modules do not depend on Streamlit.
