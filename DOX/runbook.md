# Runbook
1. Install Python 3.10–3.13.
2. Create and activate a virtual environment.
3. Install dependencies with python -m pip install -r requirements.txt.
4. Run tests with pytest -q.
5. Train demo models with python -m src.train.
6. Launch with streamlit run app.py.

If dependencies fail, use Python 3.12 or 3.13. CSVs need a numeric response_time_minutes target and at least one supported feature. Small datasets produce unstable estimates.

For reproducibility, record Python/package versions, data source and license, target definition, split method, and model settings.
