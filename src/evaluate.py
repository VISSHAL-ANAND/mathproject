"""Train/test evaluation plus a separate calibration split for uncertainty estimates."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from src.data import TARGET, validate_dataset
from src.models import build_models


def evaluate_models(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Return metrics, fitted models, test predictions, and per-model calibration residuals.

    The default split is approximately 60% train, 20% calibration, and 20% test.
    Calibration residuals are never taken from the final test set.
    """
    clean = validate_dataset(data)
    if not 0.1 <= test_size <= 0.3:
        raise ValueError("test_size must be between 0.1 and 0.3.")
    features = [c for c in clean.columns if c != TARGET]
    development, test = train_test_split(clean, test_size=test_size, random_state=random_state)
    calibration_fraction = test_size / (1.0 - test_size)
    train, calibration = train_test_split(
        development, test_size=calibration_fraction, random_state=random_state + 1
    )
    X_train, y_train = train[features], train[TARGET]
    X_cal, y_cal = calibration[features], calibration[TARGET]
    X_test, y_test = test[features], test[TARGET]
    fitted, rows, calibration_residuals = {}, [], {}
    prediction_frame = test[[TARGET]].copy()
    for name, pipeline in build_models(features, random_state).items():
        pipeline.fit(X_train, y_train)
        cal_pred = pipeline.predict(X_cal)
        calibration_residuals[name] = (y_cal.to_numpy(dtype=float) - cal_pred).tolist()
        pred = pipeline.predict(X_test)
        fitted[name] = pipeline
        rows.append({
            "Model": name,
            "MAE (minutes)": float(mean_absolute_error(y_test, pred)),
            "RMSE (minutes)": float(np.sqrt(mean_squared_error(y_test, pred))),
            "R²": float(r2_score(y_test, pred)),
        })
        prediction_frame[f"{name} prediction"] = pred
    metrics = pd.DataFrame(rows).sort_values("MAE (minutes)").reset_index(drop=True)
    best_name = metrics.iloc[0]["Model"]
    prediction_frame["Residual (actual - predicted)"] = (
        prediction_frame[TARGET] - prediction_frame[f"{best_name} prediction"]
    )
    return metrics, fitted, prediction_frame, calibration_residuals
