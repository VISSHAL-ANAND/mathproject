"""Consistent train/test evaluation for regression models."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from src.data import TARGET, validate_dataset
from src.models import build_models


def evaluate_models(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    clean = validate_dataset(data)
    features = [c for c in clean.columns if c != TARGET]
    train, test = train_test_split(clean, test_size=test_size, random_state=random_state)
    X_train, y_train = train[features], train[TARGET]
    X_test, y_test = test[features], test[TARGET]
    fitted, rows = {}, []
    prediction_frame = test[[TARGET]].copy()
    for name, pipeline in build_models(features, random_state).items():
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        fitted[name] = pipeline
        rows.append({"Model": name, "MAE (minutes)": float(mean_absolute_error(y_test, pred)),
                     "RMSE (minutes)": float(np.sqrt(mean_squared_error(y_test, pred))),
                     "R²": float(r2_score(y_test, pred))})
        prediction_frame[f"{name} prediction"] = pred
    metrics = pd.DataFrame(rows).sort_values("MAE (minutes)").reset_index(drop=True)
    best_name = metrics.iloc[0]["Model"]
    prediction_frame["Residual (actual - predicted)"] = prediction_frame[TARGET] - prediction_frame[f"{best_name} prediction"]
    return metrics, fitted, prediction_frame
