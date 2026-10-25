"""Leakage-aware model selection, final testing, and uncertainty calibration."""
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from src.data import OPENED_AT_META, TARGET, validate_dataset
from src.models import build_models


def _split_dataset(clean: pd.DataFrame, test_size: float, random_state: int):
    """Return train, validation, calibration, test sets and the split strategy."""
    features_and_target = clean
    n_rows = len(features_and_target)
    time_col = features_and_target.get(OPENED_AT_META)

    if time_col is not None and time_col.notna().all():
        ordered = features_and_target.sort_values(OPENED_AT_META, kind="mergesort")
        n_test = max(1, int(math.ceil(n_rows * test_size)))
        development = ordered.iloc[: n_rows - n_test]
        test = ordered.iloc[n_rows - n_test :]
        n_train = int(len(development) * 0.75)
        train = development.iloc[:n_train]
        validation_and_calibration = development.iloc[n_train:]
        midpoint = len(validation_and_calibration) // 2
        validation = validation_and_calibration.iloc[:midpoint]
        calibration = validation_and_calibration.iloc[midpoint:]
        return train, validation, calibration, test, "chronological"

    development, test = train_test_split(
        features_and_target, test_size=test_size, random_state=random_state
    )
    train, validation_and_calibration = train_test_split(
        development, test_size=0.25, random_state=random_state + 1
    )
    validation, calibration = train_test_split(
        validation_and_calibration, test_size=0.5, random_state=random_state + 2
    )
    return train, validation, calibration, test, "random"


def evaluate_models(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Train, select, calibrate, then evaluate without selecting on the final test set.

    With the default test_size=0.2, a dataset without timestamps is split approximately
    60% train, 10% validation, 10% calibration, and 20% test. Valid opened_at timestamps
    trigger a chronological split in the same proportions. Models are selected by
    validation MAE; calibration residuals estimate uncertainty; test rows are used only
    for final metrics and plots.
    """
    clean = validate_dataset(data)
    if not 0.1 <= test_size <= 0.2:
        raise ValueError("test_size must be between 0.1 and 0.2.")
    feature_columns = [
        column for column in clean.columns
        if column not in (TARGET, OPENED_AT_META)
    ]
    train, validation, calibration, test, split_method = _split_dataset(
        clean, test_size=test_size, random_state=random_state
    )
    if len(calibration) < 10:
        raise ValueError(
            "At least 10 calibration rows are required. Upload more usable tickets."
        )

    X_train, y_train = train[feature_columns], train[TARGET]
    X_validation, y_validation = validation[feature_columns], validation[TARGET]
    X_calibration, y_calibration = calibration[feature_columns], calibration[TARGET]
    X_test, y_test = test[feature_columns], test[TARGET]

    # Select on validation data, never on the final test set.
    selection_models = build_models(feature_columns, random_state)
    validation_mae = {}
    for name, pipeline in selection_models.items():
        pipeline.fit(X_train, y_train)
        validation_prediction = pipeline.predict(X_validation)
        validation_mae[name] = float(mean_absolute_error(y_validation, validation_prediction))

    # Refit each candidate on training + validation. Calibration and test remain unseen.
    refit_data = pd.concat([train, validation]).sort_index()
    X_refit, y_refit = refit_data[feature_columns], refit_data[TARGET]
    fitted_models = build_models(feature_columns, random_state)
    rows, calibration_residuals = [], {}
    prediction_frame = test[[TARGET]].copy()

    for name, pipeline in fitted_models.items():
        pipeline.fit(X_refit, y_refit)
        calibration_prediction = pipeline.predict(X_calibration)
        calibration_residuals[name] = (
            y_calibration.to_numpy(dtype=float) - calibration_prediction
        ).tolist()

        test_prediction = pipeline.predict(X_test)
        rows.append({
            "Model": name,
            "Validation MAE (minutes)": validation_mae[name],
            "MAE (test minutes)": float(mean_absolute_error(y_test, test_prediction)),
            "RMSE (test minutes)": float(np.sqrt(mean_squared_error(y_test, test_prediction))),
            "R² (test)": float(r2_score(y_test, test_prediction)),
        })
        prediction_frame[f"{name} prediction"] = test_prediction

    metrics = (
        pd.DataFrame(rows)
        .sort_values("Validation MAE (minutes)")
        .reset_index(drop=True)
    )
    best_name = metrics.iloc[0]["Model"]
    prediction_frame["Residual (actual - predicted)"] = (
        prediction_frame[TARGET] - prediction_frame[f"{best_name} prediction"]
    )
    metrics.attrs["split_method"] = split_method
    metrics.attrs["split_proportions"] = "approximately 60/10/10/20 (train/validation/calibration/test)"
    metrics.attrs["selected_model"] = best_name
    return metrics, fitted_models, prediction_frame, calibration_residuals
