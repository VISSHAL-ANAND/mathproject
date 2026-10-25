import numpy as np
import pandas as pd
import pytest
from src.data import TARGET, make_demo_data, validate_dataset
from src.evaluate import evaluate_models
from src.statistics import conformal_prediction_interval, probability_sla_breach, response_time_summary


def test_demo_data_reproducible_and_valid():
    a = make_demo_data(100, random_state=7)
    b = make_demo_data(100, random_state=7)
    pd.testing.assert_frame_equal(a, b)
    assert len(validate_dataset(a)) == 100
    assert (a[TARGET] >= 0).all()


def test_demo_data_rejects_too_few_rows():
    with pytest.raises(ValueError):
        make_demo_data(10)


def test_validation_requires_target():
    with pytest.raises(ValueError, match="response_time_minutes"):
        validate_dataset(pd.DataFrame({"backlog": range(40)}))


def test_evaluation_returns_models_metrics_and_calibration_data():
    metrics, models, predictions, calibration = evaluate_models(make_demo_data(150))
    assert set(metrics["Model"]) == {"Linear Regression", "Ridge Regression", "Random Forest"}
    assert len(models) == 3
    assert set(calibration) == set(models)
    assert {"MAE (minutes)", "RMSE (minutes)", "R²"}.issubset(metrics.columns)
    assert np.isfinite(metrics[["MAE (minutes)", "RMSE (minutes)", "R²"]]).all().all()
    assert len(predictions) == 30
    assert all(len(values) == 30 for values in calibration.values())


def test_negative_targets_are_removed():
    frame = make_demo_data(50)
    frame.loc[0, TARGET] = -1
    clean = validate_dataset(frame)
    assert (clean[TARGET] >= 0).all()
    assert len(clean) == 49


def test_statistical_summary_has_bootstrap_confidence_interval():
    summary = response_time_summary(make_demo_data(100))
    assert summary["count"] == 100
    assert summary["mean_ci_low"] <= summary["mean"] <= summary["mean_ci_high"]
    assert summary["std_dev"] > 0


def test_conformal_interval_contains_point_and_is_nonnegative():
    residuals = np.linspace(-5, 5, 100)
    low, high = conformal_prediction_interval(20, residuals, coverage=0.90)
    assert 0 <= low <= 20 <= high


def test_sla_breach_probability_is_empirical():
    residuals = np.array([-10, -5, 0, 5, 10] * 4, dtype=float)
    assert probability_sla_breach(50, residuals, 55) == pytest.approx(0.2)


def test_uncertainty_functions_require_calibration_data():
    with pytest.raises(ValueError):
        conformal_prediction_interval(20, [1, 2, 3], 0.9)
