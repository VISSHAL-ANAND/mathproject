import numpy as np
import pandas as pd
import pytest
from src.data import (
    OPENED_AT_META, TARGET, make_demo_data, validate_dataset,
)
from src.evaluate import evaluate_models
from src.statistics import (
    conformal_prediction_interval, probability_sla_breach, response_time_summary,
)


def test_demo_data_reproducible_and_valid():
    a = make_demo_data(100, random_state=7)
    b = make_demo_data(100, random_state=7)
    pd.testing.assert_frame_equal(a, b)
    assert len(validate_dataset(a)) == 100
    assert (a[TARGET] >= 0).all()


def test_demo_data_rejects_too_few_rows():
    with pytest.raises(ValueError):
        make_demo_data(10)


def test_validation_requires_target_or_supported_alias():
    with pytest.raises(ValueError, match="response_time_minutes"):
        validate_dataset(pd.DataFrame({"backlog": range(120)}))


def test_validation_rejects_too_few_usable_rows():
    with pytest.raises(ValueError, match="At least 100 usable rows"):
        validate_dataset(make_demo_data(99))


def test_evaluation_selects_on_validation_and_reports_held_out_test_metrics():
    metrics, models, predictions, calibration = evaluate_models(make_demo_data(150))
    expected_models = {
        "Training Mean Baseline", "Linear Regression", "Ridge Regression", "Random Forest",
    }
    assert set(metrics["Model"]) == expected_models
    assert set(models) == expected_models
    assert set(calibration) == expected_models
    assert {
        "Validation MAE (minutes)", "MAE (test minutes)",
        "RMSE (test minutes)", "R² (test)",
    }.issubset(metrics.columns)
    assert np.isfinite(
        metrics[["Validation MAE (minutes)", "MAE (test minutes)",
                 "RMSE (test minutes)", "R² (test)"]]
    ).all().all()
    assert len(predictions) == 30
    assert all(len(values) == 15 for values in calibration.values())
    assert metrics.iloc[0]["Validation MAE (minutes)"] == pytest.approx(
        metrics["Validation MAE (minutes)"].min()
    )
    assert metrics.attrs["selected_model"] == metrics.iloc[0]["Model"]
    assert metrics.attrs["split_method"] == "random"


def test_known_ticket_csv_aliases_and_chronological_split():
    frame = make_demo_data(150)
    frame["first_response_min"] = frame.pop(TARGET)
    frame["queue"] = frame.pop("category")
    frame["opened_at"] = pd.date_range("2025-01-01", periods=len(frame), freq="h")
    frame = frame.drop(columns=["hour_submitted", "day_of_week"])

    clean = validate_dataset(frame)
    assert TARGET in clean.columns
    assert "category" in clean.columns
    assert "hour_submitted" in clean.columns
    assert "day_of_week" in clean.columns
    assert OPENED_AT_META in clean.columns

    metrics, _, predictions, calibration = evaluate_models(frame)
    assert metrics.attrs["split_method"] == "chronological"
    assert len(predictions) == 30
    assert all(len(values) == 15 for values in calibration.values())


def test_negative_targets_are_removed():
    frame = make_demo_data(120)
    frame.loc[0, TARGET] = -1
    clean = validate_dataset(frame)
    assert (clean[TARGET] >= 0).all()
    assert len(clean) == 119


def test_missing_targets_are_removed():
    frame = make_demo_data(120)
    frame.loc[0, TARGET] = np.nan
    clean = validate_dataset(frame)
    assert clean[TARGET].notna().all()
    assert len(clean) == 119


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
    with pytest.raises(ValueError, match="At least 10 calibration rows"):
        conformal_prediction_interval(20, [1, 2, 3], 0.9)


def test_high_coverage_interval_requires_sufficient_calibration_rows():
    with pytest.raises(ValueError, match="At least 19 calibration rows"):
        conformal_prediction_interval(20, np.linspace(-5, 5, 10), 0.95)
