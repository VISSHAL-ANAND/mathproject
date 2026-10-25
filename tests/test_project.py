import numpy as np
import pandas as pd
import pytest
from src.data import TARGET, make_demo_data, validate_dataset
from src.evaluate import evaluate_models


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


def test_evaluation_returns_three_models_and_metrics():
    metrics, models, predictions = evaluate_models(make_demo_data(150))
    assert set(metrics["Model"]) == {"Linear Regression", "Ridge Regression", "Random Forest"}
    assert len(models) == 3
    assert {"MAE (minutes)", "RMSE (minutes)", "R²"}.issubset(metrics.columns)
    assert np.isfinite(metrics[["MAE (minutes)", "RMSE (minutes)", "R²"]]).all().all()
    assert len(predictions) == 30


def test_negative_targets_are_removed():
    frame = make_demo_data(50)
    frame.loc[0, TARGET] = -1
    clean = validate_dataset(frame)
    assert (clean[TARGET] >= 0).all()
    assert len(clean) == 49
