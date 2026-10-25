import numpy as np
from src.data import make_demo_data, validate_dataset
from src.evaluate import evaluate_models
from src.maths import (
    compare_with_sklearn, conformal_details, error_shape,
    formula_mean_ci, normal_breach_probability,
)
from src.statistics import conformal_prediction_interval


def _data():
    return validate_dataset(make_demo_data())


def test_hand_formulas_match_sklearn():
    check = compare_with_sklearn(_data())
    assert check["ols_max_diff"] < 1e-6
    assert check["ridge_max_diff"] < 1e-6
    assert check["ridge_coef_norm"] <= check["ols_coef_norm"]


def test_conformal_details_match_app_interval():
    _, _, _, cal = evaluate_models(_data())
    residuals = cal["Ridge Regression"]
    d = conformal_details(residuals, 0.90)
    low, high = conformal_prediction_interval(50.0, residuals, 0.90)
    assert abs((high - low) / 2 - d["radius"]) < 1e-9


def test_normal_breach_probability_behaves():
    rng = np.random.default_rng(0)
    residuals = rng.normal(0, 8, 300)
    assert 0 <= normal_breach_probability(50, residuals, 60) <= 1
    assert normal_breach_probability(50, residuals, 40) > normal_breach_probability(50, residuals, 70)


def test_mean_ci_and_error_shape():
    low, high = formula_mean_ci(_data()["response_time_minutes"])
    assert low < high
    shape = error_shape(np.random.default_rng(1).normal(0, 9, 5000))
    assert abs(shape["ratio"] - shape["bell_ratio"]) < 0.05
