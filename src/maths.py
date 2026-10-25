"""Hand-written maths that mirrors what the scikit-learn models and interval code do.

Nothing here changes the app's predictions. It re-derives the same numbers from
formulas so they can be explained (and checked) in the maths presentation.
"""
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

from src.data import OPENED_AT_META, TARGET
from src.models import build_models

SQRT_PI_OVER_2 = math.sqrt(math.pi / 2)  # RMSE/MAE for perfectly bell-shaped errors (~1.2533)


def design_matrix(data: pd.DataFrame):
    """Turn the table into numbers (scaled numerics + one-hot categories) exactly as the app does."""
    features = [c for c in data.columns if c not in (TARGET, OPENED_AT_META)]
    preprocessor = build_models(features)["Linear Regression"].named_steps["preprocessor"]
    X = preprocessor.fit_transform(data[features])
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=float), data[TARGET].to_numpy(dtype=float)


def least_squares(X: np.ndarray, y: np.ndarray):
    """Normal equation  beta = (X'X)^-1 X'y  on centred data. Returns (intercept, coefficients).

    One-hot columns add up to a constant, so X'X is singular; the pseudo-inverse
    picks the smallest solution, which is the same one scikit-learn picks.
    """
    x_mean, y_mean = X.mean(axis=0), y.mean()
    Xc, yc = X - x_mean, y - y_mean
    beta = np.linalg.pinv(Xc.T @ Xc) @ (Xc.T @ yc)
    return float(y_mean - x_mean @ beta), beta


def ridge_closed_form(X: np.ndarray, y: np.ndarray, alpha: float):
    """Ridge: beta = (X'X + alpha*I)^-1 X'y on centred data. The intercept is not penalised."""
    x_mean, y_mean = X.mean(axis=0), y.mean()
    Xc, yc = X - x_mean, y - y_mean
    beta = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(Xc.shape[1]), Xc.T @ yc)
    return float(y_mean - x_mean @ beta), beta


def compare_with_sklearn(data: pd.DataFrame) -> dict:
    """Fit by hand and with scikit-learn on the same numbers; report how far apart they are."""
    X, y = design_matrix(data)
    features = [c for c in data.columns if c not in (TARGET, OPENED_AT_META)]
    alpha = float(build_models(features)["Ridge Regression"].named_steps["model"].alpha)

    a_ols, b_ols = least_squares(X, y)
    a_rdg, b_rdg = ridge_closed_form(X, y, alpha)
    sk_ols = LinearRegression().fit(X, y)
    sk_rdg = Ridge(alpha=alpha).fit(X, y)

    return {
        "n_rows": int(X.shape[0]),
        "n_columns": int(X.shape[1]),
        "alpha": alpha,
        "ols_max_diff": float(np.max(np.abs((a_ols + X @ b_ols) - sk_ols.predict(X)))),
        "ridge_max_diff": float(np.max(np.abs((a_rdg + X @ b_rdg) - sk_rdg.predict(X)))),
        "ols_coef_norm": float(np.linalg.norm(b_ols)),
        "ridge_coef_norm": float(np.linalg.norm(b_rdg)),
    }


def formula_mean_ci(values, z: float = 1.96) -> tuple[float, float]:
    """Textbook 95% interval for the mean: mean +/- 1.96 * s / sqrt(n)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    half = z * np.std(values, ddof=1) / math.sqrt(len(values))
    return float(values.mean() - half), float(values.mean() + half)


def error_shape(residuals) -> dict:
    """MAE, RMSE and their ratio. A ratio near 1.2533 means the errors are close to bell-shaped."""
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    mae = float(np.mean(np.abs(r)))
    rmse = float(np.sqrt(np.mean(r ** 2)))
    return {"mae": mae, "rmse": rmse, "ratio": rmse / mae, "bell_ratio": SQRT_PI_OVER_2}


def conformal_details(calibration_residuals, coverage: float) -> dict:
    """Show the steps behind the prediction interval: how many residuals, which one is picked, the radius."""
    absolute = np.abs(np.asarray(calibration_residuals, dtype=float))
    absolute = absolute[np.isfinite(absolute)]
    n = len(absolute)
    rank = int(math.ceil((n + 1) * coverage))
    if rank > n:
        raise ValueError(f"Only {n} calibration rows: too few for a finite {coverage:.0%} interval.")
    return {"n": n, "rank": rank, "radius": float(np.sort(absolute)[rank - 1])}


def normal_breach_probability(prediction: float, calibration_residuals, sla_minutes: float) -> float:
    """Bell-curve version of deadline risk: P(Z > (deadline - prediction) / sigma)."""
    r = np.asarray(calibration_residuals, dtype=float)
    r = r[np.isfinite(r)]
    sigma = float(np.std(r, ddof=1))
    z = (float(sla_minutes) - float(prediction)) / sigma
    return float(0.5 * math.erfc(z / math.sqrt(2)))  # 1 - Phi(z)
