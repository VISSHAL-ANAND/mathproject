"""Statistical summaries and uncertainty estimates for response-time predictions."""
import numpy as np
import pandas as pd
from src.data import TARGET


def response_time_summary(data: pd.DataFrame, bootstrap_samples: int = 1000, random_state: int = 42) -> dict:
    """Return descriptive statistics and a percentile-bootstrap CI for the sample mean."""
    values = pd.to_numeric(data[TARGET], errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 2:
        raise ValueError("At least two valid response-time observations are required.")
    rng = np.random.default_rng(random_state)
    means = rng.choice(values, size=(bootstrap_samples, len(values)), replace=True).mean(axis=1)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std_dev": float(np.std(values, ddof=1)),
        "q90": float(np.quantile(values, 0.90)),
        "mean_ci_low": float(np.quantile(means, 0.025)),
        "mean_ci_high": float(np.quantile(means, 0.975)),
    }


def conformal_prediction_interval(prediction: float, calibration_residuals, coverage: float = 0.90) -> tuple[float, float]:
    """Build a split-conformal interval from held-out calibration residuals."""
    residuals = np.asarray(calibration_residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < 10:
        raise ValueError("At least 10 calibration rows (residuals) are required for an uncertainty interval.")
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between 0 and 1.")
    # Finite-sample split conformal needs ceil(coverage / (1 - coverage))
    # calibration points for a finite quantile at the requested coverage.
    rank = int(np.ceil((len(residuals) + 1) * coverage))
    if rank > len(residuals):
        minimum = int(np.ceil(coverage / (1.0 - coverage)))
        raise ValueError(
            f"At least {minimum} calibration rows are required for a finite "
            f"{coverage:.0%} prediction interval; only {len(residuals)} are available."
        )
    absolute = np.abs(residuals)
    radius = float(np.sort(absolute)[rank - 1])
    return max(0.0, float(prediction) - radius), max(0.0, float(prediction) + radius)


def probability_sla_breach(prediction: float, calibration_residuals, sla_minutes: float) -> float:
    """Estimate breach probability from empirical calibration residuals."""
    residuals = np.asarray(calibration_residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < 10:
        raise ValueError("At least 10 calibration residuals are required for a breach probability.")
    simulated_outcomes = np.maximum(0.0, float(prediction) + residuals)
    return float(np.mean(simulated_outcomes > float(sla_minutes)))
