"""Data generation and validation."""
import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["priority", "category", "channel", "hour_submitted", "day_of_week", "backlog", "agent_experience_years"]
CATEGORICAL_FEATURES = ["priority", "category", "channel"]
NUMERIC_FEATURES = ["hour_submitted", "day_of_week", "backlog", "agent_experience_years"]
TARGET = "response_time_minutes"


def make_demo_data(n_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic demo data, not empirical customer-service observations."""
    if n_samples < 30:
        raise ValueError("n_samples must be at least 30.")
    rng = np.random.default_rng(random_state)
    priority = rng.choice(["low", "medium", "high", "urgent"], n_samples, p=[.28, .42, .22, .08])
    category = rng.choice(["account", "billing", "technical", "general"], n_samples)
    channel = rng.choice(["email", "chat", "web"], n_samples, p=[.45, .35, .20])
    hour = rng.integers(0, 24, n_samples)
    day = rng.integers(0, 7, n_samples)
    backlog = rng.poisson(12, n_samples)
    experience = rng.uniform(0, 12, n_samples)
    priority_effect = pd.Series(priority).map({"low": 18, "medium": 9, "high": 0, "urgent": -5}).to_numpy()
    category_effect = pd.Series(category).map({"account": 2, "billing": 6, "technical": 12, "general": 0}).to_numpy()
    channel_effect = pd.Series(channel).map({"email": 12, "chat": -7, "web": 3}).to_numpy()
    after_hours = ((hour < 8) | (hour >= 18)).astype(float) * 10
    weekend = (day >= 5).astype(float) * 8
    noise = rng.normal(0, 7, n_samples)
    response = 12 + priority_effect + category_effect + channel_effect + hour * .35 + weekend + after_hours + backlog * 1.8 - experience * 1.2 + noise
    response = np.clip(response, 1, None)
    return pd.DataFrame({
        "priority": priority, "category": category, "channel": channel,
        "hour_submitted": hour, "day_of_week": day, "backlog": backlog,
        "agent_experience_years": np.round(experience, 2),
        TARGET: np.round(response, 2),
    })


def validate_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Validate schema and remove unusable target rows."""
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("Dataset is empty.")
    if TARGET not in data.columns:
        raise ValueError(f"Dataset must contain a '{TARGET}' column.")
    features = [col for col in FEATURE_COLUMNS if col in data.columns]
    if not features:
        raise ValueError("At least one supported feature column is required.")
    result = data[features + [TARGET]].copy()
    result[TARGET] = pd.to_numeric(result[TARGET], errors="coerce")
    for column in NUMERIC_FEATURES:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        if column in result.columns:
            result[column] = result[column].where(result[column].notna(), np.nan).astype("object")
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET])
    result = result[result[TARGET] >= 0]
    if len(result) < 30:
        raise ValueError("At least 30 usable rows with non-negative targets are required.")
    if result[features].isna().all(axis=1).all():
        raise ValueError("All predictor values are missing.")
    return result.reset_index(drop=True)
