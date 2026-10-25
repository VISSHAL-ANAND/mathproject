"""Regression estimators and a simple training-mean baseline."""
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_models(feature_columns: list[str], random_state: int = 42) -> dict[str, Pipeline]:
    """Create comparable pipelines with all preprocessing fit on each training split."""
    categorical = [c for c in feature_columns if c in CATEGORICAL_FEATURES]
    numeric = [c for c in feature_columns if c in NUMERIC_FEATURES]
    if not categorical and not numeric:
        raise ValueError("No supported predictors were supplied.")

    def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
        transformers = []
        if numeric:
            steps = [("imputer", SimpleImputer(strategy="median"))]
            if scale_numeric:
                steps.append(("scaler", StandardScaler()))
            transformers.append(("numeric", Pipeline(steps), numeric))
        if categorical:
            transformers.append((
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical,
            ))
        return ColumnTransformer(transformers=transformers, remainder="drop")

    return {
        "Training Mean Baseline": Pipeline([
            ("preprocessor", make_preprocessor(False)),
            ("model", DummyRegressor(strategy="mean")),
        ]),
        "Linear Regression": Pipeline([
            ("preprocessor", make_preprocessor(True)),
            ("model", LinearRegression()),
        ]),
        "Ridge Regression": Pipeline([
            ("preprocessor", make_preprocessor(True)),
            ("model", Ridge(alpha=1.0)),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", make_preprocessor(False)),
            ("model", RandomForestRegressor(
                n_estimators=160,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
            )),
        ]),
    }
