"""Streamlit dashboard for response-time prediction and statistical risk analysis."""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from src.data import CATEGORICAL_FEATURES, FEATURE_COLUMNS, TARGET, make_demo_data, validate_dataset
from src.evaluate import evaluate_models
from src.statistics import conformal_prediction_interval, probability_sla_breach, response_time_summary

st.set_page_config(page_title="CampusHelp Response Intelligence", page_icon="📊", layout="wide")
st.title("CampusHelp Response Intelligence")
st.caption("A student-support decision aid using regression, probability, and statistics.")
st.info("Human problem: students often report a campus issue without knowing how long help will take or whether the service deadline will be missed.")

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload support-ticket CSV (optional)", type=["csv"])
    st.info("Without an upload, the app uses synthetic demonstration data—not real campus records.")
    sla_minutes = st.number_input("Service deadline (minutes)", min_value=5, max_value=1440, value=60, step=5)
    interval_coverage = st.select_slider("Prediction interval coverage", options=[0.80, 0.90, 0.95], value=0.90)

try:
    raw = pd.read_csv(uploaded) if uploaded is not None else make_demo_data()
    source_label = "Uploaded CSV" if uploaded is not None else "Synthetic demonstration dataset"
    raw = validate_dataset(raw)
    metrics, models, predictions, calibration_residuals = evaluate_models(raw)
except Exception as exc:
    st.error(f"Could not process dataset: {exc}")
    st.stop()

st.caption(f"Data source: {source_label} • Rows used: {len(raw):,}")
st.warning("Demo statistics and model scores are illustrative only. They must not be presented as real-world findings.")
summary = response_time_summary(raw)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Average wait", f"{summary['mean']:.1f} min")
m2.metric("Median wait", f"{summary['median']:.1f} min")
m3.metric("90th percentile", f"{summary['q90']:.1f} min")
m4.metric("95% CI for mean", f"{summary['mean_ci_low']:.1f}–{summary['mean_ci_high']:.1f} min")
st.caption("The confidence interval describes uncertainty in the dataset mean, not the wait for an individual ticket.")

st.subheader("Which regression model performs best?")
st.dataframe(metrics, use_container_width=True, hide_index=True)
st.bar_chart(metrics.set_index("Model")[["MAE (minutes)", "RMSE (minutes)"]])

best_name = metrics.iloc[0]["Model"]
best_column = f"{best_name} prediction"
left, right = st.columns(2)
with left:
    st.subheader("Actual vs predicted")
    fig, ax = plt.subplots()
    ax.scatter(predictions[TARGET], predictions[best_column], alpha=0.65)
    low = min(predictions[TARGET].min(), predictions[best_column].min())
    high = max(predictions[TARGET].max(), predictions[best_column].max())
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_xlabel("Actual response time (minutes)")
    ax.set_ylabel("Predicted response time (minutes)")
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.subheader("Prediction errors")
    fig, ax = plt.subplots()
    ax.hist(predictions["Residual (actual - predicted)"], bins=24)
    ax.set_xlabel("Residual (minutes)")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Predict a support request")
st.caption("Uncertainty and deadline risk are estimated from a separate calibration split, not the final test data.")
features = [col for col in FEATURE_COLUMNS if col in raw.columns and col != TARGET]
if not features:
    st.error("No supported predictor columns were found.")
    st.stop()

with st.form("prediction_form"):
    values = {}
    columns = st.columns(2)
    for i, feature in enumerate(features):
        with columns[i % 2]:
            if feature in CATEGORICAL_FEATURES:
                choices = sorted(raw[feature].dropna().astype(str).unique().tolist())
                values[feature] = st.selectbox(feature.replace("_", " ").title(), choices or ["unknown"])
            else:
                series = pd.to_numeric(raw[feature], errors="coerce").dropna()
                default = float(series.median()) if not series.empty else 0.0
                minimum = float(series.min()) if not series.empty else 0.0
                maximum = float(series.max()) if not series.empty else max(1.0, default + 1)
                if minimum == maximum:
                    maximum = minimum + 1.0
                values[feature] = st.number_input(
                    feature.replace("_", " ").title(), min_value=minimum,
                    max_value=maximum, value=min(max(default, minimum), maximum)
                )
    model_name = st.selectbox("Model", list(models.keys()), index=list(models.keys()).index(best_name))
    submitted = st.form_submit_button("Estimate wait and risk")

if submitted:
    input_frame = pd.DataFrame([values], columns=features)
    estimate = max(0.0, float(models[model_name].predict(input_frame)[0]))
    residuals = calibration_residuals[model_name]
    lower, upper = conformal_prediction_interval(estimate, residuals, coverage=interval_coverage)
    breach_probability = probability_sla_breach(estimate, residuals, sla_minutes)
    p1, p2, p3 = st.columns(3)
    p1.metric("Estimated first response", f"{estimate:.1f} min")
    p2.metric(f"{interval_coverage:.0%} prediction interval", f"{lower:.1f}–{upper:.1f} min")
    p3.metric(f"Chance of missing {sla_minutes}-minute deadline", f"{breach_probability:.0%}")
    if breach_probability >= 0.70:
        st.error("High estimated deadline risk: consider prioritizing this request or adding support capacity.")
    elif breach_probability >= 0.35:
        st.warning("Moderate estimated deadline risk: monitor this request.")
    else:
        st.success("Lower estimated deadline risk under this model and dataset.")
    st.caption("These are empirical estimates, not guarantees. Reliability depends on representative, stable real-world data.")
