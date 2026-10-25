"""Streamlit interface for response-time regression."""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from src.data import FEATURE_COLUMNS, TARGET, make_demo_data
from src.evaluate import evaluate_models

st.set_page_config(page_title="Response Time Prediction", page_icon="📈", layout="wide")
st.title("Response Time Prediction Using Regression Models")
st.caption("College mathematics project • estimates time until first response, in minutes")

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
    st.info("Without an upload, the app uses synthetic demo data. Demo results are not real-world evidence.")

try:
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        source_label = "Uploaded CSV"
    else:
        raw = make_demo_data()
        source_label = "Synthetic demonstration dataset"
    metrics, models, predictions = evaluate_models(raw)
except Exception as exc:
    st.error(f"Could not process dataset: {exc}")
    st.stop()

st.caption(f"Data source: {source_label}")
a, b, c = st.columns(3)
a.metric("Usable rows", f"{len(raw):,}")
b.metric("Models compared", str(len(metrics)))
c.metric("Best test MAE", f"{metrics.iloc[0]['MAE (minutes)']:.2f} min")
st.warning("Metrics describe this dataset and split only. Synthetic-data results do not demonstrate real-world performance.")

st.subheader("Model comparison on held-out test data")
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
    st.subheader("Residual distribution")
    fig, ax = plt.subplots()
    ax.hist(predictions["Residual (actual - predicted)"], bins=24)
    ax.set_xlabel("Residual (minutes)")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Try a prediction")
model_name = st.selectbox("Regression model", list(models.keys()), index=list(models.keys()).index(best_name))
features = [col for col in FEATURE_COLUMNS if col in raw.columns and col != TARGET]
if not features:
    st.error("No supported predictor columns were found.")
    st.stop()

with st.form("prediction_form"):
    values = {}
    columns = st.columns(2)
    for i, feature in enumerate(features):
        with columns[i % 2]:
            if feature in raw.select_dtypes(include=["object", "category"]).columns:
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
    submitted = st.form_submit_button("Predict response time")

if submitted:
    input_frame = pd.DataFrame([values], columns=features)
    estimate = float(models[model_name].predict(input_frame)[0])
    st.success(f"Estimated first response time: {max(0.0, estimate):.1f} minutes")
    st.caption("This is an estimate, not a guarantee. Reliability depends on representative training data.")
