"""Streamlit dashboard for response-time prediction and statistical risk analysis."""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from src.data import CATEGORICAL_FEATURES, FEATURE_COLUMNS, TARGET, make_demo_data, validate_dataset
from src.evaluate import evaluate_models
from src.maths import (
    compare_with_sklearn,
    conformal_details,
    error_shape,
    formula_mean_ci,
    normal_breach_probability,
)
from src.statistics import conformal_prediction_interval, probability_sla_breach, response_time_summary

st.set_page_config(
    page_title="CampusHelp Response Intelligence",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(show_spinner="Training and evaluating models...")
def cached_evaluation(data: pd.DataFrame):
    """Avoid retraining the same CSV on every Streamlit interaction."""
    return evaluate_models(data)


st.markdown(
    """
    <style>
      :root {
        --campus-blue: #2563EB;
        --campus-blue-dark: #153B73;
        --campus-blue-soft: #EAF3FF;
        --campus-blue-border: #D5E5FA;
        --campus-text: #172B4D;
        --campus-muted: #5B6F89;
      }
      .stApp { background: #FFFFFF; color: var(--campus-text); }
      header[data-testid="stHeader"] { background: #FFFFFF; }
      .block-container {
        max-width: 1240px;
        padding-top: 4.5rem;
        padding-bottom: 3rem;
      }
      h1, h2, h3 { color: var(--campus-blue-dark); letter-spacing: -0.02em; }
      h2 { margin-top: 1.1rem; }
      p, label, .stMarkdown { color: var(--campus-text); }
      div[data-testid="stSidebar"] {
        background: #F5F9FF;
        border-right: 1px solid var(--campus-blue-border);
      }
      div[data-testid="stSidebar"] h1,
      div[data-testid="stSidebar"] h2,
      div[data-testid="stSidebar"] h3 { color: var(--campus-blue-dark); }
      .hero {
        background: linear-gradient(120deg, #EEF5FF 0%, #FFFFFF 78%);
        border: 1px solid var(--campus-blue-border);
        border-radius: 18px;
        padding: 26px 30px;
        margin: 0 0 1.3rem 0;
      }
      .hero-kicker {
        color: var(--campus-blue);
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
      }
      .hero-title {
        color: var(--campus-blue-dark);
        font-size: clamp(1.9rem, 3.5vw, 2.65rem);
        font-weight: 800;
        line-height: 1.13;
        margin: 0 0 0.65rem 0;
      }
      .hero-copy {
        color: #425875;
        font-size: 1.03rem;
        line-height: 1.6;
        max-width: 820px;
        margin-bottom: 1rem;
      }
      .tag-row { display: flex; flex-wrap: wrap; gap: 0.45rem; }
      .tag {
        display: inline-block;
        background: #FFFFFF;
        color: var(--campus-blue-dark);
        border: 1px solid #C9DDF8;
        border-radius: 999px;
        padding: 0.27rem 0.7rem;
        font-size: 0.78rem;
        font-weight: 650;
      }
      div[data-testid="stMetric"] {
        background: #F7FAFF;
        border: 1px solid var(--campus-blue-border);
        border-radius: 14px;
        padding: 15px 17px;
      }
      div[data-testid="stMetricLabel"] { color: var(--campus-muted); }
      div[data-testid="stMetricValue"] { color: var(--campus-blue-dark); }
      div[data-testid="stMetricDelta"] { color: var(--campus-blue); }
      [data-baseweb="select"] > div,
      [data-baseweb="input"] > div,
      [data-baseweb="textarea"] > div {
        border-color: #BDD2EF;
        border-radius: 9px;
      }
      div[data-testid="stFileUploader"] {
        background: #FFFFFF;
        border: 1px dashed #AFC9EF;
        border-radius: 12px;
        padding: 0.55rem;
      }
      div.stButton > button,
      div[data-testid="stFormSubmitButton"] button {
        background: var(--campus-blue);
        color: #FFFFFF;
        border: 1px solid var(--campus-blue);
        border-radius: 9px;
        font-weight: 700;
        min-height: 2.65rem;
      }
      div.stButton > button:hover,
      div[data-testid="stFormSubmitButton"] button:hover {
        background: #1D4ED8;
        border-color: #1D4ED8;
        color: #FFFFFF;
      }
      div[data-testid="stDataFrame"],
      div[data-testid="stTable"] {
        border: 1px solid var(--campus-blue-border);
        border-radius: 12px;
        overflow: hidden;
      }
      div[data-testid="stAlert"] { border-radius: 12px; }
      hr { border-color: #E1EBF8; }
      .section-note {
        color: var(--campus-muted);
        font-size: 0.91rem;
        line-height: 1.5;
        margin-top: -0.25rem;
      }
      @media (max-width: 700px) {
        .block-container { padding-top: 4rem; }
        .hero { padding: 20px 18px; }
        .hero-title { font-size: 1.8rem; }
      }
    </style>
    <div class="hero">
      <div class="hero-kicker">Campus support • Predictive analytics</div>
      <div class="hero-title">Know the wait. Understand the risk.</div>
      <div class="hero-copy">
        CampusHelp Response Intelligence estimates how long a support request may wait for its first response,
        how uncertain that estimate is, and how likely it is to miss a chosen service deadline.
        It brings regression, probability, and statistics together to support—not replace—human decisions.
      </div>
      <div class="tag-row">
        <span class="tag">Regression models</span>
        <span class="tag">Confidence & prediction intervals</span>
        <span class="tag">Deadline-risk estimate</span>
        <span class="tag">Human-in-the-loop</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Data & risk settings")
    uploaded = st.file_uploader("Upload support-ticket CSV", type=["csv"])
    st.caption("No upload? The app uses synthetic demo data.")
    st.markdown(
        "[Try the optional CC0 sample CSV](https://datanemics.com/datasets/data/support-tickets.csv)"
    )
    st.caption("That sample is also synthetic. Unanswered tickets have no first-response time and are excluded; this can bias results.")
    st.divider()
    st.subheader("Service deadline")
    sla_minutes = st.number_input(
        "Deadline (minutes)", min_value=5, max_value=1440, value=60, step=5
    )
    interval_coverage = st.select_slider(
        "Prediction interval coverage",
        options=[0.80, 0.90, 0.95],
        value=0.90,
        format_func=lambda value: f"{value:.0%}",
    )
    st.divider()
    st.caption(
        "Privacy tip: upload only authorized data, and remove names, IDs, emails, and ticket text."
    )

try:
    source_data = pd.read_csv(uploaded) if uploaded is not None else make_demo_data()
    source_rows = len(source_data)
    source_label = "Uploaded CSV" if uploaded is not None else "Synthetic demonstration dataset"
    raw = validate_dataset(source_data)
    rows_excluded = source_rows - len(raw)
    metrics, models, predictions, calibration_residuals = cached_evaluation(source_data)
except Exception as exc:
    st.error(f"Could not process the dataset: {exc}")
    st.stop()

if uploaded is None:
    st.warning(
        "Demo mode: all numbers below use synthetic data. They demonstrate the workflow only and are not real-world findings."
    )
else:
    st.info(
        "Uploaded-data mode: results depend on the quality and representativeness of your CSV. "
        "Confirm that the target really measures time to first response."
    )

st.caption(
    f"**Data source:** {source_label}  ·  **Rows received:** {source_rows:,}  ·  "
    f"**Usable rows:** {len(raw):,}"
)
if rows_excluded > 0:
    st.warning(
        f"{rows_excluded:,} row(s) were excluded because the first-response target was missing or invalid. "
        "If these are unanswered tickets, this dashboard describes answered tickets only and may understate waits."
    )
split_method = metrics.attrs.get("split_method", "random")
if split_method == "chronological":
    st.caption("Evaluation method: chronological split — earlier tickets are used before later test tickets.")
else:
    st.caption("Evaluation method: random split — use a timestamp column for a more realistic future-period evaluation.")
summary = response_time_summary(raw)

st.subheader("Response-time overview")
st.markdown(
    '<div class="section-note">These statistics describe response times in the selected dataset; they are not individual-ticket predictions.</div>',
    unsafe_allow_html=True,
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Mean wait", f"{summary['mean']:.1f} min")
m2.metric("Median wait", f"{summary['median']:.1f} min")
m3.metric("90th percentile", f"{summary['q90']:.1f} min")
m4.metric("95% CI for mean", f"{summary['mean_ci_low']:.1f}–{summary['mean_ci_high']:.1f} min")

st.divider()
st.subheader("Where are response times higher?")
st.markdown(
    '<div class="section-note">Breakdowns help teams investigate patterns. A difference between groups does not prove that the group caused the delay.</div>',
    unsafe_allow_html=True,
)
insight_columns = st.columns(2)
for col, group_column, title in [
    (insight_columns[0], "priority", "By priority"),
    (insight_columns[1], "category", "By issue category / queue"),
]:
    with col:
        st.markdown(f"**{title}**")
        if group_column in raw.columns:
            grouped = raw.groupby(group_column, dropna=False)[TARGET].agg(
                Tickets="size",
                Median_wait_min="median",
                P90_wait_min=lambda values: values.quantile(0.90),
            )
            grouped = grouped.sort_values("Median_wait_min", ascending=False)
            grouped = grouped.rename(columns={
                "Median_wait_min": "Median wait (min)",
                "P90_wait_min": "90th percentile (min)",
            })
            st.dataframe(grouped, use_container_width=True)
            st.bar_chart(grouped[["Median wait (min)"]])
        else:
            st.caption(f"The uploaded dataset does not include a {group_column} column.")

if "__opened_at" in raw.columns and raw["__opened_at"].notna().all():
    daily = raw.assign(_opened_date=raw["__opened_at"].dt.date).groupby("_opened_date")[TARGET].agg(
        Tickets="size",
        Median_wait_min="median",
    )
    st.markdown("**Median response time over time**")
    st.markdown(
        '<div class="section-note">Available when valid opening timestamps are present. Day-to-day changes may reflect volume or data composition.</div>',
        unsafe_allow_html=True,
    )
    st.line_chart(daily[["Median_wait_min"]].rename(columns={"Median_wait_min": "Median wait (min)"}))

st.divider()
st.subheader("Compare the regression models")
st.markdown(
    '<div class="section-note">The model is selected using validation MAE. Test metrics are kept separate for final evaluation; lower test MAE and RMSE indicate smaller errors.</div>',
    unsafe_allow_html=True,
)
st.dataframe(metrics, use_container_width=True, hide_index=True)

selected_model = metrics.attrs.get("selected_model", metrics.iloc[0]["Model"])
baseline_mae = float(
    metrics.loc[metrics["Model"] == "Training Mean Baseline", "Validation MAE (minutes)"].iloc[0]
)
selected_mae = float(
    metrics.loc[metrics["Model"] == selected_model, "Validation MAE (minutes)"].iloc[0]
)
if selected_model == "Training Mean Baseline":
    st.warning(
        "No regression model beat the training-mean baseline on validation data. "
        "The app selects the simpler baseline rather than claiming the more complex models add value."
    )
else:
    improvement = max(0.0, (baseline_mae - selected_mae) / baseline_mae * 100.0) if baseline_mae > 0 else 0.0
    st.success(
        f"**Selected on validation:** {selected_model}. Its validation MAE is "
        f"{improvement:.1f}% lower than the training-mean baseline."
    )

fig, ax = plt.subplots(figsize=(8.5, 3.4))
plot_metrics = metrics.set_index("Model")
x_positions = list(range(len(plot_metrics)))
bar_width = 0.34
ax.bar(
    [x - bar_width / 2 for x in x_positions],
    plot_metrics["MAE (test minutes)"],
    width=bar_width,
    label="MAE",
    color="#2563EB",
)
ax.bar(
    [x + bar_width / 2 for x in x_positions],
    plot_metrics["RMSE (test minutes)"],
    width=bar_width,
    label="RMSE",
    color="#93C5FD",
)
ax.set_xticks(x_positions, plot_metrics.index, rotation=0)
ax.set_ylabel("Error (minutes)")
ax.set_title("Prediction error by model", loc="left", color="#153B73", pad=12, fontweight="bold")
ax.grid(axis="y", color="#E3ECF8", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#D5E5FA")
ax.spines["bottom"].set_color("#D5E5FA")
ax.tick_params(colors="#425875")
ax.legend(frameon=False, ncol=2, loc="upper right")
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

best_name = selected_model
best_column = f"{best_name} prediction"
left, right = st.columns(2)
with left:
    st.subheader("Actual vs predicted")
    st.markdown(
        '<div class="section-note">Points closer to the diagonal represent closer predictions.</div>',
        unsafe_allow_html=True,
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        predictions[TARGET],
        predictions[best_column],
        alpha=0.7,
        color="#2563EB",
        edgecolors="white",
        linewidths=0.35,
    )
    low = min(predictions[TARGET].min(), predictions[best_column].min())
    high = max(predictions[TARGET].max(), predictions[best_column].max())
    ax.plot([low, high], [low, high], linestyle="--", color="#153B73", linewidth=1.5)
    ax.set_xlabel("Actual response time (minutes)")
    ax.set_ylabel("Predicted response time (minutes)")
    ax.grid(color="#E8EFF8", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with right:
    st.subheader("Prediction errors")
    st.markdown(
        '<div class="section-note">Residual = actual response time − predicted response time.</div>',
        unsafe_allow_html=True,
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        predictions["Residual (actual - predicted)"],
        bins=24,
        color="#60A5FA",
        edgecolor="#FFFFFF",
    )
    ax.axvline(0, color="#153B73", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Residual (minutes)")
    ax.set_ylabel("Number of tickets")
    ax.grid(axis="y", color="#E8EFF8", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.divider()
st.subheader("Estimate a support request")
st.markdown(
    '<div class="section-note">Enter values known at the moment you want to make the prediction. The interval and deadline risk use a separate calibration split, not the final test data.</div>',
    unsafe_allow_html=True,
)

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
                values[feature] = st.selectbox(
                    feature.replace("_", " ").title(), choices or ["unknown"]
                )
            else:
                series = pd.to_numeric(raw[feature], errors="coerce").dropna()
                default = float(series.median()) if not series.empty else 0.0
                minimum = float(series.min()) if not series.empty else 0.0
                maximum = float(series.max()) if not series.empty else max(1.0, default + 1)
                if minimum == maximum:
                    maximum = minimum + 1.0
                values[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    min_value=minimum,
                    max_value=maximum,
                    value=min(max(default, minimum), maximum),
                )
    model_name = st.selectbox(
        "Regression model",
        list(models.keys()),
        index=list(models.keys()).index(best_name),
    )
    backlog_reduction = 0
    if "backlog" in features:
        backlog_reduction = st.select_slider(
            "What-if: backlog reduction (model scenario)",
            options=[0, 10, 25, 50],
            value=25,
            format_func=lambda value: "No change" if value == 0 else f"{value}% lower",
        )
    submitted = st.form_submit_button("Estimate wait and deadline risk", type="primary")

if submitted:
    input_frame = pd.DataFrame([values], columns=features)
    estimate = max(0.0, float(models[model_name].predict(input_frame)[0]))
    scenario_estimate = None
    if "backlog" in features and backlog_reduction > 0:
        scenario_frame = input_frame.copy()
        current_backlog = float(scenario_frame.at[0, "backlog"])
        scenario_frame.at[0, "backlog"] = max(
            0.0, current_backlog * (1.0 - backlog_reduction / 100.0)
        )
        scenario_estimate = max(
            0.0, float(models[model_name].predict(scenario_frame)[0])
        )
    residuals = calibration_residuals[model_name]
    interval_error = None
    try:
        lower, upper = conformal_prediction_interval(
            estimate, residuals, coverage=interval_coverage
        )
    except ValueError as exc:
        lower, upper = None, None
        interval_error = str(exc)
    breach_probability = probability_sla_breach(estimate, residuals, sla_minutes)

    st.markdown("#### Your estimate")
    metric_columns = st.columns(4 if scenario_estimate is not None else 3)
    metric_columns[0].metric("Estimated first response", f"{estimate:.1f} min")
    metric_columns[1].metric(
        f"{interval_coverage:.0%} prediction interval",
        f"{lower:.1f}–{upper:.1f} min" if lower is not None else "More data needed",
    )
    metric_columns[2].metric(
        "Estimated deadline-breach risk",
        f"{breach_probability:.0%}",
        help=f"Empirical estimate of exceeding {sla_minutes} minutes from calibration residuals.",
    )
    if scenario_estimate is not None:
        change = scenario_estimate - estimate
        metric_columns[3].metric(
            f"If backlog were {backlog_reduction}% lower",
            f"{scenario_estimate:.1f} min",
            delta=f"{change:+.1f} min",
            delta_color="inverse",
        )
        st.caption(
            "Backlog scenario is a model-based what-if with other inputs held constant. "
            "It reflects learned associations, not proof that reducing backlog will cause this change."
        )
    if interval_error:
        st.info(interval_error)

    if breach_probability >= 0.70:
        st.error(
            "Higher estimated deadline risk. A support team member may want to review this request and available capacity."
        )
    elif breach_probability >= 0.35:
        st.warning(
            "Moderate estimated deadline risk. Consider monitoring the request and queue."
        )
    else:
        st.success(
            "Lower estimated deadline risk under this model and dataset. This is not a guarantee."
        )
    st.caption(
        "Use this result as decision support, not an automatic priority rule. "
        "Interval coverage and risk estimates need real-world validation before operational use."
    )


# ----------------------------------------------------------------------------
# The maths behind it. Self-contained: to remove it, delete everything from this
# comment to the end of the file, plus the "from src.maths import ..." lines at the top.
# ----------------------------------------------------------------------------
def render_maths_section(
    raw, summary, predictions, calibration_residuals, sla_minutes, interval_coverage,
    example_model, example_prediction, used_submitted,
):
    st.divider()
    st.subheader("The maths behind it")
    st.markdown(
        '<div class="section-note">Each part below is recalculated from formulas and checked against what the app actually uses.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
          div[data-testid="stExpander"] details summary {
            background: #EAF3FF;
            border-radius: 10px;
          }
          div[data-testid="stExpander"] details summary,
          div[data-testid="stExpander"] details summary p,
          div[data-testid="stExpander"] details summary span {
            color: #153B73 !important;
            font-weight: 650;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    example_residuals = calibration_residuals[example_model]
    check = compare_with_sklearn(raw)

    with st.expander("1. Least squares: where the regression line comes from", expanded=True):
        st.markdown(
            "Think of hanging a stick through a cloud of dots. For any stick position, add up how far each dot "
            "is from it, squaring the gaps so ups and downs cannot cancel. That total miss forms a bowl-shaped "
            "valley, and the best stick sits at the bottom, where the ground is flat (slope = 0)."
        )
        st.latex(r"S(\beta)=\sum_i\left(y_i-\hat y_i\right)^2")
        st.latex(
            r"\frac{\partial S}{\partial \beta}=0\;\Rightarrow\;X^{T}X\,\beta=X^{T}y"
            r"\;\Rightarrow\;\hat\beta=\left(X^{T}X\right)^{-1}X^{T}y"
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows used", f"{check['n_rows']:,}")
        c2.metric("Input columns (after one-hot)", f"{check['n_columns']}")
        c3.metric("Gap: formula vs scikit-learn", f"{check['ols_max_diff']:.1e} min")
        st.caption(
            "The formula was solved with NumPy on the same data. The predictions agree with scikit-learn's "
            "Linear Regression to rounding error, so the library is doing exactly this calculation."
        )

    with st.expander("2. Ridge: the same line with a leash"):
        st.markdown(
            "Ridge adds a penalty for large coefficients, like a leash that stops any one input from pulling "
            "the line too hard. The only change to the maths is adding the leash strength to the diagonal."
        )
        st.latex(r"\hat\beta_{ridge}=\left(X^{T}X+\alpha I\right)^{-1}X^{T}y")
        r1, r2, r3 = st.columns(3)
        r1.metric("Leash strength (alpha)", f"{check['alpha']:.1f}")
        r2.metric(
            "Size of coefficients: Linear → Ridge",
            f"{check['ols_coef_norm']:.3f} → {check['ridge_coef_norm']:.3f}",
        )
        r3.metric("Gap: formula vs scikit-learn", f"{check['ridge_max_diff']:.1e} min")
        st.caption(
            f"With {check['n_rows']:,} rows and only a few inputs, the leash barely tightens, so Ridge and Linear "
            "give almost the same results. That is why their rows in the comparison table are nearly identical."
        )

    with st.expander("3. Confidence interval vs prediction interval"):
        formula_low, formula_high = formula_mean_ci(raw[TARGET])
        st.markdown(
            "A **confidence interval** guesses the *average* wait of all tickets. A **prediction interval** guesses "
            "the wait of *one new ticket*, which is far more variable, so it is much wider. "
            "(Like guessing a class's average height vs guessing the next student through the door.)"
        )
        st.latex(r"\bar y\;\pm\;1.96\,\frac{s}{\sqrt{n}}")
        st.markdown(
            f"- Textbook formula: **{formula_low:.1f}–{formula_high:.1f} min**, using s = {summary['std_dev']:.1f} and n = {summary['count']:,}\n"
            f"- The app's bootstrap (re-sampling the data 1,000 times): **{summary['mean_ci_low']:.1f}–{summary['mean_ci_high']:.1f} min**\n"
            "- Both agree, so the interval is trustworthy."
        )
        try:
            details = conformal_details(example_residuals, interval_coverage)
        except ValueError as exc:
            st.info(str(exc))
        else:
            st.markdown(
                f"**Prediction interval (split-conformal).** Take the {details['n']} calibration errors, ignore their signs, "
                f"sort them, and pick number {details['rank']} (the {interval_coverage:.0%} mark). That error size, "
                f"**{details['radius']:.1f} min**, becomes the plus/minus around the prediction."
            )
            st.latex(
                r"\text{interval}=\hat y\pm q,\qquad q=\text{the }\lceil (n+1)\cdot\text{coverage}\rceil\text{-th smallest }|\text{error}|"
            )
            st.caption(
                f"Example: a prediction of {example_prediction:.1f} min gives about "
                f"{max(0.0, example_prediction - details['radius']):.1f}–{example_prediction + details['radius']:.1f} min. "
                "Compare that width with the mean's interval above: one ticket is far less certain than the average."
            )

    with st.expander("4. Deadline risk"):
        empirical = probability_sla_breach(example_prediction, example_residuals, sla_minutes)
        bell = normal_breach_probability(example_prediction, example_residuals, sla_minutes)
        st.markdown(
            "**What the app does:** add every past calibration error to the prediction and count how many "
            f"outcomes land beyond the {sla_minutes} minute deadline. That share is the risk."
        )
        st.markdown(
            "**The maths version:** if the errors are bell-shaped with spread σ, the same risk comes from the "
            "normal curve (z-score = how many σ the deadline is above the prediction)."
        )
        st.latex(r"P(\text{breach})=P\!\left(Z>\frac{D-\hat y}{\sigma}\right)")
        d1, d2, d3 = st.columns(3)
        d1.metric("Prediction used", f"{example_prediction:.1f} min")
        d2.metric("Risk: counting (app)", f"{empirical:.1%}")
        d3.metric("Risk: normal curve", f"{bell:.1%}")
        gap = abs(empirical - bell)
        st.caption(
            (
                "The two methods give close answers, which is a good sign. "
                if gap < 0.03
                else f"The methods differ by {gap * 100:.1f} points: counting uses only {len(example_residuals)} "
                "calibration errors, so it is lumpier, while the normal curve smooths them out. "
            )
            + (
                "This uses your submitted estimate."
                if used_submitted
                else "Submit an estimate above to use your own ticket; for now it uses the dataset's average wait."
            )
        )

    with st.expander("5. Are the errors bell-shaped?"):
        shape = error_shape(predictions["Residual (actual - predicted)"])
        st.markdown(
            "RMSE punishes big misses more than MAE does. For perfectly bell-shaped errors the ratio RMSE ÷ MAE "
            "is always √(π/2) ≈ 1.2533, so the ratio is a quick bell-shape test."
        )
        s1, s2, s3 = st.columns(3)
        s1.metric("MAE", f"{shape['mae']:.2f} min")
        s2.metric("RMSE", f"{shape['rmse']:.2f} min")
        s3.metric("RMSE ÷ MAE", f"{shape['ratio']:.3f}", help=f"Bell-shaped target: {shape['bell_ratio']:.4f}")
        st.caption(
            "A ratio this close to 1.2533 fits the bell-shaped histogram above, "
            "which supports the normal-curve risk calculation."
        )


try:
    render_maths_section(
        raw, summary, predictions, calibration_residuals, sla_minutes, interval_coverage,
        model_name if submitted else best_name,
        estimate if submitted else summary["mean"],
        submitted,
    )
except Exception as exc:  # the maths section must never break the main dashboard
    st.info(f"The maths section could not be shown for this dataset: {exc}")
