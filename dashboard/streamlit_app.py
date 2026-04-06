"""
Cloud Workload Prediction — Streamlit Dashboard
Interactive dashboard covering EDA, model results, learning curves,
error analysis, latency benchmarks, auto-scaling simulation & live prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")

# Page Config:
st.set_page_config(
    page_title="Cloud Workload Prediction",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths:
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")
REPORTS    = os.path.join(ROOT, "reports")
FIGURES    = os.path.join(REPORTS, "figures")

# Cache Helpers:
@st.cache_data
def load_features():
    df = pd.read_csv(os.path.join(DATA_DIR, "combined_features.csv"),
                     parse_dates=["datetime"])
    return df.sort_values(["vm_id", "datetime"]).reset_index(drop=True)

@st.cache_data
def load_cleaned():
    df = pd.read_csv(os.path.join(DATA_DIR, "combined_cleaned.csv"),
                     parse_dates=["datetime"])
    return df.sort_values(["vm_id", "datetime"]).reset_index(drop=True)

@st.cache_resource
def load_xgb_model():
    import xgboost as xgb
    return joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))

@st.cache_data
def load_json(filename):
    path = os.path.join(REPORTS, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

@st.cache_data
def load_lstm_preds():
    path = os.path.join(MODELS_DIR, "lstm_test_predictions.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def fig_path(name):
    return os.path.join(FIGURES, name)

def show_fig(name, caption=None):
    # Display a figure if it exists, otherwise show an info message.
    p = fig_path(name)
    if os.path.exists(p):
        st.image(p, caption=caption, use_container_width=True)
    else:
        st.info(f"Figure not generated yet — run the corresponding src/ script first.  \n`{name}`")

XGB_FEATURE_COLS = [
    "CPU usage [MHZ]", "Memory usage [KB]",
    "Disk read throughput [KB/s]", "Disk write throughput [KB/s]",
    "Network received throughput [KB/s]", "Network transmitted throughput [KB/s]",
    "Memory usage [%]", "hour", "day_of_week", "is_weekend",
    "day_of_month", "week",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "cpu_util_ratio",
    "CPU usage [%]_lag1", "CPU usage [%]_lag2", "CPU usage [%]_lag3",
    "CPU usage [%]_lag4", "CPU usage [%]_lag5", "CPU usage [%]_lag6",
    "CPU usage [%]_lag9", "CPU usage [%]_lag12",
    "CPU usage [%]_roll_mean_15min", "CPU usage [%]_roll_std_15min", "CPU usage [%]_roll_max_15min",
    "CPU usage [%]_roll_mean_30min", "CPU usage [%]_roll_std_30min", "CPU usage [%]_roll_max_30min",
    "CPU usage [%]_roll_mean_60min", "CPU usage [%]_roll_std_60min", "CPU usage [%]_roll_max_60min",
    "cpu_diff1", "cpu_diff2",
    "mem_pressure", "net_total_throughput", "disk_total_throughput",
]

def sanitize(name):
    return name.replace("[", "(").replace("]", ")").replace("<", "_")

# Sidebar:
st.sidebar.title("☁️ Cloud Workload Prediction")
st.sidebar.markdown("**Bitbrains fastStorage Dataset**  \n1,250 VMs · 5-min intervals · 29 days")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "📊 EDA",
        "🤖 Model Results",
        "📈 Learning Curves",
        "🔍 Error Analysis",
        "⚡ Performance",
        "🚀 Auto-Scaling",
        "🔮 Live Prediction",
        "📋 Report",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Cloud ML Project · NCI · 2026")


# PAGE 1 — Overview:
if page == "🏠 Overview":
    st.title("☁️ Cloud Workload Prediction Dashboard")
    st.markdown(
        """
        This project predicts **VM CPU usage 30 minutes into the future** using machine learning,
        trained on the [Bitbrains fastStorage](http://gwa.ewi.tudelft.nl/datasets/gwa-t-12-bitbrains)
        dataset — real workload traces from a commercial cloud datacenter.
        """
    )
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("VMs Selected",     "10")
    col2.metric("Total Rows",       "99,348")
    col3.metric("Features Built",   "39")
    col4.metric("Forecast Horizon", "30 min")

    st.divider()

    st.subheader("End-to-End Pipeline")
    steps = [
        ("1 · Data Collection",          "Bitbrains fastStorage — 1,250 VM traces (real datacenter)"),
        ("2 · Verification",             "Schema check, row counts, separator format validation"),
        ("3 · VM Selection",             "Top 10 VMs by CPU std (most dynamic workloads)"),
        ("4 · Combine VMs",              "Single combined_raw.csv (112,836 rows)"),
        ("5 · EDA",                      "Distributions, correlations, temporal patterns, outlier detection"),
        ("6 · Cleaning",                 "Clip CPU > 100%, forward-fill, remove duplicates"),
        ("7 · Feature Engineering",      "39 features: lags, rolling stats, cyclical time encodings"),
        ("8 · Model Training",           "XGBoost · LSTM (PyTorch) · Prophet — 30-min forecast horizon"),
        ("9 · Evaluation",               "MAE / RMSE / R² / sMAPE comparison across all 3 models"),
        ("10 · Learning Curves",         "XGBoost on 20/40/60/80/100% of training data"),
        ("11 · Error Analysis",          "Per-VM, per-regime, temporal & transition error breakdown"),
        ("12 · Latency Benchmark",       "Inference speed (ms/sample) and throughput (preds/sec)"),
        ("13 · Auto-Scaling Simulation", "Reactive vs Predictive scaling using XGBoost forecasts"),
        ("14 · Cloud Storage",           "AWS S3 upload/download for data, models, reports"),
        ("15 · Dashboard",               "This Streamlit app"),
    ]
    for step, desc in steps:
        st.markdown(f"**{step}** — {desc}")

    st.divider()

    metrics = load_json("model_comparison.json")
    if metrics:
        st.subheader("Best Model at a Glance")
        models_df = pd.DataFrame(metrics["models"]).set_index("Model")
        st.dataframe(
            models_df.style
            .highlight_min(subset=["MAE", "RMSE", "sMAPE"], color="#d4edda")
            .highlight_max(subset=["R2"], color="#d4edda")
            .format({"MAE": "{:.2f}%", "RMSE": "{:.2f}%", "R2": "{:.4f}", "sMAPE": "{:.1f}%"}),
            use_container_width=True,
        )
        st.success(
            f"🏆 Best model: **XGBoost** — "
            f"MAE {models_df.loc['XGBoost','MAE']:.2f}%  |  R² {models_df.loc['XGBoost','R2']:.4f}"
        )



# PAGE 2 — EDA:
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    df = load_cleaned()
    vm_ids = sorted(df["vm_id"].unique())

    st.subheader("Dataset Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows",     f"{len(df):,}")
    c2.metric("VMs",            df["vm_id"].nunique())
    c3.metric("Date Range",     f"{df['datetime'].min().date()} → {df['datetime'].max().date()}")
    c4.metric("Missing Values", "0")

    st.divider()

    st.subheader("CPU Usage Distribution")
    col1, col2 = st.columns(2)
    with col1:
        show_fig("01_cpu_distribution.png")
    with col2:
        show_fig("02_per_vm_cpu_stats.png")

    st.divider()

    st.subheader("CPU Time Series — VM Explorer")
    selected_vm = st.selectbox("Select VM", vm_ids)
    vm_df = df[df["vm_id"] == selected_vm].set_index("datetime")

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(vm_df.index, vm_df["CPU usage [%]"], linewidth=0.8, color="steelblue")
    ax.set_title(f"VM {selected_vm} — CPU Usage over Time")
    ax.set_ylabel("CPU [%]")
    ax.set_xlabel("Date")
    ax.set_ylim(-2, 105)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.divider()

    st.subheader("Temporal Patterns & Correlations")
    col1, col2 = st.columns(2)
    with col1:
        show_fig("04_temporal_patterns.png")
    with col2:
        show_fig("05_correlation_heatmap.png")

    st.divider()

    st.subheader("Per-VM CPU Statistics")
    vm_stats = (
        df.groupby("vm_id")["CPU usage [%]"]
        .agg(Mean="mean", Std="std", Min="min", Max="max", Median="median")
        .round(2)
    )
    st.dataframe(vm_stats, use_container_width=True)


# PAGE 3 — Model Results:
elif page == "🤖 Model Results":
    st.title("🤖 Model Training Results")

    metrics = load_json("model_comparison.json")
    if not metrics:
        st.error("model_comparison.json not found. Run src/09_evaluate.py first.")
        st.stop()

    models_df = pd.DataFrame(metrics["models"]).set_index("Model")

    st.subheader("Model Comparison — All Metrics")
    cols = st.columns(3)
    colors = {"XGBoost": "🟦", "LSTM": "🟧", "Prophet": "🟩"}
    for col, (model, row) in zip(cols, models_df.iterrows()):
        with col:
            st.markdown(f"### {colors[model]} {model}")
            st.metric("MAE",   f"{row['MAE']:.2f}%")
            st.metric("RMSE",  f"{row['RMSE']:.2f}%")
            st.metric("R²",    f"{row['R2']:.4f}")
            st.metric("sMAPE", f"{row['sMAPE']:.1f}%")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        show_fig("14_model_comparison_bars.png", "MAE / RMSE / sMAPE comparison")
    with col2:
        show_fig("15_model_comparison_r2.png", "R² comparison")

    st.divider()

    st.subheader("Actual vs Predicted — XGBoost & LSTM")
    show_fig("17_xgb_lstm_vs_actual.png")

    st.divider()

    st.subheader("Residual Distributions")
    show_fig("16_residual_distributions.png")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("XGBoost — Feature Importance")
        show_fig("06_xgb_feature_importance.png")
    with col2:
        st.subheader("LSTM — Training Loss Curve")
        show_fig("09_lstm_loss_curve.png")

    st.divider()

    st.subheader("Predicted vs Actual Scatter Plot")
    show_fig("18_scatter_pred_vs_actual.png")



# PAGE 4 — Learning Curves:
elif page == "📈 Learning Curves":
    st.title("📈 Learning Curves — XGBoost")
    st.markdown(
        """
        XGBoost was trained on **20%, 40%, 60%, 80% and 100%** of the training data,
        then evaluated on the **same fixed test set** each time.
        This shows how model performance scales with data volume.
        """
    )

    lc = load_json("learning_curves.json")

    if lc:
        st.divider()
        st.subheader("Results Table")

        results_df = pd.DataFrame(lc["results"])
        baseline   = lc["naive_baseline"]

        # Add a baseline row for display:
        display_df = results_df[["fraction", "train_rows", "MAE", "RMSE", "R2", "sMAPE", "train_time_sec"]].copy()
        display_df["fraction"] = display_df["fraction"].apply(lambda x: f"{int(x*100)}%")
        display_df.columns     = ["Data Used", "Train Rows", "MAE (%)", "RMSE (%)", "R²", "sMAPE (%)", "Train Time (s)"]

        st.dataframe(
            display_df.style
            .highlight_min(subset=["MAE (%)", "RMSE (%)"], color="#d4edda")
            .highlight_max(subset=["R²"], color="#d4edda")
            .format({"MAE (%)": "{:.4f}", "RMSE (%)": "{:.4f}", "R²": "{:.4f}",
                     "sMAPE (%)": "{:.2f}", "Train Rows": "{:,}", "Train Time (s)": "{:.1f}"}),
            use_container_width=True,
        )

        # Baseline comparison:
        c1, c2, c3 = st.columns(3)
        c1.metric("Naive Baseline MAE",  f"{baseline['MAE']:.2f}%", help="Predict current CPU as 30-min forecast")
        c2.metric("Naive Baseline RMSE", f"{baseline['RMSE']:.2f}%")
        c3.metric("Naive Baseline R²",   f"{baseline['R2']:.4f}")

        st.caption("Naive baseline = use current CPU as the 30-min forecast. XGBoost significantly outperforms this across all data fractions.")

        st.divider()

    st.subheader("Learning Curve Charts")
    col1, col2 = st.columns(2)
    with col1:
        show_fig("21_learning_curves.png", "MAE, RMSE, R², and Training Time vs Dataset Size")
    with col2:
        show_fig("22_learning_curves_combined.png", "Combined learning curve (dual Y-axis)")

    st.divider()
    st.subheader("Key Takeaways")
    st.markdown("""
    - **More data consistently improves performance** — MAE drops and R² rises as training size increases
    - **Diminishing returns** appear after ~60–80% of training data
    - **Training time scales approximately linearly** with dataset size for XGBoost
    - Even at **20% of training data**, XGBoost beats the naive baseline by a large margin
    - The full dataset (100%) gives the best generalisation on the test set
    """)


# PAGE 5 — Error Analysis:
elif page == "🔍 Error Analysis":
    st.title("🔍 Comprehensive Error Analysis")
    st.markdown(
        "Deep-dive into **where and why** the XGBoost model makes errors — "
        "broken down by VM, CPU regime, hour of day, day of week, and regime transitions."
    )

    ea = load_json("error_analysis.json")

    if ea:
        st.divider()
        st.subheader("Overall Error Metrics")
        overall = ea["overall_metrics"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("MAE",    f"{overall['MAE']:.4f}%")
        c2.metric("RMSE",   f"{overall['RMSE']:.4f}%")
        c3.metric("R²",     f"{overall['R2']:.4f}")
        c4.metric("sMAPE",  f"{overall['sMAPE']:.2f}%")
        c5.metric("Bias",   f"{overall['bias']:.4f}%",
                  help=overall["bias_direction"])

        st.caption(f"Model {overall['bias_direction']} — bias = {overall['bias']:.4f}% (positive = under-predict, negative = over-predict)")

        st.divider()
        st.subheader("Per-VM Error Breakdown")
        vm_df = pd.DataFrame(ea["per_vm_errors"])
        vm_df["vm_id"] = vm_df["vm_id"].astype(int)
        vm_df = vm_df.sort_values("MAE")
        st.dataframe(
            vm_df[["vm_id","MAE","RMSE","R2","sMAPE","bias","mean_actual_cpu"]]
            .rename(columns={"vm_id":"VM","mean_actual_cpu":"Mean CPU (%)"})
            .style.highlight_min(subset=["MAE"], color="#d4edda")
                  .highlight_max(subset=["R2"],  color="#d4edda")
                  .format({"MAE":"{:.3f}%","RMSE":"{:.3f}%","R2":"{:.4f}",
                            "sMAPE":"{:.2f}%","bias":"{:.3f}%","Mean CPU (%)":"{:.2f}%"}),
            use_container_width=True,
        )

        st.divider()
        st.subheader("Error by CPU Regime")
        st.markdown("Errors split by whether the **actual CPU** at that moment was idle, moderate, or high.")
        regime_df = pd.DataFrame(ea["per_regime_errors"])
        st.dataframe(
            regime_df[["cpu_regime","n_samples","MAE","RMSE","bias","pct_of_total"]]
            .rename(columns={"cpu_regime":"CPU Regime","n_samples":"Samples","pct_of_total":"% of Test"})
            .style.highlight_min(subset=["MAE"], color="#d4edda")
                  .format({"MAE":"{:.3f}%","RMSE":"{:.3f}%","bias":"{:.3f}%",
                            "Samples":"{:,}","% of Test":"{:.1f}%"}),
            use_container_width=True,
        )

        st.divider()
        st.subheader("Regime Transition Analysis")
        trans = ea["regime_transition_analysis"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Stable Period MAE",     f"{trans['stable_mean_abs_error']:.4f}%")
        c2.metric("Transition Point MAE",  f"{trans['transition_mean_abs_error']:.4f}%")
        c3.metric("Difficulty Ratio",      f"{trans['transition_difficulty_ratio']}×",
                  help="How much harder transitions are to predict vs stable periods")
        st.caption(
            f"Regime transitions (idle→high or high→idle) are **{trans['transition_difficulty_ratio']}× harder** "
            f"to predict than stable periods. These are the model's main failure mode."
        )

        st.divider()
        st.subheader("Top 10 Worst Predictions")
        worst_df = pd.DataFrame(ea["top_worst_predictions"]).head(10)
        worst_df["vm_id"] = worst_df["vm_id"].astype(int)
        st.dataframe(
            worst_df.rename(columns={"vm_id":"VM","y_true":"Actual CPU","y_pred":"Predicted CPU",
                                      "abs_error":"|Error|","residual":"Residual"})
            .style.highlight_max(subset=["|Error|"], color="#f8d7da")
                  .format({"Actual CPU":"{:.1f}%","Predicted CPU":"{:.1f}%",
                            "|Error|":"{:.1f}%","Residual":"{:.1f}%"}),
            use_container_width=True,
        )

    st.divider()
    st.subheader("Error Analysis Charts")

    col1, col2 = st.columns(2)
    with col1:
        show_fig("23_per_vm_mae.png", "Per-VM MAE")
    with col2:
        show_fig("24_error_by_regime.png", "Error by CPU regime")

    col1, col2 = st.columns(2)
    with col1:
        show_fig("25_error_temporal.png", "Error by hour and day")
    with col2:
        show_fig("26_error_transitions.png", "Error at regime transitions")

    show_fig("27_error_heatmap_vm_hour.png", "Heatmap: VM × Hour of Day mean error")



# PAGE 6 — Performance (Latency & Throughput):
elif page == "⚡ Performance":
    st.title("⚡ Inference Latency & Throughput")
    st.markdown(
        "Measures how fast each model can produce predictions in **real-time** (single sample) "
        "and **batch** scenarios. Critical for cloud deployment SLA design."
    )

    bench = load_json("latency_benchmark.json")

    if bench:
        st.divider()
        st.subheader("Single-Sample Inference (Real-Time Use Case)")
        summary = bench["single_sample_summary"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("XGBoost Latency",    f"{summary['XGBoost']['latency_ms']:.3f} ms")
        c2.metric("XGBoost Throughput", f"{summary['XGBoost']['throughput_per_sec']:,.0f} preds/s")
        c3.metric("LSTM Latency",       f"{summary['LSTM']['latency_ms']:.3f} ms")
        c4.metric("LSTM Throughput",    f"{summary['LSTM']['throughput_per_sec']:,.0f} preds/s")

        speedup = summary["LSTM"]["latency_ms"] / summary["XGBoost"]["latency_ms"]
        st.success(f"XGBoost is **{speedup:.1f}× faster** than LSTM for single-sample inference.")

        st.divider()
        st.subheader("Batch Throughput (Scoring at Scale)")

        for model_name, results in bench["full_benchmark"].items():
            bdf = pd.DataFrame(results)
            bdf.columns = ["Batch Size", "Latency (ms)", "ms / Sample", "Throughput (preds/s)"]
            st.markdown(f"**{model_name}**")
            st.dataframe(
                bdf.style.highlight_max(subset=["Throughput (preds/s)"], color="#d4edda")
                         .format({"Batch Size": "{:,}", "Latency (ms)": "{:.2f}",
                                  "ms / Sample": "{:.5f}", "Throughput (preds/s)": "{:,.0f}"}),
                use_container_width=True,
            )

        st.divider()

    st.subheader("Latency & Throughput Charts")
    col1, col2 = st.columns(2)
    with col1:
        show_fig("19_latency_throughput.png", "Throughput and per-sample latency vs batch size")
    with col2:
        show_fig("20_single_sample_latency.png", "Single-sample real-time performance comparison")

    st.divider()
    st.subheader("Key Takeaways")
    st.markdown("""
    - **XGBoost** is significantly faster than LSTM for single-sample inference — ideal for real-time auto-scaling triggers
    - **Both models** scale well in batch mode — throughput increases substantially with larger batch sizes
    - For a 5-minute interval auto-scaling scenario, both models are fast enough to be practical
    - XGBoost's advantage is largest at **batch size = 1** (online / real-time inference)
    - LSTM's relative disadvantage shrinks at large batch sizes due to GPU-like parallelism in matrix ops
    """)



# PAGE 7 — Auto-Scaling Simulation:
elif page == "🚀 Auto-Scaling":
    st.title("🚀 Auto-Scaling Simulation")
    st.markdown(
        """
        Simulates **three scaling strategies** on the test period to show the business value
        of predictive forecasting over reactive monitoring:

        - **No Scaling** — fixed 1 instance (worst-case baseline)
        - **Reactive** — scales based on *current* CPU observation
        - **Predictive** — scales based on XGBoost *30-minute forecast*
        """
    )

    sim = load_json("autoscaling_simulation.json")

    if sim:
        cfg = sim["config"]
        st.divider()
        st.subheader("Simulation Configuration")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Scale Up At",     f"CPU > {cfg['scale_up_threshold_pct']}%")
        c2.metric("Scale Down At",   f"CPU < {cfg['scale_down_threshold_pct']}%")
        c3.metric("SLA Violation",   f"Eff. CPU > {cfg['sla_violation_threshold_pct']}%")
        c4.metric("Instance Range",  f"{cfg['min_instances']}–{cfg['max_instances']}")
        c5.metric("Cooldown Period", f"{cfg['cooldown_minutes']} min")

        st.divider()
        vm_sim = sim["single_vm_simulation"]
        st.subheader(f"Single VM Results — VM {vm_sim['vm_id']}")

        strategies = {
            "No Scaling":  vm_sim["no_scaling"],
            "Reactive":    vm_sim["reactive"],
            "Predictive":  vm_sim["predictive"],
        }

        c1, c2, c3 = st.columns(3)
        icons = {"No Scaling": "⛔", "Reactive": "🔵", "Predictive": "🟢"}
        for col, (name, data) in zip([c1, c2, c3], strategies.items()):
            with col:
                st.markdown(f"### {icons[name]} {name}")
                st.metric("SLA Violations",  data["n_sla_violations"],
                          help="Timesteps where effective CPU > SLA threshold")
                st.metric("Violation Rate",  f"{data['sla_violation_rate_pct']:.1f}%")
                st.metric("Scaling Events",  data["n_total_events"])
                st.metric("Mean Instances",  f"{data['mean_instances']:.2f}")

        # Highlight improvement:
        reactive_viols   = vm_sim["reactive"]["n_sla_violations"]
        predictive_viols = vm_sim["predictive"]["n_sla_violations"]
        improvement      = reactive_viols - predictive_viols
        if improvement > 0:
            st.success(
                f"✅ Predictive scaling reduced SLA violations by **{improvement}** "
                f"compared to reactive — that's "
                f"**{vm_sim['reactive']['sla_violation_rate_pct'] - vm_sim['predictive']['sla_violation_rate_pct']:.1f}% fewer violations**."
            )
        elif improvement == 0:
            st.info("Predictive and reactive achieved the same SLA violations for this VM.")

        st.divider()
        st.subheader("All VMs Combined")
        all_sim = sim["all_vms_simulation"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🔵 Reactive (all VMs)**")
            st.metric("SLA Violations", all_sim["reactive"]["n_sla_violations"])
            st.metric("Violation Rate", f"{all_sim['reactive']['sla_violation_rate_pct']:.1f}%")
            st.metric("Scaling Events", all_sim["reactive"]["n_total_events"])
        with c2:
            st.markdown("**🟢 Predictive (all VMs)**")
            st.metric("SLA Violations", all_sim["predictive"]["n_sla_violations"])
            st.metric("Violation Rate", f"{all_sim['predictive']['sla_violation_rate_pct']:.1f}%")
            st.metric("Scaling Events", all_sim["predictive"]["n_total_events"])

        st.divider()
        st.subheader("Key Finding")
        st.info(sim.get("key_finding", "Run src/13_autoscaling_sim.py to generate findings."))

    st.divider()
    st.subheader("Simulation Charts")
    show_fig("28_autoscaling_simulation.png",
             "Timeline: CPU usage + instance count for all 3 strategies (first 24 h)")
    show_fig("29_autoscaling_comparison.png",
             "Bar comparison: SLA violations, scaling events, mean instances")

    st.divider()
    st.subheader("Why Predictive Scaling Wins")
    st.markdown("""
    - **Reactive scaling** always responds *after* the CPU spike — the scale-up happens when SLA is already breached
    - **Predictive scaling** uses the XGBoost 30-minute forecast to trigger scale-up *before* the spike arrives
    - This 30-minute lead time is exactly what the XGBoost model provides (6 × 5-min steps ahead)
    - Fewer SLA violations = better application performance for end users
    - The trade-off: predictive may occasionally scale up unnecessarily (false positives), but this is acceptable
    """)



# PAGE 8 — Live Prediction:
elif page == "🔮 Live Prediction":
    st.title("🔮 Live CPU Prediction (XGBoost)")
    st.markdown(
        "Select a VM and a point in time from the test period. "
        "The XGBoost model will predict **CPU usage 30 minutes ahead**."
    )

    df    = load_features()
    model = load_xgb_model()

    test_df = df[
        (df["datetime"] >= "2013-08-26") &
        (df["datetime"] <  "2013-09-02")
    ].copy()
    test_df["target"] = test_df.groupby("vm_id")["CPU usage [%]"].shift(-6)
    test_df = test_df.dropna(subset=["target"]).reset_index(drop=True)

    vm_ids = sorted(test_df["vm_id"].unique())

    col1, col2 = st.columns([1, 2])
    with col1:
        selected_vm = st.selectbox("VM ID", vm_ids)
        vm_data     = test_df[test_df["vm_id"] == selected_vm].reset_index(drop=True)
        n_steps     = len(vm_data)
        step_idx    = st.slider("Time step", 0, n_steps - 1, n_steps // 2)
        row         = vm_data.iloc[step_idx]

        st.markdown(f"**Selected time:** `{row['datetime']}`")
        st.markdown(f"**Current CPU:** `{row['CPU usage [%]']:.1f}%`")
        st.markdown(f"**Actual CPU +30 min:** `{row['target']:.1f}%`")

    with col2:
        safe_cols = {c: sanitize(c) for c in XGB_FEATURE_COLS}
        X_row = (row[XGB_FEATURE_COLS].to_frame().T
                 .rename(columns=safe_cols)
                 .astype(float))
        pred   = float(np.clip(model.predict(X_row)[0], 0, 100))
        actual = float(row["target"])
        error  = abs(pred - actual)

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted CPU +30 min", f"{pred:.1f}%")
        m2.metric("Actual CPU +30 min",    f"{actual:.1f}%")
        m3.metric("Absolute Error",        f"{error:.1f}%",
                  delta=f"{'↑ over' if pred > actual else '↓ under'} by {error:.1f}%",
                  delta_color="inverse")

    st.divider()
    st.subheader("CPU Context Window (last 2 h) + Prediction")
    window_start = max(0, step_idx - 24)
    context = vm_data.iloc[window_start: step_idx + 1]

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(context["datetime"], context["CPU usage [%]"],
            color="steelblue", linewidth=1.2, label="Historical CPU")
    ax.axvline(row["datetime"], color="gray", linestyle="--", linewidth=0.8)

    pred_time = row["datetime"] + pd.Timedelta(minutes=30)
    ax.scatter([pred_time], [pred],   color="tomato",    s=80, zorder=5,
               label=f"Predicted ({pred:.1f}%)")
    ax.scatter([pred_time], [actual], color="steelblue", s=80, zorder=5,
               marker="^", label=f"Actual ({actual:.1f}%)")

    ax.set_ylabel("CPU Usage [%]")
    ax.set_ylim(-2, 108)
    ax.set_title(f"VM {selected_vm} — CPU context with 30-min forecast")
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.divider()
    st.subheader("Full Test Period — Predictions vs Actual")
    safe_cols = {c: sanitize(c) for c in XGB_FEATURE_COLS}
    X_vm      = vm_data[XGB_FEATURE_COLS].rename(columns=safe_cols).astype(float)
    vm_preds  = np.clip(model.predict(X_vm), 0, 100)

    fig2, ax2 = plt.subplots(figsize=(13, 4))
    ax2.plot(vm_data["datetime"], vm_data["target"], label="Actual",    color="steelblue",  linewidth=0.9)
    ax2.plot(vm_data["datetime"], vm_preds,           label="Predicted", color="tomato",      linewidth=0.9, linestyle="--")
    ax2.axvline(vm_data.iloc[step_idx]["datetime"], color="gray", linestyle=":", linewidth=1)
    ax2.set_ylabel("CPU Usage [%]")
    ax2.set_ylim(-2, 108)
    ax2.set_title(f"VM {selected_vm} — Full test period (dashed = selected point)")
    ax2.legend()
    st.pyplot(fig2, use_container_width=True)
    plt.close()



# PAGE 9 — Report:
elif page == "📋 Report":
    st.title("📋 Project Summary Report")

    metrics = load_json("model_comparison.json")
    lc      = load_json("learning_curves.json")
    ea      = load_json("error_analysis.json")
    bench   = load_json("latency_benchmark.json")
    sim     = load_json("autoscaling_simulation.json")

    models_df = pd.DataFrame(metrics["models"]).set_index("Model") if metrics else None

    st.markdown("## Cloud Workload Prediction using Machine Learning")

    st.markdown("""
    ### 1. Objective
    Predict cloud VM CPU utilisation **30 minutes ahead** (6 × 5-min steps) to enable
    proactive auto-scaling decisions in a cloud datacenter environment.

    ### 2. Dataset
    | Property | Value |
    |----------|-------|
    | Source | Bitbrains fastStorage |
    | Total VMs available | 1,250 |
    | VMs selected | 10 (highest CPU variability) |
    | Sampling interval | 5 minutes |
    | Period | Aug 12 – Sep 11, 2013 (29 days) |
    | Total rows (post-cleaning) | 99,348 |
    | Missing values | 0 |

    ### 3. Features Engineered (39 total)
    | Category | Examples |
    |----------|---------|
    | Time-based | hour, day_of_week, is_weekend |
    | Cyclical encodings | hour_sin/cos, dow_sin/cos |
    | Lag features | CPU lag 1–12 steps (5–60 min) |
    | Rolling statistics | Mean / Std / Max over 15, 30, 60 min |
    | Rate of change | cpu_diff1, cpu_diff2 |
    | Resource ratios | cpu_util_ratio, mem_pressure |
    """)

    st.markdown("### 4. Model Comparison")
    if models_df is not None:
        st.dataframe(
            models_df.style
            .highlight_min(subset=["MAE", "RMSE", "sMAPE"], color="#d4edda")
            .highlight_max(subset=["R2"], color="#d4edda")
            .format({"MAE": "{:.2f}%", "RMSE": "{:.2f}%", "R2": "{:.4f}", "sMAPE": "{:.1f}%"}),
            use_container_width=True,
        )

    st.markdown("### 5. Learning Curves")
    if lc:
        best = min(lc["results"], key=lambda x: x["MAE"])
        st.markdown(
            f"XGBoost trained on 20–100% of data. Best result at **{int(best['fraction']*100)}%** "
            f"of training data — MAE = **{best['MAE']:.4f}%**, R² = **{best['R2']:.4f}**. "
            f"Naive baseline MAE = {lc['naive_baseline']['MAE']:.2f}% — XGBoost beats it at every fraction."
        )

    st.markdown("### 6. Error Analysis")
    if ea:
        overall = ea["overall_metrics"]
        trans   = ea["regime_transition_analysis"]
        st.markdown(
            f"- Overall bias: **{overall['bias']:.4f}%** ({overall['bias_direction']})\n"
            f"- Hardest regime: **High CPU (>70%)** — model under-predicts at saturation\n"
            f"- Regime transitions are **{trans['transition_difficulty_ratio']}× harder** to predict "
            f"than stable periods (transition MAE = {trans['transition_mean_abs_error']:.2f}% vs "
            f"stable MAE = {trans['stable_mean_abs_error']:.2f}%)\n"
            f"- Error is highest during early morning hours (2–6 AM) due to low-data regime shifts"
        )

    st.markdown("### 7. Latency & Throughput")
    if bench:
        s = bench["single_sample_summary"]
        speedup = s["LSTM"]["latency_ms"] / s["XGBoost"]["latency_ms"]
        st.markdown(
            f"| Model | Single-Sample Latency | Throughput |\n"
            f"|---|---|---|\n"
            f"| XGBoost | {s['XGBoost']['latency_ms']:.3f} ms | {s['XGBoost']['throughput_per_sec']:,.0f} preds/s |\n"
            f"| LSTM | {s['LSTM']['latency_ms']:.3f} ms | {s['LSTM']['throughput_per_sec']:,.0f} preds/s |\n\n"
            f"XGBoost is **{speedup:.1f}× faster** for real-time inference — critical for low-latency scaling triggers."
        )

    st.markdown("### 8. Auto-Scaling Simulation")
    if sim:
        st.markdown(sim.get("key_finding", ""))
        st.markdown(
            "Predictive scaling uses the 30-minute XGBoost forecast to trigger scale-up events "
            "**before** CPU spikes arrive, reducing SLA violations compared to reactive scaling "
            "which always responds after the fact."
        )

    st.markdown("""
    ### 9. Conclusions
    - **XGBoost** is the best model: MAE ≈ 6%, R² ≈ 0.85 — practical for production auto-scaling
    - **LSTM** is competitive but 10× slower for real-time inference
    - **Prophet** fails on high-frequency CPU data — not designed for 5-min spike patterns
    - **Predictive scaling** with XGBoost reduces SLA violations vs reactive monitoring
    - **More training data** consistently improves XGBoost — near-plateau at 80%
    - **Regime transitions** (idle → high CPU) are the primary failure mode

    ### 10. Future Work
    - Deploy FastAPI on AWS ECS / Fargate for production-scale inference
    - Extend to all 1,250 VMs with per-VM or multi-task models
    - Implement online learning to adapt to workload regime changes
    - Explore Transformer / Temporal Fusion Transformer for multi-step forecasting
    - Add real-time data ingestion pipeline with Kafka or AWS Kinesis
    """)

    st.divider()
    st.subheader("All Generated Figures")
    if os.path.exists(FIGURES):
        fig_files = sorted([f for f in os.listdir(FIGURES) if f.endswith(".png")])
        cols = st.columns(2)
        for i, fname in enumerate(fig_files):
            with cols[i % 2]:
                st.image(os.path.join(FIGURES, fname),
                         caption=fname.replace("_", " ").replace(".png", ""),
                         use_container_width=True)
    else:
        st.info("No figures found. Run the src/ scripts to generate them.")
