import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")   # non-interactive backend before any other plt import

# XGBoost must be imported (and its model loaded) before sklearn.preprocessing
# to avoid a segfault on macOS caused by conflicting C++ runtimes.
import xgboost as xgb
import joblib

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH   = os.path.join(PROJECT_ROOT, "data", "processed", "combined_features.csv")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR  = os.path.join(PROJECT_ROOT, "reports", "figures")
REPORTS_DIR  = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(FIGURES_DIR, exist_ok=True)

FORECAST_HORIZON = 6
SEQ_LEN          = 12

FEATURE_COLS_RAW = [
    "CPU usage [%]", "CPU usage [MHZ]", "Memory usage [%]",
    "Disk read throughput [KB/s]", "Disk write throughput [KB/s]",
    "Network received throughput [KB/s]", "Network transmitted throughput [KB/s]",
    "cpu_util_ratio", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend", "mem_pressure", "net_total_throughput", "disk_total_throughput",
]

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

def sanitize_col(name):
    return name.replace("[", "(").replace("]", ")").replace("<", "_")

# Force CPU — Prophet (Stan) + MPS in the same process segfaults on macOS.
# Inference on CPU is fast enough for evaluation (~20k samples).
DEVICE = torch.device("cpu")

# Load XGBoost model early (before sklearn.preprocessing is used) to avoid segfault
_xgb_model_early = joblib.load(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "xgboost_model.pkl"
))

print("=" * 60)
print("PHASE 7 — Model Evaluation & Comparison")
print("=" * 60)

# ── 1. Load Data & Same Split ─────────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH, parse_dates=["datetime"])
df = df.sort_values(["vm_id", "datetime"]).reset_index(drop=True)

busy_end   = pd.Timestamp("2013-09-02")
test_start = pd.Timestamp("2013-08-26")
busy_df    = df[df["datetime"] < busy_end].copy()
train_df   = busy_df[busy_df["datetime"] <  test_start].reset_index(drop=True)
test_df    = busy_df[busy_df["datetime"] >= test_start].reset_index(drop=True)

print(f"\n[1] Test set: {len(test_df):,} rows  "
      f"({test_start.date()} → {busy_end.date()})")

# ── 2. XGBoost Predictions ────────────────────────────────────────────────────
xgb_model = _xgb_model_early   # already loaded before sklearn to avoid segfault

test_with_target = test_df.copy()
test_with_target["target"] = (
    test_with_target.groupby("vm_id")["CPU usage [%]"].shift(-FORECAST_HORIZON)
)
test_with_target = test_with_target.dropna(subset=["target"])

safe_cols = {c: sanitize_col(c) for c in XGB_FEATURE_COLS}
X_test_xgb = test_with_target[XGB_FEATURE_COLS].rename(columns=safe_cols)
y_test_xgb = test_with_target["target"].values

xgb_pred = np.clip(xgb_model.predict(X_test_xgb), 0, 100)
print(f"[2] XGBoost predictions: {len(xgb_pred):,} samples")

# ── 3. LSTM Predictions — load from file saved during training ────────────────
# Avoids MPS/CPU numerical divergence: predictions were saved by 07_train_lstm.py
lstm_preds_df = pd.read_csv(os.path.join(MODELS_DIR, "lstm_test_predictions.csv"))
lstm_pred   = lstm_preds_df["y_pred"].values
y_test_lstm = lstm_preds_df["y_true"].values
vm_ids = df["vm_id"].unique()
print(f"[3] LSTM predictions loaded from file: {len(lstm_pred):,} samples")

# ── 4. Prophet Predictions ────────────────────────────────────────────────────
prophet_models = joblib.load(os.path.join(MODELS_DIR, "prophet_models.pkl"))

prophet_pred_list, prophet_true_list = [], []
for vm_id in sorted(vm_ids):
    # Prophet model keys are np.int64 — normalise lookup
    key = next((k for k in prophet_models if int(k) == int(vm_id)), None)
    if key is None:
        continue
    vm_test   = test_df[test_df["vm_id"] == vm_id]
    future_df = pd.DataFrame({"ds": vm_test["datetime"].values})
    forecast  = prophet_models[key].predict(future_df)
    y_pred_vm = np.clip(forecast["yhat"].values, 0, 100)
    y_true_vm = vm_test["CPU usage [%]"].values
    n = min(len(y_true_vm), len(y_pred_vm))
    prophet_pred_list.extend(y_pred_vm[:n])
    prophet_true_list.extend(y_true_vm[:n])

prophet_pred = np.array(prophet_pred_list)
y_test_prophet = np.array(prophet_true_list)
print(f"[4] Prophet predictions: {len(prophet_pred):,} samples")

# ── 5. Compute All Metrics ────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, name):
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    r2    = r2_score(y_true, y_pred)
    smape = np.mean(2 * np.abs(y_true - y_pred) /
                    (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
    return {"Model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2),
            "R2": round(r2, 4), "sMAPE": round(smape, 2)}

results = [
    compute_metrics(y_test_xgb,    xgb_pred,    "XGBoost"),
    compute_metrics(y_test_lstm,   lstm_pred,   "LSTM"),
    compute_metrics(y_test_prophet, prophet_pred, "Prophet"),
]
results_df = pd.DataFrame(results).set_index("Model")

print("\n[5] Model Comparison:")
print(results_df.to_string())

# ── 6. Comparison Bar Chart ───────────────────────────────────────────────────
metrics_to_plot = ["MAE", "RMSE", "sMAPE"]
colors = ["steelblue", "darkorange", "seagreen"]
models_list = ["XGBoost", "LSTM", "Prophet"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, metric in zip(axes, metrics_to_plot):
    vals = [results_df.loc[m, metric] for m in models_list]
    bars = ax.bar(models_list, vals, color=colors, width=0.5, edgecolor="white")
    ax.set_title(metric)
    ax.set_ylabel("% CPU")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

fig.suptitle("Model Comparison — MAE / RMSE / sMAPE (lower is better)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "14_model_comparison_bars.png"))
plt.close()
print("[6] Saved: 14_model_comparison_bars.png")

# ── 7. R² Comparison ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
r2_vals = [results_df.loc[m, "R2"] for m in models_list]
bars = ax.bar(models_list, r2_vals, color=colors, width=0.5, edgecolor="white")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.axhline(1, color="gray",  linewidth=0.8, linestyle=":")
ax.set_title("R² Score (higher is better, max=1)")
ax.set_ylabel("R²")
for bar, val in zip(bars, r2_vals):
    ypos = max(bar.get_height(), 0) + 0.02
    ax.text(bar.get_x() + bar.get_width() / 2, ypos,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "15_model_comparison_r2.png"))
plt.close()
print("[7] Saved: 15_model_comparison_r2.png")

# ── 8. Residual Distributions ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
for ax, (name, y_true, y_pred_vals) in zip(axes, [
    ("XGBoost", y_test_xgb,    xgb_pred),
    ("LSTM",    y_test_lstm,   lstm_pred),
    ("Prophet", y_test_prophet, prophet_pred),
]):
    resid = y_true - y_pred_vals
    ax.hist(resid, bins=60, color="steelblue" if name == "XGBoost"
            else "darkorange" if name == "LSTM" else "seagreen",
            edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="red", linewidth=1)
    ax.set_title(f"{name} Residuals\nMean={resid.mean():.2f}  Std={resid.std():.2f}")
    ax.set_xlabel("Residual (Actual − Predicted)")
    ax.set_ylabel("Frequency")

fig.suptitle("Residual Distributions", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "16_residual_distributions.png"))
plt.close()
print("[8] Saved: 16_residual_distributions.png")

# ── 9. Side-by-Side Actual vs Predicted (XGBoost & LSTM) ─────────────────────
n_plot = 288   # 24 hours worth (288 × 5 min)
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for ax, (name, y_true, y_pred_vals, color) in zip(axes, [
    ("XGBoost", y_test_xgb[:n_plot],  xgb_pred[:n_plot],  "steelblue"),
    ("LSTM",    y_test_lstm[:n_plot],  lstm_pred[:n_plot],  "darkorange"),
]):
    ax.plot(y_true,      label="Actual",    color="black",  linewidth=0.8, alpha=0.7)
    ax.plot(y_pred_vals, label=f"{name} Pred", color=color, linewidth=1.0, linestyle="--")
    ax.set_ylabel("CPU Usage [%]")
    ax.set_title(name)
    ax.legend(loc="upper right")
    ax.set_ylim(-5, 110)

axes[-1].set_xlabel("Sample (5-min steps, first 24 h of test)")
fig.suptitle("Actual vs Predicted — XGBoost & LSTM (first 24 hours)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "17_xgb_lstm_vs_actual.png"))
plt.close()
print("[9] Saved: 17_xgb_lstm_vs_actual.png")

# ── 10. Scatter: Predicted vs Actual ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, y_true, y_pred_vals, color) in zip(axes, [
    ("XGBoost", y_test_xgb,  xgb_pred,  "steelblue"),
    ("LSTM",    y_test_lstm, lstm_pred,  "darkorange"),
]):
    ax.scatter(y_true, y_pred_vals, alpha=0.05, s=3, color=color)
    lim = [0, 100]
    ax.plot(lim, lim, "r--", linewidth=1, label="Perfect")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual CPU [%]")
    ax.set_ylabel("Predicted CPU [%]")
    ax.set_title(f"{name}  (R²={results_df.loc[name,'R2']:.3f})")
    ax.legend()

fig.suptitle("Predicted vs Actual Scatter", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "18_scatter_pred_vs_actual.png"))
plt.close()
print("[10] Saved: 18_scatter_pred_vs_actual.png")

# ── 11. Save Summary Report ───────────────────────────────────────────────────
report_path = os.path.join(REPORTS_DIR, "model_comparison.json")
report = {
    "test_period": f"{test_start.date()} to {busy_end.date()}",
    "forecast_horizon_steps": FORECAST_HORIZON,
    "forecast_horizon_minutes": FORECAST_HORIZON * 5,
    "models": results,
    "best_model_mae":  min(results, key=lambda x: x["MAE"])["Model"],
    "best_model_rmse": min(results, key=lambda x: x["RMSE"])["Model"],
    "best_model_r2":   max(results, key=lambda x: x["R2"])["Model"],
}
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print("[11] Saved: reports/model_comparison.json")

# ── 12. Final Summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 7 COMPLETE — Final Model Comparison")
print("=" * 60)
print(f"\n  {'Model':<12} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'sMAPE':>8}")
print("  " + "-" * 46)
for _, row in results_df.iterrows():
    print(f"  {row.name:<12} {row['MAE']:>7.2f}% {row['RMSE']:>7.2f}% "
          f"{row['R2']:>8.4f} {row['sMAPE']:>7.1f}%")
print(f"\n  Best MAE  → {report['best_model_mae']}")
print(f"  Best RMSE → {report['best_model_rmse']}")
print(f"  Best R²   → {report['best_model_r2']}")
print(f"\n  Figures saved to: reports/figures/")
print("=" * 60)
