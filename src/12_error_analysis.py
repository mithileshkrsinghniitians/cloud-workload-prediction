import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import xgboost as xgb   # import first on macOS to avoid segfault
import joblib

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH   = os.path.join(PROJECT_ROOT, "data", "processed", "combined_features.csv")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR  = os.path.join(PROJECT_ROOT, "reports", "figures")
REPORTS_DIR  = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(FIGURES_DIR, exist_ok=True)

FORECAST_HORIZON = 6   # 30 min ahead
TOP_N_ERRORS     = 20  # how many worst predictions to inspect

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

def smape_sample(y_true, y_pred):
    """Per-sample sMAPE values."""
    return 2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8) * 100

print("=" * 60)
print("PHASE 12 — Comprehensive Error Analysis (XGBoost)")
print("=" * 60)

# ── 1. Load Data & Reproduce Test Set ─────────────────────────────────────────
df = pd.read_csv(INPUT_PATH, parse_dates=["datetime"])
df = df.sort_values(["vm_id", "datetime"]).reset_index(drop=True)

# Create target (same as training)
df["target"] = (
    df.groupby("vm_id")["CPU usage [%]"]
    .shift(-FORECAST_HORIZON)
)
df = df.dropna(subset=["target"]).reset_index(drop=True)

busy_end   = pd.Timestamp("2013-09-02")
test_start = pd.Timestamp("2013-08-26")
busy_df    = df[df["datetime"] < busy_end].reset_index(drop=True)
test_df    = busy_df[busy_df["datetime"] >= test_start].reset_index(drop=True)

print(f"\n[1] Test set: {len(test_df):,} rows  "
      f"({test_start.date()} → {busy_end.date()})")

# ── 2. Generate XGBoost Predictions ───────────────────────────────────────────
xgb_model  = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))
safe_map    = {c: sanitize_col(c) for c in XGB_FEATURE_COLS}
X_test      = test_df[XGB_FEATURE_COLS].rename(columns=safe_map)
y_pred      = np.clip(xgb_model.predict(X_test), 0, 100)
y_true      = test_df["target"].values

print(f"[2] Predictions generated: {len(y_pred):,} samples")

# ── 3. Build Error DataFrame ───────────────────────────────────────────────────
err_df = test_df[["vm_id", "datetime", "target"]].copy()
err_df = err_df.rename(columns={"target": "y_true"})
err_df["y_pred"]    = y_pred
err_df["residual"]  = err_df["y_true"] - err_df["y_pred"]  # actual - predicted
err_df["abs_error"] = np.abs(err_df["residual"])
err_df["smape"]     = smape_sample(err_df["y_true"].values, err_df["y_pred"].values)
err_df["hour"]      = err_df["datetime"].dt.hour
err_df["day_of_week"] = err_df["datetime"].dt.dayofweek   # 0=Mon … 6=Sun
err_df["day_name"]  = err_df["datetime"].dt.day_name()

# CPU regime labelling
# Idle: < 20%, Moderate: 20–70%, High: > 70%
err_df["cpu_regime"] = pd.cut(
    err_df["y_true"],
    bins=[-1, 20, 70, 101],
    labels=["Idle (<20%)", "Moderate (20–70%)", "High (>70%)"]
)

print("[3] Error dataframe built with residuals, sMAPE, CPU regime, hour, day.")

# ── 4. Overall Metrics ────────────────────────────────────────────────────────
overall_mae   = mean_absolute_error(y_true, y_pred)
overall_rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
overall_r2    = r2_score(y_true, y_pred)
overall_smape = smape_sample(y_true, y_pred).mean()
bias          = err_df["residual"].mean()   # positive = over-prediction, negative = under-prediction

print(f"\n[4] Overall Test Metrics:")
print(f"    MAE   = {overall_mae:.4f}%")
print(f"    RMSE  = {overall_rmse:.4f}%")
print(f"    R²    = {overall_r2:.4f}")
print(f"    sMAPE = {overall_smape:.2f}%")
print(f"    Bias  = {bias:.4f}% ({'over-predicts' if bias < 0 else 'under-predicts'})")

# ── 5. Per-VM Error Breakdown ─────────────────────────────────────────────────
vm_errors = (
    err_df.groupby("vm_id")
    .apply(lambda g: pd.Series({
        "n_samples": len(g),
        "MAE":   mean_absolute_error(g["y_true"], g["y_pred"]),
        "RMSE":  np.sqrt(mean_squared_error(g["y_true"], g["y_pred"])),
        "R2":    r2_score(g["y_true"], g["y_pred"]),
        "sMAPE": smape_sample(g["y_true"].values, g["y_pred"].values).mean(),
        "bias":  (g["y_true"] - g["y_pred"]).mean(),
        "mean_actual_cpu": g["y_true"].mean(),
    }))
    .reset_index()
    .sort_values("MAE")
)
vm_errors = vm_errors.round(3)

print(f"\n[5] Per-VM MAE breakdown:")
print(f"    {'VM':>5}  {'MAE':>6}  {'RMSE':>7}  {'R²':>7}  {'sMAPE':>7}  {'Bias':>6}  {'Mean CPU':>8}")
print("    " + "-" * 55)
for _, row in vm_errors.iterrows():
    print(f"    {int(row['vm_id']):>5}  {row['MAE']:>5.2f}%  {row['RMSE']:>6.2f}%  "
          f"{row['R2']:>7.4f}  {row['sMAPE']:>6.2f}%  {row['bias']:>6.2f}%  "
          f"{row['mean_actual_cpu']:>7.2f}%")

# ── 6. Error by CPU Regime ────────────────────────────────────────────────────
regime_errors = (
    err_df.groupby("cpu_regime", observed=True)
    .apply(lambda g: pd.Series({
        "n_samples": len(g),
        "MAE":   mean_absolute_error(g["y_true"], g["y_pred"]),
        "RMSE":  np.sqrt(mean_squared_error(g["y_true"], g["y_pred"])),
        "bias":  (g["y_true"] - g["y_pred"]).mean(),
        "pct_of_total": len(g) / len(err_df) * 100,
    }))
    .reset_index()
    .round(3)
)

print(f"\n[6] Error by CPU regime:")
for _, row in regime_errors.iterrows():
    print(f"    {row['cpu_regime']}:  MAE={row['MAE']:.2f}%  "
          f"RMSE={row['RMSE']:.2f}%  Bias={row['bias']:.2f}%  "
          f"({row['pct_of_total']:.1f}% of test samples)")

# ── 7. Error by Hour of Day ───────────────────────────────────────────────────
hourly_errors = (
    err_df.groupby("hour")["abs_error"]
    .mean()
    .reset_index()
    .rename(columns={"abs_error": "mean_abs_error"})
)

# ── 8. Error by Day of Week ───────────────────────────────────────────────────
day_names_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
daily_errors = (
    err_df.groupby("day_name")["abs_error"]
    .mean()
    .reindex(day_names_order)
    .reset_index()
    .rename(columns={"abs_error": "mean_abs_error"})
)

# ── 9. Worst Predictions (largest absolute errors) ────────────────────────────
worst = err_df.nlargest(TOP_N_ERRORS, "abs_error")[
    ["vm_id", "datetime", "y_true", "y_pred", "abs_error", "residual", "cpu_regime"]
].reset_index(drop=True)

print(f"\n[7] Top {TOP_N_ERRORS} worst predictions (largest |error|):")
print(f"    {'VM':>5}  {'Datetime':<20}  {'Actual':>7}  {'Pred':>7}  {'|Error|':>8}  {'Regime'}")
print("    " + "-" * 72)
for _, row in worst.head(10).iterrows():
    print(f"    {int(row['vm_id']):>5}  {str(row['datetime']):<20}  "
          f"{row['y_true']:>6.1f}%  {row['y_pred']:>6.1f}%  "
          f"{row['abs_error']:>7.1f}%  {row['cpu_regime']}")

# ── 10. Regime transition analysis ────────────────────────────────────────────
# Find rows where the CPU regime changes (idle→high or high→idle)
# These "transition points" are where models tend to struggle most
err_df_sorted = err_df.sort_values(["vm_id", "datetime"]).copy()
err_df_sorted["regime_shift"] = (
    err_df_sorted.groupby("vm_id")["cpu_regime"]
    .transform(lambda x: x != x.shift(1))
)
transition_errors = err_df_sorted[err_df_sorted["regime_shift"]]["abs_error"]
stable_errors     = err_df_sorted[~err_df_sorted["regime_shift"]]["abs_error"]

print(f"\n[8] Error at regime transitions vs stable periods:")
print(f"    Transition points: {len(transition_errors):,}  →  Mean |Error| = {transition_errors.mean():.4f}%")
print(f"    Stable periods   : {len(stable_errors):,}  →  Mean |Error| = {stable_errors.mean():.4f}%")
print(f"    Transitions are {transition_errors.mean()/stable_errors.mean():.2f}× harder to predict.")

# ── 11. Figure 1: Per-VM MAE Bar Chart ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(vm_errors)))
bars = ax.bar(vm_errors["vm_id"].astype(str), vm_errors["MAE"], color=colors, edgecolor="white")
ax.axhline(overall_mae, color="steelblue", linestyle="--", linewidth=1.5,
           label=f"Overall MAE = {overall_mae:.2f}%")
ax.set_title("Per-VM Mean Absolute Error (XGBoost — Test Set)")
ax.set_xlabel("VM ID")
ax.set_ylabel("MAE (%)")
ax.legend()
for bar, val in zip(bars, vm_errors["MAE"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f"{val:.1f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "23_per_vm_mae.png"))
plt.close()
print("\n[9] Saved: 23_per_vm_mae.png")

# ── 12. Figure 2: Error by CPU Regime ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
regime_labels = regime_errors["cpu_regime"].astype(str).tolist()
regime_colors = ["#5cb85c", "#f0ad4e", "#d9534f"]   # green, orange, red

axes[0].bar(regime_labels, regime_errors["MAE"], color=regime_colors, edgecolor="white")
axes[0].set_title("MAE by CPU Regime")
axes[0].set_ylabel("MAE (%)")
axes[0].set_xlabel("CPU Usage Regime")
for i, (label, val) in enumerate(zip(regime_labels, regime_errors["MAE"])):
    axes[0].text(i, val + 0.1, f"{val:.2f}%", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")

axes[1].bar(regime_labels, regime_errors["bias"], color=regime_colors, edgecolor="white")
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_title("Prediction Bias by CPU Regime\n(positive = under-predict, negative = over-predict)")
axes[1].set_ylabel("Mean Residual (%)")
axes[1].set_xlabel("CPU Usage Regime")
for i, (label, val) in enumerate(zip(regime_labels, regime_errors["bias"])):
    axes[1].text(i, val + (0.1 if val >= 0 else -0.4), f"{val:.2f}%",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

fig.suptitle("Error Analysis by CPU Regime", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "24_error_by_regime.png"))
plt.close()
print("[10] Saved: 24_error_by_regime.png")

# ── 13. Figure 3: Error by Hour and Day ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# By hour
axes[0].bar(hourly_errors["hour"], hourly_errors["mean_abs_error"],
            color="steelblue", edgecolor="white")
axes[0].set_title("Mean Absolute Error by Hour of Day")
axes[0].set_xlabel("Hour (0 = midnight)")
axes[0].set_ylabel("Mean |Error| (%)")
axes[0].set_xticks(range(0, 24, 2))
axes[0].axhline(overall_mae, color="tomato", linestyle="--", linewidth=1.2,
                label=f"Overall MAE={overall_mae:.2f}%")
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis="y")

# By day of week
day_colors = ["#5cb85c" if d in ["Saturday","Sunday"] else "steelblue"
              for d in daily_errors["day_name"]]
axes[1].bar(range(len(daily_errors)), daily_errors["mean_abs_error"],
            color=day_colors, edgecolor="white")
axes[1].set_title("Mean Absolute Error by Day of Week\n(green = weekend)")
axes[1].set_xlabel("Day of Week")
axes[1].set_ylabel("Mean |Error| (%)")
axes[1].set_xticks(range(len(daily_errors)))
axes[1].set_xticklabels([d[:3] for d in daily_errors["day_name"]], rotation=30)
axes[1].axhline(overall_mae, color="tomato", linestyle="--", linewidth=1.2,
                label=f"Overall MAE={overall_mae:.2f}%")
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis="y")

fig.suptitle("Temporal Error Patterns — Hour of Day & Day of Week", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "25_error_temporal.png"))
plt.close()
print("[11] Saved: 25_error_temporal.png")

# ── 14. Figure 4: Error at Regime Transitions ─────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
data_to_plot = [stable_errors.values, transition_errors.values]
bp = ax.boxplot(data_to_plot, patch_artist=True, widths=0.5,
                medianprops=dict(color="black", linewidth=2))
bp["boxes"][0].set_facecolor("steelblue")
bp["boxes"][1].set_facecolor("tomato")
ax.set_xticklabels([
    f"Stable Periods\n(n={len(stable_errors):,})",
    f"Regime Transitions\n(n={len(transition_errors):,})"
])
ax.set_ylabel("|Error| (%)")
ax.set_title("Prediction Error: Stable CPU vs Regime Transition Points")
ax.set_ylim(0, np.percentile(np.concatenate([stable_errors, transition_errors]), 99))
ax.grid(True, alpha=0.3, axis="y")
ax.text(1, stable_errors.mean() + 1, f"mean={stable_errors.mean():.2f}%",
        ha="center", fontsize=10, color="steelblue", fontweight="bold")
ax.text(2, transition_errors.mean() + 1, f"mean={transition_errors.mean():.2f}%",
        ha="center", fontsize=10, color="tomato", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "26_error_transitions.png"))
plt.close()
print("[12] Saved: 26_error_transitions.png")

# ── 15. Figure 5: Worst-case heatmap (VM × Hour) ─────────────────────────────
pivot = err_df.pivot_table(
    values="abs_error", index="vm_id", columns="hour", aggfunc="mean"
)
fig, ax = plt.subplots(figsize=(16, 6))
sns.heatmap(pivot, cmap="YlOrRd", annot=False, ax=ax, cbar_kws={"label": "Mean |Error| (%)"})
ax.set_title("Mean Absolute Error Heatmap — VM × Hour of Day")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("VM ID")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "27_error_heatmap_vm_hour.png"))
plt.close()
print("[13] Saved: 27_error_heatmap_vm_hour.png")

# ── 16. Save Error Report to JSON ─────────────────────────────────────────────
error_report = {
    "overall_metrics": {
        "MAE": round(overall_mae, 4),
        "RMSE": round(overall_rmse, 4),
        "R2": round(overall_r2, 4),
        "sMAPE": round(overall_smape, 2),
        "bias": round(bias, 4),
        "bias_direction": "under-predicts" if bias > 0 else "over-predicts",
    },
    "per_vm_errors": vm_errors.to_dict(orient="records"),
    "per_regime_errors": regime_errors.to_dict(orient="records"),
    "hourly_errors": hourly_errors.to_dict(orient="records"),
    "daily_errors": daily_errors.dropna().to_dict(orient="records"),
    "regime_transition_analysis": {
        "stable_mean_abs_error": round(float(stable_errors.mean()), 4),
        "transition_mean_abs_error": round(float(transition_errors.mean()), 4),
        "transition_difficulty_ratio": round(float(transition_errors.mean() / stable_errors.mean()), 2),
        "n_stable_samples": int(len(stable_errors)),
        "n_transition_samples": int(len(transition_errors)),
    },
    "top_worst_predictions": worst[
        ["vm_id","datetime","y_true","y_pred","abs_error","residual"]
    ].astype({"vm_id": int, "datetime": str, "y_true": float,
               "y_pred": float, "abs_error": float, "residual": float})
    .round(2).to_dict(orient="records"),
}
report_path = os.path.join(REPORTS_DIR, "error_analysis.json")
with open(report_path, "w") as f:
    json.dump(error_report, f, indent=2)
print("[14] Saved: reports/error_analysis.json")

# ── 17. Final Summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Error Analysis Complete")
print("=" * 60)
print(f"\n  Overall: MAE={overall_mae:.2f}%  RMSE={overall_rmse:.2f}%  "
      f"R²={overall_r2:.4f}  Bias={bias:.2f}%")
print(f"\n  Hardest regime: {regime_errors.loc[regime_errors['MAE'].idxmax(), 'cpu_regime']} "
      f"(MAE = {regime_errors['MAE'].max():.2f}%)")
print(f"  Easiest regime: {regime_errors.loc[regime_errors['MAE'].idxmin(), 'cpu_regime']} "
      f"(MAE = {regime_errors['MAE'].min():.2f}%)")
print(f"\n  Worst VM: {int(vm_errors.iloc[-1]['vm_id'])} (MAE = {vm_errors.iloc[-1]['MAE']:.2f}%)")
print(f"  Best VM : {int(vm_errors.iloc[0]['vm_id'])} (MAE = {vm_errors.iloc[0]['MAE']:.2f}%)")
print(f"\n  Regime transitions are {transition_errors.mean()/stable_errors.mean():.2f}× "
      f"harder to predict than stable periods.")
print("=" * 60)
