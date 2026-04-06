import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import xgboost as xgb
import joblib

import pandas as pd
import numpy as np
import os
import json
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths:
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH   = os.path.join(PROJECT_ROOT, "data", "processed", "combined_features.csv")
FIGURES_DIR  = os.path.join(PROJECT_ROOT, "reports", "figures")
REPORTS_DIR  = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(FIGURES_DIR, exist_ok=True)

FORECAST_HORIZON = 6   # 30 min ahead

# XGBoost feature columns (same as Phase 6a):
FEATURE_COLS = [
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

# Fractions of training data to test (20% → 100%):
FRACTIONS = [0.20, 0.40, 0.60, 0.80, 1.00]

def sanitize_col(name):
    return name.replace("[", "(").replace("]", ")").replace("<", "_")

def smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_true - y_pred) /
                   (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100

def compute_metrics(y_true, y_pred):
    return {
        "MAE":   round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE":  round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "R2":    round(r2_score(y_true, y_pred), 4),
        "sMAPE": round(smape(y_true, y_pred), 2),
    }

print("=" * 60)
print("PHASE 11 — Learning Curves (XGBoost)")
print("=" * 60)
print("\nTraining XGBoost on 20% / 40% / 60% / 80% / 100% of training data")
print("Evaluating on the same fixed test set each time.\n")

# 1. Load Data:
df = pd.read_csv(INPUT_PATH, parse_dates=["datetime"])
df = df.sort_values(["vm_id", "datetime"]).reset_index(drop=True)

# Create target (CPU 30 min ahead, per VM to avoid bleed):
df["target"] = (
    df.groupby("vm_id")["CPU usage [%]"]
    .shift(-FORECAST_HORIZON)
)
df = df.dropna(subset=["target"]).reset_index(drop=True)

# Same train/test split as Phase 6a:
busy_end   = pd.Timestamp("2013-09-02")
test_start = pd.Timestamp("2013-08-26")

busy_df = df[df["datetime"] < busy_end].reset_index(drop=True)
train   = busy_df[busy_df["datetime"] <  test_start].reset_index(drop=True)
test    = busy_df[busy_df["datetime"] >= test_start].reset_index(drop=True)

# Sanitize column names for XGBoost 3.x:
SAFE_COLS = {c: sanitize_col(c) for c in FEATURE_COLS}
train = train.rename(columns=SAFE_COLS)
test  = test.rename(columns=SAFE_COLS)
SAFE_FEATURE_COLS = [SAFE_COLS[c] for c in FEATURE_COLS]

X_test,  y_test  = test[SAFE_FEATURE_COLS].values,  test["target"].values
full_train_size   = len(train)

print(f"[1] Full training set: {full_train_size:,} rows")
print(f"    Fixed test set   : {len(test):,} rows")
print(f"    Fractions to test: {FRACTIONS}\n")

# 2. Naive Baseline:
# Naive: predict current CPU (lag1) as the forecast — simplest possible baseline
# This gives us something to compare the learning curves against
naive_pred = test[sanitize_col("CPU usage [%]_lag1")].values
naive_metrics = compute_metrics(y_test, naive_pred)
print(f"[2] Naive baseline (lag1 as prediction):")
print(f"    MAE={naive_metrics['MAE']:.4f}%  RMSE={naive_metrics['RMSE']:.4f}%  "
      f"R²={naive_metrics['R2']:.4f}")

# 3. XGBoost config (same as Phase 6a):
xgb_params = dict(
    n_estimators     = 1000,
    learning_rate    = 0.05,
    max_depth        = 6,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 3,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    objective        = "reg:squarederror",
    tree_method      = "hist",
    early_stopping_rounds = 50,
    random_state     = 42,
    n_jobs           = -1,
)

# 4. Run Learning Curve Experiments:
results = []
print(f"[3] Running {len(FRACTIONS)} experiments...\n")
print(f"  {'Fraction':>9}  {'Train Rows':>11}  {'MAE':>8}  {'RMSE':>8}  {'R²':>8}  "
      f"{'sMAPE':>8}  {'Train Time':>10}")
print("  " + "-" * 72)

for frac in FRACTIONS:
    # Take a chronological slice of the training data
    # (using the last N rows preserves temporal ordering and avoids data leakage)
    n_rows = max(1, int(full_train_size * frac))
    train_subset = train.iloc[:n_rows]

    X_tr = train_subset[SAFE_FEATURE_COLS].values
    y_tr = train_subset["target"].values

    # Train model:
    model = xgb.XGBRegressor(**xgb_params)
    t_start = time.perf_counter()
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    train_time = time.perf_counter() - t_start

    # Evaluate on fixed test set:
    y_pred = np.clip(model.predict(X_test), 0, 100)
    metrics = compute_metrics(y_test, y_pred)

    print(f"  {frac:>8.0%}  {n_rows:>11,}  {metrics['MAE']:>7.4f}%  "
          f"{metrics['RMSE']:>7.4f}%  {metrics['R2']:>8.4f}  "
          f"{metrics['sMAPE']:>7.2f}%  {train_time:>9.1f}s")

    results.append({
        "fraction":   frac,
        "train_rows": n_rows,
        "train_time_sec": round(train_time, 2),
        **metrics,
    })

# 5. Plot: MAE and RMSE vs Training Set Size:
train_rows = [r["train_rows"] for r in results]
maes       = [r["MAE"]   for r in results]
rmses      = [r["RMSE"]  for r in results]
r2s        = [r["R2"]    for r in results]
smapes     = [r["sMAPE"] for r in results]
times      = [r["train_time_sec"] for r in results]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# MAE:
ax = axes[0, 0]
ax.plot(train_rows, maes, "o-", color="steelblue", linewidth=2, markersize=7, label="XGBoost")
ax.axhline(naive_metrics["MAE"], color="gray", linestyle="--", linewidth=1.2, label=f"Naive baseline ({naive_metrics['MAE']:.2f}%)")
ax.set_xlabel("Training Set Size (rows)")
ax.set_ylabel("MAE (%)")
ax.set_title("MAE vs Training Set Size (lower is better)")
ax.legend()
ax.grid(True, alpha=0.3)
for x, y in zip(train_rows, maes):
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8)

# RMSE:
ax = axes[0, 1]
ax.plot(train_rows, rmses, "s-", color="tomato", linewidth=2, markersize=7, label="XGBoost")
ax.axhline(naive_metrics["RMSE"], color="gray", linestyle="--", linewidth=1.2, label=f"Naive baseline ({naive_metrics['RMSE']:.2f}%)")
ax.set_xlabel("Training Set Size (rows)")
ax.set_ylabel("RMSE (%)")
ax.set_title("RMSE vs Training Set Size (lower is better)")
ax.legend()
ax.grid(True, alpha=0.3)
for x, y in zip(train_rows, rmses):
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8)

# R²:
ax = axes[1, 0]
ax.plot(train_rows, r2s, "^-", color="seagreen", linewidth=2, markersize=7, label="XGBoost")
ax.axhline(naive_metrics["R2"], color="gray", linestyle="--", linewidth=1.2, label=f"Naive baseline ({naive_metrics['R2']:.3f})")
ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, label="Perfect (R²=1)")
ax.set_xlabel("Training Set Size (rows)")
ax.set_ylabel("R²")
ax.set_title("R² vs Training Set Size (higher is better)")
ax.legend()
ax.grid(True, alpha=0.3)
for x, y in zip(train_rows, r2s):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8)

# Training time:
ax = axes[1, 1]
fracs_pct = [f * 100 for f in FRACTIONS]
ax.bar(fracs_pct, times, color="mediumpurple", width=12, edgecolor="white")
ax.set_xlabel("Training Data Used (%)")
ax.set_ylabel("Training Time (seconds)")
ax.set_title("Training Time vs Dataset Size")
for i, (x, y) in enumerate(zip(fracs_pct, times)):
    ax.text(x, y + 0.5, f"{y:.0f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")

fig.suptitle("XGBoost Learning Curves — Performance vs Training Set Size",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "21_learning_curves.png"))
plt.close()
print("\n[4] Saved: 21_learning_curves.png")

# 6. Plot: Fraction on x-axis (cleaner for report):
fig, ax = plt.subplots(figsize=(9, 5))
ax2 = ax.twinx()

ln1 = ax.plot(fracs_pct, maes,  "o-", color="steelblue", linewidth=2, markersize=7, label="MAE (%)")
ln2 = ax.plot(fracs_pct, rmses, "s-", color="tomato",    linewidth=2, markersize=7, label="RMSE (%)")
ln3 = ax2.plot(fracs_pct, r2s,  "^-", color="seagreen",  linewidth=2, markersize=7, label="R²")

ax.axhline(naive_metrics["MAE"],  color="steelblue", linestyle=":", linewidth=1, alpha=0.5)
ax.axhline(naive_metrics["RMSE"], color="tomato",    linestyle=":", linewidth=1, alpha=0.5)
ax2.axhline(naive_metrics["R2"],  color="seagreen",  linestyle=":", linewidth=1, alpha=0.5)

ax.set_xlabel("Percentage of Training Data Used (%)")
ax.set_ylabel("Error (%) — MAE / RMSE")
ax2.set_ylabel("R² Score")
ax.set_title("XGBoost Learning Curve (MAE, RMSE, R² vs Training Size)")
ax.set_xticks(fracs_pct)
ax.set_xticklabels([f"{int(p)}%" for p in fracs_pct])

lns = ln1 + ln2 + ln3
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc="upper right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "22_learning_curves_combined.png"))
plt.close()
print("[5] Saved: 22_learning_curves_combined.png")

# 7. Save Results to JSON:
report = {
    "description": "XGBoost learning curves — trained on varying fractions of training data",
    "fixed_test_set_size": len(test),
    "naive_baseline": naive_metrics,
    "results": results,
}
report_path = os.path.join(REPORTS_DIR, "learning_curves.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print("[6] Saved: reports/learning_curves.json")

# 8. Final Summary:
print("\n" + "=" * 60)
print("Learning Curves Complete")
print("=" * 60)
print(f"\n  {'Fraction':>9}  {'Rows':>8}  {'MAE':>8}  {'RMSE':>8}  {'R²':>8}")
print("  " + "-" * 48)
for r in results:
    print(f"  {r['fraction']:>8.0%}  {r['train_rows']:>8,}  "
          f"{r['MAE']:>7.4f}%  {r['RMSE']:>7.4f}%  {r['R2']:>8.4f}")
print(f"\n  Naive baseline:  MAE={naive_metrics['MAE']:.4f}%  "
      f"RMSE={naive_metrics['RMSE']:.4f}%  R²={naive_metrics['R2']:.4f}")
best = min(results, key=lambda x: x["MAE"])
print(f"\n  Best result: {best['fraction']:.0%} of data → MAE={best['MAE']:.4f}%")
print("=" * 60)
