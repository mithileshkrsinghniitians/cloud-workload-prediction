import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH   = os.path.join(PROJECT_ROOT, "data", "processed", "combined_features.csv")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR  = os.path.join(PROJECT_ROOT, "reports", "figures")
METRICS_PATH = os.path.join(PROJECT_ROOT, "reports", "xgboost_metrics.json")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

FORECAST_HORIZON = 6   # steps ahead (6 × 5 min = 30 min)
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

def sanitize_col(name):
    """Remove characters XGBoost 3.x disallows in feature names: [ ] <"""
    return name.replace("[", "(").replace("]", ")").replace("<", "_")

print("=" * 60)
print("PHASE 6a — XGBoost Model Training")
print("=" * 60)

# ── 1. Load ────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH, parse_dates=["datetime"])
df = df.sort_values(["vm_id", "datetime"]).reset_index(drop=True)
print(f"\n[1] Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── 2. Create Target — CPU 30 min ahead ───────────────────────────────────────
# Shift within each VM so we don't bleed across VM boundaries
df["target"] = (
    df.groupby("vm_id")["CPU usage [%]"]
    .shift(-FORECAST_HORIZON)
)
df = df.dropna(subset=["target"]).reset_index(drop=True)
print(f"[2] Target = CPU usage [%] +{FORECAST_HORIZON} steps (30 min ahead).")
print(f"    Rows after target creation: {len(df):,}")

# ── 3. Train / Test Split — within busy period only ───────────────────────────
# The dataset has a regime change: busy Aug 12–Sep 1, idle from Sep 2 onward.
# To get a fair same-regime evaluation, train on first 3 weeks and test on
# the final week of the busy period (Aug 26 – Sep 1).
busy_end    = pd.Timestamp("2013-09-02")
test_start  = pd.Timestamp("2013-08-26")

busy_df = df[df["datetime"] < busy_end].reset_index(drop=True)
train   = busy_df[busy_df["datetime"] <  test_start].reset_index(drop=True)
test    = busy_df[busy_df["datetime"] >= test_start].reset_index(drop=True)

# Sanitize column names for XGBoost 3.x (no [ ] < allowed)
SAFE_COLS = {c: sanitize_col(c) for c in FEATURE_COLS}
train = train.rename(columns=SAFE_COLS)
test  = test.rename(columns=SAFE_COLS)
SAFE_FEATURE_COLS = [SAFE_COLS[c] for c in FEATURE_COLS]

X_train, y_train = train[SAFE_FEATURE_COLS], train["target"]
X_test,  y_test  = test[SAFE_FEATURE_COLS],  test["target"]

print(f"\n[3] Train/Test split (80/20 time-based per VM):")
print(f"    Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")

# ── 4. Train XGBoost ──────────────────────────────────────────────────────────
params = dict(
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

model = xgb.XGBRegressor(**params)
print(f"\n[4] Training XGBoost (max n_estimators={params['n_estimators']}, "
      f"lr={params['learning_rate']}, max_depth={params['max_depth']}, "
      f"early_stopping={params['early_stopping_rounds']})...")

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100,
)
print(f"    Best iteration: {model.best_iteration}")

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_pred = np.clip(y_pred, 0, 100)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
# sMAPE: symmetric, handles near-zero values gracefully
smape = np.mean(2 * np.abs(y_test - y_pred) /
                (np.abs(y_test) + np.abs(y_pred) + 1e-8)) * 100

print(f"\n[5] Test Set Evaluation:")
print(f"    MAE   : {mae:.4f} %")
print(f"    RMSE  : {rmse:.4f} %")
print(f"    R²    : {r2:.4f}")
print(f"    sMAPE : {smape:.2f}%")

# ── 6. Feature Importance ─────────────────────────────────────────────────────
importance = pd.Series(model.feature_importances_, index=SAFE_FEATURE_COLS)
top20 = importance.nlargest(20)

fig, ax = plt.subplots(figsize=(10, 7))
top20.sort_values().plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("XGBoost — Top 20 Feature Importances")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "06_xgb_feature_importance.png"))
plt.close()
print("[6] Saved: 06_xgb_feature_importance.png")

# ── 7. Prediction vs Actual Plot (one VM) ─────────────────────────────────────
sample_vm = test["vm_id"].iloc[0]
vm_test   = test[test["vm_id"] == sample_vm].copy()
vm_pred   = model.predict(vm_test[SAFE_FEATURE_COLS])
vm_pred   = np.clip(vm_pred, 0, 100)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(vm_test["datetime"].values, vm_test["target"].values,
        label="Actual", linewidth=1, color="steelblue")
ax.plot(vm_test["datetime"].values, vm_pred,
        label="XGBoost Predicted", linewidth=1, color="tomato", linestyle="--")
ax.set_title(f"XGBoost — Actual vs Predicted CPU [%] (VM {sample_vm})")
ax.set_xlabel("Time")
ax.set_ylabel("CPU Usage [%]")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "07_xgb_actual_vs_predicted.png"))
plt.close()
print("[7] Saved: 07_xgb_actual_vs_predicted.png")

# ── 8. Residual Plot ──────────────────────────────────────────────────────────
residuals = y_test.values - y_pred

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(y_pred, residuals, alpha=0.2, s=5, color="steelblue")
axes[0].axhline(0, color="red", linewidth=1)
axes[0].set_title("Residuals vs Predicted")
axes[0].set_xlabel("Predicted CPU [%]")
axes[0].set_ylabel("Residual")

axes[1].hist(residuals, bins=80, color="steelblue", edgecolor="white", linewidth=0.3)
axes[1].set_title("Residual Distribution")
axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "08_xgb_residuals.png"))
plt.close()
print("[8] Saved: 08_xgb_residuals.png")

# ── 9. Save Model & Metrics ───────────────────────────────────────────────────
model_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
joblib.dump(model, model_path)
print(f"[9] Model saved → models/xgboost_model.pkl")

metrics = {"MAE": round(mae, 4), "RMSE": round(rmse, 4),
           "R2": round(r2, 4), "sMAPE": round(smape, 2)}
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"[10] Metrics saved → reports/xgboost_metrics.json")

print("\n" + "=" * 60)
print("XGBoost Training Complete")
print(f"  MAE={mae:.2f}%  RMSE={rmse:.2f}%  R²={r2:.4f}  sMAPE={smape:.1f}%")
print("=" * 60)