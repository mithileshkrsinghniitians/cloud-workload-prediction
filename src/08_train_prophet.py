import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib.pyplot as plt
import warnings
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")   # suppress Prophet/Stan output

# Paths:
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH   = os.path.join(PROJECT_ROOT, "data", "processed", "combined_features.csv")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR  = os.path.join(PROJECT_ROOT, "reports", "figures")
METRICS_PATH = os.path.join(PROJECT_ROOT, "reports", "prophet_metrics.json")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

FORECAST_HORIZON = 6   # 6 steps × 5 min = 30 min ahead

print("=" * 60)
print("PHASE 6c — Prophet Baseline Training")
print("=" * 60)

# 1. Load & Same Split as XGBoost / LSTM:
df = pd.read_csv(INPUT_PATH, parse_dates=["datetime"])
df = df.sort_values(["vm_id", "datetime"]).reset_index(drop=True)

busy_end   = pd.Timestamp("2013-09-02")
test_start = pd.Timestamp("2013-08-26")
busy_df    = df[df["datetime"] < busy_end].copy()
train_df   = busy_df[busy_df["datetime"] <  test_start]
test_df    = busy_df[busy_df["datetime"] >= test_start]

print(f"\n[1] Train: {len(train_df):,} rows  |  Test: {len(test_df):,} rows")

# 2. Prophet Strategy:
# Prophet is a univariate model — fit one model per VM, forecast 6 steps,
# align with the test set target (CPU 30 min ahead).
# We collect all predictions across VMs then compute aggregate metrics.

vm_ids       = df["vm_id"].unique()
all_true     = []
all_pred     = []
vm_mae_list  = []

print(f"\n[2] Fitting Prophet per VM ({len(vm_ids)} VMs)...")
print(f"    {'VM':>6}  {'Train rows':>12}  {'Test rows':>10}  {'MAE':>8}")
print("    " + "-" * 44)

models = {}

for vm_id in sorted(vm_ids):
    vm_train = train_df[train_df["vm_id"] == vm_id][["datetime", "CPU usage [%]"]].copy()
    vm_test  = test_df[test_df["vm_id"] == vm_id][["datetime", "CPU usage [%]"]].copy()

    if len(vm_train) < 30 or len(vm_test) < FORECAST_HORIZON:
        print(f"    {vm_id:>6}  skipped (insufficient data)")
        continue

    # Prophet requires columns named 'ds' and 'y':
    vm_train_p = vm_train.rename(columns={"datetime": "ds", "CPU usage [%]": "y"})

    model = Prophet(
        changepoint_prior_scale  = 0.05,
        seasonality_prior_scale  = 10.0,
        daily_seasonality        = True,
        weekly_seasonality       = True,
        yearly_seasonality       = False,
        interval_width           = 0.95,
    )
    model.fit(vm_train_p)

    # Build future dataframe: need to predict at the exact test timestamps
    # shifted 6 steps (30 min) back — i.e., predict what CPU will be
    # 30 min after each test timestamp.
    future_times = vm_test["datetime"].values
    future_df    = pd.DataFrame({"ds": future_times})
    forecast     = model.predict(future_df)
    y_pred_vm    = np.clip(forecast["yhat"].values, 0, 100)

    # True target: CPU usage at each test timestamp (same as other models):
    y_true_vm = vm_test["CPU usage [%]"].values

    # Align lengths (Prophet predicts at same timestamps as test):
    n = min(len(y_true_vm), len(y_pred_vm))
    y_true_vm = y_true_vm[:n]
    y_pred_vm = y_pred_vm[:n]

    vm_mae = mean_absolute_error(y_true_vm, y_pred_vm)
    vm_mae_list.append({"vm_id": vm_id, "mae": round(vm_mae, 2), "n": n})
    all_true.extend(y_true_vm)
    all_pred.extend(y_pred_vm)
    models[vm_id] = model

    print(f"    {vm_id:>6}  {len(vm_train):>12,}  {len(vm_test):>10,}  {vm_mae:>8.2f}%")

# 3. Aggregate Metrics:
y_true = np.array(all_true)
y_pred = np.array(all_pred)

mae   = mean_absolute_error(y_true, y_pred)
rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
r2    = r2_score(y_true, y_pred)
smape = np.mean(2 * np.abs(y_true - y_pred) /
                (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100

print(f"\n[3] Aggregate Test Set Evaluation (all VMs):")
print(f"    MAE   : {mae:.4f} %")
print(f"    RMSE  : {rmse:.4f} %")
print(f"    R²    : {r2:.4f}")
print(f"    sMAPE : {smape:.2f}%")

# 4. Per-VM MAE Table:
print(f"\n[4] Per-VM MAE:")
vm_mae_df = pd.DataFrame(vm_mae_list).sort_values("mae")
print(vm_mae_df.to_string(index=False))

# 5. Forecast Components Plot (one VM):
sample_vm = sorted(vm_ids)[0]
sample_model = models[sample_vm]
vm_train_p = train_df[train_df["vm_id"] == sample_vm][["datetime", "CPU usage [%]"]].rename(
    columns={"datetime": "ds", "CPU usage [%]": "y"}
)
future = sample_model.make_future_dataframe(periods=48, freq="5min")
forecast_full = sample_model.predict(future)

fig = sample_model.plot(forecast_full, figsize=(14, 5))
fig.axes[0].set_title(f"Prophet Forecast — VM {sample_vm}")
fig.axes[0].set_ylabel("CPU Usage [%]")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "11_prophet_forecast.png"))
plt.close()
print("[5] Saved: 11_prophet_forecast.png")

fig = sample_model.plot_components(forecast_full, figsize=(12, 8))
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "12_prophet_components.png"))
plt.close()
print("[6] Saved: 12_prophet_components.png")

# 6. Actual vs Predicted:
plot_n = min(500, len(y_true))
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(range(plot_n), y_true[:plot_n], label="Actual",           color="steelblue",  linewidth=1)
ax.plot(range(plot_n), y_pred[:plot_n], label="Prophet Predicted", color="seagreen",   linewidth=1, linestyle="--")
ax.set_title("Prophet — Actual vs Predicted CPU [%] (first 500 test samples)")
ax.set_xlabel("Sample")
ax.set_ylabel("CPU Usage [%]")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "13_prophet_actual_vs_predicted.png"))
plt.close()
print("[7] Saved: 13_prophet_actual_vs_predicted.png")

# 7. Save Models & Metrics:
joblib.dump(models, os.path.join(MODELS_DIR, "prophet_models.pkl"))
print("[8] Models saved → models/prophet_models.pkl")

metrics = {
    "MAE":   round(float(mae),   4),
    "RMSE":  round(float(rmse),  4),
    "R2":    round(float(r2),    4),
    "sMAPE": round(float(smape), 2),
}
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)
print("[9] Metrics saved → reports/prophet_metrics.json")

print("\n" + "=" * 60)
print("Prophet Training Complete")
print(f"  MAE={mae:.2f}%  RMSE={rmse:.2f}%  R²={r2:.4f}  sMAPE={smape:.1f}%")
print("=" * 60)
