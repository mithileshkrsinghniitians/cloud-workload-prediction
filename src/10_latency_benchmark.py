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
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# Paths:
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH   = os.path.join(PROJECT_ROOT, "data", "processed", "combined_features.csv")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR  = os.path.join(PROJECT_ROOT, "reports", "figures")
REPORTS_DIR  = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(FIGURES_DIR, exist_ok=True)

FORECAST_HORIZON = 6
SEQ_LEN          = 12
DEVICE           = torch.device("cpu")
N_REPEATS        = 5   # repeat each benchmark to get stable average

# XGBoost feature columns (same as training):
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

# LSTM feature columns (same as training):
LSTM_FEATURE_COLS = [
    "CPU usage [%]", "CPU usage [MHZ]", "Memory usage [%]",
    "Disk read throughput [KB/s]", "Disk write throughput [KB/s]",
    "Network received throughput [KB/s]", "Network transmitted throughput [KB/s]",
    "cpu_util_ratio", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend", "mem_pressure", "net_total_throughput", "disk_total_throughput",
]

def sanitize_col(name):
    return name.replace("[", "(").replace("]", ")").replace("<", "_")


# LSTM architecture (must match training):
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


print("=" * 60)
print("PHASE 10 — Latency & Throughput Benchmark")
print("=" * 60)

# 1. Load Data & Prepare Test Set:
df = pd.read_csv(INPUT_PATH, parse_dates=["datetime"])
df = df.sort_values(["vm_id", "datetime"]).reset_index(drop=True)

busy_end   = pd.Timestamp("2013-09-02")
test_start = pd.Timestamp("2013-08-26")
busy_df    = df[df["datetime"] < busy_end].copy()
train_df   = busy_df[busy_df["datetime"] <  test_start].reset_index(drop=True)
test_df    = busy_df[busy_df["datetime"] >= test_start].reset_index(drop=True)

print(f"\n[1] Test set loaded: {len(test_df):,} rows")

# 2. Prepare XGBoost Test Data:
test_with_target = test_df.copy()
test_with_target["target"] = (
    test_with_target.groupby("vm_id")["CPU usage [%]"]
    .shift(-FORECAST_HORIZON)
)
test_with_target = test_with_target.dropna(subset=["target"]).reset_index(drop=True)

safe_map    = {c: sanitize_col(c) for c in XGB_FEATURE_COLS}
X_test_xgb  = test_with_target[XGB_FEATURE_COLS].rename(columns=safe_map).values
print(f"[2] XGBoost test matrix: {X_test_xgb.shape}")

# 3. Prepare LSTM Test Data:
scaler = joblib.load(os.path.join(MODELS_DIR, "lstm_scaler.pkl"))

test_scaled = test_df[LSTM_FEATURE_COLS].copy()
test_scaled[LSTM_FEATURE_COLS] = scaler.transform(test_df[LSTM_FEATURE_COLS])

vm_ids = df["vm_id"].unique()
X_lstm_list = []
for vm_id in vm_ids:
    vm_mask  = test_df["vm_id"] == vm_id
    features = test_scaled[LSTM_FEATURE_COLS][vm_mask].values
    for i in range(SEQ_LEN, len(features) - FORECAST_HORIZON + 1):
        X_lstm_list.append(features[i - SEQ_LEN : i])

X_test_lstm = np.array(X_lstm_list, dtype=np.float32)
print(f"[3] LSTM test sequences: {X_test_lstm.shape}")

# 4. Load Models:
xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))

lstm_model = LSTMForecaster(input_size=len(LSTM_FEATURE_COLS))
lstm_model.load_state_dict(
    torch.load(os.path.join(MODELS_DIR, "lstm_best.pt"), map_location=DEVICE)
)
lstm_model.eval()

print("[4] Models loaded: XGBoost + LSTM")

# 5. Benchmark Function:
def benchmark_model(name, predict_fn, data, batch_sizes, n_repeats=N_REPEATS):
    """
    Run prediction across different batch sizes and measure:
      - Latency per batch (ms)
      - Latency per sample (ms)
      - Throughput (samples / second)
    """
    results = []
    print(f"\n  --- {name} ---")
    print(f"  {'Batch':>8}  {'Latency(ms)':>12}  {'ms/sample':>10}  {'samples/s':>10}")
    print("  " + "-" * 47)

    for bs in batch_sizes:
        # slice data to batch size (cap at available samples):
        batch = data[:min(bs, len(data))]
        actual_bs = len(batch)

        # warm-up run (JIT / caching effects):
        _ = predict_fn(batch)

        # timed runs:
        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            _ = predict_fn(batch)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg_sec    = np.mean(times)
        avg_ms     = avg_sec * 1000
        ms_per_sample = avg_ms / actual_bs
        throughput = actual_bs / avg_sec

        print(f"  {actual_bs:>8,}  {avg_ms:>11.2f}  {ms_per_sample:>10.4f}  {throughput:>10.1f}")
        results.append({
            "batch_size": actual_bs,
            "latency_ms": round(avg_ms, 3),
            "ms_per_sample": round(ms_per_sample, 5),
            "throughput_per_sec": round(throughput, 1),
        })

    return results


# 6. Define Prediction Functions:
safe_feature_names = [sanitize_col(c) for c in XGB_FEATURE_COLS]

def xgb_predict(X_np):
    df_input = pd.DataFrame(X_np, columns=safe_feature_names)
    return xgb_model.predict(df_input)

def lstm_predict(X_np):
    tensor = torch.from_numpy(X_np.astype(np.float32))
    with torch.no_grad():
        return lstm_model(tensor).numpy()

# 7. Run Benchmarks:
# batch sizes from 1 (real-time inference) to full test set (batch scoring).
BATCH_SIZES = [1, 10, 50, 100, 500, 1000, 5000, len(X_test_xgb)]

print("\n[5] Running XGBoost latency benchmark...")
xgb_results = benchmark_model("XGBoost", xgb_predict, X_test_xgb, BATCH_SIZES)

print("\n[6] Running LSTM latency benchmark...")
lstm_results = benchmark_model("LSTM", lstm_predict, X_test_lstm, BATCH_SIZES)

# 8. Single-sample Latency Summary:
xgb_single  = xgb_results[0]   # batch_size = 1
lstm_single = lstm_results[0]

print("\n[7] Single-sample inference latency (real-time use case):")
print(f"    XGBoost : {xgb_single['latency_ms']:.3f} ms   "
      f"({1000/xgb_single['latency_ms']:.0f} predictions/sec)")
print(f"    LSTM    : {lstm_single['latency_ms']:.3f} ms   "
      f"({1000/lstm_single['latency_ms']:.0f} predictions/sec)")

# 9. Plot: Throughput vs Batch Size:
xgb_bs  = [r["batch_size"] for r in xgb_results]
xgb_thr = [r["throughput_per_sec"] for r in xgb_results]
xgb_lat = [r["ms_per_sample"] for r in xgb_results]

lstm_bs  = [r["batch_size"] for r in lstm_results]
lstm_thr = [r["throughput_per_sec"] for r in lstm_results]
lstm_lat = [r["ms_per_sample"] for r in lstm_results]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Throughput:
axes[0].plot(xgb_bs,  xgb_thr,  "o-", color="steelblue",  label="XGBoost", linewidth=2)
axes[0].plot(lstm_bs, lstm_thr,  "s-", color="darkorange", label="LSTM",    linewidth=2)
axes[0].set_xscale("log")
axes[0].set_xlabel("Batch Size (log scale)")
axes[0].set_ylabel("Throughput (predictions / second)")
axes[0].set_title("Throughput vs Batch Size")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Latency per sample:
axes[1].plot(xgb_bs,  xgb_lat,  "o-", color="steelblue",  label="XGBoost", linewidth=2)
axes[1].plot(lstm_bs, lstm_lat,  "s-", color="darkorange", label="LSTM",    linewidth=2)
axes[1].set_xscale("log")
axes[1].set_xlabel("Batch Size (log scale)")
axes[1].set_ylabel("Latency per Sample (ms)")
axes[1].set_title("Per-Sample Latency vs Batch Size")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.suptitle("Inference Latency & Throughput Benchmark", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "19_latency_throughput.png"))
plt.close()
print("[8] Saved: 19_latency_throughput.png")

# 10. Plot: Single-sample bar comparison:
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

models      = ["XGBoost", "LSTM"]
latencies   = [xgb_single["latency_ms"], lstm_single["latency_ms"]]
throughputs = [xgb_single["throughput_per_sec"], lstm_single["throughput_per_sec"]]
colors      = ["steelblue", "darkorange"]

bars = axes[0].bar(models, latencies, color=colors, width=0.4, edgecolor="white")
axes[0].set_title("Single-Sample Latency (ms)\n(lower is better)")
axes[0].set_ylabel("Latency (ms)")
for bar, val in zip(bars, latencies):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f} ms", ha="center", va="bottom", fontsize=11, fontweight="bold")

bars = axes[1].bar(models, throughputs, color=colors, width=0.4, edgecolor="white")
axes[1].set_title("Single-Sample Throughput (preds/sec)\n(higher is better)")
axes[1].set_ylabel("Predictions / Second")
for bar, val in zip(bars, throughputs):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

fig.suptitle("Real-Time Inference Performance (batch_size = 1)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "20_single_sample_latency.png"))
plt.close()
print("[9] Saved: 20_single_sample_latency.png")

# 11. Save Results to JSON:
report = {
    "benchmark_config": {
        "n_repeats": N_REPEATS,
        "device": str(DEVICE),
        "batch_sizes_tested": BATCH_SIZES,
    },
    "single_sample_summary": {
        "XGBoost": {
            "latency_ms": xgb_single["latency_ms"],
            "throughput_per_sec": xgb_single["throughput_per_sec"],
        },
        "LSTM": {
            "latency_ms": lstm_single["latency_ms"],
            "throughput_per_sec": lstm_single["throughput_per_sec"],
        },
    },
    "full_benchmark": {
        "XGBoost": xgb_results,
        "LSTM":    lstm_results,
    },
}

report_path = os.path.join(REPORTS_DIR, "latency_benchmark.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print("[10] Saved: reports/latency_benchmark.json")

# 12. Final Summary:
print("\n" + "=" * 60)
print("Latency Benchmark Complete")
print("=" * 60)
print(f"\n  Real-time (single-sample) inference:")
print(f"  {'Model':<12}  {'Latency':>10}  {'Throughput':>15}")
print("  " + "-" * 42)
print(f"  {'XGBoost':<12}  {xgb_single['latency_ms']:>8.3f}ms  "
      f"{xgb_single['throughput_per_sec']:>12.0f} preds/s")
print(f"  {'LSTM':<12}  {lstm_single['latency_ms']:>8.3f}ms  "
      f"{lstm_single['throughput_per_sec']:>12.0f} preds/s")
print(f"\n  XGBoost is {lstm_single['latency_ms']/xgb_single['latency_ms']:.1f}× faster than LSTM "
      f"for single-sample inference.")
print("=" * 60)
