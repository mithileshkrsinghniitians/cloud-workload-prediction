import pandas as pd
import numpy as np
import os

# Paths:
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH   = os.path.join(PROJECT_ROOT, "data", "processed", "combined_cleaned.csv")
OUTPUT_PATH  = os.path.join(PROJECT_ROOT, "data", "processed", "combined_features.csv")

print("=" * 60)
print("PHASE 5 — Feature Engineering")
print("=" * 60)

# 1. Load:
df = pd.read_csv(INPUT_PATH, parse_dates=["datetime"])
df = df.sort_values(["vm_id", "datetime"]).reset_index(drop=True)
print(f"\n[1] Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# 2. Time-Based Features:
df["hour"]        = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.dayofweek          # 0=Mon … 6=Sun
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
df["day_of_month"]= df["datetime"].dt.day
df["week"]        = df["datetime"].dt.isocalendar().week.astype(int)

# Cyclical encoding for hour and day_of_week (avoids 23→0 discontinuity):
df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"]     = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"]     = np.cos(2 * np.pi * df["day_of_week"] / 7)
print("[2] Time-based features added (hour, dow, is_weekend, cyclical encodings).")

# 3. CPU Utilisation Ratio:
mask = df["CPU capacity provisioned [MHZ]"] > 0
df["cpu_util_ratio"] = np.where(
    mask,
    df["CPU usage [MHZ]"] / df["CPU capacity provisioned [MHZ]"],
    0.0
).clip(0, 1)
print("[3] CPU utilisation ratio added (MHZ used / provisioned).")

# 4. Lag Features (per VM):
# Forecasting 6 steps ahead (30 min), so lags 1–6 are the minimum look-back.
# Add lags 1–12 (60 min) for richer short-term context.
LAG_STEPS = [1, 2, 3, 4, 5, 6, 9, 12]

def add_lags(group, col, steps):
    for s in steps:
        group[f"{col}_lag{s}"] = group[col].shift(s)
    return group

df = df.groupby("vm_id", group_keys=False).apply(
    lambda g: add_lags(g, "CPU usage [%]", LAG_STEPS)
)
print(f"[4] Lag features added: {['CPU usage [%]_lag' + str(s) for s in LAG_STEPS]}")

# 5. Rolling Window Statistics (per VM):
# Windows in steps (each step = 5 min):  3 = 15 min, 6 = 30 min, 12 = 60 min.
WINDOWS = {"15min": 3, "30min": 6, "60min": 12}

def add_rolling(group, col, windows):
    for label, w in windows.items():
        group[f"{col}_roll_mean_{label}"] = group[col].shift(1).rolling(w).mean()
        group[f"{col}_roll_std_{label}"]  = group[col].shift(1).rolling(w).std()
        group[f"{col}_roll_max_{label}"]  = group[col].shift(1).rolling(w).max()
    return group

df = df.groupby("vm_id", group_keys=False).apply(
    lambda g: add_rolling(g, "CPU usage [%]", WINDOWS)
)
print(f"[5] Rolling features added for windows: {list(WINDOWS.keys())} "
      f"(mean, std, max each).")

# 6. Diff Features (rate of change):
df = df.groupby("vm_id", group_keys=False).apply(
    lambda g: g.assign(**{
        "cpu_diff1": g["CPU usage [%]"].diff(1),   # 5-min change
        "cpu_diff2": g["CPU usage [%]"].diff(2),   # 10-min change
    })
)
print("[6] Diff features added (cpu_diff1, cpu_diff2).")

# 7. Memory Pressure:
df["mem_pressure"] = (df["Memory usage [%]"] / 100).clip(0, 1)
print("[7] Memory pressure feature added.")

# 8. Network & Disk Total:
df["net_total_throughput"]  = (df["Network received throughput [KB/s]"] +
                               df["Network transmitted throughput [KB/s]"])
df["disk_total_throughput"] = (df["Disk read throughput [KB/s]"] +
                               df["Disk write throughput [KB/s]"])
print("[8] Net/Disk total throughput features added.")

# 9. Drop Rows with NaN (introduced by lags/rolling):
rows_before = len(df)
df = df.dropna().reset_index(drop=True)
rows_dropped = rows_before - len(df)
print(f"\n[9] Dropped {rows_dropped:,} rows with NaN (from lag/rolling warmup).")
print(f"    Remaining rows: {len(df):,}")

# 10. Drop Columns Not Needed for Modeling:
# Keep raw resource columns as features but drop redundant/low-value ones.
drop_cols = [
    "CPU cores",                        # constant after cleaning check
    "CPU capacity provisioned [MHZ]",   # captured via cpu_util_ratio
    "Memory capacity provisioned [KB]", # captured via Memory usage [%]
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])
print(f"[10] Dropped low-value columns: {drop_cols}")

# 11. Column Summary:
feature_cols = [c for c in df.columns if c not in ("vm_id", "datetime", "CPU usage [%]")]
print(f"\n[11] Final feature set ({len(feature_cols)} features):")
for c in feature_cols:
    print(f"     {c}")

print(f"\n[12] Final dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(df.describe().round(2).to_string())

# 12. Save:
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n Saved → data/processed/combined_features.csv")
print("=" * 60)