import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

# Paths:
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "cleaned", "combined_raw.csv")
FIGURES_DIR  = os.path.join(PROJECT_ROOT, "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (12, 5)})
sns.set_theme(style="whitegrid")

# 1. Load:
print("=" * 60)
print("PHASE 3 — Exploratory Data Analysis")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n[1] Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Convert timestamp from ms → datetime:
df["datetime"] = pd.to_datetime(df["Timestamp [ms]"], unit="s")
df = df.sort_values(["vm_id", "datetime"]).reset_index(drop=True)

# 2. Basic Info:
print("\n[2] Columns & dtypes:")
print(df.dtypes.to_string())

print(f"\n[3] VMs in dataset : {df['vm_id'].nunique()}")
print(f"    Date range     : {df['datetime'].min()}  →  {df['datetime'].max()}")
print(f"    Duration       : {(df['datetime'].max() - df['datetime'].min()).days} days")

# 3. Missing Values:
print("\n[4] Missing values per column:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({"count": missing, "pct": missing_pct})
print(missing_report[missing_report["count"] > 0].to_string()
      if missing_report["count"].sum() > 0 else "    None — dataset is complete")

# 4. Descriptive Statistics:
numeric_cols = [
    "CPU usage [%]", "CPU usage [MHZ]", "CPU capacity provisioned [MHZ]",
    "Memory usage [KB]", "Memory capacity provisioned [KB]",
    "Disk read throughput [KB/s]", "Disk write throughput [KB/s]",
    "Network received throughput [KB/s]", "Network transmitted throughput [KB/s]",
]

print("\n[5] Descriptive statistics:")
print(df[numeric_cols].describe().round(2).to_string())

# 5. CPU Usage Distribution:
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df["CPU usage [%]"], bins=80, color="steelblue", edgecolor="white", linewidth=0.3)
axes[0].set_title("CPU Usage Distribution (All VMs)")
axes[0].set_xlabel("CPU Usage [%]")
axes[0].set_ylabel("Frequency")

axes[1].boxplot(df["CPU usage [%]"], vert=True, patch_artist=True,
                boxprops=dict(facecolor="steelblue", alpha=0.6))
axes[1].set_title("CPU Usage Boxplot (All VMs)")
axes[1].set_ylabel("CPU Usage [%]")
axes[1].set_xticks([])

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "01_cpu_distribution.png"))
plt.close()
print("\n[6] Saved: 01_cpu_distribution.png")

# 6. Per-VM CPU Statistics:
vm_stats = df.groupby("vm_id")["CPU usage [%]"].agg(
    mean="mean", std="std", min="min", max="max", median="median"
).round(2)

print(f"\n[7] Per-VM CPU stats (top 10 by mean):")
print(vm_stats.sort_values("mean", ascending=False).head(10).to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

vm_stats["mean"].sort_values(ascending=False).plot(
    kind="bar", ax=axes[0], color="steelblue", width=0.8)
axes[0].set_title("Mean CPU Usage per VM")
axes[0].set_xlabel("VM ID")
axes[0].set_ylabel("Mean CPU [%]")
axes[0].tick_params(axis="x", labelsize=5, rotation=90)

vm_stats["std"].sort_values(ascending=False).plot(
    kind="bar", ax=axes[1], color="coral", width=0.8)
axes[1].set_title("CPU Std Dev per VM (Variability)")
axes[1].set_xlabel("VM ID")
axes[1].set_ylabel("Std Dev [%]")
axes[1].tick_params(axis="x", labelsize=5, rotation=90)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "02_per_vm_cpu_stats.png"))
plt.close()
print("[8] Saved: 02_per_vm_cpu_stats.png")

# 7. Time-Series CPU for Sample VMs:
sample_vms = df["vm_id"].value_counts().head(5).index.tolist()

fig, axes = plt.subplots(len(sample_vms), 1, figsize=(14, 3 * len(sample_vms)), sharex=False)
for ax, vm in zip(axes, sample_vms):
    vm_df = df[df["vm_id"] == vm].set_index("datetime")
    ax.plot(vm_df.index, vm_df["CPU usage [%]"], linewidth=0.7, color="steelblue")
    ax.set_title(f"VM {vm} — CPU over Time")
    ax.set_ylabel("CPU [%]")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "03_cpu_timeseries_sample_vms.png"))
plt.close()
print("[9] Saved: 03_cpu_timeseries_sample_vms.png")

# 8. Temporal Patterns — Hourly & Daily:
df["hour"]       = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.day_name()

hourly_cpu = df.groupby("hour")["CPU usage [%]"].mean()
dow_order  = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
daily_cpu  = df.groupby("day_of_week")["CPU usage [%]"].mean().reindex(dow_order)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(hourly_cpu.index, hourly_cpu.values, marker="o", color="steelblue")
axes[0].set_title("Average CPU Usage by Hour of Day")
axes[0].set_xlabel("Hour")
axes[0].set_ylabel("Mean CPU [%]")
axes[0].set_xticks(range(0, 24))

axes[1].bar(daily_cpu.index, daily_cpu.values, color="steelblue")
axes[1].set_title("Average CPU Usage by Day of Week")
axes[1].set_xlabel("Day")
axes[1].set_ylabel("Mean CPU [%]")
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "04_temporal_patterns.png"))
plt.close()
print("[10] Saved: 04_temporal_patterns.png")

# 9. Correlation Heatmap:
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax)
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "05_correlation_heatmap.png"))
plt.close()
print("[11] Saved: 05_correlation_heatmap.png")

# 10. Outlier Summary (IQR method on CPU):
Q1  = df["CPU usage [%]"].quantile(0.25)
Q3  = df["CPU usage [%]"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = df[(df["CPU usage [%]"] < lower) | (df["CPU usage [%]"] > upper)]

print(f"\n[12] Outlier Detection (IQR on CPU usage [%]):")
print(f"     Q1={Q1:.2f}  Q3={Q3:.2f}  IQR={IQR:.2f}")
print(f"     Lower fence={lower:.2f}  Upper fence={upper:.2f}")
print(f"     Outlier rows: {len(outliers):,} ({len(outliers)/len(df)*100:.2f}%)")

# 11. EDA Summary:
print("\n" + "=" * 60)
print("EDA SUMMARY")
print("=" * 60)
print(f"  Total rows         : {len(df):,}")
print(f"  VMs                : {df['vm_id'].nunique()}")
print(f"  Date range         : {df['datetime'].min().date()} → {df['datetime'].max().date()}")
print(f"  CPU usage [%] mean : {df['CPU usage [%]'].mean():.2f}%")
print(f"  CPU usage [%] std  : {df['CPU usage [%]'].std():.2f}%")
print(f"  CPU usage [%] range: {df['CPU usage [%]'].min():.2f}% → {df['CPU usage [%]'].max():.2f}%")
print(f"  Missing values     : {df[numeric_cols].isnull().sum().sum()}")
print(f"  CPU outliers       : {len(outliers):,} rows ({len(outliers)/len(df)*100:.2f}%)")
print(f"\n  Figures saved to   : {FIGURES_DIR}")
print("\n EDA complete!")