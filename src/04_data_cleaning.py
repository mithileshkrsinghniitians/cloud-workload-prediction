import pandas as pd
import numpy as np
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH   = os.path.join(PROJECT_ROOT, "data", "cleaned", "combined_raw.csv")
OUTPUT_PATH  = os.path.join(PROJECT_ROOT, "data", "processed", "combined_cleaned.csv")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

print("=" * 60)
print("PHASE 4 — Data Cleaning")
print("=" * 60)

# ── 1. Load ────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH)
print(f"\n[1] Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── 2. Parse & Set Datetime Index ──────────────────────────────────────────────
# Timestamp column is actually in seconds (Unix epoch), not ms — confirmed by EDA dates
df["datetime"] = pd.to_datetime(df["Timestamp [ms]"], unit="s")
df = df.drop(columns=["Timestamp [ms]"])
df = df.sort_values(["vm_id", "datetime"]).reset_index(drop=True)
print(f"[2] Datetime parsed. Range: {df['datetime'].min()} → {df['datetime'].max()}")

# ── 3. Missing Values ──────────────────────────────────────────────────────────
missing_before = df.isnull().sum().sum()
print(f"\n[3] Missing values before cleaning: {missing_before}")

if missing_before > 0:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    # Forward-fill within each VM, then back-fill any leading NaNs
    df[numeric_cols] = (
        df.groupby("vm_id")[numeric_cols]
        .transform(lambda s: s.ffill().bfill())
    )
    print(f"    Missing values after fill: {df.isnull().sum().sum()}")
else:
    print("    No missing values — nothing to fill.")

# ── 4. CPU Clipping — cap at [0, 100] ─────────────────────────────────────────
# EDA showed CPU reaching 106.21% (overcommit artifact). Cap at 100.
cpu_over  = (df["CPU usage [%]"] > 100).sum()
cpu_under = (df["CPU usage [%]"] < 0).sum()
df["CPU usage [%]"] = df["CPU usage [%]"].clip(0, 100)
print(f"\n[4] CPU clipping:")
print(f"    Values > 100%  capped: {cpu_over:,}")
print(f"    Values < 0%    capped: {cpu_under:,}")

# ── 5. Drop constant / zero-variance columns ───────────────────────────────────
# CPU cores is always 8 across selected VMs — not useful as a feature
before_cols = df.shape[1]
zero_var = [c for c in df.select_dtypes(include=np.number).columns
            if df[c].nunique() <= 1]
df = df.drop(columns=zero_var)
print(f"\n[5] Dropped {len(zero_var)} zero-variance column(s): {zero_var}")

# ── 6. Remove Duplicate Rows ───────────────────────────────────────────────────
dups = df.duplicated(subset=["vm_id", "datetime"]).sum()
df = df.drop_duplicates(subset=["vm_id", "datetime"])
print(f"\n[6] Duplicate (vm_id, datetime) rows removed: {dups:,}")

# ── 7. Ensure Regular 5-Minute Intervals per VM ───────────────────────────────
print("\n[7] Checking time interval regularity per VM:")
EXPECTED_INTERVAL = pd.Timedelta(minutes=5)
gap_report = []
for vm_id, grp in df.groupby("vm_id"):
    grp = grp.sort_values("datetime")
    diffs = grp["datetime"].diff().dropna()
    irregular = (diffs != EXPECTED_INTERVAL).sum()
    gap_report.append({"vm_id": vm_id, "total_steps": len(grp), "irregular_gaps": irregular})

gap_df = pd.DataFrame(gap_report)
print(gap_df.to_string(index=False))

# ── 8. Derive Memory Usage % ───────────────────────────────────────────────────
# Useful normalised feature alongside raw KB values
mask = df["Memory capacity provisioned [KB]"] > 0
df["Memory usage [%]"] = np.where(
    mask,
    (df["Memory usage [KB]"] / df["Memory capacity provisioned [KB]"]) * 100,
    0.0
).clip(0, 100)
print(f"\n[8] Derived 'Memory usage [%]' column added.")

# ── 9. Final Shape & Quick Stats ──────────────────────────────────────────────
print(f"\n[9] Final dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"    Columns: {list(df.columns)}")
print(f"\n    CPU usage [%] stats after cleaning:")
print(df["CPU usage [%]"].describe().round(2).to_string())

# ── 10. Save ───────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Saved cleaned dataset → data/processed/combined_cleaned.csv")
print("=" * 60)
