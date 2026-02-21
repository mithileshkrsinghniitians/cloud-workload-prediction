import pandas as pd
import os
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "fastStorage", "2013-8")
files = glob.glob(os.path.join(DATA_PATH, "*.csv"))

vm_stats = []

print("Analyzing all 1,250 VMs... (this may take a minute)")

for f in files:
    df = pd.read_csv(f, sep=";\t", engine="python")
    vm_id = os.path.basename(f).replace(".csv", "")

    stats = {
        "vm_id": vm_id,
        "rows": len(df),
        "cpu_mean": df["CPU usage [%]"].mean(),
        "cpu_std": df["CPU usage [%]"].std(),
        "cpu_max": df["CPU usage [%]"].max(),
        "missing": df.isnull().sum().sum()
    }
    vm_stats.append(stats)

stats_df = pd.DataFrame(vm_stats)

# We want VMs that are:
# - Not idle (cpu_mean > 5%)
# - Dynamic/interesting (cpu_std > 5)
# - Complete data (missing == 0)
good_vms = stats_df[
    (stats_df["cpu_mean"] > 5) &
    (stats_df["cpu_std"] > 5) &
    (stats_df["missing"] == 0)
    ].sort_values("cpu_std", ascending=False)

print(f"\nTotal VMs: {len(stats_df)}")
print(f"VMs with interesting patterns: {len(good_vms)}")
print(f"\nTop 10 most dynamic VMs:")
print(good_vms.head(10).to_string(index=False))

# Save selected VM IDs
selected_vms = good_vms.head(10)["vm_id"].tolist()
print(f"\n✅ Selected VM IDs: {selected_vms}")

# Save the selection for later use
os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)
with open(os.path.join(PROJECT_ROOT, "data", "selected_vms.txt"), "w") as f:
    for vm in selected_vms:
        f.write(f"{vm}\n")

print("Saved to data/selected_vms.txt")