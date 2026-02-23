import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "fastStorage", "2013-8")

# Load selected VMs
with open(os.path.join(PROJECT_ROOT, "data", "selected_vms.txt"), "r") as f:
    selected_vms = [line.strip() for line in f.readlines()]

print(f"Combining {len(selected_vms)} VMs...")

all_dfs = []
for vm_id in selected_vms:
    filepath = os.path.join(DATA_PATH, f"{vm_id}.csv")
    df = pd.read_csv(filepath, sep=";\t", engine="python")
    df["vm_id"] = vm_id  # Add VM identifier column
    all_dfs.append(df)

combined = pd.concat(all_dfs, ignore_index=True)

print(f"Combined dataset shape: {combined.shape}")
print(f"VMs included: {combined['vm_id'].nunique()}")
print(f"Total rows: {len(combined)}")

# Save combined raw data
cleaned_dir = os.path.join(PROJECT_ROOT, "data", "cleaned")
os.makedirs(cleaned_dir, exist_ok=True)
combined.to_csv(os.path.join(cleaned_dir, "combined_raw.csv"), index=False)
print("\n✅ Saved to data/cleaned/combined_raw.csv")