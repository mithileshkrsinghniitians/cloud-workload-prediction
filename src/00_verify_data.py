import pandas as pd
import os
import glob

# Path to your raw data (resolve relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "fastStorage", "2013-8")

# Count total files
files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
print(f"Total VM trace files found: {len(files)}")

# Load one file and inspect
sample_file = files[0]
print(f"\nLoading sample file: {sample_file}")

# Bitbrains uses semicolon-tab separator
df = pd.read_csv(sample_file, sep=";\t", engine="python")

print(f"\nShape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nData types:")
print(df.dtypes)
print(f"\nBasic statistics:")
print(df.describe())
print(f"\nMissing values:")
print(df.isnull().sum())

# Quick check across multiple files
print("\n--- Quick Health Check Across 10 VMs ---")
for f in files[:10]:
    temp = pd.read_csv(f, sep=";\t", engine="python")
    vm_id = os.path.basename(f).replace(".csv", "")
    print(f"VM {vm_id}: {temp.shape[0]} rows, "
          f"CPU% range: {temp['CPU usage [%]'].min():.2f} - {temp['CPU usage [%]'].max():.2f}, "
          f"Missing: {temp.isnull().sum().sum()}")

print("\n✅ Data verification complete!")
print("Completed")