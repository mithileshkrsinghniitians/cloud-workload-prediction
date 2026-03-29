"""
Phase 14 — AWS S3 Cloud Storage Integration
=============================================
Demonstrates cloud computing for data collection, storage and mining.

What this script does:
  1. Upload raw/processed data to S3        (cloud data storage)
  2. Upload trained models & reports to S3  (cloud model registry)
  3. Download data from S3                  (cloud data collection / retrieval)
  4. List all objects in the S3 bucket      (cloud inventory)

Setup (before running):
  1. Install AWS CLI and configure credentials:
       aws configure
     (or set environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION)

  2. Create an S3 bucket in your AWS account:
       aws s3 mb s3://your-bucket-name --region eu-west-1

  3. Set your bucket name:
       export S3_BUCKET_NAME=your-bucket-name
     or edit S3_BUCKET_NAME below directly.

Run:
  python src/14_cloud_storage.py
"""

import os
import sys
import json
import time
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# ── Configuration ──────────────────────────────────────────────────────────────
# Read bucket name from environment variable (recommended) or set it here directly
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "cloud-workload-prediction-nci")
S3_REGION      = os.environ.get("AWS_DEFAULT_REGION", "eu-west-1")
S3_PREFIX      = "bitbrains/"   # all objects will be stored under this prefix

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Files to upload — organised by S3 folder
UPLOAD_MANIFEST = {
    # Raw & processed data
    "data/raw": [
        os.path.join(PROJECT_ROOT, "data", "selected_vms.txt"),
    ],
    "data/processed": [
        os.path.join(PROJECT_ROOT, "data", "processed", "combined_cleaned.csv"),
        os.path.join(PROJECT_ROOT, "data", "processed", "combined_features.csv"),
    ],
    "data/cleaned": [
        os.path.join(PROJECT_ROOT, "data", "cleaned", "combined_raw.csv"),
    ],
    # Trained models
    "models": [
        os.path.join(PROJECT_ROOT, "models", "xgboost_model.pkl"),
        os.path.join(PROJECT_ROOT, "models", "lstm_best.pt"),
        os.path.join(PROJECT_ROOT, "models", "lstm_scaler.pkl"),
        os.path.join(PROJECT_ROOT, "models", "lstm_test_predictions.csv"),
    ],
    # Reports & metrics
    "reports": [
        os.path.join(PROJECT_ROOT, "reports", "model_comparison.json"),
        os.path.join(PROJECT_ROOT, "reports", "xgboost_metrics.json"),
        os.path.join(PROJECT_ROOT, "reports", "lstm_metrics.json"),
        os.path.join(PROJECT_ROOT, "reports", "prophet_metrics.json"),
        os.path.join(PROJECT_ROOT, "reports", "latency_benchmark.json"),
        os.path.join(PROJECT_ROOT, "reports", "learning_curves.json"),
        os.path.join(PROJECT_ROOT, "reports", "error_analysis.json"),
        os.path.join(PROJECT_ROOT, "reports", "autoscaling_simulation.json"),
    ],
}

DOWNLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "s3_downloads")


# ── Helper Functions ───────────────────────────────────────────────────────────
def get_s3_client():
    """Create and return a boto3 S3 client."""
    return boto3.client("s3", region_name=S3_REGION)


def human_size(num_bytes):
    """Convert bytes to a human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def upload_file(s3, local_path, bucket, s3_key):
    """Upload a single file to S3. Returns True on success."""
    if not os.path.exists(local_path):
        print(f"    [SKIP] File not found locally: {os.path.basename(local_path)}")
        return False

    file_size = os.path.getsize(local_path)
    t0 = time.perf_counter()
    try:
        s3.upload_file(local_path, bucket, s3_key)
        elapsed = time.perf_counter() - t0
        print(f"    [OK] {os.path.basename(local_path):45s} → s3://{bucket}/{s3_key}  "
              f"({human_size(file_size)}, {elapsed:.1f}s)")
        return True
    except ClientError as e:
        print(f"    [ERR] {os.path.basename(local_path)}: {e}")
        return False


def download_file(s3, bucket, s3_key, local_path):
    """Download a single file from S3. Returns True on success."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    t0 = time.perf_counter()
    try:
        s3.download_file(bucket, s3_key, local_path)
        elapsed = time.perf_counter() - t0
        file_size = os.path.getsize(local_path)
        print(f"    [OK] s3://{bucket}/{s3_key}  → {os.path.basename(local_path):40s}  "
              f"({human_size(file_size)}, {elapsed:.1f}s)")
        return True
    except ClientError as e:
        print(f"    [ERR] {s3_key}: {e}")
        return False


# ── Main Script ────────────────────────────────────────────────────────────────
print("=" * 60)
print("PHASE 14 — AWS S3 Cloud Storage Integration")
print("=" * 60)
print(f"\n  Bucket : s3://{S3_BUCKET_NAME}")
print(f"  Region : {S3_REGION}")
print(f"  Prefix : {S3_PREFIX}")

# ── 1. Connect to AWS ─────────────────────────────────────────────────────────
print("\n[1] Connecting to AWS S3...")
try:
    s3 = get_s3_client()
    # Quick connectivity test — check if bucket exists
    s3.head_bucket(Bucket=S3_BUCKET_NAME)
    print(f"    Connected. Bucket '{S3_BUCKET_NAME}' is accessible.")
except NoCredentialsError:
    print("\n[ERROR] AWS credentials not found.")
    print("  Run:  aws configure")
    print("  Or set environment variables:")
    print("    export AWS_ACCESS_KEY_ID=your_key")
    print("    export AWS_SECRET_ACCESS_KEY=your_secret")
    print("    export AWS_DEFAULT_REGION=eu-west-1")
    sys.exit(1)
except ClientError as e:
    error_code = e.response["Error"]["Code"]
    if error_code == "404":
        print(f"\n[ERROR] Bucket '{S3_BUCKET_NAME}' does not exist.")
        print(f"  Create it with:")
        print(f"    aws s3 mb s3://{S3_BUCKET_NAME} --region {S3_REGION}")
    elif error_code == "403":
        print(f"\n[ERROR] Access denied to bucket '{S3_BUCKET_NAME}'.")
        print("  Check your IAM permissions (need s3:GetObject, s3:PutObject, s3:ListBucket).")
    else:
        print(f"\n[ERROR] {e}")
    sys.exit(1)

# ── 2. Upload All Files ───────────────────────────────────────────────────────
print(f"\n[2] Uploading project files to s3://{S3_BUCKET_NAME}/{S3_PREFIX} ...")

total_uploaded = 0
total_skipped  = 0

for folder, file_list in UPLOAD_MANIFEST.items():
    print(f"\n  --- {folder} ---")
    for local_path in file_list:
        filename = os.path.basename(local_path)
        s3_key   = f"{S3_PREFIX}{folder}/{filename}"
        success  = upload_file(s3, local_path, S3_BUCKET_NAME, s3_key)
        if success:
            total_uploaded += 1
        else:
            total_skipped += 1

# Also upload all figure PNGs
figures_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
if os.path.exists(figures_dir):
    print(f"\n  --- reports/figures ---")
    for fname in sorted(os.listdir(figures_dir)):
        if fname.endswith(".png"):
            local_path = os.path.join(figures_dir, fname)
            s3_key     = f"{S3_PREFIX}reports/figures/{fname}"
            success    = upload_file(s3, local_path, S3_BUCKET_NAME, s3_key)
            if success:
                total_uploaded += 1
            else:
                total_skipped += 1

print(f"\n[2] Upload summary: {total_uploaded} files uploaded, {total_skipped} skipped.")

# ── 3. List Objects in Bucket ─────────────────────────────────────────────────
print(f"\n[3] Listing objects in s3://{S3_BUCKET_NAME}/{S3_PREFIX} ...")

paginator = s3.get_paginator("list_objects_v2")
pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=S3_PREFIX)

total_objects = 0
total_size    = 0

print(f"\n  {'S3 Key':<60}  {'Size':>10}  {'Last Modified'}")
print("  " + "-" * 90)

for page in pages:
    for obj in page.get("Contents", []):
        key        = obj["Key"]
        size       = obj["Size"]
        modified   = obj["LastModified"].strftime("%Y-%m-%d %H:%M")
        total_objects += 1
        total_size    += size
        # Shorten long keys for readability
        display_key = key.replace(S3_PREFIX, "")
        print(f"  {display_key:<60}  {human_size(size):>10}  {modified}")

print(f"\n  Total: {total_objects} objects  ({human_size(total_size)})")

# ── 4. Download Sample — Processed Data from S3 ───────────────────────────────
# This demonstrates "cloud data collection" — fetching data from the cloud
print(f"\n[4] Downloading processed data from S3 → {DOWNLOAD_DIR}/")
print("    (simulates fetching data from cloud for a fresh environment)")

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Download the key files needed for model inference
files_to_download = [
    (f"{S3_PREFIX}data/processed/combined_features.csv",
     os.path.join(DOWNLOAD_DIR, "combined_features.csv")),
    (f"{S3_PREFIX}models/xgboost_model.pkl",
     os.path.join(DOWNLOAD_DIR, "xgboost_model.pkl")),
    (f"{S3_PREFIX}reports/model_comparison.json",
     os.path.join(DOWNLOAD_DIR, "model_comparison.json")),
]

dl_success = 0
for s3_key, local_path in files_to_download:
    if download_file(s3, S3_BUCKET_NAME, s3_key, local_path):
        dl_success += 1

print(f"\n[4] Downloaded {dl_success}/{len(files_to_download)} files → {DOWNLOAD_DIR}/")

# ── 5. Save Cloud Storage Report ─────────────────────────────────────────────
report = {
    "bucket":          S3_BUCKET_NAME,
    "region":          S3_REGION,
    "prefix":          S3_PREFIX,
    "files_uploaded":  total_uploaded,
    "files_skipped":   total_skipped,
    "total_objects_in_bucket": total_objects,
    "total_size_bytes": total_size,
    "total_size_human": human_size(total_size),
    "s3_uri":          f"s3://{S3_BUCKET_NAME}/{S3_PREFIX}",
    "cloud_services_used": [
        "Amazon S3 (Simple Storage Service) — data, models, reports storage",
    ],
}
report_path = os.path.join(PROJECT_ROOT, "reports", "cloud_storage_report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\n[5] Report saved → reports/cloud_storage_report.json")

# ── 6. Final Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Cloud Storage Integration Complete")
print("=" * 60)
print(f"\n  S3 URI     : s3://{S3_BUCKET_NAME}/{S3_PREFIX}")
print(f"  Region     : {S3_REGION}")
print(f"  Uploaded   : {total_uploaded} files")
print(f"  Total size : {human_size(total_size)}")
print(f"\n  Cloud services demonstrated:")
print(f"    - Data storage in AWS S3 (raw + processed datasets)")
print(f"    - Model artifact storage in AWS S3 (models + scalers)")
print(f"    - Report storage in AWS S3 (metrics + figures)")
print(f"    - Data retrieval from AWS S3 (download for inference)")
print("=" * 60)
