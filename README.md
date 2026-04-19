# Cloud Workload Prediction — NCI MSc Cloud Computing (2026)

A cloud-based machine learning system that predicts VM CPU usage 30 minutes ahead using real-world workload traces. Built as part of the NCI Cloud Machine Learning module (MSCCLOUD1), the project covers the full pipeline — from data ingestion and feature engineering to model training, evaluation, cloud deployment, and auto-scaling simulation.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [ML Pipeline](#ml-pipeline)
4. [Feature Engineering](#feature-engineering)
5. [Models & Performance](#models--performance)
6. [Latency Benchmark](#latency-benchmark)
7. [Auto-Scaling Simulation](#auto-scaling-simulation)
8. [REST API](#rest-api)
9. [Streamlit Dashboard](#streamlit-dashboard)
10. [Project Structure](#project-structure)
11. [Deployment](#deployment)
    - [Local Development](#1-local-development)
    - [AWS Elastic Beanstalk](#2-aws-elastic-beanstalk)
    - [Windows Server via Docker Hub](#3-windows-server-via-docker-hub)

---

## Project Overview

| Item | Detail |
|------|--------|
| Goal | Predict cloud VM CPU usage 30 minutes ahead for proactive auto-scaling |
| Target variable | CPU usage [%] at t+6 (6 × 5-min steps = 30 min) |
| Best model | XGBoost (MAE 6.14%, R² 0.854) |
| Dataset | Bitbrains fastStorage (Aug–Sep 2013) |
| Selected VMs | 10 (IDs: 195, 202, 207, 208, 210, 212, 214, 215, 216, 217) |
| Training samples | 38,507 rows (Aug 5–25) |
| Test samples | 20,109 rows (Aug 26–Sep 1) |
| Features | 39 engineered features per sample |

---

## Dataset

- **Source:** Bitbrains fastStorage — [GWA-T-12](http://gwa.ewi.tudelft.nl/datasets/gwa-t-12-bitbrains)
- **Scale:** 1,250 anonymised VMs, 5-minute sampling intervals
- **Period:** August–September 2013 (30 days)
- **Selected:** Top 10 VMs by CPU variance
- **After processing:** 99,348 rows × 39 engineered features
- **Raw columns:** CPU usage [%], CPU cores, memory usage/capacity, network Rx/Tx, disk Rx/Tx

---

## ML Pipeline

Fifteen sequential scripts in `src/` cover the full end-to-end workflow:

| Script | Purpose |
|--------|---------|
| `00_verify_data.py` | Raw data validation — file counts, missing timestamps, column checks |
| `01_select_vms.py` | Select top 10 VMs by CPU variance from 1,250 candidates |
| `02_combine_vms.py` | Merge selected VMs into a single combined dataset |
| `03_eda.py` | Exploratory data analysis — distributions, time-series plots, correlations |
| `04_data_cleaning.py` | Interpolate missing values, remove duplicates, enforce monotonic timestamps |
| `05_feature_engineering.py` | Build all 39 features (lags, rolling stats, cyclical encodings, derived metrics) |
| `06_train_xgboost.py` | Train XGBoost with early stopping; save `xgboost_model.pkl` |
| `07_train_lstm.py` | Train 2-layer LSTM (PyTorch, CPU-forced); save `lstm_best.pt` + `lstm_scaler.pkl` |
| `08_train_prophet.py` | Train per-VM additive Prophet baseline; save `prophet_models.pkl` |
| `09_evaluate.py` | Evaluate all models on held-out test set; produce `model_comparison.json` |
| `10_latency_benchmark.py` | Measure inference latency and throughput across batch sizes |
| `11_learning_curves.py` | Learning curve analysis (model efficiency vs training data size) |
| `12_error_analysis.py` | Error breakdown by CPU regime, temporal pattern, and VM transitions |
| `13_autoscaling_sim.py` | Simulate reactive vs predictive auto-scaling on all 10 VMs |
| `14_cloud_storage.py` | Upload models, data, and reports to AWS S3 |

---

## Feature Engineering

39 features are computed per sample, per VM:

| Category | Features |
|----------|---------|
| Time | `hour`, `day_of_week`, `is_weekend`, `day_of_month`, `week` |
| Cyclical | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` |
| Lag features | `cpu_lag1`–`cpu_lag6`, `cpu_lag9`, `cpu_lag12` (5–60 min lookback) |
| Rolling mean | `cpu_roll_mean_3`, `cpu_roll_mean_6`, `cpu_roll_mean_12` (15/30/60 min) |
| Rolling std | `cpu_roll_std_3`, `cpu_roll_std_6`, `cpu_roll_std_12` |
| Rolling max | `cpu_roll_max_3`, `cpu_roll_max_6`, `cpu_roll_max_12` |
| Momentum | `cpu_diff1`, `cpu_diff2` |
| Derived | `mem_pressure`, `net_total_throughput`, `disk_total_throughput`, `cpu_util_ratio` |

---

## Models & Performance

Test period: **Aug 26 – Sep 1, 2013** (20,109 samples across 10 VMs)

| Model | MAE (%) | RMSE (%) | R² | sMAPE (%) |
|-------|---------|----------|----|-----------|
| **XGBoost** | **6.14** | **13.69** | **0.8543** | 31.28 |
| LSTM | 7.08 | 14.58 | 0.8309 | **27.52** |
| Prophet | 81.31 | 88.95 | -5.079 | 200.0 |
| Naive (lag-1) | 4.17 | 15.48 | 0.813 | 12.56 |

**Winner: XGBoost** — best MAE, RMSE, and R². LSTM achieves lower sMAPE.
Prophet is completely unsuitable for this task (R² = -5.08).

**Model configuration:**
- **XGBoost:** 1,000 trees, learning rate 0.05, max depth 6, early stopping patience 50
- **LSTM:** 2-layer, hidden size 64, sequence length 12, PyTorch (CPU forced — MPS diverges on Apple Silicon)
- LSTM test predictions are precomputed to `lstm_test_predictions.csv` to avoid MPS issues at inference time

---

## Latency Benchmark

Benchmarked on CPU (5 repeats per batch size):

| Model | Single-sample latency | Batch throughput (20K samples) |
|-------|-----------------------|-------------------------------|
| XGBoost | 0.79 ms | 8.4M samples/sec |
| LSTM | 0.22 ms | 114K samples/sec |

XGBoost dominates at large batch sizes due to vectorised tree traversal.

---

## Auto-Scaling Simulation

Simulation config: scale-up at >70% CPU, scale-down at <30%, SLA violation at >85%, cooldown 15 min.

| Strategy | SLA Violations | Violation Rate | Scale Events |
|----------|---------------|---------------|-------------|
| No scaling | 1,550 | 77.0% | 0 |
| Reactive | **54** | **0.27%** | 373 |
| Predictive (XGBoost) | 100 | 0.50% | 299 |

**Key finding:** Reactive scaling outperforms predictive (54 vs 100 violations) because XGBoost prediction errors at CPU regime transitions introduce false positives. Predictive scaling does reduce total scaling events (299 vs 373), which is more cost-efficient.

---

## REST API

Built with FastAPI, served via Uvicorn on port 8000.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message + available endpoints |
| `/health` | GET | Health check — model loaded status |
| `/models` | GET | Model metadata and evaluation metrics |
| `/predict` | POST | Single prediction (39-feature Pydantic schema) |
| `/predict/batch` | POST | Batch prediction |
| `/docs` | GET | Auto-generated Swagger UI |

---

## Streamlit Dashboard

9-tab interactive dashboard (`dashboard/streamlit_app.py`) served on port 8501:

| Tab | Content |
|-----|---------|
| Overview | Project summary, dataset stats, model cards |
| EDA | CPU distributions, time-series plots, per-VM stats |
| Feature Engineering | Feature categories, lag structure, rolling window visualisations |
| Model Results | MAE/RMSE/R²/sMAPE comparison table and charts |
| XGBoost | Feature importance, actual vs predicted, residuals |
| LSTM | Loss curve, actual vs predicted |
| Prophet | Forecast, components, failure analysis |
| Error Analysis | Error by CPU regime, temporal heatmaps, transition analysis |
| Auto-Scaling | Reactive vs predictive simulation, SLA violation comparison |

---

## Project Structure

```
cloud-workload-prediction/
├── data/
│   ├── raw/                        # Original Bitbrains VM CSV files (1,250 VMs)
│   ├── cleaned/                    # combined_raw.csv (10 selected VMs merged)
│   └── processed/                  # combined_features.csv (99,348 × 39 features)
├── src/                            # ML pipeline scripts (00_verify → 14_cloud_storage)
├── models/
│   ├── xgboost_model.pkl           # Trained XGBoost model
│   ├── lstm_best.pt                # Trained LSTM (PyTorch)
│   ├── lstm_scaler.pkl             # LSTM feature scaler
│   ├── lstm_test_predictions.csv   # Precomputed LSTM predictions
│   └── prophet_models.pkl          # Per-VM Prophet models
├── api/
│   ├── app.py                      # FastAPI application (39-field schema, 5 endpoints)
│   └── Dockerfile
├── dashboard/
│   ├── streamlit_app.py            # Streamlit dashboard (9 tabs)
│   └── Dockerfile
├── reports/
│   ├── model_comparison.json
│   ├── xgboost_metrics.json
│   ├── lstm_metrics.json
│   ├── prophet_metrics.json
│   ├── latency_benchmark.json
│   ├── learning_curves.json
│   ├── error_analysis.json
│   ├── autoscaling_simulation.json
│   ├── cloud_storage_report.json
│   ├── NCI_Report_CloudML.md
│   └── figures/                    # 29 PNG visualisations
├── iis/                            # IIS web.config reference files
├── docker-compose.yml              # Local development
├── docker-compose.prod.yml         # AWS Elastic Beanstalk (ECR images)
├── docker-compose.hub.yml          # Windows Server deployment (Docker Hub images)
├── Dockerrun.aws.json              # EB v1 config (reference only)
├── .elasticbeanstalk/config.yml    # EB app=cwp-api, env=cwp-env, region=us-east-1
└── requirements.txt
```

---

## Deployment

### 1. Local Development

```bash
docker compose up --build
```

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |

---

### 2. AWS Elastic Beanstalk

Images hosted on ECR (`us-east-1`). Deployed via `docker-compose.prod.yml`.

```bash
# Build and push to ECR (must use --platform linux/amd64 from Mac ARM64)
docker build --platform linux/amd64 -f api/Dockerfile -t 093452070667.dkr.ecr.us-east-1.amazonaws.com/cwp-api:latest .
docker build --platform linux/amd64 -f dashboard/Dockerfile -t 093452070667.dkr.ecr.us-east-1.amazonaws.com/cwp-dashboard:latest .

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 093452070667.dkr.ecr.us-east-1.amazonaws.com
docker push 093452070667.dkr.ecr.us-east-1.amazonaws.com/cwp-api:latest
docker push 093452070667.dkr.ecr.us-east-1.amazonaws.com/cwp-dashboard:latest

eb deploy
```

| Resource | Detail |
|----------|--------|
| App | `cwp-api` |
| Environment | `cwp-env` |
| Region | `us-east-1` |
| API URL | http://cwp-env.eba-mr7hdmtp.us-east-1.elasticbeanstalk.com |
| Dashboard | http://cwp-env.eba-mr7hdmtp.us-east-1.elasticbeanstalk.com:8501 |
| S3 Bucket | `cloud-workload-prediction-nci` (eu-west-1) |

---

### 3. Windows Server via Docker Hub

Images hosted on Docker Hub (`mithileshkrsinghniitians`). Deployed to a Windows Server with Hyper-V + Linux VM at `192.168.0.203`, with IIS as the reverse proxy.

**Step 1 — Build & push (from Mac):**

```bash
docker build --platform linux/amd64 -f api/Dockerfile -t mithileshkrsinghniitians/cwp-api:latest .
docker build --platform linux/amd64 -f dashboard/Dockerfile -t mithileshkrsinghniitians/cwp-dashboard:latest .

docker push mithileshkrsinghniitians/cwp-api:latest
docker push mithileshkrsinghniitians/cwp-dashboard:latest
```

**Step 2 — Copy compose file to server:**

```bash
scp -P 19874 docker-compose.hub.yml singh-aws@ssh.cloudssfdc.com:~/docker-compose.hub.yml
```

**Step 3 — SSH and deploy:**

```bash
ssh -p 19874 singh-aws@ssh.cloudssfdc.com

docker compose -f docker-compose.hub.yml pull
docker compose -f docker-compose.hub.yml up -d
```

Both containers start with `restart: always` — they auto-start after every server reboot without manual intervention.

**Port mapping:**

| Service | Container Port | Host Port |
|---------|---------------|-----------|
| API (FastAPI) | 8000 | 8083 |
| Dashboard (Streamlit) | 8501 | 8501 |

**IIS Reverse Proxy:**

Two IIS sites are configured to route the subdomain `cml.cloudssfdc.com`:

- Port 80 → proxies to `http://192.168.0.203:8083` (API)
- Port 8501 → proxies to `http://192.168.0.203:8501` (Dashboard, WebSocket enabled via ARR)

Reference web.config files are in the `iis/` folder.

| URL | Service |
|-----|---------|
| http://cml.cloudssfdc.com | FastAPI REST API |
| http://cml.cloudssfdc.com/docs | Swagger UI |
| http://cml.cloudssfdc.com:8501 | Streamlit Dashboard |

---

## Key Insights

- XGBoost plateaus at ~60% of training data (~23K rows) — data-efficient
- Prediction errors are 2.4× higher at CPU regime transitions vs stable periods
- Moderate CPU regime (20–70%) is hardest to predict, yet represents only 5.7% of data
- Reactive auto-scaling outperforms predictive due to transition prediction errors
- XGBoost single-sample latency: 0.79ms; batch throughput at 20K samples: 8.4M/sec