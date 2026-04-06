# Cloud Workload Prediction — NCI MSc Cloud Computing (2026)

A cloud-based machine learning system that predicts VM CPU usage 30 minutes ahead using real-world workload traces. Built as part of the NCI Cloud Machine Learning module (MSCCLOUD1), the project covers the full pipeline — from data ingestion and feature engineering to model training, evaluation, cloud deployment, and auto-scaling simulation.

## Models

| Model | Role | MAE | RMSE | R² |
|-------|------|-----|------|-----|
| **XGBoost** | Primary (best performance) | 6.14% | 13.69% | 0.854 |
| **LSTM** | Deep learning (temporal) | 7.08% | 14.58% | 0.831 |
| **Prophet** | Baseline (time-series) | 81.31% | 88.95% | -5.08 |
| **Naive (lag-1)** | Persistence baseline | 4.17% | 15.48% | 0.813 |

## Dataset

- **Source:** Bitbrains fastStorage — [GWA-T-12](http://gwa.ewi.tudelft.nl/datasets/gwa-t-12-bitbrains)
- **Scale:** 1,250 anonymised VMs, 5-minute sampling intervals
- **Period:** August–September 2013 (30 days)
- **Selected:** Top 10 VMs by CPU variance (IDs: 195, 202, 207, 208, 210, 212, 214, 215, 216, 217)
- **After processing:** 99,348 rows × 39 engineered features
- **Target:** CPU usage [%], 6-step ahead forecast (30 minutes)

## Project Structure

```
cloud-workload-prediction/
├── data/
│   ├── raw/                    # Original Bitbrains VM CSV files (1,250 VMs)
│   ├── cleaned/                # Combined 10-VM dataset (combined_raw.csv)
│   └── processed/              # Feature-engineered data (39 features)
├── src/                        # ML pipeline scripts (00_verify → 14_cloud_storage)
├── models/                     # Trained model artifacts (XGBoost, LSTM, Prophet)
├── api/                        # FastAPI REST service + Dockerfile
├── dashboard/                  # Streamlit interactive dashboard + Dockerfile
├── reports/                    # Metrics JSON, error analysis, 29 visualisations
├── docker-compose.yml          # Local orchestration (API + Dashboard)
├── Dockerrun.aws.json          # AWS Elastic Beanstalk deployment config
├── requirements.txt
└── README.md
```
