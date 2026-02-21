# Cloud Workload Prediction

Predict cloud VM CPU usage using machine learning models trained on the Bitbrains fastStorage dataset. Includes auto-scaling simulation and cloud deployment via AWS.

## Models
- **XGBoost** (primary)
- **LSTM** (deep learning)
- **Prophet** (baseline)

## Dataset
Bitbrains fastStorage traces — 1,250 VMs sampled at 5-minute intervals.

## Project Structure
```
cloud-workload-prediction/
├── data/                   # Raw, cleaned, and processed data
├── notebooks/              # EDA, cleaning, feature eng, training, evaluation scripts
├── src/                    # Core modules (preprocess, features, train, predict, utils)
├── models/                 # Saved trained models
├── api/                    # FastAPI app + Dockerfile
├── dashboard/              # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Setup
```bash
source Cloud_ML/bin/activate
pip install -r requirements.txt
```

