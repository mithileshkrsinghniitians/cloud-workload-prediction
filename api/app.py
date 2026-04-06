"""
Cloud Workload Prediction — FastAPI REST API
============================================
Exposes the trained XGBoost model as a prediction service.

Endpoints:
  GET  /          → welcome message
  GET  /health    → health check
  GET  /models    → list available models and their metrics
  POST /predict   → predict CPU usage 30 min ahead (single sample)
  POST /predict/batch → predict for multiple samples at once
"""

import os
import json
import time
import numpy as np
import pandas as pd
import joblib
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Paths:
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(BASE_DIR, "models")
REPORTS_DIR  = os.path.join(BASE_DIR, "reports")
MODEL_PATH   = os.path.join(MODELS_DIR, "xgboost_model.pkl")
METRICS_PATH = os.path.join(REPORTS_DIR, "xgboost_metrics.json")

# Feature names (must match exactly what XGBoost was trained on):
FEATURE_COLS_RAW = [
    "CPU usage [MHZ]", "Memory usage [KB]",
    "Disk read throughput [KB/s]", "Disk write throughput [KB/s]",
    "Network received throughput [KB/s]", "Network transmitted throughput [KB/s]",
    "Memory usage [%]", "hour", "day_of_week", "is_weekend",
    "day_of_month", "week",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "cpu_util_ratio",
    "CPU usage [%]_lag1", "CPU usage [%]_lag2", "CPU usage [%]_lag3",
    "CPU usage [%]_lag4", "CPU usage [%]_lag5", "CPU usage [%]_lag6",
    "CPU usage [%]_lag9", "CPU usage [%]_lag12",
    "CPU usage [%]_roll_mean_15min", "CPU usage [%]_roll_std_15min", "CPU usage [%]_roll_max_15min",
    "CPU usage [%]_roll_mean_30min", "CPU usage [%]_roll_std_30min", "CPU usage [%]_roll_max_30min",
    "CPU usage [%]_roll_mean_60min", "CPU usage [%]_roll_std_60min", "CPU usage [%]_roll_max_60min",
    "cpu_diff1", "cpu_diff2",
    "mem_pressure", "net_total_throughput", "disk_total_throughput",
]

def sanitize_col(name: str) -> str:
    return name.replace("[", "(").replace("]", ")").replace("<", "_")

SAFE_FEATURE_COLS = [sanitize_col(c) for c in FEATURE_COLS_RAW]

# Load Model on Startup:
print("[API] Loading XGBoost model...")
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Run src/06_train_xgboost.py first.")

model = joblib.load(MODEL_PATH)

# Load metrics if available:
xgb_metrics = {}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH) as f:
        xgb_metrics = json.load(f)

print(f"[API] Model loaded. MAE={xgb_metrics.get('MAE', 'N/A')}%  "
      f"R²={xgb_metrics.get('R2', 'N/A')}")

# FastAPI App:
app = FastAPI(
    title="Cloud Workload Prediction API",
    description=(
        "Predicts VM CPU usage 30 minutes ahead using a trained XGBoost model. "
        "Dataset: Bitbrains fastStorage traces (Bitbrains, 2013)."
    ),
    version="1.0.0",
)


# Pydantic Schemas:
class PredictionInput(BaseModel):
    """
    Feature vector for a single prediction.
    All CPU values are in %, memory in KB, throughput in KB/s.
    Time features (hour, day_of_week, etc.) can be computed from the timestamp.
    """
    # Raw resource metrics:
    cpu_usage_mhz: float          = Field(..., ge=0,    description="Current CPU usage in MHz")
    memory_usage_kb: float        = Field(..., ge=0,    description="Memory usage in KB")
    disk_read_kbps: float         = Field(..., ge=0,    description="Disk read throughput KB/s")
    disk_write_kbps: float        = Field(..., ge=0,    description="Disk write throughput KB/s")
    net_recv_kbps: float          = Field(..., ge=0,    description="Network received KB/s")
    net_send_kbps: float          = Field(..., ge=0,    description="Network transmitted KB/s")
    memory_usage_pct: float       = Field(..., ge=0, le=100, description="Memory usage %")

    # Time features (compute from datetime before calling API):
    hour: int                     = Field(..., ge=0, le=23,  description="Hour of day (0-23)")
    day_of_week: int              = Field(..., ge=0, le=6,   description="Day of week (0=Mon, 6=Sun)")
    is_weekend: int               = Field(..., ge=0, le=1,   description="1 if Saturday/Sunday, else 0")
    day_of_month: int             = Field(..., ge=1, le=31,  description="Day of month (1-31)")
    week: int                     = Field(..., ge=1, le=53,  description="ISO week number (1-53)")

    # Cyclical time encodings (sin/cos of hour and day_of_week):
    hour_sin: float               = Field(..., description="sin(2π × hour / 24)")
    hour_cos: float               = Field(..., description="cos(2π × hour / 24)")
    dow_sin: float                = Field(..., description="sin(2π × day_of_week / 7)")
    dow_cos: float                = Field(..., description="cos(2π × day_of_week / 7)")

    # Derived resource features:
    cpu_util_ratio: float         = Field(..., ge=0, le=1,   description="CPU usage MHz / provisioned MHz")

    # Lag features — CPU usage [%] at previous timesteps:
    cpu_lag1: float               = Field(..., ge=0, le=100, description="CPU % at t-1  (5 min ago)")
    cpu_lag2: float               = Field(..., ge=0, le=100, description="CPU % at t-2  (10 min ago)")
    cpu_lag3: float               = Field(..., ge=0, le=100, description="CPU % at t-3  (15 min ago)")
    cpu_lag4: float               = Field(..., ge=0, le=100, description="CPU % at t-4  (20 min ago)")
    cpu_lag5: float               = Field(..., ge=0, le=100, description="CPU % at t-5  (25 min ago)")
    cpu_lag6: float               = Field(..., ge=0, le=100, description="CPU % at t-6  (30 min ago)")
    cpu_lag9: float               = Field(..., ge=0, le=100, description="CPU % at t-9  (45 min ago)")
    cpu_lag12: float              = Field(..., ge=0, le=100, description="CPU % at t-12 (60 min ago)")

    # Rolling statistics — 15-minute window:
    roll_mean_15min: float        = Field(..., ge=0, le=100)
    roll_std_15min: float         = Field(..., ge=0)
    roll_max_15min: float         = Field(..., ge=0, le=100)

    # Rolling statistics — 30-minute window:
    roll_mean_30min: float        = Field(..., ge=0, le=100)
    roll_std_30min: float         = Field(..., ge=0)
    roll_max_30min: float         = Field(..., ge=0, le=100)

    # Rolling statistics — 60-minute window:
    roll_mean_60min: float        = Field(..., ge=0, le=100)
    roll_std_60min: float         = Field(..., ge=0)
    roll_max_60min: float         = Field(..., ge=0, le=100)

    # Momentum features:
    cpu_diff1: float              = Field(..., description="CPU change from t-1 to t (5-min delta)")
    cpu_diff2: float              = Field(..., description="CPU change from t-2 to t (10-min delta)")

    # Aggregate resource features:
    mem_pressure: float           = Field(..., ge=0, le=1, description="Memory usage % / 100")
    net_total_throughput: float   = Field(..., ge=0, description="Total network KB/s (recv + send)")
    disk_total_throughput: float  = Field(..., ge=0, description="Total disk KB/s (read + write)")


class PredictionOutput(BaseModel):
    predicted_cpu_pct: float
    forecast_horizon_minutes: int
    model_used: str
    inference_time_ms: float
    confidence_note: str


class BatchPredictionInput(BaseModel):
    samples: List[PredictionInput] = Field(..., min_length=1, max_length=10000)


class BatchPredictionOutput(BaseModel):
    predictions: List[float]
    n_samples: int
    forecast_horizon_minutes: int
    model_used: str
    total_inference_time_ms: float
    throughput_per_sec: float


# Helper: Convert input to model DataFrame:
def input_to_dataframe(sample: PredictionInput) -> pd.DataFrame:
    raw_values = {
        "CPU usage (MHZ)":                     sample.cpu_usage_mhz,
        "Memory usage (KB)":                    sample.memory_usage_kb,
        "Disk read throughput (KB/s)":          sample.disk_read_kbps,
        "Disk write throughput (KB/s)":         sample.disk_write_kbps,
        "Network received throughput (KB/s)":   sample.net_recv_kbps,
        "Network transmitted throughput (KB/s)":sample.net_send_kbps,
        "Memory usage (%)":                     sample.memory_usage_pct,
        "hour":                                 sample.hour,
        "day_of_week":                          sample.day_of_week,
        "is_weekend":                           sample.is_weekend,
        "day_of_month":                         sample.day_of_month,
        "week":                                 sample.week,
        "hour_sin":                             sample.hour_sin,
        "hour_cos":                             sample.hour_cos,
        "dow_sin":                              sample.dow_sin,
        "dow_cos":                              sample.dow_cos,
        "cpu_util_ratio":                       sample.cpu_util_ratio,
        "CPU usage (%)_lag1":                   sample.cpu_lag1,
        "CPU usage (%)_lag2":                   sample.cpu_lag2,
        "CPU usage (%)_lag3":                   sample.cpu_lag3,
        "CPU usage (%)_lag4":                   sample.cpu_lag4,
        "CPU usage (%)_lag5":                   sample.cpu_lag5,
        "CPU usage (%)_lag6":                   sample.cpu_lag6,
        "CPU usage (%)_lag9":                   sample.cpu_lag9,
        "CPU usage (%)_lag12":                  sample.cpu_lag12,
        "CPU usage (%)_roll_mean_15min":        sample.roll_mean_15min,
        "CPU usage (%)_roll_std_15min":         sample.roll_std_15min,
        "CPU usage (%)_roll_max_15min":         sample.roll_max_15min,
        "CPU usage (%)_roll_mean_30min":        sample.roll_mean_30min,
        "CPU usage (%)_roll_std_30min":         sample.roll_std_30min,
        "CPU usage (%)_roll_max_30min":         sample.roll_max_30min,
        "CPU usage (%)_roll_mean_60min":        sample.roll_mean_60min,
        "CPU usage (%)_roll_std_60min":         sample.roll_std_60min,
        "CPU usage (%)_roll_max_60min":         sample.roll_max_60min,
        "cpu_diff1":                            sample.cpu_diff1,
        "cpu_diff2":                            sample.cpu_diff2,
        "mem_pressure":                         sample.mem_pressure,
        "net_total_throughput":                 sample.net_total_throughput,
        "disk_total_throughput":                sample.disk_total_throughput,
    }
    return pd.DataFrame([raw_values])


# Routes:
@app.get("/")
def root():
    return {
        "project": "Cloud Workload Prediction",
        "description": "Predicts VM CPU usage 30 minutes ahead using XGBoost.",
        "dataset": "Bitbrains fastStorage (2013)",
        "endpoints": ["/health", "/models", "/predict", "/predict/batch"],
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    #Returns 200 if the API is running and the model is loaded.
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": "XGBoost",
        "n_features": len(SAFE_FEATURE_COLS),
    }


@app.get("/models")
def list_models():
    # Load all metric files if available:
    metrics = {"XGBoost": xgb_metrics}

    lstm_path = os.path.join(REPORTS_DIR, "lstm_metrics.json")
    if os.path.exists(lstm_path):
        with open(lstm_path) as f:
            metrics["LSTM"] = json.load(f)

    prophet_path = os.path.join(REPORTS_DIR, "prophet_metrics.json")
    if os.path.exists(prophet_path):
        with open(prophet_path) as f:
            metrics["Prophet"] = json.load(f)

    return {
        "available_models": list(metrics.keys()),
        "active_model": "XGBoost",
        "forecast_horizon_minutes": 30,
        "metrics": metrics,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    # Predict CPU usage 30 minutes ahead for a single VM data point.
    # Provide the current engineered feature vector (lags, rolling stats, time features).
    # Returns predicted CPU % clamped to [0, 100].
    try:
        X = input_to_dataframe(data)
        t0 = time.perf_counter()
        y_pred = float(np.clip(model.predict(X)[0], 0, 100))
        inference_ms = (time.perf_counter() - t0) * 1000

        return PredictionOutput(
            predicted_cpu_pct=round(y_pred, 2),
            forecast_horizon_minutes=30,
            model_used="XGBoost",
            inference_time_ms=round(inference_ms, 3),
            confidence_note=(
                f"Model test-set MAE = {xgb_metrics.get('MAE', 'N/A')}%. "
                "Prediction accuracy may vary for workload patterns not seen during training."
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionOutput)
def predict_batch(data: BatchPredictionInput):
    # Predict CPU usage 30 minutes ahead for multiple VM data points in one call.
    # More efficient than calling /predict repeatedly for large batches.
    # Accepts up to 10,000 samples per request.
    try:
        frames = [input_to_dataframe(s) for s in data.samples]
        X = pd.concat(frames, ignore_index=True)

        t0 = time.perf_counter()
        y_pred = np.clip(model.predict(X), 0, 100).tolist()
        total_ms = (time.perf_counter() - t0) * 1000

        n = len(y_pred)
        return BatchPredictionOutput(
            predictions=[round(p, 2) for p in y_pred],
            n_samples=n,
            forecast_horizon_minutes=30,
            model_used="XGBoost",
            total_inference_time_ms=round(total_ms, 3),
            throughput_per_sec=round(n / (total_ms / 1000), 1),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run (development only):
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
