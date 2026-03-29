import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Reproducibility ────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH   = os.path.join(PROJECT_ROOT, "data", "processed", "combined_features.csv")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR  = os.path.join(PROJECT_ROOT, "reports", "figures")
METRICS_PATH = os.path.join(PROJECT_ROOT, "reports", "lstm_metrics.json")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
SEQ_LEN          = 12          # 12 × 5 min = 60 min lookback
FORECAST_HORIZON = 6           # predict 30 min ahead
BATCH_SIZE       = 256
EPOCHS           = 50
LR               = 1e-3
HIDDEN_SIZE      = 64
NUM_LAYERS       = 2
DROPOUT          = 0.2

FEATURE_COLS = [
    "CPU usage [%]", "CPU usage [MHZ]", "Memory usage [%]",
    "Disk read throughput [KB/s]", "Disk write throughput [KB/s]",
    "Network received throughput [KB/s]", "Network transmitted throughput [KB/s]",
    "cpu_util_ratio", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend", "mem_pressure", "net_total_throughput", "disk_total_throughput",
]
TARGET_COL = "CPU usage [%]"

# Force CPU to ensure train/eval numerical consistency.
# MPS (Apple Silicon) can give different floating-point results on inference.
DEVICE = torch.device("cpu")

print("=" * 60)
print("PHASE 6b — LSTM Model Training (PyTorch)")
print("=" * 60)
print(f"\n  Device : {DEVICE}")
print(f"  Seq len: {SEQ_LEN} steps ({SEQ_LEN*5} min lookback)")
print(f"  Horizon: {FORECAST_HORIZON} steps (30 min ahead)")

# ── 1. Load & Split (same regime split as XGBoost) ────────────────────────────
df = pd.read_csv(INPUT_PATH, parse_dates=["datetime"])
df = df.sort_values(["vm_id", "datetime"]).reset_index(drop=True)

busy_end   = pd.Timestamp("2013-09-02")
test_start = pd.Timestamp("2013-08-26")
busy_df    = df[df["datetime"] < busy_end].copy()
train_df   = busy_df[busy_df["datetime"] <  test_start]
test_df    = busy_df[busy_df["datetime"] >= test_start]

print(f"\n[1] Train: {len(train_df):,} rows  |  Test: {len(test_df):,} rows")

# ── 2. Scale Features ─────────────────────────────────────────────────────────
scaler = MinMaxScaler()
train_reset = train_df.reset_index(drop=True)
test_reset  = test_df.reset_index(drop=True)

train_scaled = train_reset[FEATURE_COLS].copy()
test_scaled  = test_reset[FEATURE_COLS].copy()

train_scaled[FEATURE_COLS] = scaler.fit_transform(train_reset[FEATURE_COLS])
test_scaled[FEATURE_COLS]  = scaler.transform(test_reset[FEATURE_COLS])

# Save scaler for inference
joblib.dump(scaler, os.path.join(MODELS_DIR, "lstm_scaler.pkl"))
print("[2] Features scaled (MinMaxScaler). Scaler saved.")

# ── 3. Build Sequences ────────────────────────────────────────────────────────
def build_sequences(df_scaled, df_orig, vm_ids):
    """Build (X, y) sequence pairs per VM to avoid cross-VM contamination."""
    X_list, y_list = [], []
    target_idx = FEATURE_COLS.index(TARGET_COL)

    for vm_id in vm_ids:
        vm_mask  = df_orig["vm_id"] == vm_id
        features = df_scaled[FEATURE_COLS][vm_mask].values
        targets  = df_orig[TARGET_COL][vm_mask].values

        for i in range(SEQ_LEN, len(features) - FORECAST_HORIZON + 1):
            X_list.append(features[i - SEQ_LEN : i])           # (SEQ_LEN, n_feat)
            y_list.append(targets[i + FORECAST_HORIZON - 1])   # scalar

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


vm_ids = df["vm_id"].unique()
X_train, y_train = build_sequences(train_scaled, train_reset, vm_ids)
X_test,  y_test  = build_sequences(test_scaled,  test_reset,  vm_ids)

print(f"[3] Sequences built:")
print(f"    X_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"    X_test : {X_test.shape}   y_test : {y_test.shape}")

# ── 4. Dataset & DataLoader ───────────────────────────────────────────────────
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_loader = DataLoader(TimeSeriesDataset(X_train, y_train),
                          batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(TimeSeriesDataset(X_test, y_test),
                          batch_size=BATCH_SIZE, shuffle=False)

# ── 5. LSTM Model ─────────────────────────────────────────────────────────────
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])   # last timestep → prediction


model = LSTMForecaster(
    input_size  = len(FEATURE_COLS),
    hidden_size = HIDDEN_SIZE,
    num_layers  = NUM_LAYERS,
    dropout     = DROPOUT,
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n[4] Model: {total_params:,} trainable parameters")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
criterion = nn.MSELoss()

# ── 6. Training Loop ──────────────────────────────────────────────────────────
print(f"\n[5] Training for up to {EPOCHS} epochs (early stop patience=8)...")
print(f"    {'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}  {'LR':>10}")
print("    " + "-" * 45)

best_val_loss = float("inf")
patience_counter = 0
PATIENCE = 8
train_losses, val_losses = [], []

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * len(xb)
    train_loss = running_loss / len(train_loader.dataset)

    # Validate
    model.eval()
    val_loss_sum = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            val_loss_sum += criterion(pred, yb).item() * len(xb)
    val_loss = val_loss_sum / len(test_loader.dataset)

    scheduler.step(val_loss)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]
    if epoch % 5 == 0 or epoch == 1:
        print(f"    {epoch:>6}  {train_loss:>12.4f}  {val_loss:>10.4f}  {current_lr:>10.2e}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, "lstm_best.pt"))
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n    Early stopping at epoch {epoch} (best val loss: {best_val_loss:.4f})")
            break

# ── 7. Evaluate with Best Weights ─────────────────────────────────────────────
model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "lstm_best.pt"),
                                 map_location=DEVICE))
model.eval()

all_preds, all_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        pred = model(xb).cpu().numpy().flatten()
        all_preds.extend(pred)
        all_true.extend(yb.numpy().flatten())

y_pred = np.clip(np.array(all_preds), 0, 100)
y_true = np.array(all_true)

mae   = mean_absolute_error(y_true, y_pred)
rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
r2    = r2_score(y_true, y_pred)
smape = np.mean(2 * np.abs(y_true - y_pred) /
                (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100

print(f"\n[6] Test Set Evaluation:")
print(f"    MAE   : {mae:.4f} %")
print(f"    RMSE  : {rmse:.4f} %")
print(f"    R²    : {r2:.4f}")
print(f"    sMAPE : {smape:.2f}%")

# ── 8. Loss Curve ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(train_losses, label="Train Loss (MSE)", color="steelblue")
ax.plot(val_losses,   label="Val Loss (MSE)",   color="tomato")
ax.set_title("LSTM Training & Validation Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "09_lstm_loss_curve.png"))
plt.close()
print("[7] Saved: 09_lstm_loss_curve.png")

# ── 9. Actual vs Predicted ────────────────────────────────────────────────────
plot_n = min(500, len(y_true))
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(range(plot_n), y_true[:plot_n], label="Actual",         color="steelblue",  linewidth=1)
ax.plot(range(plot_n), y_pred[:plot_n], label="LSTM Predicted", color="darkorange",  linewidth=1, linestyle="--")
ax.set_title("LSTM — Actual vs Predicted CPU [%] (first 500 test samples)")
ax.set_xlabel("Sample")
ax.set_ylabel("CPU Usage [%]")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "10_lstm_actual_vs_predicted.png"))
plt.close()
print("[8] Saved: 10_lstm_actual_vs_predicted.png")

# ── 10. Save Metrics ──────────────────────────────────────────────────────────
metrics = {"MAE": round(float(mae), 4), "RMSE": round(float(rmse), 4),
           "R2": round(float(r2), 4), "sMAPE": round(float(smape), 2)}
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)
print("[9] Metrics saved → reports/lstm_metrics.json")

# Save predictions for use in evaluation script (avoids MPS/CPU divergence)
preds_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
preds_path = os.path.join(MODELS_DIR, "lstm_test_predictions.csv")
preds_df.to_csv(preds_path, index=False)
print("[10] Predictions saved → models/lstm_test_predictions.csv")

print("\n" + "=" * 60)
print("LSTM Training Complete")
print(f"  MAE={mae:.2f}%  RMSE={rmse:.2f}%  R²={r2:.4f}  sMAPE={smape:.1f}%")
print(f"  Model saved → models/lstm_best.pt")
print("=" * 60)
