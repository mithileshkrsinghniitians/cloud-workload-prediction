import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import xgboost as xgb
import joblib

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Paths:
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH   = os.path.join(PROJECT_ROOT, "data", "processed", "combined_features.csv")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR  = os.path.join(PROJECT_ROOT, "reports", "figures")
REPORTS_DIR  = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(FIGURES_DIR, exist_ok=True)

FORECAST_HORIZON = 6   # 30 min ahead (XGBoost prediction horizon)
STEP_MINUTES     = 5   # each row = 5 minutes

# Auto-scaler Configuration:
SCALE_UP_THRESHOLD   = 70.0   # CPU % above which we scale up
SCALE_DOWN_THRESHOLD = 30.0   # CPU % below which we scale down
MIN_INSTANCES        = 1      # minimum instances in the cluster
MAX_INSTANCES        = 4      # maximum instances in the cluster
COOLDOWN_STEPS       = 3      # 3 × 5 min = 15 min cooldown between scaling decisions
SLA_VIOLATION_THRESH = 85.0   # CPU % above which an SLA violation occurs

XGB_FEATURE_COLS = [
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

def sanitize_col(name):
    return name.replace("[", "(").replace("]", ")").replace("<", "_")


# Auto-Scaler Simulation:
def simulate_autoscaler(cpu_actual, cpu_signal, label, scale_up_thresh, scale_down_thresh):
    """
    Simulate a simple threshold-based auto-scaler.

    Args:
        cpu_actual : array-like — actual CPU usage at each timestep
        cpu_signal : array-like — the signal used for scaling decisions
                     (= actual CPU for reactive, = predicted CPU for predictive)
        label      : str — name of this strategy
        scale_up_thresh   : float — scale up when signal > threshold
        scale_down_thresh : float — scale down when signal < threshold

    Returns:
        dict with simulation results and time-series arrays
    """
    n = len(cpu_actual)
    instances     = np.ones(n, dtype=int)    # instance count at each step
    scale_events  = []                       # list of (step, action, reason)
    sla_violations = []                      # steps where actual CPU > SLA threshold

    current_instances = MIN_INSTANCES
    cooldown_remaining = 0

    for t in range(n):
        # track instances:
        instances[t] = current_instances

        # SLA check: actual CPU / instances should stay < SLA threshold
        # effective CPU per instance = actual_cpu / current_instances (simplified)
        effective_cpu = cpu_actual[t] / current_instances
        if effective_cpu > SLA_VIOLATION_THRESH:
            sla_violations.append(t)

        # cooldown check:
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue

        # scaling decision based on the chosen signal:
        signal = cpu_signal[t]

        if signal > scale_up_thresh and current_instances < MAX_INSTANCES:
            current_instances += 1
            scale_events.append({
                "step": t, "action": "scale_up",
                "instances_after": current_instances,
                "signal_cpu": round(float(signal), 1),
                "actual_cpu": round(float(cpu_actual[t]), 1),
            })
            cooldown_remaining = COOLDOWN_STEPS

        elif signal < scale_down_thresh and current_instances > MIN_INSTANCES:
            current_instances -= 1
            scale_events.append({
                "step": t, "action": "scale_down",
                "instances_after": current_instances,
                "signal_cpu": round(float(signal), 1),
                "actual_cpu": round(float(cpu_actual[t]), 1),
            })
            cooldown_remaining = COOLDOWN_STEPS

    n_scale_up   = sum(1 for e in scale_events if e["action"] == "scale_up")
    n_scale_down = sum(1 for e in scale_events if e["action"] == "scale_down")

    return {
        "label":          label,
        "instances":      instances,
        "scale_events":   scale_events,
        "sla_violations": sla_violations,
        "n_scale_up":     n_scale_up,
        "n_scale_down":   n_scale_down,
        "n_total_events": len(scale_events),
        "n_sla_violations": len(sla_violations),
        "sla_violation_rate_pct": round(len(sla_violations) / n * 100, 2),
        "mean_instances": round(float(instances.mean()), 3),
        "total_instance_minutes": int(instances.sum() * STEP_MINUTES),
    }


print("=" * 60)
print("PHASE 9 — Auto-Scaling Simulation")
print("=" * 60)
print(f"\n  Scale Up  threshold : CPU > {SCALE_UP_THRESHOLD}%")
print(f"  Scale Down threshold: CPU < {SCALE_DOWN_THRESHOLD}%")
print(f"  Instance range      : {MIN_INSTANCES}–{MAX_INSTANCES}")
print(f"  Cooldown            : {COOLDOWN_STEPS} steps ({COOLDOWN_STEPS*STEP_MINUTES} min)")
print(f"  SLA violation       : effective CPU > {SLA_VIOLATION_THRESH}%")

# 1. Load Data & Build Test Set:
df = pd.read_csv(INPUT_PATH, parse_dates=["datetime"])
df = df.sort_values(["vm_id", "datetime"]).reset_index(drop=True)

df["target"] = (
    df.groupby("vm_id")["CPU usage [%]"]
    .shift(-FORECAST_HORIZON)
)
df = df.dropna(subset=["target"]).reset_index(drop=True)

busy_end   = pd.Timestamp("2013-09-02")
test_start = pd.Timestamp("2013-08-26")
busy_df    = df[df["datetime"] < busy_end].reset_index(drop=True)
test_df    = busy_df[busy_df["datetime"] >= test_start].reset_index(drop=True)

print(f"\n[1] Test set: {len(test_df):,} rows  ({test_start.date()} → {busy_end.date()})")

# 2. XGBoost Predictions:
xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))

safe_map  = {c: sanitize_col(c) for c in XGB_FEATURE_COLS if c in test_df.columns}
avail_cols = [c for c in XGB_FEATURE_COLS if c in test_df.columns]
X_test    = test_df[avail_cols].rename(columns={c: sanitize_col(c) for c in avail_cols})

# Align with model's expected features:
model_features = xgb_model.get_booster().feature_names
X_test = X_test.reindex(columns=model_features, fill_value=0)

xgb_pred = np.clip(xgb_model.predict(X_test), 0, 100)
cpu_actual = test_df["CPU usage [%]"].values

print(f"[2] XGBoost predictions generated: {len(xgb_pred):,} samples")

# 3. Pick One VM for Detailed Simulation (most dynamic = highest CPU std):
vm_ids = test_df["vm_id"].unique()
vm_variability = {
    vm: test_df[test_df["vm_id"] == vm]["CPU usage [%]"].std()
    for vm in vm_ids
}
sim_vm = max(vm_variability, key=vm_variability.get)
print(f"[3] Simulating on VM {sim_vm} (highest CPU variability: "
      f"std={vm_variability[sim_vm]:.2f}%)")

vm_mask    = test_df["vm_id"] == sim_vm
vm_actual  = cpu_actual[vm_mask]
vm_pred    = xgb_pred[vm_mask]
vm_times   = test_df[vm_mask]["datetime"].values

print(f"    VM {sim_vm} test samples: {len(vm_actual):,}")

# 4. Run Three Simulations:
print("\n[4] Running simulations...")

# Strategy 1: No scaling (fixed 1 instance — worst case baseline):
no_scale_result = simulate_autoscaler(
    cpu_actual=vm_actual,
    cpu_signal=np.zeros(len(vm_actual)),
    label="No Scaling (1 instance)",
    scale_up_thresh=200,
    scale_down_thresh=-1,
)

# Strategy 2: Reactive — decide based on current observed CPU:
reactive_result = simulate_autoscaler(
    cpu_actual=vm_actual,
    cpu_signal=vm_actual,
    label="Reactive (current CPU)",
    scale_up_thresh=SCALE_UP_THRESHOLD,
    scale_down_thresh=SCALE_DOWN_THRESHOLD,
)

# Strategy 3: Predictive — decide based on XGBoost 30-min forecast:
predictive_result = simulate_autoscaler(
    cpu_actual=vm_actual,
    cpu_signal=vm_pred,
    label="Predictive (XGBoost forecast)",
    scale_up_thresh=SCALE_UP_THRESHOLD,
    scale_down_thresh=SCALE_DOWN_THRESHOLD,
)

# 5. Print Comparison:
print(f"\n[5] Simulation Results for VM {sim_vm}:")
print(f"\n  {'Strategy':<35}  {'Scale↑':>6}  {'Scale↓':>6}  {'SLA Viol':>9}  "
      f"{'Viol %':>7}  {'Mean Inst':>9}  {'Inst-min':>8}")
print("  " + "-" * 90)

for res in [no_scale_result, reactive_result, predictive_result]:
    print(f"  {res['label']:<35}  {res['n_scale_up']:>6}  {res['n_scale_down']:>6}  "
          f"{res['n_sla_violations']:>9}  {res['sla_violation_rate_pct']:>6.1f}%  "
          f"{res['mean_instances']:>9.2f}  {res['total_instance_minutes']:>8,}")

# 6. Figure 1: Simulation Timeline:
n_plot = min(288, len(vm_actual))   # first 24 hours
t_idx  = range(n_plot)

fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

for ax, res, color in zip(axes,
                          [no_scale_result, reactive_result, predictive_result],
                          ["gray", "steelblue", "seagreen"]):
    # CPU actual:
    ax.fill_between(t_idx, 0, vm_actual[:n_plot], alpha=0.2, color="tomato", label="Actual CPU")
    ax.plot(t_idx, vm_actual[:n_plot], color="tomato", linewidth=0.8, alpha=0.8)

    # XGBoost predicted (only for predictive):
    if "Predictive" in res["label"]:
        ax.plot(t_idx, vm_pred[:n_plot], color="darkorange", linewidth=1,
                linestyle="--", alpha=0.7, label="XGBoost Forecast")

    # Instance count (right y-axis):
    ax2 = ax.twinx()
    ax2.step(t_idx, res["instances"][:n_plot], color=color, linewidth=2,
             where="post", label="Instances")
    ax2.set_ylim(0, MAX_INSTANCES + 1)
    ax2.set_yticks(range(MIN_INSTANCES, MAX_INSTANCES + 1))
    ax2.set_ylabel("Instances", color=color)

    # SLA violations in the plotted range:
    viol_in_range = [v for v in res["sla_violations"] if v < n_plot]
    if viol_in_range:
        ax.scatter(viol_in_range, vm_actual[viol_in_range],
                   color="red", s=20, zorder=5, label="SLA Violation")

    ax.axhline(SCALE_UP_THRESHOLD, color="orange", linestyle=":", linewidth=1,
               alpha=0.7, label=f"Scale-up @ {SCALE_UP_THRESHOLD}%")
    ax.axhline(SLA_VIOLATION_THRESH, color="red", linestyle=":", linewidth=1,
               alpha=0.7, label=f"SLA limit @ {SLA_VIOLATION_THRESH}%")
    ax.set_ylim(-5, 110)
    ax.set_ylabel("CPU Usage [%]")
    ax.set_title(f"{res['label']}  |  SLA violations: {res['n_sla_violations']} "
                 f"({res['sla_violation_rate_pct']:.1f}%)  |  Scaling events: {res['n_total_events']}")
    ax.legend(loc="upper right", fontsize=8)

axes[-1].set_xlabel("Time Step (5-min intervals, first 24 hours of test period)")
fig.suptitle(f"Auto-Scaling Simulation — VM {sim_vm} (Test Period: Aug 26 – Sep 1, 2013)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "28_autoscaling_simulation.png"))
plt.close()
print("\n[6] Saved: 28_autoscaling_simulation.png")

# 7. Figure 2: Comparison Bar Charts:
strategies   = ["No Scaling", "Reactive", "Predictive"]
sla_viols    = [no_scale_result["n_sla_violations"],
                reactive_result["n_sla_violations"],
                predictive_result["n_sla_violations"]]
scale_events = [no_scale_result["n_total_events"],
                reactive_result["n_total_events"],
                predictive_result["n_total_events"]]
mean_inst    = [no_scale_result["mean_instances"],
                reactive_result["mean_instances"],
                predictive_result["mean_instances"]]
colors       = ["gray", "steelblue", "seagreen"]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# SLA violations:
bars = axes[0].bar(strategies, sla_viols, color=colors, edgecolor="white")
axes[0].set_title("SLA Violations\n(lower is better)")
axes[0].set_ylabel("Count")
for bar, val in zip(bars, sla_viols):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")

# Scaling events (cost proxy):
bars = axes[1].bar(strategies, scale_events, color=colors, edgecolor="white")
axes[1].set_title("Total Scaling Events\n(fewer = lower ops cost)")
axes[1].set_ylabel("Count")
for bar, val in zip(bars, scale_events):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")

# Mean instances (resource cost):
bars = axes[2].bar(strategies, mean_inst, color=colors, edgecolor="white")
axes[2].set_title("Mean Instances Running\n(fewer = lower resource cost)")
axes[2].set_ylabel("Instances")
axes[2].set_ylim(0, MAX_INSTANCES + 0.5)
for bar, val in zip(bars, mean_inst):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

fig.suptitle(f"Auto-Scaling Strategy Comparison — VM {sim_vm}", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "29_autoscaling_comparison.png"))
plt.close()
print("[7] Saved: 29_autoscaling_comparison.png")

# 8. Also Run on ALL VMs combined:
print("\n[8] Running simulation on all VMs combined...")

all_reactive   = simulate_autoscaler(cpu_actual, cpu_actual, "Reactive (all VMs)",
                                     SCALE_UP_THRESHOLD, SCALE_DOWN_THRESHOLD)
all_predictive = simulate_autoscaler(cpu_actual, xgb_pred,  "Predictive (all VMs)",
                                     SCALE_UP_THRESHOLD, SCALE_DOWN_THRESHOLD)

print(f"    {'Strategy':<30}  {'SLA Viol':>9}  {'Viol %':>7}  {'Scaling Events':>14}")
print("    " + "-" * 68)
for res in [all_reactive, all_predictive]:
    print(f"    {res['label']:<30}  {res['n_sla_violations']:>9}  "
          f"{res['sla_violation_rate_pct']:>6.1f}%  {res['n_total_events']:>14}")

# 9. Save Report:
report = {
    "config": {
        "scale_up_threshold_pct":   SCALE_UP_THRESHOLD,
        "scale_down_threshold_pct": SCALE_DOWN_THRESHOLD,
        "sla_violation_threshold_pct": SLA_VIOLATION_THRESH,
        "min_instances": MIN_INSTANCES,
        "max_instances": MAX_INSTANCES,
        "cooldown_steps": COOLDOWN_STEPS,
        "cooldown_minutes": COOLDOWN_STEPS * STEP_MINUTES,
        "forecast_horizon_steps": FORECAST_HORIZON,
        "forecast_horizon_minutes": FORECAST_HORIZON * STEP_MINUTES,
    },
    "single_vm_simulation": {
        "vm_id": int(sim_vm),
        "n_timesteps": len(vm_actual),
        "no_scaling": {k: v for k, v in no_scale_result.items()
                       if k not in ("instances", "scale_events", "sla_violations")},
        "reactive": {k: v for k, v in reactive_result.items()
                     if k not in ("instances", "scale_events", "sla_violations")},
        "predictive": {k: v for k, v in predictive_result.items()
                       if k not in ("instances", "scale_events", "sla_violations")},
    },
    "all_vms_simulation": {
        "n_timesteps": len(cpu_actual),
        "reactive": {k: v for k, v in all_reactive.items()
                     if k not in ("instances", "scale_events", "sla_violations")},
        "predictive": {k: v for k, v in all_predictive.items()
                       if k not in ("instances", "scale_events", "sla_violations")},
    },
    "key_finding": (
        f"Predictive scaling reduced SLA violations by "
        f"{reactive_result['n_sla_violations'] - predictive_result['n_sla_violations']} "
        f"({reactive_result['sla_violation_rate_pct'] - predictive_result['sla_violation_rate_pct']:.1f}% points) "
        f"compared to reactive scaling on VM {int(sim_vm)}."
    ),
}
report_path = os.path.join(REPORTS_DIR, "autoscaling_simulation.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print("[9] Saved: reports/autoscaling_simulation.json")

# 10. Final Summary:
sla_reduction = (reactive_result["n_sla_violations"] - predictive_result["n_sla_violations"])
sla_reduction_pct = (reactive_result["sla_violation_rate_pct"] -
                     predictive_result["sla_violation_rate_pct"])

print("\n" + "=" * 60)
print("Auto-Scaling Simulation Complete")
print("=" * 60)
print(f"\n  VM {int(sim_vm)} Results:")
print(f"  {'Strategy':<35}  SLA Violations  Scaling Events")
print("  " + "-" * 65)
for res in [no_scale_result, reactive_result, predictive_result]:
    print(f"  {res['label']:<35}  "
          f"{res['n_sla_violations']:>10} ({res['sla_violation_rate_pct']:.1f}%)  "
          f"{res['n_total_events']:>14}")
print(f"\n  Predictive vs Reactive: {sla_reduction} fewer SLA violations "
      f"({sla_reduction_pct:.1f}% reduction)")
print("=" * 60)
