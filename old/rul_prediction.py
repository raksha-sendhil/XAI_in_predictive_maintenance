"""
rul_prediction.py
------------------
RUL engine matched to the MATLAB data generation model.

MATLAB generates severity as:
    norm_t_eff(day) ≈ (day / L) ^ alpha      [power-law with per-step noise]
    severity(day)   = S_min + (S_max - S_min) * norm_t_eff(day)

Key problem with hardcoding S_min/S_max:
    The MATLAB severity for LeakFault is actually:
        leak_area_set_factor * TRP_Par.Check_Valve.In.Max_Area
    The 0.036 threshold is in FACTOR space, but the CSV stores AREA values
    (factor * Max_Area), so the actual S_max in the CSV is unknown without
    knowing Max_Area.

Solution — use the FULL lifecycle CSV to determine S_max empirically:
    S_max = severity on the LAST day of the CSV (that is the failure day,
            where MATLAB explicitly pins severity to its maximum value).
    S_min = severity on the FIRST day of the CSV.

Then norm_t_eff = (S - S_min) / (S_max - S_min)  runs cleanly from 0 → 1,
and we fit norm_t_eff = (day / L) ^ alpha as intended.

This is self-consistent with the MATLAB logic which pins:
    if day == totalDays: severity = S_max   (exactly at threshold)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
import pandas as pd

# =============================================================================
# USER INPUTS  ← edit this section
# =============================================================================

INPUT_CSV            = "rul_input.csv"   # produced by fault_prediction.py

# How many days of data to use for the curve fit.
# Set to None to use ALL rows in the CSV.
HOW_MANY_DAYS_TO_USE = 20

# Alpha bounds — must bracket the MATLAB generation range [1.5, 3.0]
ALPHA_LB = 1.0
ALPHA_UB = 4.0

# Maximum lifecycle the solver is allowed to predict (days)
L_MAX = 500

# =============================================================================
# FAULT CLASS METADATA  (direction only — S_min/S_max derived from data)
# =============================================================================

FAULT_CONFIG = {
    1: dict(name='LeakFault',     decreasing=False),
    2: dict(name='BlockingFault', decreasing=True),
    3: dict(name='BearingFault',  decreasing=False),
}

# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv(INPUT_CSV)
df = df.sort_values('current_day').reset_index(drop=True)

fault_class = int(df['fault_class'].iloc[0])
cfg         = FAULT_CONFIG[fault_class]
fault_name  = cfg['name']
decreasing  = cfg['decreasing']

# =============================================================================
# DERIVE S_min AND S_max FROM THE FULL CSV
#
# MATLAB pins the last day's severity exactly to the failure threshold value.
# So S_max (or S_min for blockage) = severity on the last row of the CSV.
# S_min (or S_max for blockage) = severity on the first row of the CSV.
#
# This works regardless of unit scaling — we read directly from the data.
# =============================================================================

S_first = float(df['true_severity'].iloc[0])   # day 1 value
S_last  = float(df['true_severity'].iloc[-1])  # last day = failure threshold value

if decreasing:
    # Blockage: starts HIGH (S_first), degrades DOWN to S_last
    S_max = S_first
    S_min = S_last   # S_last is the failure threshold (minimum)
else:
    # Leak / Bearing: starts LOW (S_first), grows UP to S_last
    S_min = S_first
    S_max = S_last   # S_last is the failure threshold (maximum)

S_threshold = S_last   # failure day severity = threshold by MATLAB construction
denom       = S_max - S_min

print(f"Fault class      : {fault_class} ({fault_name})")
print(f"Degradation      : {'DECREASING' if decreasing else 'INCREASING'}")
print(f"S_min (day 1)    : {S_min:.6e}  [derived from CSV first row]")
print(f"S_max (last day) : {S_max:.6e}  [derived from CSV last row = failure]")
print(f"Failure threshold: {S_threshold:.6e}")
print(f"Total rows in CSV: {len(df)}")

if denom <= 0:
    raise ValueError(
        f"S_max ({S_max:.4e}) <= S_min ({S_min:.4e}) — "
        "check that the CSV has a proper degradation trend."
    )

# =============================================================================
# SLICE DATA TO HOW_MANY_DAYS_TO_USE
# =============================================================================

if HOW_MANY_DAYS_TO_USE is None:
    HOW_MANY_DAYS_TO_USE = len(df)

HOW_MANY_DAYS_TO_USE = min(HOW_MANY_DAYS_TO_USE, len(df))

history_df    = df.iloc[:HOW_MANY_DAYS_TO_USE].copy()
current_day   = int(history_df['current_day'].iloc[-1])
days_hist     = history_df['current_day'].values.astype(float)
severity_hist = history_df['true_severity'].values.astype(float)

print(f"\nDays used for fit  : {HOW_MANY_DAYS_TO_USE}  (current day = {current_day})")
print(f"Severity on day 1  : {severity_hist[0]:.6e}")
print(f"Severity on day {current_day:>3} : {severity_hist[-1]:.6e}")

# =============================================================================
# STEP 1 — SMOOTH SEVERITY
# =============================================================================

smoothed_sev = uniform_filter1d(severity_hist, size=3, mode='nearest')

# =============================================================================
# STEP 2 — NORMALISE → recover norm_t_eff in [0, 1]
#
# Increasing: norm_t_eff = (S - S_min) / (S_max - S_min)
# Decreasing: norm_t_eff = (S_max - S) / (S_max - S_min)
#
# norm_t_eff should go from ~0 on day 1 to 1.0 on the last lifecycle day.
# We fit only on the first HOW_MANY_DAYS_TO_USE points (0 → partial).
# =============================================================================

if decreasing:
    norm_t_eff = (S_max - smoothed_sev) / denom
else:
    norm_t_eff = (smoothed_sev - S_min) / denom

norm_t_eff = np.clip(norm_t_eff, 1e-12, 1.0 - 1e-12)

print(f"\nnorm_t_eff on day 1        : {norm_t_eff[0]:.6f}")
print(f"norm_t_eff on day {current_day:>3}       : {norm_t_eff[-1]:.6f}")
print(f"  (should reach 1.0 by day {len(df)})")

if norm_t_eff[-1] < 1e-4:
    print(
        "\n⚠  WARNING: norm_t_eff is still near zero on the current day.\n"
        "   The curve fitter has very little signal — try a later cutoff day\n"
        "   (increase HOW_MANY_DAYS_TO_USE) for a reliable RUL estimate."
    )

# =============================================================================
# STEP 3 — FIT POWER LAW IN NORMALISED TIME SPACE
#
#   norm_t_eff(day) = (day / L) ^ alpha
#
# L  = total lifecycle (days)  ← what we solve for
# alpha = shape parameter, bounded [ALPHA_LB, ALPHA_UB]
# =============================================================================

def power_law(d, L, alpha):
    return np.clip(d / L, 0.0, 1.0) ** alpha

p0 = [current_day * 1.5, (ALPHA_LB + ALPHA_UB) / 2.0]
lb = [current_day + 1,   ALPHA_LB]
ub = [L_MAX,             ALPHA_UB]

try:
    best_fit, pcov = curve_fit(
        power_law,
        days_hist,
        norm_t_eff,
        p0=p0,
        bounds=(lb, ub),
        method='trf',
        maxfev=50_000,
    )
    L_pred     = best_fit[0]
    alpha_pred = best_fit[1]
    perr       = np.sqrt(np.diag(pcov))
    L_err      = perr[0]
    alpha_err  = perr[1]
except RuntimeError as e:
    print(f"\nCurve fit failed: {e}")
    print("Try increasing L_MAX, broadening ALPHA bounds, or using more days.")
    raise

RUL_pred      = L_pred - current_day
true_last_day = int(df['current_day'].max())
true_rul      = true_last_day - current_day

print(f"\n--- RUL Prediction Results (Day {current_day}) ---")
print(f"Predicted Lifecycle : {L_pred:.1f} ± {L_err:.1f} days")
print(f"Predicted Alpha     : {alpha_pred:.3f} ± {alpha_err:.3f}")
print(f"Predicted RUL       : {RUL_pred:.1f} days remaining")
print(f"True RUL (from CSV) : {true_rul} days  (lifecycle ends day {true_last_day})")
print(f"RUL Error           : {abs(RUL_pred - true_rul):.1f} days")

# =============================================================================
# STEP 4 — RECONSTRUCT SEVERITY CURVE FROM FITTED PARAMETERS
# =============================================================================

x_max     = int(np.ceil(max(L_pred, true_last_day) + 5))
plot_days = np.arange(1, x_max + 1, dtype=float)
norm_fit  = power_law(plot_days, L_pred, alpha_pred)

if decreasing:
    sev_fit = S_max - denom * norm_fit
else:
    sev_fit = S_min + denom * norm_fit

# Smoothed severity back in severity space (for orange line)
if decreasing:
    smoothed_sev_plot = S_max - denom * norm_t_eff
else:
    smoothed_sev_plot = S_min + denom * norm_t_eff

# =============================================================================
# VISUALIZE
# =============================================================================

fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
ax.grid(True, linestyle='--', alpha=0.5)

# ── 1. Raw true severity history ──────────────────────────────────────────────
ax.scatter(days_hist, severity_hist, s=40, color='red', zorder=5,
           label=f'True severity (days 1–{current_day}, used for fit)')

# ── 2. Smoothed ───────────────────────────────────────────────────────────────
ax.plot(days_hist, smoothed_sev_plot, color='orange', linewidth=1.5,
        linestyle='--', label='Smoothed (3-day avg, normalised, fitted)')

# ── 3a. Fitted curve — past ───────────────────────────────────────────────────
mask_past = plot_days <= current_day
ax.plot(plot_days[mask_past], sev_fit[mask_past],
        color='blue', linewidth=2.0, label='Fitted curve (past)')

# ── 3b. Extrapolated — future ─────────────────────────────────────────────────
mask_future = plot_days >= current_day
ax.plot(plot_days[mask_future], sev_fit[mask_future],
        color='blue', linewidth=2.0, linestyle='--',
        label=f'Extrapolated (RUL = {RUL_pred:.1f} days)')

# ── 4. True full-lifecycle ────────────────────────────────────────────────────
ax.plot(df['current_day'].values, df['true_severity'].values,
        color='green', linewidth=1.8, linestyle='-.',
        label='True severity (full lifecycle reference)')

# ── 5. Predicted severity from regressor ─────────────────────────────────────
ax.plot(df['current_day'].values, df['severity'].values,
        color='purple', linewidth=1.0, linestyle=':',
        label='Predicted severity (regressor, reference)')

# ── 6. Reference lines ────────────────────────────────────────────────────────
ax.axvline(current_day, color='magenta', linestyle=':', linewidth=2,
           label=f'Current Day ({current_day})')
ax.axhline(S_threshold, color='red', linestyle='-', linewidth=1.5,
           label=f'Failure Threshold ({S_threshold:.3g})')
ax.axvline(L_pred, color='brown', linestyle='--', linewidth=1.5,
           label=f'Predicted EOL (day {L_pred:.1f})')
ax.axvline(true_last_day, color='green', linestyle=':', linewidth=1.5,
           label=f'True EOL (day {true_last_day})')

# ── Labels ────────────────────────────────────────────────────────────────────
ax.set_title(
    f'Predictive Maintenance — {fault_name}\n'
    f'Predicted RUL = {RUL_pred:.1f} days  |  α = {alpha_pred:.3f}  |  '
    f'Predicted lifecycle = {L_pred:.1f} days  |  True lifecycle = {true_last_day} days  |  '
    f'Error = {abs(RUL_pred - true_rul):.1f} days',
    fontsize=11
)
ax.set_xlabel('Days in Operation', fontsize=12)
ax.set_ylabel('Fault Severity', fontsize=12)
ax.legend(loc='best', fontsize=9)

if decreasing:
    ax.set_ylim(S_min * 0.95, S_max * 1.05)
else:
    ax.set_ylim(0, S_max * 1.1)

plt.tight_layout()
out_path = "rul_curve_fitting_plot.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nPlot saved to '{out_path}'")
