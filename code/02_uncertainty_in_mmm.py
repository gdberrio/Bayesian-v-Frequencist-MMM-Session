"""
02 - Uncertainty in MMM Is Real (Not a Bayesian Artifact)
==========================================================
Research Point (Section 2 / report 2.1 / all docs):
  Aryma Labs claims: "In MMM, there is no uncertainty... The sales are
  already realized... Our goal is to find these precise numbers."

  VERDICT: INCORRECT. There is genuine uncertainty from:
    1. Sampling variability (finite data from a stochastic process)
    2. Model specification uncertainty (which functional form is correct?)
    3. Causal identification uncertainty (confounding)
    4. Omitted variable bias

  "If you replayed the same 3 years with identical marketing spend,
   you would NOT get identical sales each week."

This script demonstrates:
  1. Sampling variability: same DGP, different samples -> different estimates
  2. Model specification uncertainty: Hill vs log saturation -> different ROIs
  3. A Dew et al. (2024)-style point: plausible alternative models can
     still imply different optimal allocations
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from demo_utils import finalize_figure

np.random.seed(42)

# ── Helper: Media transformations ──────────────────────────────────────────


def hill_function(x, ec, slope):
    """Hill saturation function."""
    return x**slope / (ec**slope + x**slope)


def log_saturation(x, alpha):
    """Log saturation function."""
    return alpha * np.log1p(x)


def geometric_adstock(x, decay):
    """Geometric adstock (carryover)."""
    result = np.zeros_like(x)
    result[0] = x[0]
    for t in range(1, len(x)):
        result[t] = x[t] + decay * result[t - 1]
    return result


def fit_curve_or_raise(label, func, x, y, **kwargs):
    """Raise a helpful error when a nonlinear demo fit stops converging."""
    try:
        return curve_fit(func, x, y, **kwargs)
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(f"{label} fit failed: {exc}") from exc


# ══════════════════════════════════════════════════════════════════════════
# PART 1: SAMPLING VARIABILITY
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 1: SAMPLING VARIABILITY -- Same DGP, Different Estimates")
print("=" * 70)

n_obs = 150
n_simulations = 500
beta_true = np.array([2.0, 1.5, 0.5])  # TV, Digital, Radio

# Collect OLS estimates across simulations
beta_estimates = np.zeros((n_simulations, 3))

for sim in range(n_simulations):
    # Generate fresh data from the SAME data-generating process
    X = np.random.exponential(scale=[50, 30, 20], size=(n_obs, 3))
    noise = np.random.normal(0, 5.0, n_obs)
    y = X @ beta_true + 100 + noise

    # OLS estimate
    X_aug = np.column_stack([np.ones(n_obs), X])
    beta_hat = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    beta_estimates[sim] = beta_hat[1:]  # skip intercept

print(
    f"\nTrue coefficients: TV={beta_true[0]}, Digital={beta_true[1]}, Radio={beta_true[2]}"
)
print(f"\nAcross {n_simulations} simulations of {n_obs} observations each:")
for i, ch in enumerate(["TV", "Digital", "Radio"]):
    mean_est = beta_estimates[:, i].mean()
    std_est = beta_estimates[:, i].std()
    range_95 = np.percentile(beta_estimates[:, i], [2.5, 97.5])
    print(
        f"  {ch:<8}: mean={mean_est:.3f}, std={std_est:.3f}, "
        f"95% range=[{range_95[0]:.3f}, {range_95[1]:.3f}]"
    )

print("\n  -> Even with the SAME data-generating process, estimates vary")
print("     substantially across samples. This IS real uncertainty.")


# ══════════════════════════════════════════════════════════════════════════
# PART 2: MODEL SPECIFICATION UNCERTAINTY
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 2: MODEL SPECIFICATION UNCERTAINTY -- Different Models, Same Data")
print("=" * 70)

# Generate one dataset with Hill saturation as the true DGP
n = 150
spend_tv = np.random.exponential(40, n)

# True DGP: strong nonlinear media effect plus moderate noise
adstocked_tv = geometric_adstock(spend_tv, decay=0.55)
true_contribution = 120.0 * hill_function(adstocked_tv, ec=90, slope=2.2)
y_true = 200 + true_contribution + np.random.normal(0, 12.0, n)

# Fit Model A: Hill saturation (correct specification)
popt_hill, _ = fit_curve_or_raise(
    "Hill saturation",
    lambda x, beta, ec, slope: beta * hill_function(x, ec, slope),
    adstocked_tv,
    y_true - 200,
    p0=[100.0, 90, 2.0],
    bounds=([0.0, 5.0, 0.5], [400.0, 250.0, 5.0]),
    maxfev=10000,
)
contribution_hill = popt_hill[0] * hill_function(adstocked_tv, popt_hill[1], popt_hill[2])
roi_hill = contribution_hill.sum() / spend_tv.sum()

# Fit Model B: Log saturation (misspecified but plausible)
popt_log, _ = fit_curve_or_raise(
    "Log saturation",
    lambda x, alpha: alpha * np.log1p(x),
    adstocked_tv,
    y_true - 200,
    p0=[25.0],
    maxfev=10000,
)
contribution_log = popt_log[0] * np.log1p(adstocked_tv)
roi_log = contribution_log.sum() / spend_tv.sum()

# Fit Model C: Linear-through-origin (simplest, also misspecified)
coef_linear = np.linalg.lstsq(adstocked_tv[:, None], y_true - 200, rcond=None)[0][0]
contribution_linear = coef_linear * adstocked_tv
roi_linear = contribution_linear.sum() / spend_tv.sum()


# Compute in-sample R^2 for each
def r_squared(y_true_vals, y_pred):
    ss_res = np.sum((y_true_vals - y_pred) ** 2)
    ss_tot = np.sum((y_true_vals - y_true_vals.mean()) ** 2)
    return 1 - ss_res / ss_tot


r2_hill = (
    r_squared(y_true - 200, contribution_hill) if not np.isnan(roi_hill) else np.nan
)
r2_log = r_squared(y_true - 200, contribution_log) if not np.isnan(roi_log) else np.nan
r2_linear = r_squared(y_true - 200, contribution_linear)

print(f"\nAll three models fit the SAME dataset ({n} observations):")
print(f"{'Model':<20} {'R-squared':>10} {'Implied ROI':>12}")
print("-" * 44)
print(f"{'Hill (correct)':<20} {r2_hill:>10.4f} {roi_hill:>12.4f}")
print(f"{'Log (misspecified)':<20} {r2_log:>10.4f} {roi_log:>12.4f}")
print(f"{'Linear (simple)':<20} {r2_linear:>10.4f} {roi_linear:>12.4f}")
print()
print("  -> Models with similar in-sample fit can imply DIFFERENT ROIs.")
print("     Different reasonable specifications can shift fitted ROI")
print("     and recommended spend from the same observed data.")


# ══════════════════════════════════════════════════════════════════════════
# PART 3: DEW ET AL. (2024) -- NONLINEAR vs TIME-VARYING EQUIVALENCE
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 3: Same Dataset, Different Fitted Explanations")
print("  (Illustrating Dew, Padilla & Shchetkina, 2024)")
print("=" * 70)

# Generate data that is consistent with BOTH:
#   Model A: Fixed effect with saturation (nonlinear, beta constant)
#   Model B: Time-varying linear effect (beta declines over time)

n_weeks = 156  # 3 years
t_norm = np.linspace(0, 1, n_weeks)
spend = (
    25
    + 0.40 * np.arange(n_weeks)
    + 18 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)
    + np.random.normal(0, 8, n_weeks)
)
spend = np.maximum(spend, 5)

# True DGP: saturation model
baseline = 100.0
true_saturated = 140.0 * hill_function(spend, ec=50, slope=2.0)
noise = np.random.normal(0, 3.0, n_weeks)
y_observed = baseline + true_saturated + noise

# Fit Model A: nonlinear saturation
popt_sat, _ = curve_fit(
    lambda x, beta, ec, slope: beta * hill_function(x, ec, slope),
    spend,
    y_observed - baseline,
    p0=[140.0, 50.0, 2.0],
    bounds=([0.0, 5.0, 0.5], [400.0, 200.0, 5.0]),
    maxfev=10000,
)
contribution_saturated = popt_sat[0] * hill_function(spend, popt_sat[1], popt_sat[2])

# Fit Model B: time-varying linear effectiveness on the SAME observed data
design_tv = np.column_stack([spend, spend * t_norm])
coef_tv, _, _, _ = np.linalg.lstsq(design_tv, y_observed - baseline, rcond=None)
beta_tv = coef_tv[0] + coef_tv[1] * t_norm
contribution_timevarying = design_tv @ coef_tv

# Show that both fitted models explain the data similarly
residuals_A = y_observed - baseline - contribution_saturated
residuals_B = y_observed - baseline - contribution_timevarying

rmse_A = np.sqrt(np.mean(residuals_A**2))
rmse_B = np.sqrt(np.mean(residuals_B**2))

# Compute implied optimal allocation at different spend levels
spend_test = np.linspace(10, 150, 100)

marginal_A = popt_sat[0] * hill_function(
    spend_test * 1.01, ec=popt_sat[1], slope=popt_sat[2]
) - popt_sat[0] * hill_function(spend_test, ec=popt_sat[1], slope=popt_sat[2])
marginal_A /= 0.01 * spend_test  # marginal ROI

current_marginal_B = beta_tv[-1]
early_marginal_B = beta_tv[0]

print(f"\nBoth models were fit to the same {n_weeks}-week dataset:")
print(f"  Model A (fitted Hill saturation):          RMSE = {rmse_A:.3f}")
print(f"  Model B (time-varying linear approximation): RMSE = {rmse_B:.3f}")
print()
print("Budget recommendations diverge:")
print(f"  Model A says: 'TV is below saturation at spend=80' -> INCREASE spend")
print(
    f"    (Marginal ROI at spend=80: {np.interp(80, spend_test, marginal_A):.3f})"
)
print(f"  Model B says: 'TV effectiveness is DECLINING over time' -> DECREASE spend")
print(
    f"    (Current marginal effect: {current_marginal_B:.3f}, was {early_marginal_B:.3f})"
)
print()
print("  -> SAME data, DIFFERENT fitted stories, DIFFERENT actionable conclusions.")
print("     Even without claiming perfect observational equivalence, model")
print("     specification uncertainty is enough to change recommendations.")


# ══════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Sampling variability
for i, (ch, color) in enumerate(
    zip(["TV", "Digital", "Radio"], ["#2196F3", "#FF9800", "#4CAF50"])
):
    axes[0, 0].hist(
        beta_estimates[:, i],
        bins=30,
        alpha=0.6,
        color=color,
        label=f"{ch} (true={beta_true[i]})",
    )
    axes[0, 0].axvline(beta_true[i], color=color, linestyle="--", linewidth=2)
axes[0, 0].set_xlabel("Estimated Coefficient")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].set_title("Part 1: Sampling Variability\n(500 simulations, same DGP)")
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(alpha=0.3)

# Panel B: Model specification uncertainty
spend_plot = np.linspace(1, 200, 200)
adstocked_plot = spend_plot  # simplified for visualization
axes[0, 1].scatter(adstocked_tv, y_true - 200, alpha=0.3, s=10, c="gray", label="Data")
sorted_idx = np.argsort(adstocked_tv)
if not np.isnan(roi_hill):
    axes[0, 1].plot(
        adstocked_tv[sorted_idx],
        contribution_hill[sorted_idx],
        "b-",
        linewidth=2,
        label=f"Hill (ROI={roi_hill:.2f})",
    )
if not np.isnan(roi_log):
    axes[0, 1].plot(
        adstocked_tv[sorted_idx],
        contribution_log[sorted_idx],
        "r-",
        linewidth=2,
        label=f"Log (ROI={roi_log:.2f})",
    )
axes[0, 1].plot(
    adstocked_tv[sorted_idx],
    contribution_linear[sorted_idx],
    "g-",
    linewidth=2,
    label=f"Linear (ROI={roi_linear:.2f})",
)
axes[0, 1].set_xlabel("Adstocked Spend")
axes[0, 1].set_ylabel("Media Contribution")
axes[0, 1].set_title(
    "Part 2: Model Specification Uncertainty\n(Same data, different models)"
)
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3)

# Panel C: Dew et al. -- two explanations
axes[1, 0].plot(
    contribution_saturated, label="Model A: Fitted Hill saturation", alpha=0.8
)
axes[1, 0].plot(
    contribution_timevarying,
    label="Model B: Fitted time-varying linear",
    alpha=0.8,
    linestyle="--",
)
axes[1, 0].set_xlabel("Week")
axes[1, 0].set_ylabel("Estimated Media Contribution")
axes[1, 0].set_title("Part 3: Dew et al. (2024)\nSame Data, Different Fitted Explanations")
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(alpha=0.3)

# Panel D: Divergent budget recommendations
axes[1, 1].plot(
    spend_test,
    marginal_A * 100,
    "b-",
    linewidth=2,
    label="Model A: Diminishing marginal ROI",
)
axes[1, 1].axhline(
    current_marginal_B * 100,
    color="r",
    linestyle="--",
    linewidth=2,
    label="Model B: Current-week marginal effect",
)
axes[1, 1].axvline(80, color="gray", linestyle=":", alpha=0.5, label="Current spend")
axes[1, 1].set_xlabel("TV Spend Level")
axes[1, 1].set_ylabel("Marginal ROI (scaled)")
axes[1, 1].set_title("Divergent Budget Recommendations\nfrom Equivalent Models")
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=0.3)

plt.suptitle(
    "Uncertainty in MMM Is Real -- Not a Bayesian Artifact",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
finalize_figure("02_uncertainty_in_mmm.png", fig=fig)

print("\n" + "=" * 70)
print("KEY TAKEAWAY:")
print("  The claim 'there is no uncertainty in MMM' conflates the certainty")
print("  of historical data with the certainty of causal effects. Even with")
print("  FIXED historical data, there is genuine uncertainty from sampling")
print("  variability, model specification, and causal identification.")
print("  This is true REGARDLESS of whether you use Bayesian or frequentist")
print("  methods. Frequentist CIs are also uncertainty quantification tools.")
print("=" * 70)
