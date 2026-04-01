"""
06 - Multicollinearity: Neither Approach Solves It
===================================================
Research Point (Section 10 / report Section 3):
  "Neither approach solves multicollinearity. It is a data problem,
   not a method problem."

  OLS: "Assigns all credit to one channel, zero to the other. Narrow but
        wrong confidence intervals -- 'false confidence.'" (Recast 2026)

  Bayesian: "Produces similar estimates for both channels with wide
             uncertainty bands. More honest, less actionable."

  Robyn DECOMP.RSSD: implicit prior that share-of-effect should relate
  to share-of-spend -- "a strong and rarely acknowledged assumption."

  "The ONLY real solution: vary spend independently, pause channels,
   run geo-lift tests."

This script demonstrates:
  1. How multicollinearity destroys OLS estimation
  2. Ridge manages it by shrinking (biased but stable)
  3. Bayesian with wide priors honestly shows wide posteriors
  4. Bayesian with tight priors "solves" it by assuming the answer
  5. Experimental variation (independent spend) fixes it for real
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from scipy import stats

from demo_utils import finalize_figure

np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════
# PART 1: HOW MULTICOLLINEARITY DESTROYS OLS
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 1: Multicollinearity Destroys OLS")
print("=" * 70)

n_obs = 150
n_sims = 500

# True effects
beta_tv = 2.0
beta_digital = 1.5


def run_simulation(correlation, n_sims=500):
    """Run simulation with given correlation between TV and Digital."""
    ols_estimates = np.zeros((n_sims, 2))
    ridge_estimates = np.zeros((n_sims, 2))

    for sim in range(n_sims):
        # Generate correlated spend
        cov = np.array([[100, correlation * 100], [correlation * 100, 100]])
        X = np.random.multivariate_normal([50, 40], cov, n_obs)
        X = np.maximum(X, 1)

        noise = np.random.normal(0, 5.0, n_obs)
        y = beta_tv * X[:, 0] + beta_digital * X[:, 1] + 100 + noise

        X_c = X - X.mean(axis=0)
        y_c = y - y.mean()

        # OLS
        ols = LinearRegression(fit_intercept=False).fit(X_c, y_c)
        ols_estimates[sim] = ols.coef_

        # Ridge
        ridge = Ridge(alpha=10.0, fit_intercept=False).fit(X_c, y_c)
        ridge_estimates[sim] = ridge.coef_

    return ols_estimates, ridge_estimates


def gaussian_posterior(X_c, y_c, sigma_noise, prior_sigma):
    """Closed-form posterior for Gaussian likelihood + Gaussian prior."""
    prior_precision = np.eye(X_c.shape[1]) / prior_sigma**2
    likelihood_precision = (X_c.T @ X_c) / sigma_noise**2
    post_cov = np.linalg.inv(prior_precision + likelihood_precision)
    post_mean = post_cov @ (X_c.T @ y_c / sigma_noise**2)
    return post_mean, post_cov


# Test at different correlation levels
correlations = [0.0, 0.3, 0.6, 0.8, 0.9, 0.95]

print(f"\nTrue: beta_TV={beta_tv}, beta_Digital={beta_digital}")
print()
print(
    f"{'Corr':>6} | {'OLS TV':>8} {'OLS Dig':>8} {'OLS TV std':>10} | {'Ridge TV':>9} {'Ridge Dig':>10} {'Ridge TV std':>12}"
)
print("-" * 80)

results = {}
for corr in correlations:
    ols_est, ridge_est = run_simulation(corr, n_sims)
    results[corr] = (ols_est, ridge_est)
    print(
        f"{corr:>6.2f} | {ols_est[:, 0].mean():>8.3f} {ols_est[:, 1].mean():>8.3f} "
        f"{ols_est[:, 0].std():>10.3f} | {ridge_est[:, 0].mean():>9.3f} "
        f"{ridge_est[:, 1].mean():>10.3f} {ridge_est[:, 0].std():>12.3f}"
    )

print()
print("  -> As correlation increases:")
print("     OLS: estimates become progressively unstable")
print("     Ridge: estimates are biased but somewhat more stable")


# ══════════════════════════════════════════════════════════════════════════
# PART 2: BAYESIAN WITH DIFFERENT PRIOR STRENGTHS
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 2: Bayesian Prior Strength and Multicollinearity")
print("=" * 70)

corr_high = 0.95
cov_high = np.array([[100, corr_high * 100], [corr_high * 100, 100]])

prior_configs = [
    ("Wide prior (sigma=50)", 50.0, "#2196F3"),
    ("Moderate prior (sigma=2)", 2.0, "#FF9800"),
    ("Tight prior (sigma=0.5)", 0.5, "#D32F2F"),
]

print(f"\nCorrelation = {corr_high}")
print(f"True: beta_TV={beta_tv}, beta_Digital={beta_digital}")
print()
print("  One fixed collinear dataset, then exact Gaussian posteriors:")
X_fixed = np.random.multivariate_normal([50, 40], cov_high, n_obs)
X_fixed = np.maximum(X_fixed, 1)
noise_fixed = np.random.normal(0, 5.0, n_obs)
y_fixed = beta_tv * X_fixed[:, 0] + beta_digital * X_fixed[:, 1] + 100 + noise_fixed

X_fixed_c = X_fixed - X_fixed.mean(axis=0)
y_fixed_c = y_fixed - y_fixed.mean()
ols_fixed = LinearRegression(fit_intercept=False).fit(X_fixed_c, y_fixed_c).coef_

posterior_results = {}
print(f"  OLS point estimate: TV={ols_fixed[0]:.3f}, Digital={ols_fixed[1]:.3f}")
for name, prior_sigma, color in prior_configs:
    post_mean, post_cov = gaussian_posterior(
        X_fixed_c, y_fixed_c, sigma_noise=5.0, prior_sigma=prior_sigma
    )
    post_sd = np.sqrt(np.diag(post_cov))
    post_corr = post_cov[0, 1] / (post_sd[0] * post_sd[1])
    posterior_results[name] = {
        "mean": post_mean,
        "cov": post_cov,
        "sd": post_sd,
        "corr": post_corr,
        "color": color,
    }

    print(f"  {name}:")
    print(f"    Mean:     TV={post_mean[0]:.3f}, Digital={post_mean[1]:.3f}")
    print(f"    Post SD:  TV={post_sd[0]:.3f}, Digital={post_sd[1]:.3f}")
    print(f"    Post Corr between coefficients: {post_corr:.3f}")

print()
print("  -> Wide prior: elongated posterior, because the data can't cleanly")
print("     separate TV from Digital on this fixed dataset.")
print("     Tight prior: narrower posterior, but only because the prior has")
print("     supplied the information the data lacked.")


# ══════════════════════════════════════════════════════════════════════════
# PART 3: DECOMP.RSSD -- ROBYN'S IMPLICIT PRIOR
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 3: Robyn's DECOMP.RSSD -- An Implicit Prior")
print("=" * 70)

# Simulate: with highly correlated channels, different models
# give very different decompositions
n_channels = 4
channel_names = ["TV", "Digital", "Radio", "OOH"]
true_betas = np.array([2.0, 1.5, 0.8, 0.5])
true_spend_share = np.array([0.35, 0.30, 0.20, 0.15])  # share of total spend

n_models = 200
decomp_rssd_scores = []
nrmse_scores = []
model_decomps = []

for _ in range(n_models):
    # Generate correlated data
    cov_4ch = np.eye(n_channels) * 100
    for i in range(n_channels):
        for j in range(n_channels):
            if i != j:
                cov_4ch[i, j] = 75  # high correlation

    X = np.random.multivariate_normal(50 * np.ones(n_channels), cov_4ch, n_obs)
    X = np.maximum(X, 1)
    noise = np.random.normal(0, 5.0, n_obs)
    y = X @ true_betas + 100 + noise

    # OLS with some random perturbation (simulating different hyperparameter configs)
    perturbation = np.random.normal(0, 0.5, n_channels)
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0] + perturbation

    # Compute decomposition share
    contribution = X * beta_hat
    total_contribution = contribution.sum()
    if total_contribution > 0:
        decomp_share = contribution.sum(axis=0) / total_contribution
    else:
        decomp_share = np.ones(n_channels) / n_channels

    # DECOMP.RSSD: root sum squared distance between effect share and spend share
    rssd = np.sqrt(np.sum((decomp_share - true_spend_share) ** 2))

    # NRMSE
    y_pred = X @ beta_hat
    nrmse = np.sqrt(np.mean((y - y_pred) ** 2)) / y.std()

    decomp_rssd_scores.append(rssd)
    nrmse_scores.append(nrmse)
    model_decomps.append(decomp_share)

print(f"\n  DECOMP.RSSD penalizes models where share-of-effect diverges")
print(f"  from share-of-spend. This is an IMPLICIT PRIOR that channels")
print(f"  'deserve' credit proportional to their spend.")
print()
print(f"  True spend share:   {dict(zip(channel_names, true_spend_share))}")
print(
    f"  True effect share:  {dict(zip(channel_names, (true_betas * 50) / (true_betas * 50).sum()))}"
)
print()
print(f"  If spend share != effect share (which is likely!), DECOMP.RSSD")
print(f"  penalizes the model for finding the TRUTH.")


# ══════════════════════════════════════════════════════════════════════════
# PART 4: EXPERIMENTAL VARIATION FIXES IT
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 4: Experimental Variation -- The Only Real Solution")
print("=" * 70)

# Compare: correlated vs independent spend
ols_corr = np.zeros((n_sims, 2))
ols_indep = np.zeros((n_sims, 2))

for sim in range(n_sims):
    # Correlated spend (typical)
    cov_corr = np.array([[100, 95], [95, 100]])
    X_corr = np.random.multivariate_normal([50, 40], cov_corr, n_obs)

    # Independent spend (experimental design)
    X_indep = np.column_stack(
        [np.random.normal(50, 10, n_obs), np.random.normal(40, 10, n_obs)]
    )

    for X, storage in [(X_corr, ols_corr), (X_indep, ols_indep)]:
        X = np.maximum(X, 1)
        noise = np.random.normal(0, 5.0, n_obs)
        y = beta_tv * X[:, 0] + beta_digital * X[:, 1] + 100 + noise
        X_c = X - X.mean(axis=0)
        y_c = y - y.mean()
        ols = LinearRegression(fit_intercept=False).fit(X_c, y_c)
        storage[sim] = ols.coef_

print(f"\nWith CORRELATED spend (r=0.95):")
print(f"  TV:      mean={ols_corr[:, 0].mean():.3f}, std={ols_corr[:, 0].std():.3f}")
print(f"  Digital: mean={ols_corr[:, 1].mean():.3f}, std={ols_corr[:, 1].std():.3f}")
print(f"\nWith INDEPENDENT spend (experimental design):")
print(f"  TV:      mean={ols_indep[:, 0].mean():.3f}, std={ols_indep[:, 0].std():.3f}")
print(f"  Digital: mean={ols_indep[:, 1].mean():.3f}, std={ols_indep[:, 1].std():.3f}")
print(
    f"\n  -> Independent spend variation reduces OLS std by "
    f"{(1 - ols_indep[:, 0].std() / ols_corr[:, 0].std()) * 100:.0f}%!"
)
print("     No statistical method can substitute for experimental design.")


# ══════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: OLS stability vs correlation
corr_stds = [results[c][0][:, 0].std() for c in correlations]
ridge_stds = [results[c][1][:, 0].std() for c in correlations]

axes[0, 0].plot(correlations, corr_stds, "bo-", linewidth=2, markersize=8, label="OLS")
axes[0, 0].plot(
    correlations, ridge_stds, "rs-", linewidth=2, markersize=8, label="Ridge"
)
axes[0, 0].set_xlabel("Channel Correlation")
axes[0, 0].set_ylabel("Std Dev of TV Coefficient Estimate")
axes[0, 0].set_title(
    "OLS Widens Under Multicollinearity\nRidge Is Somewhat More Stable (But Biased)"
)
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Panel B: Exact posterior contours on one fixed collinear dataset
wide = posterior_results["Wide prior (sigma=50)"]
tight = posterior_results["Tight prior (sigma=0.5)"]

tv_grid = np.linspace(
    min(beta_tv, wide["mean"][0] - 4 * wide["sd"][0]),
    max(beta_tv, wide["mean"][0] + 4 * wide["sd"][0]),
    160,
)
dig_grid = np.linspace(
    min(beta_digital, wide["mean"][1] - 4 * wide["sd"][1]),
    max(beta_digital, wide["mean"][1] + 4 * wide["sd"][1]),
    160,
)
tv_mesh, dig_mesh = np.meshgrid(tv_grid, dig_grid)
pos = np.dstack((tv_mesh, dig_mesh))

for name in ["Wide prior (sigma=50)", "Moderate prior (sigma=2)", "Tight prior (sigma=0.5)"]:
    result = posterior_results[name]
    density = stats.multivariate_normal(mean=result["mean"], cov=result["cov"]).pdf(pos)
    axes[0, 1].contour(
        tv_mesh,
        dig_mesh,
        density,
        levels=np.linspace(density.max() * 0.2, density.max() * 0.8, 3),
        colors=result["color"],
        linewidths=2,
    )

axes[0, 1].scatter(
    [beta_tv], [beta_digital], c="black", s=100, zorder=5, marker="*", label="True"
)
axes[0, 1].scatter(
    [ols_fixed[0]], [ols_fixed[1]], c="#9E9E9E", s=60, zorder=5, label="OLS point estimate"
)
axes[0, 1].set_xlabel("TV Coefficient")
axes[0, 1].set_ylabel("Digital Coefficient")
axes[0, 1].set_title(
    "Exact Gaussian Posteriors on One Fixed Dataset\nWide prior leaves an elongated ridge"
)
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3)

# Panel C: DECOMP.RSSD as Pareto tradeoff
axes[1, 0].scatter(nrmse_scores, decomp_rssd_scores, alpha=0.4, s=20, c="#9C27B0")
axes[1, 0].set_xlabel("NRMSE (prediction error)")
axes[1, 0].set_ylabel("DECOMP.RSSD (effect share vs spend share)")
axes[1, 0].set_title(
    "Robyn's Multi-Objective Space\nDECOMP.RSSD is an implicit prior on decomposition"
)
axes[1, 0].grid(alpha=0.3)
axes[1, 0].annotate(
    "Low RSSD = effect share ~ spend share\n(strong implicit assumption)",
    xy=(0.5, 0.02),
    xycoords="axes fraction",
    fontsize=8,
    ha="center",
    style="italic",
    color="gray",
)

# Panel D: Correlated vs Independent spend
axes[1, 1].hist(
    ols_corr[:, 0],
    bins=40,
    alpha=0.5,
    color="#FF9800",
    label=f"Correlated (std={ols_corr[:, 0].std():.2f})",
    density=True,
)
axes[1, 1].hist(
    ols_indep[:, 0],
    bins=40,
    alpha=0.5,
    color="#4CAF50",
    label=f"Independent (std={ols_indep[:, 0].std():.2f})",
    density=True,
)
axes[1, 1].axvline(
    beta_tv, color="red", linestyle="--", linewidth=2, label=f"True={beta_tv}"
)
axes[1, 1].set_xlabel("TV Coefficient Estimate")
axes[1, 1].set_ylabel("Density")
axes[1, 1].set_title(
    "The Real Solution: Independent Spend Variation\nExperimental design > better statistics"
)
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=0.3)

plt.suptitle(
    "Multicollinearity: Neither Approach Solves It -- Only Experiments Do",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
finalize_figure("06_multicollinearity.png", fig=fig)

print("\nKEY TAKEAWAY:")
print("  Neither Bayesian nor frequentist methods SOLVE multicollinearity.")
print("  Bayesian: wider posteriors (honest) or tight priors (masks it).")
print("  Ridge: stable but biased estimates (same as Bayesian MAP).")
print("  Robyn's DECOMP.RSSD embeds an implicit prior as strong as any Bayesian prior.")
print("  The only real solution is experimental: vary spend independently.")
