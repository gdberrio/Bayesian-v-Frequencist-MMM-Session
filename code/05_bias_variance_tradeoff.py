"""
05 - Bias-Variance Tradeoff: OLS vs Ridge vs Bayesian
=======================================================
Research Point (Section 8 / report 2.6):
  "The 'two wrongs' framing is rhetorically effective but mathematically
   naive. Regularization/shrinkage is not a 'wrong' -- it's a well-
   understood tradeoff that improves estimation quality (lower MSE)
   at the cost of bias."

  "Unregularized OLS with multicollinear data produces higher MSE than
   biased regularized estimates."

  James-Stein paradox (1961): for p >= 3 parameters estimated simultaneously,
  the unbiased MLE is INADMISSIBLE -- there always exists a biased shrinkage
  estimator with lower total MSE.

  | Prior Distribution       | Equivalent Regularization              |
  |--------------------------|----------------------------------------|
  | Normal(0, tau^2)         | Ridge: lambda = sigma^2/tau^2          |
  | Laplace(0, b)            | Lasso: lambda = sigma^2/b              |
  | HalfNormal(sigma)        | Constrained Ridge (non-negative)       |

This script demonstrates:
  1. Bias-variance tradeoff across regularization strengths
  2. MSE decomposition: MSE = Bias^2 + Variance
  3. OLS can be wildly wrong for individual datasets despite being unbiased
  4. James-Stein phenomenon: simultaneous estimation benefit
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression

from demo_utils import finalize_figure

np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════
# PART 1: BIAS-VARIANCE TRADEOFF ACROSS LAMBDA
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 1: Bias-Variance-MSE Tradeoff")
print("=" * 70)

n_obs = 100
n_channels = 8
n_sims = 800
sigma_noise = 5.0

# True coefficients
beta_true = np.array([2.5, 1.8, 1.2, 0.8, 0.5, 0.3, 0.1, 3.0])

# High-collinearity MMM-like design so shrinkage has a fair chance to help
mean_X = 50 * np.ones(n_channels)
cov_X = np.full((n_channels, n_channels), 22.0)
np.fill_diagonal(cov_X, 25.0)

lambdas = np.logspace(-2, 3, 50)

# Storage for bias, variance, MSE
all_bias2 = np.zeros(len(lambdas))
all_variance = np.zeros(len(lambdas))
all_mse = np.zeros(len(lambdas))

# Also track OLS and Ridge on the SAME simulated datasets
ols_estimates = np.zeros((n_sims, n_channels))
all_ridge_estimates = np.zeros((n_sims, len(lambdas), n_channels))

for sim in range(n_sims):
    X = np.random.multivariate_normal(mean_X, cov_X, n_obs)
    noise = np.random.normal(0, sigma_noise, n_obs)
    y = X @ beta_true + 100 + noise

    X_c = X - X.mean(axis=0)
    y_c = y - y.mean()

    ols = LinearRegression(fit_intercept=False).fit(X_c, y_c)
    ols_estimates[sim] = ols.coef_

    for li, lam in enumerate(lambdas):
        ridge = Ridge(alpha=lam, fit_intercept=False).fit(X_c, y_c)
        all_ridge_estimates[sim, li, :] = ridge.coef_

# Compute bias^2, variance, MSE properly
for li in range(len(lambdas)):
    mean_est = all_ridge_estimates[:, li, :].mean(axis=0)
    bias2 = np.sum((mean_est - beta_true) ** 2)
    variance = np.sum(all_ridge_estimates[:, li, :].var(axis=0))
    all_bias2[li] = bias2
    all_variance[li] = variance
    all_mse[li] = bias2 + variance

# OLS metrics
ols_mean = ols_estimates.mean(axis=0)
ols_bias2 = np.sum((ols_mean - beta_true) ** 2)
ols_variance = np.sum(ols_estimates.var(axis=0))
ols_mse = ols_bias2 + ols_variance

# Find optimal lambda
opt_idx = np.argmin(all_mse)
opt_lambda = lambdas[opt_idx]
mse_reduction_pct = (ols_mse - all_mse[opt_idx]) / ols_mse * 100

print(f"\nOLS (lambda=0):")
print(f"  Bias^2   = {ols_bias2:.4f}")
print(f"  Variance = {ols_variance:.4f}")
print(f"  MSE      = {ols_mse:.4f}")
print(f"\nOptimal Ridge (lambda={opt_lambda:.2f}):")
print(f"  Bias^2   = {all_bias2[opt_idx]:.4f}")
print(f"  Variance = {all_variance[opt_idx]:.4f}")
print(f"  MSE      = {all_mse[opt_idx]:.4f}")
print(f"\nMSE reduction: {mse_reduction_pct:.1f}%")
if mse_reduction_pct > 0:
    print(
        "\n  -> In this small-sample, high-collinearity regime, optimal Ridge"
        " lowers total MSE by accepting a little bias to cut variance."
    )
else:
    print(
        "\n  -> In this run, Ridge is roughly tied with OLS on total MSE."
        " The tradeoff is real, but the size of the gain depends on how"
        " ill-conditioned the data are."
    )


# ══════════════════════════════════════════════════════════════════════════
# PART 2: OLS CAN BE WILDLY WRONG FOR INDIVIDUAL DATASETS
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 2: OLS Is Unbiased ON AVERAGE But Can Be Wildly Wrong")
print("=" * 70)

# Show individual dataset OLS estimates
print(f"\nTrue betas: {beta_true}")
print(f"\nFirst 5 simulation OLS estimates (each row is a 'different 3 years'):")
print(f"{'Sim':>4} ", end="")
for i in range(n_channels):
    print(f"{'Ch' + str(i + 1):>8}", end="")
print(f"{'||err||':>10}")
print("-" * 80)
for sim in range(5):
    print(f"{sim + 1:>4} ", end="")
    for i in range(n_channels):
        val = ols_estimates[sim, i]
        print(f"{val:>8.2f}", end="")
    err = np.sqrt(np.sum((ols_estimates[sim] - beta_true) ** 2))
    print(f"{err:>10.2f}")

print(f"\nMean across {n_sims} simulations (should be close to true):")
print(f"Mean ", end="")
for i in range(n_channels):
    print(f"{ols_mean[i]:>8.3f}", end="")
print()
print(f"True ", end="")
for i in range(n_channels):
    print(f"{beta_true[i]:>8.3f}", end="")
print()

# What fraction of OLS estimates have wrong sign?
wrong_sign = np.sum(ols_estimates < 0, axis=0)
print(f"\nOLS gives negative estimate (wrong sign) in:")
for i in range(n_channels):
    print(
        f"  Ch{i + 1} (true={beta_true[i]:.1f}): {wrong_sign[i]}/{n_sims} ({wrong_sign[i] / n_sims * 100:.1f}%) of simulations"
    )


# ══════════════════════════════════════════════════════════════════════════
# PART 3: JAMES-STEIN PHENOMENON
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 3: James-Stein Phenomenon (1961)")
print("=" * 70)
print()
print("  For p >= 3 parameters estimated simultaneously, the UNBIASED MLE")
print("  is INADMISSIBLE: there ALWAYS exists a shrinkage estimator with")
print("  lower total MSE. This is not a matter of opinion -- it's proven.")
print()

# Demonstrate with simple example
p = 8  # number of means to estimate
n_js_sims = 10000
sigma_js = 1.0

true_means = beta_true / beta_true.max()  # scale to be interesting

mse_mle = 0
mse_js = 0

for _ in range(n_js_sims):
    # Observe noisy versions
    x_obs = true_means + np.random.normal(0, sigma_js, p)

    # MLE: just use observations
    mse_mle += np.sum((x_obs - true_means) ** 2) / n_js_sims

    # James-Stein estimator
    shrinkage = max(0, 1 - (p - 2) * sigma_js**2 / np.sum(x_obs**2))
    x_js = shrinkage * x_obs
    mse_js += np.sum((x_js - true_means) ** 2) / n_js_sims

print(f"  MLE total MSE:          {mse_mle:.4f}")
print(f"  James-Stein total MSE:  {mse_js:.4f}")
print(f"  Improvement:            {(mse_mle - mse_js) / mse_mle * 100:.1f}%")
print()
print(f"  -> Shrinkage (= regularization = Bayesian priors) provably")
print(f"     improves estimation. 'Two wrongs' is mathematically wrong.")


# ══════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Bias-Variance-MSE tradeoff
axes[0, 0].plot(lambdas, all_bias2, "r-", linewidth=2, label="Bias$^2$")
axes[0, 0].plot(lambdas, all_variance, "b-", linewidth=2, label="Variance")
axes[0, 0].plot(lambdas, all_mse, "k-", linewidth=3, label="MSE = Bias$^2$ + Variance")
axes[0, 0].axvline(
    opt_lambda,
    color="green",
    linestyle="--",
    linewidth=2,
    label=f"Optimal lambda={opt_lambda:.1f}",
)
axes[0, 0].axhline(
    ols_mse, color="gray", linestyle=":", alpha=0.5, label=f"OLS MSE={ols_mse:.2f}"
)
axes[0, 0].set_xscale("log")
axes[0, 0].set_xlabel("Regularization Strength (lambda)")
axes[0, 0].set_ylabel("Error")
axes[0, 0].set_title("Bias-Variance Tradeoff\nMSE = Bias$^2$ + Variance")
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(alpha=0.3)

# Panel B: OLS estimate distribution for one channel
ch_plot = 4  # Ch5 with true beta = 0.5
axes[0, 1].hist(
    ols_estimates[:, ch_plot],
    bins=40,
    alpha=0.5,
    color="#2196F3",
    label=f"OLS (mean={ols_mean[ch_plot]:.2f})",
    density=True,
)

# Ridge at optimal lambda
axes[0, 1].hist(
    all_ridge_estimates[:, opt_idx, ch_plot],
    bins=40,
    alpha=0.5,
    color="#FF9800",
    label=f"Ridge (mean={all_ridge_estimates[:, opt_idx, ch_plot].mean():.2f})",
    density=True,
)
axes[0, 1].axvline(
    beta_true[ch_plot],
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"True beta={beta_true[ch_plot]}",
)
axes[0, 1].axvline(0, color="gray", linestyle=":", alpha=0.5)
axes[0, 1].set_xlabel(f"Estimated Ch{ch_plot + 1} Coefficient")
axes[0, 1].set_ylabel("Density")
axes[0, 1].set_title(
    f"OLS vs Ridge Estimates for Ch{ch_plot + 1}\nOLS is wider, Ridge is tighter around truth"
)
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3)

# Panel C: Per-channel MSE comparison
ols_ch_mse = ((ols_estimates - beta_true) ** 2).mean(axis=0)
ridge_ch_mse = ((all_ridge_estimates[:, opt_idx, :] - beta_true) ** 2).mean(axis=0)

x_pos = np.arange(n_channels)
width = 0.35
axes[1, 0].bar(
    x_pos - width / 2, ols_ch_mse, width, label="OLS", color="#2196F3", alpha=0.8
)
axes[1, 0].bar(
    x_pos + width / 2,
    ridge_ch_mse,
    width,
    label=f"Ridge (lambda={opt_lambda:.0f})",
    color="#FF9800",
    alpha=0.8,
)
axes[1, 0].set_xlabel("Channel")
axes[1, 0].set_ylabel("Mean Squared Error")
axes[1, 0].set_title(
    f"Per-Channel MSE: OLS vs Optimal Ridge\nRidge wins on {(ridge_ch_mse < ols_ch_mse).sum()}/{n_channels} channels"
)
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels([f"Ch{i + 1}\n({b:.1f})" for i, b in enumerate(beta_true)])
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(axis="y", alpha=0.3)

# Panel D: Equivalence table visualization
table_data = [
    ["Normal(0, tau^2)", "Ridge", "lambda = sigma^2/tau^2"],
    ["Laplace(0, b)", "Lasso", "lambda = sigma^2/b"],
    ["HalfNormal(sigma)", "Constrained Ridge", "Non-negative"],
    ["Horseshoe", "Adaptive shrinkage", "No closed form"],
]
axes[1, 1].axis("off")
table = axes[1, 1].table(
    cellText=table_data,
    colLabels=["Bayesian Prior", "Frequentist Equiv.", "Mapping"],
    cellLoc="center",
    loc="center",
    colColours=["#E3F2FD", "#FFF3E0", "#E8F5E9"],
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2.0)
axes[1, 1].set_title(
    "Prior-Regularization Equivalence\n'Ridge regression IS Bayesian regression with a Gaussian prior'",
    fontsize=11,
    pad=20,
)

plt.suptitle(
    "Bias-Variance Tradeoff: 'Two Wrongs' Is Mathematically Naive",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
finalize_figure("05_bias_variance_tradeoff.png", fig=fig)

print("\nKEY TAKEAWAY:")
print("  Regularization (Ridge/Bayesian priors) is NOT 'two wrongs.'")
print("  It is a deliberate, provably optimal tradeoff: accept small bias")
print("  to dramatically reduce variance. OLS is unbiased ON AVERAGE but")
print("  can be catastrophically wrong for any individual dataset.")
print(f"  In this simulation, optimal Ridge changes total MSE by {mse_reduction_pct:.1f}%.")
print("  Both Robyn (Ridge) and Meridian (Bayesian priors) use this tradeoff.")
print("  The frequentist camp criticizing Bayesian bias is using Ridge --")
print("  which IS Bayesian regression with a Gaussian prior.")
