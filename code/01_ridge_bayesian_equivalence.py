"""
01 - Ridge Regression = Bayesian MAP with Gaussian Prior
=========================================================
Research Point (Section 1 / report 1.3 / claude-Deep-Research):
  "Ridge regression is mathematically equivalent to Bayesian regression
   with a Gaussian prior."  -- Hastie et al. (ISLR, p. 249)

  Ridge minimizes:  RSS + lambda * ||beta||^2
  Bayesian MAP with Normal(0, tau) prior minimizes:
      RSS + (sigma^2 / tau^2) * ||beta||^2
  These are IDENTICAL when lambda = sigma^2 / tau^2.

This script demonstrates:
  1. Generate synthetic MMM-like data
  2. Fit Ridge regression (sklearn)
  3. Fit Bayesian MAP with Gaussian prior (closed-form)
  4. Show the coefficient estimates are numerically identical
  5. Visualize the equivalence
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

from demo_utils import finalize_figure

np.random.seed(42)

# ── 1. Simulate MMM-like data ──────────────────────────────────────────────
n_obs = 150  # ~3 years of weekly data
n_channels = 5  # media channels
sigma_true = 1.0  # noise std

# True coefficients (some small, some large)
beta_true = np.array([2.5, 0.8, 1.5, 0.3, 3.0])

# Correlated media spend (realistic: channels move together)
mean_spend = np.array([100, 80, 60, 40, 120])
cov_spend = np.diag(mean_spend) * 0.3
# Add off-diagonal correlation (Q4 effect)
for i in range(n_channels):
    for j in range(n_channels):
        if i != j:
            cov_spend[i, j] = 0.15 * np.sqrt(cov_spend[i, i] * cov_spend[j, j])

X = np.random.multivariate_normal(mean_spend, cov_spend, size=n_obs)
X = np.maximum(X, 0)  # no negative spend

# Generate response
noise = np.random.normal(0, sigma_true, n_obs)
y = X @ beta_true + 50 + noise  # intercept = 50

# Center X and y for Ridge (no intercept in the penalty)
X_mean = X.mean(axis=0)
y_mean = y.mean()
X_c = X - X_mean
y_c = y - y_mean

# ── 2. Ridge Regression (sklearn) ──────────────────────────────────────────
lambda_ridge = 5.0

ridge = Ridge(alpha=lambda_ridge, fit_intercept=False)
ridge.fit(X_c, y_c)
beta_ridge = ridge.coef_

# ── 3. Bayesian MAP with Gaussian Prior (closed-form) ──────────────────────
# MAP estimate: beta_MAP = (X'X + lambda*I)^{-1} X'y
# This is identical to Ridge when lambda = sigma^2 / tau^2
# So tau^2 = sigma^2 / lambda

XtX = X_c.T @ X_c
Xty = X_c.T @ y_c
I = np.eye(n_channels)

beta_map = np.linalg.solve(XtX + lambda_ridge * I, Xty)

# ── 4. Compare ─────────────────────────────────────────────────────────────
print("=" * 70)
print("DEMONSTRATION: Ridge Regression = Bayesian MAP with Gaussian Prior")
print("=" * 70)
print()
print(f"Number of observations:  {n_obs}")
print(f"Number of channels:      {n_channels}")
print(f"Ridge lambda:            {lambda_ridge}")
print(f"Equivalent prior tau^2:  {sigma_true**2 / lambda_ridge:.4f}")
print()

print(f"{'Channel':<10} {'True':>8} {'Ridge':>8} {'Bayes MAP':>10} {'Diff':>12}")
print("-" * 50)
for i in range(n_channels):
    diff = abs(beta_ridge[i] - beta_map[i])
    print(
        f"Ch {i + 1:<6} {beta_true[i]:>8.4f} {beta_ridge[i]:>8.4f} {beta_map[i]:>10.4f} {diff:>12.2e}"
    )

print()
max_diff = np.max(np.abs(beta_ridge - beta_map))
print(f"Maximum absolute difference: {max_diff:.2e}")
print(f"Numerically identical:       {max_diff < 1e-10}")

# ── 5. Visualize ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Coefficient comparison
channels = [f"Ch {i + 1}" for i in range(n_channels)]
x_pos = np.arange(n_channels)
width = 0.25

axes[0].bar(x_pos - width, beta_true, width, label="True", color="#2196F3", alpha=0.8)
axes[0].bar(
    x_pos, beta_ridge, width, label="Ridge (sklearn)", color="#FF9800", alpha=0.8
)
axes[0].bar(
    x_pos + width, beta_map, width, label="Bayesian MAP", color="#4CAF50", alpha=0.8
)
axes[0].set_xlabel("Media Channel")
axes[0].set_ylabel("Coefficient Value")
axes[0].set_title("Ridge vs Bayesian MAP: Identical Estimates")
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(channels)
axes[0].legend()
axes[0].grid(axis="y", alpha=0.3)

# Panel B: Scatter plot Ridge vs MAP (should be perfect diagonal)
axes[1].scatter(beta_ridge, beta_map, s=100, c="#E91E63", zorder=3)
lims = [
    min(beta_ridge.min(), beta_map.min()) - 0.2,
    max(beta_ridge.max(), beta_map.max()) + 0.2,
]
axes[1].plot(lims, lims, "k--", alpha=0.5, label="y = x")
axes[1].set_xlabel("Ridge Coefficients")
axes[1].set_ylabel("Bayesian MAP Coefficients")
axes[1].set_title(f"Perfect Equivalence (max diff = {max_diff:.2e})")
axes[1].set_xlim(lims)
axes[1].set_ylim(lims)
axes[1].set_aspect("equal")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
finalize_figure("01_ridge_bayesian_equivalence.png", fig=fig)

# ── 6. Show equivalence across different lambda values ─────────────────────
print("\n" + "=" * 70)
print("EQUIVALENCE ACROSS REGULARIZATION STRENGTHS")
print("=" * 70)

lambdas = [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
print(f"\n{'lambda':>8} {'tau^2':>10} {'Max |Ridge - MAP|':>20}")
print("-" * 40)

for lam in lambdas:
    ridge_l = Ridge(alpha=lam, fit_intercept=False).fit(X_c, y_c)
    map_l = np.linalg.solve(XtX + lam * I, Xty)
    diff = np.max(np.abs(ridge_l.coef_ - map_l))
    tau2 = sigma_true**2 / lam
    print(f"{lam:>8.2f} {tau2:>10.4f} {diff:>20.2e}")

print()
print("KEY TAKEAWAY:")
print("  Ridge regression (frequentist) and Bayesian MAP with a Gaussian prior")
print("  produce IDENTICAL coefficient estimates. The lambda parameter in Ridge")
print("  maps directly to the prior variance: tau^2 = sigma^2 / lambda.")
print("  The difference is what happens AFTER the point estimate:")
print("  - Frequentists stop here (+ bootstrap CIs)")
print("  - Bayesians compute the full posterior distribution")
