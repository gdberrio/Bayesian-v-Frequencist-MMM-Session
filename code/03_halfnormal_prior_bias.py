"""
03 - HalfNormal Prior Creates Systematic Upward Bias
=====================================================
Research Point (Section 3 / report 2.2):
  "If the true effect of a channel is zero, a HalfNormal prior creates
   systematic upward bias. The model literally cannot detect that a
   channel is wasting money."

  "The posterior mean of HalfNormal(sigma) is always positive:
   sigma * sqrt(2/pi)."

  "In a model with 8-10 media channels, some are almost certainly
   ineffective. A model that cannot detect ineffectiveness has limited
   value for budget optimization."

This script demonstrates:
  1. HalfNormal prior properties and systematic positive bias
  2. Bayesian estimation with HalfNormal prior when true effect = 0
  3. Compare Normal (allows negative) vs HalfNormal (forces positive) priors
  4. Show how HalfNormal prevents detection of ineffective channels
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from demo_utils import finalize_figure

np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════
# PART 1: HALFNORMAL PRIOR PROPERTIES
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 1: HalfNormal Prior -- Always Positive")
print("=" * 70)

sigmas = [1, 2, 5]
x = np.linspace(0, 15, 500)

print("\nHalfNormal(sigma) properties:")
print(f"{'sigma':>6} {'E[X]':>8} {'Median':>8} {'Mode':>6} {'P(X<0)':>8}")
print("-" * 40)
for s in sigmas:
    mean = s * np.sqrt(2 / np.pi)
    median = s * stats.norm.ppf(0.75)  # median of HalfNormal
    print(f"{s:>6} {mean:>8.3f} {median:>8.3f} {'0':>6} {'0%':>8}")

print()
print("  -> The posterior mean of a HalfNormal is ALWAYS positive.")
print("     A channel with zero true effect will be estimated as positive.")

# ══════════════════════════════════════════════════════════════════════════
# PART 2: BAYESIAN UPDATING WITH HALFNORMAL vs NORMAL PRIOR
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 2: Bayesian Estimation When True Effect = 0")
print("=" * 70)


def compute_posterior_grid(prior_pdf, likelihood_pdf, grid):
    """Compute posterior on a grid via Bayes' rule."""
    unnormalized = prior_pdf * likelihood_pdf
    dx = grid[1] - grid[0]
    normalizing_constant = np.trapz(unnormalized, grid)
    if normalizing_constant > 0:
        return unnormalized / normalizing_constant
    return unnormalized


# Simulate: true beta = 0, but we observe noisy data
n_obs = 150
sigma_noise = 5.0
beta_true = 0.0

# Simplified: 1 channel, X ~ Exp(50)
X = np.random.exponential(50, n_obs)
y = beta_true * X + 100 + np.random.normal(0, sigma_noise, n_obs)

# OLS estimate
X_c = X - X.mean()
y_c = y - y.mean()
beta_ols = np.sum(X_c * y_c) / np.sum(X_c**2)
se_ols = sigma_noise / np.sqrt(np.sum(X_c**2))

print(f"\nTrue effect: beta = {beta_true}")
print(f"OLS estimate: beta_hat = {beta_ols:.4f} (SE = {se_ols:.4f})")

# Grid for posterior computation
grid = np.linspace(-1.0, 1.0, 5000)

# Likelihood (normal centered at OLS estimate)
likelihood = stats.norm.pdf(grid, loc=beta_ols, scale=se_ols)

# Use a moderately informative prior so the positivity constraint is visible
prior_sigma_example = 0.2

# Prior A: Normal(0, sigma=0.2) -- allows negative
prior_normal = stats.norm.pdf(grid, loc=0, scale=prior_sigma_example)

# Prior B: HalfNormal(sigma=0.2) -- forces positive
# HalfNormal pdf: (2/sigma) * phi(x/sigma) for x >= 0
prior_halfnormal = np.where(
    grid >= 0, 2 * stats.norm.pdf(grid, loc=0, scale=prior_sigma_example), 0.0
)

# Compute posteriors
posterior_normal = compute_posterior_grid(prior_normal, likelihood, grid)
posterior_halfnormal = compute_posterior_grid(prior_halfnormal, likelihood, grid)

# Posterior means
dx = grid[1] - grid[0]
mean_normal = np.trapz(grid * posterior_normal, grid)
mean_halfnormal = np.trapz(grid * posterior_halfnormal, grid)

print(f"\nPosterior means (true beta = 0):")
print(f"  Normal prior:     E[beta|y] = {mean_normal:.4f}")
print(f"  HalfNormal prior: E[beta|y] = {mean_halfnormal:.4f}")
print(f"  Bias (HalfNormal): {mean_halfnormal - beta_true:.4f}")
print(f"  Bias (Normal):     {mean_normal - beta_true:.4f}")
print()
print("  -> HalfNormal prior produces LARGER positive bias when true effect = 0")

# ══════════════════════════════════════════════════════════════════════════
# PART 3: SIMULATION -- INEFFECTIVE CHANNELS ALWAYS LOOK EFFECTIVE
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 3: 8-Channel Model -- HalfNormal Cannot Detect Ineffective Channels")
print("=" * 70)

n_channels = 8
n_sims = 500
sigma_prior = 2.0

# TRUE effects: channels 6, 7, 8 have ZERO effect (wasting money!)
true_betas = np.array([2.5, 1.8, 1.2, 0.8, 0.4, 0.0, 0.0, 0.0])
channel_names = [f"Ch{i + 1}" for i in range(n_channels)]

estimates_hn = np.zeros((n_sims, n_channels))  # HalfNormal prior
estimates_n = np.zeros((n_sims, n_channels))  # Normal prior
estimates_ols = np.zeros((n_sims, n_channels))  # No prior (OLS)

for sim in range(n_sims):
    X = np.random.exponential(scale=50, size=(n_obs, n_channels))
    noise = np.random.normal(0, sigma_noise, n_obs)
    y = X @ true_betas + 100 + noise

    X_c = X - X.mean(axis=0)
    y_c = y - y.mean()

    for ch in range(n_channels):
        # OLS estimate for this channel (simplified univariate)
        beta_hat = np.sum(X_c[:, ch] * y_c) / np.sum(X_c[:, ch] ** 2)
        se = sigma_noise / np.sqrt(np.sum(X_c[:, ch] ** 2))

        estimates_ols[sim, ch] = beta_hat

        # Bayesian posterior mean with Normal prior
        prior_prec = 1.0 / sigma_prior**2
        data_prec = 1.0 / se**2
        post_mean_n = (prior_prec * 0 + data_prec * beta_hat) / (prior_prec + data_prec)
        estimates_n[sim, ch] = post_mean_n

        # HalfNormal: approximate posterior mean using grid
        g = np.linspace(-0.5, 5.0, 2000)
        lik = stats.norm.pdf(g, loc=beta_hat, scale=se)
        pr_hn = np.where(g >= 0, 2 * stats.norm.pdf(g, 0, sigma_prior), 0)
        post_hn = pr_hn * lik
        post_hn /= np.trapz(post_hn, g) + 1e-30
        estimates_hn[sim, ch] = np.trapz(g * post_hn, g)

print(f"\nTrue effects: {true_betas}")
print(f"Channels 6-8 have ZERO effect (wasting money!)")
print()
print(
    f"{'Channel':<8} {'True':>6} {'OLS mean':>10} {'Normal':>10} {'HalfNormal':>12} {'HN detects 0?':>14}"
)
print("-" * 62)
for ch in range(n_channels):
    ols_mean = estimates_ols[:, ch].mean()
    n_mean = estimates_n[:, ch].mean()
    hn_mean = estimates_hn[:, ch].mean()
    detects_zero = "NO" if hn_mean > 0.1 else "Maybe"
    print(
        f"{'Ch' + str(ch + 1):<8} {true_betas[ch]:>6.1f} {ols_mean:>10.3f} {n_mean:>10.3f} {hn_mean:>12.3f} {detects_zero:>14}"
    )

print()
print("  -> HalfNormal prior ALWAYS estimates positive effect, even for")
print("     channels with zero true effect. Budget optimizer would continue")
print("     allocating money to these wasted channels.")


# ══════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: HalfNormal prior shapes
for s, color in zip([1, 2, 5], ["#2196F3", "#FF9800", "#4CAF50"]):
    xp = np.linspace(0, 15, 500)
    axes[0, 0].plot(
        xp,
        stats.halfnorm.pdf(xp, scale=s),
        color=color,
        linewidth=2,
        label=f"HalfNormal(sigma={s})",
    )
    axes[0, 0].axvline(s * np.sqrt(2 / np.pi), color=color, linestyle=":", alpha=0.7)
axes[0, 0].axvline(
    0, color="red", linestyle="--", linewidth=2, alpha=0.5, label="True beta = 0"
)
axes[0, 0].set_xlabel("beta")
axes[0, 0].set_ylabel("Density")
axes[0, 0].set_title("HalfNormal Priors (dotted = E[beta])\nTrue beta = 0 shown in red")
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(alpha=0.3)

# Panel B: Normal vs HalfNormal posterior when true beta = 0
axes[0, 1].fill_between(
    grid, posterior_normal, alpha=0.4, color="#2196F3", label="Posterior (Normal prior)"
)
axes[0, 1].fill_between(
    grid,
    posterior_halfnormal,
    alpha=0.4,
    color="#FF9800",
    label="Posterior (HalfNormal prior)",
)
axes[0, 1].axvline(0, color="red", linestyle="--", linewidth=2, label="True beta = 0")
axes[0, 1].axvline(mean_normal, color="#2196F3", linestyle=":", linewidth=2)
axes[0, 1].axvline(mean_halfnormal, color="#FF9800", linestyle=":", linewidth=2)
axes[0, 1].set_xlabel("beta")
axes[0, 1].set_ylabel("Posterior Density")
axes[0, 1].set_title(
    f"Posteriors When True beta = 0\nNormal mean={mean_normal:.3f}, HalfNormal mean={mean_halfnormal:.3f}"
)
axes[0, 1].legend(fontsize=8)
axes[0, 1].set_xlim(-0.5, 0.5)
axes[0, 1].grid(alpha=0.3)

# Panel C: Estimated coefficients across channels
x_pos = np.arange(n_channels)
width = 0.25
axes[1, 0].bar(
    x_pos - width, true_betas, width, label="True", color="#9E9E9E", alpha=0.9
)
axes[1, 0].bar(
    x_pos,
    estimates_n.mean(axis=0),
    width,
    label="Normal prior",
    color="#2196F3",
    alpha=0.8,
    yerr=estimates_n.std(axis=0),
    capsize=3,
)
axes[1, 0].bar(
    x_pos + width,
    estimates_hn.mean(axis=0),
    width,
    label="HalfNormal prior",
    color="#FF9800",
    alpha=0.8,
    yerr=estimates_hn.std(axis=0),
    capsize=3,
)
axes[1, 0].axhline(0, color="black", linestyle="-", linewidth=0.5)
# Highlight ineffective channels
for ch in [5, 6, 7]:
    axes[1, 0].axvspan(ch - 0.4, ch + 0.4, alpha=0.1, color="red")
axes[1, 0].set_xlabel("Channel")
axes[1, 0].set_ylabel("Estimated Coefficient")
axes[1, 0].set_title(
    "8-Channel Model: HalfNormal Cannot Detect Zero Effects\n(Red shading = truly ineffective channels)"
)
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(channel_names)
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(axis="y", alpha=0.3)

# Panel D: Distribution of estimates for ineffective channel (Ch 6)
axes[1, 1].hist(
    estimates_ols[:, 5],
    bins=30,
    alpha=0.5,
    color="#9E9E9E",
    label=f"OLS (mean={estimates_ols[:, 5].mean():.3f})",
    density=True,
)
axes[1, 1].hist(
    estimates_n[:, 5],
    bins=30,
    alpha=0.5,
    color="#2196F3",
    label=f"Normal prior (mean={estimates_n[:, 5].mean():.3f})",
    density=True,
)
axes[1, 1].hist(
    estimates_hn[:, 5],
    bins=30,
    alpha=0.5,
    color="#FF9800",
    label=f"HalfNormal prior (mean={estimates_hn[:, 5].mean():.3f})",
    density=True,
)
axes[1, 1].axvline(0, color="red", linestyle="--", linewidth=2, label="True beta = 0")
axes[1, 1].set_xlabel("Estimated Coefficient for Ch6 (true = 0)")
axes[1, 1].set_ylabel("Density")
axes[1, 1].set_title(
    "Channel 6 (Ineffective): Estimate Distributions\nHalfNormal is ALWAYS positive"
)
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=0.3)

plt.suptitle(
    "HalfNormal Prior Creates Systematic Upward Bias",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
finalize_figure("03_halfnormal_prior_bias.png", fig=fig)

print("\nKEY TAKEAWAY:")
print("  HalfNormal priors (used by Meridian and PyMC-Marketing) make it")
print("  IMPOSSIBLE for the model to detect that a channel has zero or negative")
print("  effect. In a portfolio of 8-10 channels, some are almost certainly")
print("  ineffective. A model that cannot detect this has limited value for")
print("  budget optimization. This is a legitimate frequentist critique.")
