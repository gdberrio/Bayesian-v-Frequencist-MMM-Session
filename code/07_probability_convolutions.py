"""
07 - Probability Convolutions and Parameter Non-Identifiability
================================================================
Research Point (Section 3 Part 3 / Section 4):
  "In a Bayesian MMM, the media contribution is:
   contribution = beta * Hill(Adstock(x; alpha); ec, slope)
   This is a PRODUCT OF RANDOM VARIABLES. Products of random variables
   create skewed, heavy-tailed distributions."

  "Non-identifiability between beta and saturation parameters: if you
   double beta and halve the Hill function's output (by adjusting ec/slope),
   you get similar fit. This creates RIDGES in the posterior."

  Meridian handles this by:
  - Fixing slope=1 (cannot be identified with ec)
  - Using informative priors on ec
  - ROI reparameterization (the most important technical innovation)

  Dew et al. (2024): "nonlinear and time-varying effects are often
  not identifiable from standard marketing mix data."

This script demonstrates:
  1. Products of random variables create unexpected distributions
  2. Beta-saturation non-identifiability (the ridge in parameter space)
  3. Why fixing slope=1 is necessary
  4. How ROI reparameterization helps
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from demo_utils import finalize_figure

np.random.seed(42)


def hill_function(x, ec, slope):
    """Hill saturation: x^slope / (ec^slope + x^slope)"""
    return x**slope / (ec**slope + x**slope)


def best_matching_ec(target_curve, spend_levels, beta, slope):
    """Find the ec that best matches a target contribution curve."""
    ec_grid = np.linspace(10, 150, 2000)
    rmses = np.empty_like(ec_grid)
    for idx, ec in enumerate(ec_grid):
        candidate = beta * hill_function(spend_levels, ec, slope)
        rmses[idx] = np.sqrt(np.mean((candidate - target_curve) ** 2))

    best_idx = np.argmin(rmses)
    best_ec = ec_grid[best_idx]
    best_curve = beta * hill_function(spend_levels, best_ec, slope)
    return best_ec, best_curve, rmses[best_idx]


# ══════════════════════════════════════════════════════════════════════════
# PART 1: PRODUCTS OF RANDOM VARIABLES
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 1: Products of Random Variables Create Heavy Tails")
print("=" * 70)

n_samples = 100_000

# beta ~ HalfNormal(2)
beta_samples = np.abs(np.random.normal(0, 2, n_samples))

# Hill output with random ec ~ Beta(1,1) mapped to [20, 100]
# and random slope
ec_samples = 20 + 80 * np.random.beta(1, 1, n_samples)

# Fixed spend level for illustration
spend_fixed = 50.0

hill_samples = hill_function(spend_fixed, ec_samples, slope=1.0)

# Contribution = beta * Hill(spend)
contribution = beta_samples * hill_samples

# Properties of the product
print(f"\nbeta ~ HalfNormal(sigma=2):")
print(
    f"  mean={beta_samples.mean():.3f}, std={beta_samples.std():.3f}, skew={stats.skew(beta_samples):.3f}"
)
print(f"\nHill(spend=50; ec~Uniform(20,100), slope=1):")
print(
    f"  mean={hill_samples.mean():.3f}, std={hill_samples.std():.3f}, skew={stats.skew(hill_samples):.3f}"
)
print(f"\nContribution = beta * Hill (product):")
print(
    f"  mean={contribution.mean():.3f}, std={contribution.std():.3f}, skew={stats.skew(contribution):.3f}"
)
print()

# Variance of product: Var(XY) = Var(X)Var(Y) + Var(X)E[Y]^2 + Var(Y)E[X]^2
var_product_formula = (
    beta_samples.var() * hill_samples.var()
    + beta_samples.var() * hill_samples.mean() ** 2
    + hill_samples.var() * beta_samples.mean() ** 2
)
print(f"Var(beta*Hill) formula:  {var_product_formula:.3f}")
print(f"Var(beta*Hill) empirical: {contribution.var():.3f}")
print()
print("  -> Product is MORE skewed and heavier-tailed than either marginal.")
print("     The 'uncertainty' in contribution is amplified by the convolution.")


# ══════════════════════════════════════════════════════════════════════════
# PART 2: BETA-SATURATION NON-IDENTIFIABILITY
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 2: Beta-Saturation Non-Identifiability")
print("=" * 70)

spend_levels = np.linspace(1, 150, 200)

# Show: many (beta, ec, slope) combinations produce the same contribution
target_contribution = 2.0 * hill_function(spend_levels, ec=60, slope=2.0)

# Search for alternative parameters that genuinely track the same curve
candidate_specs = [
    ("Original", 2.0, 2.0),
    ("Lower beta", 1.7, 2.0),
    ("Higher beta", 2.3, 2.0),
    ("Lower slope", 2.1, 1.7),
    ("Higher slope", 1.9, 2.4),
]
configs = []
for name, beta, slope in candidate_specs:
    if name == "Original":
        configs.append((beta, 60.0, slope, name, target_contribution, 0.0))
    else:
        ec, curve, rmse = best_matching_ec(target_contribution, spend_levels, beta, slope)
        configs.append((beta, ec, slope, name, curve, rmse))

print(f"\nTarget: beta=2.0, ec=60, slope=2.0")
print("All configs below are searched to best-match the same target curve:\n")
print(
    f"{'Config':<30} {'beta':>6} {'ec':>6} {'slope':>6} {'Total Contrib':>14} {'RMSE vs target':>15}"
)
print("-" * 80)

for beta, ec, slope, name, contrib, rmse in configs:
    total = contrib.sum()
    print(
        f"{name:<30} {beta:>6.1f} {ec:>6.0f} {slope:>6.1f} {total:>14.1f} {rmse:>15.3f}"
    )

print()
print("  -> Multiple parameter combinations produce similar model output.")
print("     The posterior lives on a 'ridge' in parameter space where the")
print("     prior (not the data) determines which combo is selected.")


# ══════════════════════════════════════════════════════════════════════════
# PART 3: THE RIDGE IN PARAMETER SPACE
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 3: The Ridge (Contour) in Parameter Space")
print("=" * 70)

# For fixed slope=2 and fixed spend data, show the iso-contribution
# contours in (beta, ec) space
beta_range = np.linspace(0.5, 6.0, 100)
ec_range = np.linspace(20, 120, 100)
beta_grid, ec_grid = np.meshgrid(beta_range, ec_range)

# Average contribution across spend levels
spend_data = np.random.exponential(50, 150)
total_contrib_grid = np.zeros_like(beta_grid)

for i in range(len(ec_range)):
    for j in range(len(beta_range)):
        contrib = beta_grid[i, j] * hill_function(spend_data, ec_grid[i, j], slope=2.0)
        total_contrib_grid[i, j] = contrib.sum()

# The target total contribution
target_total = 2.0 * hill_function(spend_data, 60, 2.0).sum()

# Also show the (pseudo) log-likelihood surface
# RSS is similar along the ridge, different perpendicular to it
y_data = 100 + 2.0 * hill_function(spend_data, 60, 2.0) + np.random.normal(0, 2, 150)

ll_grid = np.zeros_like(beta_grid)
for i in range(len(ec_range)):
    for j in range(len(beta_range)):
        predicted = 100 + beta_grid[i, j] * hill_function(
            spend_data, ec_grid[i, j], slope=2.0
        )
        ll_grid[i, j] = -np.sum((y_data - predicted) ** 2)

print(f"  Target total contribution: {target_total:.1f}")
print(f"  Many (beta, ec) pairs achieve similar total contribution.")
print(f"  The likelihood surface has a ridge -- data cannot distinguish them.")


# ══════════════════════════════════════════════════════════════════════════
# PART 4: WHY MERIDIAN FIXES SLOPE=1
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 4: Why Meridian Fixes slope=1")
print("=" * 70)

# Show non-identifiability between ec and slope
slope_range = np.linspace(0.5, 5.0, 100)
ec_for_equiv = np.zeros_like(slope_range)

# For each slope, find ec that gives same Hill(50)
target_hill = hill_function(50, 60, 2.0)
for i, s in enumerate(slope_range):
    # Solve: 50^s / (ec^s + 50^s) = target_hill
    # ec^s = 50^s * (1 - target_hill) / target_hill
    ec_s = 50**s * (1 - target_hill) / target_hill
    ec_for_equiv[i] = ec_s ** (1.0 / s)

print(f"\nTarget Hill(50, ec=60, slope=2) = {target_hill:.4f}")
print(f"\nEquivalent (ec, slope) pairs that give the same Hill output:")
print(f"{'slope':>8} {'ec':>8}")
print("-" * 18)
for s, e in zip(
    [0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
    [
        ec_for_equiv[np.argmin(np.abs(slope_range - s))]
        for s in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    ],
):
    print(f"{s:>8.1f} {e:>8.1f}")

print()
print("  -> ec and slope are NON-IDENTIFIABLE. Meridian's documentation:")
print('     "Difficult to learn because of identifiability reasons."')
print("     Solution: fix slope=1 (a Deterministic parameter).")
print("     This is honest about the limitation but also a strong constraint.")


# ══════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Product distribution vs marginals
axes[0, 0].hist(
    beta_samples,
    bins=50,
    alpha=0.4,
    color="#2196F3",
    density=True,
    label="beta ~ HalfNormal(2)",
)
axes[0, 0].hist(
    hill_samples,
    bins=50,
    alpha=0.4,
    color="#FF9800",
    density=True,
    label="Hill(50; ec, slope=1)",
)
axes[0, 0].hist(
    contribution,
    bins=100,
    alpha=0.4,
    color="#E91E63",
    density=True,
    label="beta * Hill (product)",
)
axes[0, 0].set_xlabel("Value")
axes[0, 0].set_ylabel("Density")
axes[0, 0].set_title(
    "Products of Random Variables\nCreate Skewed, Heavy-Tailed Distributions"
)
axes[0, 0].legend(fontsize=8)
axes[0, 0].set_xlim(0, 6)
axes[0, 0].grid(alpha=0.3)

# Panel B: Multiple equivalent parameter configurations
for beta, ec, slope, name, contrib, rmse in configs:
    axes[0, 1].plot(
        spend_levels, contrib, linewidth=2, label=f"{name} (RMSE={rmse:.02f})", alpha=0.8
    )
axes[0, 1].set_xlabel("Spend")
axes[0, 1].set_ylabel("Contribution (beta * Hill(spend))")
axes[0, 1].set_title(
    "Non-Identifiability: Multiple Configs, Similar Output\n'The prior determines which combo is selected'"
)
axes[0, 1].legend(fontsize=7)
axes[0, 1].grid(alpha=0.3)

# Panel C: Ridge in parameter space (likelihood contour)
contour = axes[1, 0].contourf(beta_grid, ec_grid, ll_grid, levels=30, cmap="RdYlBu_r")
plt.colorbar(contour, ax=axes[1, 0], label="Log-likelihood (approx)")
# Show the ridge
iso = axes[1, 0].contour(
    beta_grid,
    ec_grid,
    total_contrib_grid,
    levels=[target_total * 0.95, target_total, target_total * 1.05],
    colors="white",
    linewidths=2,
    linestyles="--",
)
axes[1, 0].clabel(iso, inline=True, fontsize=8, fmt="%.0f")
axes[1, 0].scatter(
    [2.0], [60], c="white", s=100, marker="*", zorder=5, label="True params"
)
axes[1, 0].set_xlabel("beta")
axes[1, 0].set_ylabel("ec (Half-saturation)")
axes[1, 0].set_title(
    "Likelihood Ridge in (beta, ec) Space\nData cannot distinguish along the ridge"
)
axes[1, 0].legend(fontsize=8)

# Panel D: ec-slope non-identifiability
axes[1, 1].plot(slope_range, ec_for_equiv, "b-", linewidth=2)
axes[1, 1].scatter(
    [2.0], [60], c="red", s=100, zorder=5, marker="*", label="True (slope=2, ec=60)"
)
axes[1, 1].scatter(
    [1.0],
    [ec_for_equiv[np.argmin(np.abs(slope_range - 1.0))]],
    c="green",
    s=100,
    zorder=5,
    marker="s",
    label="Meridian default (slope=1)",
)
axes[1, 1].fill_between(
    slope_range, ec_for_equiv - 5, ec_for_equiv + 5, alpha=0.2, color="blue"
)
axes[1, 1].set_xlabel("slope")
axes[1, 1].set_ylabel("ec (Half-saturation)")
axes[1, 1].set_title(
    "ec-slope Non-Identifiability\nMany pairs give the same output at the current spend level"
)
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=0.3)

plt.suptitle(
    "Probability Convolutions and Parameter Non-Identifiability",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
finalize_figure("07_probability_convolutions.png", fig=fig)

print("\nKEY TAKEAWAY:")
print("  In Bayesian MMM, contributions are products of random variables")
print("  passed through nonlinear transformations. This creates:")
print("  1. Heavy-tailed, skewed posterior distributions")
print("  2. Non-identifiability ridges where prior dominates")
print("  3. The need to fix parameters (slope=1) or reparameterize (ROI priors)")
print("  The 'uncertainty' being reported is largely determined by prior choices")
print("  on non-identified parameters, not by the data.")
