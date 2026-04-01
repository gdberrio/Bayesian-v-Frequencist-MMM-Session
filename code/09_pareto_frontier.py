"""
09 - Pareto Frontier: Robyn-Style Multi-Objective Model Selection
==================================================================
Research Point (Section 11 / report Section 4):
  Robyn's multi-objective optimization is "genuinely innovative":
    1. NRMSE: How well does the model predict? (Lower = better)
    2. DECOMP.RSSD: How business-plausible is the decomposition? (Lower = better)
    3. MAPE.LIFT (optional): How well does the model match experiments?

  "This generates a Pareto frontier of models that represent different
   tradeoffs. A model with perfect prediction might have implausible
   decompositions. A model with perfect business alignment might have
   poor prediction."

  "Robyn generates thousands of Pareto-optimal models and presents the
   range of estimates across them. This is arguably MORE HONEST for
   business communication: 'Here are several plausible models. TV's
   attributed effect share ranges from X% to Y% across them.'"

This script demonstrates:
  1. Generate models with varying hyperparameters
  2. Compute NRMSE and DECOMP.RSSD for each
  3. Find the Pareto frontier
  4. Show model-level uncertainty across Pareto-optimal solutions
"""

import numpy as np
import matplotlib.pyplot as plt

from demo_utils import finalize_figure

np.random.seed(42)


def hill_function(x, ec, slope):
    """Hill saturation function."""
    return x**slope / (ec**slope + x**slope)


def geometric_adstock(x, decay):
    """Geometric adstock."""
    result = np.zeros_like(x, dtype=float)
    result[0] = x[0]
    for t in range(1, len(x)):
        result[t] = x[t] + decay * result[t - 1]
    return result


def compute_nrmse(y_true, y_pred):
    """Normalized RMSE."""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse / y_true.std()


def compute_decomp_rssd(effect_share, spend_share):
    """Root sum squared distance between effect share and spend share."""
    return np.sqrt(np.sum((effect_share - spend_share) ** 2))


def is_pareto_optimal(costs):
    """Find Pareto-optimal points (minimize all objectives)."""
    n = costs.shape[0]
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j:
                if np.all(costs[j] <= costs[i]) and np.any(costs[j] < costs[i]):
                    is_efficient[i] = False
                    break
    return is_efficient


# ══════════════════════════════════════════════════════════════════════════
# GENERATE SYNTHETIC MMM DATA
# ══════════════════════════════════════════════════════════════════════════

n_weeks = 156  # 3 years
n_channels = 4
channel_names = ["TV", "Digital", "Radio", "OOH"]

# True parameters
true_betas = np.array([2.5, 1.8, 0.8, 0.5])
true_decays = np.array([0.7, 0.3, 0.5, 0.2])
true_ecs = np.array([50, 30, 40, 25])
true_slopes = np.array([2.0, 1.5, 2.5, 1.0])

# Generate spend data
spend = np.column_stack(
    [
        50
        + 20 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)
        + np.random.normal(0, 10, n_weeks),
        30
        + 10 * np.sin(2 * np.pi * np.arange(n_weeks) / 52 + 1)
        + np.random.normal(0, 8, n_weeks),
        20
        + 5 * np.sin(2 * np.pi * np.arange(n_weeks) / 52 + 2)
        + np.random.normal(0, 5, n_weeks),
        15
        + 3 * np.sin(2 * np.pi * np.arange(n_weeks) / 52 + 0.5)
        + np.random.normal(0, 4, n_weeks),
    ]
)
spend = np.maximum(spend, 1)

# Compute true contributions
true_contributions = np.zeros((n_weeks, n_channels))
for ch in range(n_channels):
    adstocked = geometric_adstock(spend[:, ch], true_decays[ch])
    saturated = hill_function(adstocked, true_ecs[ch], true_slopes[ch])
    true_contributions[:, ch] = true_betas[ch] * saturated

# Generate target
baseline = 200 + 0.5 * np.arange(n_weeks)  # trend
seasonality = 20 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)
noise = np.random.normal(0, 10, n_weeks)
y = baseline + seasonality + true_contributions.sum(axis=1) + noise

# Spend shares
total_spend = spend.sum(axis=0)
spend_share = total_spend / total_spend.sum()


# ══════════════════════════════════════════════════════════════════════════
# GENERATE MANY MODELS (varying hyperparameters)
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("Generating 2000 models with varying hyperparameters...")
print("=" * 70)

n_models = 2000
model_nrmse = np.zeros(n_models)
model_rssd = np.zeros(n_models)
model_effect_share = np.zeros((n_models, n_channels))
model_contributions = np.zeros((n_models, n_channels))

for m in range(n_models):
    # Random hyperparameters
    decays = np.clip(true_decays + np.random.normal(0, 0.2, n_channels), 0.01, 0.99)
    ecs = np.clip(true_ecs + np.random.normal(0, 15, n_channels), 5, 150)
    slopes = np.clip(true_slopes + np.random.normal(0, 0.8, n_channels), 0.3, 5.0)

    # Transform media
    X_transformed = np.zeros((n_weeks, n_channels))
    for ch in range(n_channels):
        adstocked = geometric_adstock(spend[:, ch], decays[ch])
        X_transformed[:, ch] = hill_function(adstocked, ecs[ch], slopes[ch])

    # Add baseline features
    X_full = np.column_stack(
        [
            np.ones(n_weeks),
            np.arange(n_weeks),
            np.sin(2 * np.pi * np.arange(n_weeks) / 52),
            np.cos(2 * np.pi * np.arange(n_weeks) / 52),
            X_transformed,
        ]
    )

    # Ridge regression (like Robyn)
    lambda_ridge = np.random.uniform(1, 50)
    XtX = X_full.T @ X_full
    I = np.eye(XtX.shape[0])
    I[0, 0] = 0  # don't penalize intercept
    betas = np.linalg.solve(XtX + lambda_ridge * I, X_full.T @ y)

    y_pred = X_full @ betas
    media_betas = betas[4:]

    # Compute metrics
    model_nrmse[m] = compute_nrmse(y, y_pred)

    # Effect share (from media contributions)
    contributions = np.zeros(n_channels)
    for ch in range(n_channels):
        contributions[ch] = max(0, media_betas[ch]) * X_transformed[:, ch].sum()

    total_effect = contributions.sum()
    if total_effect > 0:
        effect_share = contributions / total_effect
    else:
        effect_share = np.ones(n_channels) / n_channels

    model_rssd[m] = compute_decomp_rssd(effect_share, spend_share)
    model_effect_share[m] = effect_share
    model_contributions[m] = contributions


# ══════════════════════════════════════════════════════════════════════════
# FIND PARETO FRONTIER
# ══════════════════════════════════════════════════════════════════════════

costs = np.column_stack([model_nrmse, model_rssd])
pareto_mask = is_pareto_optimal(costs)
n_pareto = pareto_mask.sum()

print(f"\nTotal models: {n_models}")
print(f"Pareto-optimal models: {n_pareto}")

# Sort pareto models by NRMSE
pareto_idx = np.where(pareto_mask)[0]
pareto_sorted = pareto_idx[np.argsort(model_nrmse[pareto_idx])]


# ══════════════════════════════════════════════════════════════════════════
# REPORT PARETO MODEL EFFECT-SHARE RANGES
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Effect-Share Ranges Across Pareto-Optimal Models")
print("=" * 70)

print(
    f"\n{'Channel':<10} {'Min Share':>10} {'Median':>10} {'Max Share':>10} {'Range':>10}"
)
print("-" * 50)
for ch in range(n_channels):
    effect_pareto = model_effect_share[pareto_mask, ch]
    if len(effect_pareto) > 0:
        print(
            f"{channel_names[ch]:<10} {np.min(effect_pareto) * 100:>9.1f}% "
            f"{np.median(effect_pareto) * 100:>9.1f}% {np.max(effect_pareto) * 100:>9.1f}% "
            f"{(np.max(effect_pareto) - np.min(effect_pareto)) * 100:>8.1f}pp"
        )

print()
print("  -> 'Here are several plausible models that all fit reasonably well.")
print("     TV's attributed effect share ranges from X% to Y% across them.'")
print("     This communicates uncertainty without making probability claims")
print("     that depend on prior choices.")


# ══════════════════════════════════════════════════════════════════════════
# FRONTIER ENDPOINTS: PREDICTION-FIRST vs PLAUSIBILITY-FIRST
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Two Legitimate Frontier Endpoints")
print("=" * 70)

best_nrmse_idx = pareto_sorted[0]
best_rssd_idx = pareto_sorted[-1]
for label, idx in [
    ("Prediction-first point", best_nrmse_idx),
    ("Plausibility-first point", best_rssd_idx),
]:
    shares = model_effect_share[idx] * 100
    print(f"\n  {label}:")
    print(f"    NRMSE = {model_nrmse[idx]:.3f}")
    print(f"    DECOMP.RSSD = {model_rssd[idx]:.3f}")
    print(
        "    Effect shares = "
        + ", ".join(
            f"{channel_names[ch]} {shares[ch]:.1f}%" for ch in range(n_channels)
        )
    )


# ══════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Pareto frontier
axes[0, 0].scatter(
    model_nrmse[~pareto_mask],
    model_rssd[~pareto_mask],
    alpha=0.15,
    s=10,
    c="#9E9E9E",
    label="Non-Pareto models",
)
axes[0, 0].scatter(
    model_nrmse[pareto_mask],
    model_rssd[pareto_mask],
    alpha=0.8,
    s=30,
    c="#E91E63",
    label=f"Pareto-optimal ({n_pareto})",
)
# Connect Pareto points
pareto_nrmse = model_nrmse[pareto_sorted]
pareto_rssd = model_rssd[pareto_sorted]
axes[0, 0].plot(pareto_nrmse, pareto_rssd, "r-", linewidth=1.5, alpha=0.5)
axes[0, 0].set_xlabel("NRMSE (prediction accuracy)")
axes[0, 0].set_ylabel("DECOMP.RSSD (business plausibility)")
axes[0, 0].set_title(
    "Pareto Frontier: Multi-Objective Model Selection\n(Robyn's most important innovation)"
)
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(alpha=0.3)
axes[0, 0].annotate(
    "Better prediction -->",
    xy=(0.02, 0.02),
    xycoords="axes fraction",
    fontsize=8,
    color="gray",
)
axes[0, 0].annotate(
    "More plausible -->",
    xy=(0.02, 0.06),
    xycoords="axes fraction",
    fontsize=8,
    color="gray",
    rotation=0,
)

# Panel B: ROAS distributions across Pareto models
for ch, color in zip(range(n_channels), ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]):
    effect_pareto = model_effect_share[pareto_mask, ch] * 100
    if len(effect_pareto) > 0:
        axes[0, 1].hist(
            effect_pareto,
            bins=20,
            alpha=0.5,
            color=color,
            label=f"{channel_names[ch]} (range={effect_pareto.max() - effect_pareto.min():.1f}pp)",
        )

axes[0, 1].set_xlabel("Attributed Effect Share (%)")
axes[0, 1].set_ylabel("Count (Pareto models)")
axes[0, 1].set_title(
    "Effect-Share Ranges Across Pareto-Optimal Models\n'Model-level uncertainty without probability claims'"
)
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3)

# Panel C: Tradeoff illustration -- two specific models
contributions_best_pred = model_effect_share[best_nrmse_idx] * 100
contributions_best_plaus = model_effect_share[best_rssd_idx] * 100

x_pos = np.arange(n_channels)
width = 0.35
axes[1, 0].bar(
    x_pos - width / 2,
    contributions_best_pred,
    width,
    label="Best prediction (low NRMSE)",
    color="#2196F3",
    alpha=0.8,
)
axes[1, 0].bar(
    x_pos + width / 2,
    contributions_best_plaus,
    width,
    label="Best plausibility (low RSSD)",
    color="#FF9800",
    alpha=0.8,
)
axes[1, 0].set_xlabel("Channel")
axes[1, 0].set_ylabel("Attributed Effect Share (%)")
axes[1, 0].set_title(
    "Tradeoff: Best Prediction vs Best Plausibility\nDifferent frontier points tell different stories"
)
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(channel_names)
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(axis="y", alpha=0.3)

# Panel D: Model count by NRMSE quality tier
nrmse_bins = [0, 0.3, 0.5, 0.7, 1.0, 2.0]
tier_labels = [
    "Excellent\n(<0.3)",
    "Good\n(0.3-0.5)",
    "Fair\n(0.5-0.7)",
    "Poor\n(0.7-1.0)",
    "Bad\n(>1.0)",
]
tier_colors = ["#4CAF50", "#8BC34A", "#FFC107", "#FF9800", "#F44336"]

counts_all = np.histogram(model_nrmse, bins=nrmse_bins)[0]
counts_pareto = np.histogram(model_nrmse[pareto_mask], bins=nrmse_bins)[0]

x_tiers = np.arange(len(tier_labels))
axes[1, 1].bar(
    x_tiers - 0.2, counts_all, 0.4, label="All models", color="#9E9E9E", alpha=0.6
)
axes[1, 1].bar(
    x_tiers + 0.2, counts_pareto, 0.4, label="Pareto models", color="#E91E63", alpha=0.8
)
axes[1, 1].set_xlabel("Model Quality Tier (NRMSE)")
axes[1, 1].set_ylabel("Count")
axes[1, 1].set_title(
    "Model Quality Distribution\nPareto selection filters to the best tradeoffs"
)
axes[1, 1].set_xticks(x_tiers)
axes[1, 1].set_xticklabels(tier_labels)
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(axis="y", alpha=0.3)

plt.suptitle(
    "Robyn-Style Pareto Frontier: Multi-Objective Model Selection",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
finalize_figure("09_pareto_frontier.png", fig=fig)

print("\nKEY TAKEAWAY:")
print("  Robyn's Pareto frontier is a genuinely innovative approach to model")
print("  selection. By showing the RANGE of estimates across plausible models,")
print("  it communicates uncertainty without making probability claims that")
print("  depend on prior choices. 'Here are plausible models. TV gets between")
print("  X% and Y% of attributed media effect.' That is the part this demo")
print("  can defend directly from the simulated Pareto frontier.")
