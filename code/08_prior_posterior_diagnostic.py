"""
08 - Prior-Posterior Comparison: A Uniquely Bayesian Diagnostic
================================================================
Research Point (Section 5 / Section 7 / report 2.5 / GPT-Deep-Research):
  "Prior-posterior comparison is a useful diagnostic that the Bayesian
   framework uniquely provides."

  "If prior and posterior are nearly identical for a channel, treat the
   result as an assumption, not a finding."

  "The posterior appropriately weights prior and data based on their
   relative informativeness."

  "When someone says 'posterior ~ prior' it means the data are not
   identifying the effect well -- and no estimation method can fix that."

This script demonstrates:
  1. Prior-posterior comparison plots for multiple channels
  2. Channels where data updates the prior (informative data)
  3. Channels where prior dominates (uninformative data)
  4. Prior sensitivity analysis (key diagnostic)
  5. KL divergence as a quantitative measure of prior-posterior drift
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from demo_utils import finalize_figure

np.random.seed(42)


def bayesian_update(prior_mean, prior_std, data_mean, data_std):
    """Normal-Normal conjugate update."""
    prior_prec = 1.0 / prior_std**2
    data_prec = 1.0 / data_std**2
    post_prec = prior_prec + data_prec
    post_mean = (prior_prec * prior_mean + data_prec * data_mean) / post_prec
    post_std = 1.0 / np.sqrt(post_prec)
    return post_mean, post_std


def kl_divergence_normal(mu1, sigma1, mu2, sigma2):
    """KL divergence between two Gaussians: KL(q || p) where q=N(mu1,sigma1), p=N(mu2,sigma2)."""
    return (
        np.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2) ** 2) / (2 * sigma2**2) - 0.5
    )


# ══════════════════════════════════════════════════════════════════════════
# PART 1: PRIOR-POSTERIOR COMPARISON FOR MULTIPLE CHANNELS
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 1: Prior vs Posterior Comparison Across Channels")
print("=" * 70)

# Simulate channels with different data informativeness
channels = {
    "TV": {"prior_mean": 2.0, "prior_std": 0.5, "true_roi": 2.5, "data_noise": 0.3},
    "Digital": {
        "prior_mean": 3.0,
        "prior_std": 1.0,
        "true_roi": 1.8,
        "data_noise": 0.4,
    },
    "Radio": {"prior_mean": 1.5, "prior_std": 0.5, "true_roi": 1.6, "data_noise": 2.0},
    "OOH": {"prior_mean": 1.0, "prior_std": 0.3, "true_roi": 2.0, "data_noise": 5.0},
    "Social": {"prior_mean": 2.5, "prior_std": 0.8, "true_roi": 2.3, "data_noise": 0.5},
    "Print": {"prior_mean": 0.8, "prior_std": 0.2, "true_roi": 0.3, "data_noise": 3.0},
}

print(
    f"\n{'Channel':<10} {'Prior':>12} {'Data MLE':>10} {'Posterior':>12} {'Prior Wt':>10} {'KL(post||prior)':>16} {'Verdict':<20}"
)
print("-" * 94)

channel_results = {}
for ch_name, ch_info in channels.items():
    # Simulate data: MLE is noisy version of true
    data_mean = ch_info["true_roi"] + np.random.normal(0, ch_info["data_noise"] * 0.3)
    data_std = ch_info["data_noise"]

    post_mean, post_std = bayesian_update(
        ch_info["prior_mean"], ch_info["prior_std"], data_mean, data_std
    )

    # Prior weight
    prior_prec = 1.0 / ch_info["prior_std"] ** 2
    data_prec = 1.0 / data_std**2
    prior_weight = prior_prec / (prior_prec + data_prec)

    # KL divergence (posterior vs prior)
    kl = kl_divergence_normal(
        post_mean, post_std, ch_info["prior_mean"], ch_info["prior_std"]
    )

    verdict = "Data speaks" if prior_weight < 0.5 else "Prior dominates"

    channel_results[ch_name] = {
        "prior_mean": ch_info["prior_mean"],
        "prior_std": ch_info["prior_std"],
        "data_mean": data_mean,
        "data_std": data_std,
        "post_mean": post_mean,
        "post_std": post_std,
        "prior_weight": prior_weight,
        "kl": kl,
    }

    print(
        f"{ch_name:<10} N({ch_info['prior_mean']:.1f},{ch_info['prior_std']:.1f}) "
        f"{data_mean:>10.2f} N({post_mean:.2f},{post_std:.2f}) "
        f"{prior_weight:>10.1%} {kl:>16.4f} {verdict:<20}"
    )

print()
print("  -> Channels with low data noise: posterior moves away from prior (good)")
print("     Channels with high data noise: posterior ~ prior (assumption, not finding)")


# ══════════════════════════════════════════════════════════════════════════
# PART 2: PRIOR SENSITIVITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 2: Prior Sensitivity Analysis")
print("=" * 70)

# For a channel with high data noise (OOH), show how different priors
# produce different posteriors -- indicating the data is not informative
ch_test = "OOH"
ch_info = channel_results[ch_test]

prior_means_test = [0.5, 1.0, 1.5, 2.0, 3.0]

print(f"\nChannel: {ch_test} (data noise is high)")
print(f"Data MLE: {ch_info['data_mean']:.2f}")
print()
print(
    f"{'Prior mean':>12} {'Posterior mean':>15} {'Post moved to':>15} {'Sensitivity':>12}"
)
print("-" * 56)

for pm in prior_means_test:
    pm_post, ps_post = bayesian_update(
        pm, ch_info["prior_std"], ch_info["data_mean"], ch_info["data_std"]
    )
    movement = (
        abs(pm_post - pm) / abs(ch_info["data_mean"] - pm)
        if abs(ch_info["data_mean"] - pm) > 0.01
        else 0
    )
    print(
        f"{pm:>12.1f} {pm_post:>15.3f} {'prior' if movement < 0.3 else 'data':>15} {movement:>12.1%}"
    )

print()
print("  -> If the posterior changes dramatically with different priors,")
print("     the data is not informative. This is a DATA problem, not a")
print("     method problem. No estimation method can fix it.")


# ══════════════════════════════════════════════════════════════════════════
# PART 3: WELL-IDENTIFIED vs POORLY-IDENTIFIED CHANNELS
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 3: Well-Identified vs Poorly-Identified Channels")
print("=" * 70)

# TV (well-identified) vs OOH (poorly identified)
for ch_name in ["TV", "OOH"]:
    r = channel_results[ch_name]
    print(f"\n  {ch_name}:")
    print(f"    Prior:     N({r['prior_mean']:.2f}, {r['prior_std']:.2f})")
    print(f"    Data:      MLE = {r['data_mean']:.2f} (noise = {r['data_std']:.2f})")
    print(f"    Posterior: N({r['post_mean']:.2f}, {r['post_std']:.2f})")
    print(f"    Prior weight: {r['prior_weight']:.1%}")
    print(f"    KL(post||prior): {r['kl']:.4f}")

    if r["prior_weight"] > 0.5:
        print(f"    >>> TREAT AS ASSUMPTION, NOT FINDING <<<")
    else:
        print(f"    >>> Data has genuinely updated the prior <<<")


# ══════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
x_grid = np.linspace(-2, 7, 500)

# Panel A-F: Prior-Posterior plots for each channel
for idx, (ch_name, r) in enumerate(channel_results.items()):
    ax = axes[idx // 3, idx % 3]

    # Prior
    prior_pdf = stats.norm.pdf(x_grid, r["prior_mean"], r["prior_std"])
    ax.fill_between(x_grid, prior_pdf, alpha=0.3, color="#FF9800", label="Prior")
    ax.plot(x_grid, prior_pdf, color="#FF9800", linewidth=1.5)

    # Likelihood (data)
    data_pdf = stats.norm.pdf(x_grid, r["data_mean"], r["data_std"])
    data_pdf_scaled = data_pdf / data_pdf.max() * prior_pdf.max()
    ax.plot(
        x_grid,
        data_pdf_scaled,
        color="#9E9E9E",
        linewidth=1.5,
        linestyle="--",
        label="Data (scaled)",
    )

    # Posterior
    post_pdf = stats.norm.pdf(x_grid, r["post_mean"], r["post_std"])
    ax.fill_between(x_grid, post_pdf, alpha=0.3, color="#2196F3", label="Posterior")
    ax.plot(x_grid, post_pdf, color="#2196F3", linewidth=2)

    # Annotations
    verdict = "Data speaks" if r["prior_weight"] < 0.5 else "PRIOR DOMINATES"
    color = "#4CAF50" if r["prior_weight"] < 0.5 else "#D32F2F"

    ax.set_title(
        f"{ch_name}: {verdict}\n(Prior wt={r['prior_weight']:.0%}, KL={r['kl']:.3f})",
        color=color,
        fontweight="bold",
    )
    ax.axvline(r["prior_mean"], color="#FF9800", linestyle=":", alpha=0.5)
    ax.axvline(r["post_mean"], color="#2196F3", linestyle=":", alpha=0.5)
    ax.set_xlabel("ROI")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)

plt.suptitle(
    "Prior-Posterior Diagnostic: A Uniquely Bayesian Tool\n'If posterior ~ prior, the result is an assumption, not a finding'",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
finalize_figure("08_prior_posterior_diagnostic.png", fig=fig)

# ── Additional: Prior sensitivity plot ─────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

for ax_idx, (ch_name, title) in enumerate(
    [("TV", "Well-Identified (TV)"), ("OOH", "Poorly-Identified (OOH)")]
):
    r = channel_results[ch_name]
    ax = axes2[ax_idx]

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(prior_means_test)))
    for pm_test, color in zip(prior_means_test, colors):
        pm_post, ps_post = bayesian_update(
            pm_test, r["prior_std"], r["data_mean"], r["data_std"]
        )
        post_pdf = stats.norm.pdf(x_grid, pm_post, ps_post)
        ax.plot(
            x_grid,
            post_pdf,
            color=color,
            linewidth=2,
            label=f"Prior mean={pm_test} -> Post={pm_post:.2f}",
        )

    ax.axvline(
        r["data_mean"],
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.5,
        label=f"Data MLE={r['data_mean']:.2f}",
    )
    ax.set_xlabel("ROI")
    ax.set_ylabel("Posterior Density")
    ax.set_title(f"Prior Sensitivity: {title}")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

plt.suptitle(
    "Prior Sensitivity Analysis\nWell-identified channel: posteriors converge | Poorly-identified: posteriors track priors",
    fontsize=12,
    fontweight="bold",
    y=1.05,
)
plt.tight_layout()
finalize_figure("08_prior_sensitivity.png", fig=fig2)

print("\nKEY TAKEAWAY:")
print("  The prior-posterior comparison is one of the most valuable tools")
print("  in Bayesian MMM -- and has no direct frequentist equivalent.")
print("  If posterior ~ prior:  the result is an assumption, not a finding.")
print("  If posterior != prior: the data has genuinely updated the belief.")
print("  Prior sensitivity analysis reveals which channels are data-driven")
print("  vs assumption-driven. Demand this from any Bayesian MMM vendor.")
