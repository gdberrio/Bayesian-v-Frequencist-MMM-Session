"""
10 - Attribution Data as Priors: When It IS and ISN'T Circular Reasoning
=========================================================================
Research Point (Section 9 / report 2.7):
  "If attribution data is used to setup MMM -- why MMM is needed in the
   first place? To dress up attribution data with some noise?"

  CIRCULAR when:
    - Platform-reported ROAS as tight prior (Google Ads ROAS=5 -> Meridian prior)
    - Last-click attribution -> prior (MMM just smooths attribution)
    - Tight priors in small data (prior dominates regardless)

  NOT CIRCULAR when:
    - Priors from randomized experiments (geo-lift, conversion lift)
    - Wide default priors (LogNormal(0.2, 0.9) allows 0.2 to 8.0)
    - Prior-posterior comparison shows data moved the prior

  "The ROI measured by an experiment never aligns perfectly with the
   ROI measured by MMM. They have different estimands."

This script demonstrates:
  1. Tight platform-based priors -> MMM confirms platform data (circular)
  2. Wide experiment-based priors -> MMM discovers from data (not circular)
  3. The "different estimands" problem
  4. How to detect circularity: source quality + prior strength + posterior movement
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


def overlap_coefficient(mu1, sig1, mu2, sig2, grid=None):
    """Compute overlap coefficient between two Gaussians."""
    if grid is None:
        grid = np.linspace(
            min(mu1, mu2) - 4 * max(sig1, sig2),
            max(mu1, mu2) + 4 * max(sig1, sig2),
            5000,
        )
    pdf1 = stats.norm.pdf(grid, mu1, sig1)
    pdf2 = stats.norm.pdf(grid, mu2, sig2)
    return np.trapz(np.minimum(pdf1, pdf2), grid)


def prior_weight(prior_std, data_std):
    """Weight placed on the prior mean in the conjugate posterior."""
    prior_prec = 1.0 / prior_std**2
    data_prec = 1.0 / data_std**2
    return prior_prec / (prior_prec + data_prec)


def posterior_leans_toward(prior_mean, post_mean, data_mean):
    """Whether the posterior remains closer to the prior or the data."""
    return "prior" if abs(post_mean - prior_mean) < abs(post_mean - data_mean) else "data"


# ══════════════════════════════════════════════════════════════════════════
# SCENARIO 1: CIRCULAR -- Platform ROAS as Tight Prior
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("SCENARIO 1: CIRCULAR REASONING")
print("Using Google Ads self-reported ROAS as a tight prior in Meridian")
print("=" * 70)

# Google Ads says ROAS = 5.0 (self-reported, likely inflated)
# True incremental ROAS = 2.0 (what MMM should find)
# Data suggests ROAS around 2.5 (noisy but closer to truth)

platform_roas = 5.0
true_roas = 2.0
data_roas = 2.5
data_noise = 0.8  # informative observational data, but still noisy

# Tight prior from platform data
tight_prior_mean = platform_roas
tight_prior_std = 0.5  # very confident in platform number

post_mean_tight, post_std_tight = bayesian_update(
    tight_prior_mean, tight_prior_std, data_roas, data_noise
)
tight_prior_weight = prior_weight(tight_prior_std, data_noise)

print(f"\n  Platform-reported ROAS:   {platform_roas}")
print(f"  True incremental ROAS:    {true_roas}")
print(f"  Data suggests (MLE):      {data_roas}")
print(f"  Tight prior:              N({tight_prior_mean}, {tight_prior_std})")
print(f"  Posterior:                 N({post_mean_tight:.2f}, {post_std_tight:.2f})")
print(f"  Prior weight in posterior: {tight_prior_weight:.1%}")
print(
    f"  Prior-posterior overlap:   {overlap_coefficient(tight_prior_mean, tight_prior_std, post_mean_tight, post_std_tight):.1%}"
)
print()
print(f"  The MMM 'finds' ROAS = {post_mean_tight:.1f}, which is close to the")
print(f"  platform's number ({platform_roas}) and far from the truth ({true_roas}).")
print(f"  The MMM has just LAUNDERED the platform attribution data.")
print(f"  -> THIS IS CIRCULAR REASONING")


# ══════════════════════════════════════════════════════════════════════════
# SCENARIO 2: NOT CIRCULAR -- Experimental Data with Wide Prior
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SCENARIO 2: NOT CIRCULAR REASONING")
print("Using geo-lift experiment results as a wide prior")
print("=" * 70)

# Geo-lift experiment found ROAS = 1.8 +/- 1.2
experiment_roas = 1.8
experiment_std = 1.2  # wide -- experiment has uncertainty too

# Use as a wide prior
wide_prior_mean = experiment_roas
wide_prior_std = experiment_std  # wide enough to let data speak

post_mean_wide, post_std_wide = bayesian_update(
    wide_prior_mean, wide_prior_std, data_roas, data_noise
)
wide_prior_weight = prior_weight(wide_prior_std, data_noise)

print(f"\n  Geo-lift experiment ROAS:  {experiment_roas} +/- {experiment_std}")
print(f"  True incremental ROAS:     {true_roas}")
print(f"  Data suggests (MLE):       {data_roas}")
print(f"  Wide prior:                N({wide_prior_mean}, {wide_prior_std})")
print(f"  Posterior:                  N({post_mean_wide:.2f}, {post_std_wide:.2f})")
print(f"  Prior weight in posterior:  {wide_prior_weight:.1%}")
print(
    f"  Prior-posterior overlap:    {overlap_coefficient(wide_prior_mean, wide_prior_std, post_mean_wide, post_std_wide):.1%}"
)
print()
print(
    f"  The posterior ({post_mean_wide:.2f}) is between the experiment ({experiment_roas})"
)
print(f"  and the data ({data_roas}). Both sources of information contribute.")
print(f"  -> THIS IS LEGITIMATE BAYESIAN UPDATING")


# ══════════════════════════════════════════════════════════════════════════
# SCENARIO 3: Meridian Default Prior -- How Much Room Does Data Have?
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SCENARIO 3: Meridian Default ROI Prior -- LogNormal(0.2, 0.9)")
print("=" * 70)

# LogNormal(0.2, 0.9) in log-space -> in original space:
# median = exp(0.2) = 1.22
# 95% CI: [exp(0.2 - 1.96*0.9), exp(0.2 + 1.96*0.9)] = [0.20, 7.47]
ln_mu = 0.2
ln_sigma = 0.9
median_roi = np.exp(ln_mu)
ci_low = np.exp(ln_mu - 1.96 * ln_sigma)
ci_high = np.exp(ln_mu + 1.96 * ln_sigma)

print(f"\n  LogNormal(mu={ln_mu}, sigma={ln_sigma}):")
print(f"    Median ROI:  {median_roi:.2f}")
print(f"    Mean ROI:    {np.exp(ln_mu + ln_sigma**2 / 2):.2f}")
print(f"    95% range:   [{ci_low:.2f}, {ci_high:.2f}]")
print()
print(f"  This is wide enough that data CAN meaningfully update it.")
print(f"  -> DEFENSIBLE prior choice (not circular)")


# ══════════════════════════════════════════════════════════════════════════
# SCENARIO 4: THE "DIFFERENT ESTIMANDS" PROBLEM
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SCENARIO 4: The 'Different Estimands' Problem")
print("=" * 70)

print("""
  Google Meridian documentation:
    "The ROI measured by an experiment never aligns perfectly with the
     ROI measured by MMM. (In statistical terms, the experiment and
     MMM have different estimands.)"

  Experiment: "What happens when we turn Facebook OFF in 20 geos?"
    -> Measures: Total average treatment effect of Facebook ON vs OFF
    -> Estimand: ATE of having Facebook at all

  MMM: "What is the marginal return of the NEXT dollar on Facebook?"
    -> Estimates: Marginal ROI at current spend level  
    -> Estimand: dSales/dSpend at the current operating point

  These are DIFFERENT quantities:
    - Experiment ATE = 3.0 (having Facebook at all is valuable)
    - MMM marginal ROI = 1.5 (next dollar is less valuable due to saturation)

  Using one as prior for the other introduces systematic mismatch.
""")

# Illustrate with a saturation curve
spend_range = np.linspace(0, 200, 500)
from scipy.special import expit


def hill(x, ec=80, slope=2):
    return x**slope / (ec**slope + x**slope)


contribution = 500 * hill(spend_range, ec=80, slope=2)

# ATE: total contribution at current spend vs zero
current_spend = 100
ate = 500 * hill(current_spend, 80, 2) - 0  # vs no spend

# Marginal ROI: derivative at current point
delta = 0.01
marginal = (
    500 * hill(current_spend + delta, 80, 2) - 500 * hill(current_spend, 80, 2)
) / delta

# Average ROI
avg_roi = 500 * hill(current_spend, 80, 2) / current_spend

print(f"  At current spend = {current_spend}:")
print(f"    Total contribution (ATE-like):  {ate:.1f}")
print(f"    Average ROI:                     {avg_roi:.2f}")
print(f"    Marginal ROI:                    {marginal:.2f}")
print()
print(
    f"  ATE vs marginal ROI differ by {abs(avg_roi - marginal) / marginal * 100:.0f}%!"
)
print(f"  Using ATE as prior for marginal ROI introduces systematic bias.")


# ══════════════════════════════════════════════════════════════════════════
# CIRCULARITY DETECTION METRIC
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("CIRCULARITY DETECTION: Key Questions to Ask")
print("=" * 70)

scenarios = [
    {
        "name": "Platform ROAS (tight)",
        "source": "Platform attribution",
        "causal_source": False,
        "prior_mean": tight_prior_mean,
        "prior_std": tight_prior_std,
        "post_mean": post_mean_tight,
        "post_std": post_std_tight,
    },
    {
        "name": "Geo-lift (wide)",
        "source": "Randomized experiment",
        "causal_source": True,
        "prior_mean": wide_prior_mean,
        "prior_std": wide_prior_std,
        "post_mean": post_mean_wide,
        "post_std": post_std_wide,
    },
]

print(
    f"\n{'Scenario':<22} {'Source':<22} {'Prior Wt':>10} {'Post leans':>12} {'Overlap':>10} {'Circular?':>12}"
)
print("-" * 94)
for scenario in scenarios:
    pr_m = scenario["prior_mean"]
    pr_s = scenario["prior_std"]
    po_m = scenario["post_mean"]
    po_s = scenario["post_std"]
    pw = prior_weight(pr_s, data_noise)
    lean = posterior_leans_toward(pr_m, po_m, data_roas)
    ov = overlap_coefficient(pr_m, pr_s, po_m, po_s)
    circular = "YES" if (not scenario["causal_source"] and pw > 0.5 and lean == "prior") else "NO"
    print(
        f"{scenario['name']:<22} {scenario['source']:<22} {pw:>9.1%} {lean:>12} {ov:>10.1%} {circular:>12}"
    )

print()
print("  Key questions for any Bayesian MMM vendor:")
print("  1. Where did your priors come from?")
print("  2. How tight are they?")
print("  3. How much did the posterior move from the prior?")
print("  Overlap alone is NOT enough: source quality and prior dominance matter.")
print("  If the answer is: 'platform data' + 'tight' + 'not much' -> circular.")


# ══════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
x_grid = np.linspace(-2, 10, 500)

# Panel A: Circular reasoning scenario
axes[0, 0].fill_between(
    x_grid,
    stats.norm.pdf(x_grid, tight_prior_mean, tight_prior_std),
    alpha=0.3,
    color="#FF9800",
    label=f"Prior (platform ROAS={platform_roas})",
)
axes[0, 0].fill_between(
    x_grid,
    stats.norm.pdf(x_grid, post_mean_tight, post_std_tight),
    alpha=0.3,
    color="#F44336",
    label=f"Posterior={post_mean_tight:.1f}",
)
axes[0, 0].axvline(
    data_roas,
    color="#2196F3",
    linestyle="--",
    linewidth=2,
    label=f"Data MLE={data_roas}",
)
axes[0, 0].axvline(
    true_roas,
    color="#4CAF50",
    linestyle="--",
    linewidth=2,
    label=f"True ROAS={true_roas}",
)
axes[0, 0].set_xlabel("ROAS")
axes[0, 0].set_ylabel("Density")
axes[0, 0].set_title(
    "CIRCULAR: Platform ROAS as Tight Prior\nMMM just launders attribution data",
    color="#D32F2F",
)
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(alpha=0.3)

# Panel B: Not circular scenario
axes[0, 1].fill_between(
    x_grid,
    stats.norm.pdf(x_grid, wide_prior_mean, wide_prior_std),
    alpha=0.3,
    color="#FF9800",
    label=f"Prior (experiment={experiment_roas}+/-{experiment_std})",
)
axes[0, 1].fill_between(
    x_grid,
    stats.norm.pdf(x_grid, post_mean_wide, post_std_wide),
    alpha=0.3,
    color="#4CAF50",
    label=f"Posterior={post_mean_wide:.2f}",
)
axes[0, 1].axvline(
    data_roas,
    color="#2196F3",
    linestyle="--",
    linewidth=2,
    label=f"Data MLE={data_roas}",
)
axes[0, 1].axvline(
    true_roas,
    color="black",
    linestyle="--",
    linewidth=2,
    label=f"True ROAS={true_roas}",
)
axes[0, 1].set_xlabel("ROAS")
axes[0, 1].set_ylabel("Density")
axes[0, 1].set_title(
    "NOT CIRCULAR: Experiment as Wide Prior\nBoth experiment and data contribute",
    color="#388E3C",
)
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3)

# Panel C: Meridian default prior
roi_range = np.linspace(0.01, 12, 500)
lognormal_pdf = stats.lognorm.pdf(roi_range, s=ln_sigma, scale=np.exp(ln_mu))
axes[1, 0].fill_between(roi_range, lognormal_pdf, alpha=0.4, color="#9C27B0")
axes[1, 0].axvline(
    median_roi,
    color="#9C27B0",
    linestyle="--",
    linewidth=2,
    label=f"Median={median_roi:.2f}",
)
axes[1, 0].axvspan(
    ci_low,
    ci_high,
    alpha=0.1,
    color="#9C27B0",
    label=f"95% range=[{ci_low:.1f}, {ci_high:.1f}]",
)
axes[1, 0].set_xlabel("ROI")
axes[1, 0].set_ylabel("Density")
axes[1, 0].set_title(
    "Meridian Default: LogNormal(0.2, 0.9)\nWide enough to let data speak"
)
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(alpha=0.3)
axes[1, 0].set_xlim(0, 12)

# Panel D: Different estimands illustration
axes[1, 1].plot(
    spend_range, contribution, "b-", linewidth=2, label="Contribution curve"
)
axes[1, 1].fill_between(
    [0, current_spend],
    [0, 0],
    [0, 500 * hill(current_spend, 80, 2)],
    alpha=0.2,
    color="#FF9800",
    label=f"ATE-like (avg ROI={avg_roi:.2f})",
)
# Tangent line at current spend
tangent_y = 500 * hill(current_spend, 80, 2) + marginal * (spend_range - current_spend)
valid = (tangent_y > 0) & (tangent_y < 500)
axes[1, 1].plot(
    spend_range[valid],
    tangent_y[valid],
    "r--",
    linewidth=2,
    label=f"Marginal ROI={marginal:.2f}",
)
axes[1, 1].scatter(
    [current_spend], [500 * hill(current_spend, 80, 2)], c="red", s=100, zorder=5
)
axes[1, 1].axvline(current_spend, color="gray", linestyle=":", alpha=0.5)
axes[1, 1].set_xlabel("Spend")
axes[1, 1].set_ylabel("Contribution")
axes[1, 1].set_title(
    "Different Estimands: ATE vs Marginal ROI\n'The experiment and MMM have different estimands'"
)
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_ylim(0, 500)

plt.suptitle(
    "Attribution Data as Priors: When It IS and ISN'T Circular Reasoning",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
finalize_figure("10_circular_reasoning_priors.png", fig=fig)

print("\nKEY TAKEAWAY:")
print("  The circular reasoning critique is VALID when:")
print("    - Platform-reported ROAS used as tight priors")
print("    - Last-click attribution data used as priors")
print("    - Prior dominates posterior (data cannot override)")
print("  The critique is LESS VALID when:")
print("    - Priors from randomized experiments (geo-lift, conversion lift)")
print("    - Wide priors that allow data to speak")
print("    - Prior-posterior comparison shows meaningful updating")
print("  KEY QUESTION: 'Where did your priors come from, how tight are they,")
print("  and how much did the posterior move from the prior?'")
