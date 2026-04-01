"""
11 - Ridge vs PyMC: Full Bayesian Posterior Inference
=====================================================
Research Point (Section 1 / report 1.3 / synthesis):
  Ridge regression is equivalent to Bayesian regression with a Gaussian
  prior at the point-estimate level. The Bayesian framework adds the
  full posterior distribution, posterior predictive inference, and
  principled uncertainty propagation.

This script demonstrates:
  1. Ridge coefficients match the exact Gaussian-posterior mean / MAP
  2. PyMC samples the same posterior and recovers the same coefficient means
  3. Full Bayesian inference yields coefficient intervals and posterior
     predictive intervals for observed outcomes
  4. Uncertainty can be propagated to a downstream planning scenario
"""

from __future__ import annotations

import os
import logging
import warnings
from pathlib import Path

# PyMC/ArviZ and matplotlib try to write caches in user locations. Redirect
# them into the repo so this demo remains self-contained and reproducible.
SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_ROOT = Path(os.environ.get("MMM_PYMC_DEMO_CACHE", SCRIPT_DIR / ".demo_cache"))
MPL_CONFIG = CACHE_ROOT / "mplconfig"
PLATFORM_CACHE = CACHE_ROOT / "platform_cache"
PYTENSOR_CACHE = CACHE_ROOT / "pytensor"
XDG_CACHE = CACHE_ROOT / "xdg_cache"

for path in [CACHE_ROOT, MPL_CONFIG, PLATFORM_CACHE, PYTENSOR_CACHE, XDG_CACHE]:
    path.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE))
if "base_compiledir" not in os.environ.get("PYTENSOR_FLAGS", ""):
    extra_flag = f"base_compiledir={PYTENSOR_CACHE}"
    current_flags = os.environ.get("PYTENSOR_FLAGS", "")
    os.environ["PYTENSOR_FLAGS"] = (
        f"{current_flags},{extra_flag}" if current_flags else extra_flag
    )

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"arviz(\.|$)",
)
warnings.filterwarnings(
    "ignore",
    message=".*ArviZ is undergoing a major refactor.*",
    category=FutureWarning,
)

import platformdirs


def _safe_user_cache_dir(
    appname=None, appauthor=None, version=None, opinion=True, ensure_exists=False
):
    """Return a writable cache directory inside the repo for sandboxed runs."""
    name = appname or "app"
    path = PLATFORM_CACHE / name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


platformdirs.user_cache_dir = _safe_user_cache_dir

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from scipy import stats
from sklearn.linear_model import Ridge

from demo_utils import finalize_figure

logging.getLogger("pymc").setLevel(logging.ERROR)

np.random.seed(42)


def exact_gaussian_posterior(X_c, y_c, sigma_obs, prior_sigma):
    """Closed-form posterior for Gaussian likelihood + Gaussian prior."""
    prior_precision = np.eye(X_c.shape[1]) / prior_sigma**2
    likelihood_precision = (X_c.T @ X_c) / sigma_obs**2
    posterior_cov = np.linalg.inv(prior_precision + likelihood_precision)
    posterior_mean = posterior_cov @ (X_c.T @ y_c / sigma_obs**2)
    return posterior_mean, posterior_cov


# ── 1. Simulate MMM-like linear data where Ridge = Gaussian Bayes mean/MAP ──
n_obs = 150
n_channels = 4
sigma_obs = 2.0
lambda_ridge = 5.0
prior_sigma = sigma_obs / np.sqrt(lambda_ridge)
channels = np.array(["TV", "Digital", "Radio", "OOH"])

beta_true = np.array([2.4, 1.3, 0.7, 1.8])
intercept_true = 40.0

mean_spend = np.array([100, 70, 35, 55])
cov_spend = np.array(
    [
        [220.0, 115.0, 70.0, 90.0],
        [115.0, 180.0, 55.0, 75.0],
        [70.0, 55.0, 90.0, 40.0],
        [90.0, 75.0, 40.0, 130.0],
    ]
)

X = np.random.multivariate_normal(mean_spend, cov_spend, size=n_obs)
X = np.maximum(X, 0.0)
noise = np.random.normal(0, sigma_obs, n_obs)
y = intercept_true + X @ beta_true + noise

X_c = X - X.mean(axis=0)
y_c = y - y.mean()


# ── 2. Ridge point estimate and exact posterior ──────────────────────────────
ridge = Ridge(alpha=lambda_ridge, fit_intercept=False).fit(X_c, y_c)
beta_ridge = ridge.coef_

posterior_mean_exact, posterior_cov_exact = exact_gaussian_posterior(
    X_c, y_c, sigma_obs, prior_sigma
)
posterior_sd_exact = np.sqrt(np.diag(posterior_cov_exact))
z_975 = stats.norm.ppf(0.975)
exact_interval = np.column_stack(
    [
        posterior_mean_exact - z_975 * posterior_sd_exact,
        posterior_mean_exact + z_975 * posterior_sd_exact,
    ]
)


# ── 3. PyMC full posterior inference ────────────────────────────────────────
coords = {"obs": np.arange(n_obs), "channel": channels}
with pm.Model(coords=coords) as model:
    X_data = pm.Data("X", X_c, dims=("obs", "channel"))
    beta = pm.Normal("beta", mu=0.0, sigma=prior_sigma, dims="channel")
    mu = pm.math.dot(X_data, beta)
    pm.Normal("y_obs", mu=mu, sigma=sigma_obs, observed=y_c, dims="obs")

    idata = pm.sample(
        draws=1000,
        tune=1000,
        chains=2,
        cores=1,
        random_seed=42,
        target_accept=0.92,
        progressbar=False,
        compute_convergence_checks=False,
    )
    posterior_predictive = pm.sample_posterior_predictive(
        idata,
        var_names=["y_obs"],
        random_seed=42,
        progressbar=False,
    )

beta_samples = (
    idata.posterior["beta"].stack(sample=("chain", "draw")).transpose("channel", "sample").values
)
posterior_mean_pymc = beta_samples.mean(axis=1)
pymc_interval = np.quantile(beta_samples, [0.025, 0.975], axis=1).T

y_ppc = (
    posterior_predictive.posterior_predictive["y_obs"]
    .stack(sample=("chain", "draw"))
    .transpose("obs", "sample")
    .values
)
pp_mean = y_ppc.mean(axis=1)
pp_low = np.quantile(y_ppc, 0.05, axis=1)
pp_high = np.quantile(y_ppc, 0.95, axis=1)
pp_coverage = np.mean((y_c >= pp_low) & (y_c <= pp_high))


# ── 4. Uncertainty propagation to a spend-planning scenario ─────────────────
planned_delta_spend = np.array([15.0, 10.0, 5.0, 8.0])
incremental_lift_ridge = planned_delta_spend @ beta_ridge
incremental_lift_exact_mean = planned_delta_spend @ posterior_mean_exact
incremental_lift_exact_sd = np.sqrt(
    planned_delta_spend @ posterior_cov_exact @ planned_delta_spend
)
incremental_lift_exact_interval = (
    incremental_lift_exact_mean - z_975 * incremental_lift_exact_sd,
    incremental_lift_exact_mean + z_975 * incremental_lift_exact_sd,
)
incremental_lift_pymc = beta_samples.T @ planned_delta_spend


# ── 5. Print comparison summary ──────────────────────────────────────────────
print("=" * 78)
print("RIDGE VS PYMC: POINT ESTIMATES, POSTERIOR INTERVALS, AND UNCERTAINTY")
print("=" * 78)
print()
print(f"Observations:            {n_obs}")
print(f"Channels:                {n_channels}")
print(f"Known sigma:             {sigma_obs:.2f}")
print(f"Ridge lambda:            {lambda_ridge:.2f}")
print(f"Equivalent prior sigma:  {prior_sigma:.4f}")
print()

print(
    f"{'Channel':<10} {'True':>8} {'Ridge':>8} {'Exact post mean':>16} {'PyMC mean':>12} {'|PyMC-exact|':>12}"
)
print("-" * 74)
for idx, channel in enumerate(channels):
    print(
        f"{channel:<10} {beta_true[idx]:>8.3f} {beta_ridge[idx]:>8.3f} "
        f"{posterior_mean_exact[idx]:>16.3f} {posterior_mean_pymc[idx]:>12.3f} "
        f"{abs(posterior_mean_pymc[idx] - posterior_mean_exact[idx]):>12.4f}"
    )

print()
print(
    f"Max |Ridge - exact posterior mean|: {np.max(np.abs(beta_ridge - posterior_mean_exact)):.2e}"
)
print(
    f"Max |PyMC mean - exact posterior mean|: {np.max(np.abs(posterior_mean_pymc - posterior_mean_exact)):.4f}"
)
print()
print("Posterior predictive (90% interval) coverage on observed centered y:")
print(f"  Coverage: {pp_coverage:.1%}")
print()
print("Spend-planning scenario:")
print(f"  Delta spend vector: {dict(zip(channels, planned_delta_spend))}")
print(f"  Ridge point estimate incremental lift: {incremental_lift_ridge:.2f}")
print(
    "  Exact Gaussian posterior incremental lift: "
    f"{incremental_lift_exact_mean:.2f} "
    f"[{incremental_lift_exact_interval[0]:.2f}, {incremental_lift_exact_interval[1]:.2f}]"
)
print(
    "  PyMC posterior incremental lift: "
    f"{np.mean(incremental_lift_pymc):.2f} "
    f"[{np.quantile(incremental_lift_pymc, 0.025):.2f}, {np.quantile(incremental_lift_pymc, 0.975):.2f}]"
)


# ── 6. Visualize point estimates, intervals, predictive uncertainty ─────────
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
x_pos = np.arange(n_channels)
width = 0.22

# Panel A: coefficient comparison
axes[0, 0].bar(x_pos - width, beta_true, width, label="True", color="#546E7A", alpha=0.9)
axes[0, 0].bar(x_pos, beta_ridge, width, label="Ridge", color="#FF9800", alpha=0.85)
axes[0, 0].bar(
    x_pos + width,
    posterior_mean_pymc,
    width,
    label="PyMC posterior mean",
    color="#2196F3",
    alpha=0.85,
)
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(channels)
axes[0, 0].set_ylabel("Coefficient")
axes[0, 0].set_title("Point Estimates: Ridge vs PyMC")
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(axis="y", alpha=0.3)

# Panel B: exact vs PyMC intervals
axes[0, 1].errorbar(
    x_pos - 0.08,
    posterior_mean_exact,
    yerr=np.vstack(
        [
            posterior_mean_exact - exact_interval[:, 0],
            exact_interval[:, 1] - posterior_mean_exact,
        ]
    ),
    fmt="o",
    capsize=4,
    color="#2E7D32",
    label="Exact 95% posterior interval",
)
axes[0, 1].errorbar(
    x_pos + 0.08,
    posterior_mean_pymc,
    yerr=np.vstack(
        [
            posterior_mean_pymc - pymc_interval[:, 0],
            pymc_interval[:, 1] - posterior_mean_pymc,
        ]
    ),
    fmt="o",
    capsize=4,
    color="#1565C0",
    label="PyMC 95% interval",
)
axes[0, 1].scatter(x_pos, beta_true, marker="*", s=140, color="black", label="True")
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(channels)
axes[0, 1].set_ylabel("Coefficient")
axes[0, 1].set_title("Full Posterior Inference: Coefficient Uncertainty")
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(axis="y", alpha=0.3)

# Panel C: posterior predictive intervals on the first 40 observations
obs_idx = np.arange(40)
axes[1, 0].fill_between(
    obs_idx,
    pp_low[:40],
    pp_high[:40],
    color="#90CAF9",
    alpha=0.5,
    label="PyMC posterior predictive 90% interval",
)
axes[1, 0].plot(obs_idx, pp_mean[:40], color="#1565C0", linewidth=2, label="Predictive mean")
axes[1, 0].scatter(obs_idx, y_c[:40], s=18, color="#F4511E", label="Observed centered y")
axes[1, 0].set_xlabel("Observation index")
axes[1, 0].set_ylabel("Centered outcome")
axes[1, 0].set_title(f"Posterior Predictive Uncertainty\n90% interval coverage = {pp_coverage:.0%}")
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(alpha=0.3)

# Panel D: propagated planning uncertainty
axes[1, 1].hist(
    incremental_lift_pymc,
    bins=35,
    density=True,
    alpha=0.6,
    color="#7E57C2",
    label="PyMC posterior lift distribution",
)
axes[1, 1].axvline(
    incremental_lift_ridge,
    color="#FB8C00",
    linestyle="--",
    linewidth=2,
    label=f"Ridge point estimate = {incremental_lift_ridge:.2f}",
)
axes[1, 1].axvline(
    incremental_lift_exact_mean,
    color="#2E7D32",
    linestyle=":",
    linewidth=2,
    label=f"Exact posterior mean = {incremental_lift_exact_mean:.2f}",
)
axes[1, 1].set_xlabel("Incremental lift from planned spend change")
axes[1, 1].set_ylabel("Density")
axes[1, 1].set_title("Uncertainty Propagation to a Planning Scenario")
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=0.3)

plt.suptitle(
    "Ridge vs PyMC: Same Gaussian Prior, Different Inferential Output",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
finalize_figure("11_ridge_vs_pymc_posterior.png", fig=fig)

print("\nKEY TAKEAWAY:")
print("  Ridge already gives the Gaussian-prior point estimate.")
print("  Full Bayes adds the rest: coefficient intervals, posterior predictive")
print("  uncertainty, and principled uncertainty propagation into decisions.")
print("  In this conjugate linear-Gaussian setting, PyMC should recover the")
print("  same posterior mean as the closed-form solution, then make the full")
print("  posterior available for downstream planning.")
