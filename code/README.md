# Code Demo Map

These scripts are illustrative demos for the research in [Research](../Research). They are not empirical benchmarks of real MMM frameworks.

## 01 `01_ridge_bayesian_equivalence.py`

- Intended claim: Ridge regression and Bayesian MAP with a Gaussian prior are mathematically identical at the point-estimate level.
- What it actually shows: identical coefficients from `sklearn` Ridge and the closed-form Gaussian-prior MAP update on the same centered linear problem.
- Limitations: only MAP / point-estimate equivalence is shown; it does not compare full Bayesian posterior inference, uncertainty propagation, or nonlinear MMM transforms.

## 02 `02_uncertainty_in_mmm.py`

- Intended claim: uncertainty in MMM is real, not a uniquely Bayesian artifact.
- What it actually shows: repeated sampling changes estimates; alternative reasonable response curves can imply different fitted ROIs; different fitted model classes on the same synthetic dataset can suggest different actions.
- Limitations: the Dew-style section is only an illustration, not a strict reproduction of observational equivalence; the alternative fitted model is intentionally simpler and can fit worse than the nonlinear one.

## 03 `03_halfnormal_prior_bias.py`

- Intended claim: positivity-constraining HalfNormal priors can bias weak or zero media effects upward.
- What it actually shows: a toy posterior with a HalfNormal prior stays positive when the true effect is zero, and zero-effect channels look positive more often under the constrained prior.
- Limitations: the posterior is computed in simplified one-parameter settings; real MMMs can place priors on transformed or hierarchical parameters, so this is about the direction of the bias, not its exact production magnitude.

## 04 `04_prior_dominance_small_data.py`

- Intended claim: in small-data MMM settings, priors can heavily influence posterior estimates.
- What it actually shows: a conjugate Normal-Normal update where prior weight falls with sample size, plus a parameter-count argument for why many MMM parameters are weakly identified.
- Limitations: this is a scalar conjugate toy update, not a sampled nonlinear MMM; “typical MMM” parameter counts are stylized and meant to explain why prior influence can matter, not to prove that every posterior is prior-dominated.

## 05 `05_bias_variance_tradeoff.py`

- Intended claim: the “Bayesian bias is wrong” critique misses the bias-variance tradeoff that also motivates frequentist regularization.
- What it actually shows: in a deliberately collinear, small-sample regime, Ridge can reduce total MSE relative to OLS; OLS remains unbiased on average but unstable on individual datasets; James-Stein shrinkage improves total MSE in a separate toy problem.
- Limitations: the size of the Ridge advantage depends on the simulated regime; James-Stein is an analogy for shrinkage, not a literal MMM estimator comparison.

## 06 `06_multicollinearity.py`

- Intended claim: neither Bayesian nor frequentist MMM “solves” multicollinearity; the real fix is better variation or experimentation.
- What it actually shows: OLS becomes less stable as channel correlation rises; Ridge is somewhat more stable; exact Gaussian posteriors on one fixed collinear dataset stay highly elongated under wide priors and tighten only when the prior supplies extra information.
- Limitations: the Bayesian section uses a linear Gaussian posterior, not a full nonlinear MMM with adstock and saturation; the DECOMP.RSSD section is a stylized critique of an implicit plausibility constraint, not an implementation of Robyn itself.

## 07 `07_probability_convolutions.py`

- Intended claim: nonlinear parameter interactions in Bayesian MMM can create skewed derived quantities and non-identifiability ridges.
- What it actually shows: products of random variables become more skewed; several searched `(beta, ec, slope)` combinations can trace very similar contribution curves; likelihood contours reveal a ridge in `(beta, ec)` space.
- Limitations: this is a local identifiability illustration on toy grids, not MCMC on a full MMM; “heavy tails” here refers to the simulated derived quantity distribution, not a proof about every practical posterior.

## 08 `08_prior_posterior_diagnostic.py`

- Intended claim: prior-posterior comparison is a valuable Bayesian diagnostic for distinguishing assumptions from findings.
- What it actually shows: channels with informative data move away from the prior; channels with noisy data stay close to it; sensitivity to prior changes is easy to visualize.
- Limitations: the update is Gaussian and channel-wise; it does not include hierarchical structure, posterior sampling, or the joint parameter interactions present in production MMMs.

## 09 `09_pareto_frontier.py`

- Intended claim: Robyn-style multi-objective model selection communicates model uncertainty through a frontier of plausible tradeoffs.
- What it actually shows: many synthetic hyperparameter settings produce a Pareto frontier in `NRMSE` vs `DECOMP.RSSD`, and effect-share ranges across frontier points can tell different business stories.
- Limitations: this is not Robyn, Nevergrad, or MAPE.LIFT; it now reports effect-share ranges rather than literal ROAS because that is what the simulated transformed-contribution setup supports defensibly.

## 10 `10_circular_reasoning_priors.py`

- Intended claim: using attribution-like data as priors is circular in some cases and legitimate in others.
- What it actually shows: a tight platform-derived prior can dominate the posterior and drag it toward the platform number, while a wider experiment-derived prior leaves more room for the data; the final diagnostic combines source quality, prior weight, and posterior movement.
- Limitations: this is a stylized conjugate update, not a full MMM fed by real attribution and experiment systems; the “different estimands” example is conceptual and not calibrated to a specific platform experiment design.

## 11 `11_ridge_vs_pymc_posterior.py`

- Intended claim: Ridge gives the Gaussian-prior point estimate, while full Bayesian inference adds posterior uncertainty, posterior predictive intervals, and uncertainty propagation.
- What it actually shows: Ridge matches the exact Gaussian posterior mean / MAP; PyMC recovers the same posterior mean by sampling; the posterior is then used for coefficient intervals, posterior predictive checks, and a spend-planning scenario.
- Limitations: this is still a linear-Gaussian conjugate example with fixed observation noise; it is a companion to script `01`, not a nonlinear MMM with adstock, saturation, hierarchical pooling, or unknown-variance priors.

## Suggested Reading Order

- Start with `01`, `05`, `08`, and `11` for the most defensible technical demos.
- Use `02`, `06`, `07`, `09`, and `10` as argument illustrations rather than literal reproductions of production MMM pipelines.
- Treat all scripts as communication aids for the research, not as validation that one framework wins empirically.
