# Bayesian vs Frequentist MMM

This repository bundles the simulation demos and generated figures for the "Bayesian vs. Frequentist MMM" lightning session.

The code is intentionally illustrative. These demos are communication aids, not empirical benchmarks of production MMM frameworks.

## Structure

- `code/`: 11 Python demos plus their generated figures

## Quick Start

1. Install dependencies:
   `pip install -r code/requirements.txt`
2. Run a quick syntax check:
   `make check`
3. Render the core figures used by the talk:
   `make render-core`
4. Render the full set, including the heavier PyMC posterior demo:
   `make render-all`

## Notes

- The scripts save figures relative to their own file location, so running them from the repo root or from inside `code/` produces the same output paths.
- `make render-core` is the safest default for quick refreshes. `make render-all` additionally runs the PyMC demo in `11_ridge_vs_pymc_posterior.py`.

## References

### Academic Papers

- Jin, Y., Wang, Y., Sun, Y., Chan, D., & Koehler, J. (2017). "Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects." Google Research.
- Sun, Y., Wang, Y., Jin, Y., Chan, D., & Koehler, J. (2017). "Geo-level Bayesian Hierarchical Media Mix Modeling." Google Research.
- Wang, Y., Jin, Y., Sun, Y., Chan, D., & Koehler, J. (2017). "A Hierarchical Bayesian Approach to Improve Media Mix Models Using Category Data." Google Research.
- Dew, R., Padilla, N., & Shchetkina, A. (2024). "Your MMM is Broken: Identification of Nonlinear and Time-varying Effects in Marketing Mix Models." arXiv:2408.07678.
- Zhang, Y. et al. (2024). "ROAS-parameterized Priors for Mitigating Omitted Variable Bias in MMM." Google Research.
- Rossi, P. E., Allenby, G. M., & McCulloch, R. (2005). *Bayesian Statistics and Marketing*. Wiley. (2nd ed. with Misra, S., 2024.)
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
- Park, T. & Casella, G. (2008). "The Bayesian Lasso." *Journal of the American Statistical Association*.
- Blattberg, R. C. & George, E. I. (1991). "Shrinkage Estimation of Price and Promotional Elasticities." *Journal of the American Statistical Association*.
- Manchanda, P., Rossi, P. E., & Chintagunta, P. K. (2004). "Response Modeling with Nonrandom Marketing-Mix Variables." *Journal of Marketing Research*.
- James, W. & Stein, C. (1961). "Estimation with Quadratic Loss." *Proceedings of the Fourth Berkeley Symposium*.
- McElreath, R. *Statistical Rethinking*. CRC Press.

### Framework Documentation

- Google Meridian: https://developers.google.com/meridian
- Meta Robyn: https://facebookexperimental.github.io/Robyn
- PyMC-Marketing: https://www.pymc-marketing.io

### Industry Sources

- Aryma Labs / Venkat Raman (2024). "There is no 'Uncertainty' in MMM." Substack.
- Aryma Labs (2024). "Two Key Problems that Ails Bayesian MMM." Substack.
- Aryma Labs (2024). "Want Performance Guarantees? Choose Frequentist MMM." Substack.
- Aryma Labs (2024). "One True Marketing Mix Model?" Substack.
- Recast (2026). "Understand & Manage Multicollinearity in Your MMM."
- Recast / Kaminsky (2021). "Introduction to Bayesian Methods for MMM."
- PyMC Labs (2021). "Bayesian Media Mix Modeling for Marketing Optimization."
- Juan Orduz (2022). "Media Effect Estimation with PyMC: Adstock, Saturation & Diminishing Returns."
