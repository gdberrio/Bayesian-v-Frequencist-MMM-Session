# Bayesian vs Frequentist MMM

This repository bundles the research notes, simulation demos, and slide deck for the "Bayesian vs. Frequentist MMM" lightning session.

The code is intentionally illustrative. These demos are communication aids for the attached research, not empirical benchmarks of production MMM frameworks.

## Structure

- `code/`: 11 Python demos plus their generated figures
- `presentation/`: static HTML slide deck
- `Research/`: source PDFs and research notes
- `content-drafts/`: derivative content for shorts, email, LinkedIn, and video

## Quick Start

1. Install dependencies:
   `pip install -r code/requirements.txt`
2. Run a quick syntax check:
   `make check`
3. Render the core figures used by the talk:
   `make render-core`
4. Render the full set, including the heavier PyMC posterior demo:
   `make render-all`

The slide deck lives at `presentation/index.html` and now reads figures directly from `code/`, so there is a single source of truth for generated assets.

## Notes

- The scripts save figures relative to their own file location, so running them from the repo root or from inside `code/` produces the same output paths.
- `make render-core` is the safest default for quick refreshes. `make render-all` additionally runs the PyMC demo in `11_ridge_vs_pymc_posterior.py`.
- Older duplicate files under `presentation/images/` are intentionally ignored so future commits do not reintroduce a second asset pipeline.
