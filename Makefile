PYTHON ?= python3
MPLBACKEND ?= Agg
CACHE_ROOT := $(CURDIR)/code/.demo_cache
MPLCONFIGDIR := $(CACHE_ROOT)/mplconfig
XDG_CACHE_HOME := $(CACHE_ROOT)/xdg_cache
HOME_CACHE := $(CACHE_ROOT)/home

CORE_SCRIPTS = \
	code/01_ridge_bayesian_equivalence.py \
	code/02_uncertainty_in_mmm.py \
	code/03_halfnormal_prior_bias.py \
	code/04_prior_dominance_small_data.py \
	code/05_bias_variance_tradeoff.py \
	code/06_multicollinearity.py \
	code/07_probability_convolutions.py \
	code/08_prior_posterior_diagnostic.py \
	code/09_pareto_frontier.py \
	code/10_circular_reasoning_priors.py

ALL_SCRIPTS = $(CORE_SCRIPTS) code/11_ridge_vs_pymc_posterior.py

.PHONY: check prepare-cache render-core render-all

check:
	$(PYTHON) -m py_compile $(ALL_SCRIPTS) code/demo_utils.py

prepare-cache:
	mkdir -p $(MPLCONFIGDIR) $(XDG_CACHE_HOME)/fontconfig $(HOME_CACHE)

render-core: prepare-cache
	env HOME=$(HOME_CACHE) MPLBACKEND=$(MPLBACKEND) MPLCONFIGDIR=$(MPLCONFIGDIR) XDG_CACHE_HOME=$(XDG_CACHE_HOME) $(PYTHON) code/01_ridge_bayesian_equivalence.py
	env HOME=$(HOME_CACHE) MPLBACKEND=$(MPLBACKEND) MPLCONFIGDIR=$(MPLCONFIGDIR) XDG_CACHE_HOME=$(XDG_CACHE_HOME) $(PYTHON) code/02_uncertainty_in_mmm.py
	env HOME=$(HOME_CACHE) MPLBACKEND=$(MPLBACKEND) MPLCONFIGDIR=$(MPLCONFIGDIR) XDG_CACHE_HOME=$(XDG_CACHE_HOME) $(PYTHON) code/03_halfnormal_prior_bias.py
	env HOME=$(HOME_CACHE) MPLBACKEND=$(MPLBACKEND) MPLCONFIGDIR=$(MPLCONFIGDIR) XDG_CACHE_HOME=$(XDG_CACHE_HOME) $(PYTHON) code/04_prior_dominance_small_data.py
	env HOME=$(HOME_CACHE) MPLBACKEND=$(MPLBACKEND) MPLCONFIGDIR=$(MPLCONFIGDIR) XDG_CACHE_HOME=$(XDG_CACHE_HOME) $(PYTHON) code/05_bias_variance_tradeoff.py
	env HOME=$(HOME_CACHE) MPLBACKEND=$(MPLBACKEND) MPLCONFIGDIR=$(MPLCONFIGDIR) XDG_CACHE_HOME=$(XDG_CACHE_HOME) $(PYTHON) code/06_multicollinearity.py
	env HOME=$(HOME_CACHE) MPLBACKEND=$(MPLBACKEND) MPLCONFIGDIR=$(MPLCONFIGDIR) XDG_CACHE_HOME=$(XDG_CACHE_HOME) $(PYTHON) code/07_probability_convolutions.py
	env HOME=$(HOME_CACHE) MPLBACKEND=$(MPLBACKEND) MPLCONFIGDIR=$(MPLCONFIGDIR) XDG_CACHE_HOME=$(XDG_CACHE_HOME) $(PYTHON) code/08_prior_posterior_diagnostic.py
	env HOME=$(HOME_CACHE) MPLBACKEND=$(MPLBACKEND) MPLCONFIGDIR=$(MPLCONFIGDIR) XDG_CACHE_HOME=$(XDG_CACHE_HOME) $(PYTHON) code/09_pareto_frontier.py
	env HOME=$(HOME_CACHE) MPLBACKEND=$(MPLBACKEND) MPLCONFIGDIR=$(MPLCONFIGDIR) XDG_CACHE_HOME=$(XDG_CACHE_HOME) $(PYTHON) code/10_circular_reasoning_priors.py

render-all: render-core
	env HOME=$(HOME_CACHE) MPLBACKEND=$(MPLBACKEND) MPLCONFIGDIR=$(MPLCONFIGDIR) XDG_CACHE_HOME=$(XDG_CACHE_HOME) $(PYTHON) code/11_ridge_vs_pymc_posterior.py
