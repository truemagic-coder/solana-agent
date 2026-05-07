# Minimal makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= poetry run sphinx-build
SPHINXAUTOBUILD ?= poetry run sphinx-autobuild
SOURCEDIR     = ./docs
BUILDDIR      = ./docs/_build
PYTHON        ?= poetry run python
SCENARIO      ?= success
SDK_SCENARIO  ?= sdk-success
SMOKE_STEP    ?= token_math
SMOKE_MODE    ?= smoke-step
SMOKE_BASE_URL ?= http://127.0.0.1:8000/v1
SMOKE_DOTENV  ?= ../solana-agent-agi/.env
SMOKE_PRIVY_USER_ID ?=

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean html serve smoke-step

# Target for building HTML documentation
html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

# Target for cleaning build directory
clean:
	rm -rf $(BUILDDIR)/*

# Target for serving documentation locally
serve: html
	@echo "Starting local server..."
	@cd $(BUILDDIR)/html && python3 -m http.server 8000

smoke-step:
	@$(PYTHON) scripts/smoke_step_runner.py $(SMOKE_STEP) --mode $(SMOKE_MODE) --base-url $(SMOKE_BASE_URL) --dotenv-path $(SMOKE_DOTENV) $(if $(SMOKE_PRIVY_USER_ID),--privy-user-id $(SMOKE_PRIVY_USER_ID),)

# Target for live reload during development
livehtml: html
	@echo "Starting live reload server..."
	@${SPHINXAUTOBUILD} "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
