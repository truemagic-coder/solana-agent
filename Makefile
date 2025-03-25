# Minimal makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= poetry run sphinx-build
SPHINXAUTOBUILD ?= poetry run sphinx-autobuild
SOURCEDIR     = ./docs
BUILDDIR      = ./docs/_build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean html serve

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

# Target for live reload during development
livehtml: html
	@echo "Starting live reload server..."
	@${SPHINXAUTOBUILD} "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
