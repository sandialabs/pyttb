# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = pyttb
SOURCEDIR     = ./docs/source
BUILDDIR      = ./docs/build

# Put it first so that "make" without argument is like "make help".
help:
	@echo "install: Install release build"
	@echo "install_dev: Install dev build"
	@echo "install_docs: Install docs build"
	@echo "docs_help: Show additional docs commands"

.PHONY: help install install_dev install_docs Makefile

install:
	python -m pip install -e .

install_dev:
	python -m pip install -e ".[dev]"

install_docs:
	python -m pip install -e ".[doc]"

docs_help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
