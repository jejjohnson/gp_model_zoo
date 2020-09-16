.PHONY: conda format style types black test link check docs
.DEFAULT_GOAL = help

PYTHON = python
VERSION = 3.8
NAME = py_name
ROOT = ./
PIP = pip
CONDA = conda
SHELL = bash
ENV = src
HOST = 127.0.0.1
PORT = 3002

help:	## Display this help
		@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

# DOCS
docs-build: ## Build site documentation with mkdocs
		@printf "\033[1;34mCreating full documentation with mkdocs...\033[0m\n"
		mkdocs build --config-file mkdocs.yml --clean --theme material --site-dir site/
		@printf "\033[1;34mmkdocs completed!\033[0m\n\n"

docs-live: ## Build mkdocs documentation live
		@printf "\033[1;34mStarting live docs with mkdocs...\033[0m\n"
		mkdocs serve --dev-addr $(HOST):$(PORT) --theme material

docs-live-d: ## Build mkdocs documentation live (quicker reload)
		@printf "\033[1;34mStarting live docs with mkdocs...\033[0m\n"
		mkdocs serve --dev-addr $(HOST):$(PORT) --dirtyreload --theme material

docs-deploy:  ## Deploy docs
		@printf "\033[1;34mDeploying docs...\033[0m\n"
		mkdocs gh-deploy
		@printf "\033[1;34mSuccess...\033[0m\n"
