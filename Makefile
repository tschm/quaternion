# Colors for pretty output
BLUE := \033[36m
BOLD := \033[1m
RESET := \033[0m

.DEFAULT_GOAL := uv

.PHONY: uv fmt marimo

##@ Development Setup

uv:
	@printf "$(BLUE)Creating virtual environment...$(RESET)\n"
	@curl -LsSf https://astral.sh/uv/install.sh | sh

# Mark fmt target as phony (not a file target)
fmt: uv ## Run autoformatting and linting
	# Run all pre-commit hooks on all files
	@uvx pre-commit run --all-files

marimo: uv ## Start a Marimo server
	@printf "$(BLUE)Start Marimo server...$(RESET)\n"
	@uvx marimo edit --sandbox notebooks/rot2d.py
