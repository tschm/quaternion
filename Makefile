# Colors for pretty output
BLUE := \033[36m
BOLD := \033[1m
RESET := \033[0m

.DEFAULT_GOAL := uv

.PHONY: uv fmt 2d 3d

##@ Development Setup

uv:
	@printf "$(BLUE)Creating virtual environment...$(RESET)\n"
	@curl -LsSf https://astral.sh/uv/install.sh | sh

# Mark fmt target as phony (not a file target)
fmt: uv ## Run autoformatting and linting
	# Run all pre-commit hooks on all files
	@uvx pre-commit run --all-files

2d: uv ## Start a Marimo server
	@printf "$(BLUE)Start Marimo server...$(RESET)\n"
	@uvx marimo edit --sandbox notebooks/rot2d.py

3d: uv ## Start a Marimo server
	@printf "$(BLUE)Start Marimo server...$(RESET)\n"
	@uvx marimo edit --sandbox notebooks/rot3d.py