.PHONY: install install-dev test clean format lint build examples help

# Install the package
install:
	uv pip install .

# Install the package in development mode with all dependencies
install-dev:
	uv pip install -e ".[dev,viz,examples]"

# Set up development environment
setup-dev:
	uv venv
	uv pip install -e ".[dev,viz,examples]"

# Run tests
test:
	uv run pytest tests/ -v

# Run tests with coverage
test-cov:
	uv run pytest tests/ --cov=autofillgluon --cov-report=html --cov-report=term

# Format code with black and isort
format:
	uv run black .
	uv run isort .

# Lint code
lint:
	uv run flake8 .
	uv run black --check .
	uv run isort --check-only .

# Type check with mypy
typecheck:
	uv run mypy autofillgluon/

# Clean build artifacts and caches
clean:
	rm -rf build/ dist/ *.egg-info/
	rm -rf .venv/
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*.pyd" -delete
	find . -name ".pytest_cache" -type d -exec rm -rf {} +
	find . -name ".coverage" -delete
	find . -name "htmlcov" -type d -exec rm -rf {} +
	find . -name ".mypy_cache" -type d -exec rm -rf {} +

# Build distribution package
build: clean
	uv build

# Sync dependencies
sync:
	uv pip sync

# Update dependencies
update:
	uv pip compile --upgrade pyproject.toml

# Run a specific example
example-california:
	uv run python examples/california_housing_example.py

example-survival:
	uv run python examples/survival_analysis_example.py

example-basic:
	uv run python examples/basic/simple_imputation.py

# Start Jupyter notebook
notebook:
	uv run jupyter lab

# Help
help:
	@echo "Available targets:"
	@echo "  install      - Install the package"
	@echo "  install-dev  - Install in development mode with all dependencies"
	@echo "  setup-dev    - Set up complete development environment"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage report"
	@echo "  format       - Format code with black and isort"
	@echo "  lint         - Lint code with flake8, black, and isort"
	@echo "  typecheck    - Type check with mypy"
	@echo "  clean        - Clean build artifacts and caches"
	@echo "  build        - Build distribution package"
	@echo "  sync         - Sync dependencies"
	@echo "  update       - Update dependencies"
	@echo "  example-*    - Run specific examples"
	@echo "  notebook     - Start Jupyter Lab"
	@echo "  help         - Show this help message"