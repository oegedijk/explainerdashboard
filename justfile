# Justfile for explainerdashboard development commands
# Install just: https://github.com/casey/just

# Default recipe - show available commands
default:
    @just --list

# Install dependencies
install:
    uv sync --all-extras

# Install dev dependencies (pre-commit, etc.)
install-dev:
    uv sync --group dev

# Install only production dependencies
install-prod:
    uv sync

# Install with test dependencies
install-test:
    uv sync --extra test

# Install with docs dependencies
install-docs:
    uv sync --extra docs

# Run all tests
test:
    uv run pytest tests/

# Run all tests with coverage
test-cov:
    uv run pytest --cov=explainerdashboard --cov-report=xml tests/

# Run all tests with HTML coverage report
test-cov-html:
    uv run pytest --cov=explainerdashboard --cov-report=html tests/
    @echo "Coverage report generated in htmlcov/index.html"

# Run only unit tests (exclude integration tests)
test-unit:
    uv run pytest tests/ -k "not integration"

# Run only integration tests
test-integration:
    uv run pytest tests/integration_tests/

# Run tests excluding selenium/integration tests (faster, no browser needed)
test-fast:
    uv run pytest tests/ -k "not selenium and not integration"

# Run only CLI tests
test-cli:
    uv run pytest -m cli tests/

# Run only selenium tests
test-selenium:
    uv run pytest -m selenium tests/

# Run tests excluding selenium
test-no-selenium:
    uv run pytest -m "not selenium" tests/

# Run a specific test file
test-file FILE:
    uv run pytest tests/{{FILE}}

# Run a specific test function
test-func FILE FUNC:
    uv run pytest tests/{{FILE}}::{{FUNC}}

# Lint with ruff
lint:
    uv run ruff check --ignore=F405,F403,E402 .

# Format code with ruff
format:
    uv run ruff format .

# Check formatting without making changes
format-check:
    uv run ruff format --check .

# Run lint and format check
check:
    just lint
    just format-check

# Build the package
build:
    uv build

# Clean build artifacts
clean:
    rm -rf build/
    rm -rf dist/
    rm -rf *.egg-info/
    rm -rf .eggs/
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete
    find . -type f -name "*.pyo" -delete
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".coverage" -exec rm -f {} + 2>/dev/null || true

# Install pre-commit hooks
pre-commit-install:
    uv run pre-commit install

# Run pre-commit hooks on all files
pre-commit-run:
    uv run pre-commit run --all-files

# Run pre-commit hooks on staged files only
pre-commit-staged:
    uv run pre-commit run

# Run all checks (lint, format, tests) before committing
pre-commit:
    just lint
    just format-check
    just test-fast

# Show Python version
python-version:
    uv run python --version

# Show package version
version:
    uv run python -c "import explainerdashboard; print(explainerdashboard.__version__)"
