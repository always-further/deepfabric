.PHONY: clean install format lint test-unit test-integration test-integration-verbose security build all
.PHONY: test-integration-openai test-integration-gemini test-integration-llm
.PHONY: test-integration-hubs test-integration-spin test-integration-quick

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -f .coverage
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

install:
	uv sync --all-extras

format: ## Format code with ruff (parallel)
	uv run ruff format deepfabric/ tests/

lint:
	uv run ruff check . --exclude notebooks/

test-unit:
	uv run pytest tests/unit/

test-integration:
	uv run pytest tests/integration --tb=short --maxfail=1 -v

.PHONY: test-integration-verbose
test-integration-verbose:
	uv run pytest -v -rA --durations=10 tests/integration/

test-integration-openai:
	uv run pytest tests/integration -m openai --tb=short -v

test-integration-gemini:
	uv run pytest tests/integration -m gemini --tb=short -v

test-integration-llm:
	uv run pytest tests/integration -m "openai or gemini" --tb=short -v

test-integration-hubs:
	uv run pytest tests/integration -m huggingface --tb=short -v

test-integration-spin:
	uv run pytest tests/integration -m spin --tb=short -v

test-integration-quick:
	uv run pytest tests/integration -m "not huggingface" --tb=short -v

security:
	uv run bandit -r deepfabric/

build: clean test-unit
	uv build

all: clean install format lint test-unit test-integration security build
