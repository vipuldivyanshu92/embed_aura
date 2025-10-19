.PHONY: install lint test run clean docker-build docker-run format typecheck help

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make lint         - Run linters (ruff)"
	@echo "  make format       - Format code with black"
	@echo "  make typecheck    - Run mypy type checking"
	@echo "  make test         - Run tests with pytest"
	@echo "  make run          - Run the application locally"
	@echo "  make clean        - Clean up generated files"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"

install:
	pip install -e ".[dev]"

lint:
	ruff check app tests
	black --check app tests

format:
	black app tests
	ruff check --fix app tests

typecheck:
	mypy app tests

test:
	pytest -v --cov=app --cov-report=term-missing

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev-null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build dist .coverage htmlcov

docker-build:
	docker build -t embed-service:latest .

docker-run:
	docker run -p 8000:8000 --env-file .env embed-service:latest

all: install lint typecheck test

