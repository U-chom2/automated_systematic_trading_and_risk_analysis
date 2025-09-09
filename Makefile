.PHONY: help install dev-install test lint format type-check clean run migrate docker-up docker-down

help:
	@echo "Available commands:"
	@echo "  install       Install production dependencies"
	@echo "  dev-install   Install development dependencies"
	@echo "  test          Run tests"
	@echo "  lint          Run linter"
	@echo "  format        Format code"
	@echo "  type-check    Run type checking"
	@echo "  clean         Clean up temporary files"
	@echo "  run           Run the application"
	@echo "  migrate       Run database migrations"
	@echo "  docker-up     Start Docker containers"
	@echo "  docker-down   Stop Docker containers"

install:
	uv sync

dev-install:
	uv sync --dev

test:
	uv run pytest tests/ -v --cov=src --cov-report=term-missing

test-unit:
	uv run pytest tests/unit -v -m unit

test-integration:
	uv run pytest tests/integration -v -m integration

test-e2e:
	uv run pytest tests/e2e -v -m e2e

lint:
	uv run ruff check src/ tests/

format:
	uv run black src/ tests/
	uv run ruff check --fix src/ tests/

type-check:
	uv run mypy src/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true

run:
	uv run uvicorn src.presentation.api.app:app --reload --host 0.0.0.0 --port 8000

run-prod:
	uv run gunicorn src.presentation.api.app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

migrate:
	uv run alembic upgrade head

migrate-create:
	uv run alembic revision --autogenerate -m "$(message)"

migrate-rollback:
	uv run alembic downgrade -1

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-build:
	docker-compose build

docker-clean:
	docker-compose down -v
	docker system prune -f

db-shell:
	docker exec -it trading_postgres psql -U trading_user -d trading_db

redis-shell:
	docker exec -it trading_redis redis-cli

jupyter:
	uv run jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

docs:
	uv run mkdocs serve

docs-build:
	uv run mkdocs build

pre-commit:
	uv run pre-commit run --all-files

setup-pre-commit:
	uv run pre-commit install

# Trading System Commands
screening:
	uv run python main_trading_system.py --mode screening

analysis:
	uv run python main_trading_system.py --mode analysis

recording:
	uv run python main_trading_system.py --mode recording

morning:
	uv run python main_trading_system.py --mode morning

evening:
	uv run python main_trading_system.py --mode evening

trade-full:
	uv run python main_trading_system.py --mode full

scheduler:
	uv run python scheduler.py

# Data Commands
clean-data:
	rm -f data/target.csv data/todos.json data/trade_records.json data/settlements.json

reset-portfolio:
	rm -f data/portfolio.json