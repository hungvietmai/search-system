# Makefile for Leaf Search System

.PHONY: help install setup start-milvus stop-milvus ingest run test clean docs

# Default target
help:
	@echo "Leaf Search System - Available Commands"
	@echo "========================================"
	@echo "Python 3.13 Compatible - All dependencies updated!"
	@echo ""
	@echo "Setup Commands:"
	@echo "  make install        - Install Python dependencies"
	@echo "  make setup          - Complete setup (install + create env + start Milvus)"
	@echo "  make create-env     - Create .env file with default settings"
	@echo ""
	@echo "Milvus Commands:"
	@echo "  make start-milvus   - Start Milvus services with Docker"
	@echo "  make stop-milvus    - Stop Milvus services"
	@echo "  make restart-milvus - Restart Milvus services"
	@echo "  make milvus-logs    - View Milvus logs"
	@echo "  make milvus-reset   - Reset Milvus (WARNING: deletes all data)"
	@echo ""
	@echo "Data Commands:"
	@echo "  make ingest         - Ingest dataset into system"
	@echo "  make ingest-test    - Ingest small sample (1000 images) for testing"
	@echo ""
	@echo "Application Commands:"
	@echo "  make run            - Start FastAPI server"
	@echo "  make run-dev        - Start FastAPI server in development mode"
	@echo "  make test-api       - Run API test script"
	@echo "  make test-upload    - Run database management test script"
	@echo "  make test-preprocessing - Run preprocessing demo"
	@echo "  make test-two-stage - Run two-stage search test"
	@echo "  make test-data-mgmt - Run data management test"
	@echo "  make test-all       - Run all tests"
	@echo ""
	@echo "Maintenance Commands:"
	@echo "  make clean          - Remove generated files and caches"
	@echo "  make clean-all      - Remove all generated files including data"
	@echo ""

# Install dependencies
install:
	pip install -r requirements.txt

# Create .env file
create-env:
	python scripts/create_sample_env.py

# Complete setup
setup: install create-env start-milvus
	@echo ""
	@echo "✓ Setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Verify Milvus is running: make milvus-logs"
	@echo "  2. Ingest data: make ingest"
	@echo "  3. Start API server: make run"
	@echo ""

# Milvus commands
start-milvus:
	docker-compose up -d
	@echo "Waiting for Milvus to be ready..."
	@sleep 10
	@echo "✓ Milvus services started"
	@echo "  - Milvus: http://localhost:19530"
	@echo "  - Attu Admin UI: http://localhost:8001"

stop-milvus:
	docker-compose down

restart-milvus:
	docker-compose restart

milvus-logs:
	docker-compose logs -f milvus

milvus-reset:
	@echo "WARNING: This will delete all Milvus data!"
	@echo "Press Ctrl+C to cancel, or Enter to continue..."
	@read dummy
	docker-compose down -v
	docker-compose up -d

# Data ingestion
ingest:
	python scripts/ingest_data.py

ingest-test:
	python scripts/ingest_data.py --limit 1000

# Run application
run:
	python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

run-dev:
	python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Test
test-api:
	python scripts/test_search.py

test-upload:
	python scripts/test_upload.py

test-preprocessing:
	python scripts/test_preprocessing_demo.py

test-two-stage:
	python scripts/test_two_stage_search.py

test-data-mgmt:
	python scripts/test_data_management.py

test-all: test-api test-upload test-preprocessing test-two-stage test-data-mgmt

# Clean commands
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache 2>/dev/null || true
	rm -rf temp/* 2>/dev/null || true
	rm -rf uploads/* 2>/dev/null || true

clean-all: clean
	rm -rf data/* 2>/dev/null || true
	rm -f *.db 2>/dev/null || true
	@echo "✓ All generated files cleaned"

# Docker build
docker-build:
	docker build -t leaf-search-system:latest .

docker-run:
	docker run -p 8000:8000 -v $(PWD)/dataset:/app/dataset leaf-search-system:latest

