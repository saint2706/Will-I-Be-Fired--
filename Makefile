.PHONY: setup lint test train evaluate report app reproduce clean help

# Default target
.DEFAULT_GOAL := help

# Python and pip executables
PYTHON := python
PIP := pip

# Project directories
SRC_DIR := src
TEST_DIR := tests
REPORTS_DIR := reports
MODELS_DIR := models

help:  ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup:  ## Install all dependencies (production + dev)
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov black isort flake8 pre-commit pyyaml
	pre-commit install

lint:  ## Run code quality checks (black, isort, flake8)
	@echo "Running black..."
	black --check $(SRC_DIR) $(TEST_DIR)
	@echo "Running isort..."
	isort --check-only $(SRC_DIR) $(TEST_DIR)
	@echo "Running flake8..."
	flake8 $(SRC_DIR) $(TEST_DIR) --max-line-length=120 --extend-ignore=E203,W503

format:  ## Auto-format code with black and isort
	black $(SRC_DIR) $(TEST_DIR)
	isort $(SRC_DIR) $(TEST_DIR)

test:  ## Run unit tests with coverage
	$(PYTHON) -m pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html

train:  ## Train the model from scratch
	$(PYTHON) $(SRC_DIR)/train_model.py

evaluate:  ## Evaluate the trained model and generate metrics
	@echo "Metrics are generated during training. Check $(REPORTS_DIR)/metrics_with_ci.json"

report:  ## Generate all reports (splits, metrics with CIs, ablations, failures, figures)
	@echo "Reports are generated during training. Check $(REPORTS_DIR)/ for outputs."

app:  ## Launch the Streamlit GUI
	streamlit run $(SRC_DIR)/gui_app.py

reproduce:  ## Full reproducible pipeline: setup, train, test, generate all reports
	@echo "=== Reproducible Pipeline ==="
	@echo "Step 1: Installing dependencies..."
	$(MAKE) setup
	@echo ""
	@echo "Step 2: Running linters..."
	$(MAKE) lint || $(MAKE) format
	@echo ""
	@echo "Step 3: Running tests..."
	$(MAKE) test
	@echo ""
	@echo "Step 4: Training models and generating reports..."
	$(MAKE) train
	@echo ""
	@echo "=== Pipeline Complete ==="
	@echo "Check $(REPORTS_DIR)/ for generated reports and figures."

clean:  ## Remove generated files and caches
	rm -rf __pycache__ $(SRC_DIR)/__pycache__ $(TEST_DIR)/__pycache__
	rm -rf .pytest_cache .coverage htmlcov
	rm -rf $(MODELS_DIR)/*.joblib
	rm -rf $(REPORTS_DIR)/figures/*.png
	rm -rf $(REPORTS_DIR)/*.json $(REPORTS_DIR)/*.csv $(REPORTS_DIR)/*.md
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned up generated files and caches."
