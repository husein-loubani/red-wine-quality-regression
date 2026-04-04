.PHONY: help install data lint notebook clean

# -- Settings -----------------------------------------------------------------
PYTHON   = venv/bin/python3
PIP      = venv/bin/pip
NOTEBOOK = notebooks/red_wine_quality_regression.ipynb
MODULE   = red_wine_quality

## help     : Show this help message
help:
	@grep -E '^## ' Makefile | sed 's/^## //'

## install  : Create venv and install all dependencies
install:
	python3 -m venv venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "Environment ready. Activate with: source venv/bin/activate"

## data     : Verify raw CSV is in place
data:
	@test -f data/raw/winequality-red.csv && \
		echo "data/raw/winequality-red.csv found." || \
		echo "WARNING: data/raw/winequality-red.csv missing."

## lint     : Run ruff linter on the source module
lint:
	$(PYTHON) -m ruff check $(MODULE)/

## test     : Run the test suite
test:
	$(PYTHON) -m pytest tests/ -v

## notebook : Launch Jupyter
notebook:
	$(PYTHON) -m jupyter lab $(NOTEBOOK)

## clean    : Remove compiled Python files and Jupyter checkpoints
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete."
