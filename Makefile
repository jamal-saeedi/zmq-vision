PYTHON ?= python

.PHONY: install install-torch install-tf test test-fast lint format format-check build check-dist demo
install:
	$(PYTHON) -m pip install -e '.[dev]'
install-torch:
	$(PYTHON) -m pip install -e '.[torch,dev]'
install-tf:
	$(PYTHON) -m pip install -e '.[tf,dev]'
test:
	$(PYTHON) -m pytest --cov=zmq_inference --cov-report=term-missing
test-fast:
	$(PYTHON) -m pytest -m 'not torch and not tf' -q
lint:
	$(PYTHON) -m ruff check .
format:
	$(PYTHON) -m ruff format .
format-check:
	$(PYTHON) -m ruff format --check .
build:
	$(PYTHON) -m build
check-dist:
	$(PYTHON) -m twine check dist/*
demo:
	$(PYTHON) -m zmq_inference demo
