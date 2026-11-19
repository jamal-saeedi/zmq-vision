PYTHON ?= python

.PHONY: install install-torch install-tf install-llm test test-fast lint format format-check build check-dist demo
install:
	$(PYTHON) -m pip install -e '.[dev]'
install-torch:
	$(PYTHON) -m pip install -e '.[torch,dev]'
install-tf:
	$(PYTHON) -m pip install -e '.[tf,dev]'
install-llm:
	$(PYTHON) -m pip install -e '.[llm,dev]'
test:
	$(PYTHON) -m pytest --cov=zmq_inference --cov-report=term-missing
test-fast:
	$(PYTHON) -m pytest -m 'not torch and not tf and not llm' -q
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
