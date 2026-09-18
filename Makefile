.DEFAULT_GOAL := check
.RECIPEPREFIX := >

PYTHON := .venv/bin/python
RUFF := .venv/bin/ruff
TEST ?=

.PHONY: check format format-check test

check:
> "$(RUFF)" check src tests

format:
> "$(RUFF)" format src tests

format-check:
> "$(RUFF)" format --check src tests

test:
ifeq ($(strip $(TEST)),)
> HOME_LLM_RUN_INTEGRATION=0 PYTHONPATH=src "$(PYTHON)" -B -m unittest discover -s tests -p 'test_*.py' -v
else
> HOME_LLM_RUN_INTEGRATION=0 PYTHONPATH=src:tests "$(PYTHON)" -B -m unittest "$(TEST)" -v
endif