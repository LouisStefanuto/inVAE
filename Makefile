#################################################################################
# GLOBALS                                                                       #
#################################################################################

PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up Python environment using uv
.PHONY: create-environment
create-environment:
	uv venv --python $(PYTHON_VERSION)

## Install dependencies
.PHONY: install-requirements
install-requirements:
	uv sync --extra dev --extra docs
	uv lock

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Check typing with mypy
.PHONY: mypy
mypy:
	mypy --config-file=pyproject.toml .

## Run unit tests
.PHONY: test
test:
	python -m pytest tests

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Build and run docker container
.PHONY: docker
docker:
	docker compose build invae
	docker compose run --rm invae /bin/bash

## Serve project doc locally
.PHONY: docs
docs:
	mkdocs serve -a localhost:8001

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
