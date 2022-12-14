# simple makefile to simplify repetitive build env management tasks under posix
include docker/kubernetes/Makefile

PYTHON ?= python3
PYLINT ?= pylint
MAKE_DBG ?= ""
TESTS ?= tests/
FLAKE ?= flake8
NAME = rascil
IMG ?= $(NAME)
TAG ?= ubuntu18.04
DOCKER_IMAGE = $(IMG):$(TAG)
WORKER_RASCIL_DATA ?= /rascil/data
CURRENT_DIR = $(shell pwd)
JUPYTER_PASSWORD ?= changeme

CRED=\033[0;31m
CBLUE=\033[0;34m
CEND=\033[0m
LINE:=$(shell printf '=%.0s' {1..70})

# Set default docker registry user.
ifeq ($(strip $(DOCKER_REGISTRY_USER)),)
	DOCKER_REGISTRY_USER=ci-cd
endif

ifeq ($(strip $(DOCKER_REGISTRY_HOST)),)
	DOCKER_REGISTRY_HOST=artefact.skao.int
endif


# RASCIL data directory usualy found in ./data
RASCIL_DATA = $(CURRENT_DIR)/data

-include PrivateRules.mak

.DEFAULT_GOAL := help

clean:
	$(PYTHON) setup.py clean --all
	rm -rf dist

# Use bash shell with pipefail option enabled so that the return status of a
# piped command is the value of the last (rightmost) commnand to exit with a
# non-zero status. This lets us pipe output into tee but still exit on test
# failures.
SHELL = /bin/bash
.SHELLFLAGS = -o pipefail -c

all: test lint docs

.PHONY: all test lint docs

lint:
# outputs linting.xml
	pylint --exit-zero --extension-pkg-whitelist=numpy --ignored-classes=astropy.units,astropy.constants,HDUList --output-format=pylint_junit.JUnitReporter rascil > linting.xml
	pylint --exit-zero --extension-pkg-whitelist=numpy --ignored-classes=astropy.units,astropy.constants,HDUList --output-format=parseable rascil
	black --check .

docs:  ## build docs
# Outputs docs/build/html
	$(MAKE) -C docs/src html

test-singlepass:
	HOME=`pwd` py.test -n `python3 -c "import multiprocessing;print(multiprocessing.cpu_count());exit(0)"` \
	 tests --verbose \
	--junitxml unit-tests.xml \
	--cov rascil \
	--cov-report term \
	--cov-report html:coverage  \
	--cov-report xml:coverage.xml \
	--pylint --pylint-error-types=EF --durations=30

test:
	HOME=`pwd` py.test -n `python3 -c "import multiprocessing;print(multiprocessing.cpu_count());exit(0)"` \
	tests/apps tests/data_models tests/processing_components --verbose \
	--cov=rascil \
	--junitxml unit-tests-other.xml \
	--pylint --pylint-error-types=EF --durations=30
	coverage html -d coverage

test-dask:
	HOME=`pwd` py.test tests/workflows tests/apps_rsexecute --verbose \
	--cov=rascil \
	--junitxml unit-tests-dask.xml \
	--pylint --pylint-error-types=EF --durations=30
	coverage html -d coverage

test-gpu:
	HOME=`pwd` py.test tests/wagg_gpu --verbose \
	--cov=rascil \
	--junitxml unit-tests-gpu.xml \
	--pylint --pylint-error-types=EF --durations=30
	coverage html -d coverage

upgrade_pip:  ## make sure pip is up to date.
	pip install --upgrade pip

requirements: upgrade_pip  ## update and compile requirements
	pip install -U pip-tools
	pip-compile -U --output-file requirements.txt requirements.in
	pip-compile -U --output-file requirements-test.txt requirements-test.in
	pip-compile -U --output-file requirements-docs.txt requirements-docs.in

install_requirements: upgrade_pip
	pip install -r requirements-docs.txt
	pip install -r requirements-test.txt
	pip install -r requirements.txt
	pip install -e .
	pip freeze

update_requirements: requirements install_requirements

release-patch: ## Make patch release
	bumpver update --patch -n

release-minor: ## Make minor release
	bumpver update --minor -n

release-major: ## Make major release
	bumpver update --major -n
