# simple makefile to simplify repetitive build env management tasks under posix
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
	DOCKER_REGISTRY_HOST=nexus.engageska-portugal.pt
endif


# RASCIL data directory usualy found in ./data
RASCIL_DATA = $(CURRENT_DIR)/data

-include PrivateRules.mak

.DEFAULT_GOAL := help

clean: cleantests
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
	# FIXME pylint needs to run twice since there is no way go gather the text and junit xml output at the same time
	mkdir -p ./build/reports/
	pylint --exit-zero --output-format=pylint2junit.JunitReporter rascil > ./build/reports/linting.xml
	pylint --exit-zero --output-format=parseable rascil | tee ./build/code_analysis.stdout

docs:  ## build docs
	mkdir -p ./build/reports/
	$(MAKE) -C docs/src dirhtml | tee ./build/setup_py_docs.stdout

test:
	mkdir -p ./build/reports/
	HOME=`pwd` py.test tests/workflows/test*rsexecute.py --verbose --cov=rascil --cov-report=xml:coverage \
		--cov-report=html:coverage --durations=30 --forked | tee ./build/setup_py_test.stdout
	HOME=`pwd` py.test -n 4 tests/data_models tests/processing_components tests/workflows/test*serial.py --verbose \
		--cov=rascil --cov-report=html:coverage --cov-report=xml:coverage --cov-append --durations=30 \
		 | tee -a ./build/setup_py_test.stdout

