CRED=\033[0;31m
CBLUE=\033[0;34m
CEND=\033[0m
LINE:=$(shell printf '=%.0s' {1..70})

IMAGE_PREFIX ?= artefact.skao.int
GIT_VERSION ?= $(shell git rev-parse --verify --short=8 HEAD)
GIT_ROOT_DIR=$(shell git rev-parse --show-toplevel)

# Get image name from the release file and version from rascil/version.py
NAME=$(shell awk -F= '/^name=/{print $$2}' release)
# VERSION=$(shell awk -F= '/^__version__ = /{print $$2}' $(GIT_ROOT_DIR)/rascil/version.py | sed -r "s/\\\"//g" | tr -d " ")
VERSION=0.3.0

IMAGE := $(IMAGE_PREFIX)/$(NAME)
DEFAULT_TAG := $(IMAGE):$(VERSION)-$(GIT_VERSION)

# Environment variables to capture for metadata
LABEL_ENVVARS := \
	CI_COMMIT_AUTHOR \
	CI_COMMIT_REF_NAME \
	CI_COMMIT_REF_SLUG \
	CI_COMMIT_SHA \
	CI_COMMIT_SHORT_SHA \
	CI_COMMIT_TIMESTAMP \
	CI_JOB_ID \
	CI_JOB_URL \
	CI_PIPELINE_ID \
	CI_PIPELINE_IID \
	CI_PIPELINE_URL \
	CI_PROJECT_ID \
	CI_PROJECT_PATH_SLUG \
	CI_PROJECT_URL \
	CI_REPOSITORY_URL \
	CI_RUNNER_ID \
	CI_RUNNER_REVISION \
	CI_RUNNER_TAGS \
	GITLAB_USER_EMAIL \
	GITLAB_USER_ID \
	GITLAB_USER_LOGIN \
	GITLAB_USER_NAME

# Construct --label flags
LABELS := $(foreach var,$(LABEL_ENVVARS),--label $(var)="$${$(var)}")

.DEFAULT_GOAL := help

build:  ## Build the image, tagged as :$(VERSION)-$(GIT_VERSION)
	@echo "$(CRED)$(LINE)$(CEND)"
	@echo "$(CBLUE)Building image:$(CEND) $(IMAGE):$(VERSION)-$(GIT_VERSION)"
	@echo "$(CRED)$(LINE)$(CEND)"
	docker build \
	--build-arg IMAGE_PREFIX=$(IMAGE_PREFIX) \
	$(LABELS) \
	-t $(DEFAULT_TAG) .

push: ## Push default image (tagged as :$(VERSION)-(GIT_VERSION)
	@echo "$(CBLUE)Pushing:$(CEND) $(IMAGE):$(VERSION)-$(GIT_VERSION)"
	docker push $(DEFAULT_TAG)

tag_latest:
	@echo "$(CBLUE)Tagging:$(CEND) $(DEFAULT_TAG) $(CRED)->$(CEND) $(IMAGE):latest"
	@docker tag $(DEFAULT_TAG) $(IMAGE):latest

push_latest: tag_latest ## Push the image tagged as :latest
	@echo "$(CBLUE)Pushing:$(CEND) $(IMAGE):latest"
	@docker push $(IMAGE):latest

tag_version:
	@echo "$(CBLUE)Tagging:$(CEND) $(DEFAULT_TAG) $(CRED)->$(CEND) $(IMAGE):$(VERSION)"
	@docker tag $(DEFAULT_TAG) $(IMAGE):$(VERSION)

push_version: tag_version  ## Push the image tagged as :$(VERSION) (without the git sha)
	@echo "$(CBLUE)Pushing:$(CEND) $(IMAGE):$(VERSION)"
	@docker push $(IMAGE):$(VERSION)

push_all: push push_latest push_version ## Push all tags

pull: ## Fetch the latest image
	@docker pull $(IMAGE):latest || true

pull_default: ## Fetch the default Git versioned image
	@docker pull $(DEFAULT_TAG) || true

ls:  ## List images built from this folder
	@echo "$(CBLUE)Images matching:$(CEND) $(IMAGE):*"
	@docker image ls --filter=reference="$(IMAGE):*"

rm:  ## Remove all images built from this folder
	@echo "$(CBLUE)Removing images matching:$(CEND) $(IMAGE):*"
	-@docker image rm $(shell docker image ls --filter=reference="$(IMAGE):*" --format '{{.Repository}}:{{.Tag}}') 2> /dev/null || true

help:  ## Show this help.
	@echo "$(CRED)$(LINE)$(CEND)"
	@echo "$(CBLUE)NAME        $(CEND): $(NAME)"
	@echo "$(CBLUE)IMAGE       $(CEND): $(IMAGE)"
	@echo "$(CBLUE)VERSION     $(CEND): $(VERSION)"
	@echo "$(CBLUE)GIT VERSION $(CEND): $(GIT_VERSION)"
	@echo "$(CBLUE)DEFAULT TAG $(CEND): $(DEFAULT_TAG)"
	@echo "$(CRED)$(LINE)$(CEND)"
	@echo "$(CBLUE)Targets:$(CEND)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' ../make/Makefile | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: piplock build push tag_latest push_latest tag_version push_version \
        push_all pull ls rm help
