# RASCIL Docker image

There are various docker files:

- rascil-base-root: A minimal RASCIL, without data, runs as root, suitable for singularity
- rascil-base: rascil-base suitable for running in docker
- rascil-full-root: Base with data, runs as root, suitable for singularity
- rascil-full: rascil-full suitable for running in docker

The -root files are needed to run singularity.

## Quick start

- `cd` into one of the subdirectories
- Build the image with `make build`
- Push the image to a docker registry `make push`
- To push the `:latest` tag use `make push_latest`
- To push a version tag without the git SHA use `make push_version` 

## Python packages to the docker image

- Add python packages to the requirements.txt file
- Update the version in the release file, before building and pushing the image
