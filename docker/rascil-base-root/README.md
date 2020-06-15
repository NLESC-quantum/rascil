# Experimental RASCIL Docker base image

Eventually this should be deprecated in favour of an `ska-docker` base image.

## Quick start

If making a change, update the version number in the `release` file.

Then, if the Pipenv file has been changed:

Build with: 

```bash
make clean_build
```

or, if only the Dockerfile has changed:

```bash
make build
```

Publish (to dockerhub) with:
 
```bash
make push
```

To remove old `rascil/rascil_base` image tags:

```bash
make rm_old
```

To remove all `rascil/rascil_base` image tags

```bash
make rm
```

