FROM python:3.9-slim AS first

WORKDIR /

RUN apt-get update -y && apt-get install -y git rsync
RUN git clone https://gitlab.com/ska-telescope/external/rascil.git

RUN mkdir -p /usr/share/casacore/data/ && \
    rsync -avz --timeout=120 rsync://casa-rsync.nrao.edu/casa-data/geodetic  /usr/share/casacore/data/

FROM python:3.9

LABEL author="Tim Cornwell <realtimcornwell@gmail.com>" \
      description="SKA Telescope RASCIL base root reference image" \
      license="Apache2.0"

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install packages needed to build rascil
RUN apt-get update -y && apt-get install -y gosu && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

WORKDIR /
USER root

COPY --from=first /usr/share/casacore/data /usr/share/casacore/data

# Install RASCIL from wheel built in the "build" stage of the CI pipeline
# The data directory will not be loaded correctly until LFS is enabled
COPY rascil-*whl .
RUN pip install --upgrade pip && \
    DUCC0_OPTIMIZATION=portable-debug pip install rascil-*.whl && \
    rm -rf /root/.cache

# Set runtime environment variables.
ENV RASCIL=/rascil
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8


# We choose to put the dask scratch files inside the container.
# It might be better to map it in with --volume
RUN mkdir -p /rascil/dask-worker-space && \
    chmod 777 /rascil/dask-worker-space && \
    mkdir -p /rascil/test_results && \
    chmod 777 /rascil/test_results

WORKDIR /rascil

# examples dir is needed because these are run to test the docker images
COPY --from=first /rascil/examples ./examples

# Use entrypoint script to create a user on the fly and avoid running as root.
COPY entrypoint.sh .
RUN chmod +x /rascil/entrypoint.sh
ENTRYPOINT ["/rascil/entrypoint.sh"]
CMD ["/bin/bash"]
