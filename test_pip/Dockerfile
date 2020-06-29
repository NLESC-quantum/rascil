ARG DOCKER_REGISTRY_USER
ARG DOCKER_REGISTRY_HOST
FROM ubuntu:18.04

LABEL author="Tim Cornwell <realtimcornwell@gmail.com>" \
      description="SKA Telescope RASCIL base root reference image" \
      license="Apache2.0"

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install packages needed to build rascil
RUN apt-get update -y && apt-get install -y \
    ca-certificates \
    gosu git wget curl rsync nano \
    libxml2-dev liblapack-dev libcfitsio-dev libgtkmm-3.0-dev \
    cmake g++ build-essential gfortran libpng-dev casacore-dev python-numpy \
    python-setuptools libboost-python-dev wcslib-dev \
    python3 python3-dev python3-pip python3-setuptools python3-venv python3-wheel && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

WORKDIR /
USER root

# This will clone into (and create) a directory at /rascil
# The data directory will not be loaded correctly until LFS is enabled
RUN git clone https://gitlab.com/ska-telescope/rascil.git

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

RUN pip3 install Cython setuptools && \
    pip3 install python-casacore==3.2.0 --upgrade && \
    pip3 install aotools \
    astroplan \
    astropy \
    dask \
    distributed \
    bokeh \
    h5py \
    jupyter \
    jupyter_contrib_nbextensions \
    jupyter-server-proxy \
    jupyterlab \
    matplotlib \
    numpy \
    photutils\
    reproject \
    scikit-image==0.16.2 \
    scipy \
    seqfile \
    tabulate && \
    python3 setup.py build && python3 setup.py install && \
    rm -rf /root/.cache

# Build the nifty gridder
RUN apt-get update -y && apt-get install -y \
    ca-certificates \
    pybind11-dev && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

WORKDIR /

RUN git clone https://gitlab.mpcdf.mpg.de/ift/nifty_gridder.git && \
    cd /nifty_gridder && \
    pip3 install --user .

WORKDIR /rascil

RUN mkdir /rascil/casadata && \
    chmod 777 /rascil/casadata && \
    rsync -avz rsync://casa-rsync.nrao.edu/casa-data/geodetic /var/lib/casacore/data

# Use entrypoint script to create a user on the fly and avoid running as root.
RUN chmod +x /rascil/entrypoint.sh
ENTRYPOINT ["/rascil/entrypoint.sh"]
CMD ["/bin/bash"]
