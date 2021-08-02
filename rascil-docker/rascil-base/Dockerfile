FROM python:3.7

LABEL author="Tim Cornwell <realtimcornwell@gmail.com>" \
      description="SKA Telescope RASCIL base root reference image" \
      license="Apache2.0"

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install packages needed to build rascil
#RUN apt-get update -y && apt-get install -y \
#    ca-certificates \
#    gosu git wget curl rsync nano \
#    libxml2-dev liblapack-dev libcfitsio-dev libgtkmm-3.0-dev \
#    cmake g++ build-essential gfortran libpng-dev python-numpy \
#    libboost-python-dev wcslib-dev \
#    python3 python3-dev python3-pip python3-setuptools python3-venv python3-wheel && \
#    apt-get clean -y && \
#    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

WORKDIR /
USER root

# This will clone into (and create) a directory at /rascil
# The data directory will not be loaded correctly until LFS is enabled
COPY dist/*whl ./rascil-latest-master-py-none-any.whl
RUN pip install rascil-latest-master-py-none-any.whl

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

RUN DUCC0_OPTIMIZATION=portable-debug pip3 install -r requirements.txt --upgrade && \
    DUCC0_OPTIMIZATION=portable-debug python setup.py install && \
    rm -rf /root/.cache

WORKDIR /rascil

RUN mkdir -p /usr/share/casacore/data/ && \
    rsync -avz --timeout=120 rsync://casa-rsync.nrao.edu/casa-data/geodetic  /usr/share/casacore/data/

# Use entrypoint script to create a user on the fly and avoid running as root.
RUN chmod +x /rascil/entrypoint.sh
ENTRYPOINT ["/rascil/entrypoint.sh"]
CMD ["/bin/bash"]
