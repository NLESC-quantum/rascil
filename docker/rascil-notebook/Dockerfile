ARG IMAGE_PREFIX
FROM ${IMAGE_PREFIX}/rascil-full:latest as build

LABEL author="Tim Cornwell <realtimcornwell@gmail.com>" \
      description="SKA Telescope RASCIL notebook image" \
      description="RASCIL reference image" \
      license="Apache2.0"

WORKDIR /rascil

RUN apt-get update -y && apt-get install -y \
    curl && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# We need to install nodejs for the subsequent jupyter lab features
RUN curl -sL https://deb.nodesource.com/setup_14.x | bash - && \
    apt-get install -y nodejs

# Jupyterlab does not currently support dask-extension
#RUN pip3 install dask-labextension jupyterlab \
#    && jupyter labextension install @jupyter-widgets/jupyterlab-manager dask-labextension

RUN pip install jupyterlab

CMD ["/bin/bash"]

