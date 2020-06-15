ARG DOCKER_REGISTRY_USER
ARG DOCKER_REGISTRY_HOST
FROM ${DOCKER_REGISTRY_HOST}/${DOCKER_REGISTRY_USER}/rascil-full-root:latest as build

LABEL author="Tim Cornwell <realtimcornwell@gmail.com>" \
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

RUN pip3 install dask-labextension jupyterlab \
    && jupyter labextension install @jupyter-widgets/jupyterlab-manager dask-labextension

# Use entrypoint script to create a user on the fly and avoid running as root.
RUN chmod +x /rascil/entrypoint.sh
ENTRYPOINT ["/rascil/entrypoint.sh"]
CMD ["/bin/bash"]

