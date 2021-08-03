ARG IMAGE_PREFIX
ARG IMAGE_TAG
FROM ${IMAGE_PREFIX}/rascil-base:${IMAGE_TAG} as build

LABEL author="Tim Cornwell <realtimcornwell@gmail.com>" \
      description="SKA Telescope RASCIL full reference image" \
      license="Apache2.0"

WORKDIR /rascil

# Set runtime environment variables.
ENV RASCIL=/rascil
ENV RASCIL_DATA=/rascil/data
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    git-lfs && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN git lfs install && \
    git lfs pull

# Use entrypoint script to create a user on the fly and avoid running as root.
COPY entrypoint.sh .
RUN chmod +x /rascil/entrypoint.sh
ENTRYPOINT ["/rascil/entrypoint.sh"]
CMD ["/bin/bash"]
