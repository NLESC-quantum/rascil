ARG IMAGE_PREFIX

FROM python:3.7-slim AS first

WORKDIR /

RUN apt-get update -y && apt-get install -y git-lfs git

RUN git clone https://gitlab.com/ska-telescope/external/rascil.git && \
    cd rascil && git lfs install && \
    git lfs pull

RUN ls -la

FROM ${IMAGE_PREFIX}/rascil-base:latest AS build

LABEL author="Tim Cornwell <realtimcornwell@gmail.com>" \
      description="SKA Telescope RASCIL full reference image" \
      license="Apache2.0"

WORKDIR /rascil

COPY --from=first /rascil/data ./data

# Set runtime environment variables.
ENV RASCIL=/rascil
ENV RASCIL_DATA=/rascil/data
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

#RUN apt-get update -y && apt-get install -y --no-install-recommends \
#    apt-get clean -y && \
#    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Use entrypoint script to create a user on the fly and avoid running as root.
COPY entrypoint.sh .
RUN chmod +x /rascil/entrypoint.sh
ENTRYPOINT ["/rascil/entrypoint.sh"]
CMD ["/bin/bash"]
