ARG IMAGE_PREFIX

FROM python:3.9-slim AS first

WORKDIR /

RUN apt-get update -y && apt-get install -y git-lfs git

RUN git clone --depth 1 https://gitlab.com/ska-telescope/external/rascil.git && \
    cd rascil && git lfs install && \
    git lfs pull

FROM ${IMAGE_PREFIX}/rascil-base:latest

LABEL author="Tim Cornwell <realtimcornwell@gmail.com>" \
      description="SKA Telescope RASCIL full reference image" \
      license="Apache2.0"

WORKDIR /rascil

# copy LFS data from first stage
COPY --from=first /rascil/data ./data

# Set runtime environment variables.
ENV RASCIL=/rascil
ENV RASCIL_DATA=/rascil/data
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

CMD ["/bin/bash"]
