ARG IMAGE_PREFIX
ARG IMAGE_TAG
# imaging_qa checker needs rascil data, so we need rascil-full
FROM ${IMAGE_PREFIX}/rascil-full:${IMAGE_TAG}

LABEL author="Gabriella Hodosan <hodosan.gabriella@gmail.com>" \
      description="SKA Telescope RASCIL Continuum Imaging QA tool" \
      license="Apache2.0"

WORKDIR /rascil

ENV CLI_ARGS='-h'
ENV PYTHONPATH=$RASCIL:$PYTHONPATH

ENTRYPOINT python /rascil/rascil/apps/imaging_qa_main.py $CLI_ARGS
