ARG IMAGE_PREFIX

FROM ${IMAGE_PREFIX}/rascil-base-test:latest

LABEL author="SKA RASCIL Developers" \
      description="SKA Telescope RASCIL realtime-calibration image" \
      license="Apache2.0"

COPY vis_consumer /usr/local/lib/python3.9/site-packages/rascil/vis_consumer
COPY consumer_requirements.txt .

RUN pip install -r consumer_requirements.txt

WORKDIR /rascil

# Set runtime environment variables.
ENV RASCIL=/rascil
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

CMD ["/bin/bash"]
