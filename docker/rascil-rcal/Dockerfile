ARG IMAGE_PREFIX

FROM ${IMAGE_PREFIX}/rascil-base:latest

LABEL author="SKA RASCIL Developers" \
      description="SKA Telescope RASCIL realtime-calibration image" \
      license="Apache2.0"

COPY vis_consumer /usr/local/lib/python3.9/site-packages/rascil/vis_consumer
COPY consumer_requirements.txt .

RUN pip install -r consumer_requirements.txt

# casacore measures needed to calculate uvw coordiantes of spead2 packets
RUN mkdir -p /usr/share/casacore/data/ && \
    cd /usr/share/casacore/data && \
    wget -O WSRT_Measures.ztar ftp://ftp.astron.nl/outgoing/Measures/WSRT_Measures.ztar && \
    tar xfz WSRT_Measures.ztar

WORKDIR /rascil

# Set runtime environment variables.
ENV PATH=$PATH:/home/rascil/.local/bin
ENV RASCIL=/rascil
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

CMD ["/bin/bash"]
