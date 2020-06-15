ARG DOCKER_REGISTRY_USER
ARG DOCKER_REGISTRY_HOST
FROM ${DOCKER_REGISTRY_HOST}/${DOCKER_REGISTRY_USER}/rascil-full-root:latest as build

LABEL author="Tim Cornwell <realtimcornwell@gmail.com>" \
      description="RASCIL reference image" \
      license="Apache2.0"

# Use entrypoint script to create a user on the fly and avoid running as root.
RUN chmod +x /rascil/entrypoint.sh
ENTRYPOINT ["/rascil/entrypoint.sh"]
CMD ["/bin/bash"]

