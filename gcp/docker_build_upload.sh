#!/usr/bin/env bash

SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"


# Local build 
# Build dockerfile.ci
# DOCKER_TAG=init
# DOCKER_DIR="us-central1-docker.pkg.dev/tcpfastrak-staging/nemo-megatron-wfelix/nemo-local-build"
# DOCKER_PATH="$DOCKER_DIR:$DOCKER_TAG"
# DOCKER_CTX="$SCRIPT_DIR/.."
# DOCKER_FILE="$DOCKER_CTX/Dockerfile.ci"

# DOCKER_BUILDKIT=1 docker build \
#   --build-arg NEMO_TAG=c80bd5c0c677c47f6ae495444431b15db42c0422 \
#   --build-arg NEMO_REPO=https://github.com/felixwqp/NeMo.git  \
#   --build-arg IMAGE_LABEL=init-0313 \
#   --build-arg MLM_REPO=https://github.com/felixwqp/Megatron-LM.git \
#   --build-arg MLM_TAG=ac3884aab91668eccd952916f8ffff9d272a09be \
#   -f $DOCKER_FILE \
#   -t $DOCKER_PATH .  

# docker push $DOCKER_PATH


DOCKER_TAG=${DOCKER_TAG:-init}
DOCKER_DIR="us-central1-docker.pkg.dev/tcpfastrak-staging/nemo-megatron-wfelix/nemo-local-build-launcher"
DOCKER_PATH="$DOCKER_DIR:$DOCKER_TAG"
DOCKER_CTX="$SCRIPT_DIR/docker"
DOCKER_FILE="$DOCKER_CTX/Dockerfile.launcher"

DOCKER_BUILDKIT=1 docker build \
  --build-arg BASE_IMAGE=us-central1-docker.pkg.dev/tcpfastrak-staging/nemo-megatron-wfelix/nemo-local-build:init \
  -f $DOCKER_FILE \
  -t $DOCKER_PATH .

docker push $DOCKER_PATH


# Launcher docker

