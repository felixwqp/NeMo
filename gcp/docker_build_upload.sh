#!/usr/bin/env bash
set -x

# --- Configuration ---
SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
DOCKER_TAG=init-0326-permission-fix

# --- Docker Image Options ---
IMAGE_TO_BUILD="both"  # Default: both, nemo, launcher

# --- Process command-line arguments (flags) ---
while [[ $# -gt 0 ]]; do
  case $1 in
    --image)
      shift
      IMAGE_TO_BUILD="$1"
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
  shift
done


# --- Nemo Image ---
NEMO_DOCKER_DIR="us-central1-docker.pkg.dev/tcpfastrak-staging/nemo-megatron-wfelix/nemo-local-build"
NEMO_DOCKER_PATH="$NEMO_DOCKER_DIR:$DOCKER_TAG"
NEMO_DOCKER_CTX="$SCRIPT_DIR/.."
NEMO_DOCKER_FILE="$NEMO_DOCKER_CTX/Dockerfile.ci"

# --- Launcher Image ---
LAUNCHER_DOCKER_DIR="us-central1-docker.pkg.dev/tcpfastrak-staging/nemo-megatron-wfelix/nemo-local-build-launcher"
LAUNCHER_DOCKER_PATH="$LAUNCHER_DOCKER_DIR:$DOCKER_TAG"
LAUNCHER_DOCKER_CTX="$SCRIPT_DIR/docker"
LAUNCHER_DOCKER_FILE="$LAUNCHER_DOCKER_CTX/Dockerfile.launcher"


# --- Main Script ---
if [[ "$IMAGE_TO_BUILD" == "both" || "$IMAGE_TO_BUILD" == "nemo" ]]; then
  echo "Building Nemo Docker Image..."
  DOCKER_BUILDKIT=1 docker build \
    --no-cache \
    --build-arg NEMO_TAG=c1a008c87ad1deb48e3cf6285b4c869ebb40ca3f \
    --build-arg NEMO_REPO=https://github.com/felixwqp/NeMo.git  \
    --build-arg IMAGE_LABEL=$DOCKER_TAG \
    --build-arg MLM_REPO=https://github.com/felixwqp/Megatron-LM.git \
    --build-arg MLM_TAG=0a0eeb539d1c262dfb16b8add7706c49ddeaaebe \
    -f $NEMO_DOCKER_FILE \
    -t $NEMO_DOCKER_PATH .  

  docker push $NEMO_DOCKER_PATH
  echo "Nemo Docker Image built and pushed successfully."
fi

if [[ "$IMAGE_TO_BUILD" == "both" || "$IMAGE_TO_BUILD" == "launcher" ]]; then
  echo "Building Launcher Docker Image..."
  DOCKER_BUILDKIT=1 docker build \
    --build-arg BASE_IMAGE=us-central1-docker.pkg.dev/tcpfastrak-staging/nemo-megatron-wfelix/nemo-local-build:init \
    -f $LAUNCHER_DOCKER_FILE \
    -t $LAUNCHER_DOCKER_PATH $LAUNCHER_DOCKER_CTX

  docker push $LAUNCHER_DOCKER_PATH
  echo "Launcher Docker Image built and pushed successfully."
fi

echo "Script execution completed."






# SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"



# # Local build 
# # Build dockerfile.ci
# DOCKER_TAG=init-0320
# DOCKER_DIR="us-central1-docker.pkg.dev/tcpfastrak-staging/nemo-megatron-wfelix/nemo-local-build"
# DOCKER_PATH="$DOCKER_DIR:$DOCKER_TAG"
# DOCKER_CTX="$SCRIPT_DIR/.."
# DOCKER_FILE="$DOCKER_CTX/Dockerfile.ci"

# DOCKER_BUILDKIT=1 docker build \
#   --no-cache \
#   --build-arg NEMO_TAG=c1a008c87ad1deb48e3cf6285b4c869ebb40ca3f \
#   --build-arg NEMO_REPO=https://github.com/felixwqp/NeMo.git  \
#   --build-arg IMAGE_LABEL=$DOCKER_TAG \
#   --build-arg MLM_REPO=https://github.com/felixwqp/Megatron-LM.git \
#   --build-arg MLM_TAG=0a14427596a4d9d9422a525a7deb70b8d9a82c64 \
#   -f $DOCKER_FILE \
#   -t $DOCKER_PATH .  

# docker push $DOCKER_PATH


# DOCKER_TAG=${DOCKER_TAG:-init}
# DOCKER_DIR="us-central1-docker.pkg.dev/tcpfastrak-staging/nemo-megatron-wfelix/nemo-local-build-launcher"
# DOCKER_PATH="$DOCKER_DIR:$DOCKER_TAG"
# DOCKER_CTX="$SCRIPT_DIR/docker"
# DOCKER_FILE="$DOCKER_CTX/Dockerfile.launcher"

# DOCKER_BUILDKIT=1 docker build \
#   --build-arg BASE_IMAGE=us-central1-docker.pkg.dev/tcpfastrak-staging/nemo-megatron-wfelix/nemo-local-build:init \
#   -f $DOCKER_FILE \
#   -t $DOCKER_PATH .

# docker push $DOCKER_PATH


# # Launcher docker

