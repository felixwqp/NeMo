#!/bin/bash


: "${NODE_RANK:?Must set NODE_RANK}"
: "${NNODES:?Must set NNODES}"
: "${MASTER_ADDR:?Must set MASTER_ADDR}"
: "${VERBOSE_LOG:=false}"
: "${NCCL_LIB_DIR:='/usr/local/tcpxo/lib64'}"


# MPI commands
export GPUS_PER_NODE=8
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
# MASTER_PORT can be any free port in the master node.
export MASTER_PORT=9997


# Set TCPXO environment vars.
# TODO(wfelix): Replace with NCCL config profile
NCCL_LIB_DIR=${NCCL_LIB_DIR} source ${NCCL_LIB_DIR}/nccl-env-profile.sh

# These vars will be passed to the container during runtime.
echo MASTER_ADDR:$MASTER_ADDR
echo MASTER_PORT:$MASTER_PORT
echo NODE_RANK:$NODE_RANK
echo NNODES:$NNODES
echo GPUS_PER_NODE:$GPUS_PER_NODE
echo WORLD_SIZE:$WORLD_SIZE

set -x


if [[ "$VERBOSE_LOG" = 'true' ]]; then
  export TORCH_CPP_LOG_LEVEL=INFO
  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=ALL
fi


OMP_NUM_THREADS=12 HYDRA_FULL_ERROR=1 \
torchrun --nproc_per_node=$GPUS_PER_NODE \
  --nnodes=$NNODES \
  --rdzv-backend=static \
  --node_rank=$NODE_RANK \
  --rdzv_id=nemo_$WORLD_SIZE \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
"$@"