#!/bin/bash
set -e
set -o pipefail

# This file is for setup only, e.g. set env var, sync gcs data, etc.

: "${NNODES:?Must set NNODES}"
: "${NODE_RANK:?Must set NODE_RANK}"
: "${TRAINING_FILENAME:?Must set TRAINING_FILENAME}"
: "${TRAINING_DIR:?Must set TRAINING_DIR}"

main() {
  verify_environment
  

  trap on_script_completion EXIT

  # TODO: evaluate.
  # set_nemo_specific_configuration
  share_config

  # if [[ "${USE_GCS_BUCKET:-yes}" == "yes" ]]; then
  #   create_ram_disk
  #   mount_gcs_bucket
  # fi
  # TODO: Enable gcs on training.
  sync_gcs_data

  # TODO: enable the NCCL setup.
  # Both are replaced by dist_run_entry.sh
  # set_nccl_specific_configuratio
  # set_torch_specific_configuration
  # set_torch_distributed_profiling_configuration
  # TODO: skip the profiling command.
  # set_nsight_profiling_configuration
  # set_heartbeat_metrics_configuration
  # run_nemo_megatron_pretraining

  # TODO: remove sleep
  sleep inf
}

# run_irq_balance() {
#   if [[ -n "$RUN_IRQ_BALANCE" ]]; then
#     irqbalance
#   fi
# }

verify_environment() {
  if [[ -n "$DPG_NUM" ]]; then
    : "${IN_DPG_INDEX_SOURCE:?Must set IN_DPG_INDEX_SOURCE}"
    : "${DPGS_TOTAL:?Must set DPGS_TOTAL}"
    : "${DPG_SIZE:?Must set DPG_SIZE}"
    IN_DPG_INDEX=${IN_DPG_INDEX_SOURCE##*-}
    export NNODES=$((DPGS_TOTAL * DPG_SIZE))
    export NODE_RANK=$((DPG_NUM * DPG_SIZE + IN_DPG_INDEX))
    echo "Calculated NNODES=$NNODES from DPGS_TOTAL($DPGS_TOTAL) * DPG_SIZE($DPG_SIZE)"
    echo "Calculated NODE_RANK=$NODE_RANK from DPG_NUM($DPG_NUM) * DPG_SIZE($DPG_SIZE) + IN_DPG_INDEX($IN_DPG_INDEX)"
  fi

  : "${NNODES:?Must set NNODES}"
  : "${NODE_RANK:?Must set NODE_RANK}"
  : "${TRAINING_FILENAME:?Must set TRAINING_FILENAME}"
  # : "${JOB_TIMESTAMP:?Must set JOB_TIMESTAMP}"
  # : "${IMAGE_VERSION:?Must set IMAGE_VERSION}"
  # : "${USE_TCPX:?Must set USE_TCPX}"
  # : "${USE_FASTRAK:?Must set USE_FASTRAK}"

  USE_TRAINING_INDEX_CACHE="${USE_TRAINING_INDEX_CACHE:-false}"
}

create_ram_disk() {
  RAMDISK_SIZE="800g"
  RAMDISK_DIR="/mnt/ram-disk"
  if [[ -d $RAMDISK_DIR ]]; then
    echo "Using pre-mounted $RAMDISK_DIR cache"
  else
    mkdir -p $RAMDISK_DIR

    mount -t tmpfs -o size=$RAMDISK_SIZE tmpfs $RAMDISK_DIR
    mount | grep $RAMDISK_DIR
    echo "Created $RAMDISK_DIR cache"
  fi
}

mount_gcs_bucket() {
  GCS_BUCKET="${GCS_BUCKET:-megatron-data-us}"
  mkdir -p "$RAMDISK_DIR/gcs"
  gcsfuse --client-protocol http2 --temp-dir "$RAMDISK_DIR/gcs" "$GCS_BUCKET" /gcs
}

sync_gcs_data() {
  DATA_CACHE_GCS_DIR=$(yq -r ".model.data.data_cache_gcs_dir" "$FULL_TRAINING_CONFIG_PATH")

  if [[ -n $DATA_CACHE_GCS_DIR && "$DATA_CACHE_GCS_DIR" != "null" ]]; then
    DATA_CACHE_LOCAL_DIR=$(yq -r ".model.data.data_cache_local_dir" "$FULL_TRAINING_CONFIG_PATH")
    echo "Caching training data from $DATA_CACHE_GCS_DIR to $DATA_CACHE_LOCAL_DIR"
    mkdir -p "$DATA_CACHE_LOCAL_DIR"

    SECONDS=0
    gcloud storage rsync \
      --recursive \
      "$DATA_CACHE_GCS_DIR" \
      "$DATA_CACHE_LOCAL_DIR"
    duration=$SECONDS
    echo "Transferred or synchronized $DATA_CACHE_GCS_DIR to $DATA_CACHE_LOCAL_DIR in $duration seconds."
  fi
}

launch_device_monitoring() {
  mkdir -p /usr/share/nemo
  rm -f /tmp/workload_terminated
  rm -f /usr/share/nemo/workload_terminated
  bash /workspace/monitor_hardware_util.sh &
}

set_network_specific_configuration() {
  sysctl -w net.ipv4.tcp_mtu_probing=0

  echo "Running tune_a3_settings.sh"
  bash /workspace/tune_a3_settings.sh
}

set_nccl_specific_configuration() {
  if [[ "$LOGGING_DESTINATION" ==  "local" ]]; then
    mkdir -p /var/log
    export NCCL_DEBUG_FILE="/var/log/nemo-%h-%p.log"
    echo "Writing NCCL debug to /var/log/nemo-%h-%p.log"
    export NCCL_TOPO_DUMP_FILE="/var/log/nccl_dump.topo"
    export NCCL_GRAPH_DUMP_FILE="/var/log/nccl_dump.graph"
  elif [[ "$LOGGING_DESTINATION" ==  "gcs" ]]; then
    export NCCL_DEBUG_FILE="$EXPERIMENT_LOG_PATH/nemo-%h-%p.log"
    echo "Writing NCCL debug to $EXPERIMENT_LOG_PATH/nemo-%h-%p.log"
    export NCCL_TOPO_DUMP_FILE="$EXPERIMENT_LOG_PATH/nccl_dump.topo"
    export NCCL_GRAPH_DUMP_FILE="$EXPERIMENT_LOG_PATH/nccl_dump.graph"
  else
    echo "Writing NCCL debug to stdout"
    export NCCL_TOPO_DUMP_FILE="$EXPERIMENT_LOG_PATH/nccl_dump.topo"
    export NCCL_GRAPH_DUMP_FILE="$EXPERIMENT_LOG_PATH/nccl_dump.graph"
  fi

  if [[ "$USE_FASTRAK" == "true" ]]; then
    echo "Using FASTRAK"

    if [[ -n "$USE_EXTERNAL_NCCL_ENV" ]]; then
      return
    fi

    # From google3/cloud/cluster/mlnet/release/nccl_fastrak_dockerbuild/container_setup/scripts/run-nccl-tcpxo.sh
    export NCCL_FASTRAK_CTRL_DEV=eth0
    export NCCL_FASTRAK_IFNAME=eth1,eth2,eth3,eth4,eth5,eth6,eth7,eth8
    export NCCL_SOCKET_IFNAME=eth0
    export NCCL_CROSS_NIC=0
    export NCCL_ALGO="${NCCL_ALGO:-Tree}"
    export NCCL_PROTO=Simple
    if [[ -z "$UNSET_NCCL_MAX_NCHANNELS" ]]; then
      export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-16}"
    fi
    export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-16}"
    export NCCL_SOCKET_NTHREADS=4
    export NCCL_DYNAMIC_CHUNK_SIZE=524288
    export NCCL_DYNAMIC_CHUNK_SIZE=524288
    export NCCL_P2P_NET_CHUNKSIZE=524288
    export NCCL_P2P_PCI_CHUNKSIZE=524288
    export NCCL_P2P_NVL_CHUNKSIZE=1048576
    export NCCL_FASTRAK_NUM_FLOWS="${NCCL_FASTRAK_NUM_FLOWS:-8}"
    export NCCL_FASTRAK_FLOWS_PER_GROUP="${NCCL_FASTRAK_FLOWS_PER_GROUP:-2}"
    export NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-4194304}
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export NCCL_NET_GDR_LEVEL=PIX
    if [[ "$DETAILED_FASTRAK_LOGGING" == "true" ]]; then
      # FasTrak debugging settings
      export NCCL_DEBUG_SUBSYS=ALL
      export NCCL_DEBUG="${LOGGING_LEVEL_OVERRIDE:-TRACE}"
      export NCCL_FASTRAK_ENABLE_HOTPATH_LOGGING=1
    else
      export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV,TUNING,NET,VERSION
      export NCCL_DEBUG="${LOGGING_LEVEL_OVERRIDE:-INFO}"
      export NCCL_FASTRAK_ENABLE_HOTPATH_LOGGING=0
    fi
    echo "Sourcing ${NCCL_LIB_DIR}/nccl-env-profile.sh"
    source "${NCCL_LIB_DIR}/nccl-env-profile.sh"
  elif [[ "$USE_TCPX" == "true" ]]; then
    echo "Using TCPX"

    export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV,TUNING,NET,VERSION
    export NCCL_DEBUG="${LOGGING_LEVEL_OVERRIDE:-INFO}"

    mkdir /usr/local/tcpx_exec
    mount --bind /usr/local/tcpx_exec /usr/local/tcpx_exec
    mount -o remount,exec /usr/local/tcpx_exec
    cp -r /usr/local/tcpx/lib64 /usr/local/tcpx_exec
    export LD_LIBRARY_PATH="/usr/local/tcpx_exec/lib64:${LD_LIBRARY_PATH}"

    if [[ -n "$USE_EXTERNAL_NCCL_ENV" ]]; then
      return
    fi

    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

    export NCCL_CHECK_POINTERS=0
    export NCCL_GRAPH_MIXING_SUPPORT=0
    if [[ -z "${NCCL_SOCKET_IFNAME}" ]]; then
      export NCCL_SOCKET_IFNAME=eth0
    fi
    if [[ -z "${NCCL_GPUDIRECTTCPX_CTRL_DEV}" ]]; then
      export NCCL_GPUDIRECTTCPX_CTRL_DEV=eth0
    fi
    if [[ -z "${NCCL_GPUDIRECTTCPX_SOCKET_IFNAME}" ]]; then
      export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2,eth3,eth4
    fi

    export NCCL_GPUDIRECTTCPX_TX_BINDINGS="eth1:8-21,112-125;eth2:8-21,112-125;eth3:60-73,164-177;eth4:60-73,164-177"
    export NCCL_GPUDIRECTTCPX_RX_BINDINGS="eth1:22-35,126-139;eth2:22-35,126-139;eth3:74-87,178-191;eth4:74-87,178-191"

    export NCCL_MAX_NCHANNELS=8
    export NCCL_MIN_NCHANNELS=8
    export NCCL_SOCKET_NTHREADS=4
    export NCCL_NSOCKS_PERTHREAD=4

    export NCCL_DYNAMIC_CHUNK_SIZE=524288
    export NCCL_P2P_NET_CHUNKSIZE=524288
    export NCCL_P2P_PCI_CHUNKSIZE=524288
    export NCCL_P2P_NVL_CHUNKSIZE=1048576

    export NCCL_CROSS_NIC=0
    export NCCL_ALGO="${NCCL_ALGO:-Ring}"
    export NCCL_PROTO=Simple
    export NCCL_NET_GDR_LEVEL=PIX
    export NCCL_P2P_PXN_LEVEL=0

    export NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS=1000000
    export NCCL_GPUDIRECTTCPX_FORCE_ACK=0
    export NCCL_GPUDIRECTTCPX_TX_COMPLETION_NANOSLEEP=1000
  else
    echo "NOT using TCPX"
    export NCCL_TOPO_FILE=/workspace/network_configs/a3_cos_topo_modified.xml
  fi
}

set_torch_specific_configuration() {
  export MASTER_PORT=6002
  export GPUS_PER_NODE=8
  export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
  export HYDRA_FULL_ERROR=1
}

set_torch_distributed_profiling_configuration() {
  if [[ "$TORCH_DISTRIBUTED_TRACING" == "1" \
        || "$TORCH_DISTRIBUTED_TRACING" == "CROSSNODE" \
        || "$TORCH_DISTRIBUTED_TRACING" == "ALL" ]]; then
    pushd /opt/NeMo
    git apply /workspace/patches/enable_torch_distributed_tracing.patch
    popd

    # Note: APEX's distributed adam invokes torch.distributed indirectly.
    # For now, it is easiest to patch it in order to pick-up these messages.
    # These are the key all-reduce = reduce-scatter + all-gather within a DPG.
    patch \
      /usr/local/lib/python3.8/dist-packages/apex/contrib/optimizers/distributed_fused_adam.py \
      -i /workspace/patches/distributed_fused_adam.py.patch
  fi
}

set_heartbeat_metrics_configuration() {
  # TODO: Change workload obs condition to yes after it is disabled from workload observability container
  if [[ "$PER_PROCESS_HEARTBEAT" == "yes" ]]; then
    if [[ -n "$USE_TORCHRUN" ]]; then
      echo "** Not setting heartbeat metrics configuration for torchrun **"

    elif [[ "$TORCH_DISTRIBUTED_TRACING" == "1" \
          || "$TORCH_DISTRIBUTED_TRACING" == "CROSSNODE" \
          || "$TORCH_DISTRIBUTED_TRACING" == "ALL" ]]; then
          echo "** Applying heartbeat metrics configuration for torch distributed tracing **"
          pushd /opt/NeMo
          git apply /workspace/patches/report_heartbeat_metrics_secondary.patch
          popd
          echo "** Heartbeat report patch applied successfully **"
    else
      echo "** Setting heartbeat metrics configuration for single node slice run **"
      pushd /opt/NeMo
      git apply /workspace/patches/report_heartbeat_metrics_primary.patch
      popd
      echo "** Heartbeat report patch applied successfully **"
    fi
  fi
}

set_nemo_specific_configuration() {
  # TODO: Move this to a separate script
  FULL_TRAINING_CONFIG_PATH="$TRAINING_DIR/$TRAINING_FILENAME"
  TRAINING_FOLDER_FROM_FILENAME=$(echo "${TRAINING_FILENAME}" | sed 's/\./_/g')
  if [[ "$USE_TRAINING_INDEX_CACHE" == "true" ]]; then
    TRAINING_CONFIG_HASH=$(sha1sum "${FULL_TRAINING_CONFIG_PATH}" | awk '{print $1}')
  else
    TRAINING_CONFIG_HASH="$JOB_TIMESTAMP"
  fi


  INDEX_BASE_PATH="/gcs/nemo/index_mapping_files"
  export INDEX_MAPPING_DIR="$INDEX_BASE_PATH/$IMAGE_VERSION/$TRAINING_FOLDER_FROM_FILENAME/${NNODES}_node/$TRAINING_CONFIG_HASH"

  EXPERIMENT_ROOT_PATH=$(yq -r ".exp_manager.exp_dir" "$FULL_TRAINING_CONFIG_PATH")
  EXPERIMENT_NAME=$(yq -r ".exp_manager.name" "$FULL_TRAINING_CONFIG_PATH")
  EXPERIMENT_LOG_PATH=$EXPERIMENT_ROOT_PATH/$EXPERIMENT_NAME/$JOB_TIMESTAMP
  echo "NeMO logs from each rank will be emitted to $EXPERIMENT_LOG_PATH"
}

share_config () {
  echo "Sharing config with other containers"
  mkdir -p /usr/share/nemo/workload_configs
  FULL_TRAINING_CONFIG_PATH="$TRAINING_DIR/$TRAINING_FILENAME"
  cp "$FULL_TRAINING_CONFIG_PATH" /usr/share/nemo/workload_configs
}

#######################################
# Globals:
#   NSIGHT_PROFILE - optional
#   EXPERIMENT_LOG_PATH – must be set if NSIGHT_PROFILE is true, usually by set_nemo_specific_configuration, before this routine is called
#   FULL_TRAINING_CONFIG_PATH – must be set, usually by set_nemo_specific_configuration, before this routine is called
#   JOB_TIMESTAMP – must be set, usually by the Kubernetes job spec, before this routine is called
#   RANK – must be set on invoking PROFILING_COMMAND in order for profile filename to indicate GPU rank
# Arguments:
#   None
# Sets:
#   None
# Exports:
#   PROFILING_COMMAND
#   NSIGHT_PROFILE
#######################################
set_nsight_profiling_configuration() {
  pushd /opt/NeMo
  git apply /workspace/patches/enable_nemo_nvtx_ranges.patch
  popd
  NSIGHT_CONFIG=$(yq -r ".model.nsys_profile.enabled" "$FULL_TRAINING_CONFIG_PATH")
  if [[ $NSIGHT_CONFIG == "true" ]] || \
     [[ $NSIGHT_PROFILE == "1" ]] || \
     [[ ${NSIGHT_PROFILE,,} == "true" ]]; then
    export NSIGHT_PROFILE="true"

    pushd /workspace/Megatron-LM
    git apply /workspace/patches/enable_megatron_nvtx_ranges.patch
    popd

    # When using one profile per GPU:
    PROFILE_OUTPUT="$EXPERIMENT_LOG_PATH/nemo-$JOB_TIMESTAMP-%q{RANK}"

    echo "Emitting NSIGHT profile to $PROFILE_OUTPUT"
    CMD="nsys profile"
    CMD+=" --wait primary"
    CMD+=" -o $PROFILE_OUTPUT"
    CMD+=" --force-overwrite true"
    CMD+=" -t cuda,nvtx"
    CMD+=" -s none"
    CMD+=" --capture-range=cudaProfilerApi --capture-range-end=stop"

    export PROFILING_COMMAND="$CMD"
  else
    export PROFILING_COMMAND=""
    export NSIGHT_PROFILE="false"
  fi
}

run_nemo_megatron_pretraining() {
  echo "Running NeMo with settings:"
  printenv
  NEMO_COMMAND="bash dev_training.sh"
  eval "$NEMO_COMMAND"
}

function on_script_completion {
   # Note: This semaphore is tracked by device monitoring script.
   touch /tmp/workload_terminated
   # Note: This semaphore is tracked by tcpx daemon container in gke
   mkdir -p /usr/share/nemo
   touch /usr/share/nemo/workload_terminated

   if [[ "$STALL_ON_COMPLETE" == "yes" ]]; then
     while true
     do
       echo "Run completed (or failed), awaiting manual garbage collection"
       sleep 3600
     done;
   fi;
}

# Execute main only if we are not in a unit test.
if [[ -z "$TEST_TARGET" ]]; then
  main "$@"; exit
fi
