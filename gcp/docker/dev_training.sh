#!/bin/bash

set -x

# . "$(dirname "${BASH_SOURCE[0]}")/metrics.sh"

declare -A JOB_LABELS


: "${TRAINING_FILENAME:?Must set TRAINING_FILENAME}"
: "${TRAINING_DIR:?Must set TRAINING_DIR}"

# init_metrics() {
  
#   record_global_start_time
#   get_vm_metadata
#   get_auth_token
#   create_metric "run/launched" "CUMULATIVE" "INT64" "1" "job_class"
#   create_metric "run/succeeded" "CUMULATIVE" "INT64" "1" "job_class"
#   create_metric "run/cancelled" "CUMULATIVE" "INT64" "1" "job_class"
# }

#######################################
# Globals:
#   None
# Arguments:
#   None
# Sets:
#   GLOBAL_START_TIME
#   GLOBAL_ISO8601_START_TIME
#######################################
record_global_start_time() {
   GLOBAL_START_TIME=$(date +%s)
   GLOBAL_ISO8601_START_TIME=$(date -Iseconds)
   # Buckets of length 0 seem to get dropped by Cloud Monitoring,
   # so delay to make sure the interval for the first sample
   # is at least a few seconds long.
   sleep 5
   echo "Global start time at $(date) ($GLOBAL_START_TIME)"
}

run_nemo_single_node_slice() {
  JOB_LABELS["job_class"]="$TRAINING_FILENAME-$NNODES"
  # init_metrics
  local pids=()
  for ((LOCAL_RANK=0; LOCAL_RANK <= $((GPUS_PER_NODE - 1)); LOCAL_RANK++)); do
     RANK=$(($GPUS_PER_NODE*$NODE_RANK + $LOCAL_RANK))
     if cat "/workspace/training_configs/$TRAINING_FILENAME" | \
          grep -w deterministic | grep true; then
        NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
     fi
     echo NVTE_ALLOW_NONDETERMINISTIC_ALGO=$NVTE_ALLOW_NONDETERMINISTIC_ALGO

     read -ra extra_conf_arr <<<"$EXTRA_CONF_OVERRIDE"

     if [[ "$DEBUG_PYTHON" == "1" ]]; then
       echo "Using debug python. DEBUG_PYTHON=$DEBUG_PYTHON"
       PYTHON_EXEC=/usr/bin/python3.8-dbg
     else
       echo "Not using debug python. DEBUG_PYTHON=$DEBUG_PYTHON"
       PYTHON_EXEC=/usr/bin/python3
     fi

     OMP_NUM_THREADS=12 RANK=$RANK LOCAL_RANK=$LOCAL_RANK \
     eval nice -n -5 "$PROFILING_COMMAND" "$PYTHON_EXEC" \
      /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
      --config-path="/workspace/training_configs" \
      --config-name="$TRAINING_FILENAME" \
      trainer.num_nodes="$NNODES" \
      model.data.index_mapping_dir="$INDEX_MAPPING_DIR" \
      model.nsys_profile.enabled="$NSIGHT_PROFILE" \
      +exp_manager.version="$JOB_TIMESTAMP" \
      "${extra_conf_arr[@]}" &
     last_pid=$!
     pids+=($last_pid)

     echo "Launched megatron_gpt_pretraining.py for rank $RANK with PID $last_pid"
  done
  if (( NODE_RANK == 0 )); then
    emit_metric "run/launched" 1 JOB_LABELS
    log_event "run/launched"
  fi

  wait_all_success_or_exit "${pids[@]}"
}

run_nemo_torchrun() {
  JOB_LABELS["job_class"]="$TRAINING_FILENAME-$NNODES"
  # init_metrics

  if cat "/workspace/model_configs/$TRAINING_FILENAME" | \
      grep -w deterministic | grep true; then
    NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
  fi
  echo NVTE_ALLOW_NONDETERMINISTIC_ALGO=$NVTE_ALLOW_NONDETERMINISTIC_ALGO

  read -ra extra_conf_arr <<<"$EXTRA_CONF_OVERRIDE"
  # TODO: Make hardwired torchrun configs configurable.
  if [[ "$DEBUG_PYTHON" == "1" ]]; then
    echo "Using debug python. DEBUG_PYTHON=$DEBUG_PYTHON"
    PYTHON_EXEC=/usr/bin/python3.8-dbg
  else
    echo "Not using debug python. DEBUG_PYTHON=$DEBUG_PYTHON"
    PYTHON_EXEC=/usr/bin/python3
  fi
  echo "start to run"
  DATA_LOCAL_DIR="/workspace"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  OMP_NUM_THREADS=12 \
  eval nice -n -5 "$PROFILING_COMMAND" \
  torchrun \
  --nnodes="$NNODES" \
  --nproc-per-node=8 \
  --rdzv-id="$JOB_TIMESTAMP" \
  --node_rank=$NODE_RANK \
  --rdzv_backend c10d \
  --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
  --max-restarts=4 \
  /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
  --config-path="$TRAINING_DIR" \
  --config-name="$TRAINING_FILENAME" \
  trainer.num_nodes="$NNODES"  \
  model.tokenizer.vocab_file=${DATA_LOCAL_DIR}/gpt2-vocab.json \
  model.tokenizer.merge_file=${DATA_LOCAL_DIR}/gpt2-merges.txt \
  model.data.data_prefix="[1.0,/lssd/wikipedia/wikipedia-tokenized-for-gpt2]" \
  "${extra_conf_arr[@]}" > /tmp/dist.log 2>&1 &

  last_pid=$!
  echo "Launched megatron_gpt_pretraining.py with PID $last_pid"
  # if (( NODE_RANK == 0 )); then
  #   emit_metric "run/launched" 1 JOB_LABELS
  #   log_event "run/launched"
  # fi
  wait_all_success_or_exit "$last_pid"
}

wait_all_success_or_exit() {
  # https://www.baeldung.com/linux/background-process-get-exit-code
  local pids=("$@")
  while [[ ${#pids[@]} -ne 0 ]]; do
    all_success="true"
    for pid in "${pids[@]}"; do
      code=$(non_blocking_wait "$pid")
      if [[ $code -ne 127 ]]; then
        # See b/331632724.
        # TODO(yiinho,b/353524390): Find exact condition when $code can be -1.
        if [[ $code -ne 0 ]] && [[ $code -ne 155 ]] && [[ $code -ne -1 ]]; then
          echo "PID $pid failed with exit code $code"
          exit "$code"
        else
          echo "PID $pid exited with exit code $code, which we regard as success."
        fi
      else
        all_success="false"
      fi
    done
    if [[ $all_success == "true" ]]; then
      echo "All pids succeeded"

      if (( NODE_RANK == 0)); then
        get_auth_token
        emit_metric "run/succeeded" 1 JOB_LABELS
        log_event "run/succeeded"
      fi
      break
    fi
    sleep 5
  done
}

non_blocking_wait() {
  # https://www.baeldung.com/linux/background-process-get-exit-code
  local pid=$1
  local code=127 # special code to indicate not-finished
  if [[ ! -d "/proc/$pid" ]]; then
    wait "$pid"
    code=$?
  fi
  echo $code
}

# Execute main only if we are not in a unit test.
if [[ -z "$TEST_TARGET" ]]; then
  if [[ -n "$USE_TORCHRUN" ]]; then
    run_nemo_torchrun "$@"
  else
    run_nemo_single_node_slice "$@"
  fi
fi
