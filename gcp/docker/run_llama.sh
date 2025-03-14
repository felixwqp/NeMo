# Wrapper to run the llama3 model
# Sample command:
#   TRAINING_DIR=/workspace/training_configs TRAINING_FILENAME=llama3-7b-vmg.yaml \
#   bash /workspace/run_llama.sh > /tmp/dist.log 2>&1 &

: "${NODE_RANK:?Must set NODE_RANK}"
: "${NNODES:?Must set NNODES}"
: "${MASTER_ADDR:?Must set MASTER_ADDR}"

: "${DATA_LOCAL_DIR:=/workspace}"
: "${TRAINING_DIR:=/workspace/training_configs}"
: "${TRAINING_FILENAME:=llama3-7b-vmg.yaml}"
: "${DATA_CACHE_GCS_DIR:=gs://nemo-megatron-demo/training-data/tokenized/bpe2gpt/wikipedia/}"
: "${DATA_CACHE_LOCAL_DIR:=/lssd/wikipedia/}"

# TODO(wfelix): move job name into launcher.
TIME=$(TZ="America/Los_Angeles" date +"%Y%m%d_%H%M%S")
job_name="nemo-test-${NNODES}-${TIME}"

sync_gcs_data() {
  FULL_TRAINING_CONFIG_PATH="$TRAINING_DIR/$TRAINING_FILENAME"


  if [[ -n $DATA_CACHE_GCS_DIR && "$DATA_CACHE_GCS_DIR" != "null" ]]; then
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

sync_gcs_data

/workspace/dist_run_entry.sh /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
--config-path=${TRAINING_DIR} \
--config-name=${TRAINING_FILENAME} \
trainer.num_nodes=${NNODES} \
model.tensor_model_parallel_size=4 \
trainer.max_steps=1 \
trainer.log_every_n_steps=1 \
trainer.val_check_interval=null \
trainer.limit_val_batches=1 \
exp_manager.explicit_log_dir=/tmp/$job_name \
model.tokenizer.vocab_file=${DATA_LOCAL_DIR}/gpt2-vocab.json \
model.tokenizer.merge_file=${DATA_LOCAL_DIR}/gpt2-merges.txt \
model.data.data_prefix="[1.0,/lssd/wikipedia/wikipedia-tokenized-for-gpt2]"