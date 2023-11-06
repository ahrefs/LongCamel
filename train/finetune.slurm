#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --output=%x-%j.out

set -x -e

module load python3
source ~/pyenv/deepspeed/bin/activate

SCRIPT_PATH="./train"
BASE_PATH="./output"
LOG_FILE=$BASE_PATH/output.txt
STDOUT_LOG_FILE=$BASE_PATH/stdout.txt

MODEL_NAME=meta-llama/Llama-2-7b-hf
TRAIN_DATA=./data/data1.jsonl,./data/data2.jsonl # tokenized data files sperated by ",", refer ./dataset.py for data file format
TRAIN_DATA_WEIGHT=0.5,0.5
EVAL_DATA=./data/eval_data1.jsonl,./data/eval_data2.jsonl
ZERO_STAGE=3
MICRO_BATCH=1
EVAL_MICRO_BATCH=1
BATCH_SIZE=64
NUM_EPOCHES=1
LEARNING_RATE=0.00002
WARMUP_STEPS=10
WEIGHT_DECAY=0.1
GRADIENT_CHECKPOINTING=true
ROPE_SCALING_TYPE=linear
ROPE_SCALING_FACTOR=8
MAX_SEQ_LEN=32768

DS_CONFIG=${BASE_PATH}/deepspeed_zero_config.json

mkdir -p $BASE_PATH

cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 1,

  "gradient_clipping": 1.0,

  "zero_optimization": {
    "stage": $ZERO_STAGE,
  },
  "bf16": {
   "enabled": true
  },
  "offload_optimizer": {"device": "cpu"},
  "offload_param": {"device": "cpu"},
}
EOT

GPUS_PER_NODE=8
NNODES=$SLURM_NNODES

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
export NCCL_ASYNC_ERROR_HANDLING=1


# zero++;
# "stage": 3, simply this works for zero 3 with bf16
cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 1,

  "gradient_clipping": 1.0,

  "zero_optimization": {
    "stage": $ZERO_STAGE,
    "reduce_scatter": true,
    "contiguous_gradients": true,
    "overlap_comm": true,
  },
  "bf16": {
   "enabled": true
  },
}
EOT

ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"

run_args=""
run_args=" ${run_args} --model_name $MODEL_NAME"
run_args=" ${run_args} --num_epoches $NUM_EPOCHES"
run_args=" ${run_args} --train_data $TRAIN_DATA"
run_args=" ${run_args} --train_data_weight $TRAIN_DATA_WEIGHT"
run_args=" ${run_args} --eval_data $EVAL_DATA"
run_args=" ${run_args} --train_micro_batch_size_per_gpu $MICRO_BATCH"
run_args=" ${run_args} --eval_micro_batch_size_per_gpu $EVAL_MICRO_BATCH"
run_args=" ${run_args} --train_batch_size $BATCH_SIZE"
run_args=" ${run_args} --lr $LEARNING_RATE"
run_args=" ${run_args} --warmup_steps $WARMUP_STEPS"
run_args=" ${run_args} --weight_decay $WEIGHT_DECAY"
run_args=" ${run_args} --save $BASE_PATH"
run_args=" ${run_args} --zero_stage $ZERO_STAGE"
run_args=" ${run_args} --gradient_checkpointing $GRADIENT_CHECKPOINTING"
run_args=" ${run_args} --rope_scaling_type $ROPE_SCALING_TYPE"
run_args=" ${run_args} --rope_scaling_factor $ROPE_SCALING_FACTOR"
run_args=" ${run_args} --max_seq_len $MAX_SEQ_LEN"

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role `hostname -s`: \
    --tee 3 \
    "

CMD="$SCRIPT_PATH/finetune.py ${run_args} ${ds_args}"

clear; srun --output=$STDOUT_LOG_FILE $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c " $LAUNCHER --node_rank \$SLURM_PROCID --role \$SLURMD_NODENAME: $CMD" 2>&1 | tee -a $LOG_FILE
