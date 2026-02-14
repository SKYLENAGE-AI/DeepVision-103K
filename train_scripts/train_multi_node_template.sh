#!/bin/bash
# =====================================================================
# GSPO Multi-Node Training Script (based on verl + Ray)
# =====================================================================
# Usage:
#   1. Fill in the placeholder variables marked with {YOUR_...} below
#   2. Adjust GPU settings section according to your cluster size
#   3. Submit this script to each node in your cluster
#      (RANK, WORLD_SIZE, MASTER_ADDR should be set by your scheduler)
# =====================================================================


# ===================== SwanLab Dashboard =====================
export SWANLAB_API_KEY={YOUR_SWANLAB_API_KEY}
export SWANLAB_LOG_DIR=./swanlab_logs
export SWANLAB_MODE=cloud

# ===================== Hardware Communication =====================
export NCCL_IBEXT_DISABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_IB_HCA=mlx5
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1
export NCCL_P2P_LEVEL=SYS

# ========================= Storage Settings ============================
PROJECT_NAME={YOUR_PROJECT_NAME}
ALG_TYPE=gspo

BASE_MODEL={YOUR_BASE_MODEL}
PREFIX=${ALG_TYPE}-${BASE_MODEL}
EXP_NAME=${PREFIX}
CKPTS_DIR={YOUR_ROOT_PATH}/$ALG_TYPE/$PROJECT_NAME/$EXP_NAME
ENGINE=vllm

mkdir -p $CKPTS_DIR

# =================== Record MASTER_ADDR across nodes =====================
MASTER_PORT=2222
IP_FILE="$CKPTS_DIR/master_addr.txt"

if [ "$RANK" -eq 0 ]; then
    echo "[INFO][Rank0] Removing old IP file: $IP_FILE"
    rm -f "$IP_FILE"
fi

sleep 30

if [ "$RANK" -eq 0 ]; then
    echo "[INFO][Rank0] MASTER_ADDR from env: ${MASTER_ADDR}"
    echo "${MASTER_ADDR}" > "$IP_FILE"
else
    while [ ! -f "$IP_FILE" ]; do
        echo "[INFO][Worker RANK=$RANK] Waiting for $IP_FILE ..."
        sleep 2
    done
    MASTER_ADDR_FROM_FILE=$(cat "$IP_FILE")
    echo "[INFO][Worker RANK=$RANK] Got MASTER_ADDR from file: ${MASTER_ADDR_FROM_FILE}"
fi

# ======================== Training Hyperparameters ============================
MODEL_PATH={YOUR_MODEL_PATH}
TRAIN_FILE={YOUR_TRAIN_FILE}
TEST_FILE={YOUR_TEST_FILE}

# # --------- 64 gpu setting --------------
# GEN_BATCH_SIZE=1024  
# TRAIN_BATCH_SIZE=512
# MINI_BATCH_SIZE=256
# MICRO_BATCH_SIZE=64
# MAX_PROMPT_LENGTH=2048
# MAX_RESPONSE_LENGTH=16384
# N_ROLL=16
# TOTAL_EPOCHS=4

# # --------- 32 gpu setting --------------
GEN_BATCH_SIZE=512  
TRAIN_BATCH_SIZE=256
MINI_BATCH_SIZE=64
MICRO_BATCH_SIZE=32
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=16384
N_ROLL=16
TOTAL_EPOCHS=2

# # --------- 16 gpu setting --------------
# GEN_BATCH_SIZE=512  
# TRAIN_BATCH_SIZE=128
# MINI_BATCH_SIZE=32
# MICRO_BATCH_SIZE=16
# MAX_PROMPT_LENGTH=2048
# MAX_RESPONSE_LENGTH=16384
# N_ROLL=16
# TOTAL_EPOCHS=5

# # --------- 8 gpu setting --------------
# GEN_BATCH_SIZE=256  
# TRAIN_BATCH_SIZE=64
# MINI_BATCH_SIZE=16
# MICRO_BATCH_SIZE=8
# MAX_PROMPT_LENGTH=2048
# MAX_RESPONSE_LENGTH=16384
# N_ROLL=16
# TOTAL_EPOCHS=2

sp_size=1
use_dynamic_bsz=true
actor_ppo_max_token_len=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 2))
infer_ppo_max_token_len=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 3))
gen_tp=1
entropy_checkpointing=true

# Whether to filter by accuracy
enable_filter_groups=true
filter_groups_metric=acc
max_num_gen_batches=10

# ----------------- overlong -----------------
enable_overlong_buffer=false
overlong_buffer_len=2000
overlong_penalty_factor=1.0

reward_manager=dapo
loss_mode=gspo
ADV_ESTIMATOR=grpo
USE_KL_IN_REWARD=False
USE_KL_LOSS=True
KL_COEF=0.001

LR=1e-6
CLIP_RATIO_LOW=0.0003
CLIP_RATIO_HIGH=0.0004
LOSS_AGG_MODE="seq-mean-token-mean"

# Rollout inference parameters
TEMPERATURE=1.0
TOP_P=1
TOP_K=-1

# Validation inference parameters
VAL_TEMPERATURE=0.3
VAL_TOP_P=0.95

# ======================== Launch Ray Cluster ============================
if [ "$RANK" -eq 0 ]; then
    ray start --head --node-ip-address=$MASTER_ADDR --port=$MASTER_PORT
    ray job submit -- python3 -u -m verl.trainer.my_main_dapo \
        actor_rollout_ref.nccl_timeout=3600 \
        actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE} \
        actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
        algorithm.adv_estimator=$ADV_ESTIMATOR \
        data.train_files=$TRAIN_FILE \
        data.val_files=$TEST_FILE \
        data.gen_batch_size=$GEN_BATCH_SIZE \
        data.train_batch_size=$TRAIN_BATCH_SIZE \
        data.max_prompt_length=$MAX_PROMPT_LENGTH \
        data.max_response_length=$MAX_RESPONSE_LENGTH \
        data.filter_overlong_prompts_workers=128 \
        data.filter_overlong_prompts=true \
        data.truncation='error' \
        data.image_key=images \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.actor.optim.lr=$LR \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.checkpoint.save_contents=hf_model \
        actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
        actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
        actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
        actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
        actor_rollout_ref.actor.entropy_checkpointing=${entropy_checkpointing} \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$gen_tp \
        actor_rollout_ref.rollout.name=$ENGINE \
        actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
        actor_rollout_ref.rollout.enable_chunked_prefill=true \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
        actor_rollout_ref.rollout.n=$N_ROLL \
        actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
        actor_rollout_ref.rollout.top_p=${TOP_P} \
        actor_rollout_ref.rollout.top_k=${TOP_K} \
        actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE} \
        actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P} \
        actor_rollout_ref.rollout.val_kwargs.top_k=${TOP_K} \
        actor_rollout_ref.rollout.val_kwargs.do_sample=true \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        reward_model.reward_manager=${reward_manager} \
        +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
        +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
        +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
        +reward_model.reward_kwargs.overlong_buffer_cfg.log=true \
        +reward_model.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH} \
        algorithm.filter_groups.enable=${enable_filter_groups} \
        algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
        algorithm.filter_groups.metric=${filter_groups_metric} \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger=["console","swanlab"] \
        trainer.val_before_train=False \
        trainer.project_name="${PROJECT_NAME}" \
        trainer.experiment_name="${EXP_NAME}" \
        trainer.default_local_dir="${CKPTS_DIR}" \
        +trainer.validation_data_dir=${CKPTS_DIR} \
        +trainer.rollout_data_dir=${CKPTS_DIR} \
        trainer.n_gpus_per_node=$N_GPU \
        trainer.nnodes=$WORLD_SIZE \
        trainer.save_freq=10 \
        trainer.test_freq=20 \
        trainer.total_epochs=$TOTAL_EPOCHS \
    2>&1 | tee $CKPTS_DIR/train.log
else
    # Worker nodes join the Ray cluster using the master address from file
    ray start --address=${MASTER_ADDR_FROM_FILE}:$MASTER_PORT \
              --block
fi
