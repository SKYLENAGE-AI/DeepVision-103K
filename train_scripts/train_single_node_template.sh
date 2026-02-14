#!/bin/bash
# =====================================================================
# GSPO Training Script (based on verl)
# =====================================================================
# Usage:
#   1. Fill in the placeholder variables marked with {YOUR_...} below
#   2. Adjust GPU settings section according to your cluster size
#   3. Run: bash train_template.sh
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

# ======================== Training Hyperparameters ============================
MODEL_PATH={YOUR_MODEL_PATH}
TRAIN_FILE={YOUR_TRAIN_FILE}
TEST_FILE={YOUR_TEST_FILE}

# # --------- 16 gpu setting --------------
# GEN_BATCH_SIZE=256  
# TRAIN_BATCH_SIZE=128
# MINI_BATCH_SIZE=32
# MICRO_BATCH_SIZE=16
# MAX_PROMPT_LENGTH=2048
# MAX_RESPONSE_LENGTH=16384
# N_ROLL=16
# TOTAL_EPOCHS=2

# # --------- 8 gpu setting --------------
GEN_BATCH_SIZE=128  
TRAIN_BATCH_SIZE=64
MINI_BATCH_SIZE=16
MICRO_BATCH_SIZE=8
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=16384
N_ROLL=16
TOTAL_EPOCHS=2

sp_size=1
use_dynamic_bsz=true
actor_ppo_max_token_len=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 2))
infer_ppo_max_token_len=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 3))
gen_tp=1
entropy_checkpointing=true

enable_filter_groups=true
filter_groups_metric=acc
max_num_gen_batches=10
enable_overlong_buffer=false
overlong_buffer_len=2000
overlong_penalty_factor=0.1

reward_manager=dapo
loss_mode=gspo
ADV_ESTIMATOR=grpo
USE_KL_IN_REWARD=False
USE_KL_LOSS=True
KL_COEF=0.001

LR=1e-6
CLIP_RATIO_LOW=0.0003
CLIP_RATIO_HIGH=0.0004

TEMPERATURE=1.0
TOP_P=1
TOP_K=-1
LOSS_AGG_MODE="seq-mean-token-mean"

python3 -u -m verl.trainer.my_main_dapo \
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
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
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
        reward_model.reward_manager=${reward_manager} \
        +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
        +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
        +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
        +reward_model.reward_kwargs.overlong_buffer_cfg.log=false \
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
        +trainer.train_shuffle=True \
        trainer.project_name="${PROJECT_NAME}" \
        trainer.experiment_name="${EXP_NAME}" \
        trainer.default_local_dir="${CKPTS_DIR}" \
        +trainer.validation_data_dir=${CKPTS_DIR} \
        +trainer.rollout_data_dir=${CKPTS_DIR} \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=$WORLD_SIZE \
        trainer.save_freq=10 \
        trainer.test_freq=-1 \
        trainer.total_epochs=$TOTAL_EPOCHS \
    2>&1 | tee $CKPTS_DIR/train.log
