cd /mnt/workspace/zichen.shx/verl

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /mnt/workspace/zichen.shx/ckpt/grpo/mimo-multimath300k-wrong_cases_grpo/grpo-mimo-sft-2508-filter_v1/global_step_200/actor \
    --target_dir /mnt/workspace/zichen.shx/ckpt/grpo/mimo-multimath300k-wrong_cases_grpo/grpo-mimo-sft-2508-filter_v1/global_step_200/hf