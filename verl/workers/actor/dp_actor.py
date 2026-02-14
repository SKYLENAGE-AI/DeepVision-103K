# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()


    # 前向传播函数

    """
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                         _forward_micro_batch 完整流程                            │
    └─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                        ┌─────────────────┴─────────────────┐
                        │                                   │
                use_remove_padding?                   不使用 rmpad
                        │                                   │
                        ▼                                   ▼
    ┌───────────────────────────────┐      ┌─────────────────────────────────────────┐
    │ 1. unpad_input()              │      │ 直接前向:                                │
    │    input_ids → rmpad 格式     │       │   output = model(input_ids, mask, ...)  │
    │    返回 indices, cu_seqlens   │       │   logits = output.logits                │
    └───────────────────────────────┘      │   logits = logits[:, -resp_len-1:-1, :] │
            │                              │   log_probs = logprobs_from_logits()    │
            ▼                              └─────────────────────────────────────────┘
    ┌───────────────────────────────┐
    │ 2. roll() 准备 label          │
    │    shifted_ids = roll(ids,-1) │
    └───────────────────────────────┘
            │
            ▼
    ┌───────────────────────────────┐
    │ 3. (可选) Ulysses SP          │
    │    ulysses_pad_and_slice()    │
    └───────────────────────────────┘
            │
            ▼
    ┌───────────────────────────────┐
    │ 4. 模型前向                   │
    │    output = model(rmpad_ids)  │
    │    logits_rmpad = output      │
    └───────────────────────────────┘
            │
            ▼
    ┌───────────────────────────────┐
    │ 5. 计算 log_probs             │
    │    logprobs_from_logits()     │
    │    - flash_attn 版本          │
    │    - 或标准 logsumexp 版本    │
    └───────────────────────────────┘
            │
            ▼
    ┌───────────────────────────────┐
    │ 6. (可选) 计算 entropy        │
    │    entropy_from_logits()      │
    └───────────────────────────────┘
            │
            ▼
    ┌───────────────────────────────┐
    │ 7. (如果用 SP) gather_outputs │
    │    从所有 rank 收集           │
    └───────────────────────────────┘
            │
            ▼
    ┌───────────────────────────────┐
    │ 8. pad_input()                │
    │    恢复 (bs, seqlen) 形状     │
    └───────────────────────────────┘
            │
            ▼
    ┌───────────────────────────────┐
    │ 9. 截取 response 部分         │
    │    [:, -resp_len-1:-1]        │
    └───────────────────────────────┘
            │
            ▼
        返回 (entropy, log_probs)
        形状: (bs, response_len)
    """    

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            micro_batch: dict 包含
                - input_ids: (bs, seq_len)      # prompt + response
                - attention_mask: (bs, seq_len)
                - position_ids: (bs, seq_len)
                - responses: (bs, response_len) # 仅 response 部分
                - multi_modal_inputs: (可选) 多模态输入
            temperature: float, 温度参数，用于 softmax 缩放
            calculate_entropy: bool, 是否计算熵
        
        Returns:
            entropy: (bs, response_len) 或 None
            log_probs: (bs, response_len)
        """
        # 获取一下 response 的长度，这里是 max_response_length
        # 获取的方式是直接访问这个 tensor 最后一维的维度
        response_length = micro_batch["responses"].size(-1)

        # 处理多模态输入
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                # 标准多模态：把所有样本的多模态输入 concat 起来
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None

            #  qwen2vl 的方式可能和别的不一样，要把前两个维度转置
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                # 调用 flash_attn 的 unpad_input
                """
                def unpad_input(hidden_states, attention_mask):

                # 去除 padding tokens，只保留有效 token
                
                # Args:
                #     hidden_states: (batch, seqlen, ...) 
                #     attention_mask: (batch, seqlen)，1 表示有效，0 表示 padding
                
                # Returns:
                #     hidden_states_unpad: (total_nnz, ...)  # total_nnz = 所有有效 token 总数
                #     indices: (total_nnz,)  # 每个有效 token 在原始展平序列中的位置
                #     cu_seqlens: (batch+1,)  # 累积序列长度，用于 flash_attn_varlen
                #     max_seqlen: int  # batch 中最长序列的实际长度

                # 计算每个样本的有效 token 数
                seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)  # (batch,)
                
                # 找出所有有效位置的索引
                indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
                
                # 只保留有效位置的 hidden_states
                hidden_states_unpad = hidden_states.flatten(0, 1)[indices]  # (total_nnz, ...)
                
                # 计算累积长度（用于 flash_attn_varlen_func）
                cu_seqlens = torch.zeros(batch + 1, dtype=torch.int32)
                cu_seqlens[1:] = seqlens.cumsum(0)
                
                return hidden_states_unpad, indices, cu_seqlens, max(seqlens)

                """                
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)

                """                
                原始 input_ids (bs=2, seq_len=6):
                样本0: [A, B, C, PAD, PAD, PAD]   attention_mask: [1, 1, 1, 0, 0, 0]
                样本1: [D, E, F, G, H, PAD]       attention_mask: [1, 1, 1, 1, 1, 0]

                经过 unpad_input 后:
                input_ids_rmpad: [A, B, C, D, E, F, G, H]  # (8,) 去掉了所有 PAD
                indices: [0, 1, 2, 6, 7, 8, 9, 10]          # 原始展平位置
                cu_seqlens: [0, 3, 8]                        # 样本0长度3，样本1长度5
                """                
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                # 对 position ids 也需要做上面类似的操作
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                """
                自回归模型: logits[t] 预测 token[t+1]

                input_ids_rmpad:        [A, B, C, D, E, F, G, H]
                input_ids_rmpad_rolled: [B, C, D, E, F, G, H, A]  # 左移一位

                当计算 log_prob 时:
                log_prob[0] = log P(B | A)      # logits[0] 预测 rolled[0]=B
                log_prob[1] = log P(C | A,B)    # logits[1] 预测 rolled[1]=C
                ...
                """                
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # 处理 use_ulysses_sp 的情况，这个时候把多张卡拿来一起用
                # 相当于做 tp
                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                """
                input_ids_rmpad_rolled 本来 shape (1, total_nnz)
                去掉第一维（batch=1），得到 (total_nnz,)
                total_nnz 表示去掉 PAD 后的非零 token 数量；这里可能再加上特殊 token/sp 分片导致长度 (total_nnz / sp) + pad
                """                
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                
                # fused kernel 模式下（优化版推理），可以直接在 forward 里控制 temperature，并要求返回 dict 格式结果
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor

        """
        # set to eval
        self.actor_module.eval()

        # 计算 log_prob 的时候是以一个 micro_batch 为单位
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]


        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)


        # 使用动态批次的时候，根据 token 数重新分批次
        # 会得到新的一堆平衡之后的 micro_batch, 以及其中对应的 idx 的映射
        # 这里只是为了保证每张显卡的计算开销均等，后面还会根据 idx 把数据还回去
        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            # 如果不用 use_dynamic_bsz， 直接把数据 切分成多个 micro batch
            # 这里也是为什么 roll_out_n * mini_batch_size 要能整除 micro_batch_size 的原因
            # 不然我们没法进行 micro batch 切分了
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        # 遍历每个 micro batch，计算 log_prob
        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    micro_batch=model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        #  根据 idx 来恢复原始顺序
        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys


    """
        完整 Batch (例如 1024 samples)
            │
            ▼
    ┌─────────────────────────────────────────────────────┐
    │  Split into Mini-Batches (例如每个 256 samples)      │
    │  mini_batch_1, mini_batch_2, mini_batch_3, mini_batch_4 │
    └─────────────────────────────────────────────────────┘
            │
            │  ×ppo_epochs (例如 4 轮)
            ▼
    ┌─────────────────────────────────────────────────────┐
    │  For each Mini-Batch:                                │
    │  ┌───────────────────────────────────────────────┐  │
    │  │ Split into Micro-Batches (例如每个 64 samples) │  │
    │  │ micro_1, micro_2, micro_3, micro_4            │  │
    │  └───────────────────────────────────────────────┘  │
    │          │                                          │
    │          ▼                                          │
    │  ┌───────────────────────────────────────────────┐  │
    │  │ For each Micro-Batch:                         │  │
    │  │   1. Forward → 计算 log_prob                  │  │
    │  │   2. 计算 PPO loss                            │  │
    │  │   3. loss.backward() → 累积梯度               │  │
    │  └───────────────────────────────────────────────┘  │
    │          │                                          │
    │          ▼                                          │
    │  ┌───────────────────────────────────────────────┐  │
    │  │ optimizer.step() → 一次参数更新               │  │
    │  └───────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────┘

    === 问题 ===
    - 完整 batch 太大，GPU 放不下
    - 但我们希望用大 batch 的梯度来更新（更稳定）

    === 解决方案：梯度累积 ===

    Mini-Batch (256 samples): 一次参数更新使用的总样本数
        │
        ├─ Micro-Batch 1 (64 samples) → forward + backward → 累积梯度
        ├─ Micro-Batch 2 (64 samples) → forward + backward → 累积梯度
        ├─ Micro-Batch 3 (64 samples) → forward + backward → 累积梯度
        └─ Micro-Batch 4 (64 samples) → forward + backward → 累积梯度
                                                    │
                                                    ▼
                                        optimizer.step() (用累积的梯度更新)

    效果：等价于用 256 samples 的梯度更新，但 GPU 每次只需放 64 samples    

    on policy 和 off policy 问题
    在一个 mini batch 内 的所有 micro batch 共享同一个 old log prob，通过不同的 query 不断累积梯度
    如果 train batch 拆成的 mini batch 的数量越少，所谓 off policy 问题越小

    输入 data (1024 个样本)
            │
            ├─── mini_batch 0 (256)  ─┬─ micro_batch 0 (64) → loss.backward()
            │                         ├─ micro_batch 1 (64) → loss.backward()
            │                         ├─ micro_batch 2 (64) → loss.backward()
            │                         └─ micro_batch 3 (64) → loss.backward()
            │                                                     │
            │                                                     ▼
            │                                          optimizer.step() → θ₀ → θ₁
            │
            ├─── mini_batch 1 (256)  ─┬─ ... → loss.backward() × 4
            │                         └─ optimizer.step() → θ₁ → θ₂
            │
            ├─── mini_batch 2 (256)  ─┬─ ... → loss.backward() × 4
            │                         └─ optimizer.step() → θ₂ → θ₃
            │
            └─── mini_batch 3 (256)  ─┬─ ... → loss.backward() × 4
                                    └─ optimizer.step() → θ₃ → θ₄
                                                │
                                                ▼
                                        [Epoch 0 完成]
                                                │
    """
    # 策略更新
    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        """
        PPO 策略更新的核心函数
        
        流程：
        1. 将 batch 分成多个 mini-batch
        2. 对于每个 ppo_epoch:
        对于每个 mini-batch:
            - 将 mini-batch 分成多个 micro-batch
            - 对每个 micro-batch 计算 loss 并累积梯度
            - 累积完成后执行一次 optimizer.step()
        
        Args:
            data: 包含 responses, old_log_probs, advantages 等的数据
        
        Returns:
            metrics: 训练过程中的各种指标
        """

        # ==================== 第一部分：准备工作 ====================
        
        # 确保模型处于训练模式（启用 dropout 等）
        self.actor_module.train()

        # 获取生成时使用的温度参数
        temperature = data.meta_info["temperature"]
        
        # 选择需要的数据字段
        select_keys = [
            "responses",        # 生成的 response token ids
            "response_mask",    # response 的有效 token 掩码
            "input_ids",        # 完整输入 (prompt + response)
            "attention_mask",   # attention 掩码
            "position_ids",     # 位置编码
            "old_log_probs",    # 旧策略下的 log 概率（用于计算 ratio）
            "advantages",       # 优势估计值
        ]
        
        # 如果使用 KL loss，还需要参考模型的 log 概率
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        
        # 处理多模态输入（如图像）
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        
        # 只保留需要的字段
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)



        # ==================== 第二部分：分割 Mini-Batch ====================
        #
        # PPO 论文建议：将数据分成多个 mini-batch，多次遍历
        # 这样可以更充分地利用采集的数据
        # 
        # 例如：1024 samples → 4 个 mini-batch，每个 256 samples
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        metrics = {}

        # ==================== 第三部分：PPO Epochs 循环 ====================
        #
        # PPO 的一个特点：同一批数据可以用多个 epoch 更新
        # 因为有 clip 机制保护，不会偏离太远
        #
        # 典型值：ppo_epochs = 4
        for _ in range(self.config.ppo_epochs):

            # ==================== 第四部分：Mini-Batch 循环 ====================
            for batch_idx, mini_batch in enumerate(mini_batches):

                # ==================== 第五部分：分割 Micro-Batch ====================
                #
                # 由于 GPU 显存限制，需要将 mini-batch 进一步分割
                # 使用梯度累积技术：多个 micro-batch 的梯度累加后再更新

                if self.config.use_dynamic_bsz:
                    # 动态 batch：根据 token 数量而非样本数量分割
                    # 这样可以更好地利用 GPU 显存
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    # 静态 batch：固定样本数量分割
                    # 梯度累积步数 = mini_batch_size / micro_batch_size
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    # 把 mini batch 拆分成 mirco batch
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                # ==================== 第六部分：清零梯度 ====================
                #
                # 重要：在累积梯度之前，必须先清零
                # 否则会和上一个 mini-batch 的梯度混在一起
                self.actor_optimizer.zero_grad()

                # ==================== 第七部分：Micro-Batch 循环（梯度累积）====================
                for micro_batch in micro_batches:
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    # 确定是否考虑entropy 进入损失
                    # 使用何种方式计算损失
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # ==================== 第八部分：计算 Loss 缩放因子 ====================
                    #
                    # 梯度累积时，需要对每个 micro-batch 的 loss 进行缩放
                    # 这样累积后的梯度等价于用完整 mini-batch 计算的梯度
                    #
                    # 例如：4 个 micro-batch，每个 loss 乘以 1/4
                    # 累积后：grad = grad1/4 + grad2/4 + grad3/4 + grad4/4
                    #              = (grad1 + grad2 + grad3 + grad4) / 4
                    #              = 等价于 mini-batch 的平均梯度

                    if self.config.use_dynamic_bsz:
                        # 动态 batch：按样本数量比例缩放
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        # 静态 batch：按累积步数缩放
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    # ==================== 第九部分：前向传播 ====================
                    #
                    # 计算当前策略下的 log 概率和熵
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True

                    # log_prob: (batch_size, response_length) - 每个 token 的 log 概率
                    # entropy: (batch_size, response_length) - 每个 token 的熵
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    
                    # ==================== 第十部分：计算 PPO Loss ====================
                    #
                    # 根据配置选择不同的 loss 计算方式：
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla
                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # pg_loss: PPO 策略梯度 loss（标量）
                    # pg_clipfrac: 被 clip 的比例（监控）
                    # ppo_kl: 新旧策略 KL 散度（监控）
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                    )

                    # ==================== 第十一部分：添加熵正则化 ====================
                    #
                    # 熵正则化的作用：鼓励探索
                    # - 高熵 = 策略更随机 = 更多探索
                    # - 我们想最大化熵，所以在 loss 中减去熵
                    #
                    # 最终：loss = pg_loss - entropy_coeff * entropy
                    #            = pg_loss - 0.01 * entropy（典型值）
                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        # 总 loss = PPO loss - 熵 * 系数
                        # 减号是因为我们想最大化熵（鼓励探索）
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    # ==================== 第十二部分：添加 KL 惩罚 ====================
                    #
                    # KL 惩罚的作用：防止策略偏离参考模型太远
                    # - 参考模型通常是 SFT 模型
                    # - 避免 reward hacking（通过奇怪的方式获得高分）
                    #
                    # 最终：loss = pg_loss - entropy * coeff + kl_loss * coeff
                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        

                        # 总 loss += KL loss * 系数
                        # 目的是防止一步的更新过大
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    # ==================== 第十三部分：缩放 Loss 并反向传播 ====================
                    #
                    # 关键点：
                    # 1. loss 乘以 scale_factor，使得累积后的梯度正确
                    # 2. backward() 计算梯度并累积（因为没有 zero_grad）
                    # 3. 梯度存储在 param.grad 中，等待 optimizer.step()
                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor


                    loss.backward()
                    """
                    ┌─────────────────────────────────────────────────────────────────────────────────┐
                    │                        loss.backward() 执行过程                                  │
                    └─────────────────────────────────────────────────────────────────────────────────┘

                    前向传播时，PyTorch 构建计算图：

                    input_ids → Embedding → Transformer Layers → lm_head → logits
                                    │              │                 │
                                    W₁             W₂               W₃
                                    │              │                 │
                                    ▼              ▼                 ▼
                                保存中间结果用于反向传播

                                logits → log_softmax → log_prob → ratio → loss
                                                                            │
                                                                            ▼
                                                                        loss.backward()
                                                                            │
                                                                            ▼
                                ┌───────────────────────────────────────────────────────────┐
                                │ 反向传播（链式法则）:                                      │
                                │                                                           │
                                │ ∂loss/∂W₃ = ∂loss/∂logits × ∂logits/∂W₃                  │
                                │ ∂loss/∂W₂ = ∂loss/∂logits × ∂logits/∂hidden × ∂hidden/∂W₂│
                                │ ∂loss/∂W₁ = ...                                           │
                                │                                                           │
                                │ 梯度累积到每个参数的 .grad 属性:                           │
                                │   W₁.grad += ∂loss/∂W₁                                    │
                                │   W₂.grad += ∂loss/∂W₂                                    │
                                │   W₃.grad += ∂loss/∂W₃                                    │
                                └───────────────────────────────────────────────────────────┘
                    """         
                           
                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                # 每个 mini batch 做一次梯度更新
                # 每个 mini batch 的梯度是由每个 micro batch 累加得到的
                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
