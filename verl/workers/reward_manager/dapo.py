# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
'''
┌──────────────────────────────────────────────────────────────────────────────┐
│                     DAPORewardManager.__call__()                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  输入: DataProto (bs*n 条样本)                                                 │
│  输出: reward_tensor (bs*n, response_len)，只在最后一个 token 有值             │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
     ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
     │ 样本 0      │         │ 样本 1      │         │ 样本 ...    │
     └──────┬──────┘         └──────┬──────┘         └──────┬──────┘
            │                       │                       │
            ▼                       ▼                       ▼
     ┌─────────────────────────────────────────────────────────────┐
     │  Step 1: 提取 prompt 和 response                             │
     │    - 从 attention_mask 计算 valid_prompt_length              │
     │    - 从 attention_mask 计算 valid_response_length            │
     │    - 解码成字符串                                             │
     └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
     ┌─────────────────────────────────────────────────────────────┐
     │  Step 2: 调用 compute_score() 计算基础奖励                   │
     │    - 比对 response 和 ground_truth                           │
     │    - 返回 score（通常是 0 或 1）                              │
     └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
     ┌─────────────────────────────────────────────────────────────┐
     │  Step 3: 应用 Overlong Buffer 惩罚（可选）                   │
     │    - 如果 response 太长，扣分                                 │
     │    - reward += overlong_penalty                              │
     └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
     ┌─────────────────────────────────────────────────────────────┐
     │  Step 4: 放入 reward_tensor                                  │
     │    reward_tensor[i, valid_response_length - 1] = reward     │
     │    (只在最后一个有效 token 位置放 reward)                     │
     └─────────────────────────────────────────────────────────────┘

'''
from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("dapo")
class DAPORewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # 先看下有没有使用奖励模型评分的
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        # 新建一个 torch 张量，为每个 response 分配一个 reward
        # 形状是 (batch_size * n_roll, max_response_len)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            # 张量（tensor）的 shape 属性返回一个描述张量维度大小的元组。
            # 对于不同维度的张量，shape[-1] 表示最后一个维度的大小。
            # 假设张量的维度为 (batch_size, seq_len)，那么 shape[-1] 就等于 seq_len。
            # 注意这个 seq length 并不是真正的 prompt 的长度，而是设定的 max_prompt_len
            # 为了推理的方便，会把所有 prompt 左 padding 到 max_prompt_len
            '''
序列总长 = prompt_length + response_length (加padding补齐到 max_seq_len)

  ┌──────────────────────────────────────────────────────────────────┐
  │ [PAD] ... [PAD] │  tok_p1 tok_p2 ... tok_pN │ tok_r1 tok_r2 ... tok_rM │ [PAD] ... [PAD] │
  │   prompt 左padding部分  │   有效 prompt 部分  │    有效 response 部分 │ response右padding部分 │
  └──────────────────────────────────────────────────────────────────┘
  ↑                ↑                          ↑                      ↑
  |                |                          |                      |
左pad结束     prompt有效结束             response有效结束        右pad结束

MASK（针对loss计算的例子）：
  prompt mask(输入用)                     response mask(输出/loss用)
  ┌──────────────────────────────────────────────────────────────────┐
  │   0   ...   0   │   1   1   1...1 │     1   1   ...  1   │  0    ...   0   │
  └──────────────────────────────────────────────────────────────────┘
  0 = padding位，忽略
  1 = 有效位，计算loss时参与

            '''
            prompt_ids = data_item.batch["prompts"]

            # 这里的 prompt_length 是 max_prompt_len
            prompt_length = prompt_ids.shape[-1]

            # 因为输入会左 padding 到一样的长度，除了有效的 prompt，左边的位置都是 padding token
            # padding token 的 id 是 0
            # 所以对 mask 中前 max_prompt_len 的位置求和，就能知道真正的 prompt 的长度
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()

            # 取 prompt 张量末尾往前数 valid_prompt_length 个元素，就是 valid_prompt_ids
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            # response 矩阵单独存储 response
            response_ids = data_item.batch["responses"]

            # padding 后的 response 是在 mask 矩阵中 max_prompt_len 往后 
            # 所以这里的索引是 prompt_length 而不是 valid_prompt_length
            # 会被右 padding 到 max_response_len, sum 一下之后得到 response 的长度
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()

            # 从头往后数 valid_response_length 个元素，就是 valid_response_ids
            valid_response_ids = response_ids[:valid_response_length]

            # 记录 response 的长度
            reward_extra_info["response_token_length"].append(int(valid_response_length))
            
            # 把 prompt 和 response 都解码
            # 注意这里已经把 token 解码成 string 了
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token

            # 总体长度 - eos_token 的长度是 response 的长度
            # eos_token 有自己的长度，比如 qwen 是 <|end_of_text|>
            # 从开头到 倒数 len(eos_token) 的位置就是 response string
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            # 获取 ground truth
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            # 在我的设计中我加入了 equivalent_answers，来降低错误的奖励信号
            equivalent_answers = data_item.non_tensor_batch["reward_model"].get("equivalent_answers", [])

            # 获取 data source
            # 不同的 data source 计算 reward 的方式可能不同
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            # 把一些必要信息传给 compute_score
            # 这里我们就知道为什前面要转成 str 了，因为 ground truth 是用 str 存储的，这样才能比较
            # 进行奖励的计算，返回的是一个 float
            '''
            def compute_score(model_output: str, ground_truth: str, equivalent_answers: list = None, timeout_score: float = 0) -> float:
            '''
            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                equivalent_answers=equivalent_answers,
                extra_info=extra_info,
            )

            score: float
            # 可能有些时候返回的 result 是一个 dict
            # 但是在 rule based 的计算下是一个 float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            # 记录下这条 rollout 是否做对，存为 acc
            # 注意这个 score 并不是最终奖励，后面还可以对这个 score 做各种操作 
            else:
                score = result
                reward_extra_info["acc"].append(score)

            # 这里使用 compute_score 比对 ground truth 完成初步的奖励计算
            reward = score

            # 如果开了 overlong buffer
            # 根据 response 占有 buffer 的长度，给 reward 加一个长度的惩罚

            """
            Overlong reward 计算机制示意图：

            假设：
            max_resp_len      = 1000   # 允许生成的最大长度
            buffer_len        =  100   # 缓冲区长度
            penalty_factor    = 1.0    # 惩罚系数
            expected_len      = max_resp_len - buffer_len = 900

            # 可视化：
            序列长度刻度：
            0 ──────────────────────────────900───────────────1000─────────▶  （token数）

            │<──────────  允许生成的部分  ──────────│<── 缓冲区部分 ──>│
            │             无惩罚区                 │   惩罚区（超出）  │

            举例：
            valid_response_length = 950   # 实际生成长度（有效token数）
            
            exceed_len = valid_response_length - expected_len
                        = 950 - 900 = 50   # 超出 50 个 token（在缓冲区范围内）

            overlong_reward = min( -(exceed_len / buffer_len) × penalty_factor , 0 )
                            = min( -(50 / 100) × 1.0 , 0 )
                            = min( -0.5 , 0 )
                            = -0.5   # 惩罚 0.5 分
            
            """
            # 可以说是一个对 overlong 非常严重的惩罚了 
            if self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)


            """
            回顾一下这个 reward_tensor 是个啥
            # 新建一个 torch 张量，为每个 response 分配一个 reward
            # 形状是 (batch_size * n_roll, max_response_len)
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            
            这里把奖励给到 response 的最后一个 token, 其实就是 eos token
           
            response:     [token1] [token2] [token3] [EOS] [PAD] [PAD] ...
            reward_tensor:  0        0        0      1.0    0     0    ...
                                                    ↑
                                        valid_response_length - 1
            """
            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
