## DeepVision-103k Training & Evaluation Toolkit

This repository provides a complete pipeline for DeepVision-103k. We use [GSPO](https://arxiv.org/abs/2507.18071) for training and [vllm](https://github.com/vllm-project/vllm) for async batch evaluation. The training code is built on top of [verl](https://github.com/volcengine/verl). We use [swanlab](https://github.com/SwanHubX/SwanLab) for experiment tracking.

---

### Project Structure

```
.
├── verl/                    # Core verl library (install via pip install -e .)
├── train_scripts/           # Training launch scripts
│   ├── train_single_node_template.sh   # Single-node training
│   └── train_multi_node_template.sh    # Multi-node training (Ray cluster)
├── eval_scripts/            # Evaluation / inference scripts
│   ├── caller_async.py      # Orchestrator: launches vLLM servers + dispatches inference
│   ├── infer-v6.py          # Async continuous-batching inference client
│   └── caller.sh            # Example launch script
├── setup.py
├── pyproject.toml
└── requirements.txt
```

---
### Installation
#### Recommended Environment
We recommend the following environment configuration:
- CUDA 12.8
- PyTorch 2.8.0
- vLLM 0.11.0
- Transformers 4.57.1

#### Setup Steps
```bash
# Clone the repo
git clone <repo_url> && cd deepvision-103k

# Install mathverify for rule-based verification
pip install mathverify

# Install qwen_vl_utils for model training
pip install qwen_vl_utils

# Install verl in editable mode
pip install -e .
```
---

### Training

Two training templates are provided under `train_scripts/`. Both use the GSPO algorithm with GRPO advantage estimation.

#### Quick Start

1. **Search for `{YOUR_`** in the script to find all placeholders that need to be filled in:

| Placeholder | Description |
|---|---|
| `{YOUR_SWANLAB_API_KEY}` | Your SwanLab API key (for experiment tracking) |
| `{YOUR_PROJECT_NAME}` | Project name for experiment grouping |
| `{YOUR_BASE_MODEL}` | Base model identifier (used in experiment naming) |
| `{YOUR_ROOT_PATH}` | Root directory for saving checkpoints |
| `{YOUR_MODEL_PATH}` | Path to the pretrained model (e.g. HuggingFace format) |
| `{YOUR_TRAIN_FILE}` | Path to training data (`.parquet` format) |
| `{YOUR_TEST_FILE}` | Path to validation data (`.parquet` format) |

2. **Uncomment the GPU setting block** that matches your cluster size (8 / 16 / 32 / 64 GPUs).

3. **Run the script.**

#### Single-Node Training (8/16 GPUs on one machine)

```bash
bash train_scripts/train_single_node_template.sh
```

- Uses `python -m verl.trainer.my_main_dapo` directly
- Set `trainer.n_gpus_per_node` and `trainer.nnodes` accordingly
- Logs are saved to `$CKPTS_DIR/train.log`

#### Multi-Node Training (Ray cluster across multiple machines)

```bash
# Submit to each node via your job scheduler
# Environment variables RANK, WORLD_SIZE, MASTER_ADDR must be set by the scheduler
bash train_scripts/train_multi_node_template.sh
```

- **Rank 0** starts the Ray head node and submits the training job
- **Worker nodes** automatically join the Ray cluster via a shared IP file
- Additional features vs. single-node:
  - Environment auto-setup (pip installs, shared memory resize)
  - Separate validation inference parameters (`VAL_TEMPERATURE`, `VAL_TOP_P`)
  - Automatic MASTER_ADDR discovery across nodes

#### Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `LR` | `1e-6` | Learning rate |
| `KL_COEF` | `0.001` | KL divergence loss coefficient |
| `CLIP_RATIO_LOW/HIGH` | `0.0003/0.0004` | PPO clipping range |
| `N_ROLL` | 8-16 | Number of rollout samples per prompt |
| `TEMPERATURE` | `1.0` | Rollout sampling temperature |
| `MAX_PROMPT_LENGTH` | `2048` | Maximum prompt token length |
| `MAX_RESPONSE_LENGTH` | `16384` | Maximum response token length |
| `enable_filter_groups` | `true` | Filter rollout groups by accuracy |
| `loss_mode` | `gspo` | Loss function (gspo) |
| `LOSS_AGG_MODE` | `seq-mean-token-mean` | Loss aggregation mode |

#### GPU Setting Presets

| GPUs | GEN_BATCH | TRAIN_BATCH | MINI_BATCH | MICRO_BATCH | N_ROLL |
|------|-----------|-------------|------------|-------------|--------|
| 8    | 128       | 64          | 16         | 8           | 16      |
| 16   | 256       | 128         | 32         | 16          | 16     |
| 32   | 512       | 256         | 64         | 32          | 16     |
| 64   | 1024      | 512         | 128        | 64          | 16     |


### Experiment Tracking

Both training and evaluation support [SwanLab](https://swanlab.cn/) for experiment tracking. Set the following environment variables:

```bash
export SWANLAB_API_KEY=your_api_key
export SWANLAB_LOG_DIR=./swanlab_logs
export SWANLAB_MODE=cloud   # or "local" for offline
```


---

### Evaluation / Inference

The evaluation pipeline under `eval_scripts/` provides a fully automated workflow:
**launch vLLM servers -> run async inference -> save results -> shutdown servers**.

#### Architecture

```
caller.sh  ->  caller_async.py (orchestrator)  ->  infer-v6.py (async client)
                    |                                    |
                    |-- Launches N vLLM instances         |-- Continuous batching
                    |-- Waits for readiness               |-- Multi-endpoint load balancing
                    |-- Dispatches inference               |-- Checkpoint/resume support
                    +-- Graceful shutdown                  +-- Multi-process data preloading
```

#### Quick Start

1. **Fill in placeholders** in `caller.sh`:

```bash
python caller_async.py \
    --model /path/to/your/model \
    --input /path/to/input.jsonl \
    --output /path/to/output.jsonl \
    --hyperparam mimo \
    --prompt-field prompt \
    --gpu-devices "0,1,2,3,4,5,6,7" \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --concurrent-per-endpoint 16 \
    --max-tokens 16384 \
    --n 8
```

2. **Run:**

```bash
cd eval_scripts
bash caller.sh
```

#### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | (required) | Path to the model directory |
| `--input` | (required) | Input JSONL file |
| `--output` | (required) | Output JSONL file |
| `--hyperparam` | `qwen_ins` | Preset profile: `qwen_ins`, `qwen_think`, `mimo`, `greedy` |
| `--gpu-devices` | `0,1,2,3` | Comma-separated GPU IDs |
| `--tensor-parallel-size` | `4` | Tensor parallel size (TP) |
| `--data-parallel-size` | `1` | Data parallel size (DP), launches multiple vLLM instances |
| `--concurrent-per-endpoint` | `64` | Async concurrency per vLLM instance |
| `--n` | `1` | Number of return sequences per prompt |
| `--max-tokens` | `16384` | Max tokens to generate |
| `--prompt-field` | auto-detect | Field name(s) for prompt in input data |
| `--image-field` | auto-detect | Field name(s) for image paths |
| `--prompt-file` | (none) | Path to a prompt prefix/template file |
| `--max-retry-rounds` | `5` | Retry rounds for failed requests |

#### Parallelism Configuration

Total GPUs = `tensor-parallel-size` x `data-parallel-size`

| Setup | TP | DP | Instances | GPUs/Instance |
|-------|----|----|-----------|---------------|
| 8 GPU, max throughput | 1 | 8 | 8 | 1 |
| 8 GPU, large model | 4 | 2 | 2 | 4 |
| 8 GPU, very large model | 8 | 1 | 1 | 8 |

#### Concurrency Guidelines

| Output Length | Recommended Concurrency/Endpoint |
|---|---|
| Short (< 256 tokens) | 128-256 |
| Medium (256-1024) | 64-128 |
| Long (> 1024 tokens) | 32-64 |
| VLM with images | 32-64 |

#### Input Data Format

The input file should be in JSONL format. The prompt field is auto-detected in this order:

`messages` > `prompt` > `question` > `text` > `problem` > `query` > `input`

You can override this with `--prompt-field your_field_name`.

#### Hyperparameter Presets

| Preset | Temperature | Top-P | Top-K | Other |
|--------|-------------|-------|-------|-------|
| `qwen_ins` | 0.7 | 0.8 | 20 | repetition_penalty=1.0, presence_penalty=1.5 |
| `qwen_think` | 1.0 | 0.95 | 20 | repetition_penalty=1.0 |
| `mimo` | 0.3 | 0.95 | - | - |
| `greedy` | 0.01 | - | - | - |

---

### Post-Inference Evaluation

After inference is complete, use the evaluation tools under `eval_scripts/evaluation/` to score and analyze results.

#### Step 1: Math-Verify Rule-Based Evaluation

Run the math-verify judge to compute accuracy and automatically export error cases:

```bash
python eval_scripts/evaluation/mathverify_judge.py -i /path/to/your_output.jsonl
```

This will:

- Print overall accuracy statistics (correct / wrong / error counts)
- Save a detailed evaluation summary to `your_output_evaluation.json`
- Export all incorrect cases to `your_output_mathverify_error.jsonl`

**Optional flags:**

| Flag                      | Description                            |
| ------------------------- | -------------------------------------- |
| `-o /path/to/output.json` | Custom path for the evaluation summary |
| `--no-samples`            | Suppress sample prediction display     |
| `--no-export-errors`      | Skip exporting error cases             |

#### Step 2: GPT-5-mini Re-Judge on Error Cases

For the exported error cases (`*_mathverify_error.jsonl`), use GPT-5-mini as a secondary judge to catch false negatives from rule-based matching.

The judge prompt template is defined in `eval_scripts/evaluation/gpt5-mini-judge_prompt.md`. For each error case, construct the prompt by filling in the template:

```
PlainTextQuestion: {question}
Standard Answer:{gdt}
Model Answer:{box_answer}   # extracted \boxed{} content
```

Call GPT-5-mini with this prompt. The model will reply with exactly one word: **"Correct"** or **"Incorrect"**.

Cases marked **"Correct"** by GPT-5-mini are false negatives from math-verify and should be added back to the correct count for the final accuracy.

In our work, we roll 16 for each test query then cacululate the avg accuracy as pass@1 accuracy.
