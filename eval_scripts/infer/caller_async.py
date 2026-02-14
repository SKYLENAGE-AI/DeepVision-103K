#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Async Inference Orchestrator (TP + DP)

This script manages the full inference lifecycle:
  1. Launches one or more vLLM serving instances (Tensor Parallel + Data Parallel)
  2. Waits for all instances to become ready
  3. Dispatches inference via the async continuous-batching client (infer-v6.py)
  4. Gracefully shuts down all vLLM instances on completion or interruption
"""
import os
import sys
import json
import time
import subprocess
import argparse
import signal
import requests
import random
import threading
from typing import List, Optional

# ============= Default Configuration =============

# vLLM environment
DEFAULT_VLLM_ENV = "verl1108"
DEFAULT_PORT = 8000
DEFAULT_TENSOR_PARALLEL_SIZE = 4
DEFAULT_DATA_PARALLEL_SIZE = 1
DEFAULT_GPU_DEVICES = "0,1,2,3"

# vLLM serve defaults
DEFAULT_MAX_NUM_SEQS = 512
DEFAULT_MAX_MODEL_LEN = 20000
DEFAULT_GPU_MEMORY_UTILIZATION = 0.8

# Inference defaults
DEFAULT_PROMPT_FILE = ""
DEFAULT_MAX_TOKENS = 16384
DEFAULT_MAX_RETRY_ROUNDS = 5
DEFAULT_N = 1

# Concurrency
DEFAULT_CONCURRENT_PER_ENDPOINT = 64

# Async inference client script (co-located with this file)
INFERENCE_CLIENT_ASYNC_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "infer-v6.py")

# Preset hyperparameter profiles
HYPERPARAMS = {
    "qwen_ins": {
        "temperature": 0.7,
        "top_k": 20,
        "top_p": 0.8,
        "repetition_penalty": 1.0,
        "presence_penalty": 1.5,
    },
    "qwen_think": {
        "temperature": 1.0,
        "top_k": 20,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        "presence_penalty": 0,
    },
    "mimo": {
        "temperature": 0.3,
        "top_p": 0.95,
    },
    "greedy": {
        "temperature": 0.01
    },
}


class InferenceManager:
    """Manages vLLM server lifecycle and inference dispatch."""

    def __init__(self,
                 # Server configuration
                 port=DEFAULT_PORT,
                 tensor_parallel_size=DEFAULT_TENSOR_PARALLEL_SIZE,
                 data_parallel_size=DEFAULT_DATA_PARALLEL_SIZE,
                 gpu_devices=DEFAULT_GPU_DEVICES,
                 vllm_env=DEFAULT_VLLM_ENV,
                 max_num_seqs=DEFAULT_MAX_NUM_SEQS,
                 max_model_len=DEFAULT_MAX_MODEL_LEN,
                 gpu_memory_utilization=DEFAULT_GPU_MEMORY_UTILIZATION,
                 # Inference configuration
                 concurrent_per_endpoint=DEFAULT_CONCURRENT_PER_ENDPOINT,
                 prompt_file=DEFAULT_PROMPT_FILE,
                 prompt_fields=None,
                 image_fields=None,
                 max_tokens=DEFAULT_MAX_TOKENS,
                 max_retry_rounds=DEFAULT_MAX_RETRY_ROUNDS,
                 n=DEFAULT_N,
                 lb_strategy='round_robin'):
        # Server config
        self.base_port = port
        self.tensor_parallel_size = tensor_parallel_size
        self.data_parallel_size = data_parallel_size
        self.gpu_devices = gpu_devices
        self.vllm_env = vllm_env
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization

        # Inference config
        self.concurrent_per_endpoint = concurrent_per_endpoint
        self.prompt_file = prompt_file
        self.prompt_fields = prompt_fields or []
        self.image_fields = image_fields or []
        self.max_tokens = max_tokens
        self.max_retry_rounds = max_retry_rounds
        self.n = n
        self.lb_strategy = lb_strategy

        # Process management
        self.vllm_processes: List[subprocess.Popen] = []
        self.instance_ports: List[int] = []
        self.instance_gpus: List[str] = []

        # Which instance streams output to console
        self._display_instance_id = 0

        self._parse_gpu_allocation()

    def _parse_gpu_allocation(self):
        """Parse GPU device string and allocate GPUs across DP instances."""
        all_gpus = [g.strip() for g in self.gpu_devices.split(',')]
        total_gpus = len(all_gpus)
        gpus_per_instance = self.tensor_parallel_size

        required_gpus = gpus_per_instance * self.data_parallel_size
        if required_gpus > total_gpus:
            raise ValueError(
                f"Insufficient GPUs: need {required_gpus} (TP={self.tensor_parallel_size} × DP={self.data_parallel_size}), "
                f"but only {total_gpus} available ({self.gpu_devices})"
            )

        self.instance_gpus = []
        self.instance_ports = []

        for i in range(self.data_parallel_size):
            start_idx = i * gpus_per_instance
            end_idx = start_idx + gpus_per_instance
            instance_gpu_list = all_gpus[start_idx:end_idx]
            self.instance_gpus.append(','.join(instance_gpu_list))
            self.instance_ports.append(self.base_port + i)

        print(f"\n{'='*60}")
        print(f"🔧 GPU Allocation")
        print(f"{'='*60}")
        print(f"Total GPUs: {total_gpus}")
        print(f"Tensor Parallel (TP): {self.tensor_parallel_size}")
        print(f"Data Parallel (DP): {self.data_parallel_size}")
        print(f"GPUs per instance: {gpus_per_instance}")
        print(f"\nInstance mapping:")
        for i, (gpus, port) in enumerate(zip(self.instance_gpus, self.instance_ports)):
            print(f"  Instance {i}: GPU [{gpus}] -> Port {port}")
        print(f"{'='*60}\n")

    def start_vllm_server(self, model_path: str) -> bool:
        """Launch all vLLM serving instances and wait until ready."""
        self._display_instance_id = random.randint(0, self.data_parallel_size - 1)

        print(f"\n{'='*60}")
        print(f"🚀 Starting vLLM Server (TP={self.tensor_parallel_size}, DP={self.data_parallel_size})")
        print(f"{'='*60}")
        print(f"Model path: {model_path}")
        print(f"max_num_seqs: {self.max_num_seqs}")
        print(f"max_model_len: {self.max_model_len}")
        print(f"gpu_memory_utilization: {self.gpu_memory_utilization}")
        print(f"📺 Console output from instance: {self._display_instance_id} (port {self.instance_ports[self._display_instance_id]})")
        print(f"{'='*60}\n")

        for i in range(self.data_parallel_size):
            gpu_devices = self.instance_gpus[i]
            port = self.instance_ports[i]

            print(f"\nLaunching instance {i}: GPU [{gpu_devices}], port {port}")

            if not self._start_single_instance(model_path, gpu_devices, port, i):
                print(f"\n✗ Instance {i} failed to start")
                self.stop_vllm_server()
                return False

        print(f"\nWaiting for all {self.data_parallel_size} instances to become ready...")
        if self._wait_for_all_servers():
            print(f"\n✓ All {self.data_parallel_size} vLLM instances are ready\n")
            return True
        else:
            print("\n✗ Some instances failed to start")
            self.stop_vllm_server()
            return False

    def _start_single_instance(self, model_path: str, gpu_devices: str, port: int, instance_id: int) -> bool:
        """Start a single vLLM serve process."""
        env = os.environ.copy()
        env['PATH'] = f"/opt/conda/envs/{self.vllm_env}/bin:{env.get('PATH', '')}"
        env['CUDA_VISIBLE_DEVICES'] = gpu_devices

        cmd = [
            'vllm', 'serve', model_path,
            '--host', '0.0.0.0',
            '--port', str(port),
            '--tensor-parallel-size', str(self.tensor_parallel_size),
            '--limit-mm-per-prompt.image', '10',
            '--limit-mm-per-prompt.video', '0',
            '--gpu-memory-utilization', str(self.gpu_memory_utilization),
            '--enable-chunked-prefill',
            '--max-num-batched-tokens', '16384',
            '--max-num-seqs', str(self.max_num_seqs),
            '--max-model-len', str(self.max_model_len),
            '--trust-remote-code'
        ]

        log_file = f"/tmp/vllm_instance_{instance_id}.log"

        try:
            show_output = (instance_id == self._display_instance_id)

            if show_output:
                # Stream this instance's output to console + log file
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    universal_newlines=True
                )

                def stream_output(proc, log_path):
                    with open(log_path, 'w') as log_f:
                        try:
                            for line in iter(proc.stdout.readline, ''):
                                if line:
                                    print(line, end='', flush=True)
                                    log_f.write(line)
                                    log_f.flush()
                        except:
                            pass

                thread = threading.Thread(
                    target=stream_output,
                    args=(process, log_file),
                    daemon=True
                )
                thread.start()
            else:
                # Other instances write to log file only
                log_f = open(log_file, 'w')
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=log_f,
                    stderr=subprocess.STDOUT
                )

            self.vllm_processes.append(process)
            display_mark = " 📺" if show_output else ""
            print(f"  ✓ Instance {instance_id} started (PID: {process.pid}, log: {log_file}){display_mark}")
            return True
        except Exception as e:
            print(f"  ✗ Failed to start instance {instance_id}: {e}")
            return False

    def _wait_for_all_servers(self, timeout: int = 6000, check_interval: int = 5) -> bool:
        """Poll all instances until they respond to health checks or timeout."""
        start_time = time.time()
        ready_instances = set()

        while time.time() - start_time < timeout:
            for i, port in enumerate(self.instance_ports):
                if i in ready_instances:
                    continue

                api_url = f"http://localhost:{port}/v1/models"
                try:
                    response = requests.get(api_url, timeout=5)
                    if response.status_code == 200:
                        ready_instances.add(i)
                        print(f"\n  ✓ Instance {i} (port {port}) ready")
                except requests.exceptions.RequestException:
                    pass

            if len(ready_instances) == self.data_parallel_size:
                elapsed = int(time.time() - start_time)
                print(f"\n✓ All instances ready ({elapsed}s elapsed)")
                return True

            # Check for crashed processes
            for i, process in enumerate(self.vllm_processes):
                if process.poll() is not None:
                    print(f"\n✗ Instance {i} exited (return code: {process.returncode})")
                    return False

            elapsed = int(time.time() - start_time)
            print(f"\rWaiting... {len(ready_instances)}/{self.data_parallel_size} ready, {elapsed}/{timeout}s",
                  end="", flush=True)
            time.sleep(check_interval)

        print("\n✗ Timed out waiting for servers")
        return False

    def stop_vllm_server(self):
        """Gracefully stop all vLLM instances (SIGTERM then SIGKILL)."""
        if not self.vllm_processes:
            return

        print(f"\nStopping {len(self.vllm_processes)} vLLM instance(s)...")

        for i, process in enumerate(self.vllm_processes):
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                        print(f"  ✓ Instance {i} stopped")
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                        print(f"  ✓ Instance {i} force-killed")
            except Exception as e:
                print(f"  ✗ Error stopping instance {i}: {e}")

        self.vllm_processes.clear()
        print("✓ All vLLM instances stopped")

    def get_api_urls(self) -> str:
        """Build comma-separated API URL string for all instances."""
        urls = [f"http://localhost:{port}/v1/chat/completions" for port in self.instance_ports]
        return ','.join(urls)

    def run_inference(self, model_path: str, input_file: str, output_file: str,
                     hyperparam_type: str = "qwen_ins") -> bool:
        """Dispatch inference to the async client script."""
        hyperparams = HYPERPARAMS.get(hyperparam_type, HYPERPARAMS["qwen_ins"])

        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        api_urls = self.get_api_urls()
        total_concurrent = self.concurrent_per_endpoint * self.data_parallel_size

        print(f"\n{'='*60}")
        print(f"🔥 Starting Inference")
        print(f"{'='*60}")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        print(f"Hyperparam profile: {hyperparam_type}")
        print(f"Hyperparams: {hyperparams}")
        print(f"\n⚡ Concurrency:")
        print(f"  Client: Async (Continuous Batching)")
        print(f"  Endpoints (DP): {self.data_parallel_size}")
        print(f"  Per-endpoint concurrency: {self.concurrent_per_endpoint}")
        print(f"  Total concurrency: {self.concurrent_per_endpoint} × {self.data_parallel_size} = {total_concurrent}")
        print(f"\n🎯 Generation:")
        print(f"  Return sequences (n): {self.n}")
        print(f"  max_tokens: {self.max_tokens}")
        print(f"\n📡 API Endpoints:")
        for i, port in enumerate(self.instance_ports):
            print(f"  [{i}] http://localhost:{port}/v1/chat/completions")
        print(f"\n📝 Prompt Config:")
        print(f"  Prompt file: {self.prompt_file if self.prompt_file else '(none)'}")
        if self.prompt_fields:
            print(f"  Prompt fields: {self.prompt_fields}")
        else:
            print(f"  Prompt fields: auto-detect (messages > prompt > question > text > ...)")
        if self.image_fields:
            print(f"  Image fields: {self.image_fields}")
        print(f"{'='*60}\n")

        env = os.environ.copy()
        env['PATH'] = f"/opt/conda/envs/{self.vllm_env}/bin:{env.get('PATH', '')}"

        cmd = [
            'python', INFERENCE_CLIENT_ASYNC_SCRIPT,
            '--input_jsonl', input_file,
            '--output_jsonl', output_file,
            '--api_url', api_urls,
            '--model', model_path,
            '--max_tokens', str(self.max_tokens),
            '--max_retry_rounds', str(self.max_retry_rounds),
            '--request_timeout', '600',
            '--n', str(self.n),
            '--concurrent_per_endpoint', str(self.concurrent_per_endpoint)
        ]

        if self.prompt_file and os.path.exists(self.prompt_file):
            cmd.extend(['--prompt_file', self.prompt_file])

        if self.prompt_fields:
            cmd.append('--prompt_field')
            cmd.extend(self.prompt_fields)

        if self.image_fields:
            cmd.append('--image_field')
            cmd.extend(self.image_fields)

        for key, value in hyperparams.items():
            cmd.extend([f'--{key}', str(value)])

        try:
            print("Command:")
            print(" ".join(cmd))
            print()

            result = subprocess.run(cmd, env=env, check=True)
            print("\n✓ Inference complete")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Inference failed: {e}")
            return False
        except Exception as e:
            print(f"\n✗ Inference error: {e}")
            return False


def signal_handler(sig, frame):
    print("\n\nInterrupt received, cleaning up...")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description='Async Inference Orchestrator (TP + DP)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (TP=4, DP=1, n=1)
  python caller_async.py --model /path/to/model --input data.jsonl --output result.jsonl

  # Specify prompt field
  python caller_async.py --model /path/to/model --input data.jsonl --output result.jsonl \\
      --prompt-field query

  # Multiple candidate prompt fields (tried in order)
  python caller_async.py --model /path/to/model --input data.jsonl --output result.jsonl \\
      --prompt-field query input_text data.prompt

  # Specify image field
  python caller_async.py --model /path/to/model --input data.jsonl --output result.jsonl \\
      --image-field img_url photo_path

  # 8 GPUs: TP=4, DP=2, 64 concurrent per endpoint
  python caller_async.py --model /path/to/model --input data.jsonl --output result.jsonl \\
      --gpu-devices "0,1,2,3,4,5,6,7" --tensor-parallel-size 4 --data-parallel-size 2 \\
      --concurrent-per-endpoint 64

  # 8 GPUs: TP=2, DP=4, 128 concurrent per endpoint (total 512)
  python caller_async.py --model /path/to/model --input data.jsonl --output result.jsonl \\
      --gpu-devices "0,1,2,3,4,5,6,7" --tensor-parallel-size 2 --data-parallel-size 4 \\
      --concurrent-per-endpoint 128

Parallelism:
  Total GPUs = tensor-parallel-size × data-parallel-size

  8 GPU examples:
    TP=4, DP=2: 2 instances, 4 GPUs each
    TP=2, DP=4: 4 instances, 2 GPUs each
    TP=8, DP=1: 1 instance,  8 GPUs
    TP=1, DP=8: 8 instances, 1 GPU each

Field Mapping:
  --prompt-field: Prompt field name(s) in the input data
    - Multiple candidates tried in order
    - Supports nested paths, e.g. data.prompt, input.text
    - Default priority: messages > prompt > question > text > problem > query > input

  --image-field: Image path field name(s) in the input data
    - Multiple candidates tried in order
    - Default priority: images > image > img_path > image_path

Concurrency Guidelines:
  - Short output  (< 256 tokens):   128-256 per endpoint
  - Medium output (256-1024):       64-128 per endpoint
  - Long output   (> 1024 tokens):  32-64 per endpoint
  - VLM with images:                32-64 per endpoint
        """
    )

    # ============ Required Arguments ============
    required = parser.add_argument_group('Required')
    required.add_argument('--model', type=str, required=True,
                          help='Model path')
    required.add_argument('--input', type=str, required=True,
                          help='Input JSONL file path')
    required.add_argument('--output', type=str, required=True,
                          help='Output JSONL file path')

    # ============ Hyperparameter Profile ============
    hyperparam = parser.add_argument_group('Hyperparameter Profile')
    hyperparam.add_argument('--hyperparam', type=str, default='qwen_ins',
                            choices=['qwen_ins', 'qwen_think', 'mimo', 'greedy'],
                            help='Hyperparameter preset (default: qwen_ins)')

    # ============ Parallelism ============
    parallel = parser.add_argument_group('Parallelism')
    parallel.add_argument('--gpu-devices', type=str, default=DEFAULT_GPU_DEVICES,
                          help=f'GPU device list (default: {DEFAULT_GPU_DEVICES})')
    parallel.add_argument('--tensor-parallel-size', type=int, default=DEFAULT_TENSOR_PARALLEL_SIZE,
                          help=f'Tensor parallel size (default: {DEFAULT_TENSOR_PARALLEL_SIZE})')
    parallel.add_argument('--data-parallel-size', type=int, default=DEFAULT_DATA_PARALLEL_SIZE,
                          help=f'Data parallel size (default: {DEFAULT_DATA_PARALLEL_SIZE})')

    # ============ Server Configuration ============
    serve = parser.add_argument_group('Server (vllm serve)')
    serve.add_argument('--port', type=int, default=DEFAULT_PORT,
                       help=f'Base port for vLLM instances (default: {DEFAULT_PORT})')
    serve.add_argument('--vllm-env', type=str, default=DEFAULT_VLLM_ENV,
                       help=f'Conda environment for vLLM (default: {DEFAULT_VLLM_ENV})')
    serve.add_argument('--max-num-seqs', type=int, default=DEFAULT_MAX_NUM_SEQS,
                       help=f'Max concurrent sequences per instance (default: {DEFAULT_MAX_NUM_SEQS})')
    serve.add_argument('--max-model-len', type=int, default=DEFAULT_MAX_MODEL_LEN,
                       help=f'Max model context length (default: {DEFAULT_MAX_MODEL_LEN})')
    serve.add_argument('--gpu-memory-utilization', type=float, default=DEFAULT_GPU_MEMORY_UTILIZATION,
                       help=f'GPU memory utilization ratio (default: {DEFAULT_GPU_MEMORY_UTILIZATION})')

    # ============ Inference Configuration ============
    infer = parser.add_argument_group('Inference')
    infer.add_argument('--concurrent-per-endpoint', type=int, default=DEFAULT_CONCURRENT_PER_ENDPOINT,
                       help=f'Concurrent requests per endpoint (default: {DEFAULT_CONCURRENT_PER_ENDPOINT})')
    infer.add_argument('--n', type=int, default=DEFAULT_N,
                       help=f'Number of return sequences (default: {DEFAULT_N})')
    infer.add_argument('--prompt-file', type=str, default=DEFAULT_PROMPT_FILE,
                       help=f'Prompt template file path')
    infer.add_argument('--max-tokens', type=int, default=DEFAULT_MAX_TOKENS,
                       help=f'Max tokens to generate (default: {DEFAULT_MAX_TOKENS})')
    infer.add_argument('--max-retry-rounds', type=int, default=DEFAULT_MAX_RETRY_ROUNDS,
                       help=f'Max retry rounds (default: {DEFAULT_MAX_RETRY_ROUNDS})')
    infer.add_argument('--lb-strategy', type=str, default='partition',
                       choices=['partition', 'round_robin', 'random', 'least_pending'],
                       help='Load balancing strategy (default: partition)')

    # ============ Field Mapping ============
    field = parser.add_argument_group('Field Mapping')
    field.add_argument('--prompt-field', type=str, nargs='*', default=None,
                       help='Prompt field name(s), tried in order; supports nested paths like data.prompt')
    field.add_argument('--image-field', type=str, nargs='*', default=None,
                       help='Image field name(s), tried in order')

    args = parser.parse_args()

    # ============ Validation ============
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"Error: Model path not found: {args.model}")
        sys.exit(1)

    if not os.path.exists(INFERENCE_CLIENT_ASYNC_SCRIPT):
        print(f"Error: Async inference client not found: {INFERENCE_CLIENT_ASYNC_SCRIPT}")
        sys.exit(1)

    all_gpus = [g.strip() for g in args.gpu_devices.split(',')]
    required_gpus = args.tensor_parallel_size * args.data_parallel_size
    if required_gpus > len(all_gpus):
        print(f"Error: Insufficient GPUs")
        print(f"  Required: TP={args.tensor_parallel_size} × DP={args.data_parallel_size} = {required_gpus}")
        print(f"  Available: {len(all_gpus)} ({args.gpu_devices})")
        sys.exit(1)

    if args.n < 1:
        print(f"Error: n must be >= 1, got: {args.n}")
        sys.exit(1)

    if args.concurrent_per_endpoint < 1:
        print(f"Error: concurrent-per-endpoint must be >= 1, got: {args.concurrent_per_endpoint}")
        sys.exit(1)

    if args.prompt_file and not os.path.exists(args.prompt_file):
        print(f"Warning: Prompt file not found: {args.prompt_file}")

    # ============ Signal Handling ============
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # ============ Create Manager ============
    try:
        manager = InferenceManager(
            port=args.port,
            tensor_parallel_size=args.tensor_parallel_size,
            data_parallel_size=args.data_parallel_size,
            gpu_devices=args.gpu_devices,
            vllm_env=args.vllm_env,
            max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            concurrent_per_endpoint=args.concurrent_per_endpoint,
            prompt_file=args.prompt_file,
            prompt_fields=args.prompt_field,
            image_fields=args.image_field,
            max_tokens=args.max_tokens,
            max_retry_rounds=args.max_retry_rounds,
            n=args.n,
            lb_strategy=args.lb_strategy
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # ============ Run Inference ============
    try:
        if not manager.start_vllm_server(args.model):
            print("Server startup failed")
            sys.exit(1)

        success = manager.run_inference(
            args.model, args.input, args.output, args.hyperparam
        )

        if success:
            print(f"\n✓ Inference complete, results saved to: {args.output}")
        else:
            print("\n✗ Inference failed")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nUser interrupted")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        manager.stop_vllm_server()


if __name__ == "__main__":
    main()
