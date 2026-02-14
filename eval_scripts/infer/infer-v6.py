#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Async Inference Client with Continuous Batching for vLLM

Features:
  - True continuous batching: maintains a constant number of in-flight requests
  - Multi-endpoint support with round-robin load balancing
  - Automatic checkpoint/resume via output JSONL deduplication
  - Multi-process data preloading with image base64 encoding
  - Configurable retry rounds for failed requests
"""
import json
import os
import base64
import argparse
import asyncio
import aiohttp
import time
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from multiprocessing import Process, Manager, cpu_count
from collections import deque
from typing import List, Tuple, Optional


# ============= Utility Functions =============

def get_unique_id(item):
    """Return a unique identifier for the item, preferring 'merge_id' over 'id'."""
    if 'merge_id' in item and item['merge_id'] is not None and item['merge_id'] != '':
        return str(item['merge_id'])
    if 'id' in item and item['id'] is not None and item['id'] != '':
        return str(item['id'])
    return None


def ensure_id_field(data):
    """Ensure every item has an 'id' field; auto-assign if missing."""
    id_field_added = False
    has_merge_id = False
    for item in data:
        if 'merge_id' in item and item['merge_id'] is not None and item['merge_id'] != '':
            has_merge_id = True
            break
    for idx, item in enumerate(data, start=1):
        if 'id' not in item or item['id'] is None or item['id'] == '':
            item['id'] = f"auto_id_{idx}"
            id_field_added = True
    return data, id_field_added, has_merge_id


def load_completed_ids(output_jsonl):
    """Load IDs of already-completed items from the output file (for resume)."""
    completed_ids = set()
    if os.path.exists(output_jsonl):
        with open(output_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    unique_id = get_unique_id(obj)
                    if unique_id:
                        completed_ids.add(unique_id)
                except:
                    continue
    return completed_ids


def encode_image_to_base64(image_path):
    """Read an image file and return its base64-encoded JPEG data URI."""
    with Image.open(image_path) as image:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"


def load_prompt_prefix(prompt_file):
    """Load a prompt prefix string from a text file."""
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return ""


def extract_prompt_from_messages(messages):
    """Extract prompt text from an OpenAI-style messages list."""
    system_prompt = None
    user_prompt = None
    for msg in messages:
        if msg['role'] == 'system':
            content = msg['content']
            if isinstance(content, str):
                system_prompt = content
            elif isinstance(content, list):
                text_parts = [part.get('text', '') for part in content if isinstance(part, dict) and part.get('type') == 'text']
                system_prompt = '\n'.join(text_parts)
            else:
                system_prompt = str(content)
        elif msg['role'] == 'user':
            content = msg['content']
            if isinstance(content, str):
                user_prompt = content
            elif isinstance(content, list):
                text_parts = [part.get('text', '') for part in content if isinstance(part, dict) and part.get('type') == 'text']
                user_prompt = '\n'.join(text_parts)
            else:
                user_prompt = str(content)
    if system_prompt and user_prompt:
        return f"{system_prompt}\n\n{user_prompt}"
    return user_prompt or system_prompt or ""


def get_image_paths_from_item(item):
    """Extract image file paths from the item, trying multiple candidate field names."""
    image_paths = []
    if 'images' in item:
        images = item['images']
        if isinstance(images, list):
            image_paths = [p for p in images if p and isinstance(p, str) and p.strip()]
        elif isinstance(images, str) and images.strip():
            image_paths = [images.strip()]
    if not image_paths:
        for key in ['image', 'img_path', 'image_path']:
            if key in item:
                value = item[key]
                if isinstance(value, list):
                    image_paths = [p for p in value if p and isinstance(p, str) and p.strip()]
                elif isinstance(value, str) and value.strip():
                    image_paths = [value.strip()]
                if image_paths:
                    break
    return image_paths


def get_nested_value(item, field_path):
    """
    Retrieve a nested field value using dot-separated path.
    Example: "data.content.text" -> item['data']['content']['text']
    """
    if not field_path:
        return None

    keys = field_path.split('.')
    value = item

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None

    return value if isinstance(value, str) else None


def get_prompt_from_item(item, prompt_fields=None):
    """
    Extract prompt text from a data item.

    Resolution order:
      1. 'messages' field (OpenAI chat format) — always takes priority
      2. User-specified fields via prompt_fields (supports nested paths like "data.prompt")
      3. Default fallback fields: prompt, question, text, problem, query, input, instruction, content

    Args:
        item: A single data record (dict).
        prompt_fields: Optional list of candidate field names to try.

    Returns:
        The extracted prompt string.
    """
    # OpenAI chat format always takes priority
    if 'messages' in item:
        return extract_prompt_from_messages(item['messages'])

    # Try user-specified fields
    if prompt_fields:
        for field in prompt_fields:
            value = get_nested_value(item, field)
            if value:
                return value

    # Fallback to common field names
    default_fields = ['prompt', 'question', 'text', 'problem', 'query', 'input', 'instruction', 'content']
    for field in default_fields:
        if field in item:
            value = item[field]
            if isinstance(value, str) and value.strip():
                return value

    return ''


def preprocess_item_with_serialization(item, prompt_prefix, image_cache, sampling_params, prompt_fields=None):
    """Preprocess a single item: encode images, build payload, serialize to bytes."""
    try:
        image_paths = get_image_paths_from_item(item)
        has_images = len(image_paths) > 0
        image_contents = []
        if has_images:
            for image_path in image_paths:
                if not os.path.exists(image_path):
                    return None, None, f'Image file not found: {image_path}'
                if image_path not in image_cache:
                    image_cache[image_path] = encode_image_to_base64(image_path)
                image_contents.append(image_cache[image_path])

        original_prompt = get_prompt_from_item(item, prompt_fields)

        if not original_prompt:
            return None, None, 'No valid prompt found'

        final_prompt = f"{prompt_prefix}\n{original_prompt}" if prompt_prefix else original_prompt

        if has_images and image_contents:
            content = []
            for img_content in image_contents:
                content.append({"type": "image_url", "image_url": {"url": img_content}})
            content.append({"type": "text", "text": final_prompt})
            messages = [{"role": "user", "content": content}]
        else:
            messages = [{"role": "user", "content": final_prompt}]

        payload = {"model": sampling_params['model'], "messages": messages}
        for k, v in sampling_params.items():
            if k != 'model' and v is not None:
                payload[k] = v

        payload_bytes = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        return payload_bytes, item, None
    except Exception as e:
        return None, None, f'Preprocessing failed: {str(e)}'


def preload_all_data_with_serialization(data, prompt_prefix, sampling_params, prompt_fields=None, num_workers=None):
    """Preload and serialize all data items in parallel using multiprocessing."""
    if num_workers is None:
        num_workers = min(max(cpu_count() - 2, 4), len(data))

    print(f"\n🔄 Preloading & serializing data with {num_workers} workers...")

    manager = Manager()
    shared_image_cache = manager.dict()
    result_list = manager.list([None] * len(data))

    def worker(start_idx, end_idx, data_slice, prompt_prefix, sampling_params, prompt_fields, shared_cache, result_list):
        local_cache = {}
        for i, item in enumerate(data_slice):
            actual_idx = start_idx + i
            payload_bytes, result_item, error = preprocess_item_with_serialization(
                item, prompt_prefix, local_cache, sampling_params, prompt_fields
            )
            result_list[actual_idx] = (payload_bytes, result_item, error)
        for k, v in local_cache.items():
            if k not in shared_cache:
                shared_cache[k] = v

    chunk_size = max(1, len(data) // num_workers)
    processes = []

    for i in range(num_workers):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(data)) if i < num_workers - 1 else len(data)
        if start_idx >= len(data):
            break
        p = Process(
            target=worker,
            args=(start_idx, end_idx, data[start_idx:end_idx], prompt_prefix, sampling_params, prompt_fields, shared_image_cache, result_list)
        )
        p.start()
        processes.append(p)

    with tqdm(total=len(data), desc="Preloading") as pbar:
        last_count = 0
        while any(p.is_alive() for p in processes):
            current_count = sum(1 for x in result_list if x is not None)
            pbar.update(current_count - last_count)
            last_count = current_count
            time.sleep(0.1)
        current_count = sum(1 for x in result_list if x is not None)
        pbar.update(current_count - last_count)

    for p in processes:
        p.join()

    processed_data = list(result_list)
    valid_count = sum(1 for pb, _, _ in processed_data if pb is not None)
    print(f"✅ Preloading complete! Valid tasks: {valid_count}\n")
    return processed_data


# ============= Async File Writer =============

class AsyncFileWriter:
    """Buffered async writer that flushes results to disk in batches."""

    def __init__(self, output_file):
        self.output_file = output_file
        self.queue = asyncio.Queue()
        self.total_written = 0
        self.running = True
        self._task = None
        self.file_handle = None

    async def start(self):
        self.file_handle = open(self.output_file, 'a', encoding='utf-8', buffering=262144)
        self._task = asyncio.create_task(self._writer_loop())

    async def _writer_loop(self):
        buffer = []
        last_flush = time.time()

        while self.running or not self.queue.empty():
            try:
                result = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                if result is not None:
                    buffer.append(result)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                break

            now = time.time()
            if len(buffer) >= 50 or (buffer and now - last_flush > 0.5):
                for item in buffer:
                    self.file_handle.write(json.dumps(item, ensure_ascii=False) + '\n')
                self.file_handle.flush()
                self.total_written += len(buffer)
                buffer.clear()
                last_flush = now

        if buffer:
            for item in buffer:
                self.file_handle.write(json.dumps(item, ensure_ascii=False) + '\n')
            self.file_handle.flush()
            self.total_written += len(buffer)

    async def write(self, result):
        if result is not None:
            await self.queue.put(result)

    async def close(self):
        self.running = False
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except:
                self._task.cancel()
        if self.file_handle:
            self.file_handle.close()


# ============= Continuous Batching Inference Engine =============

async def continuous_batching_infer(
    tasks: List[Tuple[bytes, dict]],
    api_urls: List[str],
    output_file: str,
    total_concurrent: int,
    request_timeout: int = 6000,
):
    """
    Continuous batching inference engine.

    Maintains exactly `total_concurrent` in-flight requests at all times.
    As soon as one request completes, a new one is immediately dispatched.
    """

    num_endpoints = len(api_urls)
    total_tasks = len(tasks)

    timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_read=request_timeout)
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0, ttl_dns_cache=300)
    session = aiohttp.ClientSession(timeout=timeout, connector=connector)

    writer = AsyncFileWriter(output_file)
    await writer.start()

    completed = 0
    success = 0
    errors = 0
    endpoint_counter = 0

    task_iter = iter(tasks)
    tasks_submitted = 0

    pbar = tqdm(total=total_tasks, desc="Inference", unit="req")
    rates = deque(maxlen=20)
    last_update_time = time.time()
    last_completed = 0
    start_time = time.time()

    async def send_one_request(payload_bytes: bytes, item: dict) -> Optional[dict]:
        """Send a single request with up to 3 retries."""
        nonlocal endpoint_counter

        for attempt in range(3):
            try:
                url = api_urls[endpoint_counter % num_endpoints]
                endpoint_counter += 1

                async with session.post(
                    url,
                    data=payload_bytes,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status != 200:
                        if attempt < 2:
                            await asyncio.sleep(0.2 * (attempt + 1))
                            continue
                        return None

                    result = await response.json()
                    choices = result.get('choices', [])

                    # Handle single vs. multiple return sequences (n > 1)
                    if len(choices) == 1:
                        return {
                            **item,
                            'model_response': choices[0]['message']['content'],
                            'usage': result.get('usage', {}),
                            'finish_reason': choices[0].get('finish_reason', '')
                        }
                    else:
                        return {
                            **item,
                            'model_responses': [
                                {
                                    'index': c.get('index', i),
                                    'content': c['message']['content'],
                                    'finish_reason': c.get('finish_reason', '')
                                }
                                for i, c in enumerate(choices)
                            ],
                            'usage': result.get('usage', {}),
                            'n': len(choices)
                        }
            except asyncio.TimeoutError:
                if attempt < 2:
                    await asyncio.sleep(0.5 * (attempt + 1))
            except asyncio.CancelledError:
                raise
            except Exception:
                if attempt < 2:
                    await asyncio.sleep(0.2 * (attempt + 1))
        return None

    def update_progress():
        """Update the progress bar with throughput and ETA."""
        nonlocal last_update_time, last_completed

        now = time.time()
        delta = completed - last_completed

        if delta > 0:
            pbar.update(delta)

            elapsed = now - last_update_time
            if elapsed > 0:
                rate = delta / elapsed
                rates.append(rate)

            avg_rate = sum(rates) / len(rates) if rates else 0
            remaining = total_tasks - completed
            eta = remaining / avg_rate if avg_rate > 0 else 0

            pbar.set_postfix({
                'ok': success,
                'fail': errors,
                'flying': len(pending),
                'rate': f'{avg_rate:.1f}/s',
                'eta': f'{eta/60:.1f}m' if eta > 60 else f'{eta:.0f}s'
            })

            last_completed = completed
            last_update_time = now

    # === Core loop: maintain constant concurrency ===
    pending = set()

    try:
        # Fill the concurrency pool
        for _ in range(min(total_concurrent, total_tasks)):
            try:
                payload_bytes, item = next(task_iter)
                tasks_submitted += 1
                task = asyncio.create_task(send_one_request(payload_bytes, item))
                task._item = item
                pending.add(task)
            except StopIteration:
                break

        # Main loop: as one completes, immediately submit another
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                completed += 1
                try:
                    result = task.result()
                    if result is not None:
                        success += 1
                        await writer.write(result)
                    else:
                        errors += 1
                except Exception:
                    errors += 1

                # Immediately replenish with a new task
                try:
                    payload_bytes, item = next(task_iter)
                    tasks_submitted += 1
                    new_task = asyncio.create_task(send_one_request(payload_bytes, item))
                    new_task._item = item
                    pending.add(new_task)
                except StopIteration:
                    pass

            update_progress()

    except asyncio.CancelledError:
        for task in pending:
            task.cancel()
        raise
    finally:
        pbar.close()
        await writer.close()
        await session.close()

        total_time = time.time() - start_time
        print(f"\n✅ Done: {success}/{total_tasks} | Failed: {errors} | Time: {total_time:.1f}s | Throughput: {completed/total_time:.1f} req/s")

    return {'completed': completed, 'success': success, 'error': errors}


# ============= Main Inference Orchestrator =============

def main_async_infer(
    input_jsonl,
    output_jsonl,
    api_url,
    concurrent_per_endpoint=64,
    prompt_file=None,
    prompt_fields=None,
    resume=True,
    request_timeout=300,
    preload_workers=None,
    max_retry_rounds=5,
    retry_delay=5,
    **sampling_kwargs
):
    # Parse API URLs (comma-separated string or list)
    if isinstance(api_url, str):
        api_urls = [url.strip() for url in api_url.split(',') if url.strip()]
    else:
        api_urls = [url.strip() for url in api_url if url.strip()]

    if not api_urls:
        raise ValueError("At least one API URL is required")

    num_endpoints = len(api_urls)
    total_concurrent = concurrent_per_endpoint * num_endpoints

    print(f"\n{'='*70}")
    print(f"🚀 Continuous Batching Inference Client")
    print(f"{'='*70}")
    print(f"API endpoints: {num_endpoints}")
    for i, url in enumerate(api_urls):
        print(f"  [{i}] {url}")
    print(f"Concurrency per endpoint: {concurrent_per_endpoint}")
    print(f"Total concurrency: {total_concurrent}")
    print(f"Request timeout: {request_timeout}s")

    if prompt_fields:
        print(f"Prompt fields: {prompt_fields}")
    else:
        print(f"Prompt fields: auto-detect (messages > prompt > question > text > ...)")

    print(f"{'='*70}\n")

    # Load data
    print("📂 Loading data...")
    all_data = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                all_data.append(json.loads(line))
            except:
                continue

    all_data, id_added, has_merge_id = ensure_id_field(all_data)
    print(f"✅ Loaded {len(all_data)} items\n")

    # Resume from checkpoint
    all_ids = {get_unique_id(item) for item in all_data if get_unique_id(item)}
    completed_ids = load_completed_ids(output_jsonl) if resume else set()
    pending_ids = all_ids - completed_ids

    print(f"Total: {len(all_data)} | Completed: {len(completed_ids)} | Pending: {len(pending_ids)}\n")

    if not pending_ids:
        print("🎉 All items already completed!")
        return

    # Sampling parameters
    prompt_prefix = load_prompt_prefix(prompt_file)
    sampling_params = {
        'model': sampling_kwargs.get('model', 'default-model'),
        'temperature': sampling_kwargs.get('temperature'),
        'top_p': sampling_kwargs.get('top_p'),
        'top_k': sampling_kwargs.get('top_k'),
        'max_tokens': sampling_kwargs.get('max_tokens', 16384),
        'seed': sampling_kwargs.get('seed'),
        'repetition_penalty': sampling_kwargs.get('repetition_penalty'),
        'presence_penalty': sampling_kwargs.get('presence_penalty'),
        'frequency_penalty': sampling_kwargs.get('frequency_penalty'),
        'n': sampling_kwargs.get('n'),
        'stop': sampling_kwargs.get('stop'),
    }

    print(f"📋 Sampling parameters:")
    for k, v in sampling_params.items():
        if v is not None:
            print(f"  {k}: {v}")
    print()

    # Filter pending items
    data_to_process = [item for item in all_data if get_unique_id(item) in pending_ids]

    # Preload and serialize (with prompt_fields support)
    processed_data = preload_all_data_with_serialization(
        data_to_process, prompt_prefix, sampling_params, prompt_fields, preload_workers
    )

    valid_tasks = [(pb, item) for pb, item, _ in processed_data if pb is not None]

    if not valid_tasks:
        print("❌ No valid tasks to process")
        return

    print(f"✅ Valid tasks: {len(valid_tasks)}\n")

    # Retry loop
    round_num = 0
    current_pending_ids = pending_ids.copy()
    total_start = time.time()

    while current_pending_ids and round_num < max_retry_rounds:
        round_num += 1
        print(f"\n{'='*70}")
        print(f"🔄 Round {round_num} | Pending: {len(current_pending_ids)}")
        print(f"{'='*70}\n")

        current_tasks = [(pb, item) for pb, item in valid_tasks if get_unique_id(item) in current_pending_ids]

        if not current_tasks:
            break

        stats = asyncio.run(continuous_batching_infer(
            current_tasks,
            api_urls,
            output_jsonl,
            total_concurrent,
            request_timeout,
        ))

        newly_completed = load_completed_ids(output_jsonl)
        current_pending_ids = current_pending_ids - newly_completed

        if not current_pending_ids:
            print(f"\n✅ All completed after round {round_num}!")
            break

        if round_num < max_retry_rounds:
            print(f"\n⏳ {len(current_pending_ids)} items remaining, retrying in {retry_delay}s...")
            time.sleep(retry_delay)

    # Final summary
    total_time = time.time() - total_start
    final_completed = len(load_completed_ids(output_jsonl))
    print(f"\n{'='*70}")
    print(f"✅ Finished! Total time: {total_time:.1f}s | Completed: {final_completed}/{len(all_ids)}")
    print(f"{'='*70}\n")


# ============= CLI Entry Point =============

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Continuous Batching vLLM Inference Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python %(prog)s \\
    --input_jsonl input.jsonl \\
    --output_jsonl output.jsonl \\
    --api_url "http://gpu0:8000/v1/chat/completions" \\
    --model your-model

  # Specify custom prompt field
  python %(prog)s \\
    --input_jsonl input.jsonl \\
    --output_jsonl output.jsonl \\
    --api_url "http://gpu0:8000/v1/chat/completions" \\
    --model your-model \\
    --prompt_field query

  # Multiple candidate prompt fields (tried in order)
  python %(prog)s \\
    --prompt_field query input_text data.prompt

  # Multi-endpoint with high concurrency
  python %(prog)s \\
    --api_url "http://gpu0:8000/v1/chat/completions,http://gpu1:8000/v1/chat/completions" \\
    --concurrent_per_endpoint 64 \\
    --model your-model

Prompt Field Resolution:
  - Default priority: messages > prompt > question > text > problem > query > input > instruction > content
  - Use --prompt_field to specify one or more candidate fields
  - Supports nested fields, e.g.: data.prompt, input.text, meta.query
  - The 'messages' field always takes priority (OpenAI chat format)

Concurrency Guidelines:
  - Short output  (< 256 tokens):   128-256 per endpoint
  - Medium output (256-1024):       64-128 per endpoint
  - Long output   (> 1024 tokens):  32-64 per endpoint
  - VLM with images:                32-64 per endpoint
        """
    )

    parser.add_argument("--input_jsonl", type=str, required=True,
                        help="Path to input JSONL file")
    parser.add_argument("--output_jsonl", type=str, required=True,
                        help="Path to output JSONL file")
    parser.add_argument("--api_url", type=str, required=True,
                        help="API URL(s), comma-separated for multiple endpoints")
    parser.add_argument("--concurrent_per_endpoint", type=int, default=64,
                        help="Concurrent requests per endpoint (default: 64)")
    parser.add_argument("--preload_workers", type=int, default=None,
                        help="Number of preloading workers (default: cpu_count - 2)")
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="Path to prompt prefix file")
    parser.add_argument("--prompt_field", type=str, nargs='*', default=None,
                        help="Prompt field name(s), tried in order; supports nested paths like data.prompt")
    parser.add_argument("--no_resume", action='store_true',
                        help="Do not resume from checkpoint; start fresh")
    parser.add_argument("--request_timeout", type=int, default=12000,
                        help="Per-request timeout in seconds (default: 12000)")
    parser.add_argument("--max_retry_rounds", type=int, default=5,
                        help="Maximum retry rounds for failed items (default: 5)")
    parser.add_argument("--retry_delay", type=int, default=5,
                        help="Delay between retry rounds in seconds (default: 5)")

    # Sampling parameters
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or path")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Top-P (nucleus) sampling")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Top-K sampling")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Maximum tokens to generate (default: 4096)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--repetition_penalty", type=float, default=None,
                        help="Repetition penalty")
    parser.add_argument("--presence_penalty", type=float, default=None,
                        help="Presence penalty")
    parser.add_argument("--frequency_penalty", type=float, default=None,
                        help="Frequency penalty")
    parser.add_argument("--n", type=int, default=None,
                        help="Number of return sequences per prompt")
    parser.add_argument("--stop", type=str, nargs='*', default=None,
                        help="Stop token list")

    args = parser.parse_args()

    main_async_infer(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        api_url=args.api_url,
        concurrent_per_endpoint=args.concurrent_per_endpoint,
        preload_workers=args.preload_workers,
        prompt_file=args.prompt_file,
        prompt_fields=args.prompt_field,
        resume=not args.no_resume,
        request_timeout=args.request_timeout,
        max_retry_rounds=args.max_retry_rounds,
        retry_delay=args.retry_delay,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        n=args.n,
        stop=args.stop,
    )
