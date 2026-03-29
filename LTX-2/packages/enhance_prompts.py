#!/usr/bin/env python3
"""
Prompt Enhancement Script for Distillation Dataset

使用 OpenRouter API 和 Gemini 模型批量扩充 prompts，添加视觉和音频描述。

使用方式:
    python enhance_prompts.py --input prompts.txt --output enhanced_prompts.txt

或者直接修改下方 CONFIG 参数后运行:
    python enhance_prompts.py
"""

import argparse
import json
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests
from tqdm import tqdm

# =============================================================================
# 配置参数 (可直接在此修改)
# =============================================================================

CONFIG = {
    # vLLM 本地 API 配置 (OpenAI 兼容格式)
    "api_key": "EMPTY",                 # 本地 vLLM 不需要 key，填任意值
    "api_base": "http://localhost:8000/v1/chat/completions",  # vLLM 默认端口
    "model": "gemma-3-12b",              # vLLM 启动时的 --served-model-name

    # 生成参数
    "temperature": 0.7,
    "max_tokens": 4096,                  # Gemma 3 本地推理，适当减小
    "top_p": 0.9,

    # 输入输出路径 (命令行参数会覆盖这些)
    "input_file": "./captions.txt",
    "output_file": "./captions_enhanced.txt",

    # System prompt file path (optional, uses built-in prompt if not provided)
    "system_prompt_file": "",

    # 并发和重试配置 (本地 vLLM 可以更激进)
    "max_workers": 32,          # 并发线程数，根据 GPU 显存调整
    "retry_times": 3,           # 失败重试次数
    "retry_delay": 1.0,         # 重试间隔(秒)
    "request_delay": 0.05,      # 请求间隔(秒)，本地可以很小

    # 断点续传
    "resume": True,             # 是否启用断点续传
    "checkpoint_interval": 10,  # 每处理多少条保存一次 checkpoint

    # 其他
    "skip_empty_lines": True,   # 跳过空行
    "verbose": True,            # 显示详细日志
    "skip_on_error": True,      # 出错时跳过该 prompt（不写入输出）
}

# =============================================================================
# 核心功能
# =============================================================================

def load_system_prompt(filepath: Path) -> str:
    """加载 system prompt"""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()


def enhance_single_prompt(
    prompt: str,
    system_prompt: str,
    config: dict,
) -> str | None:
    """
    使用 vLLM 本地 API 增强单个 prompt (OpenAI 兼容格式)

    Returns:
        增强后的 prompt，失败时返回 None（跳过该 prompt）
    """
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"user prompt: {prompt}"},
        ],
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
        "top_p": config["top_p"],
    }

    for attempt in range(config["retry_times"]):
        try:
            response = requests.post(
                config["api_base"],
                headers=headers,
                json=payload,
                timeout=120,
            )

            result = response.json()

            # 检查是否有错误信息
            if "error" in result:
                error_msg = result.get("error", {})
                if isinstance(error_msg, dict):
                    error_msg = error_msg.get("message", str(error_msg))
                raise RuntimeError(f"API error: {error_msg}")

            # 检查 rate limit
            if response.status_code == 429:
                wait_time = config["retry_delay"] * (attempt + 2)
                tqdm.write(f"[WARN] Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue

            response.raise_for_status()

            # 解析响应
            if "choices" not in result or len(result["choices"]) == 0:
                raise KeyError(f"No 'choices' in response: {str(result)[:200]}")

            enhanced = result["choices"][0]["message"]["content"].strip()

            # 清理可能的格式问题（移除开头的特殊字符）
            if enhanced.startswith('"') and enhanced.endswith('"'):
                enhanced = enhanced[1:-1]

            return enhanced

        except (requests.exceptions.RequestException, RuntimeError, KeyError, IndexError, json.JSONDecodeError) as e:
            if attempt < config["retry_times"] - 1:
                wait_time = config["retry_delay"] * (attempt + 1)
                tqdm.write(f"[WARN] Attempt {attempt + 1} failed: {str(e)[:100]}, retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                tqdm.write(f"[ERROR] Failed after {config['retry_times']} attempts: {str(e)[:150]}")
                return None  # 返回 None，跳过该 prompt

    return None


def load_checkpoint(output_file: Path) -> set[int]:
    """加载已处理的行号（用于断点续传）"""
    checkpoint_file = output_file.with_suffix(".checkpoint")
    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            return set(map(int, f.read().strip().split("\n"))) if f.read().strip() else set()
    return set()


def save_checkpoint(output_file: Path, processed_indices: set[int]):
    """保存 checkpoint"""
    checkpoint_file = output_file.with_suffix(".checkpoint")
    with open(checkpoint_file, "w") as f:
        f.write("\n".join(map(str, sorted(processed_indices))))


def process_prompts(
    input_file: Path,
    output_file: Path,
    config: dict,
):
    """
    批量处理 prompts
    """
    # 加载 system prompt
    system_prompt = load_system_prompt(config["system_prompt_file"])
    print(f"Loaded system prompt from: {config['system_prompt_file']}")

    # 读取输入文件
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    prompts = []
    for i, line in enumerate(lines):
        line = line.strip()
        if config["skip_empty_lines"] and not line:
            continue
        prompts.append((i, line))

    print(f"Loaded {len(prompts)} prompts from: {input_file}")

    # 断点续传：加载已处理的索引
    processed_indices = set()
    results = {}

    if config["resume"] and output_file.exists():
        # 读取已有的输出
        with open(output_file, "r", encoding="utf-8") as f:
            existing_lines = f.readlines()

        # 尝试加载 checkpoint
        checkpoint_file = output_file.with_suffix(".checkpoint")
        if checkpoint_file.exists():
            with open(checkpoint_file, "r") as f:
                content = f.read().strip()
                if content:
                    processed_indices = set(map(int, content.split("\n")))

        # 如果没有 checkpoint 但有输出文件，假设是按顺序处理的
        if not processed_indices and existing_lines:
            processed_indices = set(range(len(existing_lines)))

        # 加载已有结果
        for idx, line in enumerate(existing_lines):
            if idx < len(prompts):
                results[prompts[idx][0]] = line.strip()

        print(f"Resuming: {len(processed_indices)} prompts already processed")

    # 过滤出需要处理的 prompts
    remaining = [(i, p) for i, p in prompts if i not in processed_indices]
    print(f"Remaining: {len(remaining)} prompts to process")

    if not remaining:
        print("All prompts already processed!")
        return

    # 用于线程安全的写入
    write_lock = Lock()
    progress_bar = tqdm(total=len(remaining), desc="Enhancing prompts")

    def process_one(item):
        idx, prompt = item
        time.sleep(config["request_delay"])  # Rate limiting
        enhanced = enhance_single_prompt(prompt, system_prompt, config)
        return idx, enhanced

    # 并发处理
    skipped_count = 0
    with ThreadPoolExecutor(max_workers=config["max_workers"]) as executor:
        futures = {executor.submit(process_one, item): item for item in remaining}

        checkpoint_counter = 0
        for future in as_completed(futures):
            idx, enhanced = future.result()

            with write_lock:
                if enhanced is not None:
                    results[idx] = enhanced
                else:
                    skipped_count += 1
                processed_indices.add(idx)
                checkpoint_counter += 1

                # 定期保存 checkpoint
                if checkpoint_counter % config["checkpoint_interval"] == 0:
                    save_checkpoint(output_file, processed_indices)
                    # 也保存当前结果
                    _save_results(output_file, results, prompts)

            progress_bar.update(1)

            if config["verbose"] and checkpoint_counter % 50 == 0:
                tqdm.write(f"[Progress] {checkpoint_counter}/{len(remaining)} processed, {skipped_count} skipped")

    progress_bar.close()

    # 最终保存
    _save_results(output_file, results, prompts)

    # 删除 checkpoint 文件
    checkpoint_file = output_file.with_suffix(".checkpoint")
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    print(f"\nDone! Enhanced prompts saved to: {output_file}")
    print(f"Total: {len(results)} prompts, {skipped_count} skipped due to errors")


def _save_results(output_file: Path, results: dict, prompts: list):
    """按原始顺序保存成功的结果，跳过失败的 prompt"""
    with open(output_file, "w", encoding="utf-8") as f:
        for original_idx, _ in prompts:
            if original_idx in results and results[original_idx] is not None:
                # 确保单行输出（移除换行符）
                line = results[original_idx].replace("\n", " ").replace("\r", " ")
                f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Enhance prompts for video generation distillation"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=CONFIG["input_file"],
        help="Input file with one prompt per line"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=CONFIG["output_file"],
        help="Output file for enhanced prompts"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=CONFIG["model"],
        help="Model to use for enhancement"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=CONFIG["max_workers"],
        help="Number of concurrent workers"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable checkpoint resume"
    )

    args = parser.parse_args()

    # 更新配置
    config = CONFIG.copy()
    config["input_file"] = args.input
    config["output_file"] = args.output
    config["model"] = args.model
    config["max_workers"] = args.workers
    config["resume"] = not args.no_resume

    if args.api_key:
        config["api_key"] = args.api_key

    # 本地 vLLM 不需要检查 API key

    input_file = Path(args.input)
    output_file = Path(args.output)

    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        return

    print("=" * 60)
    print("Prompt Enhancement Script")
    print("=" * 60)
    print(f"Model:   {config['model']}")
    print(f"Input:   {input_file}")
    print(f"Output:  {output_file}")
    print(f"Workers: {config['max_workers']}")
    print(f"Resume:  {config['resume']}")
    print("=" * 60)

    process_prompts(input_file, output_file, config)


if __name__ == "__main__":
    main()
