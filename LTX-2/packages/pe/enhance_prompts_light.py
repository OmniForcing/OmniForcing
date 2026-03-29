#!/usr/bin/env python3
"""
Prompt Enhancement Script (Light Version) for Distillation Dataset

Batch-expands short captions into detailed LTX-2 prompts with visual and audio
descriptions using a local vLLM server with OpenAI-compatible API.

Usage:
    python enhance_prompts_light.py --input captions.txt --output prompts.txt

Or edit the CONFIG dict below and run directly:
    python enhance_prompts_light.py
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
# Configuration (edit directly or override via CLI args)
# =============================================================================

CONFIG = {
    # vLLM local API config (OpenAI-compatible format)
    "api_key": "EMPTY",                 # Local vLLM needs no key
    "api_base": "http://localhost:8000/v1/chat/completions",  # vLLM default port
    "model": "gemma-3-12b",              # --served-model-name when launching vLLM

    # Generation parameters
    "temperature": 0.7,
    "max_tokens": 4096,
    "top_p": 0.9,

    # Input/output paths (CLI args override these)
    "input_file": "./captions.txt",
    "output_file": "./captions_enhanced.txt",

    # System prompt file path
    "system_prompt_file": os.path.join(os.path.dirname(__file__), "light_system_prompt.txt"),

    # Concurrency and retry config
    "max_workers": 32,          # Concurrent threads, adjust by GPU memory
    "retry_times": 3,           # Retry count on failure
    "retry_delay": 1.0,         # Retry interval (seconds)
    "request_delay": 0.05,      # Inter-request delay (seconds)

    # Checkpoint resume
    "resume": True,             # Enable checkpoint resume
    "checkpoint_interval": 10,  # Save checkpoint every N items

    # Misc
    "skip_empty_lines": True,   # Skip empty lines
    "verbose": True,            # Show detailed logs
    "skip_on_error": True,      # Skip failed prompts (don't write to output)
}

# =============================================================================
# Core functions
# =============================================================================

def load_system_prompt(filepath: Path) -> str:
    """Load system prompt from file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()


def enhance_single_prompt(
    prompt: str,
    system_prompt: str,
    config: dict,
) -> str | None:
    """
    Enhance a single prompt using local vLLM API (OpenAI-compatible format).

    Returns:
        Enhanced prompt string, or None on failure (prompt will be skipped).
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

            # Check for API error
            if "error" in result:
                error_msg = result.get("error", {})
                if isinstance(error_msg, dict):
                    error_msg = error_msg.get("message", str(error_msg))
                raise RuntimeError(f"API error: {error_msg}")

            # Check rate limit
            if response.status_code == 429:
                wait_time = config["retry_delay"] * (attempt + 2)
                tqdm.write(f"[WARN] Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue

            response.raise_for_status()

            # Parse response
            if "choices" not in result or len(result["choices"]) == 0:
                raise KeyError(f"No 'choices' in response: {str(result)[:200]}")

            enhanced = result["choices"][0]["message"]["content"].strip()

            # Clean formatting issues (strip leading/trailing quotes)
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
                return None

    return None


def load_checkpoint(output_file: Path) -> set[int]:
    """Load processed line indices for checkpoint resume."""
    checkpoint_file = output_file.with_suffix(".checkpoint")
    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            return set(map(int, f.read().strip().split("\n"))) if f.read().strip() else set()
    return set()


def save_checkpoint(output_file: Path, processed_indices: set[int]):
    """Save checkpoint file."""
    checkpoint_file = output_file.with_suffix(".checkpoint")
    with open(checkpoint_file, "w") as f:
        f.write("\n".join(map(str, sorted(processed_indices))))


def process_prompts(
    input_file: Path,
    output_file: Path,
    config: dict,
):
    """Batch-process all prompts with concurrent enhancement."""
    # Load system prompt
    system_prompt = load_system_prompt(config["system_prompt_file"])
    print(f"Loaded system prompt from: {config['system_prompt_file']}")

    # Read input file
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    prompts = []
    for i, line in enumerate(lines):
        line = line.strip()
        if config["skip_empty_lines"] and not line:
            continue
        prompts.append((i, line))

    print(f"Loaded {len(prompts)} prompts from: {input_file}")

    # Checkpoint resume: load previously processed indices
    processed_indices = set()
    results = {}

    if config["resume"] and output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            existing_lines = f.readlines()

        checkpoint_file = output_file.with_suffix(".checkpoint")
        if checkpoint_file.exists():
            with open(checkpoint_file, "r") as f:
                content = f.read().strip()
                if content:
                    processed_indices = set(map(int, content.split("\n")))

        # If no checkpoint but output exists, assume sequential processing
        if not processed_indices and existing_lines:
            processed_indices = set(range(len(existing_lines)))

        for idx, line in enumerate(existing_lines):
            if idx < len(prompts):
                results[prompts[idx][0]] = line.strip()

        print(f"Resuming: {len(processed_indices)} prompts already processed")

    # Filter remaining prompts
    remaining = [(i, p) for i, p in prompts if i not in processed_indices]
    print(f"Remaining: {len(remaining)} prompts to process")

    if not remaining:
        print("All prompts already processed!")
        return

    write_lock = Lock()
    progress_bar = tqdm(total=len(remaining), desc="Enhancing prompts")

    def process_one(item):
        idx, prompt = item
        time.sleep(config["request_delay"])
        enhanced = enhance_single_prompt(prompt, system_prompt, config)
        return idx, enhanced

    # Concurrent processing
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

                # Periodic checkpoint save
                if checkpoint_counter % config["checkpoint_interval"] == 0:
                    save_checkpoint(output_file, processed_indices)
                    _save_results(output_file, results, prompts)

            progress_bar.update(1)

            if config["verbose"] and checkpoint_counter % 50 == 0:
                tqdm.write(f"[Progress] {checkpoint_counter}/{len(remaining)} processed, {skipped_count} skipped")

    progress_bar.close()

    # Final save
    _save_results(output_file, results, prompts)

    # Remove checkpoint file
    checkpoint_file = output_file.with_suffix(".checkpoint")
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    print(f"\nDone! Enhanced prompts saved to: {output_file}")
    print(f"Total: {len(results)} prompts, {skipped_count} skipped due to errors")


def _save_results(output_file: Path, results: dict, prompts: list):
    """Save successful results in original order, skip failed prompts."""
    with open(output_file, "w", encoding="utf-8") as f:
        for original_idx, _ in prompts:
            if original_idx in results and results[original_idx] is not None:
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

    # Update config from CLI args
    config = CONFIG.copy()
    config["input_file"] = args.input
    config["output_file"] = args.output
    config["model"] = args.model
    config["max_workers"] = args.workers
    config["resume"] = not args.no_resume

    if args.api_key:
        config["api_key"] = args.api_key

    input_file = Path(args.input)
    output_file = Path(args.output)

    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        return

    print("=" * 60)
    print("Prompt Enhancement Script (Light)")
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
