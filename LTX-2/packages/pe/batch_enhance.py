"""
batch_enhance.py — Batch LTX-2 prompt enhancer
================================================
Reads a txt file (one short caption per line), expands each into a
full LTX-2 prompt, writes successful results to an output file.

Failed lines are DROPPED (not written) to keep output clean for training.
A .progress file tracks which input lines have been attempted, so you can
resume interrupted runs without re-processing.

Usage:
  # Process all captions on one machine
  python batch_enhance.py captions.txt

  # Split across 2 machines (no special distributed setup needed)
  python batch_enhance.py captions.txt --shards 2 --shard-id 0   # machine A
  python batch_enhance.py captions.txt --shards 2 --shard-id 1   # machine B

  # Override duration from CLI
  python batch_enhance.py captions.txt --duration 10s

Output:
  captions_enhanced_5s.txt             (no sharding)
  captions_enhanced_5s_shard0of2.txt   (with sharding)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

from prompt_enhancer import LTX2PromptExpander

# ═══════════════════════════════════════════════════════════════════════════
# Config — edit here, one place, done
# ═══════════════════════════════════════════════════════════════════════════

DURATION       = "5s"       # "5s" / "10s" / "20s"
VLLM_BASE_URL  = "http://localhost:8000/v1"
API_KEY        = "empty"
MAX_TOKENS     = 16384
MAX_CONCURRENT = 512
MAX_RETRIES    = 3
RETRY_DELAY    = 5.0

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def read_captions(path: str) -> list[str]:
    return [l.strip() for l in Path(path).read_text("utf-8").splitlines() if l.strip()]


def make_paths(input_path: str, duration: str, num_shards: int = 1, shard_id: int = 0):
    p = Path(input_path)
    if num_shards > 1:
        suffix = f"_enhanced_{duration}_shard{shard_id}of{num_shards}"
    else:
        suffix = f"_enhanced_{duration}"
    out  = p.parent / f"{p.stem}{suffix}.txt"
    prog = p.parent / f"{p.stem}{suffix}.progress"
    return str(out), str(prog)


def load_done(prog_file: str) -> set[int]:
    """Load set of input line indices that have already been attempted."""
    p = Path(prog_file)
    if not p.exists():
        return set()
    done = set()
    for line in p.read_text("utf-8").splitlines():
        line = line.strip()
        if line.isdigit():
            done.add(int(line))
    return done


def mark_done(prog_file: str, idx: int):
    """Append one index to progress file."""
    with open(prog_file, "a", encoding="utf-8") as f:
        f.write(f"{idx}\n")
        f.flush()


def append_result(out_file: str, prompt: str):
    """Append one clean prompt line."""
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(prompt.replace("\n", " ").strip() + "\n")
        f.flush()


# ═══════════════════════════════════════════════════════════════════════════
# Core
# ═══════════════════════════════════════════════════════════════════════════

async def process_one(
    client: LTX2PromptExpander,
    caption: str,
    idx: int,
    total: int,
) -> tuple[bool, str]:
    """Try to expand one caption with retries. Returns (success, prompt_or_error)."""
    for attempt in range(1, MAX_RETRIES + 1):
        result = await client.expand(caption)
        if result.ok:
            return True, result.prompt
        err = result.error or "(empty prompt)"
        if attempt < MAX_RETRIES:
            logger.warning(f"  [{idx+1}/{total}] attempt {attempt} failed: {err}, retrying...")
            await asyncio.sleep(RETRY_DELAY)
    return False, err


async def run(input_path: str, duration: str = DURATION,
              num_shards: int = 1, shard_id: int = 0):
    captions = read_captions(input_path)
    if not captions:
        print("Error: no captions in input file")
        sys.exit(1)

    total = len(captions)

    # Shard: each machine only takes its slice (by modulo)
    if num_shards > 1:
        my_indices = [i for i in range(total) if i % num_shards == shard_id]
    else:
        my_indices = list(range(total))

    out_file, prog_file = make_paths(input_path, duration, num_shards, shard_id)
    done = load_done(prog_file)
    todo = [(i, captions[i]) for i in my_indices if i not in done]

    ok_count = 0
    fail_count = 0
    skip_count = len(done)

    print("=" * 60)
    print("LTX-2 Batch Prompt Enhancer")
    print(f"  Input:          {input_path} ({total} captions)")
    print(f"  Output:         {out_file}")
    print(f"  Duration:       {duration}")
    if num_shards > 1:
        print(f"  Shard:          {shard_id}/{num_shards} ({len(my_indices)} captions in this shard)")
    print(f"  vLLM:           {VLLM_BASE_URL}")
    print(f"  Max concurrent: {MAX_CONCURRENT}")
    print(f"  Max retries:    {MAX_RETRIES}")
    if skip_count:
        print(f"  Resuming:       {skip_count} already done, {len(todo)} remaining")
    print("=" * 60)

    if not todo:
        print("\n  All captions already processed.")
        return

    t0 = time.perf_counter()
    write_lock = asyncio.Lock()
    pbar = tqdm(total=len(todo), desc="Enhancing", unit="cap",
                dynamic_ncols=True)

    async def worker(idx: int, caption: str):
        """Process one caption, write result under lock."""
        nonlocal ok_count, fail_count
        success, result = await process_one(client, caption, idx, total)

        async with write_lock:
            if success:
                append_result(out_file, result)
                mark_done(prog_file, idx)
                ok_count += 1
            else:
                mark_done(prog_file, idx)
                fail_count += 1
            pbar.set_postfix(ok=ok_count, fail=fail_count, refresh=False)
            pbar.update(1)

    async with LTX2PromptExpander(
        api_key=API_KEY,
        base_url=VLLM_BASE_URL,
        max_tokens=MAX_TOKENS,
        max_concurrent=MAX_CONCURRENT,
        duration=duration,
    ) as client:
        print(f"  Model: {client.model}\n")

        # Fire all tasks concurrently — semaphore inside client limits inflight requests
        await asyncio.gather(*[worker(idx, cap) for idx, cap in todo])

    pbar.close()
    elapsed = time.perf_counter() - t0
    out_lines = len(Path(out_file).read_text("utf-8").splitlines()) if Path(out_file).exists() else 0

    print(f"\n{'=' * 60}")
    print(f"  Done in {elapsed:.1f}s")
    print(f"  This run: {ok_count} succeeded, {fail_count} dropped")
    print(f"  Output:   {out_file} ({out_lines} total lines)")
    if num_shards > 1:
        p = Path(input_path)
        merged = f"{p.stem}_enhanced_{duration}.txt"
        pattern = f"{p.stem}_enhanced_{duration}_shard*of{num_shards}.txt"
        print(f"\n  To merge all shards after all machines finish:")
        print(f"    cat {p.parent}/{pattern} > {p.parent}/{merged}")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch LTX-2 prompt enhancer")
    parser.add_argument("input", help="Input txt file (one caption per line)")
    parser.add_argument("--duration", default=DURATION, choices=["5s", "10s", "20s"],
                        help=f"Duration mode (default: {DURATION})")
    parser.add_argument("--shards", type=int, default=1,
                        help="Total number of machines/shards (default: 1 = no sharding)")
    parser.add_argument("--shard-id", type=int, default=0,
                        help="This machine's shard index, 0-based (default: 0)")
    args = parser.parse_args()

    if args.shard_id >= args.shards:
        print(f"Error: --shard-id {args.shard_id} must be < --shards {args.shards}")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    asyncio.run(run(args.input, args.duration, args.shards, args.shard_id))
