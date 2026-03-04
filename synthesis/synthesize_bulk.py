"""
synthesize_bulk.py - Async parallel synthesis at scale using vLLM or Claude.

On the A6000 cluster:
  - Deploy vLLM with Qwen2.5-72B-Instruct (4 GPUs per instance, tensor parallel)
  - Run 4 vLLM instances → 4 synthesis workers with zero API cost
  - Throughput: ~10,000 videos in 8-12 hours overnight

Locally / prototyping:
  - Falls back to Anthropic Claude API (rate-limited but same code path)

vLLM setup on A6000 cluster (run once before this script):
  pip install vllm
  # Instance 1 (GPUs 0-3):
  CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 --port 8001 --api-key nalana
  # Instance 2 (GPUs 4-7):
  CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 --port 8002 --api-key nalana
  # ... repeat for ports 8003, 8004 (GPUs 8-11, 12-15)
  # GPUs 16-17 spare for fine-tuning

Usage:
    # Production (4 vLLM instances on the cluster):
    python synthesize_bulk.py --backend vllm \
        --vllm-urls http://gpu01:8001 http://gpu01:8002 http://gpu01:8003 http://gpu01:8004

    # Prototyping (Claude API):
    python synthesize_bulk.py --backend claude

    # Limit for testing:
    python synthesize_bulk.py --backend vllm --vllm-urls http://localhost:8001 --limit 100
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

from synthesis.prompts import TUTORIAL_SYSTEM_PROMPT, CROSS_SOFTWARE_SYSTEM_PROMPT, PHYSICS_REASONING_SYSTEM_PROMPT

try:
    from tqdm.asyncio import tqdm as async_tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

RAW_DIR       = Path(__file__).parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).parents[1] / "data" / "processed"
MASTER_JSONL  = PROCESSED_DIR / "dataset.jsonl"

CHUNKS_PER_BATCH = 6  # ~2000 tokens of transcript per API call

# Software → system prompt routing
_SOFTWARE_PROMPTS: dict[str, str] = {
    "blender":   TUTORIAL_SYSTEM_PROMPT,
    "maya":      CROSS_SOFTWARE_SYSTEM_PROMPT,
    "houdini":   CROSS_SOFTWARE_SYSTEM_PROMPT,
    "c4d":       CROSS_SOFTWARE_SYSTEM_PROMPT,
    "cinema4d":  CROSS_SOFTWARE_SYSTEM_PROMPT,
    "rhino":     CROSS_SOFTWARE_SYSTEM_PROMPT,
    "unreal":    CROSS_SOFTWARE_SYSTEM_PROMPT,
    "substance": CROSS_SOFTWARE_SYSTEM_PROMPT,
    "zbrush":    CROSS_SOFTWARE_SYSTEM_PROMPT,
    "multi":     CROSS_SOFTWARE_SYSTEM_PROMPT,
    "physics":   PHYSICS_REASONING_SYSTEM_PROMPT,
}

def _system_prompt_for(software: str) -> str:
    return _SOFTWARE_PROMPTS.get(software.lower(), TUTORIAL_SYSTEM_PROMPT)


def build_prompt(chunks: list[dict], url: str, software: str = "blender") -> str:
    segs = "\n\n".join(
        f"[{c['start']:.0f}s–{c['end']:.0f}s]\n{c['text']}"
        for c in chunks
    )
    sw_label = software if software != "blender" else "Blender"
    action = (
        "Extract all executable cross-software 3D operations and map each to the Universal DSL."
        if software not in ("blender", "physics")
        else "Extract all executable Blender operations."
    )
    return f"Tutorial ({sw_label}): {url}\n\nTranscript:\n{segs}\n\n{action} Output JSON array."


# ─── Backend implementations ───────────────────────────────────────────────────

async def call_vllm(
    client: httpx.AsyncClient,
    base_url: str,
    prompt: str,
    system_prompt: str,
    model: str = "Qwen/Qwen2.5-72B-Instruct",
    api_key: str = "nalana",
) -> str:
    """OpenAI-compatible chat completions endpoint that vLLM exposes."""
    resp = await client.post(
        f"{base_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 4096,
            "temperature": 0.1,  # low temp for consistent structured output
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def call_claude(
    client: httpx.AsyncClient,
    prompt: str,
    api_key: str,
    system_prompt: str,
) -> str:
    resp = await client.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


def parse_pairs(text: str) -> list[dict]:
    # Try to find any JSON fence block (json, python, text, or plain ```)
    fence_match = re.search(r'```(?:json|python|text|)\s*([\s\S]*?)```', text)
    if fence_match:
        json_str = fence_match.group(1).strip()
    else:
        # Use raw_decode to find the first valid JSON value without greedy matching
        decoder = json.JSONDecoder()
        # Scan for '[' or '{' and try to parse from there
        for start_char in ('[', '{'):
            idx = text.find(start_char)
            if idx == -1:
                continue
            try:
                result, _ = decoder.raw_decode(text, idx)
                if isinstance(result, list):
                    return result
                return [result]
            except json.JSONDecodeError:
                continue
        return []
    try:
        result = json.loads(json_str)
        if isinstance(result, list):
            return result
        return [result]
    except json.JSONDecodeError:
        return []


# ─── Core synthesis logic ──────────────────────────────────────────────────────

async def synthesize_video(
    video_id: str,
    client: httpx.AsyncClient,
    backend: str,
    vllm_urls: list[str],
    vllm_model: str,
    vllm_api_key: str,
    claude_api_key: str,
    worker_idx: int,
    semaphore: asyncio.Semaphore,
) -> tuple[str, int]:
    """Returns (video_id, num_pairs). Writes to processed/{video_id}.jsonl."""
    out_path = PROCESSED_DIR / f"{video_id}.jsonl"
    if out_path.exists():
        count = sum(1 for l in out_path.read_text().splitlines() if l.strip())
        return video_id, count

    raw_path = RAW_DIR / f"{video_id}.json"
    if not raw_path.exists():
        return video_id, 0

    data = json.loads(raw_path.read_text())
    chunks = data.get("chunks", [])
    url = data.get("url", f"https://www.youtube.com/watch?v={video_id}")
    # software field is set by fetch_bulk if source metadata is available; default blender
    software = data.get("software", "blender")
    sys_prompt = _system_prompt_for(software)

    all_pairs: list[dict] = []

    for i in range(0, len(chunks), CHUNKS_PER_BATCH):
        batch = chunks[i : i + CHUNKS_PER_BATCH]
        prompt = build_prompt(batch, url, software)

        async with semaphore:
            try:
                if backend == "vllm":
                    # Round-robin across vLLM instances
                    base_url = vllm_urls[worker_idx % len(vllm_urls)]
                    raw_response = await call_vllm(client, base_url, prompt, sys_prompt, vllm_model, vllm_api_key)
                else:
                    raw_response = await call_claude(client, prompt, claude_api_key, sys_prompt)

                pairs = parse_pairs(raw_response)
                for p in pairs:
                    p["video_id"] = video_id
                    p["chunk_start"] = batch[0]["start"]
                    p["chunk_end"] = batch[-1]["end"]
                    p["transcript"] = " ".join(c["text"] for c in batch)
                all_pairs.extend(pairs)

            except Exception as e:
                # Non-fatal: log and continue with next batch
                print(f"Warning: synthesis failed for {video_id} batch {i}: {e}", file=sys.stderr)

        # Small delay between batches only for Claude (rate limits)
        if backend == "claude":
            await asyncio.sleep(0.3)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    return video_id, len(all_pairs)


async def run_all(
    video_ids: list[str],
    backend: str,
    vllm_urls: list[str],
    vllm_model: str,
    vllm_api_key: str,
    claude_api_key: str,
    concurrency: int,
):
    # vLLM: high concurrency (no external rate limits)
    # Claude: lower concurrency (50 RPM hard limit → 5-8 workers safe)
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        tasks = [
            synthesize_video(
                vid, client, backend,
                vllm_urls, vllm_model, vllm_api_key, claude_api_key,
                idx, semaphore,
            )
            for idx, vid in enumerate(video_ids)
        ]

        total_pairs = 0
        completed = 0
        start = time.time()

        if HAS_TQDM:
            pbar = async_tqdm(total=len(tasks), unit="video", ncols=80)
        else:
            pbar = None

        for coro in asyncio.as_completed(tasks):
            video_id, n_pairs = await coro
            total_pairs += n_pairs
            completed += 1

            if pbar:
                rate = completed / max(time.time() - start, 1)
                pbar.set_postfix(pairs=total_pairs, rate=f"{rate:.1f}v/s")
                pbar.update(1)
            elif completed % 50 == 0:
                elapsed = time.time() - start
                eta = (len(tasks) - completed) / max(completed / elapsed, 0.001)
                print(f"  {completed}/{len(tasks)} videos | {total_pairs} pairs | ETA {eta/60:.0f}min")

        if pbar:
            pbar.close()

    return total_pairs


def merge_master():
    """Merge all per-video JSONL into master dataset.jsonl"""
    all_lines = []
    for p in sorted(PROCESSED_DIR.glob("*.jsonl")):
        if p == MASTER_JSONL:
            continue
        all_lines.extend(l for l in p.read_text().splitlines() if l.strip())
    MASTER_JSONL.write_text("\n".join(all_lines) + "\n")
    return len(all_lines)


def main():
    parser = argparse.ArgumentParser(description="Bulk async synthesis of 3D training pairs (all software)")
    parser.add_argument("--backend", choices=["vllm", "claude"], default="claude")
    parser.add_argument("--vllm-urls", nargs="+", default=["http://localhost:8001"],
                        help="vLLM server URLs (one per instance)")
    parser.add_argument("--sources", default=None,
                        help="Path to discovered_sources.json — used to tag videos with software metadata")
    parser.add_argument("--vllm-model", default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--vllm-api-key", default=os.environ.get("VLLM_API_KEY", "nalana"))
    parser.add_argument("--claude-api-key", default=os.environ.get("ANTHROPIC_API_KEY"))
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Parallel synthesis workers (default: 40 for vLLM, 6 for claude)")
    parser.add_argument("--limit", type=int, help="Max videos to process")
    parser.add_argument("--force", action="store_true", help="Re-process already synthesized videos")
    args = parser.parse_args()

    if args.backend == "claude" and not args.claude_api_key:
        raise ValueError("Set ANTHROPIC_API_KEY environment variable")

    # Auto concurrency
    if args.concurrency is None:
        args.concurrency = 40 if args.backend == "vllm" else 6

    # Load pending video IDs
    raw_files = sorted(RAW_DIR.glob("*.json"))
    if args.limit:
        raw_files = raw_files[:args.limit]

    if not args.force:
        done = set(p.stem for p in PROCESSED_DIR.glob("*.jsonl")) if PROCESSED_DIR.exists() else set()
        raw_files = [p for p in raw_files if p.stem not in done]

    video_ids = [p.stem for p in raw_files]

    if not video_ids:
        print("Nothing to synthesize. Run fetch_bulk.py first.")
        return

    print(f"Backend:     {args.backend}")
    if args.backend == "vllm":
        print(f"vLLM URLs:   {args.vllm_urls}")
        print(f"Model:       {args.vllm_model}")
    print(f"Videos:      {len(video_ids)}")
    print(f"Concurrency: {args.concurrency} workers")
    print()

    start = time.time()
    total_pairs = asyncio.run(run_all(
        video_ids,
        backend=args.backend,
        vllm_urls=args.vllm_urls,
        vllm_model=args.vllm_model,
        vllm_api_key=args.vllm_api_key,
        claude_api_key=args.claude_api_key,
        concurrency=args.concurrency,
    ))

    elapsed = time.time() - start
    print(f"\n{'─'*40}")
    print(f"Videos processed: {len(video_ids)}")
    print(f"Training pairs:   {total_pairs}")
    print(f"Time:             {elapsed/60:.1f} min  ({len(video_ids)/max(elapsed,1):.1f} videos/s)")

    print("\nMerging master dataset...")
    total = merge_master()
    print(f"dataset.jsonl: {total} total pairs")
    print(f"\nNext step: python train_prep.py")


if __name__ == "__main__":
    main()
