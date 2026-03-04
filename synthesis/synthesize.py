"""
synthesize.py - Use Claude to extract (voice_command, blender_op) training pairs
from raw tutorial transcripts.

Each output record is a JSONL line:
{
  "video_id": str,
  "chunk_start": float,         # seconds into video
  "chunk_end": float,
  "transcript": str,            # what the instructor said
  "voice_command": str,         # natural NL a user would speak
  "scene_context": str,         # what's in the scene at this moment
  "blender_op": {
    "op": str,                  # e.g. "mesh.primitive_cube_add"
    "args": {...},              # kwargs passed to the op
    "target_object": str | null # object being acted on
  },
  "blender_python": str,        # e.g. bpy.ops.mesh.primitive_cube_add(size=2)
  "reasoning": str              # why this operation maps to the command
}

Usage:
    python synthesize.py                         # all raw files
    python synthesize.py --video-id VIDEO_ID     # single video
    python synthesize.py --force                 # re-process cached
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic

from synthesis.prompts import TUTORIAL_SYSTEM_PROMPT

RAW_DIR = Path(__file__).parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).parents[1] / "data" / "processed"

# Group chunks into batches to reduce API calls (each batch ~2000 tokens of transcript)
CHUNKS_PER_BATCH = 6


def build_user_prompt(chunks: list[dict], video_url: str) -> str:
    segments_text = "\n\n".join(
        f"[{c['start']:.0f}s - {c['end']:.0f}s]\n{c['text']}"
        for c in chunks
    )
    return f"""Tutorial URL: {video_url}

Transcript segments:
{segments_text}

Extract all executable Blender operations from these segments. Output a JSON array."""


def extract_pairs_from_batch(
    client: anthropic.Anthropic,
    chunks: list[dict],
    video_url: str,
    video_id: str,
) -> list[dict]:
    prompt = build_user_prompt(chunks, video_url)

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
            system=TUTORIAL_SYSTEM_PROMPT,
        )
        raw = message.content[0].text.strip()

        # Strip markdown code fences if Claude wraps output
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if fence_match:
            raw = fence_match.group(1).strip()
        raw = raw.strip()

        pairs = json.loads(raw)
        if not isinstance(pairs, list):
            return []

        # Annotate each pair with source metadata
        for pair in pairs:
            pair["video_id"] = video_id
            pair["chunk_start"] = chunks[0]["start"]
            pair["chunk_end"] = chunks[-1]["end"]
            pair["transcript"] = " ".join(c["text"] for c in chunks)

        return pairs

    except json.JSONDecodeError as e:
        print(f"    [PARSE ERROR] Could not parse Claude response: {e}")
        return []
    except anthropic.APIError as e:
        print(f"    [API ERROR] {e}")
        return []


def synthesize_video(video_id: str, force: bool = False) -> Path | None:
    raw_path = RAW_DIR / f"{video_id}.json"
    if not raw_path.exists():
        print(f"  [SKIP] No raw file for {video_id}")
        return None

    out_path = PROCESSED_DIR / f"{video_id}.jsonl"
    if out_path.exists() and not force:
        print(f"  [CACHED] {video_id} -> {out_path.name}")
        return out_path

    data = json.loads(raw_path.read_text())
    chunks = data["chunks"]
    url = data["url"]
    print(f"  [PROCESS] {video_id} ({len(chunks)} chunks, {len(chunks)//CHUNKS_PER_BATCH + 1} batches)")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Set ANTHROPIC_API_KEY environment variable")
    client = anthropic.Anthropic(api_key=api_key)

    all_pairs: list[dict] = []

    for i in range(0, len(chunks), CHUNKS_PER_BATCH):
        batch = chunks[i : i + CHUNKS_PER_BATCH]
        batch_num = i // CHUNKS_PER_BATCH + 1
        total_batches = (len(chunks) - 1) // CHUNKS_PER_BATCH + 1
        print(f"    batch {batch_num}/{total_batches} ({batch[0]['start']:.0f}s - {batch[-1]['end']:.0f}s)")

        pairs = extract_pairs_from_batch(client, batch, url, video_id)
        all_pairs.extend(pairs)
        print(f"      -> {len(pairs)} operation(s) extracted")

        # Be polite to the API between batches
        if i + CHUNKS_PER_BATCH < len(chunks):
            time.sleep(0.5)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"  [SAVED] {len(all_pairs)} training pairs -> {out_path.name}")
    return out_path


def merge_all_jsonl(out_path: Path) -> int:
    """Merge all per-video JSONL files into one master dataset."""
    all_lines = []
    for p in sorted(PROCESSED_DIR.glob("*.jsonl")):
        if p == out_path:
            continue
        all_lines.extend(p.read_text().splitlines())
    out_path.write_text("\n".join(all_lines) + ("\n" if all_lines else ""))
    return len(all_lines)


def main():
    parser = argparse.ArgumentParser(description="Synthesize Blender training pairs from transcripts")
    parser.add_argument("--video-id", help="Process a single video ID")
    parser.add_argument("--force", action="store_true", help="Re-process even if cached")
    parser.add_argument("--merge", action="store_true", help="Merge all JSONL into dataset.jsonl after processing")
    args = parser.parse_args()

    if args.video_id:
        video_ids = [args.video_id]
    else:
        video_ids = [p.stem for p in sorted(RAW_DIR.glob("*.json"))]

    if not video_ids:
        print(f"No raw transcripts found in {RAW_DIR}/. Run fetch.py first.")
        return

    print(f"Synthesizing {len(video_ids)} video(s)...\n")
    processed = 0
    for vid in video_ids:
        result = synthesize_video(vid, force=args.force)
        if result:
            processed += 1
        print()

    print(f"Done. {processed}/{len(video_ids)} videos processed.")

    if args.merge or not args.video_id:
        master = PROCESSED_DIR / "dataset.jsonl"
        count = merge_all_jsonl(master)
        print(f"Master dataset: {count} total training pairs -> {master}")


if __name__ == "__main__":
    main()
