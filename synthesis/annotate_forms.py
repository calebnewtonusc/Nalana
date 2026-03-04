"""
annotate_forms.py - VLM-powered form analysis and build sequence generation.

For each rendered Objaverse object (from render_pipeline.py):
  1. Sends 8-view renders to Claude (vision) or local Qwen2-VL via vLLM
  2. Gets deep form analysis + complete Blender build sequence
  3. Saves as JSONL training pairs in data/objaverse/annotations/

This is Stream 2 of the training data: 3D geometric/spatial understanding.

Usage:
    python annotate_forms.py --backend claude
    python annotate_forms.py --backend vllm --vllm-url http://gpu01:8001
    python annotate_forms.py --limit 1000 --workers 8
"""

import argparse
import asyncio
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

from synthesis.prompts import FORM_ANALYSIS_SYSTEM_PROMPT, INTENT_DECOMPOSITION_SYSTEM_PROMPT

OBJAVERSE_DIR  = Path(__file__).parents[1] / "data" / "objaverse"
RENDERS_DIR    = OBJAVERSE_DIR / "renders"
ANNOTATIONS_DIR = OBJAVERSE_DIR / "annotations"
PROCESSED_DIR  = Path(__file__).parents[1] / "data" / "processed"

# Views to include in the prompt (best signal-to-token ratio)
VIEWS_TO_USE = ["front", "iso_front", "top", "right"]


def load_image_b64(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("utf-8")


def build_vision_messages(uid: str, metadata: dict) -> list[dict]:
    render_dir = RENDERS_DIR / uid
    image_content = []

    for view in VIEWS_TO_USE:
        img_path = render_dir / f"{view}.png"
        if img_path.exists():
            image_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": load_image_b64(img_path),
                }
            })

    name = metadata.get("name", uid)
    tags = ", ".join(metadata.get("tags", [])[:10])
    image_content.append({
        "type": "text",
        "text": f"Object name: {name}\nTags: {tags}\n\nAnalyze this 3D object and produce form analysis + Blender build sequence."
    })
    return image_content


async def annotate_claude(client: httpx.AsyncClient, uid: str, metadata: dict, api_key: str) -> dict | None:
    messages_content = build_vision_messages(uid, metadata)
    try:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-opus-4-6",   # Use Opus for best form analysis quality
                "max_tokens": 8192,
                "system": FORM_ANALYSIS_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": messages_content}],
            },
            timeout=180.0,
        )
        resp.raise_for_status()
        raw = resp.json()["content"][0]["text"].strip()
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if fence_match:
            raw = fence_match.group(1).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as parse_err:
            print(f"Warning: annotate_claude JSON parse failed for {uid}: {parse_err}", file=sys.stderr)
            return None
    except Exception as e:
        print(f"Warning: annotate_claude failed for {uid}: {e}", file=sys.stderr)
        return None


async def annotate_vllm(client: httpx.AsyncClient, uid: str, metadata: dict,
                         vllm_url: str, model: str, api_key: str) -> dict | None:
    render_dir = RENDERS_DIR / uid
    image_content = []
    for view in VIEWS_TO_USE:
        img_path = render_dir / f"{view}.png"
        if img_path.exists():
            b64 = load_image_b64(img_path)
            image_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })

    name = metadata.get("name", uid)
    image_content.append({"type": "text", "text": f"Object: {name}. Analyze form and produce Blender build sequence."})

    try:
        resp = await client.post(
            f"{vllm_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": FORM_ANALYSIS_SYSTEM_PROMPT},
                    {"role": "user", "content": image_content},
                ],
                "max_tokens": 8192,
                "temperature": 0.1,
            },
            timeout=180.0,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if fence_match:
            raw = fence_match.group(1).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as parse_err:
            print(f"Warning: annotate_vllm JSON parse failed for {uid}: {parse_err}", file=sys.stderr)
            return None
    except Exception as e:
        print(f"Warning: annotate_vllm failed for {uid}: {e}", file=sys.stderr)
        return None


def annotation_to_training_pairs(uid: str, annotation: dict, metadata: dict) -> list[dict]:
    """Convert a form annotation into individual training pairs for the dataset."""
    pairs = []
    build_sequence = annotation.get("build_sequence", [])
    form = annotation.get("form_analysis", {})
    obj_name = annotation.get("object_name", metadata.get("name", uid))
    category = annotation.get("object_category", "unknown")

    for i, step in enumerate(build_sequence):
        scene_before = (
            "Empty Blender scene" if i == 0
            else f"Partially built {obj_name}: {build_sequence[i-1].get('description', '')}"
        )
        pairs.append({
            "source": "objaverse_3d",
            "uid": uid,
            "object_name": obj_name,
            "object_category": category,
            "voice_command": step.get("voice_command", ""),
            "scene_context": step.get("scene_context", scene_before),
            "blender_op": step.get("blender_op", {}),
            "blender_python": step.get("blender_python", ""),
            "reasoning": f"Step {step.get('step', i+1)} of building {obj_name}: {step.get('description', '')}",
            "form_context": {
                "primary_form": form.get("primary_form", ""),
                "proportions": form.get("proportions", ""),
                "surface_character": form.get("surface_character", ""),
                "modeling_approach": annotation.get("modeling_approach", ""),
            },
        })

    # Also generate the high-level intent → full plan pair
    if build_sequence:
        pairs.append({
            "source": "objaverse_3d_intent",
            "uid": uid,
            "object_name": obj_name,
            "object_category": category,
            "voice_command": f"create a {obj_name}",
            "scene_context": "Empty Blender scene",
            "blender_op": {"op": "MULTI_STEP_PLAN", "args": {}, "target_object": None},
            "blender_python": "\n".join(s.get("blender_python", "") for s in build_sequence[:5]),
            "reasoning": f"High-level: create {obj_name} using {annotation.get('modeling_approach', 'standard modeling')}",
            "full_plan": annotation,
        })
    return pairs


async def annotate_object(
    uid: str, metadata: dict, client: httpx.AsyncClient, semaphore: asyncio.Semaphore,
    backend: str, vllm_url: str, vllm_model: str, vllm_api_key: str, claude_api_key: str,
) -> tuple[str, int]:
    out_path = ANNOTATIONS_DIR / f"{uid}.json"
    pairs_path = ANNOTATIONS_DIR / f"{uid}_pairs.jsonl"

    if out_path.exists() and pairs_path.exists():
        count = sum(1 for l in pairs_path.read_text().splitlines() if l.strip())
        return uid, count

    render_dir = RENDERS_DIR / uid
    if not (render_dir / "front.png").exists():
        return uid, 0

    async with semaphore:
        if backend == "claude":
            annotation = await annotate_claude(client, uid, metadata, claude_api_key)
        else:
            annotation = await annotate_vllm(client, uid, metadata, vllm_url, vllm_model, vllm_api_key)

    if not annotation:
        return uid, 0

    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(annotation, indent=2))

    pairs = annotation_to_training_pairs(uid, annotation, metadata)
    with pairs_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    return uid, len(pairs)


def merge_to_processed():
    """Merge all annotation pairs into the main processed dataset."""
    out = PROCESSED_DIR / "dataset_3d.jsonl"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    all_lines = []
    for p in sorted(ANNOTATIONS_DIR.glob("*_pairs.jsonl")):
        all_lines.extend(l for l in p.read_text().splitlines() if l.strip())
    out.write_text("\n".join(all_lines) + "\n")
    return len(all_lines), out


def load_metadata() -> dict[str, dict]:
    meta_path = OBJAVERSE_DIR / "metadata.jsonl"
    if not meta_path.exists():
        return {}
    result = {}
    for line in meta_path.read_text().splitlines():
        if line.strip():
            obj = json.loads(line)
            result[obj["uid"]] = obj
    return result


async def run(
    uids: list[str], metadata_map: dict, backend: str,
    vllm_url: str, vllm_model: str, vllm_api_key: str, claude_api_key: str,
    concurrency: int,
):
    semaphore = asyncio.Semaphore(concurrency)
    total_pairs = 0
    done = 0
    start = time.time()

    async with httpx.AsyncClient() as client:
        tasks = [
            annotate_object(
                uid, metadata_map.get(uid, {}), client, semaphore,
                backend, vllm_url, vllm_model, vllm_api_key, claude_api_key,
            )
            for uid in uids
        ]
        for coro in asyncio.as_completed(tasks):
            uid, n = await coro
            total_pairs += n
            done += 1
            if done % 10 == 0:
                rate = done / max(time.time() - start, 1)
                print(f"  {done}/{len(uids)} | {total_pairs} pairs | {rate:.1f} obj/s")

    return total_pairs


def main():
    parser = argparse.ArgumentParser(description="Annotate 3D objects with form analysis + build sequences")
    parser.add_argument("--backend", choices=["claude", "vllm"], default="claude")
    parser.add_argument("--vllm-url", default="http://localhost:8001")
    parser.add_argument("--vllm-model", default="Qwen/Qwen2-VL-72B-Instruct")
    parser.add_argument("--vllm-api-key", default=os.environ.get("VLLM_API_KEY", "nalana"))
    parser.add_argument("--claude-api-key", default=os.environ.get("ANTHROPIC_API_KEY"))
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.backend == "claude" and not args.claude_api_key:
        raise ValueError("Set ANTHROPIC_API_KEY environment variable")

    if args.concurrency is None:
        args.concurrency = 8 if args.backend == "vllm" else 3  # Vision is slower

    metadata_map = load_metadata()
    rendered = [p.name for p in RENDERS_DIR.glob("*") if (RENDERS_DIR / p.name / "front.png").exists()]

    if not args.force:
        done = set(p.stem for p in ANNOTATIONS_DIR.glob("*.json")) if ANNOTATIONS_DIR.exists() else set()
        rendered = [uid for uid in rendered if uid not in done]

    if args.limit:
        rendered = rendered[:args.limit]

    if not rendered:
        print("Nothing to annotate. Run render_pipeline.py first.")
        return

    print(f"Backend:     {args.backend}")
    print(f"Objects:     {len(rendered)}")
    print(f"Concurrency: {args.concurrency}")
    print()

    start = time.time()
    total = asyncio.run(run(
        rendered, metadata_map, args.backend,
        args.vllm_url, args.vllm_model, args.vllm_api_key, args.claude_api_key,
        args.concurrency,
    ))

    print(f"\nDone in {(time.time()-start)/60:.1f}min. {total} training pairs generated.")
    count, out = merge_to_processed()
    print(f"Merged → {out} ({count} total Stream 2 pairs)")
    print("\nNext step: python train_prep.py")


if __name__ == "__main__":
    main()
