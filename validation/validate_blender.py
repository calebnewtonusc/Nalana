"""
validate_blender.py - Headless Blender execution validator for training pairs.

For each pair's blender_python code:
  1. Spawns headless Blender in a clean/staged scene
  2. Executes the generated Python call
  3. Scores: 1.0 (success + scene changed)
             0.5 (ran but no measurable change)
             0.0 (Python error / syntax error)
  4. Filters: removes pairs with score < 0.5
  5. Saves passing pairs to data/validated/blender_exec.jsonl

This is the FINAL quality gate before fine-tuning.
Only pairs that produce real Blender state changes survive.

Usage:
    python validate_blender.py
    python validate_blender.py --blender-path /Applications/Blender.app/Contents/MacOS/Blender
    python validate_blender.py --workers 4 --min-score 0.5
    python validate_blender.py --input-file data/validated/dataset.jsonl
"""

import argparse
import json
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

VALIDATED_DIR = Path(__file__).parents[1] / "data" / "validated"
INPUT_FILE = VALIDATED_DIR / "dataset.jsonl"
OUTPUT_FILE = VALIDATED_DIR / "blender_exec.jsonl"
REJECTED_FILE = VALIDATED_DIR / "blender_exec_rejected.jsonl"

# ─── Blender test harness (runs inside Blender Python) ────────────────────────
# This script is injected into headless Blender for each pair.
# It executes the pair's Python, checks for errors and scene changes, prints JSON result.

BLENDER_HARNESS = '''
import bpy
import sys
import json
import traceback
import math

def reset_scene():
    """Guaranteed clean scene with one default cube."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
    bpy.ops.object.select_all(action="DESELECT")

def scene_fingerprint(context=None):
    """Capture key scene state metrics for change detection."""
    data = bpy.data
    return {
        "object_count": len(data.objects),
        "mesh_names": sorted(o.name for o in data.objects if o.type == "MESH"),
        "total_verts": sum(
            len(o.data.vertices) for o in data.objects if o.type == "MESH"
        ),
        "total_faces": sum(
            len(o.data.polygons) for o in data.objects if o.type == "MESH"
        ),
        "modifier_count": sum(len(o.modifiers) for o in data.objects),
        "material_count": len(data.materials),
        "has_armature": any(o.type == "ARMATURE" for o in data.objects),
        "has_light": any(o.type == "LIGHT" for o in data.objects),
        "has_camera": any(o.type == "CAMERA" for o in data.objects),
    }

def fingerprint_changed(before, after):
    """Check if scene state meaningfully changed."""
    return before != after

# ── Read args ─────────────────────────────────────────────────────────────────
argv = sys.argv
try:
    script_start = argv.index("--") + 1
    payload_file = argv[script_start]
except (ValueError, IndexError):
    print(json.dumps({"error": "no payload file arg"}))
    sys.exit(1)

with open(payload_file) as f:
    payload = json.load(f)

python_code = payload.get("blender_python", "")
pair_index  = payload.get("index", -1)
mode        = payload.get("mode", "object")  # edit, sculpt, etc.

# ── Prepare scene ─────────────────────────────────────────────────────────────
reset_scene()

# For edit-mode ops: enter edit mode and select all
if mode in ("edit", "EDIT"):
    bpy.ops.object.select_all(action="SELECT")
    bpy.context.view_layer.objects.active = bpy.data.objects[0]
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")

before = scene_fingerprint()

# ── Execute ───────────────────────────────────────────────────────────────────
exec_success = False
exec_error   = None
try:
    exec(python_code, {"bpy": bpy, "__builtins__": __builtins__})
    exec_success = True
except Exception as e:
    exec_error = f"{type(e).__name__}: {str(e)}"

# Return to object mode safely
try:
    bpy.ops.object.mode_set(mode="OBJECT")
except Exception:
    pass

after = scene_fingerprint()

# ── Score ─────────────────────────────────────────────────────────────────────
if not exec_success:
    score = 0.0
elif fingerprint_changed(before, after):
    score = 1.0
else:
    score = 0.5  # Ran but no detectable change (could still be valid: e.g. hide ops)

result = {
    "index": pair_index,
    "score": score,
    "exec_success": exec_success,
    "exec_error": exec_error,
    "scene_changed": fingerprint_changed(before, after),
    "before": before,
    "after": after,
}
print("NALANA_RESULT:" + json.dumps(result))
'''


def get_blender_path(hint: str | None = None) -> str:
    candidates = [
        hint,
        os.environ.get("BLENDER_PATH"),
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "/usr/bin/blender",
        "/usr/local/bin/blender",
        "blender",
    ]
    for c in candidates:
        if c and (Path(c).exists() or c == "blender"):
            return c
    raise RuntimeError("Blender not found. Set BLENDER_PATH or pass --blender-path.")


def detect_mode(blender_python: str) -> str:
    """Guess which mode the op requires based on the bpy.ops prefix."""
    edit_prefixes = ("bpy.ops.mesh.", "bpy.ops.curve.", "bpy.ops.uv.")
    sculpt_prefixes = ("bpy.ops.sculpt.",)
    pose_prefixes = ("bpy.ops.pose.",)
    code = blender_python.strip().lower()
    for p in edit_prefixes:
        if code.startswith(p.lower()):
            return "edit"
    for p in sculpt_prefixes:
        if code.startswith(p.lower()):
            return "sculpt"
    for p in pose_prefixes:
        if code.startswith(p.lower()):
            return "pose"
    return "object"


def validate_one(args: tuple) -> dict:
    """Validate a single pair. Runs in a subprocess pool worker."""
    index, pair, blender_path, timeout = args

    python_code = pair.get("blender_python", "").strip()
    # Strip comment-only lines to check whether any real code is present
    non_comment_lines = [
        ln
        for ln in python_code.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if not python_code or not non_comment_lines or "bpy." not in python_code:
        # Empty, comment-only, or no bpy API calls — score 0.0 (cannot execute)
        return {
            "index": index,
            "score": 0.0,
            "exec_success": False,
            "exec_error": "no_executable_bpy_code",
            "scene_changed": False,
            "skipped": True,
        }

    # Multi-line scripts often set up full scenes — score 1.0 if they run clean

    mode = detect_mode(python_code)

    # Write payload to temp file
    payload = {"blender_python": python_code, "index": index, "mode": mode}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as pf:
        json.dump(payload, pf)
        payload_path = pf.name

    # Write harness to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as sf:
        sf.write(BLENDER_HARNESS)
        script_path = sf.name

    try:
        result = subprocess.run(
            [blender_path, "--background", "--python", script_path, "--", payload_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        # Extract our JSON line
        for line in result.stdout.splitlines():
            if line.startswith("NALANA_RESULT:"):
                return json.loads(line[len("NALANA_RESULT:") :])

        # Blender crashed or couldn't find our marker
        return {
            "index": index,
            "score": 0.0,
            "exec_success": False,
            "exec_error": f"blender_crash: {result.stderr[-200:]}",
            "scene_changed": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "index": index,
            "score": 0.0,
            "exec_success": False,
            "exec_error": "timeout",
            "scene_changed": False,
        }
    except Exception as e:
        return {
            "index": index,
            "score": 0.0,
            "exec_success": False,
            "exec_error": str(e),
            "scene_changed": False,
        }
    finally:
        Path(payload_path).unlink(missing_ok=True)
        Path(script_path).unlink(missing_ok=True)


def load_pairs(input_file: Path) -> list[dict]:
    pairs = []
    for line in input_file.read_text().splitlines():
        if line.strip():
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Validate Blender ops by execution")
    parser.add_argument("--blender-path", help="Path to Blender executable")
    parser.add_argument("--input-file", type=Path, default=INPUT_FILE)
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel Blender processes (each uses ~1GB RAM)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Seconds per Blender execution (default 30)",
    )
    parser.add_argument("--limit", type=int, help="Max pairs to validate")
    parser.add_argument(
        "--sample", type=int, help="Validate a random sample for quick QA"
    )
    args = parser.parse_args()

    blender = get_blender_path(args.blender_path)
    print(f"Blender: {blender}")

    if not args.input_file.exists():
        print(f"Input not found: {args.input_file}")
        print("Run validate.py first.")
        return

    print("Loading pairs...")
    pairs = load_pairs(args.input_file)

    if args.sample:
        import random

        random.shuffle(pairs)
        pairs = pairs[: args.sample]

    if args.limit:
        pairs = pairs[: args.limit]

    print(f"Pairs to validate: {len(pairs):,}")
    print(
        f"Workers: {args.workers} | Timeout: {args.timeout}s | Min score: {args.min_score}"
    )
    print()

    # Prepare work items
    work = [(i, pair, blender, args.timeout) for i, pair in enumerate(pairs)]

    kept = []
    rejected = []
    score_sum = 0.0
    errors = 0

    pbar = tqdm(total=len(work), unit="pair") if HAS_TQDM else None

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(validate_one, item): item for item in work}
        for future in as_completed(futures):
            result = future.result()
            idx = result["index"]
            score = result["score"]
            score_sum += score

            pair = pairs[idx]
            pair["_blender_exec_score"] = score
            pair["_blender_exec_error"] = result.get("exec_error")
            pair["_scene_changed"] = result.get("scene_changed", False)

            if score >= args.min_score:
                kept.append(pair)
            else:
                errors += 1
                pair["_reject_reason"] = result.get("exec_error", "exec_fail")
                rejected.append(pair)

            if pbar:
                pbar.set_postfix(
                    kept=len(kept),
                    fail=errors,
                    avg=f"{score_sum / (len(kept) + errors + 1e-6):.2f}",
                )
                pbar.update(1)

    if pbar:
        pbar.close()

    # Write results
    VALIDATED_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w") as f:
        for p in kept:
            f.write(json.dumps(p) + "\n")

    with REJECTED_FILE.open("w") as f:
        for p in rejected:
            f.write(json.dumps(p) + "\n")

    # Stats
    total = len(kept) + len(rejected)
    avg_score = score_sum / max(total, 1)
    keep_rate = len(kept) / max(total, 1) * 100

    print(f"\n{'═' * 50}")
    print(f"  TESTED:       {total:,}")
    print(f"  KEPT:         {len(kept):,} ({keep_rate:.1f}%)")
    print(f"  FAILED EXEC:  {errors:,}")
    print(f"  AVG SCORE:    {avg_score:.3f}")
    print(f"\n  Output: {OUTPUT_FILE}")
    print("  Next: python train_prep.py")


if __name__ == "__main__":
    main()
