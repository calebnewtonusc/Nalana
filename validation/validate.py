"""
validate.py - Quality gate for all training pairs before fine-tuning.

Runs every pair through:
  1. Schema check     — required fields present
  2. Op validation    — bpy.ops name is real (or maps to Universal DSL)
  3. Voice quality    — command sounds natural (heuristics)
  4. Deduplication    — exact + near-duplicate removal
  5. Length filter    — voice command not too short/long
  6. Source weighting — balance across videos / software

Outputs:
  data/validated/dataset.jsonl     — clean, deduplicated pairs
  data/validated/rejected.jsonl    — rejected pairs with reason
  data/validated/stats.json        — quality report

Usage:
    python validate.py
    python validate.py --min-score 0.8 --no-dedup
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

PROCESSED_DIR = Path(__file__).parents[1] / "data" / "processed"
VALIDATED_DIR = Path(__file__).parents[1] / "data" / "validated"
MASTER_JSONL = PROCESSED_DIR / "dataset.jsonl"

# ─── Known valid bpy.ops prefixes + common ops ────────────────────────────────

VALID_OP_PREFIXES = {
    "mesh",
    "object",
    "transform",
    "view3d",
    "sculpt",
    "paint",
    "armature",
    "pose",
    "curve",
    "surface",
    "metaball",
    "font",
    "lattice",
    "empty",
    "gpencil",
    "node",
    "material",
    "texture",
    "uv",
    "render",
    "scene",
    "world",
    "action",
    "nla",
    "sequencer",
    "clip",
    "image",
    "graph",
    "anim",
    "rigidbody",
    "fluid",
    "cloth",
    "particle",
    "constraint",
    "physics",
    "geometry",
    "preferences",
    "screen",
    "workspace",
    "outliner",
    "wm",
    "file",
    "export_scene",
    "import_scene",
    "import_mesh",
    "export_mesh",
    "cycles",
}

KNOWN_BAD_OPS = {
    "unknown",
    "todo",
    "placeholder",
    "none",
    "null",
    "undefined",
}

# Menu-path patterns that indicate a bad voice_command
MENU_PATH_PATTERNS = [
    r"add\s*>\s*mesh",
    r"object\s*>\s*",
    r"ctrl\+",
    r"shift\+",
    r"alt\+",
    r"press\s+[a-z]\b",
    r"click\s+the\s+\w+\s+(button|icon|menu|panel)",
    r"go\s+to\s+(the\s+)?(top|edit|object|modifier)",
]


def validate_pair(pair: dict) -> tuple[float, list[str]]:
    """
    Returns (score 0-1, list of issues).
    Score >= 0.6 → keep.
    """
    score = 1.0
    issues = []

    # ── Schema ───────────────────────────────────────────────────────────────
    for field in ("voice_command", "scene_context", "blender_python"):
        if not pair.get(field, "").strip():
            score -= 0.3
            issues.append(f"missing_{field}")

    blender_op = pair.get("blender_op", {})
    op_name = blender_op.get("op", "")

    # ── Op validation ─────────────────────────────────────────────────────────
    if not op_name:
        score -= 0.2
        issues.append("missing_op_name")
    elif op_name.lower() in KNOWN_BAD_OPS:
        score -= 0.4
        issues.append("invalid_op_name")
    elif "." in op_name:
        prefix = op_name.split(".")[0].lower()
        if prefix not in VALID_OP_PREFIXES:
            score -= 0.15
            issues.append(f"unknown_op_prefix:{prefix}")
    elif op_name not in ("MULTI_STEP_PLAN", "UNKNOWN"):
        # Universal DSL ops are OK
        if not any(
            op_name.startswith(p.upper() + "_") or op_name == p.upper()
            for p in [
                "ADD",
                "EXTRUDE",
                "BEVEL",
                "SUBDIVIDE",
                "INSET",
                "LOOP",
                "KNIFE",
                "BRIDGE",
                "FILL",
                "MERGE",
                "DELETE",
                "SCALE",
                "ROTATE",
                "TRANSLATE",
                "SHADE",
                "ENTER",
                "APPLY",
                "DUPLICATE",
                "JOIN",
                "UNWRAP",
                "RENDER",
                "VOXEL",
            ]
        ):
            score -= 0.1
            issues.append(f"unrecognized_op:{op_name}")

    # ── Blender Python ────────────────────────────────────────────────────────
    bp = pair.get("blender_python", "")
    if bp and not (bp.startswith("bpy.") or bp.startswith("#") or "\n" in bp):
        if not any(
            kw in bp for kw in ["bpy.", "import ", "cmds.", "c4d.", "hou.", "rs."]
        ):
            score -= 0.1
            issues.append("python_not_api_call")

    # ── Voice command quality ─────────────────────────────────────────────────
    vc = pair.get("voice_command", "").lower().strip()
    words = vc.split()

    if len(words) < 2:
        score -= 0.3
        issues.append("voice_too_short")
    elif len(words) > 20:
        score -= 0.1
        issues.append("voice_too_long")

    for pattern in MENU_PATH_PATTERNS:
        if re.search(pattern, vc, re.IGNORECASE):
            score -= 0.3
            issues.append("voice_is_menu_path")
            break

    # Bad voice commands that are just op names
    if vc in (
        "extrude",
        "bevel",
        "subdivide",
        "translate",
        "rotate",
        "scale",
        "duplicate",
        "delete",
        "join",
        "merge",
        "fill",
    ):
        score -= 0.15
        issues.append("voice_is_just_op_name")

    return max(0.0, score), issues


def make_dedup_key(pair: dict) -> str:
    """Create a hash key for deduplication."""
    vc = pair.get("voice_command", "").lower().strip()
    op = pair.get("blender_op", {}).get("op", "")
    # Normalize: remove articles, extra spaces
    vc_norm = re.sub(r"\b(a|an|the|this|that|some)\b", "", vc).strip()
    vc_norm = re.sub(r"\s+", " ", vc_norm)
    return hashlib.md5(f"{vc_norm}|{op}".encode()).hexdigest()


def load_all_pairs() -> list[dict]:
    """Load from dataset.jsonl + dataset_3d.jsonl (Stream 2)."""
    pairs = []
    for fname in ["dataset.jsonl", "dataset_3d.jsonl"]:
        path = PROCESSED_DIR / fname
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    try:
                        pairs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return pairs


def balance_sources(pairs: list[dict], max_per_video: int = 50) -> list[dict]:
    """Cap contribution per source video to prevent over-representation."""
    counts: Counter = Counter()
    balanced = []
    for pair in pairs:
        vid = pair.get("video_id") or pair.get("uid") or "unknown"
        if counts[vid] < max_per_video:
            balanced.append(pair)
            counts[vid] += 1
    return balanced


def main():
    parser = argparse.ArgumentParser(description="Validate and clean training pairs")
    parser.add_argument("--min-score", type=float, default=0.6)
    parser.add_argument("--no-dedup", action="store_true")
    parser.add_argument("--max-per-source", type=int, default=50)
    args = parser.parse_args()

    VALIDATED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading pairs...")
    all_pairs = load_all_pairs()
    print(f"Total loaded: {len(all_pairs):,}")

    kept = []
    rejected = []
    seen_keys = set()
    score_dist = Counter()
    issue_dist = Counter()

    for pair in all_pairs:
        score, issues = validate_pair(pair)
        score_dist[round(score, 1)] += 1
        for issue in issues:
            issue_dist[issue] += 1

        if score < args.min_score:
            pair["_reject_reason"] = issues
            pair["_score"] = score
            rejected.append(pair)
            continue

        if not args.no_dedup:
            key = make_dedup_key(pair)
            if key in seen_keys:
                pair["_reject_reason"] = ["duplicate"]
                rejected.append(pair)
                continue
            seen_keys.add(key)

        pair["_quality_score"] = score
        kept.append(pair)

    # Balance sources
    kept = balance_sources(kept, args.max_per_source)

    # Sort by quality score descending
    kept.sort(key=lambda p: p.get("_quality_score", 0), reverse=True)

    # Write outputs
    out_path = VALIDATED_DIR / "dataset.jsonl"
    with out_path.open("w") as f:
        for p in kept:
            f.write(json.dumps(p) + "\n")

    rej_path = VALIDATED_DIR / "rejected.jsonl"
    with rej_path.open("w") as f:
        for p in rejected:
            f.write(json.dumps(p) + "\n")

    # Stats report
    stats = {
        "total_input": len(all_pairs),
        "kept": len(kept),
        "rejected": len(rejected),
        "keep_rate": f"{len(kept) / max(len(all_pairs), 1) * 100:.1f}%",
        "score_distribution": {str(k): v for k, v in sorted(score_dist.items())},
        "top_issues": dict(issue_dist.most_common(15)),
        "unique_ops": len(set(p.get("blender_op", {}).get("op", "") for p in kept)),
        "unique_sources": len(
            set(p.get("video_id") or p.get("uid") or "" for p in kept)
        ),
    }
    (VALIDATED_DIR / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\n{'═' * 50}")
    print(f"  INPUT:    {len(all_pairs):,} pairs")
    print(f"  KEPT:     {len(kept):,} ({stats['keep_rate']})")
    print(f"  REJECTED: {len(rejected):,}")
    print(f"  UNIQUE OPS: {stats['unique_ops']}")
    print("\n  Top issues:")
    for issue, count in issue_dist.most_common(8):
        print(f"    {issue:<35} {count:>6}")
    print(f"\n  Output: {out_path}")
    print("  Next: python validate_blender.py && python train_prep.py")


if __name__ == "__main__":
    main()
