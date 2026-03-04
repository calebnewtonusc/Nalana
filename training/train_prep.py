import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
train_prep.py - Convert raw JSONL pairs into fine-tuning ready datasets.

Loads from ALL 5 training data streams:
  Stream 1: data/processed/         — YouTube tutorial pairs (35% target mix)
  Stream 2: data/spatial/           — Objaverse form analysis pairs (25% target mix)
  Stream 3: data/physics/           — Physics KB pairs (15% target mix)
  Stream 4: data/multiturn/         — Multi-turn conversation sequences (20% target mix)
  Stream 5: data/integrations/      — Spline, Matterport, cross-software pairs (5% target mix)

Quality tier weighting:
  quality >= 4.0  →  5x weight (repeat 5 times in training set)
  quality >= 2.0  →  2x weight (repeat 2 times)
  else            →  1x weight (include once)

Curriculum ordering:
  EXECUTE tasks first (simplest, most concrete), MULTI_STEP last.
  Within each tier: sort by difficulty ascending so the model sees easy examples first.

Outputs two formats:
  1. ShareGPT format (for TRL SFTTrainer / axolotl) — data/train/sharegpt.jsonl
  2. Alpaca format (for simpler trainers)            — data/train/alpaca.jsonl

Also generates a train/val split (95/5) and prints coverage stats.

Usage:
    python train_prep.py
    python train_prep.py --val-ratio 0.1 --min-quality 0.7
    python train_prep.py --stats-only
    python train_prep.py --no-curriculum   # disable curriculum sorting
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# ─── Directory layout ─────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SPATIAL_DIR   = BASE_DIR / "data" / "spatial"
PHYSICS_DIR   = BASE_DIR / "data" / "physics"
MULTITURN_DIR = BASE_DIR / "data" / "multiturn"
INTEG_DIR     = BASE_DIR / "data" / "integrations"
TRAIN_DIR     = BASE_DIR / "data" / "train"

# Target mix fractions (must sum to ~1.0)
TARGET_MIX = {
    "tutorials":    0.35,   # Stream 1
    "spatial":      0.25,   # Stream 2
    "multiturn":    0.20,   # Stream 4
    "physics":      0.15,   # Stream 3
    "integrations": 0.05,   # Stream 5
}

# Curriculum task order — EXECUTE is simplest, MULTI_STEP is most complex
CURRICULUM_ORDER = {
    "EXECUTE":     0,
    "MATERIALIZE": 1,
    "LIGHT":       2,
    "SIMULATE":    3,
    "BUILD":       4,
    "UNDERSTAND":  5,
    "MULTI_STEP":  6,
}

SYSTEM_MSG = (
    "You are Nalana, a voice-controlled 3D AI that works across Blender, Maya, Cinema 4D, "
    "Houdini, Unreal Engine, and more. Given a voice command and the current scene context, "
    "output the exact 3D operation as a JSON object and the corresponding executable code. "
    "Be precise with operation names, arguments, and physical reasoning."
)


# ─── Stream loaders ───────────────────────────────────────────────────────────

def load_stream1_tutorials() -> list[dict]:
    """Stream 1: YouTube tutorial pairs from data/processed/"""
    pairs = []
    for jsonl_path in sorted(PROCESSED_DIR.glob("*.jsonl")):
        for line in jsonl_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                p = json.loads(line)
                p.setdefault("_stream", "tutorials")
                pairs.append(p)
            except json.JSONDecodeError:
                continue
    return pairs


def load_stream2_spatial() -> list[dict]:
    """Stream 2: Objaverse form analysis pairs from data/spatial/ and data/objaverse/annotations/"""
    pairs = []
    # Primary: data/spatial/
    for jsonl_path in sorted(SPATIAL_DIR.glob("**/*.jsonl")):
        for line in jsonl_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                p = json.loads(line)
                p.setdefault("_stream", "spatial")
                pairs.append(p)
            except json.JSONDecodeError:
                continue
    # Fallback: merged objaverse annotation file
    objaverse_merged = BASE_DIR / "data" / "processed" / "dataset_3d.jsonl"
    if objaverse_merged.exists():
        for line in objaverse_merged.read_text().splitlines():
            if not line.strip():
                continue
            try:
                p = json.loads(line)
                p.setdefault("_stream", "spatial")
                pairs.append(p)
            except json.JSONDecodeError:
                continue
    return pairs


def load_stream3_physics() -> list[dict]:
    """Stream 3: Physics KB pairs from data/physics/physics_pairs.jsonl"""
    pairs = []
    physics_file = PHYSICS_DIR / "physics_pairs.jsonl"
    if not physics_file.exists():
        # Also check alternate locations
        physics_file = None
        for alt in sorted(PHYSICS_DIR.glob("**/*.jsonl")):
            physics_file = alt
            break
    if physics_file is not None and physics_file.exists():
        for line in physics_file.read_text().splitlines():
            if not line.strip():
                continue
            try:
                p = json.loads(line)
                p.setdefault("_stream", "physics")
                pairs.append(p)
            except json.JSONDecodeError:
                continue
    return pairs


def load_stream4_multiturn() -> list[dict]:
    """Stream 4: Multi-turn conversation sequences from data/multiturn/conversations.jsonl"""
    pairs = []
    mt_file = MULTITURN_DIR / "conversations.jsonl"
    if not mt_file.exists():
        for alt in sorted(MULTITURN_DIR.glob("**/*.jsonl")):
            mt_file = alt
            break
    if mt_file and mt_file.exists():
        for line in mt_file.read_text().splitlines():
            if not line.strip():
                continue
            try:
                p = json.loads(line)
                p.setdefault("_stream", "multiturn")
                pairs.append(p)
            except json.JSONDecodeError:
                continue
    return pairs


def load_stream5_integrations() -> list[dict]:
    """Stream 5: Cross-software integration pairs from data/integrations/**/*.jsonl"""
    pairs = []
    for jsonl_path in sorted(INTEG_DIR.glob("**/*.jsonl")):
        for line in jsonl_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                p = json.loads(line)
                p.setdefault("_stream", "integrations")
                pairs.append(p)
            except json.JSONDecodeError:
                continue
    return pairs


def load_all_streams() -> dict[str, list[dict]]:
    """Load all 5 streams, return dict keyed by stream name."""
    print("Loading Stream 1: YouTube tutorial pairs...")
    s1 = load_stream1_tutorials()
    print(f"  {len(s1):,} pairs")

    print("Loading Stream 2: Objaverse spatial pairs...")
    s2 = load_stream2_spatial()
    print(f"  {len(s2):,} pairs")

    print("Loading Stream 3: Physics KB pairs...")
    s3 = load_stream3_physics()
    print(f"  {len(s3):,} pairs")

    print("Loading Stream 4: Multi-turn sequences...")
    s4 = load_stream4_multiturn()
    print(f"  {len(s4):,} pairs")

    print("Loading Stream 5: Integration pairs...")
    s5 = load_stream5_integrations()
    print(f"  {len(s5):,} pairs")

    return {
        "tutorials":    s1,
        "spatial":      s2,
        "physics":      s3,
        "multiturn":    s4,
        "integrations": s5,
    }


# ─── Quality filtering ────────────────────────────────────────────────────────

def is_valid_pair(p: dict) -> bool:
    """Minimum validity check — must have inputs and outputs."""
    # Multi-turn conversations have a different structure
    if "conversations" in p:
        return len(p["conversations"]) >= 2
    # Standard pairs need voice_command + some output
    if not p.get("voice_command", "").strip():
        return False
    if not p.get("blender_python", "").strip() and not p.get("implementations"):
        return False
    return True


def get_quality_weight(p: dict) -> int:
    """Return repeat count based on quality score."""
    quality = p.get("quality", p.get("quality_score", 1.0))
    if isinstance(quality, (int, float)):
        if quality >= 4.0:
            return 5
        if quality >= 2.0:
            return 2
    return 1


def apply_quality_weights(pairs: list[dict]) -> list[dict]:
    """Expand dataset by repeating high-quality pairs."""
    expanded = []
    for p in pairs:
        weight = get_quality_weight(p)
        expanded.extend([p] * weight)
    return expanded


# ─── Curriculum ordering ──────────────────────────────────────────────────────

def get_curriculum_key(p: dict) -> tuple:
    """Sort key: (task_type_order, difficulty)"""
    task_type = p.get("task_type", p.get("blender_op", {}).get("op", "EXECUTE"))
    # Normalize to uppercase key
    tt_upper = str(task_type).upper()
    # Find matching curriculum bucket
    order = CURRICULUM_ORDER.get(tt_upper, CURRICULUM_ORDER.get("EXECUTE", 0))
    for key in CURRICULUM_ORDER:
        if key in tt_upper:
            order = CURRICULUM_ORDER[key]
            break
    difficulty = p.get("difficulty", p.get("modeling_complexity", "medium"))
    diff_map = {"low": 0, "easy": 0, "medium": 1, "high": 2, "expert": 3}
    diff_score = diff_map.get(str(difficulty).lower(), 1)
    return (order, diff_score)


def apply_curriculum_sort(pairs: list[dict]) -> list[dict]:
    """Sort pairs by curriculum order: EXECUTE first, MULTI_STEP last."""
    return sorted(pairs, key=get_curriculum_key)


# ─── Mix management ───────────────────────────────────────────────────────────

def balance_streams(streams: dict[str, list[dict]], total_target: int | None = None) -> list[dict]:
    """
    Combine streams according to TARGET_MIX ratios.
    If total_target is None, uses the sum of all available pairs.
    """
    # Count available pairs per stream
    available = {k: len(v) for k, v in streams.items()}
    total_available = sum(available.values())

    if total_target is None:
        total_target = total_available

    print(f"\nStream availability: {available}")
    print(f"Target total: {total_target:,}")

    combined = []
    for stream_name, target_frac in TARGET_MIX.items():
        stream_pairs = streams.get(stream_name, [])
        if not stream_pairs:
            print(f"  [WARN] Stream '{stream_name}' is empty — skipping")
            continue

        # How many pairs do we want from this stream?
        target_n = int(total_target * target_frac)
        # If we don't have enough, use all available (with upsampling if needed)
        if len(stream_pairs) < target_n:
            # Upsample by repeating
            repeats = (target_n // len(stream_pairs)) + 1
            pool = (stream_pairs * repeats)[:target_n]
            print(f"  {stream_name}: {len(stream_pairs):,} available → upsampled to {target_n:,} (×{repeats})")
        else:
            pool = random.sample(stream_pairs, target_n)
            print(f"  {stream_name}: {len(stream_pairs):,} available → sampled {target_n:,} ({target_frac*100:.0f}%)")

        combined.extend(pool)

    return combined


# ─── Format conversion ────────────────────────────────────────────────────────

def make_user_message(pair: dict) -> str:
    """Build user turn from any stream's pair format."""
    if "voice_command" in pair:
        scene = pair.get("scene_context", pair.get("scene_state", "Default 3D scene"))
        return f"Voice command: {pair['voice_command']}\nScene context: {scene}"
    if "input" in pair:
        return str(pair["input"])
    if "user" in pair:
        return str(pair["user"])
    return json.dumps({k: v for k, v in pair.items() if k not in ("_stream",)})[:500]


def make_assistant_message(pair: dict) -> str:
    """Build assistant turn from any stream's pair format."""
    parts = []

    # Blender operation JSON
    blender_op = pair.get("blender_op") or pair.get("op")
    if blender_op and isinstance(blender_op, dict):
        parts.append(f"```json\n{json.dumps(blender_op, indent=2)}\n```")

    # Universal DSL (for cross-software pairs)
    universal_dsl = pair.get("universal_dsl") or pair.get("dsl_op")
    if universal_dsl and isinstance(universal_dsl, dict):
        parts.append(f"```json\n{json.dumps(universal_dsl, indent=2)}\n```")

    # Executable code — prefer blender_python, fall back to implementations
    code = pair.get("blender_python") or pair.get("code")
    if not code and "implementations" in pair:
        impl = pair["implementations"]
        code = impl.get("blender") or next(iter(impl.values()), None)
    if code:
        parts.append(f"```python\n{code}\n```")

    # Cross-software implementations dict
    if "implementations" in pair and isinstance(pair["implementations"], dict):
        impl_lines = "\n".join(
            f"  {sw}: {c}" for sw, c in pair["implementations"].items() if sw != "blender"
        )
        if impl_lines:
            parts.append(f"Cross-software equivalents:\n{impl_lines}")

    # Physics analysis / reasoning
    for field in ("physics_analysis", "reasoning", "explanation", "design_rationale"):
        val = pair.get(field, "")
        if val:
            parts.append(str(val))
            break

    if not parts:
        # Fallback: serialize the whole output section
        output = pair.get("output") or pair.get("response") or pair.get("assistant", "")
        parts.append(str(output) if output else json.dumps(pair)[:200])

    return "\n\n".join(parts)


def to_sharegpt_standard(pair: dict) -> dict:
    """Convert a standard (non-multi-turn) pair to ShareGPT format."""
    return {
        "conversations": [
            {"from": "system",    "value": SYSTEM_MSG},
            {"from": "human",     "value": make_user_message(pair)},
            {"from": "assistant", "value": make_assistant_message(pair)},
        ]
    }


def to_sharegpt(pair: dict) -> dict:
    """Convert any stream pair to ShareGPT format."""
    # Multi-turn conversations already have the right structure
    if "conversations" in pair:
        convs = pair["conversations"]
        # Ensure system message is present
        if convs and convs[0].get("from") != "system":
            convs = [{"from": "system", "value": SYSTEM_MSG}] + convs
        return {"conversations": convs}
    return to_sharegpt_standard(pair)


def to_alpaca(pair: dict) -> dict:
    """Convert any stream pair to Alpaca format."""
    if "conversations" in pair:
        # Extract first human/assistant exchange
        convs = pair["conversations"]
        human = next((c["value"] for c in convs if c["from"] == "human"), "")
        asst  = next((c["value"] for c in convs if c["from"] == "assistant"), "")
        return {"instruction": SYSTEM_MSG, "input": human, "output": asst}
    return {
        "instruction": SYSTEM_MSG,
        "input": make_user_message(pair),
        "output": make_assistant_message(pair),
    }


def write_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  Written: {path} ({len(records):,} records)")


# ─── Stats ────────────────────────────────────────────────────────────────────

def print_stats(streams: dict[str, list[dict]], combined: list[dict]):
    print(f"\n{'═'*60}")
    print(f"  DATASET STATISTICS")
    print(f"{'═'*60}")

    total = sum(len(v) for v in streams.values())
    print(f"\n  Raw stream sizes:")
    for name, pairs in streams.items():
        frac = len(pairs) / max(total, 1) * 100
        print(f"    {name:<15} {len(pairs):>8,}  ({frac:.1f}%)")
    print(f"    {'TOTAL':<15} {total:>8,}")

    print(f"\n  Combined training set: {len(combined):,} pairs")

    # Op coverage (for standard pairs)
    ops = Counter()
    task_types = Counter()
    for p in combined:
        op = p.get("blender_op", {})
        if isinstance(op, dict):
            ops[op.get("op", "unknown")] += 1
        tt = p.get("task_type", "EXECUTE")
        task_types[str(tt).upper()] += 1

    if ops:
        print(f"\n  Top 15 Blender ops:")
        for op, count in ops.most_common(15):
            bar = "█" * min(count // max(len(combined) // 200, 1), 30)
            print(f"    {op:<45} {count:>5}  {bar}")

    if task_types:
        print(f"\n  Task type distribution:")
        for tt, count in task_types.most_common():
            frac = count / len(combined) * 100
            print(f"    {tt:<20} {count:>6,}  ({frac:.1f}%)")

    # Token estimate (rough: 1 token ≈ 4 chars)
    total_chars = 0
    for p in combined[:min(1000, len(combined))]:
        try:
            total_chars += len(make_user_message(p)) + len(make_assistant_message(p))
        except Exception:
            pass
    if len(combined) > 0:
        avg_chars = total_chars / min(1000, len(combined))
        est_tokens = int(avg_chars * len(combined) / 4)
        print(f"\n  Est. total tokens: ~{est_tokens:,}")
        print(f"  Avg tokens/pair:   ~{int(avg_chars / 4)}")

    print(f"{'═'*60}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare training dataset from all 5 data streams")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum ordering")
    parser.add_argument("--no-weights", action="store_true", help="Disable quality tier weighting")
    parser.add_argument("--total-pairs", type=int, default=None,
                        help="Target total pairs (default: use all available)")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load all streams
    streams = load_all_streams()
    total_raw = sum(len(v) for v in streams.values())

    if total_raw == 0:
        print("No training data found across any stream. Run the data collection scripts first.")
        print("  Stream 1: python synthesize_bulk.py")
        print("  Stream 2: python annotate_forms.py")
        print("  Stream 3: python integrations/collect_design_physics.py")
        print("  Stream 4: python multi_turn.py")
        print("  Stream 5: python integrations/collect_design_physics.py --integrations")
        return

    # Filter invalid pairs per stream
    filtered_streams = {}
    for name, pairs in streams.items():
        valid = [p for p in pairs if is_valid_pair(p)]
        if len(valid) < len(pairs):
            print(f"  [filter] {name}: {len(pairs):,} → {len(valid):,} valid pairs")
        filtered_streams[name] = valid

    # Apply quality weights before mixing (so high-quality pairs are upsampled within each stream)
    if not args.no_weights:
        weighted_streams = {}
        for name, pairs in filtered_streams.items():
            weighted = apply_quality_weights(pairs)
            if len(weighted) != len(pairs):
                print(f"  [weights] {name}: {len(pairs):,} → {len(weighted):,} after quality weighting")
            weighted_streams[name] = weighted
    else:
        weighted_streams = filtered_streams

    # Balance across streams according to target mix
    combined = balance_streams(weighted_streams, total_target=args.total_pairs)

    # Apply curriculum ordering (sort by task difficulty)
    if not args.no_curriculum:
        print("\nApplying curriculum ordering (EXECUTE → MULTI_STEP)...")
        combined = apply_curriculum_sort(combined)

    print_stats(streams, combined)

    if args.stats_only:
        return

    # Train/val split (preserve curriculum order — val is the last val_ratio fraction)
    val_n = max(1, int(len(combined) * args.val_ratio))
    # For curriculum: take val from a random sample of the full set, not just the end
    indices = list(range(len(combined)))
    random.shuffle(indices)
    val_indices  = set(indices[:val_n])
    train_pairs  = [combined[i] for i in range(len(combined)) if i not in val_indices]
    val_pairs    = [combined[i] for i in range(len(combined)) if i in val_indices]

    # Re-apply curriculum sort to training set (val can be random)
    if not args.no_curriculum:
        train_pairs = apply_curriculum_sort(train_pairs)

    print(f"Train: {len(train_pairs):,} pairs")
    print(f"Val:   {len(val_pairs):,} pairs\n")

    # Write ShareGPT format
    write_jsonl(TRAIN_DIR / "sharegpt_train.jsonl", [to_sharegpt(p) for p in train_pairs])
    write_jsonl(TRAIN_DIR / "sharegpt_val.jsonl",   [to_sharegpt(p) for p in val_pairs])

    # Write Alpaca format
    write_jsonl(TRAIN_DIR / "alpaca_train.jsonl", [to_alpaca(p) for p in train_pairs])
    write_jsonl(TRAIN_DIR / "alpaca_val.jsonl",   [to_alpaca(p) for p in val_pairs])

    print(f"\nAll files written to {TRAIN_DIR}/")
    print("  sharegpt_train.jsonl  <- use with TRL SFTTrainer / axolotl")
    print("  alpaca_train.jsonl    <- use with simpler trainers")
    print(f"\nNext step: deepspeed --num_gpus=18 train.py --deepspeed ds_config.json")


if __name__ == "__main__":
    main()
