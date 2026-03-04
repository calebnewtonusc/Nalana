"""
task_classifier.py - Labels task_type on every training pair.

Classification is rule-based (keyword matching + code pattern analysis).
Supports both single-turn pairs and multi-turn conversations.

Task types:
  EXECUTE       Single bpy.ops call, simple action
  BUILD         Multi-step, creates a complete object/scene
  MATERIALIZE   Material/texture/PBR setup
  SIMULATE      Physics simulation (rigid body, cloth, fluid, smoke)
  LIGHT         Lighting setup, HDRI, render settings
  UNDERSTAND    Explanation, Q&A, design reasoning
  CROSS_SOFTWARE Mentions multiple software, translation task
  CONVERSATION  Pure dialogue, no code

Each record gets:
  task_type    : str   — classified label
  task_confidence : float — 0.0-1.0

Usage:
    python task_classifier.py --input data/validated/dataset.jsonl --output data/validated/classified.jsonl
    python task_classifier.py --input data/multiturn/conversations.jsonl --output data/multiturn/classified.jsonl
    python task_classifier.py --stats
    python task_classifier.py --input data/validated/dataset.jsonl --output data/validated/classified.jsonl --stats
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import NamedTuple

# ─── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parents[1]
VALIDATED_DIR = BASE_DIR / "data" / "validated"
DEFAULT_INPUT = VALIDATED_DIR / "dataset.jsonl"
DEFAULT_OUTPUT = VALIDATED_DIR / "classified.jsonl"

# ─── Task type constants ───────────────────────────────────────────────────────

EXECUTE = "EXECUTE"
BUILD = "BUILD"
MATERIALIZE = "MATERIALIZE"
SIMULATE = "SIMULATE"
LIGHT = "LIGHT"
UNDERSTAND = "UNDERSTAND"
CROSS_SW = "CROSS_SOFTWARE"
CONVERSATION = "CONVERSATION"

ALL_TYPES = [
    EXECUTE,
    BUILD,
    MATERIALIZE,
    SIMULATE,
    LIGHT,
    UNDERSTAND,
    CROSS_SW,
    CONVERSATION,
]

# ─── Pattern dictionaries ──────────────────────────────────────────────────────

# Code patterns — matched against blender_python (case-insensitive)
CODE_PATTERNS: dict[str, list[str]] = {
    MATERIALIZE: [
        r"bpy\.data\.materials",
        r"Principled BSDF",
        r"node_tree",
        r"bpy\.ops\.material",
        r"material_slot",
        r"mat\.use_nodes",
        r"ShaderNode",
        r"inputs\[.Base Color.\]",
        r"inputs\[.Roughness.\]",
        r"inputs\[.Metallic.\]",
        r"inputs\[.Emission",
        r"inputs\[.IOR.\]",
        r"inputs\[.Transmission",
        r"inputs\[.Subsurface",
        r"bpy\.ops\.node\.",
    ],
    SIMULATE: [
        r"bpy\.ops\.rigidbody",
        r"rigid_body",
        r"modifier.*CLOTH",
        r"modifier.*FLUID",
        r"modifier.*SMOKE",
        r"modifier.*PARTICLE",
        r"particle_system",
        r"bpy\.ops\.particle",
        r"bpy\.ops\.fluid",
        r"bpy\.ops\.cloth",
        r"ptcache\.",
        r"bpy\.ops\.object\.forcefield_toggle",
        r"modifier.*DYNAMIC_PAINT",
        r"soft_body",
    ],
    LIGHT: [
        r"bpy\.ops\.object\.light_add",
        r"bpy\.context\.object\.data\.energy",
        r"world\.use_nodes",
        r"ShaderNodeTexEnvironment",
        r"ShaderNodeTexSky",
        r"bpy\.context\.scene\.render\.engine",
        r"bpy\.context\.scene\.cycles",
        r"bpy\.ops\.render\.",
        r"bpy\.data\.worlds",
        r"Background.*Strength",
        r"HDRI",
        r"THREE_POINT",
        r"sun_elevation",
        r"light_add.*SUN",
        r"light_add.*AREA",
        r"light_add.*POINT",
        r"light_add.*SPOT",
    ],
    CROSS_SW: [
        r"cmds\.",  # Maya
        r"pm\.mel",  # Maya
        r"c4d\.",  # Cinema 4D
        r"hou\.",  # Houdini
        r"rs\.",  # Rhino
        r"unreal\.",  # Unreal
        r"fusion\.",  # Fusion 360
        r"zb\.",  # ZBrush
    ],
}

# Voice command patterns — matched against voice_command (case-insensitive)
VOICE_PATTERNS: dict[str, list[str]] = {
    BUILD: [
        r"\b(create|build|make|model|construct|design|generate)\b",
        r"\b(scene|environment|interior|exterior|room|building)\b",
        r"\b(character|creature|vehicle|furniture|product)\b",
        r"\b(from scratch|step by step|workflow|pipeline)\b",
    ],
    MATERIALIZE: [
        r"\b(material|texture|PBR|shader|surface)\b",
        r"\b(roughness|metallic|emission|glossy|diffuse)\b",
        r"\b(IOR|fresnel|translucent|subsurface|SSS)\b",
        r"\b(color|albedo|normal map|bump map|displacement map)\b",
        r"\b(aged|weathered|worn|dirty|clean|polished|matte|glossy)\b",
        r"\b(apply material|add material|create material|new material)\b",
    ],
    SIMULATE: [
        r"\b(physics|simulation|sim|simulate)\b",
        r"\b(rigid body|cloth|fluid|smoke|fire|particles|hair)\b",
        r"\b(collide|collision|gravity|wind|force)\b",
        r"\b(bake|cache)\b",
        r"\b(falling|drop|bounce|deform|tear|flow)\b",
    ],
    LIGHT: [
        r"\b(light|lighting|illuminate|illumination|shadow)\b",
        r"\b(sun|area light|point light|spot light|hdri|environment)\b",
        r"\b(studio|three.?point|key light|fill light|rim light)\b",
        r"\b(render|rendering|cycles|eevee)\b",
        r"\b(exposure|brightness|warm|cool|color temperature)\b",
        r"\b(bloom|glow|emission|bake lighting)\b",
    ],
    UNDERSTAND: [
        r"\b(what|why|how|explain|describe|tell me|can you)\b",
        r"\b(difference between|what is|what are|what does)\b",
        r"\b(best way|recommend|advice|suggestion|should I)\b",
        r"\b(theory|concept|principle|fundamentals?)\b",
    ],
    CROSS_SW: [
        r"\b(maya|cinema 4d|c4d|houdini|rhino|rhinoceros|unreal|ue5|fusion 360|zbrush|marvelous)\b",
        r"\b(in maya|in houdini|in cinema|in unreal|convert to|equivalent in)\b",
        r"\b(translate|cross.software|workflow in|how to in)\b",
    ],
    CONVERSATION: [
        r"\b(hello|hi|hey|thanks|thank you|great|nice|cool|awesome)\b",
        r"\b(can you help|i need help|please help|what should)\b",
        r"\b(where do I start|i\'m new|beginner|first time)\b",
    ],
}

# Single-op indicators (code is one line or one bpy.ops call)
SINGLE_OP_PATTERN = re.compile(r"^bpy\.ops\.\w+\.\w+\([^)]*\)\s*$", re.MULTILINE)

# Multi-line build indicators
BUILD_CODE_PATTERNS = [
    r"\bfor\s+\w+\s+in\b",  # loops
    r"\bimport\s+bpy\b",  # script-style imports
    r"\bmat\s*=\s*bpy\.data",  # material creation mid-build
    r"\bmodifier_add\b",
    r"\bprimitive_\w+_add\b.*\nprimitive_\w+_add\b",  # multiple primitives
]

# Question indicators in voice
QUESTION_WORDS = re.compile(
    r"\b(what|why|how|when|where|which|is|are|can|should|does|do)\b", re.IGNORECASE
)


# ─── Scoring engine ────────────────────────────────────────────────────────────


class ClassificationResult(NamedTuple):
    task_type: str
    confidence: float
    scores: dict[str, float]


def score_text(text: str, patterns: list[str]) -> int:
    """Count how many patterns match in text. Case-insensitive, multiline-aware."""
    count = 0
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE | re.DOTALL | re.MULTILINE):
            count += 1
    return count


def classify_pair(pair: dict) -> ClassificationResult:
    """
    Classify a single training pair (single-turn).
    Returns (task_type, confidence, score_breakdown).
    """
    voice = pair.get("voice_command", "") or ""
    code = pair.get("blender_python", "") or ""
    reason = pair.get("reasoning", "") or ""
    op_name = (pair.get("blender_op") or {}).get("op", "") or ""

    # Combine relevant text fields for voice-side patterns
    voice_combined = f"{voice} {reason}".lower()
    code_combined = f"{code} {op_name}"

    scores: dict[str, float] = {t: 0.0 for t in ALL_TYPES}

    # ── Code analysis ────────────────────────────────────────────────────────
    has_code = bool(code.strip()) and not code.strip().startswith("#")
    lines = [
        l
        for l in code.strip().splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]
    is_single_op = (
        has_code and len(lines) == 1 and bool(SINGLE_OP_PATTERN.match(code.strip()))
    )
    is_multiline = has_code and len(lines) > 3

    # Code-based scoring
    for task_type, patterns in CODE_PATTERNS.items():
        hit = score_text(code_combined, patterns)
        if hit > 0:
            scores[task_type] += hit * 1.5

    # Build code patterns
    build_hits = score_text(code_combined, BUILD_CODE_PATTERNS)
    scores[BUILD] += build_hits * 1.2

    # Voice-based scoring
    for task_type, patterns in VOICE_PATTERNS.items():
        hit = score_text(voice_combined, patterns)
        if hit > 0:
            scores[task_type] += hit * 1.0

    # ── EXECUTE heuristics ───────────────────────────────────────────────────
    if is_single_op:
        scores[EXECUTE] += 3.0
    elif not is_multiline and has_code and len(lines) <= 2:
        scores[EXECUTE] += 1.5

    # ── UNDERSTAND heuristics ────────────────────────────────────────────────
    if not has_code or code.strip().startswith("#"):
        scores[UNDERSTAND] += 2.0
    if QUESTION_WORDS.search(voice) and not has_code:
        scores[UNDERSTAND] += 2.0

    # ── BUILD heuristics ─────────────────────────────────────────────────────
    if is_multiline:
        scores[BUILD] += 1.5
    if op_name == "MULTI_STEP_PLAN":
        scores[BUILD] += 4.0

    # ── CONVERSATION heuristics ──────────────────────────────────────────────
    if not has_code and not QUESTION_WORDS.search(voice):
        scores[CONVERSATION] += 1.5
    if len(voice.split()) < 4 and not has_code:
        scores[CONVERSATION] += 1.0

    # ── Cross-software boost ──────────────────────────────────────────────────
    sw_mentions = score_text(
        voice_combined, [r"\b(maya|cinema 4d|c4d|houdini|rhino|unreal|zbrush)\b"]
    )
    if sw_mentions > 0:
        scores[CROSS_SW] += sw_mentions * 2.5

    # ── Normalize and pick winner ─────────────────────────────────────────────
    total = sum(scores.values()) or 1.0
    normalized = {k: v / total for k, v in scores.items()}

    best_type = max(normalized, key=lambda k: normalized[k])
    best_score = normalized[best_type]

    # If all scores are very low, fallback heuristics
    if best_score < 0.2:
        if has_code:
            best_type = EXECUTE
            best_score = 0.5
        else:
            best_type = UNDERSTAND
            best_score = 0.4

    return ClassificationResult(
        task_type=best_type,
        confidence=round(best_score, 4),
        scores={k: round(v, 4) for k, v in normalized.items()},
    )


def classify_conversation(conv: dict) -> ClassificationResult:
    """
    Classify a multi-turn conversation record.
    Looks at the conversation-level task_type if present, and validates/refines it.
    Also scores individual assistant messages.
    """
    # If the conversation already has a task_type, trust it (generated from template)
    existing = conv.get("task_type")
    if existing and existing in ALL_TYPES:
        # Still compute scores but weight existing heavily
        messages = conv.get("messages", [])
        assistant_texts = [m["content"] for m in messages if m["role"] == "assistant"]
        combined_code = " ".join(assistant_texts)

        scores: dict[str, float] = {t: 0.0 for t in ALL_TYPES}
        scores[existing] = 5.0  # Strong prior from template

        for task_type, patterns in CODE_PATTERNS.items():
            scores[task_type] += score_text(combined_code, patterns) * 0.5

        total = sum(scores.values()) or 1.0
        normalized = {k: v / total for k, v in scores.items()}
        return ClassificationResult(
            task_type=existing,
            confidence=round(normalized[existing], 4),
            scores={k: round(v, 4) for k, v in normalized.items()},
        )

    # For chained pairs without existing task_type, analyze messages
    messages = conv.get("messages", [])
    if not messages:
        return ClassificationResult(CONVERSATION, 0.3, {t: 0.0 for t in ALL_TYPES})

    # Extract first user message as the intent
    first_user = next((m["content"] for m in messages if m["role"] == "user"), "")

    # Treat as a pseudo-pair for classification
    pseudo_pair = {
        "voice_command": first_user,
        "blender_python": " ".join(
            m["content"] for m in messages if m["role"] == "assistant"
        )[:2000],
        "reasoning": "",
        "blender_op": {},
    }
    return classify_pair(pseudo_pair)


# ─── Batch processing ──────────────────────────────────────────────────────────


def is_conversation_record(record: dict) -> bool:
    """Detect if this is a multi-turn conversation vs single-turn pair."""
    return "messages" in record or "conversation_id" in record


def classify_record(record: dict) -> dict:
    """Classify a record (pair or conversation) and annotate it in-place."""
    if is_conversation_record(record):
        result = classify_conversation(record)
    else:
        result = classify_pair(record)

    record["task_type"] = result.task_type
    record["task_confidence"] = result.confidence
    # Optionally keep score breakdown for debugging
    # record["_task_scores"] = result.scores
    return record


def load_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def print_stats(records: list[dict]) -> None:
    type_counts: Counter = Counter()
    conf_sum: dict[str, float] = {}
    conf_count: dict[str, int] = {}

    for r in records:
        t = r.get("task_type", "UNKNOWN")
        c = r.get("task_confidence", 0.0)
        type_counts[t] += 1
        conf_sum[t] = conf_sum.get(t, 0.0) + c
        conf_count[t] = conf_count.get(t, 0) + 1

    total = len(records)
    print(f"\n{'═' * 58}")
    print(f"  TASK TYPE DISTRIBUTION  (total: {total:,})")
    print(f"{'═' * 58}")
    print(f"  {'TYPE':<20} {'COUNT':>8}  {'%':>6}  {'AVG CONF':>10}")
    print(f"  {'-' * 20} {'-' * 8}  {'-' * 6}  {'-' * 10}")
    for t in ALL_TYPES:
        count = type_counts.get(t, 0)
        pct = count / max(total, 1) * 100
        avg_c = (conf_sum.get(t, 0) / conf_count.get(t, 1)) if conf_count.get(t) else 0
        print(f"  {t:<20} {count:>8,}  {pct:>5.1f}%  {avg_c:>9.3f}")

    # Low confidence warnings
    low_conf = [r for r in records if r.get("task_confidence", 1.0) < 0.4]
    if low_conf:
        print(f"\n  Warning: {len(low_conf)} records with confidence < 0.4 (ambiguous)")

    print(f"{'═' * 58}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify task_type for all training pairs and conversations"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input JSONL (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print distribution stats after classification",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Classify and show stats but do not write output",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Re-classify records that already have task_type",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input not found: {args.input}")
        print("Run validate.py first, then task_classifier.py.")
        return

    print(f"Loading records from {args.input}...")
    records = load_records(args.input)
    print(f"Loaded {len(records):,} records.")

    classified = 0
    skipped = 0

    try:
        from tqdm import tqdm as _tqdm

        bar = _tqdm(records, unit="record", desc="Classifying")
    except ImportError:
        bar = records

    for record in bar:
        # Skip if already has a high-confidence classification
        if (
            not args.overwrite_existing
            and "task_type" in record
            and record.get("task_confidence", 0.0) >= 0.5
        ):
            skipped += 1
            continue

        classify_record(record)
        classified += 1

    print(f"\nClassified: {classified:,} | Skipped (already tagged): {skipped:,}")

    if args.stats or args.dry_run:
        print_stats(records)

    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"Output written: {args.output}")
        print("Next: python train_prep.py && python train.py")


if __name__ == "__main__":
    main()
