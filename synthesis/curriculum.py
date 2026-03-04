"""
curriculum.py - Smart curriculum ordering for 3D AI training data.

Scores training pairs by complexity and groups them into progressive difficulty tiers.
Ensures proportional software coverage and proper concept ordering.

Complexity levels:
    1 = Primitive operations (add cube, move object, set material color)
    2 = Modifier stacks (array, mirror, bevel combinations)
    3 = Node graphs (shader nodes, geometry nodes)
    4 = Python scripts (bpy automation, add-ons)
    5 = Expert workflows (multi-step pipelines, agent-level tasks)

Software coverage targets:
    Blender: 50%
    Houdini: 10%
    Maya:    10%
    Cinema4D: 8%
    ZBrush:  7%
    Substance: 7%
    Other:   8%

Usage:
    python synthesis/curriculum.py --input data/
    python synthesis/curriculum.py --input data/ --output data/curriculum/
    python synthesis/curriculum.py --stats  # show distribution stats
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parents[1] / "data"
CURRICULUM_DIR = DATA_DIR / "curriculum"

# ─── Software detection patterns ─────────────────────────────────────────────
SOFTWARE_PATTERNS = {
    "blender": re.compile(
        r"\b(blender|bpy\.|geometry\s+nodes?|shader\s+nodes?|cycles|eevee|"
        r"grease\s+pencil|rigify|armature|sculpt\s+mode)\b",
        re.I,
    ),
    "houdini": re.compile(
        r"\b(houdini|sidefx|vex\s+code|vops|sops|dops|lops|solaris|"
        r"hqueue|mantra|karma)\b",
        re.I,
    ),
    "maya": re.compile(
        r"\b(maya|autodesk\s+maya|maya\s+python|pymel|cmds\.|mel\s+script|"
        r"bifrost|arnold\s+for\s+maya|mash|ncloth)\b",
        re.I,
    ),
    "cinema4d": re.compile(
        r"\b(cinema\s*4d|c4d|maxon|mograph|xpresso|redshift|"
        r"field\s+forces|volume\s+builder)\b",
        re.I,
    ),
    "zbrush": re.compile(
        r"\b(zbrush|pixologic|dynamesh|sculptris|fibermesh|"
        r"polypaint|zremesher|zsphere|zplugin)\b",
        re.I,
    ),
    "substance": re.compile(
        r"\b(substance\s+(painter|designer|3d)|adobe\s+substance|"
        r"smart\s+materials?|substance\s+graph|sbsar)\b",
        re.I,
    ),
    "unreal": re.compile(
        r"\b(unreal\s+engine|ue[45]|nanite|lumen|niagara|blueprints?|"
        r"fab\.com|fab\s+marketplace)\b",
        re.I,
    ),
}

# ─── Complexity scoring patterns ──────────────────────────────────────────────
COMPLEXITY_INDICATORS = {
    1: [  # Primitive operations
        r"\b(add\s+(cube|sphere|cylinder|plane)|move\s+object|rotate|scale|"
        r"duplicate|delete|hide|unhide|select\s+all|deselect|apply\s+transform|"
        r"viewport\s+shading|object\s+mode|edit\s+mode|set\s+(material|color))\b",
    ],
    2: [  # Modifier stacks and basic workflows
        r"\b(modifier|array|mirror|bevel|subdivision|solidify|"
        r"boolean|shrinkwrap|displace|wave|decimate|remesh)\b",
        r"\b(UV\s+unwrap|texture\s+(bake|paint)|weight\s+paint|shape\s+key)\b",
    ],
    3: [  # Node graphs and procedural
        r"\b(node\s+(group|graph|tree)|shader\s+node|geometry\s+node|"
        r"procedural|instance\s+on\s+points|attribute|field|"
        r"noise\s+texture|voronoi|wave\s+texture)\b",
        r"\b(xpresso|vops|material\s+graph|substance\s+graph|"
        r"grasshopper|kangaroo|galapagos)\b",
    ],
    4: [  # Scripting and automation
        r"\b(python\s+script|bpy\.|scripting|add-?on|plugin|automation|"
        r"batch\s+(render|process)|command\s+line|headless)\b",
        r"\b(mel\s+script|pymel|cmds\.|vex\s+code|hscript|"
        r"maxscript|c#\s+plugin)\b",
    ],
    5: [  # Expert multi-step pipelines and agents
        r"\b(pipeline|workflow|multi.?step|production\s+pipeline|"
        r"asset\s+management|render\s+farm|LOD\s+generation|"
        r"procedural\s+city|destruction\s+sim|fluid\s+sim|crowd\s+sim)\b",
        r"\b(photogrammetry|neural\s+radiance|nerf|gaussian\s+splat|"
        r"AI.?generated|generative\s+3D|diffusion\s+model)\b",
    ],
}

# Compiled complexity patterns
COMPILED_COMPLEXITY = {
    level: [re.compile(pat, re.I) for pat in pats]
    for level, pats in COMPLEXITY_INDICATORS.items()
}

# ─── Concept category patterns ────────────────────────────────────────────────
CONCEPT_CATEGORIES = {
    "modeling": re.compile(
        r"\b(model|mesh|polygon|vertex|edge|face|topology|retopolog|"
        r"hard\s+surface|organic|sculpt)\b",
        re.I,
    ),
    "rigging": re.compile(
        r"\b(rig|armature|bone|weight\s+paint|IK|FK|control\s+rig|"
        r"deform|skin|metarig)\b",
        re.I,
    ),
    "animation": re.compile(
        r"\b(animat|keyframe|NLA|timeline|graph\s+editor|driver|"
        r"walk\s+cycle|lip\s+sync|motion\s+path)\b",
        re.I,
    ),
    "vfx": re.compile(
        r"\b(VFX|visual\s+effect|simulation|fluid|smoke|fire|"
        r"pyro|particle|cloth|rigid\s+body|soft\s+body|ocean)\b",
        re.I,
    ),
    "rendering": re.compile(
        r"\b(render|cycles|eevee|mantra|arnold|redshift|octane|"
        r"HDRI|lighting|camera|depth\s+of\s+field|motion\s+blur)\b",
        re.I,
    ),
    "texturing": re.compile(
        r"\b(texture|UV|PBR|material|shader|substance|bake|"
        r"normal\s+map|roughness|metallic|albedo)\b",
        re.I,
    ),
    "scripting": re.compile(
        r"\b(script|python|bpy|add-?on|plugin|automat|programm)\b", re.I
    ),
    "compositing": re.compile(
        r"\b(composit|post.?process|color\s+grade|render\s+pass|"
        r"node\s+composit|after\s+effects|nuke)\b",
        re.I,
    ),
}


def detect_software(text: str) -> str:
    """Detect primary 3D software mentioned in text."""
    counts = {}
    for sw, pattern in SOFTWARE_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            counts[sw] = len(matches)
    if not counts:
        return "blender"  # default assumption for this dataset
    return max(counts, key=counts.get)


def score_complexity(text: str) -> int:
    """
    Score training pair complexity from 1 (simple) to 5 (expert).
    Returns the highest level that matches.
    """
    text_lower = text.lower()
    # Start from expert and work down — return first match
    for level in range(5, 0, -1):
        for pattern in COMPILED_COMPLEXITY[level]:
            if pattern.search(text_lower):
                return level
    return 1  # default to primitive


def detect_concepts(text: str) -> list[str]:
    """Detect 3D concepts covered in text."""
    concepts = []
    for concept, pattern in CONCEPT_CATEGORIES.items():
        if pattern.search(text):
            concepts.append(concept)
    return concepts


def compute_quality_score(record: dict) -> float:
    """
    Compute a 0-1 quality score for a training record.
    Higher = better training signal.
    """
    score = 0.5  # base score

    # Source quality bonuses
    source_type = record.get("type", "")
    if source_type == "whole_script":
        score += 0.1
    elif source_type == "function_pair":
        score += 0.15  # function pairs tend to be more targeted

    # Stack Exchange answers with high votes = high quality
    if "answer_score" in record:
        a_score = record.get("answer_score", 0)
        score += min(a_score / 100.0, 0.2)  # cap at +0.2

    # Has bpy code = most valuable for this dataset
    if record.get("has_bpy_code"):
        score += 0.2

    # Has explicit code blocks
    code_blocks = record.get("code_blocks", [])
    if code_blocks:
        score += 0.1

    # GitHub stars proxy for quality
    stars = record.get("stars", 0)
    if stars >= 100:
        score += 0.1
    elif stars >= 10:
        score += 0.05

    # Penalize very short descriptions
    desc = record.get("description", record.get("question", ""))
    if len(desc) < 30:
        score -= 0.2

    return max(0.0, min(1.0, score))


def load_jsonl(filepath: Path) -> list[dict]:
    """Load records from a JSONL file."""
    records = []
    if not filepath.exists():
        return records
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def enrich_record(record: dict) -> dict:
    """Add curriculum metadata to a training record."""
    # Combine all text for analysis
    text_parts = [
        record.get("description", ""),
        record.get("question", ""),
        record.get("answer", ""),
        record.get("code", ""),
        record.get("question_title", ""),
        str(record.get("tags", "")),
    ]
    full_text = " ".join(p for p in text_parts if p)

    record["_software"] = detect_software(full_text)
    record["_complexity"] = score_complexity(full_text)
    record["_concepts"] = detect_concepts(full_text)
    record["_quality_score"] = compute_quality_score(record)
    return record


def build_curriculum(records: list[dict]) -> dict[str, list[dict]]:
    """
    Group enriched records into curriculum tiers and ensure software balance.
    Returns dict mapping tier names to ordered record lists.
    """
    # Software distribution targets
    software_targets = {
        "blender": 0.50,
        "houdini": 0.10,
        "maya": 0.10,
        "cinema4d": 0.08,
        "zbrush": 0.07,
        "substance": 0.07,
        "unreal": 0.04,
        "other": 0.04,
    }

    # Sort all records by complexity then quality
    enriched = []
    for rec in records:
        if "_complexity" not in rec:
            rec = enrich_record(rec)
        enriched.append(rec)

    enriched.sort(key=lambda r: (r["_complexity"], r["_quality_score"]))

    # Group into complexity tiers
    tiers: dict[int, list[dict]] = defaultdict(list)
    for rec in enriched:
        tiers[rec["_complexity"]].append(rec)

    # Balance software within each tier using target proportions
    balanced_tiers = {}
    for level in range(1, 6):
        tier_records = tiers[level]
        if not tier_records:
            continue

        # Group by software
        by_software: dict[str, list[dict]] = defaultdict(list)
        for rec in tier_records:
            sw = rec.get("_software", "blender")
            if sw not in software_targets:
                sw = "other"
            by_software[sw].append(rec)

        # Sort each software group by quality score
        for sw in by_software:
            by_software[sw].sort(key=lambda r: r["_quality_score"], reverse=True)

        balanced_tiers[f"tier_{level}"] = {
            "records": tier_records,
            "by_software": dict(by_software),
            "complexity_label": {
                1: "primitive_operations",
                2: "modifier_stacks",
                3: "node_graphs",
                4: "python_scripts",
                5: "expert_workflows",
            }[level],
        }

    return balanced_tiers


def generate_curriculum_dataset(
    balanced_tiers: dict,
    total_target: Optional[int] = None,
    shuffle_within_tier: bool = True,
) -> list[dict]:
    """
    Generate the final ordered curriculum dataset.
    Progressive difficulty: tier 1 → tier 5.
    Within each tier: software-balanced, quality-sorted.
    """
    software_targets = {
        "blender": 0.50,
        "houdini": 0.10,
        "maya": 0.10,
        "cinema4d": 0.08,
        "zbrush": 0.07,
        "substance": 0.07,
        "unreal": 0.04,
        "other": 0.04,
    }

    final_records = []

    for tier_name in sorted(balanced_tiers.keys()):
        tier = balanced_tiers[tier_name]
        tier_records = tier["records"]
        by_software = tier["by_software"]
        complexity_label = tier["complexity_label"]

        # Calculate target counts per software for this tier
        tier_size = len(tier_records)
        tier_final = []

        for sw, target_frac in software_targets.items():
            sw_records = by_software.get(sw, [])
            target_count = int(tier_size * target_frac)
            selected = sw_records[:target_count]
            tier_final.extend(selected)

        # Add remaining records that weren't targeted
        selected_ids = {id(r) for r in tier_final}
        remaining = [r for r in tier_records if id(r) not in selected_ids]
        tier_final.extend(remaining)

        if shuffle_within_tier:
            random.shuffle(tier_final)

        # Tag each record with curriculum position
        for i, rec in enumerate(tier_final):
            rec["_curriculum_tier"] = tier_name
            rec["_curriculum_label"] = complexity_label
            rec["_tier_position"] = i

        final_records.extend(tier_final)

    # Apply total target limit if specified
    if total_target:
        # Sample proportionally across tiers
        if len(final_records) > total_target:
            final_records = final_records[:total_target]

    return final_records


def print_stats(records: list[dict]) -> None:
    """Print curriculum statistics."""
    by_tier: dict[str, int] = defaultdict(int)
    by_software: dict[str, int] = defaultdict(int)
    by_concept: dict[str, int] = defaultdict(int)
    quality_sum = 0.0

    for rec in records:
        tier = rec.get("_curriculum_tier", "unknown")
        by_tier[tier] += 1
        sw = rec.get("_software", "unknown")
        by_software[sw] += 1
        for concept in rec.get("_concepts", []):
            by_concept[concept] += 1
        quality_sum += rec.get("_quality_score", 0.0)

    total = len(records)
    print("\n=== CURRICULUM STATISTICS ===")
    print(f"Total records: {total}")
    print(f"Average quality score: {quality_sum / max(total, 1):.3f}")

    print("\nBy complexity tier:")
    for tier in sorted(by_tier.keys()):
        count = by_tier[tier]
        print(f"  {tier}: {count:>6} ({100 * count / max(total, 1):.1f}%)")

    print("\nBy software:")
    for sw, count in sorted(by_software.items(), key=lambda x: -x[1]):
        print(f"  {sw:15}: {count:>6} ({100 * count / max(total, 1):.1f}%)")

    print("\nBy 3D concept:")
    for concept, count in sorted(by_concept.items(), key=lambda x: -x[1])[:10]:
        print(f"  {concept:15}: {count:>6}")


def main():
    parser = argparse.ArgumentParser(
        description="Build curriculum for 3D AI training data"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_DIR,
        help="Directory containing JSONL training data files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CURRICULUM_DIR,
        help="Output directory for curriculum datasets",
    )
    parser.add_argument(
        "--stats", action="store_true", help="Show statistics only, don't write output"
    )
    parser.add_argument(
        "--total",
        type=int,
        default=None,
        help="Total training examples to include (samples proportionally)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # ── Load all available training data ─────────────────────────────────────
    input_files = [
        args.input / "bpy_scripts.jsonl",
        args.input / "stackexchange_qa.jsonl",
        args.input / "youtube_transcripts.jsonl",
        args.input / "sketchfab_metadata.jsonl",
        args.input / "artstation_tutorials.jsonl",
    ]

    all_records = []
    for f in input_files:
        if f.exists():
            records = load_jsonl(f)
            print(f"Loaded {len(records):>6} records from {f.name}")
            all_records.extend(records)
        else:
            print(f"  [skip] {f.name} not found")

    if not all_records:
        print("\nNo data found. Run discovery scripts first.")
        return

    print(f"\nTotal records loaded: {len(all_records)}")

    # ── Enrich with curriculum metadata ──────────────────────────────────────
    print("Enriching records with curriculum metadata...")
    enriched = [enrich_record(r) for r in all_records]

    # ── Build curriculum ──────────────────────────────────────────────────────
    print("Building balanced curriculum tiers...")
    balanced_tiers = build_curriculum(enriched)
    curriculum = generate_curriculum_dataset(
        balanced_tiers,
        total_target=args.total,
        shuffle_within_tier=True,
    )

    if args.stats:
        print_stats(curriculum)
        return

    print_stats(curriculum)

    # ── Write output ──────────────────────────────────────────────────────────
    args.output.mkdir(parents=True, exist_ok=True)

    # Full curriculum (ordered)
    full_path = args.output / "curriculum_full.jsonl"
    with open(full_path, "w") as f:
        for rec in curriculum:
            f.write(json.dumps(rec) + "\n")
    print(f"\nFull curriculum: {full_path} ({len(curriculum)} records)")

    # Per-tier datasets
    by_tier: dict[str, list] = defaultdict(list)
    for rec in curriculum:
        tier = rec.get("_curriculum_tier", "tier_1")
        by_tier[tier].append(rec)

    for tier_name, tier_records in sorted(by_tier.items()):
        tier_path = args.output / f"curriculum_{tier_name}.jsonl"
        with open(tier_path, "w") as f:
            for rec in tier_records:
                f.write(json.dumps(rec) + "\n")
        print(f"  {tier_name}: {tier_path} ({len(tier_records)} records)")

    # High-quality subset (quality_score >= 0.7) for fine-tuning
    hq_records = [r for r in curriculum if r.get("_quality_score", 0) >= 0.7]
    hq_path = args.output / "curriculum_high_quality.jsonl"
    with open(hq_path, "w") as f:
        for rec in hq_records:
            f.write(json.dumps(rec) + "\n")
    print(f"  high_quality: {hq_path} ({len(hq_records)} records)")

    print(
        f"\nNext step: python training/train.py --data {args.output / 'curriculum_full.jsonl'}"
    )


if __name__ == "__main__":
    main()
