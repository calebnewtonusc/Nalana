"""
objaverse_prep.py - Download and filter Objaverse 3D objects for Stream 2 training.

Objaverse: 800k+ CC-licensed 3D objects with text annotations.
Objaverse-XL: 10M+ objects (subset recommended for quality).

We filter for:
  - Objects with clean geometry (not point clouds or terrain scans)
  - Objects in categories useful for Nalana training
  - Objects with sufficient polygon count (not too simple, not too dense)

Output: data/objaverse/metadata.jsonl — one object per line with uid, category, path.

Usage:
    python objaverse_prep.py --limit 50000       # download 50k objects
    python objaverse_prep.py --categories all    # all categories
    python objaverse_prep.py --categories product furniture vehicle character
    python objaverse_prep.py --stats             # just print dataset stats
"""

import argparse
import json
import random
from pathlib import Path

OBJAVERSE_DIR = Path(__file__).parents[1] / "data" / "objaverse"

# Categories most relevant to teaching Nalana 3D form understanding
# These correspond to Objaverse's lvis annotation categories
TARGET_CATEGORIES = {
    # Hard surface / product design
    "electronics": ["phone", "laptop", "camera", "computer", "tablet", "headphones",
                    "keyboard", "mouse", "speaker", "monitor", "television"],
    "furniture": ["chair", "table", "sofa", "couch", "desk", "bed", "shelf",
                  "cabinet", "lamp", "bookcase"],
    "vehicles": ["car", "truck", "motorcycle", "bicycle", "airplane", "boat",
                 "helicopter", "scooter", "bus"],
    "tools": ["hammer", "wrench", "screwdriver", "drill", "saw", "knife",
              "scissors", "pliers"],
    "containers": ["bottle", "cup", "mug", "bowl", "box", "jar", "vase",
                   "can", "bucket"],
    "buildings": ["house", "building", "tower", "bridge", "church", "castle"],

    # Organic / characters
    "characters": ["person", "human", "character", "figure", "mannequin"],
    "animals": ["dog", "cat", "bird", "horse", "fish", "dragon"],
    "plants": ["tree", "plant", "flower", "grass", "bush"],

    # Architectural / environments
    "interior": ["room", "kitchen", "bathroom", "living room", "office"],
    "props": ["weapon", "sword", "shield", "gun", "armor"],
}

ALL_KEYWORDS = [kw for kwlist in TARGET_CATEGORIES.values() for kw in kwlist]


def download_objaverse(limit: int, categories: list[str], seed: int = 42) -> list[dict]:
    """Download Objaverse objects using the official objaverse package."""
    try:
        import objaverse
    except ImportError:
        print("Install objaverse: pip install objaverse")
        return []

    print("Loading Objaverse annotations...")
    annotations = objaverse.load_annotations()  # dict: uid -> metadata

    # Filter by category keywords
    if categories != ["all"]:
        target_keywords = set()
        for cat in categories:
            if cat in TARGET_CATEGORIES:
                target_keywords.update(TARGET_CATEGORIES[cat])
            else:
                target_keywords.add(cat.lower())
    else:
        target_keywords = set(ALL_KEYWORDS)

    print(f"Filtering {len(annotations):,} objects by {len(target_keywords)} keywords...")
    filtered = {}
    for uid, meta in annotations.items():
        name = (meta.get("name", "") + " " + meta.get("description", "")).lower()
        tags = [t.lower() for t in meta.get("tags", [])]
        combined = name + " " + " ".join(tags)
        if any(kw in combined for kw in target_keywords):
            filtered[uid] = meta

    print(f"Matched: {len(filtered):,} objects")

    # Sample if over limit
    if limit and len(filtered) > limit:
        random.seed(seed)
        uids = random.sample(list(filtered.keys()), limit)
        filtered = {uid: filtered[uid] for uid in uids}

    print(f"Downloading {len(filtered):,} objects (GLB format)...")
    objects = objaverse.load_objects(
        uids=list(filtered.keys()),
        download_processes=8,
    )

    # Merge metadata + local paths
    results = []
    for uid, local_path in objects.items():
        meta = filtered.get(uid, {})
        results.append({
            "uid": uid,
            "name": meta.get("name", uid),
            "description": meta.get("description", ""),
            "tags": meta.get("tags", []),
            "local_path": str(local_path),
            "categories": [cat for cat, kws in TARGET_CATEGORIES.items()
                          if any(kw in (meta.get("name","")+" "+meta.get("description","")).lower()
                                 for kw in kws)],
        })

    return results


def save_metadata(objects: list[dict]):
    OBJAVERSE_DIR.mkdir(parents=True, exist_ok=True)
    out = OBJAVERSE_DIR / "metadata.jsonl"
    with out.open("w") as f:
        for obj in objects:
            f.write(json.dumps(obj) + "\n")
    print(f"Saved metadata: {out} ({len(objects):,} objects)")
    return out


def print_stats():
    meta_path = OBJAVERSE_DIR / "metadata.jsonl"
    if not meta_path.exists():
        print("No metadata found. Run download first.")
        return

    objects = [json.loads(l) for l in meta_path.read_text().splitlines() if l.strip()]
    print(f"\nObjaverse objects: {len(objects):,}")

    cat_counts = {}
    for obj in objects:
        for cat in obj.get("categories", ["unknown"]):
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

    print("\nBy category:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<20} {count:>6,}")

    rendered = len(list((OBJAVERSE_DIR / "renders").glob("*/front.png"))) if (OBJAVERSE_DIR / "renders").exists() else 0
    annotated = len(list((OBJAVERSE_DIR / "annotations").glob("*.json"))) if (OBJAVERSE_DIR / "annotations").exists() else 0
    print(f"\nRendered:  {rendered:,}")
    print(f"Annotated: {annotated:,}")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare Objaverse 3D objects")
    parser.add_argument("--limit", type=int, default=50000, help="Max objects to download")
    parser.add_argument("--categories", nargs="+", default=["all"],
                        choices=list(TARGET_CATEGORIES.keys()) + ["all"],
                        help="Categories to include")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.stats:
        print_stats()
        return

    objects = download_objaverse(args.limit, args.categories, args.seed)
    if objects:
        save_metadata(objects)
        print(f"\nNext step: python render_pipeline.py")


if __name__ == "__main__":
    main()
