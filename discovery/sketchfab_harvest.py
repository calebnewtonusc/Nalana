"""
sketchfab_harvest.py - Sketchfab 3D model metadata and tutorial crawling.

Uses the Sketchfab API to crawl downloadable 3D model metadata.
Associates models with tutorials and workflow descriptions for training data.

API: https://api.sketchfab.com/v3/
Docs: https://docs.sketchfab.com/data-api/v3/

Usage:
    python discovery/sketchfab_harvest.py
    python discovery/sketchfab_harvest.py --token YOUR_SKETCHFAB_TOKEN --max-pages 200
"""

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parents[1] / "data"
SKETCHFAB_FILE = DATA_DIR / "sketchfab_metadata.jsonl"

SF_BASE = "https://api.sketchfab.com/v3"

# Categories that are most useful for 3D training
USEFUL_CATEGORIES = [
    "characters-creatures",
    "architecture",
    "vehicles-transportation",
    "science-technology",
    "weapons-military",
    "cultural-heritage-history",
    "nature-plants",
    "furniture-home",
    "electronics-gadgets",
    "sports-fitness",
]


def sf_get(endpoint: str, params: dict, token: str = "") -> dict:
    """Make Sketchfab API request."""
    url = f"{SF_BASE}/{endpoint}?" + urllib.parse.urlencode(params)
    headers = {"User-Agent": "nalana-dataset-harvester/1.0"}
    if token:
        headers["Authorization"] = f"Token {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"    [ERROR] Sketchfab {endpoint}: {e}")
        return {}


def fetch_models_page(
    cursor: str = None,
    categories: str = None,
    token: str = "",
    sort_by: str = "-likeCount",
) -> dict:
    """Fetch a page of downloadable models."""
    params = {
        "type": "downloadable",
        "count": 24,
        "sort_by": sort_by,
        "staffpicked": "false",
    }
    if cursor:
        params["cursor"] = cursor
    if categories:
        params["categories"] = categories
    return sf_get("models", params, token)


def extract_model_record(model: dict) -> dict:
    """Extract training-relevant fields from a Sketchfab model response."""
    # Build scene description from model metadata
    tags = [t.get("name", "") for t in model.get("tags", [])]
    categories = [c.get("name", "") for c in model.get("categories", [])]

    description = model.get("description", "") or ""
    name = model.get("name", "")

    # Construct natural language scene description
    scene_desc_parts = [f"3D model: {name}"]
    if categories:
        scene_desc_parts.append(f"Category: {', '.join(categories)}")
    if tags:
        scene_desc_parts.append(f"Tags: {', '.join(tags[:10])}")
    if description:
        scene_desc_parts.append(f"Description: {description[:300]}")

    # Vertex/face counts tell us mesh complexity
    vertex_count = model.get("vertexCount", 0)
    face_count = model.get("faceCount", 0)
    if face_count:
        if face_count < 1000:
            complexity = "low poly"
        elif face_count < 50000:
            complexity = "medium poly"
        elif face_count < 500000:
            complexity = "high poly"
        else:
            complexity = "very high poly / scan data"
        scene_desc_parts.append(f"Mesh complexity: {complexity} ({face_count:,} faces)")

    # Animation
    animation_count = model.get("animationCount", 0)
    if animation_count:
        scene_desc_parts.append(f"Animations: {animation_count} animation(s)")

    return {
        "uid": model.get("uid"),
        "name": name,
        "description": description[:500],
        "scene_description": ". ".join(scene_desc_parts),
        "tags": tags,
        "categories": categories,
        "vertex_count": vertex_count,
        "face_count": face_count,
        "animation_count": animation_count,
        "like_count": model.get("likeCount", 0),
        "view_count": model.get("viewCount", 0),
        "download_count": model.get("downloadCount", 0),
        "published_at": model.get("publishedAt", ""),
        "license": model.get("license", {}).get("label", "unknown") if model.get("license") else "unknown",
        "url": f"https://sketchfab.com/3d-models/{model.get('uid', '')}",
        "thumbnail_url": (
            model.get("thumbnails", {}).get("images", [{}])[0].get("url", "")
            if model.get("thumbnails", {}).get("images") else ""
        ),
        "type": "sketchfab_model",
    }


def load_seen_uids() -> set[str]:
    seen = set()
    if SKETCHFAB_FILE.exists():
        with open(SKETCHFAB_FILE) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    seen.add(rec.get("uid", ""))
                except json.JSONDecodeError:
                    pass
    return seen


def save_records(records: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SKETCHFAB_FILE, "a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Harvest Sketchfab model metadata")
    parser.add_argument("--token", default=os.environ.get("SKETCHFAB_TOKEN", ""),
                        help="Sketchfab API token (optional, increases rate limits)")
    parser.add_argument("--max-pages", type=int, default=500,
                        help="Max pages to crawl per category (24 models/page)")
    parser.add_argument("--all-categories", action="store_true",
                        help="Crawl all useful categories (default: top models only)")
    args = parser.parse_args()

    seen_uids = load_seen_uids()
    total_saved = 0

    print(f"=== SKETCHFAB METADATA HARVESTER ===")
    print(f"Starting with {len(seen_uids)} already seen models\n")

    crawl_targets = []

    # Always crawl top models (no category filter)
    crawl_targets.append(("top_overall", None))

    if args.all_categories:
        for cat in USEFUL_CATEGORIES:
            crawl_targets.append((cat, cat))

    for target_name, category_filter in crawl_targets:
        print(f"\n--- Crawling: {target_name} ---")
        cursor = None
        pages = 0

        while pages < args.max_pages:
            data = fetch_models_page(
                cursor=cursor,
                categories=category_filter,
                token=args.token,
            )

            if not data:
                break

            models = data.get("results", [])
            if not models:
                break

            new_records = []
            for model in models:
                uid = model.get("uid", "")
                if uid in seen_uids:
                    continue
                record = extract_model_record(model)
                new_records.append(record)
                seen_uids.add(uid)

            save_records(new_records)
            total_saved += len(new_records)
            pages += 1

            # Get next cursor for pagination
            next_url = data.get("next")
            if not next_url:
                break
            # Extract cursor from next URL
            parsed = urllib.parse.urlparse(next_url)
            params = urllib.parse.parse_qs(parsed.query)
            cursor = params.get("cursor", [None])[0]

            print(f"  Page {pages}: +{len(new_records)} new | total: {total_saved}")
            time.sleep(0.5)  # polite rate limiting

    print(f"\n=== DONE ===")
    print(f"Total Sketchfab records saved: {total_saved}")
    print(f"Output: {SKETCHFAB_FILE}")


if __name__ == "__main__":
    main()
