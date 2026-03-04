"""
discover.py - Auto-discover Blender tutorial URLs via YouTube Data API v3
and append new ones to urls.txt.

Requires a YouTube Data API v3 key (free, 10k quota/day; search costs 100 units/call).

Usage:
    python discover.py --api-key YOUR_KEY
    python discover.py --api-key YOUR_KEY --query "blender modeling tutorial" --max 50
    python discover.py --api-key YOUR_KEY --dry-run   # print without saving

Set YOUTUBE_API_KEY env var to avoid passing --api-key every time.
"""

import argparse
import json
import os
import re
import urllib.request
import urllib.parse
from pathlib import Path


URLS_FILE = Path(__file__).parent / "urls.txt"

# Targeted queries to maximize operation-dense tutorials
DEFAULT_QUERIES = [
    "blender 3d modeling tutorial beginners step by step",
    "blender mesh modeling fundamentals",
    "blender modifiers tutorial",
    "blender sculpting tutorial",
    "blender geometry nodes tutorial",
    "blender hard surface modeling",
    "blender character modeling tutorial",
    "blender shading materials tutorial",
    "blender rigging tutorial",
    "blender animation tutorial beginners",
]

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEO_URL  = "https://www.googleapis.com/youtube/v3/videos"


def search_videos(api_key: str, query: str, max_results: int = 10) -> list[dict]:
    """Returns list of {video_id, title, description, duration} dicts."""
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": min(max_results, 50),
        "videoDuration": "medium",   # 4-20 min — denser tutorials
        "relevanceLanguage": "en",
        "key": api_key,
    }
    url = YOUTUBE_SEARCH_URL + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read())

    results = []
    for item in data.get("items", []):
        vid_id = item["id"].get("videoId")
        if not vid_id:
            continue
        snippet = item.get("snippet", {})
        results.append({
            "video_id": vid_id,
            "title": snippet.get("title", ""),
            "description": snippet.get("description", "")[:200],
        })
    return results


def filter_blender_tutorials(videos: list[dict]) -> list[dict]:
    """Basic heuristic filter — keep videos that look like real Blender tutorials."""
    keywords = ["blender", "tutorial", "model", "mesh", "sculpt", "geometry", "modifier"]
    skip_terms = ["reaction", "meme", "speedrun", "timelapse", "no commentary"]

    filtered = []
    for v in videos:
        text = (v["title"] + " " + v["description"]).lower()
        if any(bad in text for bad in skip_terms):
            continue
        if sum(1 for kw in keywords if kw in text) >= 2:
            filtered.append(v)
    return filtered


def load_existing_ids() -> set[str]:
    if not URLS_FILE.exists():
        return set()
    content = URLS_FILE.read_text()
    return set(re.findall(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", content))


def append_to_urls_file(videos: list[dict], query: str):
    existing = load_existing_ids()
    new_videos = [v for v in videos if v["video_id"] not in existing]
    if not new_videos:
        return 0

    lines = [f"\n# === Auto-discovered: {query} ==="]
    for v in new_videos:
        title = v["title"].replace("\n", " ")
        lines.append(f"# {title}")
        lines.append(f"https://www.youtube.com/watch?v={v['video_id']}")

    with URLS_FILE.open("a") as f:
        f.write("\n".join(lines) + "\n")

    return len(new_videos)


def main():
    parser = argparse.ArgumentParser(description="Discover Blender tutorial URLs via YouTube API")
    parser.add_argument("--api-key", default=os.environ.get("YOUTUBE_API_KEY", ""), help="YouTube Data API v3 key")
    parser.add_argument("--query", help="Single search query (overrides default list)")
    parser.add_argument("--max", type=int, default=10, help="Max results per query (default: 10)")
    parser.add_argument("--dry-run", action="store_true", help="Print results without saving")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: provide --api-key or set YOUTUBE_API_KEY env var.")
        print("Get a free key at: https://console.developers.google.com/")
        return

    queries = [args.query] if args.query else DEFAULT_QUERIES
    total_added = 0

    for query in queries:
        print(f"\nSearching: \"{query}\"")
        try:
            results = search_videos(args.api_key, query, args.max)
            filtered = filter_blender_tutorials(results)
            print(f"  Found {len(results)} results, {len(filtered)} passed filter")

            if args.dry_run:
                for v in filtered:
                    print(f"  [{v['video_id']}] {v['title']}")
            else:
                added = append_to_urls_file(filtered, query)
                print(f"  Added {added} new URL(s) to urls.txt")
                total_added += added

        except Exception as e:
            print(f"  [ERROR] {e}")

    if not args.dry_run:
        print(f"\nTotal new URLs added: {total_added}")
        print(f"Run `python pipeline.py` to fetch and process them.")


if __name__ == "__main__":
    main()
