"""
fetch.py - Pull YouTube transcripts for Blender tutorials.

Usage:
    python fetch.py                      # uses urls.txt
    python fetch.py --url URL            # single URL
    python fetch.py --urls-file FILE     # custom file
"""

import argparse
import json
import re
import sys
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound


DATA_DIR = Path(__file__).parents[1] / "data" / "raw"


def extract_video_id(url: str) -> str:
    patterns = [
        r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from: {url}")


def fetch_transcript(video_id: str) -> list[dict]:
    """Returns list of {text, start, duration} segments."""
    fetched = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "en-US"])
    return [{"text": s["text"], "start": s["start"], "duration": s["duration"]} for s in fetched]


def merge_segments(segments: list[dict], window_seconds: float = 30.0) -> list[dict]:
    """
    Merge raw segments into ~30-second chunks for cleaner processing.
    Returns list of {text, start, end}.
    """
    if not segments:
        return []

    chunks = []
    current_text = []
    chunk_start = segments[0]["start"]
    chunk_end = chunk_start

    for seg in segments:
        seg_end = seg["start"] + seg["duration"]
        if seg["start"] - chunk_start >= window_seconds and current_text:
            chunks.append({
                "text": " ".join(current_text).strip(),
                "start": round(chunk_start, 2),
                "end": round(chunk_end, 2),
            })
            current_text = []
            chunk_start = seg["start"]

        current_text.append(seg["text"])
        chunk_end = seg_end

    if current_text:
        chunks.append({
            "text": " ".join(current_text).strip(),
            "start": round(chunk_start, 2),
            "end": round(chunk_end, 2),
        })

    return chunks


def fetch_and_save(url: str, force: bool = False) -> Path | None:
    try:
        video_id = extract_video_id(url)
    except ValueError as e:
        print(f"  [SKIP] {e}")
        return None

    out_path = DATA_DIR / f"{video_id}.json"
    if out_path.exists() and not force:
        print(f"  [CACHED] {video_id} -> {out_path.name}")
        return out_path

    try:
        print(f"  [FETCH] {video_id} ({url})")
        raw = fetch_transcript(video_id)
        chunks = merge_segments(raw)
        payload = {
            "video_id": video_id,
            "url": url,
            "raw_segments": raw,
            "chunks": chunks,
        }
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"  [SAVED] {len(chunks)} chunks -> {out_path.name}")
        return out_path

    except TranscriptsDisabled:
        print(f"  [ERROR] Transcripts disabled for {video_id}")
    except NoTranscriptFound:
        print(f"  [ERROR] No English transcript for {video_id}")
    except Exception as e:
        print(f"  [ERROR] {video_id}: {e}")

    return None


def load_urls(path: str) -> list[str]:
    lines = Path(path).read_text().splitlines()
    return [l.strip() for l in lines if l.strip() and not l.startswith("#")]


def main():
    parser = argparse.ArgumentParser(description="Fetch YouTube transcripts for Blender tutorials")
    parser.add_argument("--url", help="Single YouTube URL")
    parser.add_argument("--urls-file", default="urls.txt", help="File with one URL per line")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cached")
    args = parser.parse_args()

    if args.url:
        urls = [args.url]
    else:
        if not Path(args.urls_file).exists():
            print(f"No urls file found at {args.urls_file}. Pass --url or create urls.txt.")
            sys.exit(1)
        urls = load_urls(args.urls_file)

    print(f"Fetching {len(urls)} tutorial(s)...\n")
    saved = []
    for url in urls:
        result = fetch_and_save(url, force=args.force)
        if result:
            saved.append(result)

    print(f"\nDone. {len(saved)}/{len(urls)} transcripts saved to {DATA_DIR}/")


if __name__ == "__main__":
    main()
