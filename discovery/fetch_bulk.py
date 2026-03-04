"""
fetch_bulk.py - Parallel transcript fetching for thousands of videos.

Reads video IDs from data/video_ids.txt (produced by discover_v2.py).
Fetches transcripts in parallel using a thread pool (I/O bound — threads
are more efficient than async here).

Performance: ~20 workers → ~200-400 transcripts/minute depending on YouTube.

Usage:
    python fetch_bulk.py                  # fetch all pending IDs
    python fetch_bulk.py --workers 30     # more aggressive
    python fetch_bulk.py --limit 500      # test run, first 500 IDs
    python fetch_bulk.py --resume         # skip already-fetched (default)
    python fetch_bulk.py --force          # re-fetch everything
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

DATA_DIR   = Path(__file__).parents[1] / "data"
RAW_DIR    = DATA_DIR / "raw"
IDS_FILE   = DATA_DIR / "video_ids.txt"
SKIP_FILE  = DATA_DIR / "skip.txt"   # IDs with no English transcript

_print_lock = Lock()


def safe_print(*args):
    with _print_lock:
        print(*args)


def load_video_ids(limit: int | None = None) -> list[str]:
    if not IDS_FILE.exists():
        # Fall back to URLs file for backward compat
        urls_file = Path(__file__).parent / "urls.txt"
        if urls_file.exists():
            import re
            content = urls_file.read_text()
            ids = re.findall(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", content)
            return ids[:limit] if limit else ids
        return []
    ids = [l.strip() for l in IDS_FILE.read_text().splitlines() if l.strip()]
    return ids[:limit] if limit else ids


def load_skip_ids() -> set[str]:
    if not SKIP_FILE.exists():
        return set()
    return set(l.strip() for l in SKIP_FILE.read_text().splitlines() if l.strip())


def mark_skip(video_id: str):
    with _print_lock:
        with SKIP_FILE.open("a") as f:
            f.write(video_id + "\n")


def merge_segments(segments: list[dict], window_seconds: float = 30.0) -> list[dict]:
    if not segments:
        return []

    chunks = []
    current_text: list[str] = []
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


def fetch_one(video_id: str) -> tuple[str, str]:
    """Returns (video_id, status) where status is 'saved', 'cached', 'skip', or 'error:...'"""
    out_path = RAW_DIR / f"{video_id}.json"
    if out_path.exists():
        return video_id, "cached"

    try:
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id, languages=["en", "en-US"])
        raw = [{"text": s.text, "start": s.start, "duration": s.duration} for s in fetched]
        chunks = merge_segments(raw)

        payload = {
            "video_id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "raw_segments": raw,
            "chunks": chunks,
        }
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload))
        return video_id, "saved"

    except (TranscriptsDisabled, NoTranscriptFound):
        mark_skip(video_id)
        return video_id, "skip"
    except Exception as e:
        return video_id, f"error:{e}"


def main():
    parser = argparse.ArgumentParser(description="Bulk parallel transcript fetching")
    parser.add_argument("--workers", type=int, default=20, help="Parallel workers (default: 20)")
    parser.add_argument("--limit", type=int, help="Max IDs to process (for test runs)")
    parser.add_argument("--force", action="store_true", help="Re-fetch already cached transcripts")
    args = parser.parse_args()

    all_ids = load_video_ids(args.limit)
    if not all_ids:
        print(f"No video IDs found. Run discover_v2.py first, or add URLs to urls.txt.")
        return

    skip_ids = load_skip_ids()
    already_fetched = set(p.stem for p in RAW_DIR.glob("*.json")) if RAW_DIR.exists() else set()

    if args.force:
        pending = [vid for vid in all_ids if vid not in skip_ids]
    else:
        pending = [vid for vid in all_ids if vid not in skip_ids and vid not in already_fetched]

    print(f"Video IDs total:    {len(all_ids)}")
    print(f"Already fetched:    {len(already_fetched)}")
    print(f"Skipped (no EN):    {len(skip_ids)}")
    print(f"Pending:            {len(pending)}")
    print(f"Workers:            {args.workers}")
    print()

    if not pending:
        print("Nothing to fetch.")
        return

    counters = {"saved": 0, "cached": 0, "skip": 0, "error": 0}
    start_time = time.time()

    if HAS_TQDM:
        pbar = tqdm(total=len(pending), unit="video", ncols=80)
    else:
        pbar = None

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch_one, vid): vid for vid in pending}
        for future in as_completed(futures):
            video_id, status = future.result()

            if status == "saved":
                counters["saved"] += 1
            elif status == "cached":
                counters["cached"] += 1
            elif status == "skip":
                counters["skip"] += 1
            else:
                counters["error"] += 1
                safe_print(f"\n  [ERR] {video_id}: {status}")

            if pbar:
                elapsed = time.time() - start_time
                rate = (counters["saved"] + counters["skip"]) / max(elapsed, 1)
                pbar.set_postfix(saved=counters["saved"], skip=counters["skip"],
                                  err=counters["error"], rate=f"{rate:.1f}/s")
                pbar.update(1)

    if pbar:
        pbar.close()

    elapsed = time.time() - start_time
    total_raw = len(list(RAW_DIR.glob("*.json"))) if RAW_DIR.exists() else 0

    print(f"\n{'─'*40}")
    print(f"Saved:    {counters['saved']}")
    print(f"Skipped:  {counters['skip']}  (no English transcript)")
    print(f"Errors:   {counters['error']}")
    print(f"Time:     {elapsed:.0f}s  ({counters['saved']/max(elapsed,1):.1f} saved/s)")
    print(f"Total raw transcripts: {total_raw}")
    print(f"\nNext step: python synthesize_bulk.py")


if __name__ == "__main__":
    main()
