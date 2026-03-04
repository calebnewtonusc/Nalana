"""
pipeline.py - End-to-end: fetch transcripts then synthesize training pairs.

Usage:
    python pipeline.py                    # full run: fetch all urls.txt + synthesize
    python pipeline.py --fetch-only       # only download transcripts
    python pipeline.py --synth-only       # only run synthesis on existing raw files
    python pipeline.py --stats            # print dataset stats

This is the main entry point for building the Nalana training dataset.
"""

import argparse
import json
from pathlib import Path

RAW_DIR = Path(__file__).parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent / "data" / "processed"
MASTER_JSONL = PROCESSED_DIR / "dataset.jsonl"


def run_fetch(urls_file: str = "urls.txt", force: bool = False):
    from discovery.fetch import load_urls, fetch_and_save

    urls_path = Path(urls_file)
    if not urls_path.exists():
        print(f"No urls file at {urls_file}. Create it with one YouTube URL per line.")
        return 0

    urls = load_urls(urls_file)
    print(f"=== FETCH: {len(urls)} URLs ===\n")
    saved = 0
    for url in urls:
        result = fetch_and_save(url, force=force)
        if result:
            saved += 1
    print(f"\nFetch complete: {saved}/{len(urls)} transcripts\n")
    return saved


def run_synthesize(force: bool = False):
    from synthesis.synthesize import synthesize_video, merge_all_jsonl

    raw_files = sorted(RAW_DIR.glob("*.json"))
    if not raw_files:
        print(f"No raw transcripts found in {RAW_DIR}/")
        return 0

    print(f"=== SYNTHESIZE: {len(raw_files)} videos ===\n")
    processed = 0
    for p in raw_files:
        result = synthesize_video(p.stem, force=force)
        if result:
            processed += 1
        print()

    count = merge_all_jsonl(MASTER_JSONL)
    print(f"\nSynthesis complete: {processed}/{len(raw_files)} videos")
    print(f"Master dataset: {count} training pairs -> {MASTER_JSONL}\n")
    return processed


def print_stats():
    print("=== DATASET STATS ===\n")

    raw_files = list(RAW_DIR.glob("*.json"))
    print(f"Raw transcripts:    {len(raw_files)}")

    total_chunks = 0
    total_duration = 0.0
    for p in raw_files:
        data = json.loads(p.read_text())
        chunks = data.get("chunks", [])
        total_chunks += len(chunks)
        if chunks:
            total_duration += chunks[-1]["end"]

    print(f"Total chunks:       {total_chunks}")
    print(f"Total duration:     {total_duration / 60:.1f} min of tutorials\n")

    processed_files = list(PROCESSED_DIR.glob("*.jsonl"))
    processed_files = [f for f in processed_files if f != MASTER_JSONL]
    print(f"Processed videos:   {len(processed_files)}")

    if MASTER_JSONL.exists():
        lines = [l for l in MASTER_JSONL.read_text().splitlines() if l.strip()]
        print(f"Training pairs:     {len(lines)}")

        # Sample op type distribution
        op_counts: dict[str, int] = {}
        for line in lines:
            try:
                pair = json.loads(line)
                op = pair.get("blender_op", {}).get("op", "unknown")
                op_counts[op] = op_counts.get(op, 0) + 1
            except Exception:
                pass

        if op_counts:
            print("\nTop Blender operations:")
            for op, count in sorted(op_counts.items(), key=lambda x: -x[1])[:10]:
                print(f"  {op:<45} {count:>4}")
    else:
        print("Training pairs:     0 (run synthesize first)")


def main():
    parser = argparse.ArgumentParser(description="Nalana dataset pipeline")
    parser.add_argument(
        "--fetch-only", action="store_true", help="Only fetch transcripts"
    )
    parser.add_argument("--synth-only", action="store_true", help="Only run synthesis")
    parser.add_argument("--stats", action="store_true", help="Print dataset statistics")
    parser.add_argument(
        "--force", action="store_true", help="Re-fetch / re-process cached data"
    )
    parser.add_argument(
        "--urls-file", default="urls.txt", help="URL list file (default: urls.txt)"
    )
    args = parser.parse_args()

    if args.stats:
        print_stats()
        return

    if args.fetch_only:
        run_fetch(args.urls_file, args.force)
    elif args.synth_only:
        run_synthesize(args.force)
    else:
        run_fetch(args.urls_file, args.force)
        run_synthesize(args.force)


if __name__ == "__main__":
    main()
