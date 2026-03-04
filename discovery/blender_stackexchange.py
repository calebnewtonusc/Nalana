"""
blender_stackexchange.py - Blender StackExchange Q&A pair discovery.

Uses the Stack Exchange API to pull all Blender.SE questions with accepted answers,
prioritizing questions that have bpy code in the answer.

Creates (user_question, expert_answer_with_code) pairs for training.

API: https://api.stackexchange.com/2.3/
Docs: https://api.stackexchange.com/docs

Usage:
    python discovery/blender_stackexchange.py
    python discovery/blender_stackexchange.py --max-pages 100
    python discovery/blender_stackexchange.py --code-only   # only bpy code answers
"""

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parents[1] / "data"
SE_QA_FILE = DATA_DIR / "stackexchange_qa.jsonl"
SE_PROGRESS_FILE = DATA_DIR / "se_progress.json"

SE_BASE = "https://api.stackexchange.com/2.3"

# Stack Exchange filter with body content (includes question/answer body)
# Filter "withbody" returns full HTML body
SE_FILTER = "withbody"

# ─── bpy code detection patterns ─────────────────────────────────────────────
BPY_PATTERNS = [
    re.compile(r'import bpy', re.IGNORECASE),
    re.compile(r'bpy\.(ops|data|context|types|utils|props)', re.IGNORECASE),
    re.compile(r'bpy\.ops\.\w+', re.IGNORECASE),
    re.compile(r'bpy\.data\.\w+', re.IGNORECASE),
]

# ─── HTML tag removal ─────────────────────────────────────────────────────────
TAG_PATTERN = re.compile(r'<[^>]+>')
CODE_BLOCK_PATTERN = re.compile(r'<code>(.*?)</code>', re.DOTALL | re.IGNORECASE)
PRE_BLOCK_PATTERN = re.compile(r'<pre[^>]*>(.*?)</pre>', re.DOTALL | re.IGNORECASE)
HTML_ENTITIES = {
    '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"',
    '&#39;': "'", '&nbsp;': ' ', '&#xA;': '\n', '&#10;': '\n',
}


def decode_html(html: str) -> str:
    """Strip HTML tags and decode entities."""
    for entity, char in HTML_ENTITIES.items():
        html = html.replace(entity, char)
    return TAG_PATTERN.sub('', html).strip()


def extract_code_blocks(html: str) -> list[str]:
    """Extract code blocks from HTML body."""
    blocks = []
    for match in PRE_BLOCK_PATTERN.finditer(html):
        code = decode_html(match.group(1))
        if code.strip():
            blocks.append(code)
    for match in CODE_BLOCK_PATTERN.finditer(html):
        code = decode_html(match.group(1))
        if code.strip() and code not in blocks:
            blocks.append(code)
    return blocks


def has_bpy_code(html: str) -> bool:
    """Check if the HTML contains bpy code."""
    for pattern in BPY_PATTERNS:
        if pattern.search(html):
            return True
    return False


def has_python_code(html: str) -> bool:
    """Check if the HTML contains any code block."""
    code_blocks = extract_code_blocks(html)
    return len(code_blocks) > 0


def se_get(endpoint: str, params: dict) -> dict:
    """Make Stack Exchange API request."""
    url = f"{SE_BASE}/{endpoint}?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"    [ERROR] SE API {endpoint}: {e}")
        return {}


def fetch_questions_page(page: int, page_size: int = 100) -> dict:
    """Fetch a page of accepted-answer questions from Blender.SE."""
    return se_get("questions", {
        "site": "blender",
        "page": page,
        "pagesize": page_size,
        "order": "desc",
        "sort": "votes",
        "filter": SE_FILTER,
        "accepted": "True",
    })


def fetch_answer(answer_id: int) -> dict:
    """Fetch a single answer with full body."""
    data = se_get(f"answers/{answer_id}", {
        "site": "blender",
        "filter": SE_FILTER,
    })
    items = data.get("items", [])
    return items[0] if items else {}


def fetch_answers_batch(answer_ids: list[int]) -> list[dict]:
    """Fetch multiple answers in batch (up to 100 per request)."""
    if not answer_ids:
        return []
    ids_str = ";".join(str(i) for i in answer_ids[:100])
    data = se_get(f"answers/{ids_str}", {
        "site": "blender",
        "filter": SE_FILTER,
    })
    return data.get("items", [])


def build_training_record(question: dict, answer: dict, code_only: bool) -> dict | None:
    """
    Build a training (question, answer) pair.
    Returns None if the answer doesn't meet quality criteria.
    """
    q_body = question.get("body", "")
    a_body = answer.get("body", "")

    if not q_body or not a_body:
        return None

    # Quality gate: require code if code_only mode
    if code_only and not has_bpy_code(a_body) and not has_python_code(a_body):
        return None

    # Require some code in answer (pure text answers are less useful)
    code_blocks = extract_code_blocks(a_body)
    has_bpy = has_bpy_code(a_body)

    # Build clean versions
    q_clean = decode_html(q_body)
    a_clean = decode_html(a_body)
    q_title = question.get("title", "")

    # Format the question text
    question_text = f"{q_title}\n\n{q_clean}".strip()

    # Extract tags as context
    tags = question.get("tags", [])

    return {
        "question_id": question.get("question_id"),
        "answer_id": answer.get("answer_id"),
        "question": question_text[:2000],
        "answer": a_clean[:4000],
        "question_title": q_title,
        "tags": tags,
        "code_blocks": code_blocks[:5],  # top 5 code blocks
        "has_bpy_code": has_bpy,
        "question_score": question.get("score", 0),
        "answer_score": answer.get("score", 0),
        "view_count": question.get("view_count", 0),
        "is_accepted": True,
    }


def load_progress() -> dict:
    if SE_PROGRESS_FILE.exists():
        return json.loads(SE_PROGRESS_FILE.read_text())
    return {"last_page": 0, "total_saved": 0}


def save_progress(page: int, total: int) -> None:
    SE_PROGRESS_FILE.write_text(json.dumps({"last_page": page, "total_saved": total}))


def load_seen_ids() -> set[int]:
    """Load question IDs already processed."""
    seen = set()
    if SE_QA_FILE.exists():
        with open(SE_QA_FILE) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    seen.add(rec.get("question_id", 0))
                except json.JSONDecodeError:
                    pass
    return seen


def save_records(records: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SE_QA_FILE, "a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Harvest Blender StackExchange Q&A pairs")
    parser.add_argument("--max-pages", type=int, default=200,
                        help="Max pages to fetch (100 questions per page)")
    parser.add_argument("--code-only", action="store_true",
                        help="Only save answers that contain Python/bpy code")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--min-answer-score", type=int, default=0,
                        help="Minimum answer score to include")
    args = parser.parse_args()

    progress = load_progress() if args.resume else {"last_page": 0, "total_saved": 0}
    seen_ids = load_seen_ids() if args.resume else set()
    start_page = progress["last_page"] + 1

    total_saved = progress["total_saved"]
    total_bpy = 0
    total_code = 0

    print(f"=== BLENDER STACKEXCHANGE HARVESTER ===")
    print(f"Resuming from page {start_page}")
    print(f"Mode: {'code-only' if args.code_only else 'all accepted answers'}\n")

    for page in range(start_page, start_page + args.max_pages):
        print(f"  Fetching page {page}...")
        data = fetch_questions_page(page)

        if not data:
            print("    [STOP] Empty response")
            break

        items = data.get("items", [])
        if not items:
            print(f"    [STOP] No items on page {page}")
            break

        # Collect answer IDs for batch fetch
        answer_ids = []
        question_map: dict[int, dict] = {}
        for q in items:
            q_id = q.get("question_id", 0)
            if q_id in seen_ids:
                continue
            a_id = q.get("accepted_answer_id")
            if a_id:
                answer_ids.append(a_id)
                question_map[a_id] = q

        if not answer_ids:
            print(f"    (all {len(items)} already seen)")
            save_progress(page, total_saved)
            time.sleep(1.0)
            continue

        # Batch fetch answers
        answers = fetch_answers_batch(answer_ids)
        time.sleep(0.5)  # SE rate limit

        records_this_page = []
        for answer in answers:
            a_id = answer.get("answer_id", 0)
            question = question_map.get(a_id)
            if not question:
                continue

            if answer.get("score", 0) < args.min_answer_score:
                continue

            rec = build_training_record(question, answer, args.code_only)
            if rec:
                records_this_page.append(rec)
                seen_ids.add(rec["question_id"])
                if rec["has_bpy_code"]:
                    total_bpy += 1
                code_blocks = rec.get("code_blocks", [])
                if code_blocks:
                    total_code += 1

        save_records(records_this_page)
        total_saved += len(records_this_page)

        quota_remaining = data.get("quota_remaining", 9999)
        has_more = data.get("has_more", False)

        print(f"    +{len(records_this_page)} records | bpy: {total_bpy} | "
              f"code: {total_code} | total: {total_saved} | quota: {quota_remaining}")

        save_progress(page, total_saved)

        if not has_more:
            print("    [DONE] No more pages")
            break

        # SE allows 30 requests/second but recommend 1/second for polite scraping
        sleep_time = 1.5 if quota_remaining > 100 else 5.0
        time.sleep(sleep_time)

    print(f"\n=== SUMMARY ===")
    print(f"Total Q&A pairs saved: {total_saved}")
    print(f"  - With bpy code: {total_bpy}")
    print(f"  - With any code: {total_code}")
    print(f"Output: {SE_QA_FILE}")
    print(f"\nNext step: python synthesis/curriculum.py")


if __name__ == "__main__":
    main()
