"""
github_bpy_scripts.py - GitHub bpy (Blender Python API) script discovery and extraction.

Searches GitHub for repositories and files containing Blender Python API (bpy) code.
Extracts scripts with natural language descriptions to create (description, bpy_script) pairs.

Target: 50k+ verified bpy scripts.

Usage:
    python discovery/github_bpy_scripts.py --token YOUR_GITHUB_TOKEN
    python discovery/github_bpy_scripts.py --token YOUR_GITHUB_TOKEN --max-repos 5000
    python discovery/github_bpy_scripts.py --token YOUR_GITHUB_TOKEN --code-search
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Optional
import urllib.parse
import urllib.request
import urllib.error

DATA_DIR = Path(__file__).parents[1] / "data"
BPY_SCRIPTS_FILE = DATA_DIR / "bpy_scripts.jsonl"
BPY_REPOS_FILE = DATA_DIR / "bpy_repos.jsonl"

GH_BASE = "https://api.github.com"

# ─── GitHub search queries for bpy repos ─────────────────────────────────────
REPO_SEARCH_QUERIES = [
    "language:python blender bpy",
    "language:python topic:blender",
    "language:python topic:blender-addon",
    "language:python topic:blender-python",
    "language:python topic:bpy",
    "language:python blender addon",
    "language:python blender script automation",
    "language:python blender procedural",
    "language:python blender geometry-nodes",
    "language:python blender render automation",
    "language:python blender batch render",
    "language:python blender asset pipeline",
    "language:python blender rigify",
    "language:python blender animation tools",
    "language:python import bpy",
    "language:python bpy.ops modeling",
    "language:python bpy.data materials",
]

# ─── Code search queries (searches actual file contents) ─────────────────────
CODE_SEARCH_QUERIES = [
    "import bpy language:python",
    "bpy.ops.mesh language:python",
    "bpy.ops.object language:python",
    "bpy.data.objects language:python",
    "bpy.context.scene language:python",
    "bpy.ops.sculpt language:python",
    "bpy.ops.armature language:python",
    "bpy.ops.curve language:python",
    "bpy.types.Panel language:python",
    "bpy.types.Operator language:python",
    "bpy.props language:python",
    "bpy.ops.render language:python",
    "bpy.ops.material language:python",
    "bpy.ops.node language:python",
    "bpy.ops.sequencer language:python",
    "bpy.ops.particle language:python",
    "bpy.ops.cloth language:python",
    "bpy.ops.fluid language:python",
    "bpy.ops.rigidbody language:python",
]

# ─── Patterns to extract script intent ──────────────────────────────────────
DOCSTRING_PATTERN = re.compile(r'"""(.*?)"""', re.DOTALL)
SINGLE_DOCSTRING_PATTERN = re.compile(r"'''(.*?)'''", re.DOTALL)
COMMENT_PATTERN = re.compile(r"^#\s*(.+)$", re.MULTILINE)
BPY_IMPORT_PATTERN = re.compile(r"import bpy", re.MULTILINE)
CLASS_PATTERN = re.compile(r'class\s+(\w+)\s*\(.*\):\s*\n\s*"""(.*?)"""', re.DOTALL)
FUNCTION_PATTERN = re.compile(r'def\s+(\w+)\s*\([^)]*\):\s*\n\s*"""(.*?)"""', re.DOTALL)

# Minimum quality thresholds
MIN_SCRIPT_LINES = 10
MIN_BPY_OPS_CALLS = 2
MIN_STARS_FOR_REPO = 2


def gh_get(endpoint: str, params: dict, token: str) -> dict:
    """Make authenticated GitHub API request."""
    url = f"{GH_BASE}/{endpoint}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "nalana-bpy-harvester/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 403:
            # Rate limited — check headers
            reset_time = int(e.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(0, reset_time - time.time()) + 5
            print(f"    [RATE LIMIT] sleeping {wait:.0f}s")
            time.sleep(wait)
            return {}
        raise


def search_repos(query: str, token: str, max_pages: int = 10) -> list[dict]:
    """Search GitHub for repos matching query."""
    repos = []
    page = 1
    while page <= max_pages:
        data = gh_get(
            "search/repositories",
            {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": 100,
                "page": page,
            },
            token,
        )
        items = data.get("items", [])
        if not items:
            break
        repos.extend(items)
        if len(items) < 100:
            break
        page += 1
        time.sleep(0.5)
    return repos


def search_code(query: str, token: str, max_pages: int = 10) -> list[dict]:
    """Search GitHub code for bpy patterns."""
    results = []
    page = 1
    while page <= max_pages:
        data = gh_get(
            "search/code",
            {
                "q": query,
                "per_page": 100,
                "page": page,
            },
            token,
        )
        items = data.get("items", [])
        if not items:
            break
        results.extend(items)
        if len(items) < 100:
            break
        page += 1
        time.sleep(1.0)  # code search has tighter rate limits
    return results


def get_repo_python_files(
    owner: str, repo: str, token: str, path: str = ""
) -> list[dict]:
    """Get all Python files in a repository."""
    files = []
    try:
        data = gh_get(f"repos/{owner}/{repo}/contents/{path}", {}, token)
    except Exception:
        return []

    if isinstance(data, list):
        for item in data:
            if item.get("type") == "file" and item.get("name", "").endswith(".py"):
                files.append(item)
            elif item.get("type") == "dir" and not item.get("name", "").startswith("."):
                # Recurse into subdirectories (limit depth to avoid huge repos)
                sub_files = get_repo_python_files(
                    owner, repo, token, item.get("path", "")
                )
                files.extend(sub_files[:50])  # cap per directory
    elif isinstance(data, dict):
        if data.get("name", "").endswith(".py"):
            files.append(data)
    return files[:200]  # cap total files per repo


def download_file_content(download_url: str, token: str) -> Optional[str]:
    """Download raw file content from GitHub."""
    req = urllib.request.Request(
        download_url,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "nalana-bpy-harvester/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def is_valid_bpy_script(content: str) -> bool:
    """Check if a file is a meaningful bpy script worth including in training data."""
    if not BPY_IMPORT_PATTERN.search(content):
        return False
    lines = [line for line in content.split("\n") if line.strip()]
    if len(lines) < MIN_SCRIPT_LINES:
        return False
    # Count bpy API calls
    bpy_calls = len(re.findall(r"bpy\.(ops|data|context|types|utils|props)", content))
    if bpy_calls < MIN_BPY_OPS_CALLS:
        return False
    return True


def extract_script_description(content: str, filename: str, repo_name: str) -> str:
    """
    Extract a natural language description of what the script does.
    Priority: module docstring > class docstring > top comments > filename inference.
    """
    # Try module-level docstring (first triple-quoted string)
    docstring_match = DOCSTRING_PATTERN.search(content[:3000])
    if docstring_match:
        desc = docstring_match.group(1).strip()
        if len(desc) > 20:
            return desc[:500]

    single_match = SINGLE_DOCSTRING_PATTERN.search(content[:3000])
    if single_match:
        desc = single_match.group(1).strip()
        if len(desc) > 20:
            return desc[:500]

    # Try first block of comments
    comment_lines = []
    for line in content.split("\n")[:20]:
        stripped = line.strip()
        if stripped.startswith("#") and not stripped.startswith("#!"):
            comment_lines.append(stripped[1:].strip())
        elif comment_lines:
            break
    if len(comment_lines) >= 2:
        return " ".join(comment_lines)[:500]

    # Infer from filename and repo
    name_parts = filename.replace(".py", "").replace("_", " ").replace("-", " ")
    return f"Blender Python script: {name_parts} (from {repo_name})"


def extract_function_pairs(content: str) -> list[dict]:
    """Extract individual function-level (description, code) pairs from a script."""
    pairs = []
    # Match functions with docstrings
    for match in FUNCTION_PATTERN.finditer(content):
        func_name = match.group(1)
        docstring = match.group(2).strip()
        if len(docstring) < 10:
            continue
        # Extract the full function body
        start = match.start()
        # Find function end (next function at same indent or EOF)
        func_end = len(content)
        next_func = re.search(r"\ndef\s+", content[start + 10 :])
        if next_func:
            func_end = start + 10 + next_func.start()
        func_code = content[start:func_end].strip()
        if len(func_code) > 100:
            pairs.append(
                {
                    "description": docstring[:400],
                    "code": func_code[:3000],
                    "granularity": "function",
                    "function_name": func_name,
                }
            )
    return pairs


def process_bpy_file(
    content: str,
    filename: str,
    repo_name: str,
    repo_url: str,
    stars: int,
) -> list[dict]:
    """Process a single Python file and extract all training pairs."""
    if not is_valid_bpy_script(content):
        return []

    records = []

    # Whole-file pair
    description = extract_script_description(content, filename, repo_name)
    records.append(
        {
            "type": "whole_script",
            "description": description,
            "code": content[:8000],  # cap at 8k chars
            "filename": filename,
            "repo": repo_name,
            "repo_url": repo_url,
            "stars": stars,
            "granularity": "script",
        }
    )

    # Function-level pairs
    func_pairs = extract_function_pairs(content)
    for pair in func_pairs:
        pair["filename"] = filename
        pair["repo"] = repo_name
        pair["repo_url"] = repo_url
        pair["stars"] = stars
        pair["type"] = "function_pair"
        records.append(pair)

    return records


def load_seen_repos() -> set[str]:
    """Load already-processed repo full names."""
    if not BPY_REPOS_FILE.exists():
        return set()
    seen = set()
    with open(BPY_REPOS_FILE) as f:
        for line in f:
            try:
                rec = json.loads(line)
                seen.add(rec.get("full_name", ""))
            except json.JSONDecodeError:
                pass
    return seen


def load_seen_scripts() -> int:
    """Count already-extracted scripts."""
    if not BPY_SCRIPTS_FILE.exists():
        return 0
    count = 0
    with open(BPY_SCRIPTS_FILE) as f:
        for _ in f:
            count += 1
    return count


def save_repo_marker(repo: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(BPY_REPOS_FILE, "a") as f:
        f.write(
            json.dumps(
                {
                    "full_name": repo.get("full_name"),
                    "stars": repo.get("stargazers_count"),
                    "url": repo.get("html_url"),
                    "description": repo.get("description", ""),
                }
            )
            + "\n"
        )


def save_records(records: list[dict]) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(BPY_SCRIPTS_FILE, "a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return len(records)


def process_repo(repo: dict, token: str) -> int:
    """Process a single repo: find all bpy Python files and extract training pairs."""
    owner = repo.get("owner", {}).get("login", "")
    repo_name = repo.get("name", "")
    full_name = repo.get("full_name", "")
    stars = repo.get("stargazers_count", 0)
    repo_url = repo.get("html_url", "")

    if stars < MIN_STARS_FOR_REPO:
        return 0

    py_files = get_repo_python_files(owner, repo_name, token)
    total_records = 0

    for file_info in py_files:
        download_url = file_info.get("download_url")
        if not download_url:
            continue
        content = download_file_content(download_url, token)
        if not content:
            continue
        records = process_bpy_file(
            content,
            file_info.get("name", "unknown.py"),
            full_name,
            repo_url,
            stars,
        )
        total_records += save_records(records)
        time.sleep(0.05)  # gentle pacing

    return total_records


def main():
    parser = argparse.ArgumentParser(description="Harvest bpy scripts from GitHub")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""))
    parser.add_argument(
        "--max-repos",
        type=int,
        default=3000,
        help="Max repos to process per search query",
    )
    parser.add_argument(
        "--code-search",
        action="store_true",
        help="Also run code-level search (slower, finds more files)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip already-processed repos"
    )
    args = parser.parse_args()

    if not args.token:
        print("Error: provide --token or set GITHUB_TOKEN")
        return

    seen_repos = load_seen_repos() if args.resume else set()
    total_scripts = load_seen_scripts()
    print(
        f"Resuming from {len(seen_repos)} seen repos, {total_scripts} existing scripts\n"
    )

    # ── Phase 1: Repo-level search ────────────────────────────────────────────
    print("=== PHASE 1: REPO SEARCH ===")
    all_repos: dict[str, dict] = {}

    for query in REPO_SEARCH_QUERIES:
        print(f"  Searching: {query}")
        repos = search_repos(query, args.token)
        for r in repos:
            fn = r.get("full_name", "")
            if fn not in all_repos:
                all_repos[fn] = r
        print(f"    Found {len(repos)} repos (total unique: {len(all_repos)})")
        time.sleep(0.5)

    print(f"\nTotal unique repos discovered: {len(all_repos)}")

    # ── Phase 2: Process repos ────────────────────────────────────────────────
    print("\n=== PHASE 2: PROCESSING REPOS ===")
    processed = 0
    total_new_records = 0

    for full_name, repo in sorted(
        all_repos.items(),
        key=lambda x: x[1].get("stargazers_count", 0),
        reverse=True,
    )[: args.max_repos]:
        if full_name in seen_repos:
            continue
        count = process_repo(repo, args.token)
        save_repo_marker(repo)
        seen_repos.add(full_name)
        total_new_records += count
        processed += 1
        print(
            f"  [{processed}] {full_name} ({repo.get('stargazers_count', 0)} stars) → {count} records"
        )
        time.sleep(0.2)

    # ── Phase 3: Code-level search ────────────────────────────────────────────
    if args.code_search:
        print("\n=== PHASE 3: CODE SEARCH ===")
        for query in CODE_SEARCH_QUERIES:
            print(f"  Code search: {query}")
            results = search_code(query, args.token)
            for item in results:
                repo_data = item.get("repository", {})
                full_name = repo_data.get("full_name", "")
                if full_name in seen_repos:
                    continue
                item.get("url", "")
                # Code search returns API URL, need raw URL
                raw_url = (
                    item.get("html_url", "")
                    .replace("github.com", "raw.githubusercontent.com")
                    .replace("/blob/", "/")
                )
                content = download_file_content(raw_url, args.token)
                if content:
                    records = process_bpy_file(
                        content,
                        item.get("name", "script.py"),
                        full_name,
                        repo_data.get("html_url", ""),
                        repo_data.get("stargazers_count", 0),
                    )
                    total_new_records += save_records(records)
                time.sleep(0.2)
            time.sleep(1.0)

    final_count = load_seen_scripts()
    print(f"\nNew records this run: {total_new_records}")
    print(f"Total bpy training pairs in {BPY_SCRIPTS_FILE}: {final_count}")
    print("\nNext step: python synthesis/curriculum.py")


if __name__ == "__main__":
    main()
