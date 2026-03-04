"""
api_harvest.py - Harvests 3D models from public APIs for the Nalana dataset.

Sources:
  - Sketchfab     (CC-licensed GLB models)
  - Polyhaven     (CC0 HDRI, textures, models)
  - Thingiverse   (printable STL/OBJ)
  - Smithsonian   (museum artifacts, CC0 GLB)
  - NIH 3D Print  (medical/scientific, public domain)

Each source downloads model files + metadata to data/models/{source}/
Metadata is written to data/models/{source}/metadata.jsonl

Usage:
    python api_harvest.py --all
    python api_harvest.py --sketchfab --limit 5000 --categories furniture,vehicles,architecture
    python api_harvest.py --polyhaven --all-hdri
    python api_harvest.py --thingiverse --limit 10000
    python api_harvest.py --smithsonian --limit 2000
    python api_harvest.py --nih --limit 1000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterator

import httpx
from dotenv import load_dotenv

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parents[1]
MODELS_DIR = BASE_DIR / "data" / "models"

# ─── Default categories ───────────────────────────────────────────────────────

DEFAULT_CATEGORIES = [
    "furniture",
    "vehicles",
    "architecture",
    "characters",
    "nature",
    "weapons",
    "electronics",
    "food",
    "animals",
    "clothing",
    "tools",
    "buildings",
    "plants",
    "environments",
    "props",
]

# ─── Helpers ──────────────────────────────────────────────────────────────────


def file_hash(path: Path) -> str:
    """SHA-256 of file contents for deduplication."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_seen_hashes(source_dir: Path) -> set[str]:
    """Load all hashes of already-downloaded files to skip duplicates."""
    hash_file = source_dir / ".hashes"
    if not hash_file.exists():
        return set()
    return set(hash_file.read_text().splitlines())


def save_hash(source_dir: Path, digest: str) -> None:
    with (source_dir / ".hashes").open("a") as f:
        f.write(digest + "\n")


def append_metadata(source_dir: Path, record: dict) -> None:
    with (source_dir / "metadata.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")


def pbar(iterable, total: int, desc: str):
    if HAS_TQDM:
        return tqdm(iterable, total=total, desc=desc, unit="model")
    return iterable


def retry_get(
    client: httpx.Client,
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    retries: int = 3,
    backoff: float = 2.0,
) -> httpx.Response:
    for attempt in range(retries):
        try:
            resp = client.get(url, params=params, headers=headers, timeout=30)
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", backoff * (attempt + 1)))
                print(f"    Rate limited — waiting {wait:.0f}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except (httpx.HTTPError, httpx.TimeoutException):
            if attempt == retries - 1:
                raise
            time.sleep(backoff * (2**attempt))
    raise RuntimeError(f"Failed after {retries} retries: {url}")


def download_file(client: httpx.Client, url: str, dest: Path, retries: int = 3) -> bool:
    """Stream-download a file. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            with client.stream("GET", url, timeout=120, follow_redirects=True) as r:
                r.raise_for_status()
                with dest.open("wb") as f:
                    for chunk in r.iter_bytes(chunk_size=65536):
                        f.write(chunk)
            return True
        except Exception as e:
            if attempt == retries - 1:
                print(f"    Download failed: {url} — {e}")
                return False
            time.sleep(2**attempt)
    return False


# ─── Sketchfab harvester ──────────────────────────────────────────────────────

SKETCHFAB_API = "https://api.sketchfab.com/v3"
SKETCHFAB_CATEGORIES = {
    "furniture": "furniture-home",
    "vehicles": "vehicles-transportation",
    "architecture": "architecture",
    "characters": "characters-creatures",
    "nature": "nature-plants",
    "weapons": "weapons-military",
    "electronics": "electronics-gadgets",
    "food": "food-drink",
    "animals": "animals-pets",
    "buildings": "architectural-elements",
    "tools": "science-technology",
    "environments": "places-travel",
}


def sketchfab_search_page(
    client: httpx.Client,
    api_key: str,
    category: str,
    cursor: str | None = None,
    page_size: int = 24,
) -> tuple[list[dict], str | None]:
    """Fetch one page of Sketchfab search results. Returns (models, next_cursor)."""
    params: dict[str, Any] = {
        "type": "models",
        "downloadable": True,
        "license": "cc",  # All CC variants
        "categories": SKETCHFAB_CATEGORIES.get(category, category),
        "count": page_size,
        "sort_by": "-downloadCount",
        "file_format": "glb",
    }
    if cursor:
        params["cursor"] = cursor

    resp = retry_get(
        client,
        f"{SKETCHFAB_API}/search",
        params=params,
        headers={"Authorization": f"Token {api_key}"},
    )
    data = resp.json()
    results = data.get("results", [])
    cursors = data.get("cursors", {})
    return results, cursors.get("next")


def sketchfab_get_download_url(
    client: httpx.Client, api_key: str, uid: str
) -> str | None:
    """Request a download URL for a Sketchfab model (requires auth)."""
    try:
        resp = retry_get(
            client,
            f"{SKETCHFAB_API}/models/{uid}/download",
            headers={"Authorization": f"Token {api_key}"},
        )
        data = resp.json()
        glb = data.get("glb") or data.get("source") or {}
        return glb.get("url")
    except Exception as e:
        print(f"    Download URL fetch failed for {uid}: {e}")
        return None


def harvest_sketchfab(limit: int, categories: list[str]) -> None:
    api_key = os.environ.get("SKETCHFAB_API_KEY", "")
    if not api_key:
        print("[Sketchfab] SKETCHFAB_API_KEY not set in .env — skipping.")
        return

    source_dir = MODELS_DIR / "sketchfab"
    source_dir.mkdir(parents=True, exist_ok=True)
    seen_hashes = load_seen_hashes(source_dir)
    seen_uids: set[str] = set()

    total_saved = 0
    per_cat = max(1, limit // len(categories))

    with httpx.Client(follow_redirects=True) as client:
        for category in categories:
            print(f"\n[Sketchfab] Category: {category} (target {per_cat})")
            cat_dir = source_dir / category
            cat_dir.mkdir(parents=True, exist_ok=True)
            saved_in_cat = 0
            cursor = None

            while saved_in_cat < per_cat and total_saved < limit:
                try:
                    results, cursor = sketchfab_search_page(
                        client, api_key, category, cursor=cursor
                    )
                except Exception as e:
                    print(f"  Search error: {e}")
                    break

                if not results:
                    break

                for model in results:
                    if saved_in_cat >= per_cat or total_saved >= limit:
                        break

                    uid = model.get("uid", "")
                    if uid in seen_uids:
                        continue
                    seen_uids.add(uid)

                    name = model.get("name", uid)
                    license_ = model.get("license", {}).get("label", "cc-by")
                    poly_count = (
                        (model.get("faceCount") or 0) + (model.get("vertexCount") or 0)
                    ) // 2 or None

                    # Get download URL
                    dl_url = sketchfab_get_download_url(client, api_key, uid)
                    if not dl_url:
                        continue

                    dest = cat_dir / f"{uid}.glb"
                    if not download_file(client, dl_url, dest):
                        continue

                    # Dedup by hash
                    digest = file_hash(dest)
                    if digest in seen_hashes:
                        dest.unlink(missing_ok=True)
                        continue
                    seen_hashes.add(digest)
                    save_hash(source_dir, digest)

                    # Metadata
                    meta = {
                        "uid": uid,
                        "name": name,
                        "category": category,
                        "poly_count": poly_count,
                        "format": "glb",
                        "license": license_,
                        "source": "sketchfab",
                        "file": str(dest.relative_to(BASE_DIR)),
                        "url": f"https://sketchfab.com/3d-models/{uid}",
                    }
                    append_metadata(source_dir, meta)
                    saved_in_cat += 1
                    total_saved += 1

                    if HAS_TQDM:
                        print(f"  [{total_saved}/{limit}] {name[:60]}")

                if not cursor:
                    break
                time.sleep(0.5)  # Polite pause between pages

    print(f"\n[Sketchfab] Done. Saved {total_saved} models.")


# ─── Polyhaven harvester ──────────────────────────────────────────────────────

POLYHAVEN_API = "https://api.polyhaven.com"


def polyhaven_iter_assets(
    client: httpx.Client, asset_type: str
) -> Iterator[tuple[str, dict]]:
    """Yield (asset_id, asset_info) for all Polyhaven assets of given type."""
    resp = retry_get(client, f"{POLYHAVEN_API}/assets", params={"type": asset_type})
    for asset_id, info in resp.json().items():
        yield asset_id, info


def polyhaven_get_files(client: httpx.Client, asset_id: str) -> dict:
    resp = retry_get(client, f"{POLYHAVEN_API}/files/{asset_id}")
    return resp.json()


def harvest_polyhaven(
    all_hdri: bool = True, models: bool = True, textures: bool = False
) -> None:
    source_dir = MODELS_DIR / "polyhaven"
    source_dir.mkdir(parents=True, exist_ok=True)
    seen_hashes = load_seen_hashes(source_dir)
    total_saved = 0

    asset_types = []
    if all_hdri:
        asset_types.append("hdris")
    if models:
        asset_types.append("models")
    if textures:
        asset_types.append("textures")

    with httpx.Client(follow_redirects=True) as client:
        for asset_type in asset_types:
            print(f"\n[Polyhaven] Downloading {asset_type}...")
            type_dir = source_dir / asset_type
            type_dir.mkdir(parents=True, exist_ok=True)

            assets = list(polyhaven_iter_assets(client, asset_type))
            bar = tqdm(assets, desc=f"polyhaven/{asset_type}") if HAS_TQDM else assets

            for asset_id, info in bar:
                try:
                    files_data = polyhaven_get_files(client, asset_id)
                except Exception as e:
                    print(f"  Files fetch failed for {asset_id}: {e}")
                    continue

                # Pick best format
                if asset_type == "hdris":
                    # Download 2K EXR
                    url = (
                        files_data.get("hdri", {})
                        .get("2k", {})
                        .get("exr", {})
                        .get("url")
                    )
                    ext = "exr"
                elif asset_type == "models":
                    # Download GLB if available, else OBJ
                    glb_info = files_data.get("blend", {}).get("1k", {}).get(
                        "glb"
                    ) or files_data.get("gltf", {}).get("1k", {}).get("glb")
                    url = glb_info.get("url") if isinstance(glb_info, dict) else None
                    ext = "glb"
                    if not url:
                        # Fallback to blend
                        url = (
                            files_data.get("blend", {})
                            .get("1k", {})
                            .get("blend", {})
                            .get("url")
                        )
                        ext = "blend"
                elif asset_type == "textures":
                    url = (
                        files_data.get("nor_gl", {})
                        .get("1k", {})
                        .get("png", {})
                        .get("url")
                    )
                    ext = "png"

                if not url:
                    continue

                dest = type_dir / f"{asset_id}.{ext}"
                if dest.exists():
                    continue

                if not download_file(client, url, dest):
                    continue

                digest = file_hash(dest)
                if digest in seen_hashes:
                    dest.unlink(missing_ok=True)
                    continue
                seen_hashes.add(digest)
                save_hash(source_dir, digest)

                meta = {
                    "uid": asset_id,
                    "name": info.get("name", asset_id),
                    "category": info.get("categories", ["unknown"])[0]
                    if info.get("categories")
                    else "unknown",
                    "poly_count": None,
                    "format": ext,
                    "license": "CC0",
                    "source": "polyhaven",
                    "type": asset_type,
                    "file": str(dest.relative_to(BASE_DIR)),
                    "url": f"https://polyhaven.com/a/{asset_id}",
                }
                append_metadata(source_dir, meta)
                total_saved += 1
                time.sleep(0.1)

    print(f"\n[Polyhaven] Done. Saved {total_saved} assets.")


# ─── Thingiverse harvester ────────────────────────────────────────────────────

THINGIVERSE_API = "https://api.thingiverse.com"
THINGIVERSE_CATEGORIES = [
    "3D Printing",
    "Art",
    "Fashion",
    "Gadgets",
    "Hobby",
    "Household",
    "Learning",
    "Models",
    "Tools",
    "Toys & Games",
]


def thingiverse_search_page(
    client: httpx.Client, api_key: str, category: str, page: int = 1, per_page: int = 30
) -> list[dict]:
    params = {
        "q": category,
        "type": "things",
        "page": page,
        "per_page": per_page,
        "sort": "popular",
        "posted_after": "2018-01-01",
    }
    resp = retry_get(
        client,
        f"{THINGIVERSE_API}/search/{category}",
        params=params,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    return resp.json().get("hits", [])


def thingiverse_get_files(
    client: httpx.Client, api_key: str, thing_id: int
) -> list[dict]:
    resp = retry_get(
        client,
        f"{THINGIVERSE_API}/things/{thing_id}/files",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    return resp.json()


def harvest_thingiverse(limit: int) -> None:
    api_key = os.environ.get("THINGIVERSE_API_KEY", "")
    if not api_key:
        print("[Thingiverse] THINGIVERSE_API_KEY not set in .env — skipping.")
        return

    source_dir = MODELS_DIR / "thingiverse"
    source_dir.mkdir(parents=True, exist_ok=True)
    seen_hashes = load_seen_hashes(source_dir)
    seen_ids: set[int] = set()
    total_saved = 0
    per_cat = max(1, limit // len(THINGIVERSE_CATEGORIES))

    with httpx.Client(follow_redirects=True) as client:
        for category in THINGIVERSE_CATEGORIES:
            print(f"\n[Thingiverse] Category: {category} (target {per_cat})")
            cat_dir = source_dir / category.lower().replace(" ", "_").replace(
                "&", "and"
            )
            cat_dir.mkdir(parents=True, exist_ok=True)
            saved_in_cat = 0
            page = 1

            while saved_in_cat < per_cat and total_saved < limit:
                try:
                    things = thingiverse_search_page(
                        client, api_key, category, page=page
                    )
                except Exception as e:
                    print(f"  Search error page {page}: {e}")
                    break

                if not things:
                    break

                for thing in things:
                    if saved_in_cat >= per_cat or total_saved >= limit:
                        break

                    thing_id = thing.get("id") or thing.get("thing_id")
                    if not thing_id or thing_id in seen_ids:
                        continue
                    seen_ids.add(thing_id)

                    # Get file list
                    try:
                        files = thingiverse_get_files(client, api_key, thing_id)
                    except Exception as e:
                        print(f"  File list failed for {thing_id}: {e}")
                        continue

                    # Prefer STL, then OBJ
                    target_file = None
                    for fmt_pref in (".stl", ".obj", ".3mf"):
                        for f in files:
                            if f.get("name", "").lower().endswith(fmt_pref):
                                target_file = f
                                break
                        if target_file:
                            break

                    if not target_file:
                        continue

                    dl_url = target_file.get("direct_url") or target_file.get(
                        "download_url"
                    )
                    if not dl_url:
                        continue

                    ext = Path(target_file["name"]).suffix.lstrip(".")
                    dest = cat_dir / f"{thing_id}.{ext}"
                    if not download_file(client, dl_url, dest):
                        continue

                    digest = file_hash(dest)
                    if digest in seen_hashes:
                        dest.unlink(missing_ok=True)
                        continue
                    seen_hashes.add(digest)
                    save_hash(source_dir, digest)

                    meta = {
                        "uid": str(thing_id),
                        "name": thing.get("name", str(thing_id)),
                        "category": category,
                        "poly_count": None,
                        "format": ext,
                        "license": thing.get("license", "cc-by"),
                        "source": "thingiverse",
                        "file": str(dest.relative_to(BASE_DIR)),
                        "url": f"https://www.thingiverse.com/thing:{thing_id}",
                    }
                    append_metadata(source_dir, meta)
                    saved_in_cat += 1
                    total_saved += 1
                    print(f"  [{total_saved}/{limit}] {meta['name'][:60]}")

                page += 1
                time.sleep(1.0)

    print(f"\n[Thingiverse] Done. Saved {total_saved} models.")


# ─── Smithsonian harvester ────────────────────────────────────────────────────

SMITHSONIAN_API = "https://api.si.edu/openaccess/api/v1.0"


def smithsonian_search_page(
    client: httpx.Client, api_key: str, query: str, start: int = 0, rows: int = 20
) -> list[dict]:
    params = {
        "q": query,
        "api_key": api_key,
        "start": start,
        "rows": rows,
        "type": "3d_package",
        "media.type": "3d_package",
    }
    try:
        resp = retry_get(client, f"{SMITHSONIAN_API}/search", params=params)
        return resp.json().get("response", {}).get("rows", [])
    except Exception:
        return []


def smithsonian_get_glb_url(media: list[dict]) -> str | None:
    for item in media:
        if item.get("type") == "3d_package":
            for resource in item.get("resources", []):
                url = resource.get("url", "")
                if url.lower().endswith(".glb"):
                    return url
    return None


def harvest_smithsonian(limit: int) -> None:
    api_key = os.environ.get("SMITHSONIAN_API_KEY", "")
    if not api_key:
        print("[Smithsonian] SMITHSONIAN_API_KEY not set — trying unauthenticated.")

    source_dir = MODELS_DIR / "smithsonian"
    source_dir.mkdir(parents=True, exist_ok=True)
    seen_hashes = load_seen_hashes(source_dir)
    total_saved = 0

    queries = [
        "artifact sculpture",
        "ancient pottery",
        "fossil specimen",
        "natural history",
        "historical instrument",
        "gemstone mineral",
        "cultural artifact",
        "architectural model",
        "industrial object",
    ]

    with httpx.Client(follow_redirects=True) as client:
        for query in queries:
            if total_saved >= limit:
                break
            print(f"\n[Smithsonian] Query: {query}")
            start = 0

            while total_saved < limit:
                rows = smithsonian_search_page(client, api_key, query, start=start)
                if not rows:
                    break

                for row in rows:
                    if total_saved >= limit:
                        break

                    uid = row.get("id", "")
                    title = row.get("title", uid)
                    media = row.get("_media", {})

                    # Extract 3D package
                    packages = media.get("3d", []) if isinstance(media, dict) else []
                    if not packages:
                        # Try alternate path
                        content = row.get("content", {})
                        descriptive = content.get("descriptiveNonRepeating", {})
                        packages = descriptive.get("online_media", {}).get("media", [])

                    glb_url = smithsonian_get_glb_url(
                        packages if isinstance(packages, list) else []
                    )
                    if not glb_url:
                        continue

                    dest = source_dir / f"{uid.replace('/', '_')}.glb"
                    if not download_file(client, glb_url, dest):
                        continue

                    digest = file_hash(dest)
                    if digest in seen_hashes:
                        dest.unlink(missing_ok=True)
                        continue
                    seen_hashes.add(digest)
                    save_hash(source_dir, digest)

                    # Extract unit/department info
                    freetext = row.get("content", {}).get("freetext", {})
                    cat = (
                        freetext.get("setName", [{}])[0].get("content", "artifact")
                        if freetext.get("setName")
                        else "artifact"
                    )

                    meta = {
                        "uid": uid,
                        "name": title,
                        "category": cat,
                        "poly_count": None,
                        "format": "glb",
                        "license": "CC0",
                        "source": "smithsonian",
                        "file": str(dest.relative_to(BASE_DIR)),
                        "url": f"https://3d.si.edu/object/{uid}",
                    }
                    append_metadata(source_dir, meta)
                    total_saved += 1
                    print(f"  [{total_saved}/{limit}] {title[:60]}")

                start += len(rows)
                time.sleep(0.5)

    print(f"\n[Smithsonian] Done. Saved {total_saved} models.")


# ─── NIH 3D Print Exchange harvester ─────────────────────────────────────────

NIH_API = "https://3dprint.nih.gov/api"
NIH_CATEGORIES = [
    "anatomy",
    "biology",
    "chemistry",
    "medicine",
    "neuroscience",
    "paleontology",
    "physiology",
    "surgery",
    "education",
]


def nih_search_page(
    client: httpx.Client, category: str, page: int = 1, per_page: int = 20
) -> list[dict]:
    params = {
        "type": "model",
        "keywords": category,
        "page": page,
        "items_per_page": per_page,
        "sort": "downloads",
        "order": "desc",
    }
    try:
        resp = retry_get(client, f"{NIH_API}/search", params=params)
        return resp.json()
    except Exception:
        return []


def nih_get_model_detail(client: httpx.Client, nid: str) -> dict:
    try:
        resp = retry_get(client, f"{NIH_API}/node/{nid}")
        return resp.json()
    except Exception:
        return {}


def harvest_nih(limit: int) -> None:
    source_dir = MODELS_DIR / "nih"
    source_dir.mkdir(parents=True, exist_ok=True)
    seen_hashes = load_seen_hashes(source_dir)
    seen_nids: set[str] = set()
    total_saved = 0
    per_cat = max(1, limit // len(NIH_CATEGORIES))

    with httpx.Client(follow_redirects=True) as client:
        for category in NIH_CATEGORIES:
            if total_saved >= limit:
                break
            print(f"\n[NIH] Category: {category}")
            cat_dir = source_dir / category
            cat_dir.mkdir(parents=True, exist_ok=True)
            saved_in_cat = 0
            page = 1

            while saved_in_cat < per_cat and total_saved < limit:
                results = nih_search_page(client, category, page=page)
                if not results:
                    break

                for item in results:
                    if saved_in_cat >= per_cat or total_saved >= limit:
                        break

                    nid = str(item.get("nid", item.get("id", "")))
                    if not nid or nid in seen_nids:
                        continue
                    seen_nids.add(nid)

                    title = item.get("title", nid)

                    # Get download links from detail page
                    detail = nih_get_model_detail(client, nid)
                    files = detail.get("files", [])
                    if not files and isinstance(detail, list):
                        files = detail

                    dl_url = None
                    ext = None
                    for f in files if isinstance(files, list) else []:
                        fname = (f.get("filename") or f.get("name") or "").lower()
                        furl = f.get("url") or f.get("file_url") or ""
                        for fmt in (".stl", ".obj", ".glb", ".3mf"):
                            if fname.endswith(fmt) or fmt in furl.lower():
                                dl_url = furl
                                ext = fmt.lstrip(".")
                                break
                        if dl_url:
                            break

                    if not dl_url:
                        # Try constructing from known patterns
                        dl_url = f"https://3dprint.nih.gov/discover/{nid}"
                        ext = "stl"

                    dest = cat_dir / f"{nid}.{ext}"
                    if not download_file(client, dl_url, dest):
                        continue

                    if not dest.exists() or dest.stat().st_size == 0:
                        dest.unlink(missing_ok=True)
                        continue

                    digest = file_hash(dest)
                    if digest in seen_hashes:
                        dest.unlink(missing_ok=True)
                        continue
                    seen_hashes.add(digest)
                    save_hash(source_dir, digest)

                    meta = {
                        "uid": nid,
                        "name": title,
                        "category": category,
                        "poly_count": None,
                        "format": ext,
                        "license": "public_domain",
                        "source": "nih",
                        "file": str(dest.relative_to(BASE_DIR)),
                        "url": f"https://3dprint.nih.gov/discover/{nid}",
                    }
                    append_metadata(source_dir, meta)
                    saved_in_cat += 1
                    total_saved += 1
                    print(f"  [{total_saved}/{limit}] {title[:60]}")

                page += 1
                time.sleep(0.5)

    print(f"\n[NIH] Done. Saved {total_saved} models.")


# ─── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harvest 3D models from public APIs for the Nalana dataset"
    )
    parser.add_argument("--all", action="store_true", help="Run all harvesters")
    parser.add_argument("--sketchfab", action="store_true")
    parser.add_argument("--polyhaven", action="store_true")
    parser.add_argument("--thingiverse", action="store_true")
    parser.add_argument("--smithsonian", action="store_true")
    parser.add_argument("--nih", action="store_true")

    parser.add_argument(
        "--categories",
        default=",".join(DEFAULT_CATEGORIES),
        help="Comma-separated list of categories (for Sketchfab/Thingiverse)",
    )
    parser.add_argument("--limit", type=int, default=1000, help="Max models per source")
    parser.add_argument(
        "--all-hdri", action="store_true", help="Download all Polyhaven HDRIs"
    )
    parser.add_argument(
        "--ph-textures", action="store_true", help="Also download Polyhaven textures"
    )

    args = parser.parse_args()
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]

    run_all = args.all

    if run_all or args.sketchfab:
        harvest_sketchfab(limit=args.limit, categories=cats)

    if run_all or args.polyhaven:
        harvest_polyhaven(
            all_hdri=args.all_hdri or run_all,
            models=True,
            textures=args.ph_textures,
        )

    if run_all or args.thingiverse:
        harvest_thingiverse(limit=args.limit)

    if run_all or args.smithsonian:
        harvest_smithsonian(limit=args.limit)

    if run_all or args.nih:
        harvest_nih(limit=args.limit)

    print("\nAll harvesters complete.")
    print(f"Models saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
