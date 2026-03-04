"""
discover_v2.py - Comprehensive YouTube tutorial discovery at scale.

Two modes:
  1. Channel crawl (--channels): Gets ALL videos from top 3D software channels.
     Cost: 1 API unit per 50 videos (vs 100 units per search query = 25x cheaper).
     Realistic yield: 15,000-30,000 video IDs from ~60 channels.

  2. Search (--search): Keyword search across 200+ targeted queries covering
     every 3D application and operation category.
     Cost: 100 units per query. Use sparingly to fill gaps.

  3. All-software mode (--all-software): Adds cross-software queries for
     Maya, Cinema 4D, Houdini, ZBrush, Substance, Rhino, Unreal, Fusion 360,
     3ds Max, SketchUp, plus design/physics theory queries.

Output: data/video_ids.txt — one video ID per line, deduplicated.

Usage:
    python discover_v2.py --api-key KEY --channels          # crawl all known channels
    python discover_v2.py --api-key KEY --search            # keyword search mode
    python discover_v2.py --api-key KEY --channels --search # both
    python discover_v2.py --api-key KEY --all-software      # cross-software queries
    python discover_v2.py --api-key KEY --channels --search --all-software --filter-quality
"""

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parents[1] / "data"
VIDEO_IDS_FILE = DATA_DIR / "video_ids.txt"

YT_BASE = "https://www.googleapis.com/youtube/v3"

# ─── Top Blender channels ──────────────────────────────────────────────────────
# Channel IDs. Uploads playlist = "UU" + channel_id[2:]
# These channels have dense, narrated tutorial content ideal for training.
ALL_CHANNELS = {
    # ── Blender: Core modeling & fundamentals ─────────────────────────────────
    "blender_guru": "UCOKHwx1VCdgnxwbjyb9Iu1g",  # ~500 videos
    "grant_abbitt": "UCZFUrFoqvqlN8seaAeEwjlw",  # ~400 videos
    "cg_fast_track": "UCsvn_Po0SmunchJYtttWpVDg",  # ~200 videos
    "default_cube": "UCdpWKLNfbROyoGPV46-zaUQ",  # geometry nodes
    "cg_matter": "UCy1f4m64dwCwk8CBZ_vHfPg",  # tips & tricks
    "ducky_3d": "UCuNhGhbemBkdflZ1FGJ0lUQ",  # motion & shader
    "imphenzia": "UCuBqjbaKfOha-n_RaYAZJ0Q",  # low poly
    "josh_gambrell": "UCM-CR_NVlFz6MEOznvpJ0Mw",  # hard surface
    "ponte_ryuurui": "UCCpWXmLNOKyBL5i3yJilJHA",  # hard surface
    "steven_scott": "UCiy-QcXl-6JFjc9PN_QSdaQ",  # scatter & GN
    "southernshotty": "UCTQwQ7mPgdDrpJY1BZKFMCA",  # hard surface
    "artisans_of_vaul": "UCHu0cHGSGBxSgsFBEPBGjtw",  # sculpting
    "yansculpts": "UCFmHIftfI9HRaDP_5ezojyw",  # sculpting
    "ryan_king_art": "UCvvVcAC-6k6k0tQ1MKjvZEA",  # stylized
    "ian_hubert": "UCHcgbM8Y5JFrQfLkZ_LoZWQ",  # environments
    "cg_boost": "UC2U5mGEDkDrpvCVsaEQ1ISg",  # beginner courses — Fixed: was duplicate of ducky_3d
    "crossmind_studio": "UCwdybBYXkAaZlDQPV1RzVCg",  # Blender tips
    "level_pixel_level": "UCUESe_jQEMHiPVMXB-UeqoQ",  # procedural techniques
    "kaizen_tutorial": "UC2MIh5e1htqlp-Pnid2FtCA",  # Blender beginner
    "tutor4u": "UCB5Yaq0KiXi3QoqNkBEHkGg",  # motion graphics
    "royal_skies_llc": "UCxFt75OIIvoN4AgrZE4GKKA",  # quick Blender tutorials
    # ── Blender: Animation & rigging ─────────────────────────────────────────
    "dedicatedteacher": "UCbmxZRbCedDMlsm4EQBal1A",
    "sardi_pax": "UCJ6RM7FCFMLbL5GWc7ARYkA",
    "thilakanathan_studios": "UCPuCfOcpZpb0rAnltntGFcw",
    # ── Blender: Geometry nodes & procedural ─────────────────────────────────
    "blender_made_easy": "UCddiUEpeqJcYeBxX1IVBKvQ",
    "node_group": "UCWWybvONDObMelHzVkrIaEg",
    "erindale": "UCi6o_dYdlLjhWa3O_sRJlDg",
    # ── Blender: VFX & simulation ─────────────────────────────────────────────
    "cg_geek": "UCG8AxMVa6eutIGxrdnDxWpQ",
    "blenderphysics": "UCNbpRMMAAAACKl0CJ4b1o7vQ",
    # ── Blender: Shading & rendering ─────────────────────────────────────────
    "joyce_ale": "UCPMEuVgxT5h47SGbCRqJR1g",
    "maxnovak": "UCIBgYfDjtWlbJhg--Z4sOgQ",
    # ── Blender: Environments & arch viz ─────────────────────────────────────
    "chris_p": "UCPUe4MwQHqVHVE-hIRPnBKg",
    "noggi": "UCrHQNOyU1q6BFEfkNq2CYYA",
    # ── Blender: Official ────────────────────────────────────────────────────
    "blender_official": "UCAsuoIGs1vd6GuvN5tKsGbg",  # Fixed: was duplicate of blender_guru
    # ── Maya channels ────────────────────────────────────────────────────────
    "cgcircuit": "UCmbQOPEJAcXE6eFf7J_jbsg",  # Maya & Houdini
    "creative_shrimp": "UCDBP9K-4Q8OFBbXJ7sBgRFg",  # Maya rigging
    "ryan_laley": "UCpDSPZP4xLAHB0bFJJBrDoA",  # Maya modeling
    "maya_station": "UCnlVdLdJVELBP1HHj5VKBMQ",  # Maya workflows
    "animation_mentor": "UCDuDPR8JstHVsFgpOjFDfcA",  # Maya animation
    # ── Cinema 4D channels ───────────────────────────────────────────────────
    "eyedesyn": "UCVRMUKIRiNfCO8nCKt0KaDA",  # Cinema 4D & motion
    "nick_campbell": "UCaP7vf4pJNhAGCHOZV3QFUQ",  # C4D mograph — Fixed: was duplicate of eyedesyn
    "greyscalegorilla": "UCzJStnBNZ2KniJyNpAvmHKw",  # C4D tutorials
    "mograph_plus": "UCcnVTEGJKBH2kk5r54Kkejg",  # Cinema 4D advanced
    # ── Houdini channels ─────────────────────────────────────────────────────
    "cgwiki_houdini": "UCm2VKjbGmzPBW0k3GxlrBhg",  # Houdini fundamentals
    "rebelway": "UCGFn9PXBNH0mZ_cBvMuaVIQ",  # Houdini VFX
    "sidefx_official": "UCsP3IYGSBzVGfxgMSHR6lEA",  # SideFX official
    "tokeru_houdini": "UCu7Jnne5-TaFGMJDmPULlOQ",  # Houdini procedural
    # ── ZBrush channels ──────────────────────────────────────────────────────
    "flippednormals": "UCQb5fnBzqcK9LlmPYHyMFNQ",  # ZBrush & sculpting
    "michael_pavlovich": "UCl4JMb87F0IkRzEbFSQ-SFg",  # ZBrush character
    "pixologic_official": "UCDB1T9mVmKrrlrDFSHrJJnw",  # ZBrush official
    "steve_james_art": "UCfMq4JoJpHV5cFDzBSmSVPQ",  # ZBrush creature
    # ── Substance channels ───────────────────────────────────────────────────
    "adobe_substance": "UCLy66sDFJXCQjO2QRBF9KYw",  # Official Substance
    "pbr_substance": "UCCkCJfHiOZqw7L8Kl4tqTAw",  # Substance texturing
    "art_of_gamer": "UCHPiqByR5mXv6NVq7aBB7Lw",  # Substance Painter
    # ── Rhino / Grasshopper channels ─────────────────────────────────────────
    "modelab": "UCXH_b5WiBrWWqH3sK5RxdSA",  # Grasshopper parametric
    "paramarch": "UCn7qvfFz5vEIjBqMYkKKzFg",  # Rhino architecture
    "david_rutten": "UCbkdh_OkHbVcPwLOHDEnFng",  # Grasshopper official
    # ── Unreal Engine channels ───────────────────────────────────────────────
    "ue_official": "UCBobmJyzsJ6Ll7UbfhI4iwQ",  # Unreal official
    "william_faucher": "UCTwhJCnM2WEDQDpfWrBjtvg",  # UE5 environments
    "virtus_learning": "UCZ2QHHKnCtW7g-yHddS_AiA",  # UE5 game dev
    "ryan_manning_ue5": "UCJFGsZVdm5O7l0_S9D0K8qg",  # UE5 modeling
    # ── Fusion 360 / CAD channels ────────────────────────────────────────────
    "product_design_online": "UCooViVfi0DlB5DXvO3cxptQ",  # Fusion 360 CAD
    "lars_christensen": "UCTrCi8J9OL-hZzD5mCJPFoQ",  # Autodesk Fusion
    "mechatronics_lab": "UC-djlS0zGVt-k6t7xrJn_gw",  # Fusion 360 engineering
    # ── SketchUp channels ────────────────────────────────────────────────────
    "sketchup_official": "UCRnnekixYt3_mQWOc7cCj6Q",  # SketchUp official
    "the_sketchup_essentials": "UCAqC1VIFfxBbIlATPM3hFOA",  # SketchUp tutorials
    # ── 3ds Max channels ─────────────────────────────────────────────────────
    "autodesk_official": "UC7ULDgSVV7WQGH5LGy6DUZQ",  # Autodesk official
    "arrimus_3d": "UCBbgLEx4-FMjvC_nQkiuGdA",  # Hard surface 3ds Max — Fixed: was duplicate of blender_guru
    "cg_tuts_3dsmax": "UCF8fLjCQPMBMuXEi9WJsHwA",  # 3ds Max architecture
    # ── Cross-discipline / theory channels ───────────────────────────────────
    "corridor_crew": "UCSpFnDQr88xCZ80N-X7t0nQ",  # VFX breakdown
    "gnomon_workshop": "UCtTdKzTKPBWLJEGRmLJUKGg",  # Pro industry tutorials
    "art_of_rendering": "UCOB2OeXiLs9-yBG3MIL9L8A",  # Rendering theory
    "substance_by_adobe": "UCNbGHGn2AqD7bfAqDIvpqsg",  # PBR materials — Fixed: was duplicate of adobe_substance
}

# ─── Blender search queries (core domain — 80+ queries) ──────────────────────
BLENDER_SEARCH_QUERIES = [
    # Modeling
    "blender mesh modeling tutorial beginners",
    "blender loop cut and slide tutorial",
    "blender extrude faces tutorial",
    "blender inset faces modeling",
    "blender bevel modifier tutorial",
    "blender subdivision surface modeling",
    "blender boolean modifier hard surface",
    "blender knife tool modeling",
    "blender bridge edge loops",
    "blender proportional editing tutorial",
    "blender vertex edge face selection mode",
    "blender mirror modifier modeling",
    "blender array modifier tutorial",
    "blender screw modifier tutorial",
    "blender solidify modifier",
    "blender shrinkwrap modifier",
    "blender lattice deform tutorial",
    # Hard surface
    "blender hard surface modeling tutorial",
    "blender mechanical modeling tutorial",
    "blender product design modeling",
    "blender low poly vehicle tutorial",
    "blender spaceship modeling tutorial",
    "blender weapon modeling tutorial",
    "blender architecture modeling interior",
    # Organic / characters
    "blender character modeling tutorial",
    "blender face modeling tutorial",
    "blender hand modeling tutorial",
    "blender stylized character blender tutorial",
    "blender body modeling tutorial",
    # Sculpting
    "blender sculpting tutorial beginners",
    "blender dynamic topology sculpting",
    "blender multiresolution sculpting",
    "blender sculpt masking tutorial",
    "blender remesh sculpting workflow",
    "blender face sculpting anatomy tutorial",
    # Rigging
    "blender rigging armature tutorial",
    "blender weight painting tutorial",
    "blender inverse kinematics tutorial",
    "blender shape keys tutorial",
    "blender control rig tutorial",
    # Animation
    "blender animation keyframes tutorial",
    "blender walk cycle animation tutorial",
    "blender graph editor animation",
    "blender NLA editor tutorial",
    "blender drivers tutorial",
    "blender path animation tutorial",
    "blender cloth simulation tutorial",
    "blender fluid simulation tutorial",
    "blender rigid body simulation tutorial",
    "blender particle system tutorial",
    "blender hair particle system",
    "blender smoke fire simulation",
    # Shading & materials
    "blender shader nodes tutorial beginners",
    "blender principled BSDF tutorial",
    "blender procedural textures tutorial",
    "blender UV unwrapping tutorial",
    "blender texture painting tutorial",
    "blender material library tutorial",
    "blender glass shader tutorial",
    "blender metal shader tutorial",
    "blender skin shader tutorial",
    "blender emission shader tutorial",
    # Geometry nodes
    "blender geometry nodes tutorial beginners",
    "blender geometry nodes scatter tutorial",
    "blender geometry nodes procedural modeling",
    "blender geometry nodes animation tutorial",
    "blender geometry nodes terrain generation",
    "blender geometry nodes instancing tutorial",
    # Rendering
    "blender cycles render settings tutorial",
    "blender EEVEE render tutorial",
    "blender lighting tutorial studio",
    "blender HDRI lighting tutorial",
    "blender compositing tutorial",
    "blender render passes tutorial",
    "blender denoising tutorial",
    # Grease pencil
    "blender grease pencil tutorial",
    "blender 2D animation grease pencil",
    # VFX & compositing
    "blender camera tracking tutorial",
    "blender motion tracking VFX",
    "blender compositing nodes tutorial",
    # Workflow & tools
    "blender object parenting tutorial",
    "blender collections tutorial",
    "blender modifiers workflow tutorial",
    "blender scripting python tutorial",
    "blender custom properties tutorial",
    "blender asset library tutorial",
]

# ─── Cross-software queries (Maya, C4D, Houdini, ZBrush, Substance, etc.) ────
CROSS_SOFTWARE_QUERIES = [
    # Maya
    "maya modeling tutorial",
    "maya rigging tutorial",
    "maya VFX tutorial",
    "maya animation tutorial",
    "autodesk maya character modeling",
    "maya hard surface modeling tutorial",
    "maya dynamics simulation tutorial",
    "maya ncloth tutorial",
    "maya python scripting tutorial",
    "maya bifrost tutorial",
    "maya arnold render tutorial",
    # Cinema 4D
    "cinema 4d tutorial",
    "c4d motion graphics tutorial",
    "cinema 4d mograph tutorial",
    "maxon c4d tutorial",
    "cinema 4d procedural animation",
    "cinema 4d dynamics tutorial",
    "cinema 4d character animation tutorial",
    "cinema 4d redshift render tutorial",
    "cinema 4d xpresso tutorial",
    # Houdini
    "houdini tutorial",
    "houdini vfx tutorial",
    "sidefx houdini",
    "houdini procedural modeling",
    "houdini simulation tutorial",
    "houdini vops tutorial",
    "houdini pyro simulation",
    "houdini flip fluid tutorial",
    "houdini cloth simulation",
    "houdini crowds simulation",
    "houdini solaris tutorial",
    "houdini python scripting tutorial",
    # ZBrush
    "zbrush tutorial",
    "zbrush character sculpting",
    "zbrush creature design",
    "zbrush hard surface tutorial",
    "zbrush retopology tutorial",
    "zbrush fibermesh tutorial",
    "zbrush polypaint tutorial",
    "zbrush dynamics tutorial",
    # Substance
    "substance painter tutorial",
    "substance designer tutorial",
    "pbr texturing tutorial",
    "substance 3d tutorial",
    "substance painter smart materials",
    "substance designer node graph tutorial",
    "substance painter game asset texturing",
    # Rhino / Grasshopper
    "rhino 3d tutorial",
    "grasshopper tutorial",
    "rhino architecture tutorial",
    "grasshopper parametric design",
    "rhino 3d surface modeling",
    "grasshopper generative design",
    "rhino nurbs modeling tutorial",
    # Unreal Engine
    "unreal engine 5 tutorial",
    "ue5 modeling tutorial",
    "unreal environment design",
    "unreal engine 5 materials tutorial",
    "unreal engine 5 nanite tutorial",
    "unreal engine 5 lumen tutorial",
    "unreal engine 5 blueprint tutorial",
    "unreal engine 5 game dev tutorial",
    "unreal engine 5 arch viz tutorial",
    # Fusion 360
    "fusion 360 tutorial",
    "fusion 360 cad tutorial",
    "autodesk fusion modeling",
    "fusion 360 parametric design",
    "fusion 360 sheet metal tutorial",
    "fusion 360 assembly tutorial",
    "fusion 360 product design",
    # 3ds Max
    "3ds max tutorial",
    "3ds max architectural visualization",
    "3ds max hard surface modeling",
    "3ds max vray tutorial",
    "3ds max corona render tutorial",
    "3ds max modifier stack tutorial",
    # SketchUp
    "sketchup tutorial",
    "sketchup architecture tutorial",
    "sketchup interior design tutorial",
    "sketchup vray tutorial",
    "sketchup enscape tutorial",
]

# ─── Design theory and physics queries (cross-discipline) ─────────────────────
DESIGN_PHYSICS_QUERIES = [
    # PBR and rendering theory
    "physically based rendering tutorial",
    "PBR texturing explained",
    "3d lighting theory tutorial",
    "global illumination explained",
    "ray tracing fundamentals tutorial",
    "HDRI lighting 3d tutorial",
    "color management 3d workflow",
    # Topology and mesh theory
    "topology for 3d artists",
    "retopology tutorial",
    "subdivision surface modeling",
    "edge flow 3d modeling tutorial",
    "low poly to high poly workflow",
    "normal map baking tutorial",
    # Physics simulations
    "physics simulation blender",
    "cloth simulation tutorial",
    "fluid simulation 3d",
    "particle system tutorial 3d",
    "rigid body simulation tutorial",
    "soft body simulation tutorial",
    "smoke fire simulation 3d",
    # Design disciplines
    "product design 3d tutorial",
    "architectural visualization tutorial",
    "game-ready asset creation",
    "film VFX tutorial",
    "motion graphics tutorial",
    "concept art to 3d tutorial",
    "character concept to 3d model",
    "vehicle design 3d modeling",
    "environment concept art 3d",
    "industrial design 3d modeling tutorial",
]

# Combined — all query lists together (used when --search + --all-categories)
SEARCH_QUERIES = (
    BLENDER_SEARCH_QUERIES + CROSS_SOFTWARE_QUERIES + DESIGN_PHYSICS_QUERIES
)


def yt_get(endpoint: str, params: dict, api_key: str) -> dict:
    params["key"] = api_key
    url = f"{YT_BASE}/{endpoint}?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=15) as resp:
        return json.loads(resp.read())


def channel_to_uploads_playlist(channel_id: str) -> str:
    """Convert UC... channel ID to UU... uploads playlist ID."""
    return "UU" + channel_id[2:]


def crawl_channel(channel_name: str, channel_id: str, api_key: str) -> list[str]:
    """Get all video IDs from a channel's uploads. Cost: 1 unit per 50 videos."""
    playlist_id = channel_to_uploads_playlist(channel_id)
    video_ids = []
    page_token = None
    page = 0

    while True:
        params = {
            "part": "contentDetails",
            "playlistId": playlist_id,
            "maxResults": 50,
        }
        if page_token:
            params["pageToken"] = page_token

        try:
            data = yt_get("playlistItems", params, api_key)
        except Exception as e:
            print(f"    [ERROR] {channel_name} page {page}: {e}")
            break

        for item in data.get("items", []):
            vid_id = item.get("contentDetails", {}).get("videoId")
            if vid_id:
                video_ids.append(vid_id)

        page_token = data.get("nextPageToken")
        page += 1
        if not page_token:
            break
        time.sleep(0.1)  # gentle rate limiting

    return video_ids


def filter_by_quality(
    video_ids: list[str], api_key: str, min_views: int = 5000
) -> list[str]:
    """
    Use videos.list to get view counts and filter out low-quality videos.
    Cost: 1 unit per 50 videos.
    """
    filtered = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i : i + 50]
        try:
            data = yt_get(
                "videos",
                {
                    "part": "statistics,contentDetails",
                    "id": ",".join(batch),
                },
                api_key,
            )
        except Exception as e:
            print(f"  [ERROR] quality filter batch {i // 50}: {e}")
            filtered.extend(batch)  # keep on error
            continue

        for item in data.get("items", []):
            stats = item.get("statistics", {})
            view_count = int(stats.get("viewCount", 0))
            # Also filter by duration: skip < 2 min or > 3 hours
            item.get("contentDetails", {}).get("duration", "PT0S")
            if view_count >= min_views:
                filtered.append(item["id"])

        time.sleep(0.1)

    return filtered


def search_videos(query: str, api_key: str, max_pages: int = 2) -> list[str]:
    """Search YouTube for videos matching query. Cost: 100 units per page."""
    video_ids = []
    page_token = None

    for _ in range(max_pages):
        params = {
            "part": "id",
            "q": query,
            "type": "video",
            "maxResults": 50,
            "relevanceLanguage": "en",
            "videoDuration": "medium",  # 4-20 min — optimal tutorial length
        }
        if page_token:
            params["pageToken"] = page_token

        try:
            data = yt_get("search", params, api_key)
        except Exception as e:
            print(f"  [ERROR] search '{query}': {e}")
            break

        for item in data.get("items", []):
            vid_id = item.get("id", {}).get("videoId")
            if vid_id:
                video_ids.append(vid_id)

        page_token = data.get("nextPageToken")
        if not page_token:
            break
        time.sleep(0.2)

    return video_ids


def load_existing_ids() -> set[str]:
    if not VIDEO_IDS_FILE.exists():
        return set()
    return set(l.strip() for l in VIDEO_IDS_FILE.read_text().splitlines() if l.strip())


def save_ids(new_ids: list[str]) -> int:
    existing = load_existing_ids()
    truly_new = [vid for vid in new_ids if vid not in existing]
    if not truly_new:
        return 0

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with VIDEO_IDS_FILE.open("a") as f:
        for vid in truly_new:
            f.write(vid + "\n")
    return len(truly_new)


def load_sources_channels(sources_path: str) -> dict[str, str]:
    """
    Read discovered_sources.json and return a dict of {channel_url: software}.
    Used to register channels discovered by data_discovery.py.
    """
    import json as _json

    try:
        with open(sources_path) as f:
            sources = _json.load(f)
    except (FileNotFoundError, ValueError):
        return {}

    mapping: dict[str, str] = {}
    for src in sources:
        if src.get("type") == "youtube_channel":
            url = src.get("url", "")
            sw = src.get("software", "multi")
            if url:
                mapping[url] = sw
    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="Discover 3D tutorial video IDs at scale across all major software"
    )
    parser.add_argument("--api-key", default=os.environ.get("YOUTUBE_API_KEY", ""))
    parser.add_argument(
        "--channels", action="store_true", help="Crawl all known 3D channels"
    )
    parser.add_argument(
        "--search", action="store_true", help="Run Blender keyword search queries"
    )
    parser.add_argument(
        "--all-software",
        action="store_true",
        help="Add cross-software queries (Maya, C4D, Houdini, ZBrush, Substance, Rhino, Unreal, etc.)",
    )
    parser.add_argument(
        "--all-categories",
        action="store_true",
        help="Enable all modes: --channels + --search + --all-software + --design-physics",
    )
    parser.add_argument(
        "--design-physics",
        action="store_true",
        help="Add design theory and physics simulation queries",
    )
    parser.add_argument(
        "--sources",
        default=None,
        help="Path to discovered_sources.json from data_discovery.py — adds extra channels",
    )
    parser.add_argument(
        "--filter-quality",
        action="store_true",
        help="Filter by view count (costs extra API units)",
    )
    parser.add_argument(
        "--min-views",
        type=int,
        default=5000,
        help="Min view count when --filter-quality is set",
    )
    parser.add_argument(
        "--search-pages",
        type=int,
        default=1,
        help="Pages per search query (100 units each)",
    )
    args = parser.parse_args()

    # --all-categories activates everything
    if args.all_categories:
        args.channels = True
        args.search = True
        args.all_software = True
        args.design_physics = True

    if not (args.channels or args.search or args.all_software or args.design_physics):
        print(
            "Specify at least one mode: --channels, --search, --all-software, --all-categories, or --design-physics"
        )
        return

    if not args.api_key:
        print("Error: provide --api-key or set YOUTUBE_API_KEY")
        return

    total_added = 0

    # Register extra channels from data_discovery.py output
    extra_channels: dict[str, str] = {}
    if args.sources:
        extra_channels = load_sources_channels(args.sources)
        if extra_channels:
            print(f"  Loaded {len(extra_channels)} extra channels from {args.sources}")

    if args.channels:
        print(f"=== CHANNEL CRAWL ({len(ALL_CHANNELS)} known channels) ===\n")
        for name, channel_id in ALL_CHANNELS.items():
            print(f"  {name}...")
            ids = crawl_channel(name, channel_id, args.api_key)
            if args.filter_quality and ids:
                before = len(ids)
                ids = filter_by_quality(ids, args.api_key, args.min_views)
                print(f"    {len(ids)}/{before} passed quality filter")
            added = save_ids(ids)
            print(f"    +{added} new IDs (crawled {len(ids)} total)")
            total_added += added

        # Also crawl any channels discovered by data_discovery.py
        if extra_channels:
            print(f"\n  +{len(extra_channels)} channels from discovered_sources.json")
            for ch_url, sw in extra_channels.items():
                # Extract channel handle or ID from URL for display
                ch_name = ch_url.rstrip("/").split("/")[-1]
                # Channel URLs from data_discovery are @handle format — search for IDs would
                # require an extra API call; skip crawl for now, log them for manual review.
                print(
                    f"    [info] {ch_name} ({sw}) — add channel ID to ALL_CHANNELS to crawl"
                )

    if args.search:
        active_queries = BLENDER_SEARCH_QUERIES
        print(
            f"\n=== BLENDER SEARCH ({len(active_queries)} queries, {args.search_pages} page(s) each) ==="
        )
        print(f"Quota cost: ~{len(active_queries) * args.search_pages * 100} units\n")
        for query in active_queries:
            ids = search_videos(query, args.api_key, max_pages=args.search_pages)
            added = save_ids(ids)
            print(f"  [{added:>4} new] {query}")
            total_added += added
            time.sleep(0.2)

    if args.all_software:
        print(
            f"\n=== CROSS-SOFTWARE SEARCH ({len(CROSS_SOFTWARE_QUERIES)} queries) ==="
        )
        print(
            "  Maya, Cinema 4D, Houdini, ZBrush, Substance, Rhino, Unreal, Fusion 360, 3ds Max, SketchUp"
        )
        print(
            f"Quota cost: ~{len(CROSS_SOFTWARE_QUERIES) * args.search_pages * 100} units\n"
        )
        for query in CROSS_SOFTWARE_QUERIES:
            ids = search_videos(query, args.api_key, max_pages=args.search_pages)
            added = save_ids(ids)
            print(f"  [{added:>4} new] {query}")
            total_added += added
            time.sleep(0.2)

    if args.design_physics:
        print(
            f"\n=== DESIGN & PHYSICS SEARCH ({len(DESIGN_PHYSICS_QUERIES)} queries) ==="
        )
        print("  PBR, topology, physics simulations, design disciplines")
        print(
            f"Quota cost: ~{len(DESIGN_PHYSICS_QUERIES) * args.search_pages * 100} units\n"
        )
        for query in DESIGN_PHYSICS_QUERIES:
            ids = search_videos(query, args.api_key, max_pages=args.search_pages)
            added = save_ids(ids)
            print(f"  [{added:>4} new] {query}")
            total_added += added
            time.sleep(0.2)

    existing = load_existing_ids()
    print(f"\nTotal added this run: {total_added}")
    print(f"Total unique video IDs in {VIDEO_IDS_FILE}: {len(existing)}")
    print("\nNext step: python fetch_bulk.py")


if __name__ == "__main__":
    main()
