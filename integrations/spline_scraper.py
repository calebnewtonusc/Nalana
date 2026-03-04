"""
integrations/spline_scraper.py — Spline.design public gallery scraper.

Spline is a browser-based 3D tool with a massive public gallery of scenes.
Each .splinecode file is JSON under the hood: objects, materials, transforms,
animations. We scrape it, parse it, and convert to Universal DSL training pairs.

Usage:
    python integrations/spline_scraper.py --limit 1000 --output data/integrations/spline/
    python integrations/spline_scraper.py --limit 100 --delay 2.0
    python integrations/spline_scraper.py --resume  # continue from cache
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import aiohttp
except ImportError:
    print("ERROR: aiohttp required. Install with: pip install aiohttp", file=sys.stderr)
    sys.exit(1)
from tqdm import tqdm

# ─── Setup ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("spline_scraper")

BASE_DIR = Path(__file__).parents[1]
DATA_DIR = BASE_DIR / "data" / "integrations" / "spline"
CACHE_DIR = DATA_DIR / "_cache"

SPLINE_BASE = "https://spline.design"
SPLINE_API_BASE = "https://api.spline.design"
SPLINE_COMMUNITY_API = "https://community.spline.design/api"

# Documented and reverse-engineered Spline API endpoints
GALLERY_ENDPOINTS = [
    f"{SPLINE_COMMUNITY_API}/scenes?sort=trending&limit=50&offset={{offset}}",
    f"{SPLINE_BASE}/api/public/scenes?limit=50&page={{page}}",
    f"{SPLINE_API_BASE}/public/gallery?limit=50&cursor={{cursor}}",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; NalanaDatasetCollector/1.0; research)",
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

# Rate limit: 1 request/sec by default
DEFAULT_DELAY = 1.0

# ─── Material → DSL mapping ───────────────────────────────────────────────────

SPLINE_MATERIAL_MAP = {
    "standard": "PRINCIPLED_BSDF",
    "glass": "GLASS_BSDF",
    "toon": "DIFFUSE_BSDF",
    "normal": "PRINCIPLED_BSDF",
    "depth": "PRINCIPLED_BSDF",
}

SPLINE_SHAPE_MAP = {
    "box": "ADD_CUBE",
    "sphere": "ADD_SPHERE",
    "cylinder": "ADD_CYLINDER",
    "cone": "ADD_CONE",
    "torus": "ADD_TORUS",
    "plane": "ADD_PLANE",
    "text": "ADD_TEXT",
    "parametric": "ADD_CURVE",
}

# ─── Spline scene parser ───────────────────────────────────────────────────────


def parse_spline_color(
    color_data: dict | list | None,
) -> tuple[float, float, float, float]:
    """Convert Spline color format (0-255 int or 0-1 float) to RGBA 0-1 tuple."""
    if color_data is None:
        return (0.8, 0.8, 0.8, 1.0)
    if isinstance(color_data, list) and len(color_data) >= 3:
        r, g, b = color_data[:3]
        a = color_data[3] if len(color_data) > 3 else 1.0
        # Spline sometimes uses 0-255, sometimes 0-1
        if any(v > 1.0 for v in [r, g, b]):
            return (r / 255, g / 255, b / 255, a / 255)
        return (r, g, b, a)
    if isinstance(color_data, dict):
        r = color_data.get("r", 0.8)
        g = color_data.get("g", 0.8)
        b = color_data.get("b", 0.8)
        a = color_data.get("a", 1.0)
        if any(v > 1.0 for v in [r, g, b]):
            return (r / 255, g / 255, b / 255, a / 255)
        return (r, g, b, a)
    return (0.8, 0.8, 0.8, 1.0)


def parse_spline_transform(obj: dict) -> dict:
    """Extract position, rotation, scale from a Spline object node."""
    pos = obj.get("position", obj.get("transform", {}).get("position", {}))
    rot = obj.get("rotation", obj.get("transform", {}).get("rotation", {}))
    scl = obj.get("scale", obj.get("transform", {}).get("scale", {}))

    def _vec(d: dict | list | None, default: tuple) -> tuple:
        if d is None:
            return default
        if isinstance(d, list):
            return tuple(d[:3]) if len(d) >= 3 else default
        return (
            d.get("x", default[0]),
            d.get("y", default[1]),
            d.get("z", default[2]),
        )

    return {
        "location": _vec(pos, (0.0, 0.0, 0.0)),
        "rotation": _vec(rot, (0.0, 0.0, 0.0)),
        "scale": _vec(scl, (1.0, 1.0, 1.0)),
    }


def parse_spline_material(mat: dict) -> dict:
    """Extract material properties from a Spline material node."""
    color_key = next(
        (k for k in ("color", "diffuseColor", "albedo", "baseColor") if k in mat), None
    )
    color = parse_spline_color(mat.get(color_key) if color_key else None)
    mat_type = mat.get("type", mat.get("shader", "standard")).lower()

    return {
        "type": SPLINE_MATERIAL_MAP.get(mat_type, "PRINCIPLED_BSDF"),
        "color": list(color),
        "roughness": float(mat.get("roughness", 0.5)),
        "metalness": float(mat.get("metalness", mat.get("metallic", 0.0))),
        "emissive": list(parse_spline_color(mat.get("emissiveColor"))),
        "emissive_intensity": float(mat.get("emissiveIntensity", 0.0)),
        "opacity": float(mat.get("opacity", 1.0)),
        "transmission": float(mat.get("transmission", 0.0)),
    }


def describe_material(mat_props: dict) -> str:
    """Generate a human-readable material description for voice command synthesis."""
    parts = []
    r, g, b, _ = mat_props["color"]

    # Simple color naming
    if r > 0.8 and g < 0.3 and b < 0.3:
        parts.append("red")
    elif g > 0.8 and r < 0.3 and b < 0.3:
        parts.append("green")
    elif b > 0.8 and r < 0.3 and g < 0.3:
        parts.append("blue")
    elif r > 0.85 and g > 0.85 and b > 0.85:
        parts.append("white")
    elif r < 0.15 and g < 0.15 and b < 0.15:
        parts.append("black")
    elif r > 0.8 and g > 0.6 and b < 0.3:
        parts.append("orange")
    elif r > 0.7 and g > 0.7 and b < 0.3:
        parts.append("yellow")
    elif r > 0.5 and b > 0.5 and g < 0.4:
        parts.append("purple")
    else:
        parts.append(f"rgb({r:.2f},{g:.2f},{b:.2f})")

    if mat_props["metalness"] > 0.7:
        parts.append("metallic")
    if mat_props["roughness"] < 0.2:
        parts.append("glossy")
    elif mat_props["roughness"] > 0.8:
        parts.append("matte")
    if mat_props["transmission"] > 0.5:
        parts.append("glass-like")
    if mat_props["emissive_intensity"] > 0.5:
        parts.append("emissive")

    return " ".join(parts)


def spline_object_to_dsl(obj: dict) -> list[dict]:
    """Convert a single Spline object node to one or more Universal DSL operations."""
    ops = []
    obj_type = obj.get("type", obj.get("shape", "box")).lower()
    name = obj.get("name", obj.get("id", "object"))
    transform = parse_spline_transform(obj)

    # Map Spline shape type to DSL op
    dsl_op = SPLINE_SHAPE_MAP.get(obj_type, "ADD_CUBE")

    # Build args based on shape
    args: dict[str, Any] = {}
    if dsl_op == "ADD_CUBE":
        dims = obj.get("dimensions", obj.get("size", {}))
        if isinstance(dims, dict):
            w = dims.get("x", dims.get("width", 2.0))
            args["size"] = float(w)
        elif isinstance(dims, (int, float)):
            args["size"] = float(dims)
        else:
            args["size"] = 2.0
    elif dsl_op == "ADD_SPHERE":
        args["radius"] = float(obj.get("radius", 1.0))
        args["segments"] = int(obj.get("segments", 32))
    elif dsl_op == "ADD_CYLINDER":
        args["radius"] = float(obj.get("radius", 1.0))
        args["depth"] = float(obj.get("height", obj.get("depth", 2.0)))
    elif dsl_op == "ADD_CONE":
        args["radius1"] = float(obj.get("bottomRadius", obj.get("radius", 1.0)))
        args["radius2"] = float(obj.get("topRadius", 0.0))
        args["depth"] = float(obj.get("height", 2.0))
    elif dsl_op == "ADD_TORUS":
        args["major_radius"] = float(obj.get("majorRadius", 1.0))
        args["minor_radius"] = float(obj.get("minorRadius", 0.25))

    args["location"] = list(transform["location"])

    ops.append(
        {
            "op": dsl_op,
            "args": args,
            "target": name,
            "intent": f"Create {obj_type} named '{name}'",
        }
    )

    # Material operation
    mat_data = obj.get(
        "material",
        obj.get("materials", [None])[0]
        if isinstance(obj.get("materials"), list)
        else None,
    )
    if mat_data and isinstance(mat_data, dict):
        mat_props = parse_spline_material(mat_data)
        ops.append(
            {
                "op": "SET_MATERIAL",
                "args": mat_props,
                "target": name,
                "intent": f"Apply {describe_material(mat_props)} material to '{name}'",
            }
        )

    return ops


def spline_scene_to_training_pairs(scene_data: dict, scene_id: str) -> list[dict]:
    """
    Convert a full Spline scene JSON to a list of Universal DSL training pairs.
    Returns one pair per significant object with material information.
    """
    pairs = []
    scene_name = scene_data.get(
        "name", scene_data.get("title", f"scene_{scene_id[:8]}")
    )
    objects = scene_data.get(
        "objects", scene_data.get("children", scene_data.get("nodes", []))
    )

    if not objects:
        log.debug("Scene %s has no parseable objects", scene_id)
        return pairs

    for obj in objects:
        if not isinstance(obj, dict):
            continue

        obj_type = obj.get("type", obj.get("shape", "")).lower()
        if not obj_type or obj_type in ("group", "scene", "camera", "light", "empty"):
            continue

        dsl_ops = spline_object_to_dsl(obj)
        if not dsl_ops:
            continue

        shape_op = dsl_ops[0]
        shape_name = obj_type if obj_type in SPLINE_SHAPE_MAP else "shape"

        mat_props = {}
        mat_description = "default material"
        if len(dsl_ops) > 1 and dsl_ops[1]["op"] == "SET_MATERIAL":
            mat_props = dsl_ops[1]["args"]
            mat_description = describe_material(mat_props)

        # Build Blender Python for this object
        blender_lines = ["import bpy", ""]
        loc = shape_op["args"].get("location", [0, 0, 0])

        if shape_op["op"] == "ADD_CUBE":
            size = shape_op["args"].get("size", 2.0)
            blender_lines.append(
                f"bpy.ops.mesh.primitive_cube_add(size={size}, location={tuple(loc)})"
            )
        elif shape_op["op"] == "ADD_SPHERE":
            r = shape_op["args"].get("radius", 1.0)
            seg = shape_op["args"].get("segments", 32)
            blender_lines.append(
                f"bpy.ops.mesh.primitive_uv_sphere_add(radius={r}, segments={seg}, location={tuple(loc)})"
            )
        elif shape_op["op"] == "ADD_CYLINDER":
            r = shape_op["args"].get("radius", 1.0)
            d = shape_op["args"].get("depth", 2.0)
            blender_lines.append(
                f"bpy.ops.mesh.primitive_cylinder_add(radius={r}, depth={d}, location={tuple(loc)})"
            )
        elif shape_op["op"] == "ADD_CONE":
            r1 = shape_op["args"].get("radius1", 1.0)
            r2 = shape_op["args"].get("radius2", 0.0)
            d = shape_op["args"].get("depth", 2.0)
            blender_lines.append(
                f"bpy.ops.mesh.primitive_cone_add(radius1={r1}, radius2={r2}, depth={d}, location={tuple(loc)})"
            )
        elif shape_op["op"] == "ADD_TORUS":
            R = shape_op["args"].get("major_radius", 1.0)
            r = shape_op["args"].get("minor_radius", 0.25)
            blender_lines.append(
                f"bpy.ops.mesh.primitive_torus_add(major_radius={R}, minor_radius={r}, location={tuple(loc)})"
            )
        else:
            blender_lines.append(
                f"bpy.ops.mesh.primitive_cube_add(location={tuple(loc)})"
            )

        obj_var = "bpy.context.active_object"
        blender_lines += [
            f"obj = {obj_var}",
            f'obj.name = "{obj.get("name", "spline_obj")}"',
            "",
        ]

        if mat_props:
            c = mat_props.get("color", [0.8, 0.8, 0.8, 1.0])
            rough = mat_props.get("roughness", 0.5)
            metal = mat_props.get("metalness", 0.0)
            emit = mat_props.get("emissive_intensity", 0.0)
            blender_lines += [
                "mat = bpy.data.materials.new(name='SplineMaterial')",
                "mat.use_nodes = True",
                "nodes = mat.node_tree.nodes",
                "bsdf = nodes.get('Principled BSDF')",
                f"bsdf.inputs['Base Color'].default_value = ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}, 1.0)",
                f"bsdf.inputs['Roughness'].default_value = {rough:.3f}",
                f"bsdf.inputs['Metallic'].default_value = {metal:.3f}",
            ]
            if emit > 0:
                mat_props.get("emissive", [0, 0, 0, 1])
                blender_lines.append(
                    f"bsdf.inputs['Emission Strength'].default_value = {emit:.3f}"
                )
            blender_lines += [
                "obj.data.materials.append(mat)",
            ]

        blender_python = "\n".join(blender_lines)

        voice_command = f"create a {mat_description} {shape_name}"
        if loc and any(abs(v) > 0.01 for v in loc):
            voice_command += f" at position {tuple(round(v, 2) for v in loc)}"

        pair = {
            "voice_command": voice_command,
            "task_type": "BUILD" if not mat_props else "MATERIALIZE",
            "scene_context": "empty scene",
            "blender_python": blender_python,
            "universal_dsl": [op for op in dsl_ops],
            "reasoning": (
                f"Spline scene analysis: Scene '{scene_name}' contains a {obj_type} object. "
                f"Converted Spline geometry node to Blender primitive with matching "
                f"PBR material ({mat_description}). Transform preserved from Spline coordinates."
            ),
            "quality": 2.0,
            "source": "spline_gallery",
            "metadata": {
                "scene_id": scene_id,
                "scene_name": scene_name,
                "object_type": obj_type,
                "object_name": obj.get("name", ""),
            },
        }
        pairs.append(pair)

    return pairs


# ─── Scraper class ─────────────────────────────────────────────────────────────


class SplineScraper:
    def __init__(
        self,
        output_dir: Path,
        delay: float = DEFAULT_DELAY,
        limit: int = 1000,
    ):
        self.output_dir = Path(output_dir)
        self.cache_dir = self.output_dir / "_cache"
        self.delay = delay
        self.limit = limit
        self.session: aiohttp.ClientSession | None = None
        self._scene_ids_seen: set[str] = set()
        self._last_request_time: float = 0.0

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._load_seen_ids()

    def _load_seen_ids(self) -> None:
        output_file = self.output_dir / "scenes.jsonl"
        if output_file.exists():
            with output_file.open() as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        sid = rec.get("metadata", {}).get("scene_id")
                        if sid:
                            self._scene_ids_seen.add(sid)
                    except json.JSONDecodeError:
                        pass
            log.info("Resuming: %d scenes already collected", len(self._scene_ids_seen))

    async def _rate_limited_get(
        self, url: str, **kwargs
    ) -> aiohttp.ClientResponse | None:
        """GET with rate limiting and error handling."""
        if self.session is None:
            raise RuntimeError(
                "HTTP session not initialized. Use 'async with SplineScraper() as s:'"
            )
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.delay:
            await asyncio.sleep(self.delay - elapsed)

        self._last_request_time = time.monotonic()
        try:
            resp = await self.session.get(
                url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=30), **kwargs
            )
            return resp
        except Exception as e:
            log.warning("GET %s failed: %s", url, e)
            return None

    def _cache_path(self, scene_id: str) -> Path:
        return self.cache_dir / f"{scene_id}.json"

    async def _fetch_cached_scene(self, scene_id: str, url: str) -> dict | None:
        cache_path = self._cache_path(scene_id)
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text())
            except json.JSONDecodeError:
                pass

        resp = await self._rate_limited_get(url)
        if resp is None or resp.status != 200:
            return None

        try:
            text = await resp.text()
            data = json.loads(text)
            cache_path.write_text(json.dumps(data))
            return data
        except Exception as e:
            log.warning("Failed to parse scene %s: %s", scene_id, e)
            return None

    async def _discover_via_api(self, limit: int) -> list[dict]:
        """Try Spline's public API endpoints to discover scenes."""
        scenes = []
        page = 0
        offset = 0

        while len(scenes) < limit:
            url = GALLERY_ENDPOINTS[0].format(offset=offset)
            resp = await self._rate_limited_get(url)
            if resp is None or resp.status != 200:
                log.info(
                    "Community API unavailable (status %s), trying fallback",
                    resp.status if resp else "timeout",
                )
                break

            try:
                data = await resp.json(content_type=None)
            except Exception:
                break

            batch = data.get("scenes", data.get("items", data.get("data", [])))
            if not batch:
                break

            scenes.extend(batch)
            offset += len(batch)
            page += 1

            if len(batch) < 50:
                break

        log.info("API discovery found %d scenes", len(scenes))
        return scenes

    async def _discover_via_html(self, limit: int) -> list[dict]:
        """
        Fall back to scraping the Spline community gallery HTML.
        Spline uses a React SPA so we look for __NEXT_DATA__ or embedded JSON.
        """
        scenes = []

        gallery_url = f"{SPLINE_BASE}/community"
        resp = await self._rate_limited_get(gallery_url)
        if resp is None or resp.status != 200:
            log.warning("HTML fallback: could not reach %s", gallery_url)
            return []

        html = await resp.text()

        # Next.js embeds page data as JSON in __NEXT_DATA__ script tag
        match = re.search(
            r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL
        )
        if match:
            try:
                next_data = json.loads(match.group(1))
                page_props = next_data.get("props", {}).get("pageProps", {})
                scenes_data = (
                    page_props.get("scenes")
                    or page_props.get("initialScenes")
                    or page_props.get("gallery", {}).get("scenes", [])
                )
                if scenes_data:
                    scenes.extend(scenes_data[:limit])
                    log.info("HTML __NEXT_DATA__ found %d scenes", len(scenes))
                    return scenes
            except json.JSONDecodeError:
                pass

        # Look for JSON arrays of scene-like objects in the HTML
        scene_pattern = re.compile(
            r'\{"id":"([a-zA-Z0-9_-]+)","name":"([^"]+)".*?"thumbnail"', re.DOTALL
        )
        for m in scene_pattern.finditer(html):
            scenes.append({"id": m.group(1), "name": m.group(2)})
            if len(scenes) >= limit:
                break

        log.info("HTML scrape found %d scenes", len(scenes))
        return scenes

    def _extract_scene_id(self, scene_record: dict) -> str | None:
        for key in ("id", "sceneId", "scene_id", "uuid", "slug"):
            val = scene_record.get(key)
            if val:
                return str(val)
        url = scene_record.get("url", scene_record.get("embedUrl", ""))
        if url:
            parts = urlparse(url).path.strip("/").split("/")
            if parts:
                return parts[-1]
        return None

    def _scene_download_url(self, scene_id: str) -> str:
        return f"https://my.spline.design/{scene_id}/scene.splinecode"

    async def _fetch_scene_file(self, scene_id: str) -> dict | None:
        """Download the .splinecode file (JSON) for a scene."""
        url = self._scene_download_url(scene_id)
        data = await self._fetch_cached_scene(scene_id, url)
        if data:
            return data

        # Try alternate URL patterns
        for alt_url in [
            f"https://prod.spline.design/{scene_id}/scene.splinecode",
            f"https://cdn.spline.design/{scene_id}/scene.splinecode",
        ]:
            data = await self._fetch_cached_scene(f"{scene_id}_alt", alt_url)
            if data:
                return data

        return None

    async def scrape(self) -> int:
        """Main scrape loop. Returns number of training pairs written."""
        connector = aiohttp.TCPConnector(limit=5, ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            self.session = session

            # Discovery phase
            log.info("Discovering Spline public scenes (target: %d)", self.limit)
            scenes = await self._discover_via_api(self.limit)
            if not scenes:
                scenes = await self._discover_via_html(self.limit)

            if not scenes:
                log.warning(
                    "No scenes discovered. Generating synthetic Spline-style training pairs."
                )
                return self._write_synthetic_pairs()

            output_file = self.output_dir / "scenes.jsonl"
            total_pairs = 0
            processed = 0

            with (
                open(output_file, "a") as f_out,
                tqdm(total=min(len(scenes), self.limit), desc="Spline scenes") as pbar,
            ):
                for scene_record in scenes:
                    if processed >= self.limit:
                        break

                    scene_id = self._extract_scene_id(scene_record)
                    if not scene_id:
                        continue
                    if scene_id in self._scene_ids_seen:
                        pbar.update(1)
                        continue

                    scene_data = await self._fetch_scene_file(scene_id)
                    if scene_data is None:
                        # Use the gallery record itself as scene metadata
                        scene_data = scene_record

                    pairs = spline_scene_to_training_pairs(scene_data, scene_id)
                    for pair in pairs:
                        f_out.write(json.dumps(pair) + "\n")
                        total_pairs += 1

                    self._scene_ids_seen.add(scene_id)
                    processed += 1
                    pbar.update(1)
                    pbar.set_postfix(pairs=total_pairs)

        log.info(
            "Done. %d scenes → %d training pairs → %s",
            processed,
            total_pairs,
            output_file,
        )
        return total_pairs

    def _write_synthetic_pairs(self) -> int:
        """
        When live scraping fails, generate high-quality synthetic pairs that
        represent the kinds of objects found in the Spline gallery.
        This ensures the pipeline always produces output.
        """
        SYNTHETIC_SCENES = [
            {
                "shape": "sphere",
                "color": [0.2, 0.5, 1.0, 1.0],
                "roughness": 0.1,
                "metalness": 0.0,
                "description": "glossy blue sphere",
            },
            {
                "shape": "box",
                "color": [1.0, 0.3, 0.2, 1.0],
                "roughness": 0.8,
                "metalness": 0.0,
                "description": "matte red cube",
            },
            {
                "shape": "cylinder",
                "color": [0.9, 0.9, 0.9, 1.0],
                "roughness": 0.0,
                "metalness": 1.0,
                "description": "metallic white cylinder",
            },
            {
                "shape": "torus",
                "color": [0.6, 0.2, 0.8, 1.0],
                "roughness": 0.3,
                "metalness": 0.2,
                "description": "slightly metallic purple torus",
            },
            {
                "shape": "cone",
                "color": [0.1, 0.9, 0.4, 1.0],
                "roughness": 0.5,
                "metalness": 0.0,
                "description": "green cone",
            },
            {
                "shape": "sphere",
                "color": [1.0, 0.8, 0.2, 1.0],
                "roughness": 0.0,
                "metalness": 0.8,
                "description": "gold metallic sphere",
            },
            {
                "shape": "box",
                "color": [0.05, 0.05, 0.05, 1.0],
                "roughness": 0.0,
                "metalness": 0.9,
                "description": "black chrome cube",
            },
            {
                "shape": "sphere",
                "color": [0.9, 0.95, 1.0, 1.0],
                "roughness": 0.0,
                "metalness": 0.0,
                "description": "glass-like white sphere",
                "transmission": 0.9,
            },
        ]

        output_file = self.output_dir / "scenes.jsonl"
        pairs_written = 0

        shape_to_blender = {
            "sphere": "bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=64, location=(0, 0, 0))",
            "box": "bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0))",
            "cylinder": "bpy.ops.mesh.primitive_cylinder_add(radius=1.0, depth=2.0, location=(0, 0, 0))",
            "torus": "bpy.ops.mesh.primitive_torus_add(major_radius=1.0, minor_radius=0.25, location=(0, 0, 0))",
            "cone": "bpy.ops.mesh.primitive_cone_add(radius1=1.0, radius2=0.0, depth=2.0, location=(0, 0, 0))",
        }

        shape_to_dsl_op = {
            "sphere": "ADD_SPHERE",
            "box": "ADD_CUBE",
            "cylinder": "ADD_CYLINDER",
            "torus": "ADD_TORUS",
            "cone": "ADD_CONE",
        }

        with open(output_file, "a") as f_out:
            for scene in SYNTHETIC_SCENES:
                shape = scene["shape"]
                desc = scene["description"]
                c = scene["color"]
                rough = scene["roughness"]
                metal = scene["metalness"]
                trans = scene.get("transmission", 0.0)

                blender_python = f"""import bpy

# Add shape
{shape_to_blender.get(shape, "bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))")}
obj = bpy.context.active_object

# Apply PBR material
mat = bpy.data.materials.new(name='SplineMaterial')
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get('Principled BSDF')
bsdf.inputs['Base Color'].default_value = ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}, 1.0)
bsdf.inputs['Roughness'].default_value = {rough:.3f}
bsdf.inputs['Metallic'].default_value = {metal:.3f}
bsdf.inputs['Transmission Weight'].default_value = {trans:.3f}
obj.data.materials.append(mat)"""

                dsl_op = shape_to_dsl_op.get(shape, "ADD_CUBE")

                pair = {
                    "voice_command": f"create a {desc}",
                    "task_type": "MATERIALIZE",
                    "scene_context": "empty scene",
                    "blender_python": blender_python,
                    "universal_dsl": [
                        {
                            "op": dsl_op,
                            "args": {"location": [0, 0, 0]},
                            "target": shape,
                            "intent": f"Create {shape}",
                        },
                        {
                            "op": "SET_MATERIAL",
                            "args": {
                                "type": "PRINCIPLED_BSDF",
                                "color": c,
                                "roughness": rough,
                                "metalness": metal,
                                "transmission": trans,
                            },
                            "target": shape,
                            "intent": f"Apply {desc} material",
                        },
                    ],
                    "reasoning": (
                        f"Spline scene analysis: Synthetic pair representing common Spline gallery object. "
                        f"Spline's material system maps directly to Blender's Principled BSDF: "
                        f"color={c[:3]}, roughness={rough}, metalness={metal}. "
                        f"Transmission={trans} maps to Blender's Transmission Weight."
                    ),
                    "quality": 2.0,
                    "source": "spline_gallery_synthetic",
                    "metadata": {
                        "scene_id": f"synthetic_{shape}_{pairs_written}",
                        "scene_name": f"Synthetic {desc}",
                        "object_type": shape,
                        "object_name": shape,
                    },
                }
                f_out.write(json.dumps(pair) + "\n")
                pairs_written += 1

        log.info(
            "Wrote %d synthetic Spline-style pairs → %s", pairs_written, output_file
        )
        return pairs_written


# ─── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Spline.design public gallery for Nalana training data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--limit", type=int, default=1000, help="Max scenes to collect")
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "data" / "integrations" / "spline",
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY, help="Seconds between requests"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from existing cache"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    scraper = SplineScraper(
        output_dir=args.output,
        delay=args.delay,
        limit=args.limit,
    )

    total = asyncio.run(scraper.scrape())
    print(
        f"\nSpline scrape complete: {total} training pairs written to {args.output}/scenes.jsonl"
    )


if __name__ == "__main__":
    main()
