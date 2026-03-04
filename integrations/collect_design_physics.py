"""
integrations/collect_design_physics.py — Collect public domain expert texts for GPU synthesis.

This is the most important integration file. It gathers public-domain / open-access expert
texts in physics, design, architecture, and 3D/VFX theory. These chunks are then fed to
Qwen2.5-72B on GPUs to synthesize expert Q&A training pairs.

The goal: teach Nalana to reason like a genius physicist AND master designer.
Not through prompting — through actual training on synthesized expert knowledge.

Sources (all free/open):
  PHYSICS:
  - Feynman Lectures on Physics (feynmanlectures.caltech.edu) — public access
  - PBRT textbook excerpts (pbrt.org)
  - Wikipedia physics articles (CC BY-SA)

  DESIGN & ARCHITECTURE:
  - Vitruvius "De Architectura" (public domain, ~15 BCE)
  - Bauhaus manifesto (1919, public domain)
  - Wikipedia: Golden ratio, Gestalt, Color theory, Proportion

  3D/VFX KNOWLEDGE:
  - OpenSubdiv documentation (Apache license)
  - USD specification overview (ASWF, Apache)
  - glTF 2.0 specification (CC BY 4.0)
  - Open Shading Language spec (BSD)

Usage:
    python integrations/collect_design_physics.py --all --output data/design_physics_raw/
    python integrations/collect_design_physics.py --source feynman --output data/design_physics_raw/
    python integrations/collect_design_physics.py --stats  # show existing stats
    python integrations/collect_design_physics.py --dry-run  # show sources without fetching
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    print("ERROR: aiohttp required. Install with: pip install aiohttp", file=sys.stderr)
    sys.exit(1)

# ─── Setup ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("collect_design_physics")

BASE_DIR = Path(__file__).parents[1]
DEFAULT_OUTPUT = BASE_DIR / "data" / "design_physics_raw"

# Token estimation: ~4 chars per token (rough estimate for chunking)
CHARS_PER_TOKEN = 4
TARGET_CHUNK_TOKENS = 2000
CHUNK_OVERLAP_TOKENS = 200
CHUNK_SIZE_CHARS = TARGET_CHUNK_TOKENS * CHARS_PER_TOKEN
OVERLAP_SIZE_CHARS = CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN

REQUEST_DELAY = 1.5  # seconds between requests per domain

# ─── Source catalog ───────────────────────────────────────────────────────────

@dataclass
class TextSource:
    name: str
    display_name: str
    urls: list[str]
    topic: str
    expertise_level: str  # "foundational" | "intermediate" | "expert"
    license: str
    synthesis_role: str   # What aspect of 3D intelligence this builds
    synthesis_prompt_suffix: str  # Domain-specific synthesis instruction


# Each source teaches Nalana a different dimension of 3D expertise.
SOURCES: list[TextSource] = [
    # ─── PHYSICS: Feynman Lectures ────────────────────────────────────────────
    TextSource(
        name="feynman_optics",
        display_name="Feynman Lectures on Physics — Optics Chapters",
        urls=[
            "https://www.feynmanlectures.caltech.edu/I_26.html",  # Optics: The Principle of Least Time
            "https://www.feynmanlectures.caltech.edu/I_27.html",  # Geometric Optics
            "https://www.feynmanlectures.caltech.edu/I_28.html",  # Electromagnetic Radiation
            "https://www.feynmanlectures.caltech.edu/I_34.html",  # Relativistic Effects in Radiation
            "https://www.feynmanlectures.caltech.edu/II_32.html", # Refractive Index of Dense Materials
        ],
        topic="light_physics",
        expertise_level="expert",
        license="Copyright Caltech — free for educational use",
        synthesis_role="Teaches Nalana the physics of light: reflection, refraction, scattering. Directly maps to PBR materials.",
        synthesis_prompt_suffix=(
            "Connect each physics concept to 3D rendering: Snell's law → refraction in glass materials, "
            "Fresnel equations → surface reflectance falloff, geometric optics → camera lenses in Blender, "
            "electromagnetic radiation → emission shaders, refractive index → IOR values in BSDF materials."
        ),
    ),
    TextSource(
        name="feynman_em",
        display_name="Feynman Lectures — Electromagnetism",
        urls=[
            "https://www.feynmanlectures.caltech.edu/II_01.html",  # Electromagnetism
            "https://www.feynmanlectures.caltech.edu/II_02.html",  # Differential Calculus of Vector Fields
            "https://www.feynmanlectures.caltech.edu/II_33.html",  # Reflection from Surfaces
        ],
        topic="light_physics",
        expertise_level="expert",
        license="Copyright Caltech — free for educational use",
        synthesis_role="Deeper electromagnetic theory underlying all surface interaction in PBR.",
        synthesis_prompt_suffix=(
            "Focus on surface reflection physics for 3D: Maxwell equations at boundaries, "
            "how conductor vs dielectric materials differ electromagnetically, "
            "why metals have Fresnel with full color tinting while dielectrics are achromatic."
        ),
    ),
    # ─── PHYSICS: Wikipedia (CC BY-SA) ────────────────────────────────────────
    TextSource(
        name="wiki_fresnel",
        display_name="Wikipedia — Fresnel Equations",
        urls=[
            "https://en.wikipedia.org/wiki/Fresnel_equations",
            "https://en.wikipedia.org/wiki/Snell%27s_law",
            "https://en.wikipedia.org/wiki/Total_internal_reflection",
            "https://en.wikipedia.org/wiki/Brewster%27s_angle",
        ],
        topic="light_physics",
        expertise_level="intermediate",
        license="CC BY-SA 4.0",
        synthesis_role="Fresnel equations are the foundation of all physically-based rendering. Critical for glass, water, surfaces.",
        synthesis_prompt_suffix=(
            "Derive practical 3D artist insights: at what angle does glass become fully reflective (Brewster's angle)? "
            "Why does the Schlick approximation (used in all game engines) work? "
            "How do Fresnel equations translate to the 'Fresnel' input in Blender's Principled BSDF?"
        ),
    ),
    TextSource(
        name="wiki_subsurface",
        display_name="Wikipedia — Subsurface Scattering & Microfacet Model",
        urls=[
            "https://en.wikipedia.org/wiki/Subsurface_scattering",
            "https://en.wikipedia.org/wiki/Microfacet_model",
            "https://en.wikipedia.org/wiki/Bidirectional_reflectance_distribution_function",
            "https://en.wikipedia.org/wiki/Lambertian_reflectance",
            "https://en.wikipedia.org/wiki/Specular_highlight",
        ],
        topic="material_physics",
        expertise_level="expert",
        license="CC BY-SA 4.0",
        synthesis_role="BRDF theory and subsurface scattering underlie all material appearance in 3D. Skin, wax, marble, jade.",
        synthesis_prompt_suffix=(
            "Teach Nalana when to use subsurface scattering: skin (1-2mm depth), candle wax (5mm), jade (10mm+). "
            "Explain microfacet roughness physically: why rough surfaces scatter light more broadly. "
            "Connect GGX/Beckmann distribution to the Roughness slider in Blender's Principled BSDF."
        ),
    ),
    TextSource(
        name="wiki_color_theory",
        display_name="Wikipedia — Color Theory & Perception",
        urls=[
            "https://en.wikipedia.org/wiki/Color_theory",
            "https://en.wikipedia.org/wiki/Color_wheel",
            "https://en.wikipedia.org/wiki/Complementary_colors",
            "https://en.wikipedia.org/wiki/Color_temperature",
            "https://en.wikipedia.org/wiki/CIE_1931_color_space",
            "https://en.wikipedia.org/wiki/sRGB",
        ],
        topic="color_design",
        expertise_level="intermediate",
        license="CC BY-SA 4.0",
        synthesis_role="Color theory is fundamental to material palettes, lighting setup, and scene composition.",
        synthesis_prompt_suffix=(
            "Teach 3D color decisions: why complementary colors create visual tension, "
            "how color temperature (2700K warm vs 6500K cool) affects scene mood, "
            "why sRGB gamma correction matters in Blender's compositor, "
            "how to use the color wheel for material palette selection in product renders."
        ),
    ),
    # ─── DESIGN: Architecture ─────────────────────────────────────────────────
    TextSource(
        name="vitruvius",
        display_name="Vitruvius — De Architectura (Ten Books on Architecture)",
        urls=[
            "https://en.wikisource.org/wiki/Ten_Books_on_Architecture/Book_I",
            "https://en.wikisource.org/wiki/Ten_Books_on_Architecture/Book_III",
            "https://en.wikisource.org/wiki/Ten_Books_on_Architecture/Book_IV",
            "https://en.wikisource.org/wiki/Ten_Books_on_Architecture/Book_VI",
        ],
        topic="architectural_design",
        expertise_level="foundational",
        license="Public Domain (~15 BCE)",
        synthesis_role=(
            "Vitruvius's firmitas-utilitas-venustas (strength-utility-beauty) is the original "
            "design framework. Column proportions, room ratios, site orientation — all directly "
            "applicable to architectural 3D modeling."
        ),
        synthesis_prompt_suffix=(
            "Extract timeless architectural principles for 3D modeling: "
            "Vitruvius's column module system (Doric: column diameter × 6-7 = column height) → "
            "how to proportion columns in Blender. Room proportions from Book VI → "
            "generate correct room aspect ratios. Orientation and light entry → "
            "how to position windows in architectural visualization."
        ),
    ),
    TextSource(
        name="wiki_golden_ratio",
        display_name="Wikipedia — Golden Ratio & Sacred Geometry",
        urls=[
            "https://en.wikipedia.org/wiki/Golden_ratio",
            "https://en.wikipedia.org/wiki/Fibonacci_sequence",
            "https://en.wikipedia.org/wiki/Sacred_geometry",
            "https://en.wikipedia.org/wiki/Platonic_solid",
            "https://en.wikipedia.org/wiki/Archimedean_solid",
        ],
        topic="design_mathematics",
        expertise_level="foundational",
        license="CC BY-SA 4.0",
        synthesis_role="Proportional systems used in all great design: φ=1.618, Fibonacci spirals, Platonic solids.",
        synthesis_prompt_suffix=(
            "Teach 3D proportion decisions using mathematical beauty: "
            "why φ=1.618 creates pleasing object proportions, "
            "how Fibonacci spiral applies to shell modeling and scroll curves, "
            "why Platonic solids (tetrahedron, cube, octahedron, dodecahedron, icosahedron) "
            "appear in crystal structures and architectural domes, "
            "practical Blender exercises using golden ratio to set object scale and camera framing."
        ),
    ),
    TextSource(
        name="wiki_gestalt",
        display_name="Wikipedia — Gestalt Principles & Visual Perception",
        urls=[
            "https://en.wikipedia.org/wiki/Gestalt_psychology",
            "https://en.wikipedia.org/wiki/Principles_of_grouping",
            "https://en.wikipedia.org/wiki/Figure%E2%80%93ground_(perception)",
            "https://en.wikipedia.org/wiki/Rule_of_thirds",
            "https://en.wikipedia.org/wiki/Visual_hierarchy",
        ],
        topic="visual_design",
        expertise_level="intermediate",
        license="CC BY-SA 4.0",
        synthesis_role="Gestalt principles govern how viewers perceive 3D scenes: composition, grouping, hierarchy.",
        synthesis_prompt_suffix=(
            "Connect Gestalt principles to 3D scene composition: "
            "Proximity → how to group objects in a scene, "
            "Similarity → when to use repeated materials/forms, "
            "Figure-ground → how to separate hero object from background, "
            "Rule of thirds → camera framing in Blender (enable overlay grid), "
            "Visual hierarchy → material brightness hierarchy (brightest = most important)."
        ),
    ),
    TextSource(
        name="bauhaus_design",
        display_name="Bauhaus Design Principles & Form Language",
        urls=[
            "https://en.wikipedia.org/wiki/Bauhaus",
            "https://en.wikipedia.org/wiki/Form_follows_function",
            "https://en.wikipedia.org/wiki/Industrial_design",
            "https://en.wikipedia.org/wiki/Dieter_Rams",
        ],
        topic="design_philosophy",
        expertise_level="foundational",
        license="CC BY-SA 4.0",
        synthesis_role="Bauhaus and Rams's 10 principles define modern design language: reduction, function, honesty.",
        synthesis_prompt_suffix=(
            "Extract design philosophy for 3D modeling decisions: "
            "Rams's 'Good design is as little design as possible' → when to remove geometry, "
            "'Form follows function' → why product proportions are not arbitrary, "
            "Bauhaus primary shapes (circle, triangle, square) → basic 3D primitives as design vocabulary, "
            "'Good design is long-lasting' → avoiding trendy vs timeless design choices in modeling."
        ),
    ),
    # ─── 3D/VFX TECHNICAL KNOWLEDGE ───────────────────────────────────────────
    TextSource(
        name="gltf_spec",
        display_name="glTF 2.0 Specification — PBR Material Model",
        urls=[
            "https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html",
            "https://github.com/KhronosGroup/glTF/blob/main/specification/2.0/Specification.adoc",
        ],
        topic="3d_formats_pbr",
        expertise_level="expert",
        license="CC BY 4.0",
        synthesis_role="glTF defines the universal PBR material model used across all 3D tools. Understanding it = understanding all materials.",
        synthesis_prompt_suffix=(
            "Teach Nalana the glTF PBR material model as a universal standard: "
            "baseColorFactor → Blender Base Color, metallicFactor → Metallic slider, "
            "roughnessFactor → Roughness slider, normalTexture → Normal Map node, "
            "emissiveFactor → Emission Strength. "
            "Explain why this model is physically grounded and how it corresponds to "
            "real material measurements (baseColor = albedo reflectance spectrum)."
        ),
    ),
    TextSource(
        name="opensubdiv_theory",
        display_name="OpenSubdiv — Subdivision Surface Theory",
        urls=[
            "https://graphics.pixar.com/opensubdiv/docs/subdivision_surfaces.html",
            "https://en.wikipedia.org/wiki/Catmull%E2%80%93Clark_subdivision_surface",
            "https://en.wikipedia.org/wiki/Loop_subdivision_surface",
            "https://en.wikipedia.org/wiki/Doo%E2%80%93Sabin_subdivision_surface",
        ],
        topic="3d_modeling_theory",
        expertise_level="expert",
        license="Apache License 2.0 (OpenSubdiv) + CC BY-SA (Wikipedia)",
        synthesis_role="Subdivision surfaces are the foundation of production 3D modeling. Understanding poles, edge loops, topology is critical.",
        synthesis_prompt_suffix=(
            "Teach production topology rules through subdivision theory: "
            "why poles (vertices with ≠4 edges) cause pinching artifacts in Catmull-Clark, "
            "why edge loops define form and define where subdivision refinement concentrates, "
            "how to achieve creases using edge weighting vs geometry, "
            "why quads subdivide cleanly while triangles and n-gons don't, "
            "the n-pole vs e-pole problem in character modeling (face topology)."
        ),
    ),
    TextSource(
        name="usd_spec",
        display_name="USD (Universal Scene Description) — Technical Overview",
        urls=[
            "https://openusd.org/release/intro.html",
            "https://openusd.org/release/glossary.html",
            "https://openusd.org/release/tut_usd_tutorials.html",
            "https://en.wikipedia.org/wiki/Universal_Scene_Description",
        ],
        topic="3d_pipeline_theory",
        expertise_level="expert",
        license="Apache License 2.0",
        synthesis_role="USD is the future of all 3D pipelines (Pixar, Apple, NVIDIA). Nalana must understand scene description language.",
        synthesis_prompt_suffix=(
            "Teach Nalana to think in USD concepts: "
            "Prim hierarchy → scene organization, "
            "Variants → LOD systems and alternate representations, "
            "Schemas → typed 3D primitives (UsdGeomMesh, UsdShadeMaterial), "
            "Composition arcs (references, layers, variants) → how production pipelines combine assets, "
            "How USD maps to Blender's scene hierarchy and collection system."
        ),
    ),
    TextSource(
        name="osl_shading",
        display_name="Open Shading Language — Surface Shader Theory",
        urls=[
            "https://github.com/AcademySoftwareFoundation/OpenShadingLanguage/blob/main/src/doc/osl-languagespec.pdf.md",
            "https://en.wikipedia.org/wiki/Open_Shading_Language",
            "https://en.wikipedia.org/wiki/Shader",
            "https://en.wikipedia.org/wiki/Physically_based_rendering",
        ],
        topic="shader_theory",
        expertise_level="expert",
        license="BSD License",
        synthesis_role="OSL is the shader language of Blender Cycles, Arnold, RenderMan. Understanding shader construction = deep material intelligence.",
        synthesis_prompt_suffix=(
            "Teach shader concepts through OSL: "
            "closure-based shading model (BSDF closures vs emission closures) → "
            "how Blender's node system maps to closure algebra, "
            "why mixing shaders with MixShader is physically correct (energy conservation), "
            "how displacement works (shader-based vs geometric), "
            "OSL noise functions → procedural texture creation in Blender."
        ),
    ),
    TextSource(
        name="wiki_rendering",
        display_name="Wikipedia — Rendering & Ray Tracing Theory",
        urls=[
            "https://en.wikipedia.org/wiki/Ray_tracing_(graphics)",
            "https://en.wikipedia.org/wiki/Global_illumination",
            "https://en.wikipedia.org/wiki/Path_tracing",
            "https://en.wikipedia.org/wiki/Ambient_occlusion",
            "https://en.wikipedia.org/wiki/Depth_of_field",
            "https://en.wikipedia.org/wiki/Bokeh",
        ],
        topic="rendering_theory",
        expertise_level="intermediate",
        license="CC BY-SA 4.0",
        synthesis_role="Rendering algorithms determine how beautiful the final image is. Nalana should understand WHY to use each technique.",
        synthesis_prompt_suffix=(
            "Connect rendering theory to Blender practical settings: "
            "Ray tracing → Cycles ray depth (Max Bounces), "
            "Path tracing → why increasing samples reduces noise, "
            "Global illumination → why HDRI + fill light looks more realistic than just a sun, "
            "AO → when to add AO node vs rely on path tracing for occlusion, "
            "Depth of field → F-stop to Blender aperture mapping, bokeh shape from aperture blades."
        ),
    ),
    TextSource(
        name="wiki_typography_proportion",
        display_name="Wikipedia — Typography, Proportion & Grid Systems",
        urls=[
            "https://en.wikipedia.org/wiki/Typography",
            "https://en.wikipedia.org/wiki/Grid_(graphic_design)",
            "https://en.wikipedia.org/wiki/Typographic_scale",
            "https://en.wikipedia.org/wiki/White_space_(visual_arts)",
            "https://en.wikipedia.org/wiki/Readability",
        ],
        topic="visual_design",
        expertise_level="intermediate",
        license="CC BY-SA 4.0",
        synthesis_role="Typography and grid theory apply to 3D text objects, logo modeling, and interface elements in 3D scenes.",
        synthesis_prompt_suffix=(
            "Apply typography principles to 3D text and UI design in Blender: "
            "type scale systems → when modeling text objects, how to set consistent sizes, "
            "grid systems → organizing multi-object scenes with alignment, "
            "white space → why 3D scenes need 'breathing room' (negative space), "
            "typographic hierarchy → how to use scale, weight, and position to create focus."
        ),
    ),
]

# Map source names to source objects for quick lookup
SOURCE_MAP = {s.name: s for s in SOURCES}

# ─── HTML → text extraction ────────────────────────────────────────────────────

# Tags to strip completely (tag + content)
STRIP_TAGS = re.compile(
    r'<(script|style|nav|header|footer|aside|noscript|iframe|figure|figcaption)[^>]*>.*?</\1>',
    re.DOTALL | re.IGNORECASE,
)
# Tags to remove but keep content
INLINE_TAGS = re.compile(r'<[^>]+>', re.DOTALL)
# Collapse whitespace
WHITESPACE = re.compile(r'\s+')
# Footnote references like [1], [2]
FOOTNOTES = re.compile(r'\[\d+\]')
# URL-like artifacts
URL_ARTIFACTS = re.compile(r'https?://\S+')
# Edit links and Wikipedia-specific junk
WIKI_JUNK = re.compile(r'\[edit\]|\[citation needed\]|\[dubious\]|\[clarification needed\]')


def html_to_clean_text(html: str) -> str:
    """Extract clean readable text from HTML, removing navigation and boilerplate."""
    # Remove scripts, styles, nav, header, footer
    text = STRIP_TAGS.sub(' ', html)
    # Remove remaining HTML tags
    text = INLINE_TAGS.sub(' ', text)
    # Remove Wikipedia edit/citation artifacts
    text = WIKI_JUNK.sub('', text)
    text = FOOTNOTES.sub('', text)
    text = URL_ARTIFACTS.sub('', text)
    # Collapse whitespace
    text = WHITESPACE.sub(' ', text).strip()
    return text


def chunk_text(
    text: str,
    source_name: str,
    url: str,
    topic: str,
    expertise_level: str,
    synthesis_prompt: str,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = OVERLAP_SIZE_CHARS,
) -> list[dict]:
    """
    Split text into overlapping chunks of ~2000 tokens with metadata.
    Each chunk is ready for GPU synthesis.
    """
    chunks = []
    start = 0
    chunk_idx = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Try to break at sentence boundary
        if end < text_len:
            # Search backward for sentence-ending punctuation
            for boundary in ('. ', '! ', '? ', '\n\n', '\n'):
                last_boundary = text.rfind(boundary, start + chunk_size // 2, end)
                if last_boundary != -1:
                    end = last_boundary + len(boundary)
                    break

        chunk_text_content = text[start:end].strip()
        if len(chunk_text_content) < 100:
            break  # Skip tiny trailing chunks

        estimated_tokens = len(chunk_text_content) // CHARS_PER_TOKEN

        chunk = {
            "chunk_id": hashlib.md5(f"{source_name}_{chunk_idx}_{chunk_text_content[:50]}".encode()).hexdigest()[:12],
            "source": source_name,
            "source_url": url,
            "topic": topic,
            "expertise_level": expertise_level,
            "chunk_index": chunk_idx,
            "char_start": start,
            "char_end": end,
            "estimated_tokens": estimated_tokens,
            "source_text": chunk_text_content,
            "synthesis_prompt": (
                f"You are building training data for Nalana, an AI that controls 3D software via voice commands. "
                f"Nalana should reason like a master physicist AND master designer. "
                f"Generate 10 expert Q&A pairs from the following text where a 3D artist or engineer asks "
                f"about the physics/design principle and Nalana explains it deeply, connecting it to "
                f"practical 3D work (materials, lighting, rendering, modeling, simulation). "
                f"Each Q&A pair should be in JSON format: "
                f'{{\"voice_command\": \"...\", \"task_type\": \"UNDERSTAND\", \"reasoning\": \"...\", \"quality\": 4.5}}. '
                f"Additional focus: {synthesis_prompt}"
            ),
        }
        chunks.append(chunk)

        # Move forward with overlap
        start = end - overlap
        chunk_idx += 1

    return chunks


# ─── Async fetcher ─────────────────────────────────────────────────────────────

class TextFetcher:
    def __init__(self, output_dir: Path, delay: float = REQUEST_DELAY):
        self.output_dir = Path(output_dir)
        self.delay = delay
        self._last_request_by_domain: dict[str, float] = {}
        self._session: aiohttp.ClientSession | None = None
        self._cache_dir = self.output_dir / "_url_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()

    def _cache_path(self, url: str) -> Path:
        return self._cache_dir / f"{self._cache_key(url)}.txt"

    def _get_domain(self, url: str) -> str:
        from urllib.parse import urlparse
        return urlparse(url).netloc

    async def _rate_limited_get(self, url: str) -> str | None:
        """Fetch URL with per-domain rate limiting and caching."""
        cache_path = self._cache_path(url)
        if cache_path.exists():
            log.debug("Cache hit: %s", url)
            return cache_path.read_text(encoding="utf-8", errors="replace")

        domain = self._get_domain(url)
        now = time.monotonic()
        last = self._last_request_by_domain.get(domain, 0.0)
        wait = self.delay - (now - last)
        if wait > 0:
            await asyncio.sleep(wait)

        self._last_request_by_domain[domain] = time.monotonic()

        if self._session is None:
            raise RuntimeError("HTTP session not initialized. Use async context manager.")
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; NalanaDataCollector/1.0; education)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }
            async with self._session.get(
                url, headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
                ssl=False,
            ) as resp:
                if resp.status != 200:
                    log.warning("HTTP %d for %s", resp.status, url)
                    return None
                text = await resp.text(encoding="utf-8", errors="replace")
                cache_path.write_text(text, encoding="utf-8")
                log.info("Fetched %s (%d chars)", url, len(text))
                return text

        except asyncio.TimeoutError:
            log.warning("Timeout fetching %s", url)
        except Exception as e:
            log.warning("Error fetching %s: %s", url, e)
        return None

    async def fetch_source(self, source: TextSource) -> list[dict]:
        """Fetch all URLs for a source, extract text, chunk, and return ready-for-synthesis records."""
        all_chunks = []

        for url in source.urls:
            log.info("[%s] Fetching: %s", source.name, url)
            html = await self._rate_limited_get(url)
            if html is None:
                continue

            clean = html_to_clean_text(html)
            if len(clean) < 500:
                log.warning("[%s] Skipping %s — too short (%d chars)", source.name, url, len(clean))
                continue

            chunks = chunk_text(
                text=clean,
                source_name=source.name,
                url=url,
                topic=source.topic,
                expertise_level=source.expertise_level,
                synthesis_prompt=source.synthesis_prompt_suffix,
            )
            all_chunks.extend(chunks)
            log.debug("[%s] %s → %d chunks", source.name, url, len(chunks))

        return all_chunks

    async def fetch_all(
        self,
        sources: list[TextSource],
        output_dir: Path,
    ) -> dict[str, int]:
        """Fetch and chunk all sources, writing JSONL files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        source_counts: dict[str, int] = {}

        connector = aiohttp.TCPConnector(limit=10, ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            self._session = session

            all_bar = tqdm(total=sum(len(s.urls) for s in sources), desc="Fetching sources", unit="url")

            for source in sources:
                log.info("Processing source: %s (%d URLs)", source.display_name, len(source.urls))
                chunks = await self.fetch_source(source)
                all_bar.update(len(source.urls))

                if not chunks:
                    log.warning("No chunks for source: %s", source.name)
                    source_counts[source.name] = 0
                    continue

                # Write JSONL
                out_file = output_dir / f"{source.name}.jsonl"
                with open(out_file, "w", encoding="utf-8") as f:
                    for chunk in chunks:
                        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

                source_counts[source.name] = len(chunks)
                total_tokens = sum(c["estimated_tokens"] for c in chunks)
                log.info("[%s] Wrote %d chunks (~%dk tokens) → %s",
                         source.name, len(chunks), total_tokens // 1000, out_file)

            all_bar.close()

        return source_counts


# ─── Stats function ────────────────────────────────────────────────────────────

def compute_stats(output_dir: Path) -> None:
    """Print statistics for all existing chunk files."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        return

    jsonl_files = list(output_dir.glob("*.jsonl"))
    if not jsonl_files:
        print("No JSONL files found. Run collection first.")
        return

    print(f"\nDesign Physics Dataset Stats")
    print(f"{'='*60}")
    print(f"Directory: {output_dir}")
    print(f"{'─'*60}")
    print(f"{'Source':<30} {'Chunks':>8} {'Tokens':>10} {'Topic':<20}")
    print(f"{'─'*60}")

    total_chunks = 0
    total_tokens = 0
    by_topic: dict[str, dict[str, int]] = {}

    for jsonl_file in sorted(jsonl_files):
        source_name = jsonl_file.stem
        chunks = 0
        tokens = 0
        topic = "unknown"

        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    chunks += 1
                    tokens += rec.get("estimated_tokens", 0)
                    topic = rec.get("topic", "unknown")
                except json.JSONDecodeError:
                    pass

        print(f"{source_name:<30} {chunks:>8} {tokens:>10,} {topic:<20}")
        total_chunks += chunks
        total_tokens += tokens

        by_topic.setdefault(topic, {"chunks": 0, "tokens": 0})
        by_topic[topic]["chunks"] += chunks
        by_topic[topic]["tokens"] += tokens

    print(f"{'─'*60}")
    print(f"{'TOTAL':<30} {total_chunks:>8} {total_tokens:>10,}")

    print(f"\nBy topic:")
    for topic, stats in sorted(by_topic.items()):
        print(f"  {topic:<28} {stats['chunks']:>6} chunks  {stats['tokens']:>8,} tokens")

    print(f"\nEstimated synthesis output:")
    print(f"  {total_chunks} chunks × 10 Q&A pairs/chunk = ~{total_chunks * 10:,} training pairs")
    print(f"  At quality=4.5 (highest tier) — this is Nalana's physics/design genius layer")
    print(f"  GPU time: ~{total_chunks * 3 // 60}h on 4× A6000 (Qwen2.5-72B @ ~30 chunks/min)")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect public-domain expert texts for Nalana GPU synthesis pipeline.\n"
            "These chunks become training input for Qwen2.5-72B to generate "
            "physics/design Q&A pairs at quality=4.5."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--all", dest="all_sources", action="store_true",
        help="Fetch all configured sources",
    )
    parser.add_argument(
        "--source", nargs="+",
        choices=list(SOURCE_MAP.keys()),
        help="Fetch specific source(s) by name",
    )
    parser.add_argument(
        "--output", type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory for JSONL chunk files",
    )
    parser.add_argument(
        "--delay", type=float, default=REQUEST_DELAY,
        help="Seconds between requests per domain",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print stats for existing output directory and exit",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show sources that would be fetched without fetching",
    )
    parser.add_argument(
        "--list-sources", action="store_true",
        help="List all available sources and exit",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.stats:
        compute_stats(args.output)
        return

    if args.list_sources:
        print("\nAvailable sources:")
        print(f"{'Name':<25} {'Topic':<22} {'URLs':>5}  Display Name")
        print("─" * 90)
        for s in SOURCES:
            print(f"{s.name:<25} {s.topic:<22} {len(s.urls):>5}  {s.display_name[:40]}")
        print(f"\nTotal: {len(SOURCES)} sources, {sum(len(s.urls) for s in SOURCES)} URLs")
        return

    # Select sources to fetch
    if args.all_sources:
        selected = SOURCES
    elif args.source:
        selected = [SOURCE_MAP[name] for name in args.source]
    else:
        parser.print_help()
        print("\nError: specify --all or --source <name(s)>")
        print("Use --list-sources to see available sources")
        sys.exit(1)

    if args.dry_run:
        print(f"\nDry run — would fetch {len(selected)} sources:")
        total_urls = 0
        for s in selected:
            print(f"\n  [{s.name}] {s.display_name}")
            print(f"    Topic: {s.topic} | Level: {s.expertise_level}")
            print(f"    License: {s.license}")
            print(f"    URLs ({len(s.urls)}):")
            for url in s.urls:
                print(f"      {url}")
            total_urls += len(s.urls)
        print(f"\nTotal: {total_urls} URLs to fetch")
        print(f"Estimated chunks: {total_urls * 5}-{total_urls * 15}")
        print(f"Estimated synthesis pairs: {total_urls * 5 * 10}-{total_urls * 15 * 10}")
        return

    # Run fetcher
    log.info("Fetching %d sources → %s", len(selected), args.output)

    fetcher = TextFetcher(output_dir=args.output, delay=args.delay)
    source_counts = asyncio.run(fetcher.fetch_all(selected, args.output))

    # Summary
    total_chunks = sum(source_counts.values())
    print(f"\nCollection complete:")
    print(f"  Sources processed: {len(source_counts)}")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Total estimated synthesis pairs: ~{total_chunks * 10:,}")
    print(f"  Output: {args.output}/")
    print()
    print("Next step: run GPU synthesis with Qwen2.5-72B:")
    print(f"  python synthesize_bulk.py --source-dir {args.output} --gpu 0-11")
    print()
    compute_stats(args.output)


if __name__ == "__main__":
    main()
