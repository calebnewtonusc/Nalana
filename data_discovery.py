"""
data_discovery.py - Autonomous self-expanding dataset source hunter.

Step 2 in the Nalana pipeline (see scripts/run_all.sh). Discovers and scores
ALL relevant data sources across every category Nalana needs: YouTube channels,
3D geometry datasets, physics/optics references, academic papers, community
forums, design assets, and simulation libraries.

Uses an LLM (vLLM on cluster, fallback to Claude API) to:
  1. Score pre-seeded sources for relevance
  2. Generate new search queries from each discovered source
  3. Recursively expand coverage until depth limit is reached

Output: data/discovered_sources.json — array of scored, categorized source
objects consumed by downstream pipeline steps (discover_v2.py, api_harvest.py,
synthesize_bulk.py, collect_design_physics.py).

Usage:
    # Full run (all categories, LLM expansion enabled):
    python3 data_discovery.py --all --output data/discovered_sources.json

    # Quick seed-only run (no LLM expansion):
    python3 data_discovery.py --all --no-expand --output data/discovered_sources.json

    # Single category:
    python3 data_discovery.py --category youtube --output data/youtube_sources.json

    # Use Claude fallback:
    python3 data_discovery.py --all --backend claude --output data/discovered_sources.json

    # Use vLLM on cluster:
    python3 data_discovery.py --all --backend vllm \\
        --vllm-url http://localhost:8001 --output data/discovered_sources.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

# ─── Source record ─────────────────────────────────────────────────────────────

@dataclass
class DataSource:
    """A single discovered data source with metadata."""
    type: str             # youtube_channel | dataset | paper | forum | docs | api
    name: str             # Human-readable name
    url: str              # Canonical URL
    software: str         # blender | maya | houdini | c4d | rhino | unreal | multi | physics | geometry
    category: str         # tutorial | simulation | geometry | physics | design_theory | community | reference
    stream: int           # 1-5 Nalana training stream this feeds
    relevance_score: float = 0.0   # LLM-assigned 0.0–1.0
    priority: str = "medium"       # high | medium | low
    notes: str = ""
    tags: list[str] = field(default_factory=list)


# ─── Seed catalog ──────────────────────────────────────────────────────────────
# These are known-good sources. The LLM expansion step adds more.

SEED_SOURCES: list[dict[str, Any]] = [

    # ── Stream 1: YouTube channels (tutorials) ─────────────────────────────────
    {"type": "youtube_channel", "name": "Blender Guru",           "url": "https://www.youtube.com/@blenderguru",           "software": "blender",  "category": "tutorial",       "stream": 1, "priority": "high"},
    {"type": "youtube_channel", "name": "Grant Abbitt",           "url": "https://www.youtube.com/@grabbitt",              "software": "blender",  "category": "tutorial",       "stream": 1, "priority": "high"},
    {"type": "youtube_channel", "name": "CGFastTrack",            "url": "https://www.youtube.com/@CGFastTrack",           "software": "blender",  "category": "tutorial",       "stream": 1, "priority": "high"},
    {"type": "youtube_channel", "name": "Default Cube",           "url": "https://www.youtube.com/@DefaultCube",           "software": "blender",  "category": "tutorial",       "stream": 1, "priority": "high"},
    {"type": "youtube_channel", "name": "Ducky 3D",               "url": "https://www.youtube.com/@Ducky3D",               "software": "blender",  "category": "tutorial",       "stream": 1, "priority": "high"},
    {"type": "youtube_channel", "name": "CG Geek",                "url": "https://www.youtube.com/@CGGeek",                "software": "blender",  "category": "tutorial",       "stream": 1, "priority": "high"},
    {"type": "youtube_channel", "name": "Josh Gambrell",          "url": "https://www.youtube.com/@JoshGambrell",          "software": "blender",  "category": "tutorial",       "stream": 1, "priority": "high"},
    {"type": "youtube_channel", "name": "Polygon Runway",         "url": "https://www.youtube.com/@PolygonRunway",         "software": "blender",  "category": "tutorial",       "stream": 1, "priority": "high"},
    {"type": "youtube_channel", "name": "FlippedNormals",         "url": "https://www.youtube.com/@FlippedNormals",        "software": "multi",    "category": "tutorial",       "stream": 1, "priority": "high"},
    {"type": "youtube_channel", "name": "YanSculpts",             "url": "https://www.youtube.com/@YanSculpts",            "software": "blender",  "category": "tutorial",       "stream": 1, "priority": "high"},
    {"type": "youtube_channel", "name": "Derek Elliott",          "url": "https://www.youtube.com/@DerekElliott",          "software": "blender",  "category": "tutorial",       "stream": 1, "priority": "medium"},
    {"type": "youtube_channel", "name": "SouthernShotty",         "url": "https://www.youtube.com/@SouthernShotty3D",      "software": "blender",  "category": "tutorial",       "stream": 1, "priority": "high"},
    {"type": "youtube_channel", "name": "Remington Creative",     "url": "https://www.youtube.com/@RemingtonCreative",     "software": "blender",  "category": "tutorial",       "stream": 1, "priority": "medium"},
    {"type": "youtube_channel", "name": "Noggi 3D",               "url": "https://www.youtube.com/@Noggi3D",               "software": "blender",  "category": "tutorial",       "stream": 1, "priority": "medium"},
    # Maya
    {"type": "youtube_channel", "name": "Autodesk Maya",          "url": "https://www.youtube.com/@AutodeskMaya",          "software": "maya",     "category": "tutorial",       "stream": 1, "priority": "high"},
    {"type": "youtube_channel", "name": "Maya Central",           "url": "https://www.youtube.com/@MayaCentral3D",         "software": "maya",     "category": "tutorial",       "stream": 1, "priority": "high"},
    {"type": "youtube_channel", "name": "Arrimus 3D",             "url": "https://www.youtube.com/@Arrimus3D",             "software": "multi",    "category": "tutorial",       "stream": 1, "priority": "high"},
    {"type": "youtube_channel", "name": "Alex Cheparev",          "url": "https://www.youtube.com/@alexcheparev",          "software": "maya",     "category": "tutorial",       "stream": 1, "priority": "high"},
    {"type": "youtube_channel", "name": "CGCircuit",              "url": "https://www.youtube.com/@cgcircuit",             "software": "maya",     "category": "tutorial",       "stream": 1, "priority": "high"},
    # Houdini
    {"type": "youtube_channel", "name": "SideFX",                 "url": "https://www.youtube.com/@SideFXHoudini",         "software": "houdini",  "category": "tutorial",       "stream": 1, "priority": "high"},
    {"type": "youtube_channel", "name": "Houdini Tutorial",       "url": "https://www.youtube.com/@houdinitutorial",       "software": "houdini",  "category": "tutorial",       "stream": 1, "priority": "high"},
    {"type": "youtube_channel", "name": "Tim van Helsdingen",     "url": "https://www.youtube.com/@TimvanHelsdingen",      "software": "houdini",  "category": "tutorial",       "stream": 1, "priority": "high"},
    {"type": "youtube_channel", "name": "Indie-Pixel",            "url": "https://www.youtube.com/@Indie-Pixel",           "software": "houdini",  "category": "tutorial",       "stream": 1, "priority": "medium"},
    # Cinema 4D
    {"type": "youtube_channel", "name": "Maxon",                  "url": "https://www.youtube.com/@Maxon3D",               "software": "c4d",      "category": "tutorial",       "stream": 1, "priority": "high"},
    {"type": "youtube_channel", "name": "School of Motion",       "url": "https://www.youtube.com/@SchoolofMotion",        "software": "c4d",      "category": "tutorial",       "stream": 1, "priority": "high"},
    {"type": "youtube_channel", "name": "Greyscalegorilla",       "url": "https://www.youtube.com/@Greyscalegorilla",      "software": "c4d",      "category": "tutorial",       "stream": 1, "priority": "high"},
    # Substance
    {"type": "youtube_channel", "name": "Adobe Substance 3D",    "url": "https://www.youtube.com/@AdobeSubstance3D",      "software": "substance","category": "tutorial",       "stream": 1, "priority": "high"},
    {"type": "youtube_channel", "name": "JROTools",               "url": "https://www.youtube.com/@JROTools",              "software": "substance","category": "tutorial",       "stream": 1, "priority": "medium"},
    # Rhino / Grasshopper
    {"type": "youtube_channel", "name": "McNeel",                 "url": "https://www.youtube.com/@McNeel",                "software": "rhino",    "category": "tutorial",       "stream": 1, "priority": "high"},
    {"type": "youtube_channel", "name": "parametric architecture","url": "https://www.youtube.com/@ParametricArchitecture", "software": "rhino",    "category": "tutorial",       "stream": 1, "priority": "medium"},
    # Unreal Engine
    {"type": "youtube_channel", "name": "Unreal Engine",          "url": "https://www.youtube.com/@UnrealEngine",          "software": "unreal",   "category": "tutorial",       "stream": 1, "priority": "high"},
    {"type": "youtube_channel", "name": "William Faucher",        "url": "https://www.youtube.com/@WilliamFaucher",        "software": "unreal",   "category": "tutorial",       "stream": 1, "priority": "high"},
    # Cross-discipline / theory
    {"type": "youtube_channel", "name": "Corridor Crew",          "url": "https://www.youtube.com/@CorridorCrew",          "software": "multi",    "category": "design_theory",  "stream": 1, "priority": "medium"},
    {"type": "youtube_channel", "name": "Gnomon Workshop",        "url": "https://www.youtube.com/@GnomonWorkshop",        "software": "multi",    "category": "tutorial",       "stream": 1, "priority": "high"},
    {"type": "youtube_channel", "name": "Art of Rendering",       "url": "https://www.youtube.com/@ArtofRendering",        "software": "multi",    "category": "design_theory",  "stream": 1, "priority": "medium"},
    # Industry conferences
    {"type": "youtube_channel", "name": "GDC",                    "url": "https://www.youtube.com/@Gdconf",                "software": "multi",    "category": "design_theory",  "stream": 1, "priority": "medium"},
    {"type": "youtube_channel", "name": "ACM SIGGRAPH",           "url": "https://www.youtube.com/@ACMSIGGRAPH",           "software": "multi",    "category": "design_theory",  "stream": 1, "priority": "medium"},

    # ── Stream 2: 3D geometry datasets ─────────────────────────────────────────
    {"type": "dataset", "name": "Objaverse",        "url": "https://huggingface.co/datasets/allenai/objaverse",      "software": "multi",    "category": "geometry",       "stream": 2, "priority": "high",   "notes": "800k+ CC-BY 3D objects"},
    {"type": "dataset", "name": "Objaverse-XL",     "url": "https://huggingface.co/datasets/allenai/objaverse-xl",   "software": "multi",    "category": "geometry",       "stream": 2, "priority": "high",   "notes": "10M+ 3D objects"},
    {"type": "dataset", "name": "ShapeNet",         "url": "https://huggingface.co/datasets/ShapeNet/ShapeNetCore",  "software": "multi",    "category": "geometry",       "stream": 2, "priority": "high",   "notes": "55k+ 3D CAD models, 55 categories"},
    {"type": "dataset", "name": "ABC Dataset",      "url": "https://deep-geometry.github.io/abc-dataset/",           "software": "cad",      "category": "geometry",       "stream": 2, "priority": "high",   "notes": "1M+ CAD models with ground truth"},
    {"type": "dataset", "name": "Thingi10K",        "url": "https://ten-thousand-models.appspot.com/",               "software": "multi",    "category": "geometry",       "stream": 2, "priority": "medium", "notes": "10k real-world 3D printing models"},
    {"type": "dataset", "name": "ScanNet",          "url": "http://www.scan-net.org/",                               "software": "multi",    "category": "geometry",       "stream": 2, "priority": "medium", "notes": "Indoor RGB-D scans with semantic labels"},
    {"type": "dataset", "name": "ABO",              "url": "https://amazon-berkeley-objects.s3.amazonaws.com/index.html", "software": "multi", "category": "geometry",      "stream": 2, "priority": "medium", "notes": "Amazon Berkeley Objects — product 3D"},
    {"type": "dataset", "name": "LVIS",             "url": "https://www.lvisdataset.org/",                           "software": "multi",    "category": "geometry",       "stream": 2, "priority": "low"},
    {"type": "dataset", "name": "3D-FUTURE",        "url": "https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future", "software": "multi", "category": "geometry", "stream": 2, "priority": "medium", "notes": "Furniture 3D models with CAD"},
    {"type": "api",     "name": "Sketchfab API",    "url": "https://sketchfab.com/developers/data-api/v3",           "software": "multi",    "category": "geometry",       "stream": 2, "priority": "high",   "notes": "CC-licensed 3D models, Python client available"},
    {"type": "api",     "name": "Polyhaven API",    "url": "https://api.polyhaven.com/",                             "software": "blender",  "category": "geometry",       "stream": 2, "priority": "high",   "notes": "CC0 HDRIs, materials, meshes"},
    {"type": "api",     "name": "Thingiverse API",  "url": "https://www.thingiverse.com/developers",                "software": "multi",    "category": "geometry",       "stream": 2, "priority": "medium"},
    {"type": "api",     "name": "GrabCAD Library",  "url": "https://grabcad.com/library",                           "software": "cad",      "category": "geometry",       "stream": 2, "priority": "medium", "notes": "7M+ engineering CAD files"},
    {"type": "dataset", "name": "Printables",       "url": "https://www.printables.com/",                           "software": "multi",    "category": "geometry",       "stream": 2, "priority": "medium", "notes": "Prusa 3D printing community"},
    {"type": "dataset", "name": "NASA 3D Resources","url": "https://nasa3d.arc.nasa.gov/",                           "software": "multi",    "category": "geometry",       "stream": 2, "priority": "medium", "notes": "Public domain spacecraft, planetary models"},
    {"type": "dataset", "name": "NIH 3D Print Exchange", "url": "https://3dprint.nih.gov/",                         "software": "multi",    "category": "geometry",       "stream": 2, "priority": "medium", "notes": "Medical and anatomical 3D models"},
    {"type": "dataset", "name": "Smithsonian 3D",   "url": "https://3d.si.edu/",                                    "software": "multi",    "category": "geometry",       "stream": 2, "priority": "medium", "notes": "Museum objects, CC0"},

    # ── Stream 3: Physics & optics knowledge ───────────────────────────────────
    {"type": "paper",   "name": "Physically Based Rendering (PBR book)", "url": "https://pbr-book.org/",            "software": "multi",    "category": "physics",        "stream": 3, "priority": "high",   "notes": "Pharr, Jakob, Humphreys — full text online"},
    {"type": "paper",   "name": "Cook-Torrance 1982",   "url": "https://dl.acm.org/doi/10.1145/357290.357293",     "software": "multi",    "category": "physics",        "stream": 3, "priority": "high"},
    {"type": "paper",   "name": "Oren-Nayar 1994",      "url": "https://dl.acm.org/doi/10.1145/192161.192213",     "software": "multi",    "category": "physics",        "stream": 3, "priority": "medium"},
    {"type": "paper",   "name": "GGX / Walter 2007",    "url": "https://www.graphics.cornell.edu/~bjw/microfacetbsdf.pdf", "software": "multi", "category": "physics",   "stream": 3, "priority": "high"},
    {"type": "paper",   "name": "Donner & Jensen SSS",  "url": "https://graphics.ucsd.edu/papers/sss-spectral/",   "software": "multi",    "category": "physics",        "stream": 3, "priority": "high",   "notes": "Measured skin scattering data"},
    {"type": "paper",   "name": "MERL BRDF Database",   "url": "https://merl.com/brdf/",                           "software": "multi",    "category": "physics",        "stream": 3, "priority": "high",   "notes": "Measured BRDFs for 100 real materials"},
    {"type": "docs",    "name": "refractiveindex.info",  "url": "https://refractiveindex.info/",                    "software": "multi",    "category": "physics",        "stream": 3, "priority": "high",   "notes": "IOR database for all real materials"},
    {"type": "docs",    "name": "ShaderToy",             "url": "https://www.shadertoy.com/",                       "software": "multi",    "category": "physics",        "stream": 3, "priority": "medium", "notes": "GLSL shader examples, procedural techniques"},
    {"type": "docs",    "name": "Feynman Lectures EM",   "url": "https://feynmanlectures.caltech.edu/II_toc.html",  "software": "multi",    "category": "physics",        "stream": 3, "priority": "medium"},
    {"type": "docs",    "name": "OpenGL Spec",           "url": "https://registry.khronos.org/OpenGL/specs/gl/",    "software": "multi",    "category": "reference",      "stream": 3, "priority": "low"},
    {"type": "docs",    "name": "Vulkan Spec",           "url": "https://registry.khronos.org/vulkan/specs/",       "software": "multi",    "category": "reference",      "stream": 3, "priority": "low"},
    {"type": "docs",    "name": "OpenFOAM Docs",         "url": "https://www.openfoam.com/documentation/",          "software": "multi",    "category": "simulation",     "stream": 3, "priority": "medium", "notes": "CFD solver — fluid simulation ground truth"},
    # Architecture & design theory
    {"type": "docs",    "name": "ArchDaily",             "url": "https://www.archdaily.com/",                       "software": "rhino",    "category": "design_theory",  "stream": 3, "priority": "medium", "notes": "Architecture visualization references"},
    {"type": "dataset", "name": "IFC Open Building Models", "url": "https://www.ifcwiki.org/index.php/Open_BIM_Models", "software": "rhino", "category": "geometry",      "stream": 3, "priority": "medium", "notes": "IFC BIM models for architectural training"},
    {"type": "dataset", "name": "CyArk",                 "url": "https://www.cyark.org/data/",                     "software": "multi",    "category": "geometry",       "stream": 3, "priority": "low",    "notes": "Photogrammetry of heritage sites"},

    # ── Stream 4: Community & multi-turn ───────────────────────────────────────
    {"type": "forum",   "name": "Blender Artists",       "url": "https://blenderartists.org/",                      "software": "blender",  "category": "community",      "stream": 4, "priority": "high",   "notes": "Q&A threads with expert replies"},
    {"type": "forum",   "name": "Polycount",             "url": "https://polycount.com/forum/",                     "software": "multi",    "category": "community",      "stream": 4, "priority": "high",   "notes": "Game art topology crits, feedback threads"},
    {"type": "forum",   "name": "CGSociety",             "url": "https://forums.cgsociety.org/",                    "software": "multi",    "category": "community",      "stream": 4, "priority": "medium"},
    {"type": "forum",   "name": "r/blender",             "url": "https://www.reddit.com/r/blender/",                "software": "blender",  "category": "community",      "stream": 4, "priority": "medium"},
    {"type": "forum",   "name": "r/3Dmodeling",          "url": "https://www.reddit.com/r/3Dmodeling/",             "software": "multi",    "category": "community",      "stream": 4, "priority": "medium"},
    {"type": "forum",   "name": "SideFX Houdini Forum",  "url": "https://www.sidefx.com/forum/",                   "software": "houdini",  "category": "community",      "stream": 4, "priority": "high"},
    {"type": "forum",   "name": "ArtStation",            "url": "https://www.artstation.com/",                      "software": "multi",    "category": "community",      "stream": 4, "priority": "low",    "notes": "Process posts with workflow breakdowns"},

    # ── Stream 5: Cross-software & integration ─────────────────────────────────
    {"type": "dataset", "name": "AMASS Motion Capture",  "url": "https://amass.is.tue.mpg.de/",                     "software": "multi",    "category": "animation",      "stream": 5, "priority": "high",   "notes": "50h+ of MoCap data for rigging training"},
    {"type": "dataset", "name": "CMU MoCap",             "url": "http://mocap.cs.cmu.edu/",                         "software": "multi",    "category": "animation",      "stream": 5, "priority": "medium"},
    {"type": "api",     "name": "Matterport API",        "url": "https://matterport.com/developers",                "software": "multi",    "category": "geometry",       "stream": 5, "priority": "medium", "notes": "3D space scans"},
    {"type": "api",     "name": "Spline API",            "url": "https://spline.design/",                           "software": "multi",    "category": "geometry",       "stream": 5, "priority": "medium"},
    {"type": "docs",    "name": "glTF Spec",             "url": "https://registry.khronos.org/glTF/specs/2.0/",     "software": "multi",    "category": "reference",      "stream": 5, "priority": "medium"},
    {"type": "docs",    "name": "USD Spec (Pixar)",      "url": "https://openusd.org/docs/",                        "software": "multi",    "category": "reference",      "stream": 5, "priority": "medium"},
    {"type": "docs",    "name": "Blender Python API",    "url": "https://docs.blender.org/api/current/",            "software": "blender",  "category": "reference",      "stream": 5, "priority": "high"},
    {"type": "docs",    "name": "Maya Python API",       "url": "https://help.autodesk.com/view/MAYAUL/2024/ENU/?guid=Maya_SDK_Maya_Python_API_html", "software": "maya", "category": "reference", "stream": 5, "priority": "high"},
    {"type": "docs",    "name": "Houdini Python API",    "url": "https://www.sidefx.com/docs/houdini/hom/hou/index.html", "software": "houdini", "category": "reference",   "stream": 5, "priority": "high"},
    {"type": "docs",    "name": "Three.js Docs",         "url": "https://threejs.org/docs/",                        "software": "web",      "category": "reference",      "stream": 5, "priority": "medium"},

    # GitHub awesome lists
    {"type": "github",  "name": "awesome-blender",       "url": "https://github.com/agmmnn/awesome-blender",        "software": "blender",  "category": "reference",      "stream": 1, "priority": "medium"},
    {"type": "github",  "name": "awesome-3d-generation", "url": "https://github.com/justimyhxu/awesome-3D-generation", "software": "multi",  "category": "reference",      "stream": 2, "priority": "medium"},
    {"type": "github",  "name": "awesome-point-cloud",   "url": "https://github.com/Yochengliu/awesome-point-cloud-analysis", "software": "multi", "category": "geometry", "stream": 2, "priority": "low"},

    # HuggingFace datasets
    {"type": "dataset", "name": "HF 3D datasets hub",    "url": "https://huggingface.co/datasets?task_categories=task_categories%3Aother&sort=likes&search=3d", "software": "multi", "category": "geometry", "stream": 2, "priority": "medium"},
    {"type": "dataset", "name": "Papers With Code 3D",   "url": "https://paperswithcode.com/area/computer-vision/3d-vision", "software": "multi", "category": "geometry",  "stream": 2, "priority": "medium"},
    {"type": "dataset", "name": "arXiv 3D ML papers",    "url": "https://arxiv.org/list/cs.GR/recent",              "software": "multi",    "category": "physics",        "stream": 3, "priority": "medium"},
]


# ─── LLM scoring ───────────────────────────────────────────────────────────────

SCORING_SYSTEM_PROMPT = """You are a data quality analyst for a 3D AI training pipeline.

Given a list of data sources, score each one for relevance to training Nalana — a voice-to-3D AI that:
- Executes real 3D software operations (Blender, Maya, Houdini, Cinema 4D, Rhino, Unreal)
- Understands expert 3D workflows, topology, physics, and materials
- Reasons about WHY operations are done, not just WHAT

For each source, output a JSON object with:
{
  "url": "<source url>",
  "relevance_score": <0.0-1.0>,
  "priority": "high|medium|low",
  "notes": "<one sentence on why it's valuable>",
  "tags": ["topology", "physics", "rigging", etc]
}

Scoring rubric:
- 0.9–1.0: Primary training signal — expert workflows, narrated tutorials, physics ground truth
- 0.7–0.9: Strong secondary signal — high-quality 3D data, community Q&A, API references
- 0.5–0.7: Useful supplementary — design theory, geometry datasets, academic papers
- 0.3–0.5: Low signal — sparse or tangential content
- 0.0–0.3: Not useful for this specific task

Output a JSON ARRAY of score objects. Order by relevance_score descending."""


EXPANSION_SYSTEM_PROMPT = """You are a dataset discovery agent for a 3D AI training pipeline.

Given a list of data sources that have already been found, generate NEW sources that haven't been listed yet.

Focus on:
1. YouTube channels teaching 3D software workflows (Blender, Maya, Houdini, C4D, Rhino, Unreal, Substance, ZBrush)
2. 3D geometry datasets (Hugging Face, academic repos, CC-licensed repositories)
3. Physics/optics references (papers, databases, measured material data)
4. Community forums where experts discuss 3D techniques
5. Software documentation and API references
6. Motion capture and animation datasets

Output a JSON ARRAY of new source objects:
{
  "type": "youtube_channel|dataset|paper|forum|docs|api|github",
  "name": "human readable name",
  "url": "canonical URL",
  "software": "blender|maya|houdini|c4d|rhino|unreal|substance|multi|physics|geometry|cad|web|animation",
  "category": "tutorial|simulation|geometry|physics|design_theory|community|reference|animation",
  "stream": <1-5>,
  "priority": "high|medium|low",
  "notes": "one sentence on value"
}

Only include sources you are confident exist and are publicly accessible. Do NOT invent URLs."""


async def score_sources(
    sources: list[dict],
    backend: str,
    vllm_url: str,
    vllm_api_key: str,
    claude_api_key: str,
    client: httpx.AsyncClient,
    batch_size: int = 20,
) -> list[dict]:
    """Ask LLM to score each source for relevance. Returns updated source list."""
    scored = []

    for i in range(0, len(sources), batch_size):
        batch = sources[i : i + batch_size]
        batch_summary = [{"url": s["url"], "name": s["name"], "category": s["category"], "notes": s.get("notes", "")} for s in batch]
        prompt = f"Score these {len(batch)} data sources for Nalana relevance:\n\n{json.dumps(batch_summary, indent=2)}"

        try:
            raw = await _call_llm(client, backend, vllm_url, vllm_api_key, claude_api_key,
                                  SCORING_SYSTEM_PROMPT, prompt)
            scores = _parse_json_array(raw)

            score_map = {s["url"]: s for s in scores if "url" in s}
            for src in batch:
                rating = score_map.get(src["url"], {})
                scored.append({
                    **src,
                    "relevance_score": rating.get("relevance_score", 0.5),
                    "priority":        rating.get("priority", src.get("priority", "medium")),
                    "notes":           rating.get("notes", src.get("notes", "")),
                    "tags":            rating.get("tags", []),
                })
        except Exception:
            # On LLM failure, keep sources with default score
            scored.extend(batch)

        await asyncio.sleep(0.5)

    return scored


async def expand_sources(
    existing: list[dict],
    backend: str,
    vllm_url: str,
    vllm_api_key: str,
    claude_api_key: str,
    client: httpx.AsyncClient,
    rounds: int = 2,
) -> list[dict]:
    """Ask LLM to generate new sources not already in the list. Returns only new sources."""
    all_sources = list(existing)
    existing_urls = {s["url"] for s in all_sources}

    for rnd in range(rounds):
        print(f"  Expansion round {rnd + 1}/{rounds} ({len(all_sources)} sources so far)...")
        # Give LLM a summary of what we already have by category
        summary = {}
        for s in all_sources:
            cat = s.get("category", "other")
            summary[cat] = summary.get(cat, 0) + 1

        prompt = (
            f"We have {len(all_sources)} sources. Category counts: {json.dumps(summary)}.\n"
            f"Existing source URLs (do NOT repeat these):\n"
            + "\n".join(f"  - {s['url']}" for s in all_sources[:60])
            + "\n\nGenerate 15-25 NEW data sources not in this list."
        )

        try:
            raw = await _call_llm(client, backend, vllm_url, vllm_api_key, claude_api_key,
                                  EXPANSION_SYSTEM_PROMPT, prompt)
            new_sources = _parse_json_array(raw)
            added = 0
            for src in new_sources:
                if src.get("url") and src["url"] not in existing_urls:
                    all_sources.append(src)
                    existing_urls.add(src["url"])
                    added += 1
            print(f"    +{added} new sources")
        except Exception as e:
            print(f"    Expansion round {rnd + 1} failed: {e}")

        await asyncio.sleep(1.0)

    return [s for s in all_sources if s["url"] not in {e["url"] for e in existing}]


# ─── LLM backends ──────────────────────────────────────────────────────────────

async def _call_llm(
    client: httpx.AsyncClient,
    backend: str,
    vllm_url: str,
    vllm_api_key: str,
    claude_api_key: str,
    system: str,
    prompt: str,
) -> str:
    if backend == "vllm":
        resp = await client.post(
            f"{vllm_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {vllm_api_key}"},
            json={
                "model": "Qwen/Qwen2.5-72B-Instruct",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                "max_tokens": 4096,
                "temperature": 0.1,
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    else:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": claude_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 4096,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]


def _parse_json_array(raw: str) -> list[dict]:
    raw = raw.strip()
    # Strip markdown code fences
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    try:
        result = json.loads(raw)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        # Try to extract first [...] block
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
    return []


# ─── Category filters ──────────────────────────────────────────────────────────

CATEGORIES = {
    "youtube":   lambda s: s["type"] == "youtube_channel",
    "datasets":  lambda s: s["type"] in ("dataset", "api") and s["stream"] == 2,
    "physics":   lambda s: s["stream"] == 3,
    "community": lambda s: s["type"] == "forum",
    "docs":      lambda s: s["type"] in ("docs", "github"),
    "all":       lambda s: True,
}


# ─── Main ──────────────────────────────────────────────────────────────────────

async def _run(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter seeds by requested category
    cat_filter = CATEGORIES.get(args.category, CATEGORIES["all"])
    sources: list[dict] = [s for s in SEED_SOURCES if cat_filter(s)]

    print(f"\nNalana Data Discovery")
    print(f"  Category:  {args.category}")
    print(f"  Backend:   {args.backend}")
    print(f"  Seeds:     {len(sources)}")
    print(f"  Expand:    {'yes' if not args.no_expand else 'no'}")
    print(f"  Output:    {output_path}\n")

    async with httpx.AsyncClient() as client:

        if not args.no_expand:
            # Step 1: LLM-score the seeds
            print("Scoring seeds...")
            t0 = time.time()
            sources = await score_sources(
                sources, args.backend, args.vllm_url,
                args.vllm_api_key, args.claude_api_key, client,
            )
            print(f"  Done ({time.time()-t0:.0f}s). Sample scores: "
                  + ", ".join(f"{s['name']}={s.get('relevance_score',0):.2f}" for s in sources[:4]))

            # Step 2: LLM-expand to find new sources
            print("\nExpanding sources with LLM...")
            new_sources = await expand_sources(
                sources, args.backend, args.vllm_url,
                args.vllm_api_key, args.claude_api_key, client,
                rounds=args.expand_rounds,
            )

            # Step 3: Score the new sources too
            if new_sources:
                print(f"\nScoring {len(new_sources)} new sources...")
                new_sources = await score_sources(
                    new_sources, args.backend, args.vllm_url,
                    args.vllm_api_key, args.claude_api_key, client,
                )
                sources.extend(new_sources)
        else:
            # No-expand: assign default scores from priority
            priority_to_score = {"high": 0.85, "medium": 0.60, "low": 0.35}
            for s in sources:
                s.setdefault("relevance_score", priority_to_score.get(s.get("priority", "medium"), 0.5))
                s.setdefault("tags", [])

    # Sort by score desc, then stream asc
    sources.sort(key=lambda s: (-s.get("relevance_score", 0), s.get("stream", 5)))

    # Deduplicate by URL
    seen_urls: set[str] = set()
    deduped: list[dict] = []
    for s in sources:
        url = s.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            deduped.append(s)

    output_path.write_text(json.dumps(deduped, indent=2))

    # Summary stats
    by_stream: dict[int, int] = {}
    by_type:   dict[str, int] = {}
    high_count = sum(1 for s in deduped if s.get("relevance_score", 0) >= 0.7)
    for s in deduped:
        by_stream[s.get("stream", 0)] = by_stream.get(s.get("stream", 0), 0) + 1
        by_type[s.get("type", "?")] = by_type.get(s.get("type", "?"), 0) + 1

    print(f"\n{'─'*50}")
    print(f"Total sources:    {len(deduped)}")
    print(f"High relevance:   {high_count} (score >= 0.7)")
    print(f"By stream:        " + "  ".join(f"S{k}:{v}" for k, v in sorted(by_stream.items())))
    print(f"By type:          " + "  ".join(f"{k}:{v}" for k, v in sorted(by_type.items())))
    print(f"Output:           {output_path}")
    print(f"\nNext step: bash scripts/run_all.sh --from-step 3")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous dataset source hunter for the Nalana training pipeline"
    )
    parser.add_argument("--all", dest="category", action="store_const", const="all",
                        help="Discover all source categories (equivalent to --category all)")
    parser.add_argument("--category", default="all",
                        choices=list(CATEGORIES.keys()),
                        help="Source category to discover (default: all)")
    parser.add_argument("--output", default="data/discovered_sources.json",
                        help="Output JSON file path")
    parser.add_argument("--backend", choices=["vllm", "claude"], default="claude",
                        help="LLM backend for scoring and expansion")
    parser.add_argument("--vllm-url", default=os.environ.get("VLLM_URL", "http://localhost:8001"))
    parser.add_argument("--vllm-api-key", default=os.environ.get("VLLM_API_KEY", "nalana"))
    parser.add_argument("--claude-api-key", default=os.environ.get("ANTHROPIC_API_KEY"))
    parser.add_argument("--no-expand", action="store_true",
                        help="Skip LLM expansion — just score and output seeds")
    parser.add_argument("--expand-rounds", type=int, default=2,
                        help="Number of LLM expansion rounds (default: 2)")
    args = parser.parse_args()

    if not args.no_expand:
        if args.backend == "claude" and not args.claude_api_key:
            parser.error("Set ANTHROPIC_API_KEY or pass --claude-api-key (or use --no-expand)")
        if args.backend == "vllm":
            print(f"Using vLLM at {args.vllm_url}")

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
