"""
asset_manager.py — Nalana Asset Management Domain Agent

Generates training data for 3D asset library management tasks:
- Semantic search across asset collections
- Auto-tagging and categorization
- Duplicate detection
- Asset validation and cleanup
- Library organization recommendations
- Batch operations on asset collections
"""

from __future__ import annotations

import argparse
import json
import re
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ─── Output paths ──────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parents[1]
ASSET_DATA_DIR = BASE_DIR / "data" / "asset_management"
ASSET_DATA_DIR.mkdir(parents=True, exist_ok=True)
PAIRS_OUTPUT = ASSET_DATA_DIR / "asset_management_pairs.jsonl"

# ─── Asset Taxonomy ────────────────────────────────────────────────────────────

ASSET_TAXONOMY: dict[str, dict[str, Any]] = {
    "category": {
        "architecture": {
            "description": "Buildings, rooms, structural elements, walls, floors, doors, windows",
            "subcategories": [
                "exterior",
                "interior",
                "modular",
                "hero",
                "background",
                "kit_piece",
            ],
            "typical_poly_range": (500, 500_000),
            "naming_prefix": "SM_Arch_",
        },
        "character": {
            "description": "Humans, creatures, monsters, NPCs, hero characters",
            "subcategories": ["hero", "npc", "crowd", "creature", "robot", "humanoid"],
            "typical_poly_range": (3_000, 80_000),
            "naming_prefix": "SK_Char_",
        },
        "vehicle": {
            "description": "Cars, aircraft, boats, bikes, spacecraft, military vehicles",
            "subcategories": ["land", "air", "sea", "space", "military", "civilian"],
            "typical_poly_range": (5_000, 150_000),
            "naming_prefix": "SM_Veh_",
        },
        "prop": {
            "description": "Objects, furniture, tools, weapons, clothing, everyday items",
            "subcategories": [
                "hero",
                "background",
                "weapon",
                "furniture",
                "electronics",
                "food",
            ],
            "typical_poly_range": (100, 30_000),
            "naming_prefix": "SM_Prop_",
        },
        "environment": {
            "description": "Terrain, foliage, rocks, cliffs, water bodies, nature elements",
            "subcategories": [
                "terrain",
                "foliage",
                "rock",
                "water",
                "sky",
                "modular_tile",
            ],
            "typical_poly_range": (50, 200_000),
            "naming_prefix": "SM_Env_",
        },
        "fx": {
            "description": "Particle emitters, VFX meshes, fluid caches, shader-driven effects",
            "subcategories": ["fire", "smoke", "water", "sparks", "magic", "explosion"],
            "typical_poly_range": (10, 50_000),
            "naming_prefix": "FX_",
        },
        "material": {
            "description": "Shader node groups, PBR material definitions",
            "subcategories": [
                "metal",
                "organic",
                "fabric",
                "glass",
                "plastic",
                "stone",
                "wood",
                "liquid",
            ],
            "typical_poly_range": (0, 0),
            "naming_prefix": "M_",
        },
        "texture": {
            "description": "Image textures including maps: diffuse, normal, roughness, etc.",
            "subcategories": [
                "diffuse",
                "normal",
                "roughness",
                "metallic",
                "ao",
                "emissive",
                "height",
                "opacity",
            ],
            "typical_poly_range": (0, 0),
            "naming_prefix": "T_",
        },
        "hdri": {
            "description": "High dynamic range equirectangular images for lighting and IBL",
            "subcategories": [
                "outdoor",
                "indoor",
                "studio",
                "sky",
                "night",
                "overcast",
            ],
            "typical_poly_range": (0, 0),
            "naming_prefix": "HDRI_",
        },
        "rig": {
            "description": "Armatures and control rigs, separate from character meshes",
            "subcategories": [
                "biped",
                "quadruped",
                "face",
                "hand",
                "vehicle",
                "mechanical",
            ],
            "typical_poly_range": (0, 0),
            "naming_prefix": "RIG_",
        },
    },
    "style": {
        "photorealistic": {
            "description": "Physically accurate materials, real-world scale, 4K+ textures",
            "texture_resolution": "2K-16K",
            "polygon_density": "high",
        },
        "stylized": {
            "description": "Artistic interpretation of reality, exaggerated forms, painterly",
            "texture_resolution": "512-2K",
            "polygon_density": "medium",
        },
        "cartoon": {
            "description": "Flat colors, strong outlines, exaggerated proportions, minimal textures",
            "texture_resolution": "256-1K",
            "polygon_density": "low-medium",
        },
        "lowpoly": {
            "description": "Deliberately low triangle count, visible faceting, often stylized",
            "texture_resolution": "64-512",
            "polygon_density": "very_low",
        },
        "highpoly": {
            "description": "Sculpted detail, ZBrush or Blender Sculpt origin, baked to low for games",
            "texture_resolution": "4K-16K",
            "polygon_density": "very_high",
        },
        "scifi": {
            "description": "Futuristic, technological, often hard-surface, greebles, neon",
            "texture_resolution": "1K-4K",
            "polygon_density": "medium-high",
        },
        "fantasy": {
            "description": "Medieval, magical, organic forms, ornate detail, worn surfaces",
            "texture_resolution": "1K-4K",
            "polygon_density": "medium-high",
        },
        "historical": {
            "description": "Period-accurate props and architecture from specific eras",
            "texture_resolution": "1K-4K",
            "polygon_density": "medium",
        },
        "modern": {
            "description": "Contemporary real-world aesthetics, clean lines, realistic wear",
            "texture_resolution": "1K-4K",
            "polygon_density": "medium",
        },
        "abstract": {
            "description": "Non-representational geometry, procedural forms, experimental",
            "texture_resolution": "any",
            "polygon_density": "any",
        },
    },
    "state": {
        "finished": {
            "description": "Production-ready, fully textured, LODs complete, named correctly",
            "ready_for": ["render", "game", "delivery"],
        },
        "WIP": {
            "description": "Work-in-progress, placeholder materials may be present, naming unstable",
            "ready_for": ["internal_review"],
        },
        "placeholder": {
            "description": "Temporary block-out or proxy mesh, will be replaced",
            "ready_for": ["layout", "blockout"],
        },
        "optimized": {
            "description": "Decimated/retopologized for real-time use, LOD chain present",
            "ready_for": ["game", "real_time", "xr"],
        },
        "hero": {
            "description": "Highest-quality version, will be seen close-up, maximum detail",
            "ready_for": ["render", "film", "archviz"],
        },
        "background": {
            "description": "Simplified version for distant placement, fewer polygons and texture detail",
            "ready_for": ["game", "render"],
        },
    },
    "software_compatibility": {
        "blender": {
            "native_format": ".blend",
            "import_formats": [".fbx", ".obj", ".gltf", ".glb", ".abc", ".usd"],
            "version_note": "Blender 3.x+ recommended; geometry nodes require 3.0+",
        },
        "maya": {
            "native_format": ".ma / .mb",
            "import_formats": [".fbx", ".obj", ".abc", ".usd"],
            "version_note": "Maya 2022+ for full USD support; Arnold renderer integration",
        },
        "c4d": {
            "native_format": ".c4d",
            "import_formats": [".fbx", ".obj", ".abc", ".usd"],
            "version_note": "Cinema 4D 2023+ for full Redshift/Arnold support",
        },
        "unreal": {
            "native_format": ".uasset",
            "import_formats": [".fbx", ".obj", ".abc", ".usd"],
            "version_note": "UE5 supports Nanite for high-poly meshes; Lumen for dynamic lighting",
        },
        "unity": {
            "native_format": ".prefab",
            "import_formats": [".fbx", ".obj", ".gltf", ".glb"],
            "version_note": "Unity 2021+ for URP/HDRP; glTF support via UnityGLTF package",
        },
        "houdini": {
            "native_format": ".hip / .hda",
            "import_formats": [".abc", ".fbx", ".obj", ".usd", ".bgeo"],
            "version_note": "Solaris USD pipeline recommended; SideFX Labs for asset tools",
        },
        "substance": {
            "native_format": ".spp / .sbsar",
            "import_formats": [".fbx", ".obj", ".gltf"],
            "version_note": "Substance Painter for texturing; Substance Designer for procedural materials",
        },
        "all": {
            "description": "Asset uses only universal formats (.obj, .fbx) with no proprietary nodes",
            "recommended_format": ".gltf or .fbx",
        },
    },
    "topology_quality": {
        "excellent": {
            "description": "All quads, edge loops follow muscle/form flow, subdivision-ready",
            "polygon_count_examples": {
                "game_character": "8K-20K tris",
                "film_character": "50K-200K quads",
                "vehicle_game": "15K-40K tris",
            },
            "characteristics": [
                "No n-gons",
                "No triangles except at poles",
                "Even quad distribution",
                "Clean edge loops",
                "Subdivision-friendly",
            ],
        },
        "good": {
            "description": "Mostly quads, minor triangles at hard edges, functionally clean",
            "characteristics": [
                "Triangles present but not disruptive",
                "No n-gons larger than 5 sides",
                "Renders and deforms acceptably",
                "Not fully subdivision-ready",
            ],
        },
        "fair": {
            "description": "Mix of quads, triangles, and occasional n-gons; functional but not clean",
            "characteristics": [
                "N-gons present (may cause shading artifacts)",
                "Uneven polygon distribution",
                "Deformation issues in animated areas",
                "Usable for static renders",
            ],
        },
        "poor": {
            "description": "Heavily triangulated, non-manifold edges, n-gons, overlapping faces",
            "characteristics": [
                "Likely auto-retopologized or scan data",
                "Non-manifold geometry",
                "Self-intersections possible",
                "Not suitable for rigging or subdivision",
            ],
        },
    },
    "uv_status": {
        "unwrapped": {
            "description": "Clean UV islands, no overlapping, optimal texel density",
            "suitable_for": [
                "hand_painting",
                "baking",
                "pbr_texturing",
                "udim_expansion",
            ],
        },
        "overlapping": {
            "description": "UV islands overlap — fine for tiling textures, bad for baking",
            "suitable_for": ["tiling_textures_only"],
            "issues": ["Cannot bake AO, normals, or shadows correctly"],
        },
        "no_uvs": {
            "description": "No UV map assigned; procedural or un-textured asset",
            "suitable_for": ["procedural_materials", "preview_only"],
        },
        "udim": {
            "description": "UDIM layout (multiple UV tiles), each tile maps to a separate texture",
            "suitable_for": ["film", "vfx", "archviz", "high_detail_characters"],
            "tile_naming": "1001, 1002, 1011 etc. following Mari/Mudbox convention",
        },
        "tileable": {
            "description": "UV islands extend beyond 0-1 space to tile seamlessly",
            "suitable_for": ["large_surfaces", "terrain", "architectural_surfaces"],
        },
    },
    "rig_status": {
        "none": {"description": "No armature, static mesh only"},
        "basic": {
            "description": "Simple bone chain, FK-only, no IK or controls",
            "bone_count_range": (1, 30),
        },
        "advanced": {
            "description": "IK/FK switching, stretch bones, custom shape controllers",
            "bone_count_range": (30, 100),
        },
        "facial": {
            "description": "Facial rig with shape keys (blend shapes) for expressions and phonemes",
            "bone_count_range": (20, 80),
            "shape_key_count_typical": (30, 120),
        },
        "full_body": {
            "description": "Complete character rig: body IK, facial shapes, finger controls, cloth simulation bones",
            "bone_count_range": (80, 300),
            "features": [
                "IK/FK body",
                "facial shapes",
                "corrective shapes",
                "squash/stretch bones",
            ],
        },
    },
}

# ─── Duplicate Detection Strategies ───────────────────────────────────────────

DUPLICATE_DETECTION_STRATEGIES: list[dict[str, Any]] = [
    {
        "name": "exact_filename_match",
        "description": "Compare base filenames (without extension) for identical names across the library.",
        "reliability": "low",
        "false_positive_risk": "high",
        "notes": (
            "Fast to compute but unreliable — different assets may share names. "
            "Use as a first pass to flag candidates, not as a final decision."
        ),
        "implementation": "Set intersection of {Path(f).stem for f in file_list}",
        "blender_example": """
import bpy

def find_duplicate_names():
    seen = {}
    duplicates = []
    for obj in bpy.data.objects:
        base = obj.name.rstrip('.0123456789')
        if base in seen:
            duplicates.append((seen[base], obj))
        else:
            seen[base] = obj
    return duplicates
""",
    },
    {
        "name": "perceptual_hash_thumbnail",
        "description": (
            "Render a small thumbnail (64x64) of each asset and compare using perceptual hash (pHash). "
            "Assets with Hamming distance < 8 are candidates."
        ),
        "reliability": "medium",
        "false_positive_risk": "medium",
        "notes": (
            "Works well for visually identical assets even if renamed or reformatted. "
            "Sensitive to camera angle — always render from the same canonical angle (front/iso)."
        ),
        "implementation": "imagehash.phash() from the imagehash library; Hamming distance threshold <= 8",
        "blender_example": """
import bpy

def render_asset_thumbnail(obj_name, output_path, size=64):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects[obj_name].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects[obj_name]
    bpy.context.scene.render.resolution_x = size
    bpy.context.scene.render.resolution_y = size
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
""",
    },
    {
        "name": "polygon_count_bbox_similarity",
        "description": (
            "Compare vertex count + polygon count + bounding box dimensions. "
            "Assets within 2% vertex count AND 5% bounding box volume are flagged."
        ),
        "reliability": "medium-high",
        "false_positive_risk": "low-medium",
        "notes": (
            "Very effective for detecting re-exported versions of the same mesh (e.g., .obj vs .fbx). "
            "May miss copies that have been scaled or slightly modified."
        ),
        "implementation": "abs(v1 - v2) / max(v1, v2) < 0.02 for vertices; bbox volume ratio within 5%",
        "blender_example": """
import bpy

def get_mesh_signature(obj):
    if obj.type != 'MESH':
        return None
    mesh = obj.data
    verts = len(mesh.vertices)
    faces = len(mesh.polygons)
    dims = obj.dimensions
    volume = dims.x * dims.y * dims.z
    return {'verts': verts, 'faces': faces, 'volume': volume}

def find_by_signature(objects, tol_verts=0.02, tol_vol=0.05):
    sigs = [(o, get_mesh_signature(o)) for o in objects if get_mesh_signature(o)]
    groups = []
    used = set()
    for i, (oa, sa) in enumerate(sigs):
        if i in used:
            continue
        group = [oa]
        for j, (ob, sb) in enumerate(sigs[i+1:], i+1):
            if j in used:
                continue
            dv = abs(sa['verts'] - sb['verts']) / max(sa['verts'], sb['verts'], 1)
            dvol = abs(sa['volume'] - sb['volume']) / max(sa['volume'], sb['volume'], 1e-6)
            if dv < tol_verts and dvol < tol_vol:
                group.append(ob)
                used.add(j)
        if len(group) > 1:
            used.add(i)
            groups.append(group)
    return groups
""",
    },
    {
        "name": "material_name_overlap",
        "description": "Assets that share >=50% of their material slot names are likely duplicates or variants.",
        "reliability": "low-medium",
        "false_positive_risk": "medium",
        "notes": (
            "Useful for finding re-imported assets that kept the original material names. "
            "Less useful when studios use generic material names (Material.001, etc.)."
        ),
        "implementation": "Jaccard similarity: |A & B| / |A | B| >= 0.5",
        "blender_example": """
import bpy

def material_jaccard(obj_a, obj_b):
    mats_a = {slot.material.name for slot in obj_a.material_slots if slot.material}
    mats_b = {slot.material.name for slot in obj_b.material_slots if slot.material}
    if not mats_a and not mats_b:
        return 0.0
    intersection = len(mats_a & mats_b)
    union = len(mats_a | mats_b)
    return intersection / union if union > 0 else 0.0
""",
    },
    {
        "name": "vertex_count_exact_match",
        "description": "Exact vertex count match. Fastest computation, high precision but narrow recall.",
        "reliability": "high (for exact copies)",
        "false_positive_risk": "low",
        "notes": (
            "Exact vertex count match almost certainly indicates a copy. "
            "Use as a fast first-pass before more expensive perceptual or bbox checks. "
            "May miss assets that have been slightly modified (vertex welded, subdivided, etc.)."
        ),
        "implementation": "Group all objects by len(obj.data.vertices); any group with size > 1 is a candidate",
        "blender_example": """
import bpy
from collections import defaultdict

def find_exact_vertex_duplicates():
    buckets = defaultdict(list)
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            vcount = len(obj.data.vertices)
            buckets[vcount].append(obj)
    return {k: v for k, v in buckets.items() if len(v) > 1}
""",
    },
]

# ─── File Format Support ───────────────────────────────────────────────────────

FILE_FORMAT_SUPPORT: dict[str, dict[str, Any]] = {
    ".blend": {
        "read": True,
        "write": True,
        "notes": (
            "Blender native format. Stores everything: meshes, materials, node trees, "
            "particles, linked libraries, render settings. Single-file project OR library. "
            "Not compatible with other software without export."
        ),
        "fidelity": "perfect",
        "animation_support": True,
        "material_support": "full_procedural",
    },
    ".fbx": {
        "read": True,
        "write": True,
        "notes": (
            "Most common game engine interchange format. Supports meshes, armatures, "
            "animations, blend shapes (shape keys), and basic Lambert/Phong materials. "
            "Does NOT preserve procedural node trees. Binary FBX preferred over ASCII. "
            "Autodesk proprietary spec — some quirks in Blender exporter."
        ),
        "fidelity": "medium-high",
        "animation_support": True,
        "material_support": "basic_phong_lambert",
    },
    ".obj": {
        "read": True,
        "write": True,
        "notes": (
            "Oldest universal mesh format. Supports geometry, UVs, normals, and basic "
            "MTL material references. No animation, no armatures. Widely supported. "
            "ASCII-only, slow for large meshes. OBJ+MTL is a two-file pair."
        ),
        "fidelity": "medium",
        "animation_support": False,
        "material_support": "mtl_file",
    },
    ".gltf": {
        "read": True,
        "write": True,
        "notes": (
            "glTF 2.0 (JSON + separate .bin + textures). 'JPEG of 3D' — designed for "
            "web and real-time. Supports PBR metallic-roughness workflow, animation, "
            "morph targets, skinning. Widely supported in game engines and web (Three.js, Babylon.js)."
        ),
        "fidelity": "medium-high",
        "animation_support": True,
        "material_support": "pbr_metallic_roughness",
    },
    ".glb": {
        "read": True,
        "write": True,
        "notes": (
            "Binary glTF — same as .gltf but single self-contained file. "
            "Ideal for distribution and runtime loading. Same feature set as .gltf."
        ),
        "fidelity": "medium-high",
        "animation_support": True,
        "material_support": "pbr_metallic_roughness",
    },
    ".abc": {
        "read": True,
        "write": True,
        "notes": (
            "Alembic — designed for animated geometry caches. Stores per-frame "
            "vertex positions, UV, normals. Used for cloth sim, fluid sim, crowd caches. "
            "No rigging/armatures — just baked geometry. Industry standard for VFX pipelines."
        ),
        "fidelity": "high",
        "animation_support": True,
        "material_support": "none_baked_geo_only",
    },
    ".usd": {
        "read": True,
        "write": True,
        "notes": (
            "Universal Scene Description (Pixar/Apple). Emerging VFX and game standard. "
            "Supports layering, variants, references, and full scene hierarchy. "
            "Text format. Pair with .usdc for binary efficiency."
        ),
        "fidelity": "very_high",
        "animation_support": True,
        "material_support": "usd_preview_surface_and_mdl",
    },
    ".usda": {
        "read": True,
        "write": True,
        "notes": (
            "USD ASCII variant — human readable, diffable. Slower to load than binary .usdc. "
            "Best for debugging USD scenes and pipeline development."
        ),
        "fidelity": "very_high",
        "animation_support": True,
        "material_support": "usd_preview_surface",
    },
    ".usdc": {
        "read": True,
        "write": True,
        "notes": (
            "USD Crate (binary) — compact and fast to load. Preferred format for "
            "production USD pipelines. Same data model as .usda, just compressed binary."
        ),
        "fidelity": "very_high",
        "animation_support": True,
        "material_support": "usd_preview_surface",
    },
    ".usdz": {
        "read": True,
        "write": True,
        "notes": (
            "USD ZIP — single-file USD bundle (scene + textures). Apple AR format. "
            "Used for iOS/macOS AR Quick Look."
        ),
        "fidelity": "high",
        "animation_support": True,
        "material_support": "usd_preview_surface",
    },
    ".dae": {
        "read": True,
        "write": False,
        "notes": (
            "COLLADA — XML-based interchange format, largely superseded by glTF. "
            "Still used in some older pipelines and game engines (SketchUp, Godot). "
            "Supports animation, materials. Import-only recommended in modern pipelines."
        ),
        "fidelity": "medium",
        "animation_support": True,
        "material_support": "phong_lambert",
    },
    ".ply": {
        "read": True,
        "write": False,
        "notes": (
            "Polygon File Format — common for point clouds and scan data. "
            "Supports per-vertex color, normals, arbitrary properties. "
            "No animation, no materials. ASCII and binary variants."
        ),
        "fidelity": "medium",
        "animation_support": False,
        "material_support": "vertex_color_only",
    },
    ".stl": {
        "read": True,
        "write": True,
        "notes": (
            "Stereolithography — triangles only, no UVs, no materials, no scale units. "
            "Standard for 3D printing. Binary STL preferred (smaller than ASCII). "
            "Always watertight (manifold) for successful printing."
        ),
        "fidelity": "low",
        "animation_support": False,
        "material_support": "none",
    },
    ".3ds": {
        "read": True,
        "write": False,
        "notes": (
            "3ds Max legacy format (pre-2010). Supports mesh, materials, cameras, lights. "
            "Hard limits: 65535 vertices per mesh, 8-char material names. "
            "Import only — export to FBX instead."
        ),
        "fidelity": "low-medium",
        "animation_support": False,
        "material_support": "basic",
    },
    ".dxf": {
        "read": True,
        "write": False,
        "notes": (
            "AutoCAD Drawing Exchange Format. Used in CAD and architectural workflows. "
            "Supports 2D and 3D geometry. Common for floor plans, technical drawings. "
            "Poor mesh support — use for CAD import, not general 3D assets."
        ),
        "fidelity": "low",
        "animation_support": False,
        "material_support": "none",
    },
}

# ─── Asset Naming Conventions ──────────────────────────────────────────────────

ASSET_NAMING_CONVENTIONS: dict[str, Any] = {
    "prefix_by_type": {
        "SM_": {
            "meaning": "Static Mesh",
            "software_origin": "Unreal Engine",
            "description": "Any non-animated, non-skeletal mesh asset. Buildings, props, environment pieces.",
            "examples": ["SM_Rock_01", "SM_Chair_Wood_A", "SM_Pillar_Stone"],
        },
        "SK_": {
            "meaning": "Skeletal Mesh",
            "software_origin": "Unreal Engine",
            "description": "Animated mesh bound to an armature/skeleton. Characters, cloth simulations.",
            "examples": ["SK_Character_Hero", "SK_Enemy_Grunt", "SK_Creature_Dragon"],
        },
        "M_": {
            "meaning": "Material (master)",
            "software_origin": "Unreal Engine",
            "description": "Master material node graph, parameterized. Material instances derive from this.",
            "examples": ["M_Metal_Base", "M_Organic_Skin", "M_Glass_Clear"],
        },
        "MI_": {
            "meaning": "Material Instance",
            "software_origin": "Unreal Engine",
            "description": "Parameter override of a master material. Faster to create, inherits master's logic.",
            "examples": ["MI_Metal_Rusty", "MI_Skin_Human_01", "MI_Glass_Tinted_Red"],
        },
        "T_": {
            "meaning": "Texture",
            "software_origin": "Unreal Engine / Industry",
            "description": "Individual image texture. Combined with suffix to indicate map type.",
            "examples": ["T_Brick_Wall_D", "T_Skin_Face_N", "T_Metal_Panel_ORM"],
        },
        "BP_": {
            "meaning": "Blueprint / Prefab",
            "software_origin": "Unreal Engine",
            "description": "Composite asset grouping mesh + logic + child objects. Interactive prop in Unreal.",
            "examples": [
                "BP_Door_Sliding",
                "BP_Light_Ceiling_Dimmable",
                "BP_Crate_Destructible",
            ],
        },
        "FX_": {
            "meaning": "Visual Effect / Particle System",
            "software_origin": "Universal",
            "description": "Particle emitters, Niagara systems, or VFX mesh assets.",
            "examples": ["FX_Fire_Small", "FX_Sparks_Welding", "FX_Explosion_Medium"],
        },
        "S_": {
            "meaning": "Sound / Audio",
            "software_origin": "Unreal Engine",
            "description": "Sound wave or sound cue asset.",
            "examples": ["S_Footstep_Concrete", "S_Door_Creak", "S_Ambient_Forest"],
        },
        "CH_": {
            "meaning": "Character (Blender convention)",
            "software_origin": "Blender community",
            "description": "Character mesh and rig combined in Blender. SK_ is preferred for Unreal.",
            "examples": ["CH_Hero_Knight", "CH_NPC_Merchant", "CH_Creature_Goblin"],
        },
    },
    "texture_suffixes": {
        "_D": {
            "meaning": "Diffuse / Base Color / Albedo",
            "channel": "RGB",
            "color_space": "sRGB",
            "notes": "The base color of the surface, no lighting information baked in.",
        },
        "_N": {
            "meaning": "Normal Map",
            "channel": "RGB (XYZ encoded)",
            "color_space": "Linear (non-color)",
            "notes": (
                "OpenGL convention (Y-up, green channel up). DirectX convention (Y-down, green flipped). "
                "Blender uses OpenGL; Unreal and Unity use DirectX — flip G channel when switching."
            ),
        },
        "_R": {
            "meaning": "Roughness",
            "channel": "Grayscale (R channel)",
            "color_space": "Linear",
            "notes": "0=mirror, 1=fully rough. Inverse of glossiness maps from older workflows.",
        },
        "_M": {
            "meaning": "Metallic",
            "channel": "Grayscale (B or R channel)",
            "color_space": "Linear",
            "notes": (
                "Binary mask: 0=dielectric, 1=metal. Metals reflect colored specular; "
                "dielectrics reflect white specular."
            ),
        },
        "_AO": {
            "meaning": "Ambient Occlusion",
            "channel": "Grayscale",
            "color_space": "Linear",
            "notes": "Pre-baked shadow in crevices. Often packed into ORM as red channel.",
        },
        "_E": {
            "meaning": "Emissive",
            "channel": "RGB",
            "color_space": "Linear (HDR capable)",
            "notes": "Self-illuminated surfaces. Often HDR values > 1.0 for bloom/glow effects.",
        },
        "_A": {
            "meaning": "Alpha / Opacity",
            "channel": "Grayscale",
            "color_space": "Linear",
            "notes": "0=transparent, 1=opaque. Often packed into diffuse alpha channel (RGBA texture).",
        },
        "_ORM": {
            "meaning": "Packed: Occlusion (R), Roughness (G), Metallic (B)",
            "channel": "RGB each carrying a grayscale map",
            "color_space": "Linear",
            "notes": (
                "Industry-standard packing for game engines. Saves 2 texture samples vs separate maps. "
                "Unreal Engine convention. Some pipelines swap R/G channels — verify before use."
            ),
        },
        "_H": {
            "meaning": "Height / Displacement",
            "channel": "Grayscale",
            "color_space": "Linear",
            "notes": "Used for parallax occlusion mapping in games or actual displacement in renders.",
        },
    },
    "level_of_detail": {
        "_LOD0": {
            "description": "Highest detail — seen at close distance (< 5m)",
            "polygon_budget_multiplier": 1.0,
            "use_case": "Hero view, close-up interaction",
        },
        "_LOD1": {
            "description": "Mid-detail — seen at medium distance (5-20m)",
            "polygon_budget_multiplier": 0.5,
            "use_case": "Standard gameplay distance",
        },
        "_LOD2": {
            "description": "Reduced detail — seen at far distance (20-50m)",
            "polygon_budget_multiplier": 0.25,
            "use_case": "Background, distant gameplay",
        },
        "_LOD3": {
            "description": "Heavily decimated — seen at very far distance (50-100m)",
            "polygon_budget_multiplier": 0.1,
            "use_case": "Distant background fill",
        },
        "_LOD4": {
            "description": "Imposter / billboard replacement (>100m)",
            "polygon_budget_multiplier": 0.02,
            "use_case": "Horizon fill, forests, distant buildings — usually a quad with texture",
        },
    },
    "general_rules": [
        "No spaces in names — use underscores",
        "No special characters except underscores and hyphens",
        "Capitalize each word: SM_Wooden_Barrel not SM_wooden_barrel",
        "Suffix numbers with leading zeros for sequencing: _01, _02 not _1, _2",
        "Avoid version suffixes in final assets: never deliver Asset_v07_FINAL_FINAL",
        "Keep names unique within a project to avoid collision on import",
        "Material names should reflect content: M_Metal_Steel not Material.003",
    ],
}

# ─── Asset Management Training Pairs ──────────────────────────────────────────

ASSET_MANAGEMENT_TRAINING_PAIRS: list[dict[str, str]] = [
    {
        "input": "Find all my metal materials",
        "output": """import bpy

def find_metal_materials():
    metal_keywords = ['metal', 'steel', 'iron', 'copper', 'brass', 'bronze', 'aluminum',
                      'chrome', 'gold', 'silver', 'titanium', 'pewter', 'zinc']
    results = []
    for mat in bpy.data.materials:
        name_lower = mat.name.lower()
        if any(kw in name_lower for kw in metal_keywords):
            results.append(mat)
            continue
        if mat.get('category') == 'metal' or mat.get('material_type') == 'metal':
            results.append(mat)
    print(f"Found {len(results)} metal material(s):")
    for m in results:
        print(f"  {m.name}")
    return results

find_metal_materials()""",
    },
    {
        "input": "Rename all my textures to follow Unreal Engine naming conventions",
        "output": """import bpy
import re

SUFFIX_MAP = {
    'diffuse': '_D', 'diff': '_D', 'albedo': '_D', 'color': '_D', 'col': '_D',
    'normal': '_N', 'nrm': '_N', 'nor': '_N', 'bump': '_N',
    'roughness': '_R', 'rough': '_R', 'rgh': '_R', 'gloss': '_R',
    'metallic': '_M', 'metal': '_M', 'met': '_M',
    'ambient_occlusion': '_AO', 'ao': '_AO', 'occlusion': '_AO',
    'emissive': '_E', 'emit': '_E', 'emission': '_E',
    'alpha': '_A', 'opacity': '_A', 'mask': '_A',
    'height': '_H', 'displacement': '_H', 'disp': '_H',
    'orm': '_ORM',
}

def detect_suffix(name):
    lower = name.lower()
    for key, suffix in SUFFIX_MAP.items():
        if key in lower:
            return suffix
    return ''

def ue_rename_texture(img):
    stem = re.sub(r'\\.(png|jpg|jpeg|tga|tif|tiff|exr|hdr)$', '', img.name, flags=re.IGNORECASE)
    for key in SUFFIX_MAP:
        stem = re.sub(rf'[_\\-]?{re.escape(key)}[_\\-]?', '_', stem, flags=re.IGNORECASE)
    stem = re.sub(r'_+', '_', stem).strip('_')
    suffix = detect_suffix(img.name)
    return f"T_{stem}{suffix}"

renamed = []
for img in bpy.data.images:
    if img.name.startswith('T_'):
        continue
    new = ue_rename_texture(img)
    print(f"  {img.name}  ->  {new}")
    renamed.append((img, new))

for img, new_name in renamed:
    img.name = new_name

print(f"Renamed {len(renamed)} textures.")""",
    },
    {
        "input": "Find duplicate meshes in my scene",
        "output": """import bpy
from collections import defaultdict

def find_duplicate_meshes():
    buckets = defaultdict(list)
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        mesh = obj.data
        key = (len(mesh.vertices), len(mesh.polygons))
        buckets[key].append(obj)

    duplicates = []
    for key, group in buckets.items():
        if len(group) < 2:
            continue
        vol_groups = []
        for obj in group:
            d = obj.dimensions
            vol = d.x * d.y * d.z
            placed = False
            for vg in vol_groups:
                ref_vol = vg[0][1]
                if ref_vol < 1e-9:
                    if vol < 1e-9:
                        vg.append((obj, vol))
                        placed = True
                        break
                elif abs(vol - ref_vol) / ref_vol < 0.01:
                    vg.append((obj, vol))
                    placed = True
                    break
            if not placed:
                vol_groups.append([(obj, vol)])
        for vg in vol_groups:
            if len(vg) >= 2:
                duplicates.append([item[0] for item in vg])

    print(f"Found {len(duplicates)} duplicate group(s):")
    for i, group in enumerate(duplicates):
        print(f"  Group {i+1}: {[o.name for o in group]}")
    return duplicates

find_duplicate_meshes()""",
    },
    {
        "input": "Auto-tag all my assets with metadata",
        "output": """import bpy

CATEGORY_KEYWORDS = {
    'character': ['human', 'char', 'person', 'npc', 'hero', 'villain', 'creature'],
    'vehicle': ['car', 'truck', 'bike', 'plane', 'ship', 'boat', 'aircraft', 'drone'],
    'prop': ['chair', 'table', 'box', 'crate', 'barrel', 'lamp', 'weapon', 'gun'],
    'architecture': ['wall', 'floor', 'ceiling', 'door', 'window', 'building', 'arch'],
    'environment': ['rock', 'tree', 'grass', 'terrain', 'ground', 'cliff', 'mountain'],
    'fx': ['smoke', 'fire', 'explosion', 'spark', 'particle', 'fluid'],
}

def infer_category(obj):
    name_lower = obj.name.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return cat
    if obj.type == 'MESH' and len(obj.data.vertices) > 5000:
        if obj.find_armature():
            return 'character'
    return 'prop'

def infer_topology_quality(obj):
    if obj.type != 'MESH':
        return 'unknown'
    mesh = obj.data
    if len(mesh.polygons) == 0:
        return 'empty'
    n_gon_count = sum(1 for f in mesh.polygons if len(f.vertices) > 4)
    tri_count = sum(1 for f in mesh.polygons if len(f.vertices) == 3)
    total = len(mesh.polygons)
    n_gon_ratio = n_gon_count / total
    tri_ratio = tri_count / total
    if n_gon_ratio > 0.1:
        return 'poor'
    if n_gon_ratio > 0.02 or tri_ratio > 0.5:
        return 'fair'
    if tri_ratio > 0.1:
        return 'good'
    return 'excellent'

tagged = 0
for obj in bpy.data.objects:
    if obj.type not in {'MESH', 'ARMATURE'}:
        continue
    obj['nalana_category'] = infer_category(obj)
    obj['nalana_topology'] = infer_topology_quality(obj)
    obj['nalana_uv_status'] = 'unwrapped' if (obj.type == 'MESH' and obj.data.uv_layers) else 'no_uvs'
    obj['nalana_rig_status'] = 'rigged' if obj.find_armature() else 'none'
    obj['nalana_poly_count'] = len(obj.data.polygons) if obj.type == 'MESH' else 0
    tagged += 1
    print(f"Tagged: {obj.name}")

print(f"Auto-tagged {tagged} objects.")""",
    },
    {
        "input": "Organize my blend file into a clean collection hierarchy",
        "output": """import bpy

STRUCTURE = {
    'Characters': ['SK_', 'CH_', 'RIG_'],
    'Vehicles':   ['SM_Veh_', 'SK_Veh_'],
    'Architecture': ['SM_Arch_', 'SM_Building_'],
    'Props':      ['SM_Prop_', 'SM_'],
    'Environment': ['SM_Env_', 'SM_Rock_', 'SM_Tree_', 'SM_Terrain_'],
    'VFX':        ['FX_', 'PS_'],
    'Lights':     [],
    'Cameras':    [],
    '_Unsorted':  [],
}

def get_or_create_collection(name, parent=None):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        target = parent if parent else bpy.context.scene.collection
        target.children.link(col)
    return col

def find_target_collection(obj):
    if obj.type == 'LIGHT':
        return get_or_create_collection('Lights')
    if obj.type == 'CAMERA':
        return get_or_create_collection('Cameras')
    for col_name, prefixes in STRUCTURE.items():
        if col_name in ('_Unsorted', 'Lights', 'Cameras'):
            continue
        for prefix in prefixes:
            if obj.name.startswith(prefix):
                return get_or_create_collection(col_name)
    return get_or_create_collection('_Unsorted')

for col_name in STRUCTURE:
    get_or_create_collection(col_name)

for obj in bpy.data.objects:
    target_col = find_target_collection(obj)
    for col in obj.users_collection:
        col.objects.unlink(obj)
    target_col.objects.link(obj)
    print(f"  {obj.name} -> {target_col.name}")

print("Library organization complete.")""",
    },
    {
        "input": "Export all my LODs as a single FBX with LOD groups",
        "output": """import bpy
import re

def export_lod_groups(output_path: str):
    lod_groups = {}
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        match = re.match(r'^(.+?)_LOD(\\d)$', obj.name)
        if match:
            base_name = match.group(1)
            lod_idx = int(match.group(2))
            if base_name not in lod_groups:
                lod_groups[base_name] = {}
            lod_groups[base_name][lod_idx] = obj

    if not lod_groups:
        print("No LOD objects found. Objects must follow: AssetName_LOD0, AssetName_LOD1 ...")
        return

    bpy.ops.object.select_all(action='DESELECT')
    for base_name, lods in lod_groups.items():
        print(f"  LOD group: {base_name} ({len(lods)} levels)")
        for lod_idx, obj in sorted(lods.items()):
            obj.select_set(True)

    bpy.ops.export_scene.fbx(
        filepath=output_path,
        use_selection=True,
        apply_scale_options='FBX_SCALE_ALL',
        object_types={'MESH'},
        mesh_smooth_type='FACE',
        use_mesh_modifiers=True,
        add_leaf_bones=False,
        bake_anim=False,
        path_mode='COPY',
        embed_textures=False,
    )
    print(f"Exported LOD FBX to: {output_path}")

export_lod_groups("/tmp/asset_lods.fbx")""",
    },
    {
        "input": "Find all objects with overlapping UVs",
        "output": """import bpy
import bmesh

def check_uv_overlaps(obj):
    if obj.type != 'MESH':
        return False
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    uv_layer = bm.loops.layers.uv.active
    if uv_layer is None:
        bm.free()
        return False

    triangles = []
    for face in bm.faces:
        if len(face.loops) < 3:
            continue
        uvs = [loop[uv_layer].uv.copy() for loop in face.loops]
        for i in range(1, len(uvs) - 1):
            triangles.append((uvs[0], uvs[i], uvs[i+1]))
    bm.free()

    def tri_aabb(tri):
        xs = [v.x for v in tri]
        ys = [v.y for v in tri]
        return min(xs), max(xs), min(ys), max(ys)

    aabbs = [tri_aabb(t) for t in triangles]
    for i in range(len(aabbs)):
        for j in range(i + 1, len(aabbs)):
            ax1, ax2, ay1, ay2 = aabbs[i]
            bx1, bx2, by1, by2 = aabbs[j]
            if ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1:
                return True
    return False

overlapping = [obj.name for obj in bpy.data.objects if check_uv_overlaps(obj)]
if overlapping:
    print(f"Objects with overlapping UVs ({len(overlapping)}):")
    for name in overlapping:
        print(f"  {name}")
else:
    print("No overlapping UVs found.")""",
    },
    {
        "input": "Create a preview render of every asset in my library",
        "output": """import bpy
from pathlib import Path

OUTPUT_DIR = Path("/tmp/asset_previews")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

bpy.context.scene.render.engine = 'BLENDER_EEVEE'
bpy.context.scene.render.resolution_x = 512
bpy.context.scene.render.resolution_y = 512
bpy.context.scene.render.film_transparent = True
bpy.context.scene.render.image_settings.file_format = 'PNG'

camera_obj = next((o for o in bpy.data.objects if o.type == 'CAMERA'), None)
if camera_obj is None:
    cam_data = bpy.data.cameras.new("Preview_Camera")
    camera_obj = bpy.data.objects.new("Preview_Camera", cam_data)
    bpy.context.scene.collection.objects.link(camera_obj)
bpy.context.scene.camera = camera_obj

def frame_camera(obj, camera):
    import mathutils
    bbox_corners = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    center = sum(bbox_corners, mathutils.Vector()) / 8
    radius = max((c - center).length for c in bbox_corners)
    offset = mathutils.Vector((1, -1, 0.7)).normalized() * radius * 2.5
    camera.location = center + offset
    direction = center - camera.location
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

rendered = 0
for obj in bpy.data.objects:
    if obj.type not in {'MESH', 'CURVE', 'META', 'FONT', 'VOLUME'}:
        continue
    for o in bpy.data.objects:
        o.hide_render = (o != obj)
    obj.hide_render = False
    frame_camera(obj, camera_obj)
    safe_name = obj.name.replace('/', '_').replace('\\\\', '_')
    bpy.context.scene.render.filepath = str(OUTPUT_DIR / f"{safe_name}.png")
    bpy.ops.render.render(write_still=True)
    rendered += 1
    print(f"  Rendered: {obj.name}")

for o in bpy.data.objects:
    o.hide_render = False

print(f"Completed {rendered} preview renders in {OUTPUT_DIR}")""",
    },
    {
        "input": "What objects in my scene aren't used anywhere?",
        "output": """import bpy

def find_unused_assets():
    report = {
        'meshes': [],
        'materials': [],
        'textures': [],
        'node_groups': [],
        'actions': [],
        'objects_not_in_scene': [],
    }
    scene_objects = set(bpy.context.scene.objects)
    for obj in bpy.data.objects:
        if obj not in scene_objects:
            report['objects_not_in_scene'].append(obj.name)
    for mesh in bpy.data.meshes:
        if mesh.users == 0:
            report['meshes'].append(mesh.name)
    for mat in bpy.data.materials:
        if mat.users == 0:
            report['materials'].append(mat.name)
    for img in bpy.data.images:
        if img.users == 0:
            report['textures'].append(img.name)
    for ng in bpy.data.node_groups:
        if ng.users == 0:
            report['node_groups'].append(ng.name)
    for action in bpy.data.actions:
        if action.users == 0:
            report['actions'].append(action.name)

    total = sum(len(v) for v in report.values())
    print(f"Found {total} unused asset(s):")
    for category, names in report.items():
        if names:
            print(f"  {category} ({len(names)}):")
            for n in names:
                print(f"    - {n}")
    return report

find_unused_assets()""",
    },
    {
        "input": "Pack all textures into the blend file for delivery",
        "output": """import bpy
from pathlib import Path

def pack_and_report():
    missing = []
    packed_before = []
    external = []
    for img in bpy.data.images:
        if img.source == 'FILE':
            if img.packed_file:
                packed_before.append(img.name)
            else:
                filepath = bpy.path.abspath(img.filepath)
                if not Path(filepath).exists():
                    missing.append((img.name, img.filepath))
                else:
                    external.append(img.name)

    if missing:
        print(f"WARNING: {len(missing)} texture(s) missing from disk (cannot be packed):")
        for name, path in missing:
            print(f"  {name}: {path}")

    bpy.ops.file.pack_all()
    newly_packed = [img.name for img in bpy.data.images if img.source == 'FILE' and img.packed_file]

    print(f"Texture packing report:")
    print(f"  Already packed: {len(packed_before)}")
    print(f"  Newly packed:   {len(newly_packed) - len(packed_before)}")
    print(f"  Missing:        {len(missing)}")
    bpy.ops.wm.save_mainfile()
    print("Blend file saved with all textures packed.")

pack_and_report()""",
    },
    {
        "input": "List all assets sorted by polygon count",
        "output": """import bpy

asset_data = []
for obj in bpy.data.objects:
    if obj.type != 'MESH':
        continue
    asset_data.append({
        'name': obj.name,
        'polys': len(obj.data.polygons),
        'verts': len(obj.data.vertices),
        'materials': len(obj.material_slots),
    })

asset_data.sort(key=lambda x: x['polys'], reverse=True)
print(f"{'Name':<40} {'Polys':>10} {'Verts':>10} {'Mats':>6}")
print("-" * 70)
for a in asset_data:
    print(f"{a['name']:<40} {a['polys']:>10,} {a['verts']:>10,} {a['materials']:>6}")
print(f"\\nTotal objects: {len(asset_data)}")
print(f"Total polygons in scene: {sum(a['polys'] for a in asset_data):,}")""",
    },
    {
        "input": "Check my scene for non-manifold geometry",
        "output": """import bpy
import bmesh

def find_non_manifold(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()
    nm_edges = [e for e in bm.edges if not e.is_manifold]
    nm_verts = [v for v in bm.verts if not v.is_manifold]
    boundary = [e for e in bm.edges if e.is_boundary]
    result = {
        'non_manifold_edges': len(nm_edges),
        'non_manifold_verts': len(nm_verts),
        'boundary_edges': len(boundary),
        'is_clean': len(nm_edges) == 0 and len(nm_verts) == 0,
    }
    bm.free()
    return result

print(f"{'Object':<40} {'NM Edges':>10} {'NM Verts':>10} {'Clean':>8}")
print("-" * 72)
for obj in bpy.data.objects:
    if obj.type != 'MESH':
        continue
    r = find_non_manifold(obj)
    clean_str = "YES" if r['is_clean'] else "NO"
    print(f"{obj.name:<40} {r['non_manifold_edges']:>10} {r['non_manifold_verts']:>10} {clean_str:>8}")""",
    },
    {
        "input": "Generate a LOD chain from this high-res asset",
        "output": """import bpy

LOD_RATIOS = {
    'LOD0': 1.0,
    'LOD1': 0.5,
    'LOD2': 0.25,
    'LOD3': 0.1,
    'LOD4': 0.02,
}

def generate_lod_chain(source_obj_name: str):
    source = bpy.data.objects.get(source_obj_name)
    if source is None or source.type != 'MESH':
        print(f"Object '{source_obj_name}' not found or not a mesh.")
        return

    original_polys = len(source.data.polygons)
    print(f"Source: {source_obj_name} ({original_polys:,} polygons)")

    lod_col_name = f"{source_obj_name}_LODs"
    lod_col = bpy.data.collections.get(lod_col_name) or bpy.data.collections.new(lod_col_name)
    if lod_col.name not in [c.name for c in bpy.context.scene.collection.children]:
        bpy.context.scene.collection.children.link(lod_col)

    for lod_name, ratio in LOD_RATIOS.items():
        new_mesh = source.data.copy()
        new_obj = bpy.data.objects.new(f"{source_obj_name}_{lod_name}", new_mesh)
        lod_col.objects.link(new_obj)
        if ratio < 1.0:
            dec = new_obj.modifiers.new(name="Decimate", type='DECIMATE')
            dec.decimate_type = 'COLLAPSE'
            dec.ratio = ratio
            bpy.context.view_layer.objects.active = new_obj
            bpy.ops.object.modifier_apply(modifier="Decimate")
        result_polys = len(new_obj.data.polygons)
        print(f"  {lod_name}: {result_polys:,} polygons ({ratio*100:.0f}% of original)")

    print(f"LOD chain created in collection '{lod_col_name}'")

generate_lod_chain("SM_Hero_Prop")""",
    },
    {
        "input": "Find all objects that are missing materials",
        "output": """import bpy

no_material = []
empty_slots = []
for obj in bpy.data.objects:
    if obj.type != 'MESH':
        continue
    if len(obj.material_slots) == 0:
        no_material.append(obj.name)
    elif any(slot.material is None for slot in obj.material_slots):
        empty_slots.append(obj.name)

print(f"No material slots ({len(no_material)}): {no_material}")
print(f"Empty material slots ({len(empty_slots)}): {empty_slots}")
print(f"Total needing assignment: {len(no_material) + len(empty_slots)}")""",
    },
    {
        "input": "Create a full asset inventory of my entire library",
        "output": """import bpy
import json
from pathlib import Path

def build_asset_inventory() -> list[dict]:
    inventory = []
    for obj in bpy.data.objects:
        entry = {
            'name': obj.name,
            'type': obj.type,
            'collection': [col.name for col in obj.users_collection],
            'polygon_count': len(obj.data.polygons) if obj.type == 'MESH' else 0,
            'vertex_count': len(obj.data.vertices) if obj.type == 'MESH' else 0,
            'material_count': len(obj.material_slots),
            'materials': [s.material.name if s.material else None for s in obj.material_slots],
            'has_uvs': bool(obj.data.uv_layers) if obj.type == 'MESH' else False,
            'uv_layer_count': len(obj.data.uv_layers) if obj.type == 'MESH' else 0,
            'has_armature': obj.find_armature() is not None,
            'has_shape_keys': bool(getattr(obj.data, 'shape_keys', None)),
            'is_instanced': obj.data.users > 1 if obj.data else False,
            'modifiers': [m.name for m in obj.modifiers],
            'custom_properties': dict(obj.items()),
            'dimensions': list(obj.dimensions),
        }
        inventory.append(entry)

    inventory.sort(key=lambda x: x['polygon_count'], reverse=True)
    return inventory

inventory = build_asset_inventory()
output_path = Path("/tmp/asset_inventory.json")
with open(output_path, 'w') as f:
    json.dump(inventory, f, indent=2, default=str)

print(f"Asset inventory written to: {output_path}")
print(f"Total assets: {len(inventory)}")
print(f"Total polygons: {sum(a['polygon_count'] for a in inventory):,}")""",
    },
    {
        "input": "Batch apply a studio color palette to all placeholder materials",
        "output": """import bpy

STUDIO_PALETTE = {
    'concrete':    ((0.65, 0.62, 0.58), 0.90, 0.0),
    'wood':        ((0.55, 0.35, 0.15), 0.85, 0.0),
    'metal_dark':  ((0.15, 0.15, 0.18), 0.30, 1.0),
    'metal_light': ((0.80, 0.80, 0.82), 0.20, 1.0),
    'fabric':      ((0.40, 0.30, 0.25), 0.95, 0.0),
    'plastic':     ((0.20, 0.20, 0.22), 0.60, 0.0),
    'glass':       ((0.85, 0.92, 1.00), 0.05, 0.0),
    'default':     ((0.80, 0.80, 0.80), 0.70, 0.0),
}

def is_placeholder(mat):
    if mat is None:
        return True
    name_lower = mat.name.lower()
    return any(p in name_lower for p in ['material', 'mat.', 'placeholder', 'default', 'untitled'])

def classify_material(name_lower):
    if any(k in name_lower for k in ['concrete', 'stone', 'cement', 'plaster']):
        return 'concrete'
    if any(k in name_lower for k in ['wood', 'timber', 'plank', 'oak']):
        return 'wood'
    if any(k in name_lower for k in ['steel', 'iron', 'chrome', 'dark_metal']):
        return 'metal_dark'
    if any(k in name_lower for k in ['aluminum', 'silver', 'light_metal']):
        return 'metal_light'
    if any(k in name_lower for k in ['fabric', 'cloth', 'textile', 'leather']):
        return 'fabric'
    if any(k in name_lower for k in ['plastic', 'rubber', 'polymer']):
        return 'plastic'
    if any(k in name_lower for k in ['glass', 'crystal', 'transparent']):
        return 'glass'
    return 'default'

applied = 0
for mat in bpy.data.materials:
    if not is_placeholder(mat):
        continue
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if bsdf is None:
        bsdf = mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
    category = classify_material(mat.name.lower())
    color, roughness, metallic = STUDIO_PALETTE[category]
    bsdf.inputs['Base Color'].default_value = (*color, 1.0)
    bsdf.inputs['Roughness'].default_value = roughness
    bsdf.inputs['Metallic'].default_value = metallic
    applied += 1
    print(f"  Applied '{category}' palette to: {mat.name}")

print(f"Studio palette applied to {applied} placeholder material(s).")""",
    },
    {
        "input": "Merge all duplicate material slots on every object",
        "output": """import bpy

def deduplicate_material_slots(obj):
    if obj.type != 'MESH' or len(obj.material_slots) <= 1:
        return 0
    seen = {}
    slot_remap = []
    for i, slot in enumerate(obj.material_slots):
        mat_key = slot.material.name if slot.material else None
        if mat_key not in seen:
            seen[mat_key] = i
            slot_remap.append(i)
        else:
            slot_remap.append(seen[mat_key])
    if len(set(slot_remap)) == len(slot_remap):
        return 0
    for poly in obj.data.polygons:
        poly.material_index = slot_remap[poly.material_index]
    unique_indices = sorted(set(slot_remap))
    slots_to_remove = [i for i in range(len(obj.material_slots)) if i not in unique_indices]
    for i in reversed(slots_to_remove):
        obj.active_material_index = i
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.material_slot_remove()
    return len(slots_to_remove)

total_removed = 0
for obj in bpy.data.objects:
    removed = deduplicate_material_slots(obj)
    if removed > 0:
        print(f"  {obj.name}: removed {removed} duplicate slot(s)")
        total_removed += removed
print(f"Total duplicate slots removed: {total_removed}")""",
    },
    {
        "input": "Find all objects with unapplied scale or rotation transforms",
        "output": """import bpy

SCALE_EPSILON = 1e-4
ROTATION_EPSILON = 1e-5

problems = []
for obj in bpy.data.objects:
    if obj.type not in {'MESH', 'CURVE', 'ARMATURE'}:
        continue
    issues = {}
    scale = obj.scale
    if any(abs(s - 1.0) > SCALE_EPSILON for s in scale):
        issues['scale'] = tuple(round(s, 4) for s in scale)
    rot = obj.rotation_euler
    if any(abs(r) > ROTATION_EPSILON for r in rot):
        issues['rotation_deg'] = tuple(round(r * 57.2958, 3) for r in rot)
    if issues:
        problems.append((obj.name, issues))

if problems:
    print(f"Objects with unapplied transforms ({len(problems)}):")
    for name, issues in problems:
        print(f"  {name}: {issues}")
    print("Fix: bpy.ops.object.transform_apply(rotation=True, scale=True)")
else:
    print("All objects have clean transforms.")""",
    },
    {
        "input": "Export my scene as a USD file with all textures included",
        "output": """import bpy
from pathlib import Path

def export_usd_with_textures(output_path="/tmp/scene_export.usdc"):
    bpy.ops.wm.usd_export(
        filepath=str(output_path),
        selected_objects_only=False,
        visible_objects_only=True,
        export_animation=True,
        export_hair=False,
        export_uvmaps=True,
        export_normals=True,
        export_materials=True,
        export_textures=True,
        overwrite_textures=False,
        relative_paths=True,
        export_lights=True,
        export_cameras=True,
    )
    out = Path(output_path)
    if out.exists():
        size_mb = out.stat().st_size / 1_048_576
        print(f"USD exported to: {output_path} ({size_mb:.2f} MB)")
        tex_dir = out.parent / "textures"
        if tex_dir.exists():
            print(f"Textures exported: {len(list(tex_dir.glob('*')))}")

export_usd_with_textures()""",
    },
    {
        "input": "What's the total file size of all external textures in my project?",
        "output": """import bpy
from pathlib import Path

total_size = 0
file_report = []
missing = []

for img in bpy.data.images:
    if img.source != 'FILE' or img.packed_file:
        continue
    abs_path = bpy.path.abspath(img.filepath)
    p = Path(abs_path)
    if p.exists():
        size_bytes = p.stat().st_size
        total_size += size_bytes
        file_report.append({
            'name': img.name,
            'size_mb': round(size_bytes / 1_048_576, 2),
            'resolution': f"{img.size[0]}x{img.size[1]}",
        })
    else:
        missing.append({'name': img.name, 'path': img.filepath})

file_report.sort(key=lambda x: x['size_mb'], reverse=True)
print(f"{'Name':<35} {'Size (MB)':>10} {'Resolution':>12}")
print("-" * 60)
for r in file_report:
    print(f"{r['name']:<35} {r['size_mb']:>10.2f} {r['resolution']:>12}")
print(f"\\nTotal texture storage: {total_size / 1_048_576:.2f} MB")
if missing:
    print(f"Missing textures ({len(missing)}): {[m['name'] for m in missing]}")""",
    },
    {
        "input": "Remove all empty collections from my scene",
        "output": """import bpy

def remove_empty_collections():
    def is_empty_recursive(col):
        if col.objects:
            return False
        return all(is_empty_recursive(child) for child in col.children)

    removed = []
    changed = True
    while changed:
        changed = False
        for col in list(bpy.data.collections):
            if col.name == bpy.context.scene.collection.name:
                continue
            if is_empty_recursive(col):
                bpy.data.collections.remove(col)
                removed.append(col.name)
                changed = True

    if removed:
        print(f"Removed {len(removed)} empty collection(s): {removed}")
    else:
        print("No empty collections found.")

remove_empty_collections()""",
    },
    {
        "input": "Show me all assets that need UV unwrapping",
        "output": """import bpy

needs_uv = []
has_uv = []
for obj in bpy.data.objects:
    if obj.type != 'MESH':
        continue
    if not obj.data.uv_layers:
        needs_uv.append(obj.name)
    else:
        has_uv.append(obj.name)

print(f"Meshes WITHOUT UV maps ({len(needs_uv)}):")
for n in needs_uv:
    print(f"  {n}")
print(f"\\nMeshes with UVs: {len(has_uv)}")

# Auto-UV-unwrap all meshes missing UVs
if needs_uv:
    print("\\nAuto-unwrapping...")
    bpy.ops.object.select_all(action='DESELECT')
    for name in needs_uv:
        obj = bpy.data.objects[name]
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(angle_limit=66.0, margin_method='SCALED', island_margin=0.02)
        bpy.ops.object.mode_set(mode='OBJECT')
        obj.select_set(False)
    print(f"Auto-unwrapped {len(needs_uv)} objects.")""",
    },
    {
        "input": "Validate all my assets are export-ready for Unreal Engine",
        "output": """import bpy

def validate_for_unreal(obj) -> list[str]:
    issues = []
    if obj.type != 'MESH':
        return issues
    mesh = obj.data

    # Check polygon count
    if len(mesh.polygons) == 0:
        issues.append("Empty mesh — no polygons")

    # Check UVs
    if not mesh.uv_layers:
        issues.append("Missing UV map — required for Unreal import")

    # Check scale
    if any(abs(s - 1.0) > 0.001 for s in obj.scale):
        issues.append(f"Unapplied scale: {tuple(round(s,3) for s in obj.scale)}")

    # Check materials
    if len(obj.material_slots) == 0:
        issues.append("No material assigned")
    elif any(slot.material is None for slot in obj.material_slots):
        issues.append("Empty material slot(s)")

    # Check name follows UE convention
    ue_prefixes = ('SM_', 'SK_', 'M_', 'MI_', 'T_', 'FX_', 'BP_')
    if not obj.name.startswith(ue_prefixes):
        issues.append(f"Name doesn't follow UE convention (should start with SM_, SK_, etc.)")

    # Check for N-gons (Unreal handles them but it's bad practice)
    n_gon_count = sum(1 for f in mesh.polygons if len(f.vertices) > 4)
    if n_gon_count > 0:
        issues.append(f"{n_gon_count} n-gon(s) — triangulate before export")

    return issues

all_clear = True
for obj in bpy.data.objects:
    if obj.type != 'MESH':
        continue
    issues = validate_for_unreal(obj)
    if issues:
        all_clear = False
        print(f"{obj.name}:")
        for issue in issues:
            print(f"  FAIL: {issue}")
    else:
        print(f"{obj.name}: PASS")

if all_clear:
    print("\\nAll assets are Unreal Engine export-ready.")
else:
    print("\\nFix the above issues before exporting.")""",
    },
    {
        "input": "Create a collection of all assets that use a specific texture",
        "output": """import bpy

TARGET_TEXTURE = "T_Brick_Wall_D"  # Change to your target texture name

def find_objects_using_texture(texture_name: str) -> list:
    target_img = bpy.data.images.get(texture_name)
    if target_img is None:
        print(f"Texture '{texture_name}' not found in blend file.")
        return []

    using_objects = []
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        for slot in obj.material_slots:
            if slot.material is None or not slot.material.use_nodes:
                continue
            for node in slot.material.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image == target_img:
                    using_objects.append(obj)
                    break

    return using_objects

objects_using = find_objects_using_texture(TARGET_TEXTURE)
print(f"Objects using '{TARGET_TEXTURE}' ({len(objects_using)}):")
for obj in objects_using:
    print(f"  {obj.name}")

# Optionally group them into a collection
if objects_using:
    col_name = f"Uses_{TARGET_TEXTURE}"
    col = bpy.data.collections.get(col_name) or bpy.data.collections.new(col_name)
    if col.name not in [c.name for c in bpy.context.scene.collection.children]:
        bpy.context.scene.collection.children.link(col)
    for obj in objects_using:
        if obj.name not in col.objects:
            col.objects.link(obj)
    print(f"Grouped in collection '{col_name}'")""",
    },
]

# ─── Dataclass ─────────────────────────────────────────────────────────────────


@dataclass
class AssetLibraryEntry:
    """Metadata record for a single 3D asset in the library."""

    name: str
    category: str
    style: str
    state: str
    poly_count: int
    vert_count: int
    uv_status: str
    rig_status: str
    software_compatibility: list[str]
    topology_quality: str
    file_path: str
    file_format: str
    materials: list[str]
    tags: list[str]
    custom_properties: dict[str, Any] = field(default_factory=dict)


# ─── Class ─────────────────────────────────────────────────────────────────────


class AssetManager:
    """
    Asset library management domain agent for Nalana.

    Generates and manages training data for 3D asset library tasks including
    semantic search, auto-tagging, duplicate detection, and batch operations.
    """

    def __init__(self, library_path: str | None = None) -> None:
        self.library_path = Path(library_path) if library_path else None
        self.taxonomy = ASSET_TAXONOMY
        self.naming_conventions = ASSET_NAMING_CONVENTIONS
        self.format_support = FILE_FORMAT_SUPPORT
        self.duplicate_strategies = DUPLICATE_DETECTION_STRATEGIES
        self._rng = random.Random(42)

    # ── Library Scanning ───────────────────────────────────────────────────────

    def scan_library(self, path: str) -> list[dict[str, Any]]:
        """
        Scan a directory for 3D asset files and extract metadata from filenames
        and directory structure. Returns a list of asset metadata dicts.

        Metadata is inferred from naming conventions and file extensions without
        opening each file — for production use, extend with format-specific readers.
        """
        scan_path = Path(path)
        if not scan_path.exists():
            return []

        assets: list[dict[str, Any]] = []
        supported_extensions = set(self.format_support.keys())

        for file_path in scan_path.rglob("*"):
            if file_path.suffix.lower() not in supported_extensions:
                continue
            entry = self._infer_metadata_from_path(file_path, scan_path)
            assets.append(entry)

        return assets

    def _infer_metadata_from_path(
        self, file_path: Path, library_root: Path
    ) -> dict[str, Any]:
        """Infer asset metadata from file path, name, and directory structure."""
        stem = file_path.stem
        relative = file_path.relative_to(library_root)
        parts = [p.lower() for p in relative.parts]
        stem_lower = stem.lower()

        # Infer category from UE-style prefix first
        category = "prop"
        prefix_map = {
            "sm_": "prop",
            "sk_": "character",
            "ch_": "character",
            "rig_": "rig",
            "m_": "material",
            "mi_": "material",
            "t_": "texture",
            "fx_": "fx",
            "hdri_": "hdri",
            "bp_": "prop",
        }
        for prefix, cat in prefix_map.items():
            if stem_lower.startswith(prefix):
                category = cat
                break

        # Override with directory-level hints
        category_dir_map = {
            "character": "character",
            "characters": "character",
            "vehicle": "vehicle",
            "vehicles": "vehicle",
            "architecture": "architecture",
            "props": "prop",
            "environment": "environment",
            "environments": "environment",
            "fx": "fx",
            "vfx": "fx",
            "textures": "texture",
            "materials": "material",
            "hdri": "hdri",
        }
        for part in parts[:-1]:
            if part in category_dir_map:
                category = category_dir_map[part]
                break

        lod_match = re.search(r"_lod(\d)", stem_lower)
        lod_level = int(lod_match.group(1)) if lod_match else None

        state = "finished"
        for state_keyword in ["wip", "draft", "placeholder", "proxy", "blockout"]:
            if state_keyword in stem_lower:
                state = "WIP" if state_keyword == "wip" else "placeholder"
                break

        ext = file_path.suffix.lower()
        fmt_info = self.format_support.get(ext, {})

        return {
            "name": stem,
            "file_path": str(file_path),
            "file_format": ext,
            "relative_path": str(relative),
            "category": category,
            "state": state,
            "lod_level": lod_level,
            "has_animation_support": fmt_info.get("animation_support", False),
            "material_support": fmt_info.get("material_support", "unknown"),
            "inferred_tags": self.auto_tag(
                {"name": stem, "category": category, "state": state, "file_format": ext}
            ),
        }

    # ── Auto-Tagging ───────────────────────────────────────────────────────────

    def auto_tag(self, asset_metadata: dict[str, Any]) -> list[str]:
        """
        Generate suggested tags for an asset based on its metadata.
        Returns a sorted, deduplicated list of tag strings from the taxonomy.
        """
        tags: list[str] = []
        name = asset_metadata.get("name", "").lower()
        category = asset_metadata.get("category", "")
        state = asset_metadata.get("state", "")
        poly_count = asset_metadata.get("poly_count", 0)
        file_format = asset_metadata.get("file_format", "")

        if category in self.taxonomy.get("category", {}):
            tags.append(f"category:{category}")

        if state in self.taxonomy.get("state", {}):
            tags.append(f"state:{state}")

        style_keywords: dict[str, list[str]] = {
            "stylized": ["stylized", "stylised", "cartoon", "toony", "toon"],
            "lowpoly": ["lowpoly", "low_poly", "lp_", "_lp", "lo_poly"],
            "scifi": ["scifi", "sci_fi", "cyber", "neon", "mech", "robot"],
            "fantasy": ["fantasy", "magic", "medieval", "dragon", "sword", "castle"],
            "photorealistic": ["scan", "photogrammetry", "real", "photo", "lidar"],
        }
        for style, keywords in style_keywords.items():
            if any(kw in name for kw in keywords):
                tags.append(f"style:{style}")
                break

        if poly_count > 500_000:
            tags.append("polygon_tier:highpoly")
            tags.append("state:hero")
        elif poly_count > 50_000:
            tags.append("polygon_tier:medium_high")
        elif 0 < poly_count < 5_000:
            tags.append("polygon_tier:lowpoly")

        for i in range(5):
            if f"_lod{i}" in name:
                tags.append(f"lod:LOD{i}")
                break

        if file_format == ".blend":
            tags.append("software_compatibility:blender")
        elif file_format in (".fbx", ".obj"):
            tags.append("software_compatibility:all")
        elif file_format in (".gltf", ".glb"):
            tags.append("software_compatibility:unity")
            tags.append("software_compatibility:unreal")
        elif file_format in (".usd", ".usda", ".usdc", ".usdz"):
            tags.append("software_compatibility:unreal")
            tags.append("software_compatibility:houdini")

        if "udim" in name:
            tags.append("uv_status:udim")
        elif "tile" in name or "tileable" in name:
            tags.append("uv_status:tileable")

        return sorted(set(tags))

    # ── Duplicate Detection ────────────────────────────────────────────────────

    def find_duplicates(
        self, assets: list[dict[str, Any]]
    ) -> list[list[dict[str, Any]]]:
        """
        Group assets that are likely duplicates using multi-pass heuristics.

        Pass 1: Exact stem name match (case-insensitive, version suffixes stripped)
        Pass 2: Polygon count + bounding box volume similarity within tolerance
        Pass 3: Material name Jaccard similarity >= 0.5

        Returns a list of groups; each group contains the duplicate asset dicts.
        """
        duplicate_groups: list[list[dict[str, Any]]] = []
        used_indices: set[int] = set()

        # Pass 1: name similarity
        name_buckets: dict[str, list[int]] = {}
        for i, asset in enumerate(assets):
            stem = Path(
                asset.get("file_path", asset.get("name", f"asset_{i}"))
            ).stem.lower()
            stem = re.sub(r"_v\d+$", "", stem)
            stem = re.sub(r"_copy\d*$", "", stem)
            stem = re.sub(r"\.\d+$", "", stem)
            name_buckets.setdefault(stem, []).append(i)

        for indices in name_buckets.values():
            if len(indices) >= 2:
                group = [assets[i] for i in indices]
                duplicate_groups.append(group)
                used_indices.update(indices)

        # Pass 2: polygon count + bounding box
        remaining = [
            (i, assets[i]) for i in range(len(assets)) if i not in used_indices
        ]
        for i, (idx_a, asset_a) in enumerate(remaining):
            if idx_a in used_indices:
                continue
            poly_a = asset_a.get("poly_count", -1)
            vol_a = asset_a.get("bbox_volume", -1.0)
            if poly_a < 0 or vol_a < 0:
                continue
            group = [asset_a]
            for idx_b, asset_b in remaining[i + 1 :]:
                if idx_b in used_indices:
                    continue
                poly_b = asset_b.get("poly_count", -1)
                vol_b = asset_b.get("bbox_volume", -1.0)
                if poly_b < 0 or vol_b < 0:
                    continue
                max_poly = max(poly_a, poly_b, 1)
                if abs(poly_a - poly_b) / max_poly > 0.02:
                    continue
                max_vol = max(vol_a, vol_b, 1e-9)
                if abs(vol_a - vol_b) / max_vol > 0.05:
                    continue
                group.append(asset_b)
                used_indices.add(idx_b)
            if len(group) >= 2:
                used_indices.add(idx_a)
                duplicate_groups.append(group)

        # Pass 3: material Jaccard
        remaining = [
            (i, assets[i]) for i in range(len(assets)) if i not in used_indices
        ]
        for i, (idx_a, asset_a) in enumerate(remaining):
            if idx_a in used_indices:
                continue
            mats_a = set(asset_a.get("materials", []))
            if not mats_a:
                continue
            group = [asset_a]
            for idx_b, asset_b in remaining[i + 1 :]:
                if idx_b in used_indices:
                    continue
                mats_b = set(asset_b.get("materials", []))
                if not mats_b:
                    continue
                union = mats_a | mats_b
                intersection = mats_a & mats_b
                jaccard = len(intersection) / len(union) if union else 0.0
                if jaccard >= 0.5:
                    group.append(asset_b)
                    used_indices.add(idx_b)
            if len(group) >= 2:
                used_indices.add(idx_a)
                duplicate_groups.append(group)

        return duplicate_groups

    # ── Training Pair Generation ───────────────────────────────────────────────

    def generate_training_pairs(self, n: int = 150) -> list[dict[str, Any]]:
        """
        Generate n training pairs for asset management tasks.
        Mixes all static curated pairs with dynamically generated variants.
        """
        pairs: list[dict[str, Any]] = []

        for pair in ASSET_MANAGEMENT_TRAINING_PAIRS:
            pairs.append(
                {
                    "input": pair["input"],
                    "output": pair["output"].strip(),
                    "domain": "asset_management",
                    "source": "curated",
                    "task_type": self._classify_task_type(pair["input"]),
                }
            )

        search_queries = [
            "Find all character meshes",
            "Search for vehicle assets under 20K polygons",
            "Find all assets tagged as WIP",
            "Show me all HDRI lighting assets",
            "Find props with more than 10K polygons",
            "Search for assets with UDIM UV layouts",
            "Find all fully rigged characters",
            "Show environment assets with photorealistic style",
            "Find all unreal-compatible assets",
            "Search for lowpoly background assets",
            "Find all assets without materials",
            "Show me all skeletal meshes",
            "Find every object larger than 5 meters",
            "Search for FX assets in the scene",
            "List all assets from the Architecture collection",
        ]

        batch_ops = [
            "apply_smooth_shading",
            "rename_to_ue_convention",
            "add_subdivision_modifier",
            "remove_doubles",
            "triangulate",
            "apply_scale",
            "merge_by_distance",
            "recalculate_normals",
            "set_origin_to_geometry",
        ]

        naming_targets = ["unreal", "unity", "maya", "studio_x", "blender"]

        while len(pairs) < n:
            roll = self._rng.random()
            if roll < 0.35:
                query = self._rng.choice(search_queries)
                pairs.append(self.generate_search_pair(query))
            elif roll < 0.65:
                op = self._rng.choice(batch_ops)
                pairs.append(self.generate_batch_op_pair(op))
            else:
                target = self._rng.choice(naming_targets)
                pairs.append(self.generate_naming_pair(target))

        return pairs[:n]

    def generate_search_pair(self, query: str) -> dict[str, Any]:
        """
        Generate a semantic search training pair for an asset library query.
        Parses the query to determine filter criteria and emits executable Blender Python.
        """
        query_lower = query.lower()
        filters: list[str] = []
        code_conditions: list[str] = []

        # Category detection
        for cat in self.taxonomy["category"]:
            if cat in query_lower:
                filters.append(f"category == '{cat}'")
                code_conditions.append(
                    f"(obj.get('nalana_category') == '{cat}' or '{cat}' in obj.name.lower())"
                )
                break

        # Polygon count ceiling
        poly_match = re.search(r"under\s+(\d+)[Kk]?\s+polygon", query_lower)
        if poly_match:
            raw = poly_match.group(1)
            limit = int(raw) * 1000 if "k" in poly_match.group(0).lower() else int(raw)
            filters.append(f"poly_count < {limit}")
            code_conditions.append(
                f"(len(obj.data.polygons) < {limit} if obj.type == 'MESH' else False)"
            )

        # State detection
        for state in self.taxonomy["state"]:
            if state.lower() in query_lower:
                filters.append(f"state == '{state}'")
                code_conditions.append(f"obj.get('nalana_state') == '{state}'")
                break

        # Rig status
        if "rigged" in query_lower or "skeletal" in query_lower:
            filters.append("rig_status != 'none'")
            code_conditions.append("obj.find_armature() is not None")

        # Software compatibility
        for sw in self.taxonomy["software_compatibility"]:
            if sw in query_lower:
                filters.append(f"software_compatibility includes '{sw}'")
                code_conditions.append(f"obj.get('nalana_software') == '{sw}'")
                break

        # UV status
        if "udim" in query_lower:
            filters.append("uv_status == 'udim'")
            code_conditions.append("obj.get('nalana_uv_status') == 'udim'")

        if not code_conditions:
            words = [w for w in re.split(r"\\W+", query_lower) if len(w) > 3]
            code_conditions.append(f"any(kw in obj.name.lower() for kw in {words!r})")

        condition_str = " and ".join(code_conditions) if code_conditions else "True"

        output_code = (
            f"import bpy\n\n"
            f"def search_assets():\n"
            f'    """Semantic search: {query}"""\n'
            f"    results = []\n"
            f"    for obj in bpy.data.objects:\n"
            f"        if obj.type not in {{'MESH', 'ARMATURE', 'CURVE'}}:\n"
            f"            continue\n"
            f"        if {condition_str}:\n"
            f"            results.append(obj)\n"
            f'    print(f"Query: \\"{query}\\"  |  Found {{len(results)}} asset(s):")\n'
            f"    for obj in results:\n"
            f"        poly_count = len(obj.data.polygons) if obj.type == 'MESH' else 0\n"
            f'        print(f"  {{obj.name}} ({{poly_count:,}} polys)")\n'
            f"    return results\n\n"
            f"search_assets()"
        )

        return {
            "input": query,
            "output": output_code,
            "domain": "asset_management",
            "source": "generated",
            "task_type": "semantic_search",
            "filters_applied": filters,
        }

    def generate_batch_op_pair(self, operation: str) -> dict[str, Any]:
        """
        Generate a batch operation training pair for a given operation type.
        Returns a dict with 'input' and fully implemented 'output' Blender Python.
        """
        op_templates: dict[str, dict[str, str]] = {
            "apply_smooth_shading": {
                "input": "Apply smooth shading to all mesh objects in my scene",
                "output": """import bpy

bpy.ops.object.select_all(action='DESELECT')
smooth_applied = 0
for obj in bpy.data.objects:
    if obj.type != 'MESH':
        continue
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_smooth()
    obj.data.use_auto_smooth = True
    obj.data.auto_smooth_angle = 0.523599  # 30 degrees
    obj.select_set(False)
    smooth_applied += 1
print(f"Applied smooth shading to {smooth_applied} mesh objects.")""",
            },
            "remove_doubles": {
                "input": "Remove duplicate vertices on all mesh objects",
                "output": """import bpy
import bmesh

total_removed = 0
for obj in bpy.data.objects:
    if obj.type != 'MESH':
        continue
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    before = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)
    after = len(bm.verts)
    removed = before - after
    if removed > 0:
        bm.to_mesh(obj.data)
        obj.data.update()
        total_removed += removed
        print(f"  {obj.name}: removed {removed} verts")
    bm.free()
print(f"Total vertices removed: {total_removed:,}")""",
            },
            "triangulate": {
                "input": "Triangulate all meshes for game engine export",
                "output": """import bpy
import bmesh

triangulated = 0
for obj in bpy.data.objects:
    if obj.type != 'MESH':
        continue
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method='BEAUTY', ngon_method='BEAUTY')
    bm.to_mesh(obj.data)
    obj.data.update()
    bm.free()
    triangulated += 1
print(f"Triangulated {triangulated} mesh object(s).")""",
            },
            "apply_scale": {
                "input": "Apply scale transforms to all objects",
                "output": """import bpy

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
bpy.ops.object.select_all(action='DESELECT')
print(f"Applied scale to all selected objects.")""",
            },
            "merge_by_distance": {
                "input": "Merge vertices by distance on all objects",
                "output": """import bpy
import bmesh

MERGE_DIST = 0.001  # 1mm threshold
total_merged = 0
for obj in bpy.data.objects:
    if obj.type != 'MESH':
        continue
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    before = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=MERGE_DIST)
    merged = before - len(bm.verts)
    total_merged += merged
    bm.to_mesh(obj.data)
    obj.data.update()
    bm.free()
    if merged > 0:
        print(f"  {obj.name}: merged {merged} vertices")
print(f"Total merged: {total_merged:,} vertices across all objects.")""",
            },
            "recalculate_normals": {
                "input": "Recalculate normals on all mesh objects to fix flipped faces",
                "output": """import bpy
import bmesh

fixed = 0
for obj in bpy.data.objects:
    if obj.type != 'MESH':
        continue
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    obj.data.update()
    bm.free()
    fixed += 1
print(f"Recalculated normals on {fixed} mesh objects.")""",
            },
            "set_origin_to_geometry": {
                "input": "Set origin to geometry center on all mesh objects",
                "output": """import bpy

bpy.ops.object.select_all(action='DESELECT')
reset_count = 0
for obj in bpy.data.objects:
    if obj.type != 'MESH':
        continue
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.select_set(False)
    reset_count += 1
print(f"Reset origin to geometry center on {reset_count} mesh objects.")""",
            },
            "add_subdivision_modifier": {
                "input": "Add a subdivision surface modifier at level 2 to all character meshes",
                "output": """import bpy

CHAR_KEYWORDS = ['char', 'character', 'npc', 'hero', 'villain', 'sk_', 'ch_']
added = 0
for obj in bpy.data.objects:
    if obj.type != 'MESH':
        continue
    if not any(kw in obj.name.lower() for kw in CHAR_KEYWORDS):
        continue
    if any(m.type == 'SUBSURF' for m in obj.modifiers):
        print(f"  Skipped (already has subsurf): {obj.name}")
        continue
    subsurf = obj.modifiers.new(name="Subdivision", type='SUBSURF')
    subsurf.levels = 2
    subsurf.render_levels = 3
    added += 1
    print(f"  Added subdivision to: {obj.name}")
print(f"Subdivision modifier added to {added} character object(s).")""",
            },
            "rename_to_ue_convention": {
                "input": "Rename all objects to follow Unreal Engine naming conventions",
                "output": """import bpy
import re

PREFIX_MAP = {
    'MESH': 'SM_', 'ARMATURE': 'SK_', 'LIGHT': 'L_',
    'CAMERA': 'CAM_', 'CURVE': 'SP_', 'EMPTY': 'PT_',
}
KNOWN_PREFIXES = ('SM_', 'SK_', 'M_', 'MI_', 'T_', 'BP_', 'FX_', 'CH_', 'RIG_', 'L_', 'CAM_')

def to_pascal_case(name: str) -> str:
    for p in KNOWN_PREFIXES:
        if name.startswith(p):
            name = name[len(p):]
            break
    words = name.replace('-', '_').replace('.', '_').split('_')
    return '_'.join(w.capitalize() for w in words if w)

renamed = 0
for obj in bpy.data.objects:
    prefix = PREFIX_MAP.get(obj.type, 'SM_')
    base = to_pascal_case(obj.name)
    new_name = f"{prefix}{base}"
    if new_name != obj.name:
        print(f"  {obj.name} -> {new_name}")
        obj.name = new_name
        renamed += 1
print(f"Renamed {renamed} object(s) to Unreal Engine convention.")""",
            },
        }

        if operation in op_templates:
            tmpl = op_templates[operation]
        else:
            tmpl = {
                "input": f"Apply '{operation}' to all objects in the scene",
                "output": (
                    f"import bpy\n\n"
                    f"# Operation: {operation}\n"
                    f"for obj in bpy.data.objects:\n"
                    f"    pass  # Implement {operation} logic here\n"
                    f"print('{operation} complete.')"
                ),
            }

        return {
            "input": tmpl["input"],
            "output": tmpl["output"].strip(),
            "domain": "asset_management",
            "source": "generated",
            "task_type": "batch_operation",
            "operation": operation,
        }

    def generate_naming_pair(self, target_convention: str) -> dict[str, Any]:
        """
        Generate a naming convention transformation training pair for the target pipeline.
        Returns a dict with 'input' and 'output' (Blender Python script).
        """
        conventions: dict[str, dict[str, str]] = {
            "unreal": {
                "description": "Unreal Engine naming: SM_ SK_ M_ MI_ T_ FX_ BP_",
                "example_before": "Barrel_Wood_01",
                "example_after": "SM_Barrel_Wood_01",
            },
            "unity": {
                "description": "Unity naming: PascalCase, no prefix convention required",
                "example_before": "sm_barrel_wood_01",
                "example_after": "BarrelWood01",
            },
            "maya": {
                "description": "Maya naming: lowercase with underscores, _geo _grp _jnt suffixes",
                "example_before": "SM_Barrel_Wood",
                "example_after": "barrel_wood_geo",
            },
            "blender": {
                "description": "Blender community: descriptive names, no strict prefix",
                "example_before": "SM_Barrel",
                "example_after": "Barrel.Wood.01",
            },
            "studio_x": {
                "description": "Studio X convention: CATEGORY_Name_Variant_LOD",
                "example_before": "SM_Barrel",
                "example_after": "PROP_Barrel_Wood_LOD0",
            },
        }

        conv = conventions.get(target_convention, conventions["unreal"])
        fn_name = f"rename_for_{target_convention.replace('-', '_')}"
        desc = conv["description"]
        ex_before = conv["example_before"]
        ex_after = conv["example_after"]

        output_code = (
            f"import bpy\n"
            f"import re\n\n"
            f"# {desc}\n"
            f"# Example: '{ex_before}' -> '{ex_after}'\n\n"
            f"KNOWN_PREFIXES = ('SM_', 'SK_', 'M_', 'MI_', 'T_', 'FX_', 'BP_', 'CH_', 'RIG_')\n\n"
            f"def {fn_name}(obj) -> str:\n"
            f"    name = obj.name\n"
            f"    for p in KNOWN_PREFIXES:\n"
            f"        if name.startswith(p):\n"
            f"            name = name[len(p):]\n"
            f"            break\n"
            f"    name = name.strip('_')\n"
            f"    return name  # Apply {target_convention}-specific transformation here\n\n"
            f"renamed = 0\n"
            f"for obj in bpy.data.objects:\n"
            f"    new_base = {fn_name}(obj)\n"
            f"    new_name = new_base  # Add prefix/suffix logic for {target_convention}\n"
            f"    if new_name != obj.name:\n"
            f"        obj.name = new_name\n"
            f"        renamed += 1\n"
            f"print(f'Renamed {{renamed}} assets to {target_convention} convention.')"
        )

        return {
            "input": f"Rename all my assets to follow {target_convention} naming conventions",
            "output": output_code,
            "domain": "asset_management",
            "source": "generated",
            "task_type": "naming_convention",
            "target_convention": target_convention,
        }

    @staticmethod
    def _classify_task_type(query: str) -> str:
        """Classify a natural language query into an asset management task type."""
        q = query.lower()
        if any(
            k in q
            for k in ["find", "search", "show me", "list", "which", "what objects"]
        ):
            return "semantic_search"
        if any(k in q for k in ["rename", "naming", "convention"]):
            return "naming_convention"
        if any(k in q for k in ["export", "pack", "deliver", "send"]):
            return "export_operation"
        if any(
            k in q for k in ["organize", "sort", "move to", "collection", "hierarchy"]
        ):
            return "organization"
        if any(k in q for k in ["duplicate", "same", "identical", "copy"]):
            return "duplicate_detection"
        if any(k in q for k in ["tag", "metadata", "property", "label", "auto-tag"]):
            return "auto_tagging"
        if any(
            k in q for k in ["validate", "check", "audit", "verify", "clean", "missing"]
        ):
            return "validation"
        return "batch_operation"


# ─── Entry Point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nalana Asset Management Domain Agent — Training Data Generator"
    )
    parser.add_argument(
        "--generate-pairs",
        action="store_true",
        help="Generate training pairs and save to data/asset_management/",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=150,
        help="Number of training pairs to generate (default: 150)",
    )
    parser.add_argument(
        "--scan",
        type=str,
        default=None,
        metavar="LIBRARY_PATH",
        help="Scan a local asset library directory and print metadata",
    )
    parser.add_argument(
        "--find-duplicates",
        action="store_true",
        help="Run duplicate detection demo on a synthetic asset list",
    )
    args = parser.parse_args()

    manager = AssetManager()

    if args.generate_pairs:
        print(f"Generating {args.n_pairs} asset management training pairs...")
        pairs = manager.generate_training_pairs(n=args.n_pairs)
        with open(PAIRS_OUTPUT, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        print(f"Saved {len(pairs)} pairs to: {PAIRS_OUTPUT}")
        json_out = ASSET_DATA_DIR / "asset_management_pairs_readable.json"
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)
        print(f"Formatted version: {json_out}")

    if args.scan:
        print(f"\nScanning library: {args.scan}")
        assets = manager.scan_library(args.scan)
        print(f"Found {len(assets)} assets:")
        for asset in assets[:20]:
            print(f"  [{asset['category']:12}] {asset['name']}")
        if len(assets) > 20:
            print(f"  ... and {len(assets) - 20} more")

    if args.find_duplicates:
        print("\nRunning duplicate detection demo...")
        demo_assets = [
            {
                "name": "Barrel_Wood",
                "file_path": "props/Barrel_Wood.blend",
                "poly_count": 1024,
                "bbox_volume": 2.5,
                "materials": ["M_Wood", "M_Metal"],
            },
            {
                "name": "Barrel_Wood_v2",
                "file_path": "props/Barrel_Wood_v2.blend",
                "poly_count": 1024,
                "bbox_volume": 2.5,
                "materials": ["M_Wood", "M_Metal"],
            },
            {
                "name": "barrel_wood",
                "file_path": "old/barrel_wood.fbx",
                "poly_count": 1024,
                "bbox_volume": 2.48,
                "materials": ["M_WoodOld"],
            },
            {
                "name": "Chair_01",
                "file_path": "furniture/Chair_01.fbx",
                "poly_count": 800,
                "bbox_volume": 1.2,
                "materials": ["M_Fabric"],
            },
            {
                "name": "Table_Round",
                "file_path": "furniture/Table_Round.fbx",
                "poly_count": 2000,
                "bbox_volume": 4.1,
                "materials": ["M_Glass", "M_Steel"],
            },
        ]
        groups = manager.find_duplicates(demo_assets)
        print(f"Found {len(groups)} duplicate group(s):")
        for i, group in enumerate(groups):
            print(f"  Group {i + 1}: {[a['name'] for a in group]}")

    if not any([args.generate_pairs, args.scan, args.find_duplicates]):
        parser.print_help()


if __name__ == "__main__":
    main()
