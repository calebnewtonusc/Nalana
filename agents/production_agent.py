"""
production_agent.py - Nalana Production Pipeline

Automates the "everyone hates this" work:
  - Retopology (via Blender's Remesh modifier + Quadriflow)
  - UV unwrapping (smart UV project + manual seam suggestions)
  - LOD chain generation (Decimation with animation-zone preservation)
  - Normal map baking (cage generation, ray distance, multi-object)
  - Collision mesh generation (convex decomposition via VHACD)
  - Export validation and packaging

Each operation is available as:
  1. A voice command handler (returns executable Blender Python)
  2. A training pair generator
  3. A standalone function with quality parameters

Usage:
  python production_agent.py --retopo object_name --target-faces 5000
  python production_agent.py --uv object_name --margin 0.01
  python production_agent.py --full-pipeline object_name --platform game_pc
  python production_agent.py --generate-pairs  # generate all training pairs
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Output dir
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parents[1] / "data" / "production"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Platform target specs (mirrors qa_agent profiles for cross-module consistency)
# ---------------------------------------------------------------------------

PLATFORM_SPECS = {
    "mobile": {
        "target_faces": 2_500,
        "lod_levels": [1.0, 0.5, 0.25],
        "texture_size": 512,
        "bake_samples": 32,
        "collision": "simple_box",
        "notes": "Mobile AR/game — 2.5k faces, 512px textures",
    },
    "game_pc": {
        "target_faces": 12_500,
        "lod_levels": [1.0, 0.5, 0.25, 0.1],
        "texture_size": 2048,
        "bake_samples": 128,
        "collision": "convex",
        "notes": "PC game hero — 12.5k faces, 2048px, LOD0-LOD3",
    },
    "cinematics": {
        "target_faces": 125_000,
        "lod_levels": [1.0],
        "texture_size": 4096,
        "bake_samples": 512,
        "collision": "none",
        "notes": "VFX/cinematic — 125k faces, 4096px, no collision needed",
    },
    "arch_viz": {
        "target_faces": 25_000,
        "lod_levels": [1.0, 0.5],
        "texture_size": 2048,
        "bake_samples": 256,
        "collision": "none",
        "notes": "Arch viz — 25k faces, real-time preview quality",
    },
}


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _save_pairs(pairs: List[dict], filename: str) -> Path:
    out_path = DATA_DIR / filename
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Saved {len(pairs)} pairs → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# RetopologyAgent
# ---------------------------------------------------------------------------


class RetopologyAgent:
    """
    Handles all retopology operations: Quadriflow remesh, topology zone pinning,
    face count suggestions, and training pair generation.
    """

    # Voice command templates → (variant, target_faces, smooth)
    _VOICE_TEMPLATES = [
        ("retopo this to {faces} faces", "{faces}", True),
        ("remesh {obj} to {faces} polygons", "{faces}", True),
        ("quadriflow retopology, target {faces} faces", "{faces}", True),
        ("clean up the topology, I want {faces} faces", "{faces}", True),
        ("retopologize for {platform}", None, True),
        ("reduce this to {faces} quads using quadriflow", "{faces}", True),
        ("remesh with {faces} target faces, smooth on", "{faces}", True),
        ("do a quadriflow remesh at {faces} faces", "{faces}", True),
        ("retopo {obj} down to {faces} polys for {platform}", "{faces}", False),
        ("clean topology — {faces} face target, no smoothing", "{faces}", False),
    ]

    def quadriflow_remesh(
        self,
        obj_name: str,
        target_faces: int,
        smooth: bool = True,
    ) -> str:
        """
        Generate Blender Python for Quadriflow remesh operation.

        Args:
            obj_name:     Name of the object in Blender scene.
            target_faces: Target face count for remesh.
            smooth:       Use smooth normals on result.

        Returns:
            Blender Python string.
        """
        return f"""import bpy

obj = bpy.data.objects.get('{obj_name}')
if obj is None:
    raise ValueError("Object '{obj_name}' not found in scene")

# Select and make active
bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)
bpy.context.view_layer.objects.active = obj

# Add Remesh modifier set to Voxel mode first to get a clean base
# then switch to QuadriFlow for all-quad output
mod = obj.modifiers.new(name='QuadriflowRemesh', type='REMESH')
mod.mode = 'QUAD'
mod.target_faces = {target_faces}
mod.use_smooth_normals = {smooth}
mod.use_project_mesh = True  # project back onto original surface

# Apply the modifier
bpy.ops.object.modifier_apply(modifier=mod.name)

# Remove doubles that may appear on seams
bpy.ops.object.editmode_toggle()
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.remove_doubles(threshold=0.0001)
bpy.ops.object.editmode_toggle()

print(f"QuadriFlow remesh complete: {{len(obj.data.polygons)}} faces on '{obj_name}'")
"""

    def preserve_topology_zones(
        self,
        obj_name: str,
        zones: List[str],
    ) -> str:
        """
        Generate Blender Python to pin vertices in named vertex groups
        so they are preserved during remesh/decimate operations.

        Args:
            obj_name: Object name.
            zones:    List of vertex group names to pin (e.g. ["face", "hands", "feet"]).

        Returns:
            Blender Python string.
        """
        zones_repr = repr(zones)
        return f"""import bpy

obj = bpy.data.objects.get('{obj_name}')
if obj is None:
    raise ValueError("Object '{obj_name}' not found")

bpy.context.view_layer.objects.active = obj
bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)

# Enter edit mode and pin vertices in preservation zones
bpy.ops.object.editmode_toggle()
bpy.ops.mesh.select_all(action='DESELECT')

for zone_name in {zones_repr}:
    vg = obj.vertex_groups.get(zone_name)
    if vg is None:
        print(f"Warning: vertex group '{{zone_name}}' not found on '{obj_name}' — skipping")
        continue
    # Select vertices in this group
    obj.vertex_groups.active_index = vg.index
    bpy.ops.object.vertex_group_select()

# Mark selected vertices as 'PINNED' in the sculpt mask
# (Remesh respects the sculpt mask — pinned = 1.0 = fully preserved)
bpy.ops.sculpt.mask_flood_fill(mode='VALUE', value=1.0)
bpy.ops.object.editmode_toggle()

print(f"Pinned topology zones: {zones_repr} on '{obj_name}'")
print("Run QuadriFlow remesh now — these zones will be preserved")
"""

    def suggest_face_count(
        self,
        obj_name: str,
        platform: str,
    ) -> dict:
        """
        Suggest an appropriate face count for a given platform target,
        with reasoning based on the object's current detail level.

        Args:
            obj_name: Object name (used in the returned reasoning string).
            platform: Platform key from PLATFORM_SPECS.

        Returns:
            dict with: recommended_faces, reasoning, platform_budget, reduction_ratio
        """
        spec = PLATFORM_SPECS.get(platform, PLATFORM_SPECS["game_pc"])
        target = spec["target_faces"]

        suggestions = {
            "mobile": {
                "hero_prop": 2_500,
                "background_prop": 500,
                "character": 3_000,
                "environment_piece": 1_000,
                "vehicle": 4_000,
            },
            "game_pc": {
                "hero_prop": 12_500,
                "background_prop": 2_000,
                "character": 15_000,
                "environment_piece": 5_000,
                "vehicle": 20_000,
            },
            "cinematics": {
                "hero_prop": 125_000,
                "background_prop": 25_000,
                "character": 200_000,
                "environment_piece": 50_000,
                "vehicle": 250_000,
            },
            "arch_viz": {
                "hero_prop": 10_000,
                "background_prop": 2_000,
                "character": 8_000,
                "environment_piece": 15_000,
                "vehicle": 12_000,
            },
        }

        type_suggestions = suggestions.get(platform, suggestions["game_pc"])

        return {
            "recommended_faces": target,
            "platform_budget": spec,
            "type_suggestions": type_suggestions,
            "reasoning": (
                f"For {platform} ({spec['notes']}), target {target:,} faces for a hero prop. "
                f"Background props should be 80% lower. Characters can be 20% higher. "
                f"LOD system: {spec['lod_levels']} ratios applied after base retopo."
            ),
            "reduction_ratio_examples": {
                "if_current_is_50k": round(target / 50_000, 3),
                "if_current_is_200k": round(target / 200_000, 3),
                "if_current_is_1m": round(target / 1_000_000, 4),
            },
        }

    def generate_training_pairs(self, n_pairs: int = 200) -> List[dict]:
        """Generate 200 retopology training pairs."""
        pairs: List[dict] = []

        face_counts = [
            500,
            1000,
            2000,
            2500,
            3000,
            5000,
            8000,
            10000,
            12500,
            15000,
            20000,
            25000,
            50000,
        ]
        platforms = list(PLATFORM_SPECS.keys())
        obj_names = [
            "hero_character",
            "weapon",
            "environment_prop",
            "vehicle",
            "tree",
            "building_piece",
            "rock_formation",
            "furniture",
        ]
        zone_options = [
            ["face", "hands"],
            ["eyes", "mouth", "fingers"],
            ["face"],
            ["high_detail_panel", "edge_loops"],
            [],
        ]

        reasonings = [
            "QuadriFlow produces clean all-quad topology ideal for subdivision and skinning. "
            "The target_faces parameter controls density; smooth normals project back to preserve the silhouette of the original high-poly.",
            "Retopology is required before rigging — the remeshed quad topology gives clean edge loops for proper deformation at joints.",
            "Lower polycount reduces draw calls and GPU vertex throughput. "
            "QuadriFlow is preferred over Voxel remesh because it outputs quads rather than triangles.",
            "Preserving topology zones prevents the remesher from collapsing critical high-detail areas like the face or hands. "
            "The sculpt mask pins those vertices while QuadriFlow retopologizes the lower-detail body.",
            "For game engine assets, topology quality affects LOD generation. "
            "Well-spaced quad topology decimates more cleanly than triangulated or ngon meshes.",
        ]

        prefixes = ["", "", "", "hey nalana, ", "can you ", "please "]
        smooth_variants = [True, False]

        for i in range(n_pairs):
            face_count = random.choice(face_counts)
            platform = random.choice(platforms)
            obj_name = random.choice(obj_names)
            smooth = random.choice(smooth_variants)
            zones = random.choice(zone_options)
            prefix = random.choice(prefixes)

            # Pick voice command style
            style = i % 5
            if style == 0:
                voice = f"{prefix}retopo {obj_name} to {face_count:,} faces"
            elif style == 1:
                voice = f"{prefix}remesh {obj_name} with quadriflow, target {face_count:,} polygons"
            elif style == 2:
                voice = f"{prefix}clean up the topology for {platform}, about {face_count:,} faces"
            elif style == 3:
                voice = f"{prefix}quadriflow remesh — {face_count:,} face target, smooth {'on' if smooth else 'off'}"
            else:
                voice = f"{prefix}retopologize {obj_name} for {platform} export"
                face_count = PLATFORM_SPECS[platform]["target_faces"]

            if zones and i % 4 == 0:
                code = (
                    self.preserve_topology_zones(obj_name, zones)
                    + "\n"
                    + self.quadriflow_remesh(obj_name, face_count, smooth)
                )
                voice = f"{prefix}retopo {obj_name} to {face_count:,} faces, preserve the {', '.join(zones)}"
            else:
                code = self.quadriflow_remesh(obj_name, face_count, smooth)

            pairs.append(
                {
                    "voice_command": voice,
                    "task_type": "RETOPO",
                    "blender_python": code,
                    "reasoning": random.choice(reasonings),
                    "quality": 2.5,
                    "source": "production_pipeline_synthetic",
                    "metadata": {
                        "target_faces": face_count,
                        "platform": platform,
                        "smooth": smooth,
                        "zones_preserved": zones,
                    },
                }
            )

        _save_pairs(pairs, "retopo_pairs.jsonl")
        return pairs


# ---------------------------------------------------------------------------
# UVAgent
# ---------------------------------------------------------------------------


class UVAgent:
    """
    UV unwrapping, seam suggestion, texel density analysis, UDIM layout, training pairs.
    """

    def smart_unwrap(
        self,
        obj_name: str,
        margin: float = 0.02,
        angle_limit: float = 66.0,
    ) -> str:
        """
        Generate Blender Python for Smart UV Project on a specific object.
        """
        return f"""import bpy

obj = bpy.data.objects.get('{obj_name}')
if obj is None:
    raise ValueError("Object '{obj_name}' not found in scene")

bpy.context.view_layer.objects.active = obj
bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)

# Ensure a UV map exists
if not obj.data.uv_layers:
    obj.data.uv_layers.new(name='UVMap')

# Smart UV Project — works on all faces, respects angle breaks
bpy.ops.object.editmode_toggle()
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project(
    angle_limit=math.radians({angle_limit}),
    margin_method='SCALED',
    island_margin={margin},
    area_weight=0.0,
    correct_aspect=True,
    scale_to_bounds=False,
)
bpy.ops.object.editmode_toggle()

print(f"Smart UV Project complete on '{obj_name}' "
      f"(angle_limit={angle_limit}°, margin={margin})")
"""

    def suggest_seams(self, obj_name: str) -> str:
        """
        Generate Blender Python to auto-select seam candidates
        using the UV Unwrap seam generation heuristic
        (marks edges with sharp angles or UV stretch as seams).
        """
        return f"""import bpy
import bmesh

obj = bpy.data.objects.get('{obj_name}')
if obj is None:
    raise ValueError("Object '{obj_name}' not found")

bpy.context.view_layer.objects.active = obj
obj.select_set(True)

bpy.ops.object.editmode_toggle()
bm = bmesh.from_edit_mesh(obj.data)
bm.edges.ensure_lookup_table()

# Heuristic: mark edges as seams if they are:
# 1. Sharp edges (angle > 60 degrees between adjacent faces)
# 2. Boundary edges (mesh border)
# 3. UV stretch edges (high distortion under angle-based unwrap)

SHARP_ANGLE_THRESHOLD = 1.047  # 60 degrees in radians

seam_candidates = []
for edge in bm.edges:
    if edge.is_boundary:
        edge.seam = True
        seam_candidates.append(edge.index)
        continue
    if len(edge.link_faces) == 2:
        angle = edge.calc_face_angle(fallback=0.0)
        if angle > SHARP_ANGLE_THRESHOLD:
            edge.seam = True
            seam_candidates.append(edge.index)

bmesh.update_edit_mesh(obj.data)

print(f"Marked {{len(seam_candidates)}} seam candidates on '{obj_name}'")
print("Suggested seams are now visible — adjust in UV editor before unwrapping")
bpy.ops.object.editmode_toggle()
"""

    def check_texel_density(
        self,
        obj_name: str,
        target_density: float = 10.24,
        texture_size: int = 1024,
    ) -> str:
        """
        Generate Blender Python to compute and report texel density per UV island.
        target_density in texels/cm (10.24 texels/cm = 1024px/100cm standard).
        """
        return f"""import bpy
import bmesh
import math

obj = bpy.data.objects.get('{obj_name}')
if obj is None:
    raise ValueError("Object '{obj_name}' not found")

mesh = obj.data
if not mesh.uv_layers:
    raise ValueError(f"Object '{obj_name}' has no UV maps")

bm = bmesh.new()
bm.from_mesh(mesh)
bm.faces.ensure_lookup_table()

uv_layer = bm.loops.layers.uv.active
texture_size = {texture_size}
target_density = {target_density}  # texels per cm

density_report = []
for face in bm.faces:
    # UV area (in 0-1 UV space)
    uvs = [loop[uv_layer].uv for loop in face.loops]
    uv_area = 0.0
    n = len(uvs)
    for i in range(n):
        j = (i + 1) % n
        uv_area += uvs[i].x * uvs[j].y - uvs[j].x * uvs[i].y
    uv_area = abs(uv_area) * 0.5

    # World-space area (in scene units, assuming 1 unit = 1m = 100cm)
    world_area = face.calc_area() * 10000  # convert m² to cm²

    if world_area > 0 and uv_area > 0:
        # texel density = sqrt(UV pixels² / world area cm²)
        uv_pixels_sq = uv_area * (texture_size ** 2)
        density = math.sqrt(uv_pixels_sq / world_area)
        deviation = abs(density - target_density) / target_density
        density_report.append({{
            "face_index": face.index,
            "density": round(density, 2),
            "target": target_density,
            "deviation_pct": round(deviation * 100, 1),
            "status": "ok" if deviation < 0.15 else "warn" if deviation < 0.40 else "error",
        }})

bm.free()

# Summary
ok = sum(1 for d in density_report if d["status"] == "ok")
warn = sum(1 for d in density_report if d["status"] == "warn")
error = sum(1 for d in density_report if d["status"] == "error")

print(f"Texel Density Report for '{obj_name}':")
print(f"  Target: {{target_density}} tx/cm  |  Texture: {{texture_size}}px")
print(f"  OK: {{ok}} faces  |  Warn (±15-40%): {{warn}}  |  Error (>40%): {{error}}")
if density_report:
    avg = sum(d["density"] for d in density_report) / len(density_report)
    print(f"  Average density: {{avg:.2f}} tx/cm")
"""

    def create_udim_layout(
        self,
        object_names: List[str],
        udim_count: int = 4,
    ) -> str:
        """
        Generate Blender Python to assign UV islands across UDIM tiles
        for a list of objects. Objects are distributed across tiles 1001-100N.
        """
        names_repr = repr(object_names)
        return f"""import bpy
import bmesh

object_names = {names_repr}
udim_count = {udim_count}

# Assign each object's UV islands to a specific UDIM tile
# UDIM tile 1001 = UV space (0-1, 0-1)
# UDIM tile 1002 = UV space (1-2, 0-1), etc.

for tile_idx, obj_name in enumerate(object_names[:udim_count]):
    obj = bpy.data.objects.get(obj_name)
    if obj is None:
        print(f"Warning: '{{obj_name}}' not found — skipping")
        continue

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    bpy.ops.object.editmode_toggle()
    bm = bmesh.from_edit_mesh(obj.data)
    uv_layer = bm.loops.layers.uv.active

    if uv_layer is None:
        print(f"Warning: '{{obj_name}}' has no UV map — skipping")
        bpy.ops.object.editmode_toggle()
        continue

    # Offset UV islands to the correct UDIM tile
    tile_u_offset = float(tile_idx % 10)   # tiles wrap at column 10
    tile_v_offset = float(tile_idx // 10)

    for face in bm.faces:
        for loop in face.loops:
            uv = loop[uv_layer].uv
            uv.x = (uv.x % 1.0) + tile_u_offset
            uv.y = (uv.y % 1.0) + tile_v_offset

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.editmode_toggle()

    tile_number = 1001 + tile_idx
    print(f"Assigned '{{obj_name}}' → UDIM tile {{tile_number}}")

print(f"UDIM layout complete: {{min(len(object_names), udim_count)}} objects across {{udim_count}} tiles")
"""

    def generate_training_pairs(self, n_pairs: int = 200) -> List[dict]:
        """Generate 200 UV unwrapping training pairs."""
        pairs: List[dict] = []

        margins = [0.005, 0.01, 0.02, 0.03, 0.05]
        angle_limits = [45.0, 60.0, 66.0, 75.0, 85.0]
        obj_names = [
            "character",
            "weapon_rifle",
            "helmet",
            "car_body",
            "building_facade",
            "tree_trunk",
            "rock",
            "crate",
        ]
        texture_sizes = [512, 1024, 2048, 4096]
        densities = [5.12, 10.24, 20.48, 40.96]

        reasonings = [
            "Smart UV Project is the fastest way to get a usable UV layout on arbitrary meshes. "
            "The angle_limit determines where UV islands break — lower angles = more islands = fewer stretches but more wasted UV space.",
            "Seam placement is critical for minimizing UV stretching. "
            "Sharp edges and boundaries are natural seam candidates because they're already visually distinct — the seam won't be noticeable.",
            "Consistent texel density across all objects in a scene ensures no single object looks blurry next to others. "
            "Target density depends on texture budget and viewing distance.",
            "UDIM tiles let production pipelines pack multiple objects into a single texture set with per-tile resolution. "
            "Game engines like Unreal support UDIM natively for hero characters.",
            "UV margin (island padding) prevents texture bleeding between islands during mip-mapping. "
            "0.02 (2%) is the minimum safe margin for 1024px textures; use 0.005 for 4096px.",
        ]

        prefixes = ["", "", "", "hey nalana, ", "can you ", "please "]

        for i in range(n_pairs):
            obj_name = random.choice(obj_names)
            margin = random.choice(margins)
            angle_limit = random.choice(angle_limits)
            texture_size = random.choice(texture_sizes)
            density = random.choice(densities)
            prefix = random.choice(prefixes)

            style = i % 6
            if style == 0:
                voice = f"{prefix}unwrap {obj_name} with smart UV project"
                code = self.smart_unwrap(obj_name, margin, angle_limit)
                task = "UV_UNWRAP"
            elif style == 1:
                voice = f"{prefix}unwrap {obj_name}, margin {margin}, angle limit {angle_limit} degrees"
                code = self.smart_unwrap(obj_name, margin, angle_limit)
                task = "UV_UNWRAP"
            elif style == 2:
                voice = f"{prefix}suggest seams for {obj_name}"
                code = self.suggest_seams(obj_name)
                task = "UV_UNWRAP"
            elif style == 3:
                voice = (
                    f"{prefix}mark seam candidates on {obj_name} based on hard edges"
                )
                code = self.suggest_seams(obj_name)
                task = "UV_UNWRAP"
            elif style == 4:
                voice = f"{prefix}check texel density on {obj_name} for {texture_size}px texture"
                code = self.check_texel_density(obj_name, density, texture_size)
                task = "UV_UNWRAP"
            else:
                udim_objects = random.sample(obj_names, 3)
                voice = f"{prefix}set up UDIM layout for {', '.join(udim_objects)}"
                code = self.create_udim_layout(udim_objects, len(udim_objects))
                task = "UV_UNWRAP"
                obj_name = udim_objects[0]

            pairs.append(
                {
                    "voice_command": voice,
                    "task_type": task,
                    "blender_python": code,
                    "reasoning": random.choice(reasonings),
                    "quality": 2.5,
                    "source": "production_pipeline_synthetic",
                }
            )

        _save_pairs(pairs, "uv_pairs.jsonl")
        return pairs


# ---------------------------------------------------------------------------
# LODAgent
# ---------------------------------------------------------------------------


class LODAgent:
    """
    LOD chain generation, silhouette-preserving decimation, imposter/billboard generation.
    """

    def generate_lod_chain(
        self,
        obj_name: str,
        levels: Optional[List[float]] = None,
    ) -> str:
        """
        Generate Blender Python for a full LOD chain (LOD0 - LOD3).
        Each level is a fraction of the original face count.

        Args:
            obj_name: Source object name.
            levels:   List of decimation ratios [LOD0, LOD1, LOD2, LOD3].
                      Default: [1.0, 0.5, 0.25, 0.1]
        """
        if levels is None:
            levels = [1.0, 0.5, 0.25, 0.1]
        levels_repr = repr(levels)
        return f"""import bpy

obj = bpy.data.objects.get('{obj_name}')
if obj is None:
    raise ValueError("Object '{obj_name}' not found")

levels = {levels_repr}
lod_objects = []

for lod_idx, ratio in enumerate(levels):
    lod_name = f'{obj_name}_LOD{{lod_idx}}'

    if ratio == 1.0:
        # LOD0 = original mesh, just duplicate and rename
        lod_obj = obj.copy()
        lod_obj.data = obj.data.copy()
        lod_obj.name = lod_name
        bpy.context.collection.objects.link(lod_obj)
        lod_obj.location.x = obj.location.x
    else:
        # Duplicate and apply decimation
        lod_obj = obj.copy()
        lod_obj.data = obj.data.copy()
        lod_obj.name = lod_name
        bpy.context.collection.objects.link(lod_obj)

        # Add Decimate modifier
        dec_mod = lod_obj.modifiers.new(name='Decimate_LOD', type='DECIMATE')
        dec_mod.decimate_type = 'COLLAPSE'
        dec_mod.ratio = ratio
        dec_mod.use_collapse_triangulate = True

        # Apply modifier
        bpy.context.view_layer.objects.active = lod_obj
        bpy.ops.object.modifier_apply(modifier=dec_mod.name)

    face_count = len(lod_obj.data.polygons)
    tri_count = sum(len(p.vertices) - 2 for p in lod_obj.data.polygons)
    print(f"LOD{{lod_idx}} ({ratio:.0%}): {{face_count}} faces / {{tri_count}} tris → '{{lod_name}}'")

    lod_objects.append(lod_obj.name)

print(f"LOD chain complete: {{lod_objects}}")
"""

    def preserve_silhouette_decimate(
        self,
        obj_name: str,
        ratio: float,
    ) -> str:
        """
        Generate Blender Python for weighted decimation that preserves
        silhouette edges using a vertex group to protect outline edges.
        """
        return f"""import bpy
import bmesh
import math

obj = bpy.data.objects.get('{obj_name}')
if obj is None:
    raise ValueError("Object '{obj_name}' not found")

bpy.context.view_layer.objects.active = obj
obj.select_set(True)

# --- Step 1: Identify silhouette/boundary vertices and mark with a vertex group ---
bm = bmesh.new()
bm.from_mesh(obj.data)
bm.edges.ensure_lookup_table()
bm.verts.ensure_lookup_table()

# Silhouette edges: high curvature (angle between adjacent faces > 45°)
SILHOUETTE_ANGLE = math.radians(45)
silhouette_vert_indices = set()

for edge in bm.edges:
    if len(edge.link_faces) == 2:
        angle = edge.calc_face_angle(fallback=0.0)
        if angle > SILHOUETTE_ANGLE:
            for vert in edge.verts:
                silhouette_vert_indices.add(vert.index)
    elif edge.is_boundary:
        for vert in edge.verts:
            silhouette_vert_indices.add(vert.index)

bm.to_mesh(obj.data)
bm.free()

# --- Step 2: Create vertex group with silhouette verts weighted at 1.0 ---
vg = obj.vertex_groups.get('silhouette_protect')
if vg is None:
    vg = obj.vertex_groups.new(name='silhouette_protect')

# Assign all verts weight 0 first
all_indices = list(range(len(obj.data.vertices)))
vg.add(all_indices, 0.0, 'REPLACE')

# Assign silhouette verts weight 1.0 (fully protected)
if silhouette_vert_indices:
    vg.add(list(silhouette_vert_indices), 1.0, 'REPLACE')

# --- Step 3: Apply Decimate with vertex group ---
dec_mod = obj.modifiers.new(name='SilhouetteDecimate', type='DECIMATE')
dec_mod.decimate_type = 'COLLAPSE'
dec_mod.ratio = {ratio}
dec_mod.vertex_group = 'silhouette_protect'
dec_mod.invert_vertex_group = False  # protected verts (weight=1) resist decimation

bpy.ops.object.modifier_apply(modifier=dec_mod.name)

face_count = len(obj.data.polygons)
print(f"Silhouette-preserving decimation complete on '{obj_name}'")
print(f"Result: {{face_count}} faces (ratio={ratio:.2f})")
print(f"Silhouette verts protected: {{len(silhouette_vert_indices)}}")
"""

    def generate_imposter(self, obj_name: str) -> str:
        """
        Generate Blender Python to create a camera-facing billboard (imposter)
        for use as LOD_max. Creates a plane with a baked texture.
        """
        return f"""import bpy
import math

obj = bpy.data.objects.get('{obj_name}')
if obj is None:
    raise ValueError("Object '{obj_name}' not found")

# --- Step 1: Create billboard plane at object bounds center ---
bbox = [obj.matrix_world @ v.co for v in obj.data.vertices]
min_x = min(v.x for v in bbox)
max_x = max(v.x for v in bbox)
min_y = min(v.y for v in bbox)
max_y = max(v.y for v in bbox)
min_z = min(v.z for v in bbox)
max_z = max(v.z for v in bbox)

width  = max(max_x - min_x, max_y - min_y)
height = max_z - min_z
center_x = (min_x + max_x) / 2
center_y = (min_y + max_y) / 2
center_z = (min_z + max_z) / 2

bpy.ops.mesh.primitive_plane_add(
    size=1,
    location=(center_x, center_y, center_z),
)
billboard = bpy.context.active_object
billboard.name = f'{obj_name}_imposter'
billboard.scale = (width, 1.0, height)
bpy.ops.object.transform_apply(scale=True)

# --- Step 2: Orient billboard to face camera (Z-up, Y-forward) ---
billboard.rotation_euler = (math.radians(90), 0, 0)

# --- Step 3: Add billboard material with PNG texture slot ---
mat = bpy.data.materials.new(name=f'{obj_name}_imposter_mat')
mat.use_nodes = True
mat.blend_method = 'CLIP'  # alpha clip for transparency

nodes = mat.node_tree.nodes
links = mat.node_tree.links
nodes.clear()

output = nodes.new('ShaderNodeOutputMaterial')
principled = nodes.new('ShaderNodeBsdfPrincipled')
tex = nodes.new('ShaderNodeTexImage')
tex.label = 'Imposter Texture'

# Note: texture path should be set after baking
tex.image = bpy.data.images.new(
    name=f'{obj_name}_imposter_tex',
    width=512,
    height=512,
    alpha=True,
)

links.new(tex.outputs['Color'], principled.inputs['Base Color'])
links.new(tex.outputs['Alpha'], principled.inputs['Alpha'])
links.new(principled.outputs['BSDF'], output.inputs['Surface'])

billboard.data.materials.append(mat)

print(f"Imposter billboard created: '{billboard.name}'")
print(f"Dimensions: {{width:.2f}}w x {{height:.2f}}h")
print("Bake the original object's diffuse onto '{obj_name}_imposter_tex' to complete setup")
"""

    def generate_training_pairs(self, n_pairs: int = 100) -> List[dict]:
        """Generate 100 LOD training pairs."""
        pairs: List[dict] = []

        obj_names = [
            "tree",
            "rock_cluster",
            "building",
            "vehicle",
            "character_npc",
            "prop_barrel",
            "foliage_bush",
            "lamp_post",
        ]
        platform_levels = {
            "mobile": [1.0, 0.5, 0.2],
            "game_pc": [1.0, 0.5, 0.25, 0.1],
            "arch_viz": [1.0, 0.5],
        }
        ratios = [0.5, 0.25, 0.1, 0.05]

        reasonings = [
            "LOD chains are essential for real-time performance. "
            "LOD0 is the full-res mesh used close-up; LOD3 at <0.1 ratio is used at maximum draw distance. "
            "Unreal Engine selects LODs by screen percentage automatically.",
            "Silhouette-preserving decimation is better than standard collapse decimation for organic shapes. "
            "It prevents the characteristic 'crunched' look where outline edges collapse and the silhouette degrades.",
            "Imposter billboards replace low-LOD geometry entirely at extreme distances. "
            "Trees are the classic use case — a flat plane with an alpha-clipped texture costs almost zero render time.",
            "LOD transitions should be tuned per asset. "
            "A large building can switch at larger screen percentages; a small prop should switch at smaller ones.",
        ]

        prefixes = ["", "", "can you ", "please ", "hey nalana, "]

        for i in range(n_pairs):
            obj_name = random.choice(obj_names)
            platform = random.choice(list(platform_levels.keys()))
            levels = platform_levels[platform]
            ratio = random.choice(ratios)
            prefix = random.choice(prefixes)

            style = i % 3
            if style == 0:
                voice = f"{prefix}generate a LOD chain for {obj_name} for {platform}"
                code = self.generate_lod_chain(obj_name, levels)
                task = "LOD"
            elif style == 1:
                voice = f"{prefix}decimate {obj_name} to {int(ratio * 100)}% while preserving the silhouette"
                code = self.preserve_silhouette_decimate(obj_name, ratio)
                task = "LOD"
            else:
                voice = f"{prefix}create an imposter billboard for {obj_name}"
                code = self.generate_imposter(obj_name)
                task = "LOD"

            pairs.append(
                {
                    "voice_command": voice,
                    "task_type": task,
                    "blender_python": code,
                    "reasoning": random.choice(reasonings),
                    "quality": 2.5,
                    "source": "production_pipeline_synthetic",
                }
            )

        _save_pairs(pairs, "lod_pairs.jsonl")
        return pairs


# ---------------------------------------------------------------------------
# BakeAgent
# ---------------------------------------------------------------------------


class BakeAgent:
    """
    Normal map baking setup, cage generation, batch baking.
    """

    BAKE_MAP_TYPES = ["normal", "ao", "curvature", "thickness", "color"]

    def setup_bake(
        self,
        high_poly_name: str,
        low_poly_name: str,
        map_type: str = "normal",
        texture_size: int = 2048,
        samples: int = 128,
        ray_distance: float = 0.1,
    ) -> str:
        """
        Generate Blender Python to configure and execute a bake pass
        from high-poly to low-poly mesh.
        """
        bake_type_map = {
            "normal": "NORMAL",
            "ao": "AO",
            "curvature": "ROUGHNESS",  # baked curvature via material
            "thickness": "TRANSMISSION",
            "color": "DIFFUSE",
        }
        blender_bake_type = bake_type_map.get(map_type, "NORMAL")

        return f"""import bpy

high_poly = bpy.data.objects.get('{high_poly_name}')
low_poly  = bpy.data.objects.get('{low_poly_name}')

if high_poly is None:
    raise ValueError("High-poly object '{high_poly_name}' not found")
if low_poly is None:
    raise ValueError("Low-poly object '{low_poly_name}' not found")

# --- Ensure low-poly has a UV map ---
if not low_poly.data.uv_layers:
    raise ValueError(f"Low-poly '{low_poly_name}' needs a UV map before baking")

# --- Create bake target image ---
img_name = f'{low_poly_name}_{map_type}_{texture_size}px'
existing = bpy.data.images.get(img_name)
if existing:
    bpy.data.images.remove(existing)

bake_image = bpy.data.images.new(
    name=img_name,
    width={texture_size},
    height={texture_size},
    alpha=False,
    float_buffer=True,  # 32-bit float for normal maps
)

# --- Set image node active on low-poly material ---
for mat_slot in low_poly.material_slots:
    mat = mat_slot.material
    if mat is None or not mat.use_nodes:
        continue
    # Add image texture node if not present
    img_node = mat.node_tree.nodes.get('BakeTarget')
    if img_node is None:
        img_node = mat.node_tree.nodes.new('ShaderNodeTexImage')
        img_node.name = 'BakeTarget'
        img_node.label = 'Bake Target'
    img_node.image = bake_image
    # Make it active (required for Cycles baking target selection)
    mat.node_tree.nodes.active = img_node

# --- Configure render for Cycles baking ---
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.samples = {samples}

# --- Select high-poly as source, low-poly as bake target ---
bpy.ops.object.select_all(action='DESELECT')
high_poly.select_set(True)
low_poly.select_set(True)
bpy.context.view_layer.objects.active = low_poly  # active = bake destination

# --- Run bake ---
bpy.ops.object.bake(
    type='{blender_bake_type}',
    use_selected_to_active=True,
    use_cage=False,
    cage_extrusion={ray_distance},
    normal_space='TANGENT',
    save_mode='INTERNAL',
)

# --- Save to disk ---
output_path = f'/tmp/{low_poly_name}_{map_type}.exr'
bake_image.filepath_raw = output_path
bake_image.file_format = 'OPEN_EXR'
bake_image.save()

print(f"Bake complete: {map_type} map → '{{output_path}}'")
print(f"Image: {{bake_image.name}}  |  Size: {texture_size}px  |  Samples: {samples}")
"""

    def generate_cage(
        self,
        low_poly_name: str,
        offset: float = 0.05,
    ) -> str:
        """
        Generate Blender Python to create a cage mesh for baking
        by expanding the low-poly mesh along normals.
        """
        return f"""import bpy

low_poly = bpy.data.objects.get('{low_poly_name}')
if low_poly is None:
    raise ValueError("Object '{low_poly_name}' not found")

# Duplicate low-poly to create cage
bpy.ops.object.select_all(action='DESELECT')
low_poly.select_set(True)
bpy.context.view_layer.objects.active = low_poly
bpy.ops.object.duplicate(linked=False)

cage = bpy.context.active_object
cage.name = f'{low_poly_name}_cage'

# Add Shrinkwrap modifier to push cage outward along normals
# Use Displace modifier with generated normals
disp_mod = cage.modifiers.new(name='CageExpand', type='DISPLACE')
disp_mod.strength = {offset}
disp_mod.direction = 'NORMAL'
disp_mod.mid_level = 0.0

# Texture for uniform displacement (white = full outward)
tex = bpy.data.textures.new(name='{low_poly_name}_cage_tex', type='NONE')
tex.type = 'BLEND'  # Constant white = uniform expansion
disp_mod.texture = tex
disp_mod.texture_coords = 'NORMAL'

bpy.context.view_layer.objects.active = cage
bpy.ops.object.modifier_apply(modifier=disp_mod.name)

print(f"Cage mesh created: '{{cage.name}}' (offset={offset})")
print("Use this cage in: Object > Bake > Cage Object")
"""

    def batch_bake(
        self,
        pairs: List[tuple],
        output_dir: str = "/tmp/bakes",
        texture_size: int = 2048,
    ) -> str:
        """
        Generate Blender Python to batch bake multiple high/low poly pairs.

        Args:
            pairs:       List of (high_poly_name, low_poly_name) tuples.
            output_dir:  Directory to save baked textures.
            texture_size: Output texture resolution.
        """
        pairs_repr = repr(pairs)
        return f"""import bpy
import os

pairs = {pairs_repr}
output_dir = '{output_dir}'
texture_size = {texture_size}
os.makedirs(output_dir, exist_ok=True)

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.samples = 64  # lower for batch speed

results = []
for high_name, low_name in pairs:
    high_poly = bpy.data.objects.get(high_name)
    low_poly  = bpy.data.objects.get(low_name)

    if high_poly is None or low_poly is None:
        print(f"SKIP: high='{{high_name}}' or low='{{low_name}}' not found")
        continue

    for map_type, bake_type in [('normal', 'NORMAL'), ('ao', 'AO'), ('color', 'DIFFUSE')]:
        img_name = f'{{low_name}}_{{map_type}}'
        bake_img = bpy.data.images.new(img_name, width=texture_size, height=texture_size, float_buffer=True)

        # Set active image on all low-poly materials
        for slot in low_poly.material_slots:
            mat = slot.material
            if mat and mat.use_nodes:
                node = mat.node_tree.nodes.new('ShaderNodeTexImage')
                node.image = bake_img
                node.name = 'BakeTarget'
                mat.node_tree.nodes.active = node

        bpy.ops.object.select_all(action='DESELECT')
        high_poly.select_set(True)
        low_poly.select_set(True)
        bpy.context.view_layer.objects.active = low_poly

        try:
            bpy.ops.object.bake(
                type=bake_type,
                use_selected_to_active=True,
                cage_extrusion=0.05,
                normal_space='TANGENT',
            )
            out_path = os.path.join(output_dir, f'{{img_name}}.exr')
            bake_img.filepath_raw = out_path
            bake_img.file_format = 'OPEN_EXR'
            bake_img.save()
            results.append(('ok', low_name, map_type, out_path))
            print(f"OK: {{low_name}} {{map_type}} → {{out_path}}")
        except Exception as e:
            results.append(('error', low_name, map_type, str(e)))
            print(f"ERROR: {{low_name}} {{map_type}}: {{e}}")

print(f"Batch bake complete: {{sum(1 for r in results if r[0]=='ok')}} / {{len(results)}} succeeded")
"""

    def generate_training_pairs(self, n_pairs: int = 100) -> List[dict]:
        """Generate 100 bake training pairs."""
        pairs: List[dict] = []

        obj_pairs = [
            ("character_highpoly", "character_lowpoly"),
            ("armor_sculpt", "armor_game"),
            ("rock_detail", "rock_lod0"),
            ("weapon_zbrush", "weapon_fbx"),
            ("face_sculpt", "face_realtime"),
        ]
        map_types = ["normal", "ao", "curvature", "thickness", "color"]
        texture_sizes = [1024, 2048, 4096]
        offsets = [0.02, 0.05, 0.1, 0.15]

        reasonings = [
            "Normal baking transfers surface detail from a high-poly sculpt to a low-poly game mesh. "
            "The normal map stores the high-poly surface direction per texel, so the shader can simulate the detail without the geometry cost.",
            "Cage meshes prevent ray-casting errors during baking. "
            "Without a cage, rays from the low-poly can miss the high-poly on concave areas, leaving black patches in the normal map.",
            "AO baking captures ambient occlusion — how much light reaches each surface point. "
            "Pre-baked AO is cheaper than real-time SSAO and bakes contact shadows that SSAO misses.",
            "Curvature maps encode convexity/concavity — used for edge highlights and dirt masks in Substance Painter. "
            "Bake with a thin ray distance to capture only surface-level curvature.",
            "Batch baking automates what would otherwise be 30+ manual bake operations for a full character. "
            "Always bake normal first, then AO, as the normal map is used to check bake quality.",
        ]

        prefixes = ["", "", "can you ", "please ", "hey nalana, "]

        for i in range(n_pairs):
            high, low = random.choice(obj_pairs)
            map_type = random.choice(map_types)
            tex_size = random.choice(texture_sizes)
            offset = random.choice(offsets)
            prefix = random.choice(prefixes)

            style = i % 4
            if style == 0:
                voice = f"{prefix}bake a {map_type} map from {high} to {low}"
                code = self.setup_bake(high, low, map_type, tex_size)
                task = "BAKE"
            elif style == 1:
                voice = (
                    f"{prefix}bake {map_type} at {tex_size}px from {high} onto {low}"
                )
                code = self.setup_bake(high, low, map_type, tex_size)
                task = "BAKE"
            elif style == 2:
                voice = f"{prefix}create a bake cage for {low} with {offset} offset"
                code = self.generate_cage(low, offset)
                task = "BAKE"
            else:
                sample_pairs = random.sample(obj_pairs, 2)
                voice = f"{prefix}batch bake all objects to {tex_size}px textures"
                code = self.batch_bake(sample_pairs, "/tmp/bakes", tex_size)
                task = "BAKE"

            pairs.append(
                {
                    "voice_command": voice,
                    "task_type": task,
                    "blender_python": code,
                    "reasoning": random.choice(reasonings),
                    "quality": 2.5,
                    "source": "production_pipeline_synthetic",
                }
            )

        _save_pairs(pairs, "bake_pairs.jsonl")
        return pairs


# ---------------------------------------------------------------------------
# CollisionAgent
# ---------------------------------------------------------------------------


class CollisionAgent:
    """
    Collision mesh generation: convex decomposition, box, capsule.
    """

    def convex_decomposition(
        self,
        obj_name: str,
        resolution: int = 64,
        max_hulls: int = 8,
    ) -> str:
        """
        Generate Blender Python for approximate convex decomposition
        using V-HACD parameters via Blender's built-in VHACD support.
        """
        return f"""import bpy

obj = bpy.data.objects.get('{obj_name}')
if obj is None:
    raise ValueError("Object '{obj_name}' not found")

bpy.context.view_layer.objects.active = obj
bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)

# Use Blender's V-HACD operator (requires Blender 3.6+ or Cell Fracture addon)
# V-HACD decomposes a mesh into approximate convex hulls for physics
try:
    bpy.ops.object.vhacd(
        resolution={resolution},
        max_num_vertices_per_ch=32,
        max_convex_hulls={max_hulls},
        min_volume_per_convex_hull=0.0001,
        pca=False,
        mode='VOXEL',
    )
    print(f"V-HACD decomposition complete on '{obj_name}'")
    print(f"Generated up to {max_hulls} convex hull meshes")
except AttributeError:
    # Fallback: manual convex hull per object via bmesh
    import bmesh
    print("V-HACD not available — using single convex hull fallback")

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    hull_result = bmesh.ops.convex_hull(bm, input=bm.verts, use_existing_faces=False)

    # Keep only hull faces
    geom_interior = hull_result.get('geom_interior', [])
    bmesh.ops.delete(bm, geom=geom_interior, context='FACES')

    hull_mesh = bpy.data.meshes.new(f'{obj_name}_col_convex')
    bm.to_mesh(hull_mesh)
    bm.free()

    hull_obj = bpy.data.objects.new(f'UCX_{obj_name}_01', hull_mesh)
    bpy.context.collection.objects.link(hull_obj)
    hull_obj.location = obj.location.copy()

    print(f"Convex hull collision created: 'UCX_{obj_name}_01'")
    print("Prefix 'UCX_' is the Unreal Engine collision naming convention")
"""

    def simple_box_collision(self, obj_name: str) -> str:
        """
        Generate Blender Python to create a simple box (AABB) collision
        around the object's bounding box.
        """
        return f"""import bpy
from mathutils import Vector

obj = bpy.data.objects.get('{obj_name}')
if obj is None:
    raise ValueError("Object '{obj_name}' not found")

# Get world-space bounding box
bbox_world = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
min_v = Vector((
    min(v.x for v in bbox_world),
    min(v.y for v in bbox_world),
    min(v.z for v in bbox_world),
))
max_v = Vector((
    max(v.x for v in bbox_world),
    max(v.y for v in bbox_world),
    max(v.z for v in bbox_world),
))

center = (min_v + max_v) / 2
size_x = max_v.x - min_v.x
size_y = max_v.y - min_v.y
size_z = max_v.z - min_v.z

# Create box collision mesh (UBX_ prefix = Unreal Box Collision)
bpy.ops.mesh.primitive_cube_add(size=1, location=center)
box_col = bpy.context.active_object
box_col.name = f'UBX_{obj_name}_01'
box_col.scale = (size_x, size_y, size_z)
bpy.ops.object.transform_apply(scale=True)

# Make wireframe display to distinguish from render mesh
box_col.display_type = 'WIRE'

print(f"Box collision created: '{{box_col.name}}'")
print(f"Dimensions: {{size_x:.3f}} x {{size_y:.3f}} x {{size_z:.3f}}")
print("Use 'UBX_' prefix for Unreal Engine box collision, 'UCP_' for capsule, 'UCX_' for convex")
"""

    def capsule_collision(self, obj_name: str) -> str:
        """
        Generate Blender Python to create a capsule collision mesh
        for character or cylindrical objects.
        """
        return f"""import bpy
import math
from mathutils import Vector

obj = bpy.data.objects.get('{obj_name}')
if obj is None:
    raise ValueError("Object '{obj_name}' not found")

bbox_world = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
min_z = min(v.z for v in bbox_world)
max_z = max(v.z for v in bbox_world)
min_x = min(v.x for v in bbox_world)
max_x = max(v.x for v in bbox_world)
min_y = min(v.y for v in bbox_world)
max_y = max(v.y for v in bbox_world)

height = max_z - min_z
radius = max(max_x - min_x, max_y - min_y) / 2
center_z = (min_z + max_z) / 2
center_x = (min_x + max_x) / 2
center_y = (min_y + max_y) / 2

# Build capsule from cylinder + two UV spheres (hemisphere caps)
bpy.ops.mesh.primitive_cylinder_add(
    radius=radius,
    depth=max(0.01, height - 2 * radius),
    location=(center_x, center_y, center_z),
    vertices=16,
)
cylinder = bpy.context.active_object

# Top hemisphere
bpy.ops.mesh.primitive_uv_sphere_add(
    radius=radius,
    location=(center_x, center_y, center_z + height / 2 - radius),
    segments=16,
    ring_count=8,
)
top_cap = bpy.context.active_object

# Bottom hemisphere
bpy.ops.mesh.primitive_uv_sphere_add(
    radius=radius,
    location=(center_x, center_y, center_z - height / 2 + radius),
    segments=16,
    ring_count=8,
)
bot_cap = bpy.context.active_object

# Join into single capsule mesh
bpy.ops.object.select_all(action='DESELECT')
cylinder.select_set(True)
top_cap.select_set(True)
bot_cap.select_set(True)
bpy.context.view_layer.objects.active = cylinder
bpy.ops.object.join()

capsule = bpy.context.active_object
capsule.name = f'UCP_{obj_name}_01'
capsule.display_type = 'WIRE'

print(f"Capsule collision created: '{{capsule.name}}'")
print(f"Radius: {{radius:.3f}}  |  Height: {{height:.3f}}")
"""

    def generate_training_pairs(self, n_pairs: int = 50) -> List[dict]:
        """Generate 50 collision mesh training pairs."""
        pairs: List[dict] = []

        obj_names = [
            "crate",
            "barrel",
            "car",
            "building_column",
            "character_npc",
            "rock",
            "tree_trunk",
            "machine_part",
        ]
        resolutions = [32, 64, 128]
        max_hulls_options = [4, 6, 8, 12]

        reasonings = [
            "Convex decomposition creates a set of convex hulls that approximate the mesh. "
            "Physics engines require convex shapes for performance — concave meshes need to be decomposed. "
            "V-HACD is the industry standard algorithm for this.",
            "Box collision is the cheapest possible collision shape. "
            "Use it for rectangular objects like crates and buildings — the physics overhead is nearly zero.",
            "Capsule collision is the standard for characters. "
            "It's cheap, has no corners to snag on geometry, and rolls smoothly over terrain edges.",
            "The 'UCX_', 'UBX_', and 'UCP_' prefixes are Unreal Engine naming conventions for collision meshes. "
            "Blender FBX exporter auto-links collision meshes with matching prefixes to their parent mesh.",
        ]

        prefixes = ["", "", "can you ", "please "]

        for i in range(n_pairs):
            obj_name = random.choice(obj_names)
            resolution = random.choice(resolutions)
            max_hulls = random.choice(max_hulls_options)
            prefix = random.choice(prefixes)

            style = i % 3
            if style == 0:
                voice = f"{prefix}generate convex collision for {obj_name}"
                code = self.convex_decomposition(obj_name, resolution, max_hulls)
                task = "COLLISION"
            elif style == 1:
                voice = f"{prefix}create a box collision mesh for {obj_name}"
                code = self.simple_box_collision(obj_name)
                task = "COLLISION"
            else:
                voice = f"{prefix}add capsule collision to {obj_name}"
                code = self.capsule_collision(obj_name)
                task = "COLLISION"

            pairs.append(
                {
                    "voice_command": voice,
                    "task_type": task,
                    "blender_python": code,
                    "reasoning": random.choice(reasonings),
                    "quality": 2.5,
                    "source": "production_pipeline_synthetic",
                }
            )

        _save_pairs(pairs, "collision_pairs.jsonl")
        return pairs


# ---------------------------------------------------------------------------
# ProductionPipeline
# ---------------------------------------------------------------------------


class ProductionPipeline:
    """
    Orchestrates the full production pipeline: retopo → UV → bake → LOD → collision → export.
    Also generates multi-turn training sequences.
    """

    def __init__(self):
        self.retopo = RetopologyAgent()
        self.uv = UVAgent()
        self.lod = LODAgent()
        self.bake = BakeAgent()
        self.collision = CollisionAgent()

    def full_pipeline(
        self,
        obj_name: str,
        target_platform: str = "game_pc",
    ) -> str:
        """
        Generate a complete production pipeline script for one object.
        Sequence: retopo → UV → LOD chain → normal bake → collision → FBX export.
        """
        spec = PLATFORM_SPECS.get(target_platform, PLATFORM_SPECS["game_pc"])
        target_faces = spec["target_faces"]
        lod_levels = spec["lod_levels"]
        tex_size = spec["texture_size"]
        collision_type = spec["collision"]

        retopo_code = self.retopo.quadriflow_remesh(obj_name, target_faces)
        uv_code = self.uv.smart_unwrap(obj_name, margin=0.02, angle_limit=66.0)
        lod_code = self.lod.generate_lod_chain(obj_name, lod_levels)

        high_poly_name = f"{obj_name}_highpoly"
        bake_code = self.bake.setup_bake(high_poly_name, obj_name, "normal", tex_size)

        if collision_type == "convex":
            col_code = self.collision.convex_decomposition(obj_name)
        elif collision_type == "simple_box":
            col_code = self.collision.simple_box_collision(obj_name)
        else:
            col_code = f"# No collision needed for {target_platform}"

        return f"""# ============================================================
# Nalana Production Pipeline — {target_platform.upper()}
# Object: {obj_name}
# Platform: {target_platform} | {spec["notes"]}
# ============================================================

import bpy
import sys

print("=" * 60)
print(f"Production Pipeline: '{obj_name}' → {target_platform}")
print("=" * 60)

# --- STEP 1: Retopology ---
print("\\n[1/5] Retopology — target {target_faces:,} faces...")
{retopo_code}

# --- STEP 2: UV Unwrap ---
print("\\n[2/5] UV Unwrapping...")
{uv_code}

# --- STEP 3: LOD Chain ---
print("\\n[3/5] LOD Chain Generation ({lod_levels})...")
{lod_code}

# --- STEP 4: Normal Map Bake ---
print("\\n[4/5] Normal Map Bake ({tex_size}px)...")
{bake_code}

# --- STEP 5: Collision Mesh ---
print("\\n[5/5] Collision Mesh...")
{col_code}

# --- Export ---
print("\\n[6/6] FBX Export...")
export_path = f"/tmp/{obj_name}_{target_platform}.fbx"
bpy.ops.export_scene.fbx(
    filepath=export_path,
    use_selection=False,
    global_scale=1.0,
    apply_unit_scale=True,
    bake_space_transform=False,
    object_types={{'MESH', 'ARMATURE', 'EMPTY'}},
    use_mesh_modifiers=True,
    mesh_smooth_type='FACE',
    use_tspace=True,
    embed_textures=False,
    path_mode='RELATIVE',
    use_metadata=True,
    axis_forward='-Z',
    axis_up='Y',
)

print(f"\\nPipeline complete! Exported → {{export_path}}")
print("=" * 60)
"""

    def generate_multi_turn_pipeline_sequence(
        self,
        n_sequences: int = 20,
    ) -> List[dict]:
        """
        Generate multi-turn conversation sequences showing a complete production pipeline.
        Each sequence is a full back-and-forth: user asks steps, Nalana executes each.
        """
        sequences: List[dict] = []

        obj_names = [
            "hero_character",
            "weapon_rifle",
            "vehicle_car",
            "environment_rock",
            "building_facade",
        ]
        platforms = list(PLATFORM_SPECS.keys())

        for i in range(n_sequences):
            obj_name = random.choice(obj_names)
            platform = random.choice(platforms)
            spec = PLATFORM_SPECS[platform]

            sequence = {
                "conversation_type": "multi_turn_pipeline",
                "object": obj_name,
                "platform": platform,
                "quality": 3.0,
                "source": "production_pipeline_synthetic",
                "turns": [
                    {
                        "role": "user",
                        "content": f"I need to get {obj_name} production-ready for {platform}. Where do we start?",
                    },
                    {
                        "role": "assistant",
                        "content": (
                            f"For {platform} ({spec['notes']}), the production pipeline for {obj_name} is:\n\n"
                            f"1. Retopology → {spec['target_faces']:,} faces (QuadriFlow)\n"
                            f"2. UV Unwrap → Smart project, 2% margin\n"
                            f"3. LOD Chain → {spec['lod_levels']} ratios\n"
                            f"4. Bake → Normal map at {spec['texture_size']}px\n"
                            f"5. Collision → {spec['collision']}\n"
                            f"6. Export → FBX\n\n"
                            f"Starting with retopology now."
                        ),
                        "blender_python": self.retopo.quadriflow_remesh(
                            obj_name, spec["target_faces"]
                        ),
                    },
                    {
                        "role": "user",
                        "content": "Good, retopo looks clean. Now unwrap it.",
                    },
                    {
                        "role": "assistant",
                        "content": f"Running Smart UV Project on {obj_name} with 2% island margin.",
                        "blender_python": self.uv.smart_unwrap(obj_name),
                    },
                    {
                        "role": "user",
                        "content": "UVs look good. Generate the LOD chain.",
                    },
                    {
                        "role": "assistant",
                        "content": f"Generating LOD0 through LOD{len(spec['lod_levels']) - 1} for {obj_name}.",
                        "blender_python": self.lod.generate_lod_chain(
                            obj_name, spec["lod_levels"]
                        ),
                    },
                    {
                        "role": "user",
                        "content": "Now bake the normal map from the high-poly.",
                    },
                    {
                        "role": "assistant",
                        "content": f"Baking {spec['texture_size']}px normal map from {obj_name}_highpoly onto {obj_name}.",
                        "blender_python": self.bake.setup_bake(
                            f"{obj_name}_highpoly",
                            obj_name,
                            "normal",
                            spec["texture_size"],
                        ),
                    },
                    {
                        "role": "user",
                        "content": "Generate the collision mesh.",
                    },
                    {
                        "role": "assistant",
                        "content": f"Creating {spec['collision']} collision for {obj_name}.",
                        "blender_python": (
                            self.collision.convex_decomposition(obj_name)
                            if spec["collision"] == "convex"
                            else self.collision.simple_box_collision(obj_name)
                        ),
                    },
                    {
                        "role": "user",
                        "content": "Export it as FBX.",
                    },
                    {
                        "role": "assistant",
                        "content": f"{obj_name} is fully production-ready. Exporting as FBX.",
                        "blender_python": (
                            f"import bpy\n"
                            f"bpy.ops.export_scene.fbx(\n"
                            f"    filepath='/tmp/{obj_name}_{platform}.fbx',\n"
                            f"    use_selection=False,\n"
                            f"    global_scale=1.0,\n"
                            f"    apply_unit_scale=True,\n"
                            f"    object_types={{'MESH', 'ARMATURE'}},\n"
                            f"    use_mesh_modifiers=True,\n"
                            f"    axis_forward='-Z',\n"
                            f"    axis_up='Y',\n"
                            f")"
                        ),
                    },
                ],
            }
            sequences.append(sequence)

        out_path = DATA_DIR / "pipeline_sequences.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for s in sequences:
                f.write(json.dumps(s) + "\n")
        print(f"Saved {len(sequences)} multi-turn pipeline sequences → {out_path}")

        return sequences


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Nalana Production Pipeline — retopo, UV, LOD, bake, collision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python production_agent.py --retopo hero_character --target-faces 5000
  python production_agent.py --uv weapon --margin 0.01
  python production_agent.py --full-pipeline building --platform game_pc
  python production_agent.py --generate-pairs
        """,
    )

    parser.add_argument(
        "--retopo",
        metavar="OBJ_NAME",
        help="Retopologize object (requires --target-faces)",
    )
    parser.add_argument(
        "--target-faces", type=int, default=5000, help="Retopo target face count"
    )
    parser.add_argument("--uv", metavar="OBJ_NAME", help="Smart UV unwrap object")
    parser.add_argument("--margin", type=float, default=0.02, help="UV island margin")
    parser.add_argument(
        "--full-pipeline", metavar="OBJ_NAME", help="Run full production pipeline"
    )
    parser.add_argument(
        "--platform",
        default="game_pc",
        choices=list(PLATFORM_SPECS.keys()),
        help="Target platform",
    )
    parser.add_argument(
        "--generate-pairs", action="store_true", help="Generate all training pairs"
    )
    parser.add_argument(
        "--output", help="Output script path (prints to stdout if not set)"
    )

    args = parser.parse_args()

    pipeline = ProductionPipeline()

    if args.generate_pairs:
        print("Generating all production training pairs...")
        retopo_pairs = pipeline.retopo.generate_training_pairs(200)
        uv_pairs = pipeline.uv.generate_training_pairs(200)
        lod_pairs = pipeline.lod.generate_training_pairs(100)
        bake_pairs = pipeline.bake.generate_training_pairs(100)
        col_pairs = pipeline.collision.generate_training_pairs(50)
        sequences = pipeline.generate_multi_turn_pipeline_sequence(20)

        total = (
            len(retopo_pairs)
            + len(uv_pairs)
            + len(lod_pairs)
            + len(bake_pairs)
            + len(col_pairs)
        )
        print(f"\nTotal pairs generated: {total}")
        print(f"  RETOPO: {len(retopo_pairs)}")
        print(f"  UV:     {len(uv_pairs)}")
        print(f"  LOD:    {len(lod_pairs)}")
        print(f"  BAKE:   {len(bake_pairs)}")
        print(f"  COLLSN: {len(col_pairs)}")
        print(f"  MULTI:  {len(sequences)} sequences")
        print(f"\nAll saved to: {DATA_DIR}/")
        return

    code = None

    if args.retopo:
        code = pipeline.retopo.quadriflow_remesh(args.retopo, args.target_faces)

    elif args.uv:
        code = pipeline.uv.smart_unwrap(args.uv, margin=args.margin)

    elif args.full_pipeline:
        code = pipeline.full_pipeline(args.full_pipeline, args.platform)

    else:
        parser.print_help()
        return

    if code:
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(code)
            print(f"Script saved to: {args.output}")
        else:
            print(code)


if __name__ == "__main__":
    main()
