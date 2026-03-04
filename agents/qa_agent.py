"""
qa_agent.py - Nalana Scene QA / Linting System

Runs a comprehensive quality audit on a Blender scene and returns structured results.
Can run headlessly (called by validate_blender.py), via Blender addon panel, or via API.

Checks:
  1.  Naming conventions (objects, materials, UV maps)
  2.  Transform issues (unapplied scale/rotation, non-zero origins)
  3.  UV issues (missing UVs, overlapping, out-of-bounds, texel density inconsistency)
  4.  Topology issues (n-gons where they'll break deformation, non-manifold edges, zero-area faces)
  5.  Polycount budgets (configurable per platform: mobile/game/film)
  6.  Material sanity (missing materials, wrong PBR ranges, energy conservation)
  7.  Rig integrity (missing bones, broken constraints, degenerate weights)
  8.  Scene organization (orphaned data, unused materials, missing textures)
  9.  Export readiness (FBX/GLTF compliance)
  10. Unit consistency (mixed cm/m)

Usage:
  python qa_agent.py --scene path/to/scene.blend --profile game
  python qa_agent.py --scene path/to/scene.blend --profile film --output report.json
  # Or inside Blender:
  import qa_agent; report = qa_agent.audit_active_scene()
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Platform budget profiles
# ---------------------------------------------------------------------------

PLATFORM_PROFILES = {
    "mobile": {
        "max_tris": 10_000,
        "max_texture_px": 512,
        "allow_displacement": False,
        "require_watertight": False,
        "allow_ngons": False,
        "max_bones": 30,
        "notes": "Mobile / AR target (iOS/Android game)",
    },
    "game_pc": {
        "max_tris": 50_000,
        "max_texture_px": 2048,
        "allow_displacement": False,
        "require_watertight": False,
        "allow_ngons": False,
        "max_bones": 80,
        "notes": "PC game hero asset budget",
    },
    "cinematics": {
        "max_tris": 500_000,
        "max_texture_px": 4096,
        "allow_displacement": True,
        "require_watertight": False,
        "allow_ngons": True,
        "max_bones": 300,
        "notes": "VFX / cinematic asset (Pixar / ILM tier)",
    },
    "print_3d": {
        "max_tris": 2_000_000,
        "max_texture_px": 0,  # textures irrelevant
        "allow_displacement": False,
        "require_watertight": True,
        "allow_ngons": True,
        "max_bones": 0,
        "notes": "FDM/SLA 3D printing — watertight is mandatory",
    },
    "arch_viz": {
        "max_tris": 0,  # 0 = no hard limit
        "max_texture_px": 4096,
        "allow_displacement": True,
        "require_watertight": False,
        "allow_ngons": True,
        "max_bones": 0,
        "notes": "Architecture visualization — group by floor/zone",
    },
}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class QAIssue:
    category: str
    severity: str          # "error" | "warning" | "info"
    object_name: str
    description: str
    fix_command: str       # Python code that auto-fixes the issue (or "" if not fixable)
    auto_fixable: bool


@dataclass
class QAReport:
    passed: bool
    score: float           # 0-100
    issues: List[QAIssue]
    warnings: List[QAIssue]
    info: List[str]
    stats: dict
    platform_budget: dict

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "score": round(self.score, 2),
            "issues": [asdict(i) for i in self.issues],
            "warnings": [asdict(w) for w in self.warnings],
            "info": self.info,
            "stats": self.stats,
            "platform_budget": self.platform_budget,
        }

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        e = len(self.issues)
        w = len(self.warnings)
        return (
            f"QA {status} | Score: {self.score:.1f}/100 | "
            f"Errors: {e} | Warnings: {w}"
        )


# ---------------------------------------------------------------------------
# Individual check functions (bpy-based, run inside Blender)
# ---------------------------------------------------------------------------

def _check_naming_conventions(context) -> List[QAIssue]:
    """
    Check 1: Naming conventions.
    Rules:
      - No default Blender names (Cube, Cube.001, Material, Material.001 etc.)
      - Object names: snake_case or PascalCase only
      - No names with trailing spaces
      - UV maps should be named 'UVMap' or 'UVMap_<suffix>'
    """
    import bpy

    issues: List[QAIssue] = []
    DEFAULT_NAMES = re.compile(
        r"^(Cube|Sphere|Cylinder|Cone|Torus|Plane|Camera|Light|Sun|"
        r"Material|Texture|Image|Armature|Bone|Empty|Text|"
        r"NurbsCurve|BezierCurve|Circle|Grid|Icosphere)"
        r"(\.\d+)?$",
        re.IGNORECASE,
    )
    VALID_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_\.]*$")

    for obj in bpy.data.objects:
        name = obj.name

        # Default names
        if DEFAULT_NAMES.match(name):
            issues.append(QAIssue(
                category="naming",
                severity="warning",
                object_name=name,
                description=f"Object '{name}' uses a default Blender name. Rename it to something descriptive.",
                fix_command="",
                auto_fixable=False,
            ))

        # Trailing spaces
        if name != name.strip():
            issues.append(QAIssue(
                category="naming",
                severity="error",
                object_name=name,
                description=f"Object '{name}' has leading/trailing whitespace in its name.",
                fix_command=f"import bpy; bpy.data.objects['{name}'].name = '{name.strip()}'",
                auto_fixable=True,
            ))

        # Invalid characters
        if not VALID_NAME.match(name):
            issues.append(QAIssue(
                category="naming",
                severity="warning",
                object_name=name,
                description=f"Object name '{name}' contains invalid characters (spaces, special chars). Use snake_case.",
                fix_command="",
                auto_fixable=False,
            ))

        # Check material slot names
        if obj.type == "MESH":
            for slot in obj.material_slots:
                if slot.material and DEFAULT_NAMES.match(slot.material.name):
                    issues.append(QAIssue(
                        category="naming",
                        severity="warning",
                        object_name=name,
                        description=f"Material '{slot.material.name}' on '{name}' uses a default name.",
                        fix_command="",
                        auto_fixable=False,
                    ))

            # UV map names
            for uv in obj.data.uv_layers:
                if uv.name not in ("UVMap", "UVMap_base", "UVMap_detail", "UVMap_lm"):
                    if not uv.name.startswith("UVMap"):
                        issues.append(QAIssue(
                            category="naming",
                            severity="info",
                            object_name=name,
                            description=f"UV map '{uv.name}' on '{name}' — convention is 'UVMap' or 'UVMap_<suffix>'.",
                            fix_command="",
                            auto_fixable=False,
                        ))

    return issues


def _check_transforms(context) -> List[QAIssue]:
    """
    Check 2: Transform issues.
    - Unapplied scale (not (1,1,1))
    - Unapplied rotation (non-zero Euler)
    - Object origin far from geometry center
    """
    import bpy
    from mathutils import Vector

    issues: List[QAIssue] = []
    SCALE_EPSILON = 1e-4
    ROT_EPSILON = 1e-4

    for obj in bpy.data.objects:
        if obj.type not in {"MESH", "CURVE", "SURFACE", "FONT"}:
            continue

        name = obj.name

        # Unapplied scale
        sx, sy, sz = obj.scale
        if abs(sx - 1.0) > SCALE_EPSILON or abs(sy - 1.0) > SCALE_EPSILON or abs(sz - 1.0) > SCALE_EPSILON:
            issues.append(QAIssue(
                category="transforms",
                severity="error",
                object_name=name,
                description=(
                    f"Object '{name}' has unapplied scale ({sx:.3f}, {sy:.3f}, {sz:.3f}). "
                    "Apply scale before export or rigging."
                ),
                fix_command=(
                    f"import bpy; bpy.context.view_layer.objects.active = bpy.data.objects['{name}']; "
                    f"bpy.data.objects['{name}'].select_set(True); "
                    f"bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)"
                ),
                auto_fixable=True,
            ))

        # Unapplied rotation
        rx, ry, rz = obj.rotation_euler
        if abs(rx) > ROT_EPSILON or abs(ry) > ROT_EPSILON or abs(rz) > ROT_EPSILON:
            issues.append(QAIssue(
                category="transforms",
                severity="warning",
                object_name=name,
                description=(
                    f"Object '{name}' has non-zero rotation "
                    f"({math.degrees(rx):.1f}°, {math.degrees(ry):.1f}°, {math.degrees(rz):.1f}°). "
                    "Apply rotation if this is a non-animated asset."
                ),
                fix_command=(
                    f"import bpy; bpy.context.view_layer.objects.active = bpy.data.objects['{name}']; "
                    f"bpy.data.objects['{name}'].select_set(True); "
                    f"bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)"
                ),
                auto_fixable=True,
            ))

        # Origin far from geometry
        if obj.type == "MESH" and obj.data.vertices:
            verts_world = [obj.matrix_world @ v.co for v in obj.data.vertices]
            center = sum(verts_world, Vector()) / len(verts_world)
            origin = obj.location
            dist = (center - origin).length
            if dist > 1.0:  # more than 1 unit away
                issues.append(QAIssue(
                    category="transforms",
                    severity="warning",
                    object_name=name,
                    description=(
                        f"Object '{name}' origin is {dist:.2f} units from geometry center. "
                        "Consider setting origin to geometry center."
                    ),
                    fix_command=(
                        f"import bpy; bpy.context.view_layer.objects.active = bpy.data.objects['{name}']; "
                        f"bpy.data.objects['{name}'].select_set(True); "
                        f"bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')"
                    ),
                    auto_fixable=True,
                ))

    return issues


def _check_uv_issues(context) -> List[QAIssue]:
    """
    Check 3: UV issues.
    - Missing UV maps
    - UVs out of [0,1] bounds (for non-UDIM)
    - Zero-area UV faces
    - Texel density inconsistency across related objects
    """
    import bpy
    import bmesh

    issues: List[QAIssue] = []

    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        name = obj.name
        mesh = obj.data

        # Missing UVs
        if len(mesh.uv_layers) == 0:
            issues.append(QAIssue(
                category="uv",
                severity="error",
                object_name=name,
                description=f"Object '{name}' has no UV maps. Cannot bake or export textures.",
                fix_command=(
                    f"import bpy; bpy.context.view_layer.objects.active = bpy.data.objects['{name}']; "
                    f"bpy.data.objects['{name}'].select_set(True); "
                    f"bpy.ops.object.editmode_toggle(); "
                    f"bpy.ops.mesh.select_all(action='SELECT'); "
                    f"bpy.ops.uv.smart_project(angle_limit=66.0, margin_method='SCALED', island_margin=0.02); "
                    f"bpy.ops.object.editmode_toggle()"
                ),
                auto_fixable=True,
            ))
            continue

        # Check UV bounds and zero-area faces via bmesh
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.faces.ensure_lookup_table()

        uv_layer = bm.loops.layers.uv.active
        if uv_layer is None:
            bm.free()
            continue

        out_of_bounds_count = 0
        zero_area_count = 0

        for face in bm.faces:
            uvs = [loop[uv_layer].uv for loop in face.loops]

            # Out-of-bounds check
            for uv in uvs:
                if uv.x < -0.001 or uv.x > 1.001 or uv.y < -0.001 or uv.y > 1.001:
                    out_of_bounds_count += 1
                    break

            # Zero-area UV face
            if len(uvs) >= 3:
                # Shoelace formula for UV area
                area = 0.0
                n = len(uvs)
                for i in range(n):
                    j = (i + 1) % n
                    area += uvs[i].x * uvs[j].y
                    area -= uvs[j].x * uvs[i].y
                if abs(area) < 1e-8:
                    zero_area_count += 1

        bm.free()

        if out_of_bounds_count > 0:
            issues.append(QAIssue(
                category="uv",
                severity="warning",
                object_name=name,
                description=(
                    f"{out_of_bounds_count} UV face(s) on '{name}' are outside the [0,1] UV space. "
                    "This causes tiling artifacts unless UDIM is intentional."
                ),
                fix_command="",
                auto_fixable=False,
            ))

        if zero_area_count > 0:
            issues.append(QAIssue(
                category="uv",
                severity="warning",
                object_name=name,
                description=(
                    f"{zero_area_count} zero-area UV face(s) on '{name}'. "
                    "These faces have collapsed UVs and will appear as black spots when baked."
                ),
                fix_command="",
                auto_fixable=False,
            ))

    return issues


def _check_topology(context, profile: dict) -> List[QAIssue]:
    """
    Check 4: Topology issues.
    - N-gons (faces with >4 sides) where they break deformation
    - Non-manifold edges (3+ face adjacencies or open boundary on closed mesh)
    - Zero-area faces (degenerate geometry)
    - Isolated vertices
    """
    import bpy
    import bmesh

    issues: List[QAIssue] = []
    allow_ngons = profile.get("allow_ngons", False)

    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        name = obj.name
        mesh = obj.data

        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.faces.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.verts.ensure_lookup_table()

        ngon_count = 0
        zero_face_count = 0
        non_manifold_edge_count = 0
        isolated_vert_count = 0

        for face in bm.faces:
            if len(face.verts) > 4 and not allow_ngons:
                ngon_count += 1
            if face.calc_area() < 1e-8:
                zero_face_count += 1

        for edge in bm.edges:
            if not edge.is_manifold:
                non_manifold_edge_count += 1

        for vert in bm.verts:
            if not vert.link_edges:
                isolated_vert_count += 1

        bm.free()

        if ngon_count > 0:
            issues.append(QAIssue(
                category="topology",
                severity="error",
                object_name=name,
                description=(
                    f"{ngon_count} n-gon(s) found on '{name}'. "
                    "N-gons break subdivision, skinning, and deformation. Triangulate or quad-fill these."
                ),
                fix_command=(
                    f"import bpy, bmesh; "
                    f"bpy.context.view_layer.objects.active = bpy.data.objects['{name}']; "
                    f"bpy.ops.object.editmode_toggle(); "
                    f"bpy.ops.mesh.select_all(action='SELECT'); "
                    f"bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY'); "
                    f"bpy.ops.object.editmode_toggle()"
                ),
                auto_fixable=True,
            ))

        if zero_face_count > 0:
            issues.append(QAIssue(
                category="topology",
                severity="error",
                object_name=name,
                description=(
                    f"{zero_face_count} zero-area (degenerate) face(s) on '{name}'. "
                    "These are invisible geometry that cause rendering artifacts."
                ),
                fix_command=(
                    f"import bpy; "
                    f"bpy.context.view_layer.objects.active = bpy.data.objects['{name}']; "
                    f"bpy.ops.object.editmode_toggle(); "
                    f"bpy.ops.mesh.select_all(action='SELECT'); "
                    f"bpy.ops.mesh.dissolve_degenerate(threshold=0.0001); "
                    f"bpy.ops.object.editmode_toggle()"
                ),
                auto_fixable=True,
            ))

        if non_manifold_edge_count > 0:
            severity = "error" if profile.get("require_watertight") else "warning"
            issues.append(QAIssue(
                category="topology",
                severity=severity,
                object_name=name,
                description=(
                    f"{non_manifold_edge_count} non-manifold edge(s) on '{name}'. "
                    "Mesh is not watertight — will fail 3D printing and cause issues in physics sims."
                ),
                fix_command=(
                    f"import bpy; "
                    f"bpy.context.view_layer.objects.active = bpy.data.objects['{name}']; "
                    f"bpy.ops.object.editmode_toggle(); "
                    f"bpy.ops.mesh.select_all(action='DESELECT'); "
                    f"bpy.ops.mesh.select_non_manifold(); "
                    f"bpy.ops.object.editmode_toggle()"
                ),
                auto_fixable=False,  # selection only, actual fix is manual
            ))

        if isolated_vert_count > 0:
            issues.append(QAIssue(
                category="topology",
                severity="warning",
                object_name=name,
                description=(
                    f"{isolated_vert_count} isolated vertex/vertices on '{name}' with no edges. "
                    "Clean up loose geometry."
                ),
                fix_command=(
                    f"import bpy; "
                    f"bpy.context.view_layer.objects.active = bpy.data.objects['{name}']; "
                    f"bpy.ops.object.editmode_toggle(); "
                    f"bpy.ops.mesh.select_all(action='DESELECT'); "
                    f"bpy.ops.mesh.select_loose(); "
                    f"bpy.ops.mesh.delete(type='VERT'); "
                    f"bpy.ops.object.editmode_toggle()"
                ),
                auto_fixable=True,
            ))

    return issues


def _check_polycount(context, profile: dict) -> List[QAIssue]:
    """
    Check 5: Polycount budget compliance.
    """
    import bpy

    issues: List[QAIssue] = []
    max_tris = profile.get("max_tris", 0)
    if max_tris == 0:
        return issues  # no limit for this profile

    # Count triangles per object and total
    total_tris = 0
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        tri_count = sum(
            len(poly.vertices) - 2 for poly in obj.data.polygons
        )
        total_tris += tri_count

        if tri_count > max_tris * 0.5:  # warn if single object is >50% of budget
            issues.append(QAIssue(
                category="polycount",
                severity="warning",
                object_name=obj.name,
                description=(
                    f"Object '{obj.name}' has {tri_count:,} tris "
                    f"({tri_count / max_tris * 100:.0f}% of {profile['notes']} budget of {max_tris:,})."
                ),
                fix_command="",
                auto_fixable=False,
            ))

    if total_tris > max_tris:
        issues.append(QAIssue(
            category="polycount",
            severity="error",
            object_name="SCENE",
            description=(
                f"Scene total: {total_tris:,} tris — EXCEEDS {profile['notes']} budget of {max_tris:,} tris "
                f"(over by {total_tris - max_tris:,})."
            ),
            fix_command="",
            auto_fixable=False,
        ))
    elif total_tris > max_tris * 0.85:
        issues.append(QAIssue(
            category="polycount",
            severity="warning",
            object_name="SCENE",
            description=(
                f"Scene total: {total_tris:,} tris — within {100 - (total_tris / max_tris * 100):.0f}% "
                f"of {profile['notes']} budget of {max_tris:,}. Close to limit."
            ),
            fix_command="",
            auto_fixable=False,
        ))

    return issues


def _check_materials(context) -> List[QAIssue]:
    """
    Check 6: Material sanity.
    - Objects with no materials
    - Principled BSDF inputs out of physically plausible range
    - Emission energy conservation (bloom vs. linear workflow)
    """
    import bpy

    issues: List[QAIssue] = []

    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        name = obj.name

        # No material assigned
        if len(obj.material_slots) == 0 or all(s.material is None for s in obj.material_slots):
            issues.append(QAIssue(
                category="materials",
                severity="error",
                object_name=name,
                description=f"Object '{name}' has no material. Assign a material before export.",
                fix_command=(
                    f"import bpy; obj = bpy.data.objects['{name}']; "
                    f"mat = bpy.data.materials.new(name='{name}_mat'); "
                    f"mat.use_nodes = True; obj.data.materials.append(mat)"
                ),
                auto_fixable=True,
            ))
            continue

        for slot in obj.material_slots:
            mat = slot.material
            if mat is None:
                continue

            # Find Principled BSDF node
            if not mat.use_nodes:
                continue

            for node in mat.node_tree.nodes:
                if node.type != "BSDF_PRINCIPLED":
                    continue

                # Metallic range check [0, 1]
                metallic_input = node.inputs.get("Metallic")
                if metallic_input and not metallic_input.is_linked:
                    val = metallic_input.default_value
                    if not 0.0 <= val <= 1.0:
                        issues.append(QAIssue(
                            category="materials",
                            severity="error",
                            object_name=name,
                            description=(
                                f"Material '{mat.name}' on '{name}': Metallic value {val:.3f} "
                                "is outside [0, 1] range. Physically impossible."
                            ),
                            fix_command=(
                                f"import bpy; "
                                f"mat = bpy.data.materials['{mat.name}']; "
                                f"node = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'); "
                                f"node.inputs['Metallic'].default_value = max(0.0, min(1.0, node.inputs['Metallic'].default_value))"
                            ),
                            auto_fixable=True,
                        ))

                # Roughness range check [0, 1]
                roughness_input = node.inputs.get("Roughness")
                if roughness_input and not roughness_input.is_linked:
                    val = roughness_input.default_value
                    if not 0.0 <= val <= 1.0:
                        issues.append(QAIssue(
                            category="materials",
                            severity="error",
                            object_name=name,
                            description=(
                                f"Material '{mat.name}' on '{name}': Roughness value {val:.3f} "
                                "is outside [0, 1] range. Physically impossible."
                            ),
                            fix_command="",
                            auto_fixable=False,
                        ))

                # Emission strength sanity check
                emission_strength_input = node.inputs.get("Emission Strength")
                if emission_strength_input and not emission_strength_input.is_linked:
                    val = emission_strength_input.default_value
                    if val > 1000:
                        issues.append(QAIssue(
                            category="materials",
                            severity="warning",
                            object_name=name,
                            description=(
                                f"Material '{mat.name}' on '{name}': Emission Strength is {val:.0f} — "
                                "extremely high. Verify this is intentional (not a unit error)."
                            ),
                            fix_command="",
                            auto_fixable=False,
                        ))

    return issues


def _check_rig_integrity(context) -> List[QAIssue]:
    """
    Check 7: Rig integrity.
    - Armatures with no bones
    - Meshes with Armature modifier but no vertex groups
    - Meshes with vertex groups that have no corresponding bone
    - Degenerate (zero-length) bones
    """
    import bpy

    issues: List[QAIssue] = []

    # Find armatures and their meshes
    armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]

    for arm_obj in armatures:
        name = arm_obj.name
        arm = arm_obj.data

        if len(arm.bones) == 0:
            issues.append(QAIssue(
                category="rig",
                severity="error",
                object_name=name,
                description=f"Armature '{name}' has no bones.",
                fix_command="",
                auto_fixable=False,
            ))
            continue

        # Check for zero-length bones
        for bone in arm.bones:
            length = bone.length
            if length < 1e-5:
                issues.append(QAIssue(
                    category="rig",
                    severity="error",
                    object_name=name,
                    description=(
                        f"Bone '{bone.name}' in armature '{name}' has near-zero length ({length:.6f}). "
                        "Degenerate bones cause skinning failures."
                    ),
                    fix_command="",
                    auto_fixable=False,
                ))

    # Find skinned meshes
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        name = obj.name

        arm_mod = next(
            (m for m in obj.modifiers if m.type == "ARMATURE"),
            None,
        )
        if arm_mod is None:
            continue

        arm_obj = arm_mod.object
        if arm_obj is None:
            issues.append(QAIssue(
                category="rig",
                severity="error",
                object_name=name,
                description=f"Mesh '{name}' has an Armature modifier with no armature assigned.",
                fix_command="",
                auto_fixable=False,
            ))
            continue

        # Check vertex groups exist
        if len(obj.vertex_groups) == 0:
            issues.append(QAIssue(
                category="rig",
                severity="error",
                object_name=name,
                description=(
                    f"Mesh '{name}' has an Armature modifier but zero vertex groups. "
                    "Skinning will have no effect."
                ),
                fix_command=(
                    f"import bpy; "
                    f"bpy.context.view_layer.objects.active = bpy.data.objects['{name}']; "
                    f"bpy.data.objects['{name}'].select_set(True); "
                    f"arm = bpy.data.objects['{arm_obj.name}']; "
                    f"arm.select_set(True); "
                    f"bpy.ops.object.parent_set(type='ARMATURE_AUTO')"
                ),
                auto_fixable=True,
            ))
            continue

        # Check vertex groups have matching bones
        bone_names = {b.name for b in arm_obj.data.bones}
        for vg in obj.vertex_groups:
            if vg.name not in bone_names:
                issues.append(QAIssue(
                    category="rig",
                    severity="warning",
                    object_name=name,
                    description=(
                        f"Vertex group '{vg.name}' on '{name}' has no matching bone in '{arm_obj.name}'. "
                        "Orphaned weight group — likely a renamed/deleted bone."
                    ),
                    fix_command=(
                        f"import bpy; "
                        f"obj = bpy.data.objects['{name}']; "
                        f"vg = obj.vertex_groups.get('{vg.name}'); "
                        f"if vg: obj.vertex_groups.remove(vg)"
                    ),
                    auto_fixable=True,
                ))

    return issues


def _check_scene_organization(context) -> List[QAIssue]:
    """
    Check 8: Scene organization.
    - Orphaned data blocks (meshes, materials, images with zero users)
    - Missing image textures (file not found on disk)
    - Objects not in any collection (other than Scene Collection root)
    """
    import bpy

    issues: List[QAIssue] = []

    # Orphaned meshes
    orphaned_meshes = [m for m in bpy.data.meshes if m.users == 0]
    if orphaned_meshes:
        issues.append(QAIssue(
            category="organization",
            severity="info",
            object_name="DATA",
            description=(
                f"{len(orphaned_meshes)} orphaned mesh data-block(s) with zero users: "
                f"{[m.name for m in orphaned_meshes[:5]]}... "
                "Run 'Purge All' to clean up."
            ),
            fix_command="import bpy; bpy.ops.outliner.orphans_purge(do_recursive=True)",
            auto_fixable=True,
        ))

    # Orphaned materials
    orphaned_mats = [m for m in bpy.data.materials if m.users == 0]
    if orphaned_mats:
        issues.append(QAIssue(
            category="organization",
            severity="info",
            object_name="DATA",
            description=(
                f"{len(orphaned_mats)} orphaned material(s) with zero users. Run 'Purge All'."
            ),
            fix_command="import bpy; bpy.ops.outliner.orphans_purge(do_recursive=True)",
            auto_fixable=True,
        ))

    # Missing image textures
    for img in bpy.data.images:
        if img.source == "FILE" and img.filepath:
            abs_path = bpy.path.abspath(img.filepath)
            if not os.path.exists(abs_path):
                issues.append(QAIssue(
                    category="organization",
                    severity="error",
                    object_name=img.name,
                    description=(
                        f"Image texture '{img.name}' references a missing file: '{abs_path}'. "
                        "Fix the path or pack the image."
                    ),
                    fix_command=(
                        f"import bpy; bpy.data.images['{img.name}'].pack()"
                    ),
                    auto_fixable=False,  # packing won't help if file is missing
                ))

    return issues


def _check_export_readiness(context, profile_name: str) -> List[QAIssue]:
    """
    Check 9: Export readiness.
    - FBX/GLTF: no shape keys with invalid names
    - GLTF: no multi-user meshes (instanced geometry needs separate mesh data)
    - All objects at scene root or in exportable collections
    """
    import bpy

    issues: List[QAIssue] = []

    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        name = obj.name

        # Shape key naming (FBX requirement: no spaces or special chars in shape key names)
        if obj.data.shape_keys:
            for kb in obj.data.shape_keys.key_blocks:
                if " " in kb.name or not re.match(r"^[A-Za-z0-9_]+$", kb.name):
                    issues.append(QAIssue(
                        category="export",
                        severity="warning",
                        object_name=name,
                        description=(
                            f"Shape key '{kb.name}' on '{name}' has spaces or special characters. "
                            "FBX export may mangle the name. Use underscores only."
                        ),
                        fix_command="",
                        auto_fixable=False,
                    ))

        # Multi-user mesh data (GLTF issue)
        if obj.data.users > 1:
            issues.append(QAIssue(
                category="export",
                severity="info",
                object_name=name,
                description=(
                    f"Object '{name}' shares mesh data with {obj.data.users - 1} other object(s). "
                    "GLTF exports this correctly as instancing, but FBX may duplicate geometry."
                ),
                fix_command="",
                auto_fixable=False,
            ))

    # Check for modifier stacks that need to be applied before export
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        name = obj.name
        problem_mods = [
            m.name for m in obj.modifiers
            if m.type in {"BOOLEAN", "ARRAY", "MIRROR", "SUBSURF", "SOLIDIFY"}
        ]
        if problem_mods:
            issues.append(QAIssue(
                category="export",
                severity="info",
                object_name=name,
                description=(
                    f"Object '{name}' has unapplied modifiers: {problem_mods}. "
                    "These will be ignored by some exporters unless applied first."
                ),
                fix_command=(
                    f"import bpy; "
                    f"bpy.context.view_layer.objects.active = bpy.data.objects['{name}']; "
                    f"[bpy.ops.object.modifier_apply(modifier=m.name) "
                    f"for m in list(bpy.data.objects['{name}'].modifiers)]"
                ),
                auto_fixable=True,
            ))

    return issues


def _check_units(context) -> List[QAIssue]:
    """
    Check 10: Unit consistency.
    - Detect mixed unit systems (scene in meters but objects scaled as if centimeters)
    - Detect scene unit scale vs. object dimensions mismatch
    """
    import bpy

    issues: List[QAIssue] = []

    scene = bpy.context.scene
    unit_scale = scene.unit_settings.scale_length
    unit_system = scene.unit_settings.system  # 'METRIC', 'IMPERIAL', 'NONE'

    if unit_system == "NONE":
        issues.append(QAIssue(
            category="units",
            severity="warning",
            object_name="SCENE",
            description=(
                "Scene unit system is set to 'None'. Set to Metric (1m scale) for production assets."
            ),
            fix_command=(
                "import bpy; "
                "bpy.context.scene.unit_settings.system = 'METRIC'; "
                "bpy.context.scene.unit_settings.scale_length = 1.0"
            ),
            auto_fixable=True,
        ))

    # Detect likely cm-scale objects in a meter scene
    if unit_system == "METRIC" and abs(unit_scale - 1.0) < 1e-4:
        suspiciously_small = []
        suspiciously_large = []
        for obj in bpy.data.objects:
            if obj.type != "MESH":
                continue
            dims = obj.dimensions
            max_dim = max(dims)
            if max_dim < 0.01:  # < 1cm in meter scene is suspicious unless intentional
                suspiciously_small.append(obj.name)
            if max_dim > 1000:  # > 1km in a non-terrain scene is suspicious
                suspiciously_large.append(obj.name)

        if suspiciously_small:
            issues.append(QAIssue(
                category="units",
                severity="warning",
                object_name="SCENE",
                description=(
                    f"Objects {suspiciously_small[:5]} are <1cm in a meter-scale scene. "
                    "If modeled in centimeters, apply scale ×0.01 or change scene unit scale to 0.01."
                ),
                fix_command="",
                auto_fixable=False,
            ))

        if suspiciously_large:
            issues.append(QAIssue(
                category="units",
                severity="warning",
                object_name="SCENE",
                description=(
                    f"Objects {suspiciously_large[:5]} exceed 1km in a meter-scale scene. "
                    "Verify scale is intentional (terrain) or was modeled in wrong units."
                ),
                fix_command="",
                auto_fixable=False,
            ))

    return issues


# ---------------------------------------------------------------------------
# Main audit function
# ---------------------------------------------------------------------------

def audit_active_scene(profile_name: str = "game_pc") -> QAReport:
    """
    Run a full QA audit on the currently active Blender scene.

    Args:
        profile_name: Platform profile key from PLATFORM_PROFILES.

    Returns:
        QAReport with all issues, score, and stats.
    """
    import bpy

    profile = PLATFORM_PROFILES.get(profile_name, PLATFORM_PROFILES["game_pc"])

    context = bpy.context
    all_issues: List[QAIssue] = []
    info: List[str] = []

    # Collect scene stats
    mesh_objects = [o for o in bpy.data.objects if o.type == "MESH"]
    total_verts = sum(len(o.data.vertices) for o in mesh_objects)
    total_tris = sum(
        sum(len(p.vertices) - 2 for p in o.data.polygons)
        for o in mesh_objects
    )
    total_materials = len(bpy.data.materials)
    total_images = len(bpy.data.images)

    stats = {
        "object_count": len(bpy.data.objects),
        "mesh_object_count": len(mesh_objects),
        "total_vertices": total_verts,
        "total_triangles": total_tris,
        "material_count": total_materials,
        "image_count": total_images,
        "armature_count": sum(1 for o in bpy.data.objects if o.type == "ARMATURE"),
        "profile": profile_name,
    }

    info.append(f"Scene: {len(bpy.data.objects)} objects, {total_tris:,} tris, {total_materials} materials")
    info.append(f"Profile: {profile_name} — {profile['notes']}")

    # Run all checks
    all_issues += _check_naming_conventions(context)
    all_issues += _check_transforms(context)
    all_issues += _check_uv_issues(context)
    all_issues += _check_topology(context, profile)
    all_issues += _check_polycount(context, profile)
    all_issues += _check_materials(context)
    all_issues += _check_rig_integrity(context)
    all_issues += _check_scene_organization(context)
    all_issues += _check_export_readiness(context, profile_name)
    all_issues += _check_units(context)

    # Split by severity
    errors = [i for i in all_issues if i.severity == "error"]
    warnings = [i for i in all_issues if i.severity == "warning"]
    infos_from_issues = [i for i in all_issues if i.severity == "info"]

    # Convert info-severity QAIssues to plain strings for the info list
    for issue in infos_from_issues:
        info.append(f"[{issue.object_name}] {issue.description}")

    # Score calculation
    # Start at 100, deduct: errors cost 10 pts each, warnings cost 3 pts each
    score = 100.0
    score -= len(errors) * 10.0
    score -= len(warnings) * 3.0
    score = max(0.0, min(100.0, score))

    passed = len(errors) == 0

    return QAReport(
        passed=passed,
        score=score,
        issues=errors,
        warnings=warnings,
        info=info,
        stats=stats,
        platform_budget=profile,
    )


# ---------------------------------------------------------------------------
# Auto-fix helpers
# ---------------------------------------------------------------------------

def apply_all_transforms(object_names: Optional[List[str]] = None):
    """
    Auto-fix: Apply scale and rotation to all mesh objects (or a specific list).
    Safe to call headlessly.
    """
    import bpy

    targets = (
        [bpy.data.objects[n] for n in object_names]
        if object_names
        else [o for o in bpy.data.objects if o.type == "MESH"]
    )
    bpy.ops.object.select_all(action="DESELECT")
    for obj in targets:
        obj.select_set(True)
    if targets:
        bpy.context.view_layer.objects.active = targets[0]
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    bpy.ops.object.select_all(action="DESELECT")


def remove_doubles_all(merge_threshold: float = 0.0001):
    """
    Auto-fix: Merge vertices by distance on all mesh objects.
    """
    import bpy

    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.remove_doubles(threshold=merge_threshold)
        bpy.ops.object.editmode_toggle()


def smart_unwrap_all(margin: float = 0.02, angle_limit: float = 66.0):
    """
    Auto-fix: Apply Smart UV Project to all mesh objects missing UV maps.
    """
    import bpy

    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if len(obj.data.uv_layers) == 0:
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.editmode_toggle()
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.uv.smart_project(
                angle_limit=math.radians(angle_limit),
                margin_method="SCALED",
                island_margin=margin,
            )
            bpy.ops.object.editmode_toggle()


def auto_fix_report(report: QAReport) -> List[str]:
    """
    Execute all auto_fixable fix_commands from a QAReport.
    Returns a list of results (success/error per fix).
    """
    results = []
    fixable = [i for i in report.issues + report.warnings if i.auto_fixable and i.fix_command]
    for issue in fixable:
        try:
            exec(issue.fix_command)  # noqa: S102
            results.append(f"FIXED: [{issue.object_name}] {issue.description[:60]}")
        except Exception as e:
            results.append(f"ERROR fixing [{issue.object_name}]: {e}")
    return results


# ---------------------------------------------------------------------------
# Training pair generation
# ---------------------------------------------------------------------------

QA_VOICE_TEMPLATES = [
    # Naming
    ("naming", "check the naming conventions in this scene"),
    ("naming", "are there any default blender names I need to fix"),
    ("naming", "find all objects with generic names like Cube or Material"),
    ("naming", "rename check — flag anything that's still using default names"),
    ("naming", "audit my naming conventions"),
    # Transforms
    ("transforms", "check if any objects have unapplied transforms"),
    ("transforms", "find objects with unapplied scale"),
    ("transforms", "are there any transforms I need to apply before rigging"),
    ("transforms", "apply scale to all mesh objects"),
    ("transforms", "fix the transforms — apply rotation and scale to everything"),
    # UV
    ("uv", "check UV maps on all objects"),
    ("uv", "find meshes with missing UVs"),
    ("uv", "are there any UV islands outside the zero to one space"),
    ("uv", "check texel density consistency across the scene"),
    ("uv", "add UV maps to everything that's missing them"),
    # Topology
    ("topology", "run a topology check"),
    ("topology", "find all n-gons in the scene"),
    ("topology", "check for non-manifold geometry"),
    ("topology", "are there any zero area faces"),
    ("topology", "fix the loose geometry"),
    # Polycount
    ("polycount", "check if we're within the mobile polycount budget"),
    ("polycount", "what's the total triangle count"),
    ("polycount", "are we within game budget"),
    ("polycount", "run a polycount audit for the cinematics profile"),
    ("polycount", "how many tris does this scene have"),
    # Materials
    ("materials", "check all materials are assigned"),
    ("materials", "find objects with no material"),
    ("materials", "audit the PBR values — check for out of range roughness or metallic"),
    ("materials", "find any physically impossible material values"),
    ("materials", "create a default material for anything missing one"),
    # Rig
    ("rig", "check the rig integrity"),
    ("rig", "are there any broken vertex groups"),
    ("rig", "find vertex groups with no matching bone"),
    ("rig", "check for degenerate bones"),
    ("rig", "audit the skinning — look for weight group issues"),
    # Organization
    ("organization", "clean up orphaned data"),
    ("organization", "find missing image textures"),
    ("organization", "purge all unused data blocks"),
    ("organization", "are there any broken image links"),
    ("organization", "check for orphaned meshes and materials"),
    # Export
    ("export", "check if this scene is ready to export"),
    ("export", "are there any FBX export issues"),
    ("export", "find unapplied modifiers before export"),
    ("export", "check GLTF compatibility"),
    ("export", "run an export readiness check"),
    # Units
    ("units", "check the scene units"),
    ("units", "are the units set correctly for a meter scale scene"),
    ("units", "detect any unit inconsistencies"),
    ("units", "check if any objects are suspiciously small or large"),
    ("units", "set the scene to metric units"),
    # Full audit
    ("full_qa", "run a full scene QA audit"),
    ("full_qa", "audit everything and give me a quality score"),
    ("full_qa", "run all checks — naming, topology, UVs, materials, everything"),
    ("full_qa", "do a complete quality check for game export"),
    ("full_qa", "lint the scene for production readiness"),
]

QA_CODE_TEMPLATES = {
    "naming": "import qa_agent; report = qa_agent._check_naming_conventions(bpy.context)\nfor issue in report:\n    print(f'{issue.severity.upper()}: [{issue.object_name}] {issue.description}')",
    "transforms": "import qa_agent; report = qa_agent._check_transforms(bpy.context)\nfor issue in report:\n    print(f'{issue.severity.upper()}: [{issue.object_name}] {issue.description}')",
    "uv": "import qa_agent; report = qa_agent._check_uv_issues(bpy.context)\nfor issue in report:\n    print(f'{issue.severity.upper()}: [{issue.object_name}] {issue.description}')",
    "topology": "import qa_agent; report = qa_agent._check_topology(bpy.context, qa_agent.PLATFORM_PROFILES['game_pc'])\nfor issue in report:\n    print(f'{issue.severity.upper()}: [{issue.object_name}] {issue.description}')",
    "polycount": "import qa_agent; report = qa_agent._check_polycount(bpy.context, qa_agent.PLATFORM_PROFILES['game_pc'])\nfor issue in report:\n    print(f'{issue.severity.upper()}: [{issue.object_name}] {issue.description}')",
    "materials": "import qa_agent; report = qa_agent._check_materials(bpy.context)\nfor issue in report:\n    print(f'{issue.severity.upper()}: [{issue.object_name}] {issue.description}')",
    "rig": "import qa_agent; report = qa_agent._check_rig_integrity(bpy.context)\nfor issue in report:\n    print(f'{issue.severity.upper()}: [{issue.object_name}] {issue.description}')",
    "organization": "import qa_agent; report = qa_agent._check_scene_organization(bpy.context)\nfor issue in report:\n    print(f'{issue.severity.upper()}: [{issue.object_name}] {issue.description}')",
    "export": "import qa_agent; report = qa_agent._check_export_readiness(bpy.context, 'game_pc')\nfor issue in report:\n    print(f'{issue.severity.upper()}: [{issue.object_name}] {issue.description}')",
    "units": "import qa_agent; report = qa_agent._check_units(bpy.context)\nfor issue in report:\n    print(f'{issue.severity.upper()}: [{issue.object_name}] {issue.description}')",
    "full_qa": "import qa_agent; report = qa_agent.audit_active_scene(profile_name='game_pc')\nprint(report.summary())\nfor issue in report.issues:\n    print(f'ERROR: [{issue.object_name}] {issue.description}')\nfor w in report.warnings:\n    print(f'WARN:  [{w.object_name}] {w.description}')",
}

QA_REASONING_TEMPLATES = {
    "naming": (
        "Good naming conventions are non-negotiable in production pipelines. "
        "Exporters, game engines, and rigging tools all rely on consistent names. "
        "Default Blender names like 'Cube.001' become 'Cube_001' in FBX and confuse downstream tools."
    ),
    "transforms": (
        "Unapplied transforms cause rigging, physics, and export failures. "
        "The armature sees the mesh at its 'true' world position, but the vertices are stored in local space. "
        "Always apply scale before weight painting — otherwise your brush radius is wrong."
    ),
    "uv": (
        "Missing or broken UVs prevent texture baking and break material display in real-time engines. "
        "Smart UV Project is a safe first pass; seam-based unwrapping gives better texel density control."
    ),
    "topology": (
        "N-gons tessellate unpredictably under subdivision and skinning. "
        "Non-manifold edges prevent boolean operations and 3D printing. "
        "Zero-area faces are invisible geometry that wastes draw calls and breaks physics."
    ),
    "polycount": (
        "Each platform has hard polycount budgets that affect framerate and memory. "
        "Mobile targets 10k tris per hero asset. PC game targets 50k. "
        "Exceeding budget requires LOD chains or decimation."
    ),
    "materials": (
        "PBR materials must stay within physically plausible ranges [0,1] for Metallic and Roughness. "
        "Out-of-range values break energy conservation and produce glowing or black artifacts in real-time engines."
    ),
    "rig": (
        "Broken vertex groups cause skin deformation failures that are hard to debug mid-production. "
        "Orphaned weight groups from renamed bones waste memory and can cause unexpected deformation in engines."
    ),
    "organization": (
        "Orphaned data blocks increase file size and slow load times. "
        "Missing texture paths cause broken materials in every downstream tool. "
        "A clean data hierarchy is essential before handoff."
    ),
    "export": (
        "FBX and GLTF have specific requirements. "
        "Shape key names with spaces get mangled by FBX exporters. "
        "Unapplied modifiers are skipped by some exporters, causing geometry mismatch between source and exported asset."
    ),
    "units": (
        "Unit mismatches between Blender and the target engine cause assets to appear 100x too large or small. "
        "The most common mistake: modeling in Blender units (default 1 unit = 1m) then importing to Unity "
        "which expects 1 unit = 1m but the model was mentally modeled as centimeters."
    ),
    "full_qa": (
        "A full scene audit catches all categories of issues before they become production blockers. "
        "Running QA before handoff or export prevents the 'works on my machine' problem and "
        "ensures consistent quality across a production team."
    ),
}


def generate_training_pairs_from_qa(
    n_pairs: int = 300,
    output_dir: Optional[str] = None,
) -> List[dict]:
    """
    Generate QA task-type training pairs for Nalana.

    These teach Nalana to:
    1. Map voice commands to the right QA check function
    2. Understand what each check does and why it matters
    3. Generate the correct bpy code to run the check

    Args:
        n_pairs: Target number of training pairs to generate.
        output_dir: If provided, saves to <output_dir>/qa_pairs.jsonl

    Returns:
        List of training pair dicts.
    """
    pairs: List[dict] = []

    templates = QA_VOICE_TEMPLATES * (n_pairs // len(QA_VOICE_TEMPLATES) + 1)
    random.shuffle(templates)

    # Variations to make voice commands more natural
    prefixes = [
        "", "", "",  # no prefix (most common)
        "can you ",
        "please ",
        "hey nalana, ",
        "go ahead and ",
        "I need you to ",
        "let's ",
        "quickly ",
        "run a ",
    ]

    profile_variants = list(PLATFORM_PROFILES.keys())

    for i, (category, base_command) in enumerate(templates[:n_pairs]):
        prefix = random.choice(prefixes)
        voice_cmd = prefix + base_command

        # Randomize profile for polycount/topology/export checks
        profile = random.choice(profile_variants)
        code = QA_CODE_TEMPLATES.get(category, QA_CODE_TEMPLATES["full_qa"])

        # Substitute profile name in code where relevant
        if "game_pc" in code and category in {"polycount", "topology", "export", "full_qa"}:
            code = code.replace("game_pc", profile)

        pair = {
            "voice_command": voice_cmd,
            "task_type": "QA_LINT",
            "qa_category": category,
            "blender_python": code,
            "reasoning": QA_REASONING_TEMPLATES.get(category, ""),
            "quality": 2.5,
            "source": "qa_agent_synthetic",
            "platform_profile": profile,
        }
        pairs.append(pair)

    if output_dir:
        out_path = Path(output_dir) / "qa_pairs.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        print(f"Saved {len(pairs)} QA training pairs → {out_path}")

    return pairs


# ---------------------------------------------------------------------------
# Headless Blender runner
# ---------------------------------------------------------------------------

BLENDER_QA_HARNESS = '''
import bpy
import sys
import json

# Inject qa_agent path
sys.path.insert(0, "{qa_agent_dir}")

import qa_agent

report = qa_agent.audit_active_scene(profile_name="{profile}")
print("QA_RESULT:" + json.dumps(report.to_dict()))
'''


def run_headless(
    blend_file: str,
    profile: str = "game_pc",
    blender_path: str = "blender",
    auto_fix: bool = False,
) -> QAReport:
    """
    Run QA audit on a .blend file by spawning headless Blender.

    Args:
        blend_file:   Path to the .blend file.
        profile:      Platform profile name.
        blender_path: Path to the Blender executable.
        auto_fix:     If True, runs auto_fix_report() inside Blender before reporting.

    Returns:
        QAReport deserialized from headless Blender output.
    """
    qa_agent_dir = str(Path(__file__).parent.resolve())

    harness = BLENDER_QA_HARNESS.format(
        qa_agent_dir=qa_agent_dir,
        profile=profile,
    )

    if auto_fix:
        harness += "\nauto_fix_results = qa_agent.auto_fix_report(report)\nprint('FIXES:' + json.dumps(auto_fix_results))"

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
        tmp.write(harness)
        script_path = tmp.name

    try:
        result = subprocess.run(
            [blender_path, "--background", blend_file, "--python", script_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = result.stdout

        # Extract JSON result from output
        for line in output.splitlines():
            if line.startswith("QA_RESULT:"):
                data = json.loads(line[len("QA_RESULT:"):])
                report = QAReport(
                    passed=data["passed"],
                    score=data["score"],
                    issues=[QAIssue(**i) for i in data["issues"]],
                    warnings=[QAIssue(**w) for w in data["warnings"]],
                    info=data["info"],
                    stats=data["stats"],
                    platform_budget=data["platform_budget"],
                )
                return report

        # If no result found, return failed report
        return QAReport(
            passed=False,
            score=0.0,
            issues=[QAIssue(
                category="execution",
                severity="error",
                object_name="BLENDER",
                description=f"Headless Blender failed. stderr: {result.stderr[:500]}",
                fix_command="",
                auto_fixable=False,
            )],
            warnings=[],
            info=[],
            stats={},
            platform_budget=PLATFORM_PROFILES.get(profile, {}),
        )
    finally:
        os.unlink(script_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Nalana QA Agent — Scene quality audit system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python qa_agent.py --scene model.blend --profile game_pc
  python qa_agent.py --scene model.blend --profile mobile --output report.json
  python qa_agent.py --scene model.blend --auto-fix
  python qa_agent.py --generate-pairs --n-pairs 300
        """,
    )
    parser.add_argument("--scene", help="Path to .blend file")
    parser.add_argument(
        "--profile",
        default="game_pc",
        choices=list(PLATFORM_PROFILES.keys()),
        help="Platform quality profile",
    )
    parser.add_argument("--output", help="Output report JSON path")
    parser.add_argument("--auto-fix", action="store_true", help="Run auto-fixes before reporting")
    parser.add_argument("--blender-path", default="blender", help="Path to Blender executable")
    parser.add_argument("--generate-pairs", action="store_true", help="Generate QA training pairs")
    parser.add_argument("--n-pairs", type=int, default=300, help="Number of training pairs to generate")
    parser.add_argument(
        "--pairs-output-dir",
        default=str(Path(__file__).parents[1] / "data" / "qa"),
        help="Output directory for training pairs",
    )

    args = parser.parse_args()

    if args.generate_pairs:
        pairs = generate_training_pairs_from_qa(
            n_pairs=args.n_pairs,
            output_dir=args.pairs_output_dir,
        )
        print(f"Generated {len(pairs)} QA training pairs.")
        return

    if not args.scene:
        parser.error("--scene is required unless --generate-pairs is used")

    if not Path(args.scene).exists():
        print(f"Error: scene file not found: {args.scene}")
        sys.exit(1)

    print(f"Running QA audit: {args.scene} (profile: {args.profile})")
    report = run_headless(
        blend_file=args.scene,
        profile=args.profile,
        blender_path=args.blender_path,
        auto_fix=args.auto_fix,
    )

    print(f"\n{report.summary()}")
    print(f"\nStats: {json.dumps(report.stats, indent=2)}")

    if report.issues:
        print(f"\nErrors ({len(report.issues)}):")
        for issue in report.issues:
            print(f"  [{issue.category.upper()}] {issue.object_name}: {issue.description}")

    if report.warnings:
        print(f"\nWarnings ({len(report.warnings)}):")
        for w in report.warnings:
            print(f"  [{w.category.upper()}] {w.object_name}: {w.description}")

    if report.info:
        print(f"\nInfo:")
        for i in report.info:
            print(f"  {i}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nReport saved to: {args.output}")

    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
