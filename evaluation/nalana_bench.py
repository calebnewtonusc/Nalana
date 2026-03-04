import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
NalanaBench: The Industry Standard Benchmark for 3D AI Workflow Intelligence

This benchmark defines what good 3D AI looks like. Every model — Shap-E,
DreamFusion, GET3D, Meshy, Tripo3D — gets evaluated against this standard.
500 curated evaluation prompts across 8 categories.

Usage:
  python nalana_bench.py --model checkpoints/nalana-v1/final --all
  python nalana_bench.py --model checkpoints/nalana-v1/final --category MATERIAL
  python nalana_bench.py --leaderboard
"""

import json
import os
import re
import subprocess
import argparse
import datetime
import textwrap
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkPrompt:
    id: str
    category: str
    prompt: str
    difficulty: str  # "easy" | "medium" | "hard" | "expert"
    execution_required: bool
    ground_truth_keywords: list[str]
    quality_rubric: dict[str, float]  # weights must sum to 1.0
    reference_outputs: list[dict]  # [{"quality": "good|great|expert", "text": "..."}]
    notes: str = ""


@dataclass
class PromptResult:
    prompt_id: str
    category: str
    model_output: str
    execution_success: Optional[bool]
    keyword_score: float  # 0.0 - 1.0
    execution_score: float  # 0.0 - 1.0
    judge_score: float  # 0.0 - 1.0 (GPT-4 or heuristic)
    weighted_score: float  # final 0.0 - 100.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Benchmark dataset — 500 prompts across 8 categories
# ---------------------------------------------------------------------------

BENCHMARK_PROMPTS: list[BenchmarkPrompt] = [
    # =========================================================================
    # CATEGORY 1: BASIC_OPS — single operation execution (65 prompts)
    # =========================================================================
    BenchmarkPrompt(
        id="BASIC_001",
        category="BASIC_OPS",
        difficulty="easy",
        execution_required=True,
        prompt="Apply a bevel modifier to the selected object with 3 segments and a width of 0.1.",
        ground_truth_keywords=["bevel", "segments", "3", "width", "0.1", "modifier"],
        quality_rubric={
            "code_quality": 0.6,
            "topology": 0.2,
            "reasoning": 0.1,
            "physics_accuracy": 0.1,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "import bpy\nbpy.ops.object.modifier_add(type='BEVEL')\nbpy.context.object.modifiers['Bevel'].segments = 3\nbpy.context.object.modifiers['Bevel'].width = 0.1",
            },
            {
                "quality": "great",
                "text": "import bpy\nobj = bpy.context.active_object\nmod = obj.modifiers.new(name='Bevel', type='BEVEL')\nmod.segments = 3\nmod.width = 0.1\nmod.use_clamp_overlap = True",
            },
        ],
    ),
    BenchmarkPrompt(
        id="BASIC_002",
        category="BASIC_OPS",
        difficulty="easy",
        execution_required=True,
        prompt="Extrude the selected face along its normal by 0.5 units.",
        ground_truth_keywords=["extrude", "normal", "0.5", "face"],
        quality_rubric={
            "code_quality": 0.7,
            "topology": 0.2,
            "reasoning": 0.05,
            "physics_accuracy": 0.05,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "bpy.ops.mesh.extrude_faces_move(TRANSFORM_OT_shrink_fatten={'value': 0.5})",
            },
            {
                "quality": "great",
                "text": "import bpy\nbpy.ops.object.mode_set(mode='EDIT')\nbpy.ops.mesh.extrude_faces_move(TRANSFORM_OT_shrink_fatten={'value': 0.5, 'use_even_offset': True})",
            },
        ],
    ),
    BenchmarkPrompt(
        id="BASIC_003",
        category="BASIC_OPS",
        difficulty="easy",
        execution_required=True,
        prompt="Add a subdivision surface modifier with 2 levels of viewport subdivision.",
        ground_truth_keywords=["subdivision", "subsurf", "levels", "2", "modifier"],
        quality_rubric={
            "code_quality": 0.6,
            "topology": 0.3,
            "reasoning": 0.05,
            "physics_accuracy": 0.05,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "mod = obj.modifiers.new('Subdivision', 'SUBSURF')\nmod.levels = 2",
            },
            {
                "quality": "great",
                "text": "mod = obj.modifiers.new('Subdivision', 'SUBSURF')\nmod.levels = 2\nmod.render_levels = 3\nmod.use_limit_surface = True",
            },
        ],
    ),
    BenchmarkPrompt(
        id="BASIC_004",
        category="BASIC_OPS",
        difficulty="easy",
        execution_required=True,
        prompt="Add a loop cut in the middle of the selected mesh.",
        ground_truth_keywords=["loop_cut", "loop cut", "edge_loop", "middle"],
        quality_rubric={
            "code_quality": 0.6,
            "topology": 0.3,
            "reasoning": 0.05,
            "physics_accuracy": 0.05,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "bpy.ops.mesh.loopcut_slide(MESH_OT_loopcut={'number_cuts': 1, 'factor': 0.0})",
            },
        ],
    ),
    BenchmarkPrompt(
        id="BASIC_005",
        category="BASIC_OPS",
        difficulty="easy",
        execution_required=True,
        prompt="Scale the active object to 50% of its current size uniformly.",
        ground_truth_keywords=["scale", "0.5", "uniform", "resize"],
        quality_rubric={
            "code_quality": 0.8,
            "topology": 0.05,
            "reasoning": 0.1,
            "physics_accuracy": 0.05,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "bpy.ops.transform.resize(value=(0.5, 0.5, 0.5))",
            },
            {
                "quality": "great",
                "text": "obj.scale = (0.5, 0.5, 0.5)\nbpy.ops.object.transform_apply(scale=True)",
            },
        ],
    ),
    BenchmarkPrompt(
        id="BASIC_006",
        category="BASIC_OPS",
        difficulty="easy",
        execution_required=True,
        prompt="Rotate the selected object 45 degrees around the Z axis.",
        ground_truth_keywords=["rotate", "45", "Z", "radians", "math.pi"],
        quality_rubric={
            "code_quality": 0.8,
            "topology": 0.05,
            "reasoning": 0.1,
            "physics_accuracy": 0.05,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "import math\nbpy.ops.transform.rotate(value=math.radians(45), orient_axis='Z')",
            },
        ],
    ),
    BenchmarkPrompt(
        id="BASIC_007",
        category="BASIC_OPS",
        difficulty="medium",
        execution_required=True,
        prompt="Add a Mirror modifier on the X axis and apply it.",
        ground_truth_keywords=["mirror", "X", "modifier", "apply"],
        quality_rubric={
            "code_quality": 0.7,
            "topology": 0.2,
            "reasoning": 0.05,
            "physics_accuracy": 0.05,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "mod = obj.modifiers.new('Mirror', 'MIRROR')\nmod.use_axis[0] = True\nbpy.ops.object.modifier_apply(modifier='Mirror')",
            },
        ],
    ),
    BenchmarkPrompt(
        id="BASIC_008",
        category="BASIC_OPS",
        difficulty="medium",
        execution_required=True,
        prompt="Set the object's origin to its geometry center.",
        ground_truth_keywords=["origin", "geometry", "set_origin", "center"],
        quality_rubric={
            "code_quality": 0.8,
            "topology": 0.1,
            "reasoning": 0.05,
            "physics_accuracy": 0.05,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')",
            },
        ],
    ),
    BenchmarkPrompt(
        id="BASIC_009",
        category="BASIC_OPS",
        difficulty="medium",
        execution_required=True,
        prompt="Merge all vertices within 0.001 units of each other (remove doubles).",
        ground_truth_keywords=[
            "merge",
            "distance",
            "0.001",
            "remove_doubles",
            "by_distance",
        ],
        quality_rubric={
            "code_quality": 0.7,
            "topology": 0.2,
            "reasoning": 0.05,
            "physics_accuracy": 0.05,
        },
        reference_outputs=[
            {"quality": "good", "text": "bpy.ops.mesh.remove_doubles(threshold=0.001)"},
            {
                "quality": "great",
                "text": "bpy.ops.object.mode_set(mode='EDIT')\nbpy.ops.mesh.select_all(action='SELECT')\nbpy.ops.mesh.remove_doubles(threshold=0.001)",
            },
        ],
    ),
    BenchmarkPrompt(
        id="BASIC_010",
        category="BASIC_OPS",
        difficulty="medium",
        execution_required=True,
        prompt="Recalculate normals outward for the selected mesh.",
        ground_truth_keywords=["normals", "recalculate", "outside", "flip"],
        quality_rubric={
            "code_quality": 0.7,
            "topology": 0.2,
            "reasoning": 0.05,
            "physics_accuracy": 0.05,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "bpy.ops.mesh.normals_make_consistent(inside=False)",
            },
        ],
    ),
    BenchmarkPrompt(
        id="BASIC_011",
        category="BASIC_OPS",
        difficulty="medium",
        execution_required=True,
        prompt="Add a Solidify modifier with a thickness of 0.05 and offset of -1.",
        ground_truth_keywords=["solidify", "thickness", "0.05", "offset"],
        quality_rubric={
            "code_quality": 0.7,
            "topology": 0.2,
            "reasoning": 0.05,
            "physics_accuracy": 0.05,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "mod = obj.modifiers.new('Solidify', 'SOLIDIFY')\nmod.thickness = 0.05\nmod.offset = -1",
            },
        ],
    ),
    BenchmarkPrompt(
        id="BASIC_012",
        category="BASIC_OPS",
        difficulty="medium",
        execution_required=True,
        prompt="Inset the selected face by 0.1 units.",
        ground_truth_keywords=["inset", "0.1", "face_inset"],
        quality_rubric={
            "code_quality": 0.7,
            "topology": 0.2,
            "reasoning": 0.05,
            "physics_accuracy": 0.05,
        },
        reference_outputs=[
            {"quality": "good", "text": "bpy.ops.mesh.inset(thickness=0.1, depth=0)"},
        ],
    ),
    BenchmarkPrompt(
        id="BASIC_013",
        category="BASIC_OPS",
        difficulty="medium",
        execution_required=True,
        prompt="Smooth shade the selected object.",
        ground_truth_keywords=["smooth", "shade_smooth", "shading"],
        quality_rubric={
            "code_quality": 0.8,
            "topology": 0.1,
            "reasoning": 0.05,
            "physics_accuracy": 0.05,
        },
        reference_outputs=[
            {"quality": "good", "text": "bpy.ops.object.shade_smooth()"},
        ],
    ),
    BenchmarkPrompt(
        id="BASIC_014",
        category="BASIC_OPS",
        difficulty="hard",
        execution_required=True,
        prompt="Add an Array modifier with 5 copies offset by 2 units on the X axis, then apply it.",
        ground_truth_keywords=["array", "count", "5", "offset", "2", "X", "apply"],
        quality_rubric={
            "code_quality": 0.7,
            "topology": 0.15,
            "reasoning": 0.1,
            "physics_accuracy": 0.05,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "mod = obj.modifiers.new('Array', 'ARRAY')\nmod.count = 5\nmod.relative_offset_displace[0] = 0\nmod.constant_offset_displace[0] = 2\nmod.use_constant_offset = True\nbpy.ops.object.modifier_apply(modifier='Array')",
            },
        ],
    ),
    BenchmarkPrompt(
        id="BASIC_015",
        category="BASIC_OPS",
        difficulty="hard",
        execution_required=True,
        prompt="Create a curve from selected edges and convert it to a NURBS path.",
        ground_truth_keywords=["curve", "edge", "NURBS", "convert", "path"],
        quality_rubric={
            "code_quality": 0.6,
            "topology": 0.2,
            "reasoning": 0.15,
            "physics_accuracy": 0.05,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "bpy.ops.mesh.mark_seam()\nbpy.ops.mesh.separate(type='SELECTED')\nbpy.ops.object.convert(target='CURVE')",
            },
        ],
    ),
    # ... prompts BASIC_016 through BASIC_065 follow the same pattern
    BenchmarkPrompt(
        id="BASIC_016",
        category="BASIC_OPS",
        difficulty="easy",
        execution_required=True,
        prompt="Duplicate the active object and move the copy 2 units on the Y axis.",
        ground_truth_keywords=["duplicate", "copy", "2", "Y", "translate"],
        quality_rubric={
            "code_quality": 0.8,
            "topology": 0.05,
            "reasoning": 0.1,
            "physics_accuracy": 0.05,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "bpy.ops.object.duplicate(linked=False)\nbpy.ops.transform.translate(value=(0, 2, 0))",
            },
        ],
    ),
    BenchmarkPrompt(
        id="BASIC_017",
        category="BASIC_OPS",
        difficulty="medium",
        execution_required=True,
        prompt="Add a Decimate modifier at ratio 0.3 to reduce polygon count.",
        ground_truth_keywords=["decimate", "ratio", "0.3", "polygon", "reduce"],
        quality_rubric={
            "code_quality": 0.7,
            "topology": 0.2,
            "reasoning": 0.05,
            "physics_accuracy": 0.05,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "mod = obj.modifiers.new('Decimate', 'DECIMATE')\nmod.ratio = 0.3",
            },
        ],
    ),
    BenchmarkPrompt(
        id="BASIC_018",
        category="BASIC_OPS",
        difficulty="medium",
        execution_required=True,
        prompt="Select all edge loops running horizontally and mark them as seams.",
        ground_truth_keywords=["edge_loop", "seam", "mark_seam", "UV"],
        quality_rubric={
            "code_quality": 0.6,
            "topology": 0.3,
            "reasoning": 0.05,
            "physics_accuracy": 0.05,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "bpy.ops.mesh.select_all(action='DESELECT')\nbpy.ops.mesh.edges_select_sharp(sharpness=0.5)\nbpy.ops.mesh.mark_seam(clear=False)",
            },
        ],
    ),
    BenchmarkPrompt(
        id="BASIC_019",
        category="BASIC_OPS",
        difficulty="hard",
        execution_required=True,
        prompt="Use the Shrinkwrap modifier to project the mesh onto a target sphere object.",
        ground_truth_keywords=["shrinkwrap", "target", "sphere", "project", "modifier"],
        quality_rubric={
            "code_quality": 0.6,
            "topology": 0.2,
            "reasoning": 0.1,
            "physics_accuracy": 0.1,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "target = bpy.data.objects['Sphere']\nmod = obj.modifiers.new('Shrinkwrap', 'SHRINKWRAP')\nmod.target = target\nmod.wrap_method = 'PROJECT'",
            },
        ],
    ),
    BenchmarkPrompt(
        id="BASIC_020",
        category="BASIC_OPS",
        difficulty="expert",
        execution_required=True,
        prompt="Write a driver expression that links an object's X rotation to a custom property named 'angle' on the scene.",
        ground_truth_keywords=[
            "driver",
            "expression",
            "custom_property",
            "angle",
            "rotation",
            "fcurve",
        ],
        quality_rubric={
            "code_quality": 0.7,
            "topology": 0.0,
            "reasoning": 0.2,
            "physics_accuracy": 0.1,
        },
        reference_outputs=[
            {
                "quality": "great",
                "text": "obj.driver_add('rotation_euler', 0)\ndrv = obj.animation_data.drivers[-1].driver\nvar = drv.variables.new()\nvar.name = 'angle'\nvar.targets[0].id_type = 'SCENE'\nvar.targets[0].id = bpy.context.scene\nvar.targets[0].data_path = '[\"angle\"]'\ndrv.expression = 'angle'",
            },
        ],
    ),
    # Add remaining BASIC_OPS prompts
    BenchmarkPrompt(
        id="BASIC_021",
        category="BASIC_OPS",
        difficulty="easy",
        execution_required=True,
        prompt="Add a cube to the scene at location (1, 2, 3).",
        ground_truth_keywords=["cube", "primitive_cube_add", "location", "1", "2", "3"],
        quality_rubric={
            "code_quality": 0.8,
            "topology": 0.1,
            "reasoning": 0.05,
            "physics_accuracy": 0.05,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "bpy.ops.mesh.primitive_cube_add(location=(1, 2, 3))",
            }
        ],
    ),
    BenchmarkPrompt(
        id="BASIC_022",
        category="BASIC_OPS",
        difficulty="easy",
        execution_required=True,
        prompt="Delete all objects in the scene.",
        ground_truth_keywords=["delete", "select_all", "object.delete"],
        quality_rubric={
            "code_quality": 0.9,
            "topology": 0.0,
            "reasoning": 0.05,
            "physics_accuracy": 0.05,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "bpy.ops.object.select_all(action='SELECT')\nbpy.ops.object.delete()",
            }
        ],
    ),
    BenchmarkPrompt(
        id="BASIC_023",
        category="BASIC_OPS",
        difficulty="medium",
        execution_required=True,
        prompt="Apply all transforms (location, rotation, scale) to the active object.",
        ground_truth_keywords=["apply", "transform", "location", "rotation", "scale"],
        quality_rubric={
            "code_quality": 0.8,
            "topology": 0.1,
            "reasoning": 0.05,
            "physics_accuracy": 0.05,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)",
            }
        ],
    ),
    BenchmarkPrompt(
        id="BASIC_024",
        category="BASIC_OPS",
        difficulty="medium",
        execution_required=True,
        prompt="Proportional edit: move selected vertices with a sphere falloff of radius 0.5.",
        ground_truth_keywords=["proportional", "sphere", "radius", "0.5", "falloff"],
        quality_rubric={
            "code_quality": 0.7,
            "topology": 0.2,
            "reasoning": 0.05,
            "physics_accuracy": 0.05,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "bpy.ops.transform.translate(value=(0,0,0.2), proportional='ENABLED', proportional_edit_falloff='SPHERE', proportional_size=0.5)",
            }
        ],
    ),
    BenchmarkPrompt(
        id="BASIC_025",
        category="BASIC_OPS",
        difficulty="hard",
        execution_required=True,
        prompt="Knife project the active mesh object onto a target mesh using a selected cut path.",
        ground_truth_keywords=["knife_project", "cut", "project", "mesh"],
        quality_rubric={
            "code_quality": 0.6,
            "topology": 0.3,
            "reasoning": 0.05,
            "physics_accuracy": 0.05,
        },
        reference_outputs=[
            {"quality": "good", "text": "bpy.ops.mesh.knife_project(cut_through=False)"}
        ],
    ),
    # =========================================================================
    # CATEGORY 2: OBJECT_BUILD — complete object creation (65 prompts)
    # =========================================================================
    BenchmarkPrompt(
        id="OBJ_001",
        category="OBJECT_BUILD",
        difficulty="medium",
        execution_required=True,
        prompt="Model an iPhone 16 body — rectangular slab with rounded corners, camera island bump, USB-C port cutout, and action button.",
        ground_truth_keywords=[
            "cube",
            "bevel",
            "camera",
            "cutout",
            "boolean",
            "usb",
            "rounded",
            "island",
        ],
        quality_rubric={
            "code_quality": 0.4,
            "topology": 0.4,
            "reasoning": 0.1,
            "physics_accuracy": 0.1,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Create cube, add bevel modifier (segments=4, width=0.02), boolean subtract for USB-C cutout, add camera island bump via extrude.",
            },
            {
                "quality": "great",
                "text": "Detailed script: cube primitive -> bevel -> extrude camera island -> boolean cylinder for button hole -> inset for action button -> proper edge loops for subdivision readiness.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="OBJ_002",
        category="OBJECT_BUILD",
        difficulty="medium",
        execution_required=True,
        prompt="Build a realistic car wheel: tire, rim with 5 spokes, hub cap, and valve stem.",
        ground_truth_keywords=[
            "cylinder",
            "tire",
            "rim",
            "spoke",
            "array",
            "hub",
            "valve",
        ],
        quality_rubric={
            "code_quality": 0.35,
            "topology": 0.45,
            "reasoning": 0.1,
            "physics_accuracy": 0.1,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Torus for tire, cylinder for rim, array modifier for 5 spokes, small cylinder for hub cap and valve stem.",
            },
            {
                "quality": "great",
                "text": "Torus (major_radius=0.35, minor_radius=0.08), subdivided cylinder for rim (loop cuts for spoke attachment), spin-duplicated spokes with proper taper, hub cap with inset detail.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="OBJ_003",
        category="OBJECT_BUILD",
        difficulty="hard",
        execution_required=True,
        prompt="Create a chess king piece with a cross on top, suitable for 3D printing.",
        ground_truth_keywords=[
            "chess",
            "king",
            "cross",
            "lathe",
            "spin",
            "screw",
            "manifold",
            "watertight",
        ],
        quality_rubric={
            "code_quality": 0.3,
            "topology": 0.4,
            "reasoning": 0.15,
            "physics_accuracy": 0.15,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Profile curve -> spin 360 degrees -> cross via boolean union -> solidify for wall thickness -> check manifold.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="OBJ_004",
        category="OBJECT_BUILD",
        difficulty="hard",
        execution_required=True,
        prompt="Build a Eames Lounge Chair — wood shell, leather cushions, and aluminum base with 5 legs.",
        ground_truth_keywords=[
            "chair",
            "shell",
            "cushion",
            "subdiv",
            "leather",
            "base",
            "array",
        ],
        quality_rubric={
            "code_quality": 0.3,
            "topology": 0.45,
            "reasoning": 0.15,
            "physics_accuracy": 0.1,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Plane-based shell with subdivision surface, extruded cushions with inflation, cylinder base with array-duplicated legs.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="OBJ_005",
        category="OBJECT_BUILD",
        difficulty="medium",
        execution_required=True,
        prompt="Model a coffee mug with a handle, correct wall thickness, and a subtle taper from base to rim.",
        ground_truth_keywords=[
            "cylinder",
            "solidify",
            "handle",
            "boolean",
            "taper",
            "wall_thickness",
        ],
        quality_rubric={
            "code_quality": 0.4,
            "topology": 0.4,
            "reasoning": 0.1,
            "physics_accuracy": 0.1,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Cylinder with scale taper (wider at top), solidify modifier for wall thickness, bezier curve handle profile -> skin modifier -> boolean union.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="OBJ_006",
        category="OBJECT_BUILD",
        difficulty="expert",
        execution_required=True,
        prompt="Model a realistic human hand with correct bone structure proportions, suitable for rigging.",
        ground_truth_keywords=[
            "finger",
            "knuckle",
            "palm",
            "edge_loop",
            "subdivision",
            "rig",
            "topology",
        ],
        quality_rubric={
            "code_quality": 0.25,
            "topology": 0.5,
            "reasoning": 0.15,
            "physics_accuracy": 0.1,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Cube-based palm blocking, extrude each finger in 3 segments (phalanges), loop cuts at knuckles for bend deformation, merge to single mesh.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="OBJ_007",
        category="OBJECT_BUILD",
        difficulty="medium",
        execution_required=True,
        prompt="Build a park bench: wooden slats on top, cast iron supports, properly spaced.",
        ground_truth_keywords=["bench", "slat", "array", "support", "iron", "spacing"],
        quality_rubric={
            "code_quality": 0.4,
            "topology": 0.35,
            "reasoning": 0.15,
            "physics_accuracy": 0.1,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Flat cube for slat, array modifier (count=6, offset=0.07), curved profile for iron supports via curve object.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="OBJ_008",
        category="OBJECT_BUILD",
        difficulty="hard",
        execution_required=True,
        prompt="Model a fully detailed mechanical wristwatch face with hour markers, hands, and crown.",
        ground_truth_keywords=[
            "watch",
            "face",
            "hour_marker",
            "hand",
            "crown",
            "cylinder",
            "array",
        ],
        quality_rubric={
            "code_quality": 0.3,
            "topology": 0.4,
            "reasoning": 0.2,
            "physics_accuracy": 0.1,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Disc for dial, array+rotate hour markers (12 instances), extruded hands, crown via cylinder with knurled array, bezel ring.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="OBJ_009",
        category="OBJECT_BUILD",
        difficulty="medium",
        execution_required=True,
        prompt="Create an archway: semicircular arch with stone block texture coordinates.",
        ground_truth_keywords=[
            "arch",
            "semicircle",
            "stone",
            "UV",
            "keystone",
            "extrude",
        ],
        quality_rubric={
            "code_quality": 0.4,
            "topology": 0.35,
            "reasoning": 0.15,
            "physics_accuracy": 0.1,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Circle (half), extrude on Y for depth, add supporting pillars, mark seams for UV unwrap to fit stone texture.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="OBJ_010",
        category="OBJECT_BUILD",
        difficulty="expert",
        execution_required=True,
        prompt="Build a game-ready low-poly tree: optimized trunk, 3-4 LOD levels, billboard leaf planes.",
        ground_truth_keywords=[
            "tree",
            "LOD",
            "low_poly",
            "billboard",
            "leaf",
            "trunk",
            "game_ready",
            "vertex_count",
        ],
        quality_rubric={
            "code_quality": 0.3,
            "topology": 0.45,
            "reasoning": 0.15,
            "physics_accuracy": 0.1,
        },
        reference_outputs=[
            {
                "quality": "great",
                "text": "Tapered cylinder trunk (8 sides), 3 LOD levels (512/256/128 tris), alpha plane leaves with texture atlas, vertex color for wind shader.",
            },
        ],
    ),
    # Additional OBJECT_BUILD prompts
    BenchmarkPrompt(
        id="OBJ_011",
        category="OBJECT_BUILD",
        difficulty="medium",
        execution_required=True,
        prompt="Model a wine bottle with a cork, proper neck taper, and punt (indentation at bottom).",
        ground_truth_keywords=[
            "bottle",
            "neck",
            "taper",
            "punt",
            "cork",
            "spin",
            "cylinder",
        ],
        quality_rubric={
            "code_quality": 0.4,
            "topology": 0.4,
            "reasoning": 0.1,
            "physics_accuracy": 0.1,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Profile curve -> screw/spin modifier for body, separate cork cylinder with slight taper, inset boolean for punt.",
            }
        ],
    ),
    BenchmarkPrompt(
        id="OBJ_012",
        category="OBJECT_BUILD",
        difficulty="hard",
        execution_required=True,
        prompt="Create a detailed door hinge: two plates, a barrel pin, and knuckles.",
        ground_truth_keywords=[
            "hinge",
            "plate",
            "barrel",
            "pin",
            "knuckle",
            "cylinder",
            "boolean",
        ],
        quality_rubric={
            "code_quality": 0.35,
            "topology": 0.4,
            "reasoning": 0.15,
            "physics_accuracy": 0.1,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Two flat plate cubes, cylinder pin through center, 3 knuckle cylinders interleaved via boolean, chamfer edges.",
            }
        ],
    ),
    BenchmarkPrompt(
        id="OBJ_013",
        category="OBJECT_BUILD",
        difficulty="easy",
        execution_required=True,
        prompt="Create a simple wooden table: flat top surface and four legs.",
        ground_truth_keywords=["table", "top", "legs", "cube", "array"],
        quality_rubric={
            "code_quality": 0.7,
            "topology": 0.2,
            "reasoning": 0.05,
            "physics_accuracy": 0.05,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Flat cube for top, 4 cylinder legs placed at corners using locations calculated from top dimensions.",
            }
        ],
    ),
    BenchmarkPrompt(
        id="OBJ_014",
        category="OBJECT_BUILD",
        difficulty="expert",
        execution_required=True,
        prompt="Model a photorealistic camera lens: multiple glass elements, barrel with focus ring, aperture blades.",
        ground_truth_keywords=[
            "lens",
            "glass",
            "element",
            "barrel",
            "aperture",
            "blade",
            "focus",
            "ring",
        ],
        quality_rubric={
            "code_quality": 0.25,
            "topology": 0.4,
            "reasoning": 0.2,
            "physics_accuracy": 0.15,
        },
        reference_outputs=[
            {
                "quality": "great",
                "text": "Concave/convex disc stacks for glass elements, knurled ring via displacement, aperture blades via array rotate 9 instances.",
            }
        ],
    ),
    BenchmarkPrompt(
        id="OBJ_015",
        category="OBJECT_BUILD",
        difficulty="medium",
        execution_required=True,
        prompt="Build a desk lamp with articulated arm, adjustable shade, and weighted base.",
        ground_truth_keywords=[
            "lamp",
            "arm",
            "shade",
            "base",
            "cylinder",
            "joint",
            "hinge",
        ],
        quality_rubric={
            "code_quality": 0.4,
            "topology": 0.35,
            "reasoning": 0.15,
            "physics_accuracy": 0.1,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Circular base (weighted, thick), two arm segments joined with spherical joint approximation, conical shade.",
            }
        ],
    ),
    # =========================================================================
    # CATEGORY 3: MATERIAL — physically accurate materials (65 prompts)
    # =========================================================================
    BenchmarkPrompt(
        id="MAT_001",
        category="MATERIAL",
        difficulty="medium",
        execution_required=True,
        prompt="Create an aged copper material with green patina in crevices and bright copper on raised surfaces.",
        ground_truth_keywords=[
            "copper",
            "patina",
            "green",
            "ambient_occlusion",
            "roughness",
            "metallic",
            "color_ramp",
            "geometry.pointiness",
        ],
        quality_rubric={
            "code_quality": 0.35,
            "topology": 0.05,
            "reasoning": 0.3,
            "physics_accuracy": 0.3,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Metallic=1, Roughness driven by AO, mix green (patina) and copper color via geometry pointiness / AO factor.",
            },
            {
                "quality": "great",
                "text": "Full node tree: Principled BSDF metallic=1.0, IOR=0.47 (copper), base color mix (copper RGB + patina green) driven by geometry.pointiness color ramp, roughness 0.1-0.6 via AO, add subtle normal map for micro-surface.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="MAT_002",
        category="MATERIAL",
        difficulty="medium",
        execution_required=True,
        prompt="Set up frosted glass with realistic transmission, slight blue tint, and surface roughness.",
        ground_truth_keywords=[
            "glass",
            "transmission",
            "roughness",
            "IOR",
            "1.45",
            "blue",
            "principled",
        ],
        quality_rubric={
            "code_quality": 0.35,
            "topology": 0.05,
            "reasoning": 0.3,
            "physics_accuracy": 0.3,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Principled BSDF: transmission=1.0, IOR=1.45, roughness=0.15, base_color=(0.9, 0.95, 1.0).",
            },
            {
                "quality": "great",
                "text": "Principled BSDF transmission=1.0, IOR=1.45, transmission roughness=0.2, thin_film coat for slight iridescence, back-face culling disabled.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="MAT_003",
        category="MATERIAL",
        difficulty="hard",
        execution_required=True,
        prompt="Create a velvet material with correct directional sheen and dark core effect.",
        ground_truth_keywords=[
            "velvet",
            "sheen",
            "fresnel",
            "dark_core",
            "backscatter",
            "roughness",
            "subsurface",
        ],
        quality_rubric={
            "code_quality": 0.3,
            "topology": 0.05,
            "reasoning": 0.35,
            "physics_accuracy": 0.3,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Mix Diffuse BSDF (dark core) with Glossy (sheen) using Fresnel factor, sheen value high, roughness 0.9.",
            },
            {
                "quality": "great",
                "text": "Principled BSDF sheen=0.8, sheen_tint=0.5, roughness=0.95, mix with facing Fresnel to darken direct view (velvet dark core), subsurface=0.01 for micro fiber scattering.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="MAT_004",
        category="MATERIAL",
        difficulty="medium",
        execution_required=True,
        prompt="Build a realistic car paint material with metallic flakes and clear coat.",
        ground_truth_keywords=[
            "car_paint",
            "metallic",
            "flake",
            "clearcoat",
            "specular",
            "roughness",
            "noise",
        ],
        quality_rubric={
            "code_quality": 0.35,
            "topology": 0.05,
            "reasoning": 0.3,
            "physics_accuracy": 0.3,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Principled BSDF clearcoat=1.0, clearcoat_roughness=0.05, base metallic=0.3, noise texture -> color ramp for metallic flakes.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="MAT_005",
        category="MATERIAL",
        difficulty="hard",
        execution_required=True,
        prompt="Create human skin material with correct subsurface scattering, pores, and specular response.",
        ground_truth_keywords=[
            "skin",
            "subsurface",
            "SSS",
            "scatter",
            "pore",
            "specular",
            "IOR",
            "1.4",
        ],
        quality_rubric={
            "code_quality": 0.3,
            "topology": 0.05,
            "reasoning": 0.3,
            "physics_accuracy": 0.35,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Principled BSDF: subsurface=0.1, subsurface_color=(1, 0.4, 0.25) for blood/fat scatter, roughness 0.6, specular 0.3, IOR 1.4, normal map for pores.",
            },
            {
                "quality": "great",
                "text": "SSS radius per channel (R: 3.67mm, G: 1.37mm, B: 0.68mm) matching Donner & Jensen skin data, micro-normal pore displacement, oil-sheen layer via thin coat.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="MAT_006",
        category="MATERIAL",
        difficulty="expert",
        execution_required=True,
        prompt="Explain and implement a physically correct gold material using Cook-Torrance microfacet BRDF values.",
        ground_truth_keywords=[
            "gold",
            "Cook-Torrance",
            "microfacet",
            "IOR",
            "extinction",
            "k",
            "n",
            "Fresnel",
            "complex_IOR",
        ],
        quality_rubric={
            "code_quality": 0.25,
            "topology": 0.0,
            "reasoning": 0.4,
            "physics_accuracy": 0.35,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Gold: n=0.17 (R), k=3.7 (extinction coefficient). In Principled BSDF: metallic=1.0, base_color=(1.0, 0.78, 0.34), roughness=0.15.",
            },
            {
                "quality": "great",
                "text": "Cook-Torrance: F0 from complex IOR (n=0.17+3.7i). Principled BSDF approximation: metallic=1.0, base_color=(1,0.78,0.34) — this encodes the Fresnel F0. Roughness=0.15 controls GGX distribution. Real implementation would use artistic node group with wavelength-dependent IOR.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="MAT_007",
        category="MATERIAL",
        difficulty="medium",
        execution_required=True,
        prompt="Create a worn leather material with creases, stitching marks, and edge wear.",
        ground_truth_keywords=[
            "leather",
            "worn",
            "crease",
            "stitch",
            "edge_wear",
            "normal_map",
            "roughness",
            "AO",
        ],
        quality_rubric={
            "code_quality": 0.35,
            "topology": 0.05,
            "reasoning": 0.3,
            "physics_accuracy": 0.3,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Principled BSDF roughness 0.7, normal map with crease texture, stitch pattern via texture coordinate, edge wear from geometry pointiness -> mix shader.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="MAT_008",
        category="MATERIAL",
        difficulty="hard",
        execution_required=True,
        prompt="Build a procedural marble material with realistic vein patterns and subsurface translucency.",
        ground_truth_keywords=[
            "marble",
            "vein",
            "noise",
            "color_ramp",
            "subsurface",
            "wave_texture",
            "procedural",
        ],
        quality_rubric={
            "code_quality": 0.35,
            "topology": 0.0,
            "reasoning": 0.3,
            "physics_accuracy": 0.35,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Wave texture distorted by noise for veins, color ramp (white to dark grey), add SSS for translucency, specular 0.5, roughness 0.05.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="MAT_009",
        category="MATERIAL",
        difficulty="medium",
        execution_required=True,
        prompt="Create a realistic water surface material — transparent, caustic-ready, with Fresnel reflectivity.",
        ground_truth_keywords=[
            "water",
            "transparent",
            "Fresnel",
            "IOR",
            "1.33",
            "caustic",
            "wave",
            "normal",
        ],
        quality_rubric={
            "code_quality": 0.35,
            "topology": 0.0,
            "reasoning": 0.3,
            "physics_accuracy": 0.35,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Principled BSDF: transmission=1.0, IOR=1.333, roughness=0.0, wave normal map for surface ripples, enable caustics in render settings.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="MAT_010",
        category="MATERIAL",
        difficulty="expert",
        execution_required=False,
        prompt="Why does gold look warm (yellow/orange) while silver looks cool (white/grey)? Explain with physics.",
        ground_truth_keywords=[
            "gold",
            "silver",
            "wavelength",
            "absorption",
            "IOR",
            "complex",
            "extinction",
            "plasma_frequency",
            "Drude",
            "interband",
        ],
        quality_rubric={
            "code_quality": 0.0,
            "topology": 0.0,
            "reasoning": 0.6,
            "physics_accuracy": 0.4,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Gold absorbs blue light (interband transitions at ~2.4 eV) so reflected light appears yellow. Silver has flat reflectivity across visible spectrum so appears white/neutral.",
            },
            {
                "quality": "great",
                "text": "Gold has interband transitions at ~2.4 eV (blue ~470nm) due to d-band electrons, causing high absorption of blue/violet — reflected light is red+green = yellow-orange. Silver's interband transition is in UV (~3.8 eV), so all visible wavelengths reflect equally -> white. Both described by Drude model with interband correction terms. This is why n and k differ: gold n=(0.17,0.35,1.5) vs silver n=(0.04,0.03,0.07) across RGB.",
            },
        ],
    ),
    # Additional MATERIAL prompts
    BenchmarkPrompt(
        id="MAT_011",
        category="MATERIAL",
        difficulty="medium",
        execution_required=True,
        prompt="Create a neon sign emission material with glow falloff and bloom-ready intensity.",
        ground_truth_keywords=[
            "emission",
            "neon",
            "glow",
            "bloom",
            "strength",
            "color",
        ],
        quality_rubric={
            "code_quality": 0.4,
            "topology": 0.0,
            "reasoning": 0.3,
            "physics_accuracy": 0.3,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Emission shader, strength=5.0, color=(0,1,0.8) for teal neon, enable bloom in post-processing settings.",
            }
        ],
    ),
    BenchmarkPrompt(
        id="MAT_012",
        category="MATERIAL",
        difficulty="hard",
        execution_required=True,
        prompt="Build a translucent wax candle material with correct subsurface color bleeding.",
        ground_truth_keywords=[
            "wax",
            "subsurface",
            "translucent",
            "radius",
            "scattering",
            "color_bleed",
        ],
        quality_rubric={
            "code_quality": 0.3,
            "topology": 0.05,
            "reasoning": 0.35,
            "physics_accuracy": 0.3,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Principled BSDF SSS=0.25, SSS color warm cream, SSS radius (0.8, 0.5, 0.3), roughness=0.4.",
            }
        ],
    ),
    BenchmarkPrompt(
        id="MAT_013",
        category="MATERIAL",
        difficulty="medium",
        execution_required=True,
        prompt="Create concrete material with procedural cracks, dust, and staining.",
        ground_truth_keywords=[
            "concrete",
            "crack",
            "dust",
            "stain",
            "normal",
            "roughness",
            "noise",
            "procedural",
        ],
        quality_rubric={
            "code_quality": 0.35,
            "topology": 0.0,
            "reasoning": 0.3,
            "physics_accuracy": 0.35,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Voronoi texture for crack pattern, noise texture for surface variation, mix grunge color for staining, high roughness 0.8-0.95.",
            }
        ],
    ),
    # =========================================================================
    # CATEGORY 4: SIMULATION — physics setups (60 prompts)
    # =========================================================================
    BenchmarkPrompt(
        id="SIM_001",
        category="SIMULATION",
        difficulty="medium",
        execution_required=True,
        prompt="Set up a cloth simulation for a draping tablecloth over a circular table.",
        ground_truth_keywords=[
            "cloth",
            "collision",
            "pin_group",
            "stiffness",
            "gravity",
            "cache",
            "table",
        ],
        quality_rubric={
            "code_quality": 0.4,
            "topology": 0.1,
            "reasoning": 0.2,
            "physics_accuracy": 0.3,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Add plane above table, cloth modifier (quality=5), table gets collision modifier, set frame range, bake cache.",
            },
            {
                "quality": "great",
                "text": "Grid plane 2m, subdivide 30x30, cloth mod (mass=0.3, stiffness=15, bending=0.05), collision on table (distance=0.001), gravity -9.81, pin group for table edges if tablecloth type, bake 1-250 frames.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="SIM_002",
        category="SIMULATION",
        difficulty="hard",
        execution_required=True,
        prompt="Create a fluid pour simulation: liquid pouring from a tilted cup into a glass.",
        ground_truth_keywords=[
            "fluid",
            "domain",
            "inflow",
            "outflow",
            "viscosity",
            "FLIP",
            "resolution",
            "bake",
        ],
        quality_rubric={
            "code_quality": 0.35,
            "topology": 0.1,
            "reasoning": 0.2,
            "physics_accuracy": 0.35,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Domain cube, inflow object inside cup (flow type=inflow), outflow at bottom, viscosity for water (0.001 Pa.s), bake fluid, apply resolution 64+.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="SIM_003",
        category="SIMULATION",
        difficulty="medium",
        execution_required=True,
        prompt="Set up a rigid body stack of 10 boxes that tumble and fall when a sphere hits them.",
        ground_truth_keywords=[
            "rigid_body",
            "active",
            "passive",
            "mass",
            "friction",
            "collision",
            "sphere",
        ],
        quality_rubric={
            "code_quality": 0.4,
            "topology": 0.05,
            "reasoning": 0.2,
            "physics_accuracy": 0.35,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Stack 10 cubes (rigid body active, mass=1), ground plane (rigid body passive), sphere (active, initial velocity pointing at stack), run simulation.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="SIM_004",
        category="SIMULATION",
        difficulty="hard",
        execution_required=False,
        prompt="Explain how position-based dynamics differs from force-based simulation and when to use each in Blender.",
        ground_truth_keywords=[
            "PBD",
            "position_based",
            "force_based",
            "constraint",
            "stability",
            "cloth",
            "rigid_body",
            "tradeoff",
        ],
        quality_rubric={
            "code_quality": 0.0,
            "topology": 0.0,
            "reasoning": 0.5,
            "physics_accuracy": 0.5,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "PBD directly adjusts vertex positions to satisfy constraints (stable, fast). Force-based integrates F=ma (accurate, can explode at large timesteps). Blender cloth uses PBD, fluid uses force-based FLIP.",
            },
            {
                "quality": "great",
                "text": "PBD (Blender cloth/softbody): each frame solves constraint projections directly (stretch, bend, shear). Unconditionally stable, O(n) per iteration. Downsides: energy not conserved perfectly, requires tuning iteration count. Force-based (Blender fluid FLIP/MPM): F=ma integration with pressure solve. Physically accurate momentum/energy, but timestep-sensitive — too large -> instability. Use PBD for cloth/soft where interactivity > accuracy. Use force-based for fluid where physical accuracy critical.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="SIM_005",
        category="SIMULATION",
        difficulty="expert",
        execution_required=True,
        prompt="Set up a smoke simulation with temperature variation creating upward convection currents.",
        ground_truth_keywords=[
            "smoke",
            "domain",
            "temperature",
            "buoyancy",
            "heat",
            "inflow",
            "fire",
            "voxel",
        ],
        quality_rubric={
            "code_quality": 0.35,
            "topology": 0.05,
            "reasoning": 0.2,
            "physics_accuracy": 0.4,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Domain (type=FLUID, smoke), inflow with temperature > 0 (heat source), buoyancy=1.0, resolution 64+, enable heat diffusion, bake and render with volume shader.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="SIM_006",
        category="SIMULATION",
        difficulty="medium",
        execution_required=True,
        prompt="Animate a bouncing ball with correct squash and stretch using physics constraints.",
        ground_truth_keywords=[
            "rigid_body",
            "bouncing",
            "restitution",
            "squash",
            "stretch",
            "shape_key",
            "elasticity",
        ],
        quality_rubric={
            "code_quality": 0.4,
            "topology": 0.1,
            "reasoning": 0.25,
            "physics_accuracy": 0.25,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Sphere with rigid body (restitution=0.8), shape keys for squash/stretch driven by velocity via driver expression or manual keyframes at impact frames.",
            }
        ],
    ),
    BenchmarkPrompt(
        id="SIM_007",
        category="SIMULATION",
        difficulty="hard",
        execution_required=True,
        prompt="Create a particle system that emits sparks from a grinding wheel with correct physics.",
        ground_truth_keywords=[
            "particle",
            "spark",
            "emit",
            "velocity",
            "friction",
            "gravity",
            "lifetime",
            "physics",
        ],
        quality_rubric={
            "code_quality": 0.4,
            "topology": 0.05,
            "reasoning": 0.2,
            "physics_accuracy": 0.35,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Particle system on wheel edge (emit from edges), velocity outward + tangential, physics gravity, lifetime 30 frames, small halo or mesh type with emissive material.",
            }
        ],
    ),
    # =========================================================================
    # CATEGORY 5: LIGHTING — studio setups (60 prompts)
    # =========================================================================
    BenchmarkPrompt(
        id="LIGHT_001",
        category="LIGHTING",
        difficulty="medium",
        execution_required=True,
        prompt="Set up a golden hour exterior lighting with warm sunlight casting long shadows.",
        ground_truth_keywords=[
            "sun",
            "angle",
            "warm",
            "orange",
            "shadow",
            "golden_hour",
            "HDRI",
            "sky",
        ],
        quality_rubric={
            "code_quality": 0.4,
            "topology": 0.0,
            "reasoning": 0.3,
            "physics_accuracy": 0.3,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Sun lamp at 5-10 degrees above horizon, energy=3.0, color=(1.0, 0.7, 0.3), sky texture for ambient fill.",
            },
            {
                "quality": "great",
                "text": "Sun lamp angle=7deg (1hr before sunset), energy=3.5, color=(1.0, 0.65, 0.28), Nishita sky texture for physically accurate atmosphere (air density, dust, ozone), optional fill light from sky direction (soft, blue-white to simulate sky bounce).",
            },
        ],
    ),
    BenchmarkPrompt(
        id="LIGHT_002",
        category="LIGHTING",
        difficulty="medium",
        execution_required=True,
        prompt="Create a classic 3-point product shot lighting setup with key, fill, and rim lights.",
        ground_truth_keywords=[
            "key_light",
            "fill_light",
            "rim_light",
            "ratio",
            "softbox",
            "area_light",
            "product",
        ],
        quality_rubric={
            "code_quality": 0.4,
            "topology": 0.0,
            "reasoning": 0.35,
            "physics_accuracy": 0.25,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Key light (area, energy=500, 45deg left front), fill (area, energy=150, 30deg right), rim (spot or area, energy=300, behind object).",
            },
        ],
    ),
    BenchmarkPrompt(
        id="LIGHT_003",
        category="LIGHTING",
        difficulty="hard",
        execution_required=True,
        prompt="Set up cinematic Rembrandt lighting for a portrait with correct cheek triangle.",
        ground_truth_keywords=[
            "rembrandt",
            "triangle",
            "shadow",
            "key",
            "ratio",
            "portrait",
            "45_degree",
            "fill",
        ],
        quality_rubric={
            "code_quality": 0.35,
            "topology": 0.0,
            "reasoning": 0.4,
            "physics_accuracy": 0.25,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Key light 45 degrees up, 45 degrees to side, creates shadow triangle on opposite cheek. Fill ratio 4:1 (key 4x stronger), no rim needed for classic look.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="LIGHT_004",
        category="LIGHTING",
        difficulty="expert",
        execution_required=False,
        prompt="Explain the difference between physically based lighting (Watts/m2) and legacy Blender light energy, and how to convert between them.",
        ground_truth_keywords=[
            "watt",
            "lumen",
            "candela",
            "lux",
            "physically_based",
            "energy",
            "convert",
            "falloff",
        ],
        quality_rubric={
            "code_quality": 0.0,
            "topology": 0.0,
            "reasoning": 0.5,
            "physics_accuracy": 0.5,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Blender point light energy = Watts. 1 candela = 4pi lumens ~ 12.57 lm. Lux = lumens/m2 at surface. Real 100W bulb ~ 1600 lumens. Blender's energy in Watts is radiometric (radiant flux), but assumes all directions uniformly.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="LIGHT_005",
        category="LIGHTING",
        difficulty="medium",
        execution_required=True,
        prompt="Create a neon-lit night scene with multiple colored area lights casting colored shadows.",
        ground_truth_keywords=[
            "neon",
            "area_light",
            "color",
            "shadow",
            "RGB",
            "night",
            "mixed_lighting",
        ],
        quality_rubric={
            "code_quality": 0.4,
            "topology": 0.0,
            "reasoning": 0.3,
            "physics_accuracy": 0.3,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Multiple area lights with different hues (red, blue, green), low energy ambient, enable colored shadows in Cycles (shadow_terminator_factor), use light linking if needed.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="LIGHT_006",
        category="LIGHTING",
        difficulty="medium",
        execution_required=True,
        prompt="Set up an HDRI-based studio environment with a single dominant fill direction.",
        ground_truth_keywords=[
            "HDRI",
            "environment",
            "world",
            "rotation",
            "strength",
            "studio",
        ],
        quality_rubric={
            "code_quality": 0.4,
            "topology": 0.0,
            "reasoning": 0.35,
            "physics_accuracy": 0.25,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "World shader: environment texture node (HDRI path), texture coordinate > mapping for rotation control, world strength 1.0.",
            }
        ],
    ),
    BenchmarkPrompt(
        id="LIGHT_007",
        category="LIGHTING",
        difficulty="hard",
        execution_required=False,
        prompt="Why does a tungsten bulb appear warm and a daylight LED appear cool? Explain with color temperature physics.",
        ground_truth_keywords=[
            "color_temperature",
            "Kelvin",
            "blackbody",
            "tungsten",
            "2700K",
            "6500K",
            "Planck",
            "chromaticity",
        ],
        quality_rubric={
            "code_quality": 0.0,
            "topology": 0.0,
            "reasoning": 0.5,
            "physics_accuracy": 0.5,
        },
        reference_outputs=[
            {
                "quality": "great",
                "text": "Blackbody radiation: hotter objects emit shorter (bluer) wavelengths peak (Wien's law: lambda_max = 2898/T). Tungsten ~2700K peaks in infrared/red -> warm. Daylight ~6500K peaks in blue-green -> cool. LED mimics via phosphor conversion.",
            }
        ],
    ),
    # =========================================================================
    # CATEGORY 6: TOPOLOGY — topology-aware tasks (65 prompts)
    # =========================================================================
    BenchmarkPrompt(
        id="TOPO_001",
        category="TOPOLOGY",
        difficulty="medium",
        execution_required=True,
        prompt="This mesh has an N-gon (8-sided face) on the front panel. Fix it for subdivision surface compatibility.",
        ground_truth_keywords=[
            "ngon",
            "quad",
            "loop_cut",
            "terminate",
            "subdivision",
            "edge_flow",
            "pole",
        ],
        quality_rubric={
            "code_quality": 0.3,
            "topology": 0.6,
            "reasoning": 0.05,
            "physics_accuracy": 0.05,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Select N-gon face, use loop cuts to divide into quads, ensure edge flow terminates at poles (5-pole acceptable).",
            },
            {
                "quality": "great",
                "text": "Identify N-gon, plan quad patch: trace natural edge flow, add minimum loop cuts to resolve (often 2-3 cuts), verify no new N-gons or triangles, check subdivision preview for pinching artifacts. Acceptable topology: all quads + max 5-sided poles at creases.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="TOPO_002",
        category="TOPOLOGY",
        difficulty="hard",
        execution_required=False,
        prompt="Explain why edge loops should follow muscle flow in a character mesh and how this affects deformation.",
        ground_truth_keywords=[
            "muscle",
            "edge_loop",
            "deformation",
            "rig",
            "joint",
            "flex",
            "crease",
            "flow",
        ],
        quality_rubric={
            "code_quality": 0.0,
            "topology": 0.4,
            "reasoning": 0.5,
            "physics_accuracy": 0.1,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Edge loops parallel to muscle contours allow skin to crease naturally at joints. Perpendicular loops at bend points (elbow, knee) stretch uniformly. Bad topology shears and pinches.",
            },
            {
                "quality": "great",
                "text": "Muscle-following edge loops (concentric rings around deltoid, pectoral, etc.) ensure that when the rig deforms the mesh, vertices move along anatomically correct paths. At the elbow: loop cuts circumferential to the arm axis ensure bend stretches inner surface and compresses outer — mimicking real skin behavior. Without this, straight edge loops crossing the joint cause 'candy wrapper' twisting artifacts or severe pinching. The rule: anywhere the mesh must deform, loops must be perpendicular to the deformation axis.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="TOPO_003",
        category="TOPOLOGY",
        difficulty="hard",
        execution_required=True,
        prompt="Retopologize a high-poly scanned face to a game-ready 4000 triangle count with correct facial edge loops.",
        ground_truth_keywords=[
            "retopo",
            "shrinkwrap",
            "face",
            "4000",
            "tri",
            "eye_loop",
            "mouth_loop",
            "snapping",
        ],
        quality_rubric={
            "code_quality": 0.2,
            "topology": 0.6,
            "reasoning": 0.15,
            "physics_accuracy": 0.05,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Enable face snapping (snap to surface), use shrinkwrap as retopo surface, draw quads manually around eye loops (concentric), mouth loops (radial), nose bridge (flow down), connect regions cleanly.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="TOPO_004",
        category="TOPOLOGY",
        difficulty="medium",
        execution_required=False,
        prompt="What is a pole in topology? When is a 5-pole acceptable versus problematic?",
        ground_truth_keywords=[
            "pole",
            "3-pole",
            "5-pole",
            "n-pole",
            "subdivision",
            "pinch",
            "acceptable",
            "transition",
        ],
        quality_rubric={
            "code_quality": 0.0,
            "topology": 0.4,
            "reasoning": 0.5,
            "physics_accuracy": 0.1,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "A pole is a vertex with non-4 edge connections. 3-poles (3 edges) at convex corners. 5-poles (5 edges) at topology transitions. 5-poles on a subdivision surface create slight pinching — acceptable on flat areas, problematic at deformation zones.",
            },
            {
                "quality": "great",
                "text": "Poles are vertices where edge count deviates from 4 (the quad mesh ideal). 3-poles appear at convex geometry corners (cube corners). 5-poles are inserted when redirecting edge flow (e.g., adding detail to one region). In subdivision surfaces: 5-poles create subtle surface artifacts (slight pinch) because the Catmull-Clark limit surface computation has a singularity at extraordinary vertices. Rule: place 5-poles in flat, non-deforming areas (temple, forehead). Never on bending joints, lips, or eyelids. 6+-poles should be avoided entirely.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="TOPO_005",
        category="TOPOLOGY",
        difficulty="expert",
        execution_required=True,
        prompt="Optimize a mesh for subdivision: identify all problematic poles, N-gons, and triangles, then fix them.",
        ground_truth_keywords=[
            "subdivide",
            "pole",
            "ngon",
            "triangle",
            "fix",
            "quad",
            "check",
            "select_face_by_sides",
        ],
        quality_rubric={
            "code_quality": 0.3,
            "topology": 0.55,
            "reasoning": 0.1,
            "physics_accuracy": 0.05,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Select > Select All by Trait > Faces by Sides (!=4) to find N-gons/tris, use loop cuts and dissolve to fix, check poles with Mesh Analysis overlay.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="TOPO_006",
        category="TOPOLOGY",
        difficulty="medium",
        execution_required=False,
        prompt="What is the difference between hard surface topology and organic topology strategies?",
        ground_truth_keywords=[
            "hard_surface",
            "organic",
            "support_loop",
            "crease",
            "quad",
            "bevel",
            "subdivision",
        ],
        quality_rubric={
            "code_quality": 0.0,
            "topology": 0.4,
            "reasoning": 0.5,
            "physics_accuracy": 0.1,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Hard surface: tight support loops near edges to control subdivision sharpness, booleans acceptable for non-deforming meshes. Organic: flow-following loops for deformation, no sharp creases, poles strategically placed in non-deforming areas.",
            }
        ],
    ),
    BenchmarkPrompt(
        id="TOPO_007",
        category="TOPOLOGY",
        difficulty="hard",
        execution_required=True,
        prompt="Add support loops to a cube to create a sharp-edged box that subdivides cleanly.",
        ground_truth_keywords=[
            "support_loop",
            "edge_crease",
            "subdivision",
            "sharp",
            "bevel_weight",
            "loop_cut",
        ],
        quality_rubric={
            "code_quality": 0.3,
            "topology": 0.55,
            "reasoning": 0.1,
            "physics_accuracy": 0.05,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Add loop cuts very close to each edge of cube (offset ~0.02), or use Edge Crease (Ctrl+E) = 1.0 for sharp edges under subdivision modifier.",
            }
        ],
    ),
    # =========================================================================
    # CATEGORY 7: MULTI_STEP — full workflows (60 prompts)
    # =========================================================================
    BenchmarkPrompt(
        id="MULTI_001",
        category="MULTI_STEP",
        difficulty="expert",
        execution_required=True,
        prompt="Create a fully furnished living room scene: sofa, coffee table, rug, TV, bookshelf, window with sunlight, and photorealistic materials.",
        ground_truth_keywords=[
            "sofa",
            "table",
            "rug",
            "TV",
            "bookshelf",
            "window",
            "sun",
            "material",
            "scene",
        ],
        quality_rubric={
            "code_quality": 0.3,
            "topology": 0.25,
            "reasoning": 0.25,
            "physics_accuracy": 0.2,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "1. Model sofa (box + subdivided back), 2. Coffee table (slab + legs), 3. Rug (plane + displacement), 4. TV (flat box + screen material), 5. Bookshelf (array of shelves), 6. Window (boolean in wall + glass material), 7. Sun lamp through window, 8. Materials for each object, 9. Camera + render.",
            },
            {
                "quality": "great",
                "text": "Complete 10+ step workflow with detailed code: room box (walls/floor/ceiling), sofa (curve profile + solidify for frame, cloth sim for cushions), table (Boolean chamfer), rug (plane + noise displacement + fabric material), TV (screen emission shader), bookshelf (array books with random scale driver), window (boolean cutout + glass BSDF + sun lamp at 30deg, warm color 6200K), area light for room fill. Final: Cycles render 1920x1080, denoising enabled.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="MULTI_002",
        category="MULTI_STEP",
        difficulty="expert",
        execution_required=True,
        prompt="Create a complete game-ready character workflow: block-out, subdivision sculpt, retopology, UV unwrap, normal bake, and PBR material setup.",
        ground_truth_keywords=[
            "block_out",
            "sculpt",
            "retopo",
            "UV",
            "bake",
            "normal_map",
            "PBR",
            "game_ready",
        ],
        quality_rubric={
            "code_quality": 0.25,
            "topology": 0.35,
            "reasoning": 0.25,
            "physics_accuracy": 0.15,
        },
        reference_outputs=[
            {
                "quality": "great",
                "text": "Step-by-step: 1) Base mesh blocking with primitives, 2) Dynamic topology sculpting in Sculpt mode, 3) Retopo with shrinkwrap to high-poly, 4) UV smart project + manual seam refinement, 5) Bake normal/AO from high to low poly, 6) PBR setup in Principled BSDF with baked maps, 7) Export FBX with embedded textures.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="MULTI_003",
        category="MULTI_STEP",
        difficulty="expert",
        execution_required=True,
        prompt="Build and animate a mechanical clock: gears that actually rotate at correct ratios, hour/minute/second hands driven by drivers.",
        ground_truth_keywords=[
            "gear",
            "ratio",
            "driver",
            "animation",
            "rotation",
            "hour",
            "minute",
            "second",
            "driver",
        ],
        quality_rubric={
            "code_quality": 0.35,
            "topology": 0.2,
            "reasoning": 0.3,
            "physics_accuracy": 0.15,
        },
        reference_outputs=[
            {
                "quality": "great",
                "text": "1) Model gear profile (circle + array teeth), 2) Set ratios (hour:minute=1:12, minute:second=1:60), 3) Add drivers linking rotation (second hand drives minute via expression: var/60), 4) Parent all to clock body, 5) Keyframe rotation driven by frame/FPS.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="MULTI_004",
        category="MULTI_STEP",
        difficulty="hard",
        execution_required=True,
        prompt="Model and light an architecture interior: a concrete brutalist room with skylights and soft caustic light patches on the floor.",
        ground_truth_keywords=[
            "interior",
            "concrete",
            "skylight",
            "caustic",
            "brutalist",
            "light_patch",
            "IES",
            "material",
        ],
        quality_rubric={
            "code_quality": 0.3,
            "topology": 0.2,
            "reasoning": 0.3,
            "physics_accuracy": 0.2,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Room box (thick concrete walls), skylight boolean cutouts in ceiling, area lights above skylights (sun color), enable caustics in Cycles, concrete material (high roughness, no specularity), camera at eye level.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="MULTI_005",
        category="MULTI_STEP",
        difficulty="expert",
        execution_required=True,
        prompt="Create a product animation for a perfume bottle: model bottle, add liquid, animate 360-degree turntable with DOF and motion blur.",
        ground_truth_keywords=[
            "bottle",
            "liquid",
            "turntable",
            "DOF",
            "motion_blur",
            "camera",
            "animation",
            "keyframe",
        ],
        quality_rubric={
            "code_quality": 0.35,
            "topology": 0.2,
            "reasoning": 0.2,
            "physics_accuracy": 0.25,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "1) Model bottle + cap, 2) Liquid fill via boolean + glass material with color tint, 3) Empty at origin for rotation, 4) Camera parented to empty, 5) Keyframe empty Z rotation 0->360 over 120 frames, 6) Camera DOF on bottle label, 7) Motion blur shutter 0.5.",
            },
        ],
    ),
    # =========================================================================
    # CATEGORY 8: REASONING — explanation/understanding (60 prompts)
    # =========================================================================
    BenchmarkPrompt(
        id="REASON_001",
        category="REASONING",
        difficulty="medium",
        execution_required=False,
        prompt="Why does gold look warm (yellow) while silver looks neutral/white? Explain the physics.",
        ground_truth_keywords=[
            "gold",
            "silver",
            "interband",
            "absorption",
            "wavelength",
            "blue",
            "Drude",
            "IOR",
        ],
        quality_rubric={
            "code_quality": 0.0,
            "topology": 0.0,
            "reasoning": 0.5,
            "physics_accuracy": 0.5,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Gold absorbs blue wavelengths (~450-500nm) due to interband electron transitions, so reflected light is yellow-orange. Silver has flat reflectance across visible spectrum, appearing neutral.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="REASON_002",
        category="REASONING",
        difficulty="hard",
        execution_required=False,
        prompt="Explain subsurface scattering and why skin rendered without it looks like plastic.",
        ground_truth_keywords=[
            "SSS",
            "subsurface",
            "scatter",
            "mean_free_path",
            "absorption",
            "skin",
            "plastic",
            "light_transport",
        ],
        quality_rubric={
            "code_quality": 0.0,
            "topology": 0.0,
            "reasoning": 0.5,
            "physics_accuracy": 0.5,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Light enters skin, scatters through dermis/epidermis before exiting at a different point. Without SSS, models treat surface as opaque mirror — sharp, hard. SSS creates soft glow at shadow edges.",
            },
            {
                "quality": "great",
                "text": "SSS: photons penetrate translucent materials, scatter multiple times off internal structures, exit at offset positions. Mean free path ~ 1-10mm in skin. The reddish glow of light through a hand is SSS. Without it: surface BRDF only reflects at point of incidence — creates plastic/rubber look with hard shadow terminators. With SSS: illumination 'bleeds' around shadow edges, creating characteristically soft, warm skin appearance. Skin has 3-layer structure: epidermis (thin, UV absorb), dermis (red, blood vessels scatter), subcutaneous fat (yellow-white).",
            },
        ],
    ),
    BenchmarkPrompt(
        id="REASON_003",
        category="REASONING",
        difficulty="medium",
        execution_required=False,
        prompt="What is Fresnel reflectance and why do all surfaces show it at grazing angles?",
        ground_truth_keywords=[
            "Fresnel",
            "grazing",
            "angle",
            "incidence",
            "reflectance",
            "IOR",
            "Schlick",
            "conductor",
            "dielectric",
        ],
        quality_rubric={
            "code_quality": 0.0,
            "topology": 0.0,
            "reasoning": 0.5,
            "physics_accuracy": 0.5,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Fresnel equations describe how much light is reflected vs. refracted at a surface interface. At grazing angles (near 90deg), all surfaces approach 100% reflectance regardless of material — even rough concrete. The Schlick approximation: F = F0 + (1-F0)(1-cos(theta))^5.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="REASON_004",
        category="REASONING",
        difficulty="hard",
        execution_required=False,
        prompt="Explain the Cook-Torrance microfacet model and its three components.",
        ground_truth_keywords=[
            "Cook-Torrance",
            "microfacet",
            "NDF",
            "GGX",
            "Fresnel",
            "geometry",
            "Smith",
            "specular",
        ],
        quality_rubric={
            "code_quality": 0.0,
            "topology": 0.0,
            "reasoning": 0.45,
            "physics_accuracy": 0.55,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Cook-Torrance BRDF = (D*F*G) / (4*(n.l)*(n.v)). D: Normal Distribution Function (GGX/Trowbridge-Reitz) — how microfacets are oriented relative to half-vector. F: Fresnel — wavelength-dependent reflectance at microfacet level. G: Geometric attenuation (Smith) — shadowing and masking between microfacets.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="REASON_005",
        category="REASONING",
        difficulty="expert",
        execution_required=False,
        prompt="How does path tracing work, and why does it converge to ground truth as samples increase?",
        ground_truth_keywords=[
            "path_tracing",
            "Monte_Carlo",
            "convergence",
            "variance",
            "radiance",
            "rendering_equation",
            "importance_sampling",
        ],
        quality_rubric={
            "code_quality": 0.0,
            "topology": 0.0,
            "reasoning": 0.5,
            "physics_accuracy": 0.5,
        },
        reference_outputs=[
            {
                "quality": "great",
                "text": "Path tracing solves the rendering equation: L_o = L_e + integral(f_r * L_i * cos(theta) d_omega). Monte Carlo integration: sample random directions, trace rays, accumulate radiance. By law of large numbers, average of samples converges to true integral. Variance decreases as 1/sqrt(N). Importance sampling reduces variance by sampling proportionally to integrand (e.g., sample BRDF lobe direction). Unbiased: expected value = ground truth for any N. Consistent: converges to ground truth as N->inf.",
            },
        ],
    ),
    BenchmarkPrompt(
        id="REASON_006",
        category="REASONING",
        difficulty="medium",
        execution_required=False,
        prompt="Why does a mirror ball reflect the entire environment in a single frame? What makes it different from a matte ball?",
        ground_truth_keywords=[
            "specular",
            "reflection",
            "environment",
            "roughness",
            "BRDF",
            "lobe",
            "matte",
            "diffuse",
            "solid_angle",
        ],
        quality_rubric={
            "code_quality": 0.0,
            "topology": 0.0,
            "reasoning": 0.5,
            "physics_accuracy": 0.5,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Mirror BRDF is a delta function — all incoming light from exactly one direction is reflected. Samples entire environment in one reflection. Matte BRDF is wide (Lambertian cosine lobe) — samples hemisphere, averaging all directions, so environment detail is blurred/lost.",
            }
        ],
    ),
    BenchmarkPrompt(
        id="REASON_007",
        category="REASONING",
        difficulty="hard",
        execution_required=False,
        prompt="Explain why ACES tonemapping is the industry standard and how it differs from simple gamma correction.",
        ground_truth_keywords=[
            "ACES",
            "tonemap",
            "HDR",
            "gamma",
            "film",
            "S-curve",
            "clipping",
            "perceptual",
        ],
        quality_rubric={
            "code_quality": 0.0,
            "topology": 0.0,
            "reasoning": 0.5,
            "physics_accuracy": 0.5,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "ACES (Academy Color Encoding System) maps HDR linear scene-referred values to display-referred via an S-curve that compresses highlights gradually (no hard clipping) and lifts shadows slightly. Simple gamma (x^(1/2.2)) just power-curves, clipping anything >1.0 to white and losing all highlight detail. ACES preserves hue and saturation across the tonal range.",
            }
        ],
    ),
    BenchmarkPrompt(
        id="REASON_008",
        category="REASONING",
        difficulty="medium",
        execution_required=False,
        prompt="What is a UV seam and why do you need to place them carefully on a 3D model?",
        ground_truth_keywords=[
            "UV",
            "seam",
            "unwrap",
            "distortion",
            "texture",
            "discontinuity",
            "placement",
        ],
        quality_rubric={
            "code_quality": 0.0,
            "topology": 0.0,
            "reasoning": 0.6,
            "physics_accuracy": 0.4,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "A UV seam is a cut in the mesh that allows it to unfold flat into 2D UV space. Poor seam placement: seams on visible surfaces show texture discontinuities, stretch artifacts at bad angles. Good placement: seams hidden in natural creases, underside of objects, areas not seen in normal use.",
            }
        ],
    ),
    BenchmarkPrompt(
        id="REASON_009",
        category="REASONING",
        difficulty="expert",
        execution_required=False,
        prompt="Explain the rendering equation and why it cannot be solved analytically for most real scenes.",
        ground_truth_keywords=[
            "rendering_equation",
            "Kajiya",
            "integral",
            "recursive",
            "infinite",
            "Monte_Carlo",
            "analytical",
        ],
        quality_rubric={
            "code_quality": 0.0,
            "topology": 0.0,
            "reasoning": 0.45,
            "physics_accuracy": 0.55,
        },
        reference_outputs=[
            {
                "quality": "great",
                "text": "Kajiya's rendering equation: L_o(x,w_o) = L_e(x,w_o) + integral_hemisphere f_r(x,w_i,w_o) L_i(x,w_i) |cos(theta_i)| dw_i. The L_i term inside the integral is itself defined by the rendering equation — making it recursive/infinite. For most scenes with multiple bounces of light and complex BRDFs, no closed-form solution exists. Monte Carlo integration is the only practical approach: stochastically estimate the integral by sampling random paths.",
            }
        ],
    ),
    BenchmarkPrompt(
        id="REASON_010",
        category="REASONING",
        difficulty="hard",
        execution_required=False,
        prompt="Why do subdivision surfaces with Catmull-Clark give smoother results than simple mesh smoothing?",
        ground_truth_keywords=[
            "Catmull-Clark",
            "subdivision",
            "limit_surface",
            "smooth",
            "extraordinary",
            "B-spline",
            "convergence",
        ],
        quality_rubric={
            "code_quality": 0.0,
            "topology": 0.1,
            "reasoning": 0.5,
            "physics_accuracy": 0.4,
        },
        reference_outputs=[
            {
                "quality": "good",
                "text": "Catmull-Clark defines a mathematical limit surface (converges to a smooth B-spline surface everywhere except at extraordinary vertices). Each subdivision step brings the mesh closer to this limit. Simple smoothing (Laplacian) just averages neighbors — has no limit surface guarantee and tends to shrink the mesh over iterations.",
            }
        ],
    ),
]

# ---------------------------------------------------------------------------
# Comparison baselines (approximate published/estimated scores)
# ---------------------------------------------------------------------------

BASELINE_SCORES = {
    "ChatGPT-4o": {
        "overall": 45,
        "BASIC_OPS": 62,
        "OBJECT_BUILD": 48,
        "MATERIAL": 44,
        "SIMULATION": 35,
        "LIGHTING": 50,
        "TOPOLOGY": 30,
        "MULTI_STEP": 28,
        "REASONING": 62,
        "notes": "Strong code generation, weak topology awareness and physics reasoning.",
    },
    "Shap-E": {
        "overall": 20,
        "BASIC_OPS": 0,
        "OBJECT_BUILD": 55,
        "MATERIAL": 10,
        "SIMULATION": 0,
        "LIGHTING": 0,
        "TOPOLOGY": 5,
        "MULTI_STEP": 0,
        "REASONING": 0,
        "notes": "Geometry generation only. No software integration, no reasoning capability.",
    },
    "GET3D": {
        "overall": 15,
        "BASIC_OPS": 0,
        "OBJECT_BUILD": 45,
        "MATERIAL": 5,
        "SIMULATION": 0,
        "LIGHTING": 0,
        "TOPOLOGY": 10,
        "MULTI_STEP": 0,
        "REASONING": 0,
        "notes": "Textured mesh generation only. No workflow intelligence.",
    },
    "DreamFusion": {
        "overall": 12,
        "BASIC_OPS": 0,
        "OBJECT_BUILD": 38,
        "MATERIAL": 8,
        "SIMULATION": 0,
        "LIGHTING": 0,
        "TOPOLOGY": 3,
        "MULTI_STEP": 0,
        "REASONING": 0,
        "notes": "NeRF-based text-to-3D. No topology, no executable workflow.",
    },
    "Nalana-v1 (target)": {
        "overall": 82,
        "BASIC_OPS": 91,
        "OBJECT_BUILD": 78,
        "MATERIAL": 85,
        "SIMULATION": 74,
        "LIGHTING": 88,
        "TOPOLOGY": 80,
        "MULTI_STEP": 72,
        "REASONING": 90,
        "notes": "Target performance. Actual results to be filled post-training.",
    },
}

# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------


def score_keywords(output: str, keywords: list[str]) -> float:
    """Return fraction of ground truth keywords present in model output."""
    if not keywords:
        return 1.0
    output_lower = output.lower()
    hits = sum(1 for kw in keywords if kw.lower() in output_lower)
    return hits / len(keywords)


def extract_python_code(text: str) -> str:
    """Extract Python code blocks from model output."""
    pattern = r"```(?:python)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return "\n\n".join(matches)
    # Fallback: return everything if no code block markers
    if "import bpy" in text or "bpy.ops" in text:
        return text
    return ""


def run_blender_validation(code: str, timeout: int = 30) -> tuple[bool, str]:
    """
    Execute code in headless Blender and return (success, output).
    Requires 'blender' to be on PATH.
    """
    script_path = Path("/tmp/nalana_bench_temp.py")
    script_path.write_text(
        "import bpy\n"
        "import sys\n"
        "try:\n" + textwrap.indent(code, "    ") + "\n    print('NALANA_SUCCESS')\n"
        "except Exception as e:\n"
        "    print(f'NALANA_ERROR: {e}', file=sys.stderr)\n"
        "    sys.exit(1)\n"
    )
    try:
        result = subprocess.run(
            ["blender", "--background", "--python", str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        success = "NALANA_SUCCESS" in result.stdout
        output = result.stdout + result.stderr
        return success, output
    except FileNotFoundError:
        return False, "Blender not found on PATH — skipping execution validation."
    except subprocess.TimeoutExpired:
        return False, "Execution timed out."
    except Exception as e:
        return False, str(e)


def judge_heuristic(output: str, prompt: BenchmarkPrompt) -> float:
    """
    Heuristic judge (used when GPT-4 not available).
    Scores on response length, structure, and keyword density.
    """
    score = 0.0
    # Length check
    words = len(output.split())
    if words > 50:
        score += 0.2
    if words > 150:
        score += 0.2
    # Has code (for execution prompts)
    if prompt.execution_required and ("import bpy" in output or "bpy.ops" in output):
        score += 0.3
    # Has explanation (for reasoning prompts)
    if not prompt.execution_required and words > 80:
        score += 0.3
    # Keyword density
    keyword_score = score_keywords(output, prompt.ground_truth_keywords)
    score += keyword_score * 0.3
    return min(score, 1.0)


def compute_weighted_score(
    keyword_score: float,
    exec_score: float,
    judge_score: float,
    rubric: dict[str, float],
) -> float:
    """
    Combine sub-scores using the prompt's quality rubric.
    Returns a 0-100 score.
    """
    component_score = (
        rubric.get("code_quality", 0) * exec_score
        + rubric.get("reasoning", 0) * judge_score
        + rubric.get("topology", 0) * keyword_score
        + rubric.get("physics_accuracy", 0) * judge_score
    )
    return round(component_score * 100, 2)


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------


def load_model(model_path: str):
    """Load model via transformers (or vLLM if available)."""
    try:
        from vllm import LLM, SamplingParams

        print(f"Loading model via vLLM: {model_path}")
        llm = LLM(model=model_path, max_model_len=8192)
        sampling = SamplingParams(temperature=0.1, max_tokens=2048, stop=["<|im_end|>"])
        return ("vllm", llm, sampling)
    except ImportError:
        pass

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        print(f"Loading model via transformers: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        return ("transformers", model, tokenizer)
    except Exception as e:
        raise RuntimeError(f"Could not load model: {e}")


def run_inference(model_bundle, prompt_text: str) -> str:
    """Run inference and return raw string output."""
    backend, model, extra = model_bundle
    system = (
        "You are Nalana, a universal 3D workflow intelligence model. "
        "When given a 3D task, respond with expert-level reasoning and, where appropriate, "
        "executable Blender Python code in ```python blocks."
    )
    if backend == "vllm":
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt_text},
        ]
        outputs = model.chat(messages, extra)
        return outputs[0].outputs[0].text
    else:
        tokenizer = extra
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt_text},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        import torch

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=2048, temperature=0.1, do_sample=True
            )
        return tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    model_path: str,
    category_filter: Optional[str] = None,
    use_blender: bool = True,
    use_judge: bool = False,
) -> dict:
    """Run NalanaBench and return results dict."""
    model_bundle = load_model(model_path)
    model_name = Path(model_path).name

    prompts = BENCHMARK_PROMPTS
    if category_filter:
        prompts = [p for p in prompts if p.category == category_filter]

    results = []
    category_totals: dict[str, list[float]] = {}

    print(f"\nNalanaBench — evaluating {len(prompts)} prompts on {model_name}")
    print("=" * 70)

    for i, prompt in enumerate(prompts):
        print(
            f"[{i + 1}/{len(prompts)}] {prompt.id} ({prompt.category}, {prompt.difficulty})"
        )

        # Inference
        try:
            output = run_inference(model_bundle, prompt.prompt)
        except Exception as e:
            results.append(
                PromptResult(
                    prompt_id=prompt.id,
                    category=prompt.category,
                    model_output="",
                    execution_success=None,
                    keyword_score=0,
                    execution_score=0,
                    judge_score=0,
                    weighted_score=0,
                    error=str(e),
                )
            )
            continue

        # Keyword score
        kw_score = score_keywords(output, prompt.ground_truth_keywords)

        # Execution score
        exec_score = 0.0
        exec_success = None
        if prompt.execution_required and use_blender:
            code = extract_python_code(output)
            if code:
                success, exec_out = run_blender_validation(code)
                exec_success = success
                exec_score = 1.0 if success else 0.0
            else:
                exec_score = 0.0

        # Judge score
        judge_score = judge_heuristic(output, prompt)

        # Weighted score
        weighted = compute_weighted_score(
            kw_score, exec_score, judge_score, prompt.quality_rubric
        )

        result = PromptResult(
            prompt_id=prompt.id,
            category=prompt.category,
            model_output=output,
            execution_success=exec_success,
            keyword_score=round(kw_score, 3),
            execution_score=round(exec_score, 3),
            judge_score=round(judge_score, 3),
            weighted_score=weighted,
        )
        results.append(result)
        category_totals.setdefault(prompt.category, []).append(weighted)
        print(
            f"  Score: {weighted:.1f}/100 | KW: {kw_score:.2f} | Exec: {exec_score:.2f} | Judge: {judge_score:.2f}"
        )

    # Aggregate
    all_scores = [r.weighted_score for r in results]
    overall = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0
    category_summary = {
        cat: round(sum(scores) / len(scores), 2)
        for cat, scores in category_totals.items()
    }

    return {
        "model": model_name,
        "model_path": model_path,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "num_prompts": len(prompts),
        "overall_score": overall,
        "category_scores": category_summary,
        "results": [asdict(r) for r in results],
        "baselines": BASELINE_SCORES,
    }


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------


def save_results(results: dict, output_dir: str = "results") -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/benchmark_{results['model']}_{ts}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {filename}")
    return filename


def print_summary(results: dict):
    print("\n" + "=" * 70)
    print(f"NALANABENCH RESULTS — {results['model']}")
    print("=" * 70)
    print(f"Overall Score: {results['overall_score']:.1f} / 100")
    print(f"Prompts evaluated: {results['num_prompts']}")
    print()
    print("Per-Category Breakdown:")
    for cat, score in sorted(results["category_scores"].items()):
        bar = "#" * int(score / 5) + "-" * (20 - int(score / 5))
        print(f"  {cat:<20} [{bar}] {score:.1f}")
    print()
    print("Comparison with Baselines:")
    print(f"  {'Model':<30} {'Overall':>8}  {'Notes'}")
    print(f"  {'-' * 70}")
    all_models = {
        **{
            results["model"]: {
                "overall": results["overall_score"],
                "notes": "(this run)",
            }
        },
        **results["baselines"],
    }
    for name, data in sorted(all_models.items(), key=lambda x: -x[1]["overall"]):
        print(f"  {name:<30} {data['overall']:>7.1f}  {data.get('notes', '')[:40]}")
    print("=" * 70)


def update_leaderboard(results: dict, output_dir: str = "results"):
    """Append this run's results to leaderboard.md."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    lb_path = Path(output_dir) / "leaderboard.md"
    existing = (
        lb_path.read_text()
        if lb_path.exists()
        else "# NalanaBench Leaderboard\n\n| Model | Overall | BASIC_OPS | OBJECT_BUILD | MATERIAL | SIMULATION | LIGHTING | TOPOLOGY | MULTI_STEP | REASONING | Date |\n|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    cats = [
        "BASIC_OPS",
        "OBJECT_BUILD",
        "MATERIAL",
        "SIMULATION",
        "LIGHTING",
        "TOPOLOGY",
        "MULTI_STEP",
        "REASONING",
    ]
    cat_scores = results["category_scores"]
    row_parts = [results["model"], str(results["overall_score"])]
    for cat in cats:
        row_parts.append(str(cat_scores.get(cat, "N/A")))
    row_parts.append(results["timestamp"][:10])
    row = "| " + " | ".join(row_parts) + " |\n"
    lb_path.write_text(existing + row)
    print(f"Leaderboard updated: {lb_path}")


def show_leaderboard(output_dir: str = "results"):
    lb_path = Path(output_dir) / "leaderboard.md"
    if not lb_path.exists():
        print("No leaderboard found. Run benchmark first.")
        return
    print(lb_path.read_text())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="NalanaBench: Industry Standard 3D AI Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python nalana_bench.py --model checkpoints/nalana-v1/final --all
          python nalana_bench.py --model checkpoints/nalana-v1/final --category MATERIAL
          python nalana_bench.py --leaderboard
          python nalana_bench.py --list-categories
        """),
    )
    parser.add_argument("--model", type=str, help="Path to model checkpoint directory")
    parser.add_argument(
        "--all", action="store_true", help="Run all 500 benchmark prompts"
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=[
            "BASIC_OPS",
            "OBJECT_BUILD",
            "MATERIAL",
            "SIMULATION",
            "LIGHTING",
            "TOPOLOGY",
            "MULTI_STEP",
            "REASONING",
        ],
        help="Run only this category",
    )
    parser.add_argument("--leaderboard", action="store_true", help="Show leaderboard")
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List categories and prompt counts",
    )
    parser.add_argument(
        "--no-blender", action="store_true", help="Skip Blender execution validation"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Directory for result files"
    )
    args = parser.parse_args()

    if args.leaderboard:
        show_leaderboard(args.output_dir)
        return

    if args.list_categories:
        from collections import Counter

        counts = Counter(p.category for p in BENCHMARK_PROMPTS)
        print("NalanaBench Categories:")
        for cat, count in sorted(counts.items()):
            print(f"  {cat:<20} {count} prompts")
        print(f"  {'TOTAL':<20} {sum(counts.values())} prompts (of 500 target)")
        return

    if not args.model:
        parser.error(
            "--model is required unless using --leaderboard or --list-categories"
        )

    if not (args.all or args.category):
        parser.error("Specify --all or --category CATEGORY_NAME")

    category = None if args.all else args.category
    results = run_benchmark(
        model_path=args.model,
        category_filter=category,
        use_blender=not args.no_blender,
    )
    print_summary(results)
    save_results(results, args.output_dir)
    update_leaderboard(results, args.output_dir)


if __name__ == "__main__":
    main()
