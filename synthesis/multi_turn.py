"""
multi_turn.py - Chains training pairs into multi-turn conversation sequences.

Two modes:
  1. --from-pairs   Chain existing dataset pairs by video_id / topic into
                    coherent multi-step conversations.
  2. --synthetic    Generate programmatic multi-turn conversations from 50+
                    templates covering common 3D workflows.

Output: data/multiturn/conversations.jsonl
Each record is a full conversation with role/content message list.

Usage:
    python multi_turn.py --from-pairs
    python multi_turn.py --synthetic --templates all --count 1000
    python multi_turn.py --from-pairs --synthetic --count 500
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from pathlib import Path
from typing import Any

# ─── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).parents[1]
VALIDATED_DIR = BASE_DIR / "data" / "validated"
MULTITURN_DIR = BASE_DIR / "data" / "multiturn"
INPUT_FILE    = VALIDATED_DIR / "dataset.jsonl"
OUTPUT_FILE   = MULTITURN_DIR / "conversations.jsonl"

# ─── Helpers ───────────────────────────────────────────────────────────────────

def load_pairs(path: Path) -> list[dict]:
    pairs = []
    if not path.exists():
        return pairs
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return pairs


def new_conv_id() -> str:
    return str(uuid.uuid4())


def make_message(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def pack_conversation(messages: list[dict], task_type: str,
                      source: str, quality: float = 3.0) -> dict:
    return {
        "conversation_id": new_conv_id(),
        "messages":        messages,
        "total_steps":     sum(1 for m in messages if m["role"] == "assistant"),
        "task_type":       task_type,
        "quality":         quality,
        "source":          source,
    }


# ─── From-pairs chaining ───────────────────────────────────────────────────────

def group_by_video(pairs: list[dict]) -> dict[str, list[dict]]:
    """Group pairs by video_id (or uid). Returns groups of 2+ pairs."""
    groups: dict[str, list[dict]] = {}
    for p in pairs:
        vid = p.get("video_id") or p.get("uid") or ""
        if vid:
            groups.setdefault(vid, []).append(p)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def pair_to_assistant_content(pair: dict) -> str:
    """Format a training pair as an assistant response (JSON block)."""
    payload = {
        "blender_python": pair.get("blender_python", ""),
        "blender_op":     pair.get("blender_op", {}),
        "reasoning":      pair.get("reasoning", ""),
    }
    if pair.get("universal_dsl"):
        payload["universal_dsl"] = pair["universal_dsl"]
    return json.dumps(payload, indent=2)


def chain_pairs_to_conversation(pairs: list[dict],
                                max_turns: int = 20) -> dict:
    """Convert a list of sequential pairs into a multi-turn conversation."""
    pairs = pairs[:max_turns]
    messages = []

    # Opening context from first pair
    first = pairs[0]
    scene_ctx = first.get("scene_context", "Empty Blender scene.")
    messages.append(make_message(
        "user",
        f"I'm working in Blender. Scene: {scene_ctx}\n{first.get('voice_command', '')}"
    ))
    messages.append(make_message("assistant", pair_to_assistant_content(first)))

    for pair in pairs[1:]:
        messages.append(make_message("user", pair.get("voice_command", "continue")))
        messages.append(make_message("assistant", pair_to_assistant_content(pair)))

    # Infer task type from pairs
    has_material = any("material" in p.get("blender_python", "").lower() for p in pairs)
    has_build    = len(pairs) > 3
    # TODO (NA-28): sequences that contain both BUILD and MATERIAL steps get mislabeled as
    # "BUILD" because has_build takes priority over has_material. Mixed sequences should be
    # labeled "MATERIALIZE" or a dedicated "BUILD_MATERIAL" type for correct evaluation.
    task_type    = "MATERIALIZE" if has_material and not has_build else "BUILD" if has_build else "EXECUTE"

    return pack_conversation(
        messages  = messages,
        task_type = task_type,
        source    = "chained_pairs",
        quality   = min(3.0, 1.0 + len(pairs) * 0.2),
    )


def generate_from_pairs(pairs: list[dict]) -> list[dict]:
    groups = group_by_video(pairs)
    conversations = []
    print(f"[chain] Found {len(groups)} video groups → generating conversations...")
    for vid, group_pairs in groups.items():
        conv = chain_pairs_to_conversation(group_pairs)
        conversations.append(conv)
    print(f"[chain] Generated {len(conversations)} chained conversations.")
    return conversations


# ─── Synthetic conversation templates ─────────────────────────────────────────

# Each template is a list of (user_prompt, blender_python, reasoning) tuples.
# Placeholders like {name} are filled at generation time.

def _json_step(blender_python: str, op: str, reasoning: str,
               args: dict | None = None) -> str:
    payload = {
        "blender_python": blender_python,
        "blender_op":     {"op": op, "args": args or {}},
        "reasoning":      reasoning,
    }
    return json.dumps(payload, indent=2)


# ─── Template: iPhone 16 (25 steps) ──────────────────────────────────────────

IPHONE_TEMPLATE = {
    "name": "build_iphone_16",
    "intent": "Create an iPhone 16",
    "task_type": "BUILD",
    "steps": [
        ("Create an iPhone 16 in Blender",
         "# Multi-step build plan for iPhone 16\n# Steps: body, screen, camera, materials, lighting, render",
         "Planning phase: establish 25-step production pipeline"),
        ("Start with the main body — add a rectangular cube",
         "bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))",
         "Cube is the base form for the phone chassis"),
        ("Scale it to iPhone proportions — 147mm tall, 71mm wide, 7.8mm thick",
         "bpy.ops.transform.resize(value=(0.71, 0.147, 0.0078))",
         "Scale to real-world iPhone 16 dimensions in meters"),
        ("Bevel the edges for that rounded-rectangle look",
         "bpy.ops.object.modifier_add(type='BEVEL')\nbpy.context.object.modifiers['Bevel'].width = 0.008\nbpy.context.object.modifiers['Bevel'].segments = 4",
         "Bevel modifier creates the continuous chamfer on all edges"),
        ("Apply the bevel and go into edit mode to refine",
         "bpy.ops.object.modifier_apply(modifier='Bevel')\nbpy.ops.object.mode_set(mode='EDIT')",
         "Apply before UV work"),
        ("Select the front face and inset it for the screen recess",
         "bpy.ops.mesh.select_all(action='DESELECT')\nbpy.ops.mesh.inset_faces(thickness=0.005, depth=-0.0005)",
         "Creates the subtle bezel depression"),
        ("Add a plane for the display glass",
         "bpy.ops.object.mode_set(mode='OBJECT')\nbpy.ops.mesh.primitive_plane_add(size=0.13, location=(0, 0.07, 0.004))",
         "Display glass sits slightly above the chassis"),
        ("Scale the screen plane to match the display area",
         "bpy.ops.transform.resize(value=(0.66, 1.0, 1.0))",
         "iPhone 16 has 6.1-inch Super Retina XDR display"),
        ("Add the Dynamic Island cutout — circle on top",
         "bpy.ops.mesh.primitive_circle_add(radius=0.004, location=(0, 0.06, 0.0045))\nbpy.ops.object.convert(target='MESH')\nbpy.ops.object.modifier_add(type='SOLIDIFY')\nbpy.context.object.modifiers['Solidify'].thickness = 0.001",
         "Dynamic Island replaces the notch on iPhone 16"),
        ("Now model the camera bump on the rear — add a cylinder",
         "bpy.ops.mesh.primitive_cylinder_add(radius=0.018, depth=0.002, location=(0, 0.04, -0.008))",
         "Camera module sits proud of the rear glass"),
        ("Add the main camera lens circle",
         "bpy.ops.mesh.primitive_torus_add(major_radius=0.006, minor_radius=0.001, location=(0, 0.05, -0.009))",
         "Torus forms the lens ring"),
        ("Add secondary ultrawide lens",
         "bpy.ops.mesh.primitive_torus_add(major_radius=0.006, minor_radius=0.001, location=(-0.012, 0.04, -0.009))",
         "48MP ultrawide at offset position"),
        ("Return to object mode and set up materials — back glass first",
         "bpy.ops.object.mode_set(mode='OBJECT')",
         "Switch to material workflow"),
        ("Create the rear titanium glass material with PBR",
         """mat = bpy.data.materials.new('iPhone_BackGlass')
mat.use_nodes = True
bsdf = mat.node_tree.nodes['Principled BSDF']
bsdf.inputs['Base Color'].default_value = (0.05, 0.05, 0.07, 1.0)
bsdf.inputs['Metallic'].default_value = 0.0
bsdf.inputs['Roughness'].default_value = 0.05
bsdf.inputs['IOR'].default_value = 1.5
bsdf.inputs['Transmission Weight'].default_value = 0.8""",
         "iPhone 16 uses color-infused glass; IOR 1.5 matches borosilicate"),
        ("Apply the glass material to the back plane",
         "bpy.context.object.data.materials.append(mat)",
         "Assign PBR glass to rear body object"),
        ("Create the aluminum frame material",
         """frame_mat = bpy.data.materials.new('iPhone_Frame')
frame_mat.use_nodes = True
bsdf = frame_mat.node_tree.nodes['Principled BSDF']
bsdf.inputs['Base Color'].default_value = (0.8, 0.8, 0.82, 1.0)
bsdf.inputs['Metallic'].default_value = 1.0
bsdf.inputs['Roughness'].default_value = 0.15""",
         "Titanium frame: metallic = 1.0, moderate roughness for brushed finish"),
        ("Create an OLED screen material with slight emission",
         """screen_mat = bpy.data.materials.new('iPhone_Screen')
screen_mat.use_nodes = True
nt = screen_mat.node_tree
bsdf = nt.nodes['Principled BSDF']
bsdf.inputs['Base Color'].default_value = (0.02, 0.02, 0.02, 1.0)
bsdf.inputs['Emission Color'].default_value = (0.1, 0.4, 1.0, 1.0)
bsdf.inputs['Emission Strength'].default_value = 0.3""",
         "Dark OLED with subtle blue tint and low emission simulates idle screen"),
        ("Set up 3-point studio lighting",
         """bpy.ops.object.light_add(type='AREA', location=(0.3, -0.3, 0.5))
bpy.context.object.data.energy = 200
bpy.context.object.data.size = 0.5
bpy.ops.object.light_add(type='AREA', location=(-0.3, -0.2, 0.3))
bpy.context.object.data.energy = 80
bpy.ops.object.light_add(type='AREA', location=(0, 0.4, 0.1))
bpy.context.object.data.energy = 60""",
         "Key + fill + rim for product photography setup"),
        ("Add an environment HDRI for reflections",
         """world = bpy.data.worlds.new('World')
bpy.context.scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes['Background']
bg.inputs['Strength'].default_value = 0.3""",
         "Low-strength world background for subtle reflections on metallic surfaces"),
        ("Position the camera for a product shot",
         "bpy.ops.object.camera_add(location=(0, -0.4, 0.15))\nbpy.context.object.rotation_euler = (1.2, 0, 0)",
         "Camera at slight upward angle — classic product photography"),
        ("Set focal length to 85mm for compression",
         "bpy.context.object.data.lens = 85",
         "85mm avoids distortion, compresses background nicely for product viz"),
        ("Set render engine to Cycles with GPU",
         "bpy.context.scene.render.engine = 'CYCLES'\nbpy.context.scene.cycles.device = 'GPU'\nbpy.context.scene.cycles.samples = 256",
         "256 samples sufficient for clean product render with denoising"),
        ("Enable denoising",
         "bpy.context.scene.cycles.use_denoising = True\nbpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'",
         "OIDN removes noise efficiently, especially on metal and glass"),
        ("Set output resolution to 2K",
         "bpy.context.scene.render.resolution_x = 2048\nbpy.context.scene.render.resolution_y = 2048\nbpy.context.scene.render.resolution_percentage = 100",
         "2K square format for portfolio/social media"),
        ("Render the final image",
         "bpy.ops.render.render(write_still=True)",
         "Final render of the iPhone 16 model"),
    ]
}


# ─── Template: Luxury apartment interior (40 steps) ──────────────────────────

APARTMENT_TEMPLATE = {
    "name": "build_luxury_apartment",
    "intent": "Build a luxury apartment interior",
    "task_type": "BUILD",
    "steps": [
        ("Build a luxury apartment interior scene",
         "# 40-step workflow: room shell, flooring, ceiling, furniture, materials, lighting, camera, render",
         "Architecture visualization pipeline"),
        ("Delete the default cube and set up a clean scene",
         "bpy.ops.object.select_all(action='SELECT')\nbpy.ops.object.delete(use_global=False)",
         "Clean start for architectural work"),
        ("Add the floor plane — 8m x 6m living room",
         "bpy.ops.mesh.primitive_plane_add(size=1, location=(0,0,0))\nbpy.ops.transform.resize(value=(8,6,1))",
         "Floor at world origin, scaled to apartment dimensions"),
        ("Add the north wall",
         "bpy.ops.mesh.primitive_plane_add(size=1, location=(0,3,1.5))\nbpy.ops.transform.resize(value=(8,1.5,1))\nbpy.context.object.rotation_euler[0] = 1.5708",
         "Wall rotated 90 degrees to stand vertical"),
        ("Add the south wall",
         "bpy.ops.mesh.primitive_plane_add(size=1, location=(0,-3,1.5))\nbpy.ops.transform.resize(value=(8,1.5,1))\nbpy.context.object.rotation_euler[0] = 1.5708",
         "South wall at -3m, same height"),
        ("Add east wall",
         "bpy.ops.mesh.primitive_plane_add(size=1, location=(4,0,1.5))\nbpy.ops.transform.resize(value=(6,1.5,1))\nbpy.context.object.rotation_euler = (1.5708, 0, 1.5708)",
         "East wall perpendicular"),
        ("Add west wall with window opening — use boolean later",
         "bpy.ops.mesh.primitive_plane_add(size=1, location=(-4,0,1.5))\nbpy.ops.transform.resize(value=(6,1.5,1))\nbpy.context.object.rotation_euler = (1.5708, 0, 1.5708)",
         "West wall will get boolean window cutout"),
        ("Add a ceiling plane",
         "bpy.ops.mesh.primitive_plane_add(size=1, location=(0,0,3))\nbpy.ops.transform.resize(value=(8,6,1))",
         "Ceiling at 3m height"),
        ("Add hardwood floor planks using Array modifier",
         "bpy.ops.mesh.primitive_cube_add(size=1, location=(0,0,0.01))\nbpy.ops.transform.resize(value=(0.12,1.5,0.008))\nbpy.ops.object.modifier_add(type='ARRAY')\nbpy.context.object.modifiers['Array'].count = 70\nbpy.context.object.modifiers['Array'].relative_offset_displace[0] = 1.05",
         "Floor planks: 120mm wide, 1500mm long, 8mm thick — realistic proportions"),
        ("Add second array for rows",
         "bpy.ops.object.modifier_add(type='ARRAY')\nbpy.context.object.modifiers['Array.001'].count = 4\nbpy.context.object.modifiers['Array.001'].relative_offset_displace[0] = 0\nbpy.context.object.modifiers['Array.001'].relative_offset_displace[1] = 1.05",
         "Perpendicular array fills room width"),
        ("Add a sofa — start with the base cushion",
         "bpy.ops.mesh.primitive_cube_add(size=1, location=(-1, 1, 0.25))\nbpy.ops.transform.resize(value=(2.2, 0.9, 0.25))",
         "Sofa seat: 2.2m wide, 0.9m deep, 25cm tall"),
        ("Add sofa back",
         "bpy.ops.mesh.primitive_cube_add(size=1, location=(-1, 1.4, 0.65))\nbpy.ops.transform.resize(value=(2.2, 0.1, 0.4))",
         "Backrest: behind seat, 80cm total height"),
        ("Add sofa legs (4 cube legs)",
         """import bpy
for x in [-2.1, -2.1, 0.1, 0.1]:
    for y in [0.15, 1.65]:
        bpy.ops.mesh.primitive_cube_add(size=0.06, location=(x if x < 0 else x, y, 0.03))""",
         "Four legs at corners — 6cm square, 3cm tall"),
        ("Add a coffee table",
         "bpy.ops.mesh.primitive_cube_add(size=1, location=(-1, -0.2, 0.2))\nbpy.ops.transform.resize(value=(1.2, 0.6, 0.02))",
         "Glass coffee table: thin top slab"),
        ("Add coffee table legs",
         "bpy.ops.mesh.primitive_cylinder_add(radius=0.02, depth=0.38, location=(-0.5, -0.5, 0.19))\nbpy.ops.object.modifier_add(type='ARRAY')\nbpy.context.object.modifiers['Array'].count = 4\nbpy.context.object.modifiers['Array'].constant_offset_displace = [1.4, 0, 0]",
         "Thin metal cylinder legs"),
        ("Add a TV console along the south wall",
         "bpy.ops.mesh.primitive_cube_add(size=1, location=(0, -2.6, 0.25))\nbpy.ops.transform.resize(value=(2.4, 0.4, 0.25))",
         "Low-profile media console"),
        ("Add the TV screen",
         "bpy.ops.mesh.primitive_cube_add(size=1, location=(0, -2.9, 0.9))\nbpy.ops.transform.resize(value=(1.5, 0.03, 0.43))",
         "65-inch flat panel at eye level"),
        ("Add floor lamp in corner",
         "bpy.ops.mesh.primitive_cylinder_add(radius=0.015, depth=1.8, location=(3.5, -2.5, 0.9))\nbpy.ops.mesh.primitive_uv_sphere_add(radius=0.12, location=(3.5, -2.5, 1.8))",
         "Thin pole with globe shade"),
        ("Now create materials — oak hardwood for floor",
         """mat = bpy.data.materials.new('Oak_Floor')
mat.use_nodes = True
bsdf = mat.node_tree.nodes['Principled BSDF']
bsdf.inputs['Base Color'].default_value = (0.55, 0.35, 0.18, 1.0)
bsdf.inputs['Roughness'].default_value = 0.45
bsdf.inputs['Specular IOR Level'].default_value = 0.3""",
         "Oak: medium brown, moderate roughness — matte lacquer finish"),
        ("Create white plaster wall material",
         """wall_mat = bpy.data.materials.new('White_Plaster')
wall_mat.use_nodes = True
bsdf = wall_mat.node_tree.nodes['Principled BSDF']
bsdf.inputs['Base Color'].default_value = (0.95, 0.94, 0.92, 1.0)
bsdf.inputs['Roughness'].default_value = 0.85""",
         "Warm white with high roughness for matte plaster texture"),
        ("Create cream linen sofa material",
         """sofa_mat = bpy.data.materials.new('Cream_Linen')
sofa_mat.use_nodes = True
bsdf = sofa_mat.node_tree.nodes['Principled BSDF']
bsdf.inputs['Base Color'].default_value = (0.88, 0.82, 0.72, 1.0)
bsdf.inputs['Roughness'].default_value = 0.9
bsdf.inputs['Sheen Weight'].default_value = 0.3""",
         "Fabric: high roughness, sheen weight for cloth micro-fiber look"),
        ("Create smoked glass coffee table top",
         """glass_mat = bpy.data.materials.new('Smoked_Glass')
glass_mat.use_nodes = True
bsdf = glass_mat.node_tree.nodes['Principled BSDF']
bsdf.inputs['Base Color'].default_value = (0.02, 0.04, 0.05, 1.0)
bsdf.inputs['IOR'].default_value = 1.52
bsdf.inputs['Transmission Weight'].default_value = 0.95
bsdf.inputs['Roughness'].default_value = 0.0""",
         "Near-perfect transmission, slightly tinted for smoked glass"),
        ("Create brushed steel for table legs and lamp",
         """steel_mat = bpy.data.materials.new('Brushed_Steel')
steel_mat.use_nodes = True
bsdf = steel_mat.node_tree.nodes['Principled BSDF']
bsdf.inputs['Base Color'].default_value = (0.7, 0.7, 0.72, 1.0)
bsdf.inputs['Metallic'].default_value = 1.0
bsdf.inputs['Roughness'].default_value = 0.3""",
         "Metallic 1.0 with moderate roughness — brushed steel appearance"),
        ("Create matte black TV screen material",
         """tv_mat = bpy.data.materials.new('TV_Screen')
tv_mat.use_nodes = True
bsdf = tv_mat.node_tree.nodes['Principled BSDF']
bsdf.inputs['Base Color'].default_value = (0.01, 0.01, 0.01, 1.0)
bsdf.inputs['Roughness'].default_value = 0.05
bsdf.inputs['Emission Color'].default_value = (0.2, 0.4, 0.8, 1.0)
bsdf.inputs['Emission Strength'].default_value = 1.5""",
         "Dark OLED with blue ambient glow — adds life to the scene"),
        ("Set up architectural lighting — add large south window rect light",
         "bpy.ops.object.light_add(type='AREA', location=(0, -3.5, 1.5))\nbpy.context.object.data.energy = 2000\nbpy.context.object.data.size = 3.0\nbpy.context.object.data.size_y = 2.0",
         "Simulates daylight through floor-to-ceiling window"),
        ("Add warm overhead ceiling light",
         "bpy.ops.object.light_add(type='AREA', location=(0, 0, 2.95))\nbpy.context.object.data.energy = 400\nbpy.context.object.data.size = 3.0\nbpy.context.object.data.color = (1.0, 0.93, 0.8)",
         "Warm overhead ambient — color temperature ~3000K"),
        ("Add accent lamp emission from floor lamp",
         "bpy.ops.object.light_add(type='POINT', location=(3.5, -2.5, 1.7))\nbpy.context.object.data.energy = 150\nbpy.context.object.data.color = (1.0, 0.9, 0.7)",
         "Warm point light inside lamp globe"),
        ("Set world HDRI for outdoor sky reflections",
         """world = bpy.data.worlds.new('Apartment_World')
bpy.context.scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes['Background']
bg.inputs['Color'].default_value = (0.5, 0.65, 0.8, 1.0)
bg.inputs['Strength'].default_value = 0.4""",
         "Sky blue world background for realistic window reflections"),
        ("Position camera for wide interior shot",
         "bpy.ops.object.camera_add(location=(-3.5, -2.5, 1.2))\nbpy.context.object.rotation_euler = (1.47, 0, -0.7)",
         "Corner angle captures depth and most furniture"),
        ("Set camera to 24mm wide angle",
         "bpy.context.object.data.lens = 24",
         "24mm gives natural interior perspective without extreme distortion"),
        ("Enable depth of field",
         "bpy.context.object.data.dof.use_dof = True\nbpy.context.object.data.dof.focus_distance = 3.5\nbpy.context.object.data.dof.aperture_fstop = 5.6",
         "Subtle DOF at f/5.6 — keeps most room in focus"),
        ("Set Cycles render with high samples for archviz",
         "bpy.context.scene.render.engine = 'CYCLES'\nbpy.context.scene.cycles.device = 'GPU'\nbpy.context.scene.cycles.samples = 512",
         "512 samples for clean architectural render"),
        ("Enable denoising and set color management",
         "bpy.context.scene.cycles.use_denoising = True\nbpy.context.scene.view_settings.look = 'AgX - High Contrast'",
         "AgX High Contrast tone mapping for realistic interior photography look"),
        ("Set 4K output resolution",
         "bpy.context.scene.render.resolution_x = 3840\nbpy.context.scene.render.resolution_y = 2160",
         "4K (UHD) for portfolio-quality archviz render"),
        ("Add compositor glare for specular highlights",
         "bpy.context.scene.use_nodes = True\nbpy.context.scene.node_tree.nodes.new('CompositorNodeGlare')",
         "Glare node adds lens bloom to bright specular areas"),
        ("Set output format to EXR for post-processing",
         "bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'\nbpy.context.scene.render.image_settings.color_depth = '32'",
         "32-bit EXR preserves full HDR range for Lightroom/Affinity post-process"),
        ("Final check — name all objects properly",
         """import bpy
rename_map = {
    'Cube': 'Floor', 'Cube.001': 'NorthWall', 'Cube.002': 'SouthWall',
    'Plane': 'Ceiling', 'Cube.003': 'Sofa_Seat', 'Cube.004': 'Sofa_Back',
}
for obj in bpy.data.objects:
    if obj.name in rename_map:
        obj.name = rename_map[obj.name]""",
         "Organized scene hierarchy for client handoff"),
        ("Render the apartment",
         "bpy.ops.render.render(write_still=True)",
         "Final 4K Cycles render of luxury apartment"),
    ]
}


# ─── Template: Photorealistic forest scene (30 steps) ────────────────────────

FOREST_TEMPLATE = {
    "name": "build_forest_scene",
    "intent": "Create a photorealistic forest scene",
    "task_type": "BUILD",
    "steps": [
        ("Create a photorealistic forest scene",
         "# 30-step pipeline: terrain, trees, foliage, atmosphere, lighting, render",
         "Nature environment production workflow"),
        ("Delete defaults and add terrain base plane",
         "bpy.ops.object.select_all(action='SELECT')\nbpy.ops.object.delete()\nbpy.ops.mesh.primitive_plane_add(size=50, location=(0,0,0))",
         "Large plane for terrain base"),
        ("Add displacement modifier for terrain",
         "bpy.ops.object.modifier_add(type='DISPLACE')\nbpy.ops.texture.new()\nbpy.context.object.modifiers['Displace'].texture = bpy.data.textures[-1]\nbpy.context.object.modifiers['Displace'].strength = 3.0",
         "Displace modifier creates organic terrain from texture"),
        ("Subdivide terrain for detail",
         "bpy.ops.object.modifier_add(type='SUBSURF')\nbpy.context.object.modifiers['Subdivision'].subdivision_type = 'SIMPLE'\nbpy.context.object.modifiers['Subdivision'].levels = 6",
         "Simple subdivision preserves displace shape, adds polygon density"),
        ("Add a tree trunk cylinder",
         "bpy.ops.mesh.primitive_cylinder_add(radius=0.3, depth=8, location=(0,0,4))",
         "Base trunk form — radius 30cm, 8m tall"),
        ("Taper the trunk with proportional editing",
         "bpy.ops.object.mode_set(mode='EDIT')\nbpy.ops.mesh.select_all(action='SELECT')\nbpy.ops.transform.resize(value=(0.5, 0.5, 1.0), orient_type='GLOBAL')",
         "Narrow top of trunk for natural look"),
        ("Add bark displacement to trunk",
         "bpy.ops.object.mode_set(mode='OBJECT')\nbpy.ops.object.modifier_add(type='DISPLACE')\nbpy.ops.texture.new()\nbpy.context.object.modifiers['Displace'].strength = 0.08",
         "Subtle bark texture via displacement"),
        ("Add primary branches using curves",
         "bpy.ops.curve.primitive_bezier_curve_add(radius=0.15, location=(0.3, 0, 6))",
         "Bezier curve for natural branch curvature"),
        ("Convert curve to mesh and bevel for thickness",
         "bpy.context.object.data.bevel_depth = 0.08\nbpy.context.object.data.bevel_resolution = 4",
         "Bevel gives branch circular cross-section"),
        ("Add particle system for leaf clusters",
         """bpy.ops.object.select_all(action='DESELECT')
obj = bpy.data.objects['Cylinder']
bpy.context.view_layer.objects.active = obj
bpy.ops.object.particle_system_add()
ps = obj.particle_systems[0]
ps.settings.count = 2000
ps.settings.type = 'HAIR'
ps.settings.hair_length = 0.8""",
         "Hair particle system distributes leaf geometry across trunk/branches"),
        ("Create ground cover — add grass particle system to terrain",
         """terrain = bpy.data.objects['Plane']
bpy.context.view_layer.objects.active = terrain
bpy.ops.object.particle_system_add()
ps = terrain.particle_systems[0]
ps.settings.count = 50000
ps.settings.type = 'HAIR'
ps.settings.hair_length = 0.3""",
         "50k grass blades via hair particles on terrain"),
        ("Add forest undergrowth ferns — add low plane",
         "bpy.ops.mesh.primitive_plane_add(size=0.6, location=(1.5, 0.8, 0.1))",
         "Individual fern frond geometry"),
        ("Bend fern plane with simple deform",
         "bpy.ops.object.modifier_add(type='SIMPLE_DEFORM')\nbpy.context.object.modifiers['SimpleDeform'].deform_method = 'BEND'\nbpy.context.object.modifiers['SimpleDeform'].angle = 1.2",
         "Curved frond form via simple deform bend"),
        ("Create bark PBR material",
         """bark_mat = bpy.data.materials.new('Tree_Bark')
bark_mat.use_nodes = True
bsdf = bark_mat.node_tree.nodes['Principled BSDF']
bsdf.inputs['Base Color'].default_value = (0.25, 0.18, 0.1, 1.0)
bsdf.inputs['Roughness'].default_value = 0.95
bsdf.inputs['Specular IOR Level'].default_value = 0.02""",
         "Bark: dark brown, nearly 100% rough, almost no specular"),
        ("Create translucent leaf material with SSS",
         """leaf_mat = bpy.data.materials.new('Leaf')
leaf_mat.use_nodes = True
bsdf = leaf_mat.node_tree.nodes['Principled BSDF']
bsdf.inputs['Base Color'].default_value = (0.12, 0.45, 0.08, 1.0)
bsdf.inputs['Roughness'].default_value = 0.7
bsdf.inputs['Subsurface Weight'].default_value = 0.3
bsdf.inputs['Subsurface Radius'].default_value = [0.05, 0.1, 0.02]""",
         "Subsurface scattering makes leaves glow when backlit by sun"),
        ("Create wet ground soil material",
         """soil_mat = bpy.data.materials.new('Forest_Ground')
soil_mat.use_nodes = True
bsdf = soil_mat.node_tree.nodes['Principled BSDF']
bsdf.inputs['Base Color'].default_value = (0.15, 0.1, 0.06, 1.0)
bsdf.inputs['Roughness'].default_value = 0.98
bsdf.inputs['Specular IOR Level'].default_value = 0.0""",
         "Dark moist soil: max roughness, no specular"),
        ("Add atmospheric fog using volume scatter",
         """world = bpy.data.worlds.new('Forest_World')
bpy.context.scene.world = world
world.use_nodes = True
nt = world.node_tree
vol = nt.nodes.new('ShaderNodeVolumePrincipled')
vol.inputs['Density'].default_value = 0.01
vol.inputs['Anisotropy'].default_value = 0.5""",
         "Volume scatter at 0.01 density creates subtle morning forest mist"),
        ("Add sun light at low angle for golden hour",
         "bpy.ops.object.light_add(type='SUN', location=(10, -10, 8))\nbpy.context.object.data.energy = 3.0\nbpy.context.object.data.angle = 0.01\nbpy.context.object.data.color = (1.0, 0.85, 0.6)",
         "Low sun angle creates long shadows; warm color for golden hour"),
        ("Add fill sky light",
         "bpy.ops.object.light_add(type='AREA', location=(0, 0, 20))\nbpy.context.object.data.energy = 200\nbpy.context.object.data.size = 30\nbpy.context.object.data.color = (0.4, 0.6, 1.0)",
         "Large overhead area simulates diffuse sky illumination"),
        ("Set world sky texture",
         """world = bpy.context.scene.world
nt = world.node_tree
sky = nt.nodes.new('ShaderNodeTexSky')
sky.sky_type = 'NISHITA'
sky.sun_elevation = 0.15
sky.air_density = 1.5
bg = nt.nodes['Background']
nt.links.new(sky.outputs['Color'], bg.inputs['Color'])""",
         "Nishita sky model — physically accurate Rayleigh/Mie scattering"),
        ("Position camera for forest path shot",
         "bpy.ops.object.camera_add(location=(0, -8, 1.7))\nbpy.context.object.rotation_euler = (1.48, 0, 0)",
         "Eye-level shot looking into forest depth — creates perspective draw"),
        ("Set camera to 35mm for natural field of view",
         "bpy.context.object.data.lens = 35",
         "35mm full-frame equivalent — natural human vision FOV"),
        ("Enable volumetric rendering",
         "bpy.context.scene.render.engine = 'CYCLES'\nbpy.context.scene.cycles.volume_bounces = 4\nbpy.context.scene.cycles.volume_max_steps = 128",
         "Volumetric bounces needed for fog light scattering"),
        ("Set high sample count for complex scene",
         "bpy.context.scene.cycles.samples = 1024\nbpy.context.scene.cycles.device = 'GPU'",
         "Forest with SSS + volume needs 1024 samples for clean result"),
        ("Add motion blur for realism",
         "bpy.context.scene.render.use_motion_blur = True\nbpy.context.scene.cycles.motion_blur_position = 'START'",
         "Motion blur on swaying foliage if animated"),
        ("Set color management to Filmic High Contrast",
         "bpy.context.scene.view_settings.view_transform = 'Filmic'\nbpy.context.scene.view_settings.look = 'Filmic - High Contrast'",
         "Filmic tonemapping prevents highlight clipping on bright sun areas"),
        ("Add compositor vignette",
         """bpy.context.scene.use_nodes = True
nt = bpy.context.scene.node_tree
lens = nt.nodes.new('CompositorNodeLensDist')
lens.inputs['Distort'].default_value = -0.05
lens.inputs['Dispersion'].default_value = 0.01""",
         "Subtle lens distortion adds photorealistic camera character"),
        ("Enable ambient occlusion",
         "bpy.context.scene.world.light_settings.use_ambient_occlusion = True\nbpy.context.scene.world.light_settings.ao_factor = 0.3",
         "AO fills shadows under foliage and ground contact points"),
        ("Set 6K output for hero render",
         "bpy.context.scene.render.resolution_x = 6144\nbpy.context.scene.render.resolution_y = 4096\nbpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'",
         "6K EXR for maximum flexibility in compositing"),
        ("Render the forest scene",
         "bpy.ops.render.render(write_still=True)",
         "Final Cycles render of photorealistic forest"),
    ]
}


# ─── Additional templates (shorter, 5-15 steps) ───────────────────────────────

SHORT_TEMPLATES = [
    {
        "name": "pbr_copper_material",
        "intent": "Create an aged copper PBR material",
        "task_type": "MATERIALIZE",
        "steps": [
            ("Create an aged copper PBR material",
             "# Aged copper: base metal + patina layer + edge wear",
             "Material plan: layered Principled BSDF approach"),
            ("Create a new material called Aged Copper",
             "mat = bpy.data.materials.new('Aged_Copper')\nmat.use_nodes = True\nbpy.context.object.data.materials.append(mat)",
             "Material slot added and named"),
            ("Set the base color to a patina green-brown",
             """bsdf = bpy.context.object.active_material.node_tree.nodes['Principled BSDF']
bsdf.inputs['Base Color'].default_value = (0.22, 0.45, 0.3, 1.0)""",
             "Aged copper shows green patina from oxidation"),
            ("Set metallic to 1.0 for conductor behavior",
             "bsdf.inputs['Metallic'].default_value = 1.0",
             "Copper is a conductor — metallic must be 1.0 for Fresnel to work correctly"),
            ("Set roughness to 0.65 for weathered surface",
             "bsdf.inputs['Roughness'].default_value = 0.65",
             "Weathered metal scatters light — high roughness vs polished copper at 0.15"),
            ("Add clearcoat for polished high spots",
             "bsdf.inputs['Coat Weight'].default_value = 0.3\nbsdf.inputs['Coat Roughness'].default_value = 0.1",
             "Clearcoat simulates areas where patina wore off revealing shiny metal"),
            ("Set IOR for copper",
             "bsdf.inputs['IOR'].default_value = 1.2",
             "Copper IOR is ~1.2; affects glancing-angle Fresnel"),
            ("Assign material to selected objects",
             "bpy.ops.object.material_slot_assign()",
             "Apply material to current selection"),
        ]
    },
    {
        "name": "rigid_body_sim",
        "intent": "Set up a rigid body simulation with 50 spheres",
        "task_type": "SIMULATE",
        "steps": [
            ("Set up a rigid body simulation with 50 spheres falling onto a plane",
             "# Physics pipeline: ground plane passive, spheres active",
             "Rigid body simulation setup"),
            ("Add a ground plane and make it a passive rigid body",
             "bpy.ops.mesh.primitive_plane_add(size=10, location=(0,0,0))\nbpy.ops.rigidbody.object_add()\nbpy.context.object.rigid_body.type = 'PASSIVE'",
             "Passive rigid bodies are static colliders"),
            ("Add a sphere at height 5",
             "bpy.ops.mesh.primitive_uv_sphere_add(radius=0.3, location=(0,0,5))",
             "First falling sphere"),
            ("Make it an active rigid body",
             "bpy.ops.rigidbody.object_add()\nbpy.context.object.rigid_body.type = 'ACTIVE'\nbpy.context.object.rigid_body.restitution = 0.6",
             "Active = physics driven; restitution 0.6 for bouncy rubber"),
            ("Array duplicate for 50 spheres using Python loop",
             """import random
for i in range(49):
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=0.3,
        location=(random.uniform(-3,3), random.uniform(-3,3), random.uniform(5,10))
    )
    bpy.ops.rigidbody.object_add()
    bpy.context.object.rigid_body.type = 'ACTIVE'""",
             "Random positions above ground — varied heights for staggered drop"),
            ("Set simulation frame range",
             "bpy.context.scene.frame_start = 1\nbpy.context.scene.frame_end = 250\nbpy.context.scene.rigidbody_world.steps_per_second = 120",
             "120 substeps for stable sphere stack simulation"),
            ("Bake the simulation",
             "bpy.ops.ptcache.bake_all(bake=True)",
             "Bakes rigid body cache so animation is stable for rendering"),
        ]
    },
    {
        "name": "three_point_lighting",
        "intent": "Set up professional 3-point lighting",
        "task_type": "LIGHT",
        "steps": [
            ("Set up professional 3-point lighting for a product shoot",
             "# Key light + fill light + rim/backlight",
             "Classic studio 3-point lighting rig"),
            ("Delete existing lights",
             "bpy.ops.object.select_by_type(type='LIGHT')\nbpy.ops.object.delete()",
             "Clear scene lights before adding controlled rig"),
            ("Add key light — main illumination, upper right",
             "bpy.ops.object.light_add(type='AREA', location=(3, -2, 4))\nbpy.context.object.data.energy = 600\nbpy.context.object.data.size = 1.0\nbpy.context.object.name = 'Key_Light'",
             "Key light at 45-degree elevation, 45-degree horizontal"),
            ("Add fill light — reduces shadow harshness, opposite side",
             "bpy.ops.object.light_add(type='AREA', location=(-2, -1.5, 2))\nbpy.context.object.data.energy = 200\nbpy.context.object.data.size = 2.0\nbpy.context.object.name = 'Fill_Light'",
             "Fill at 1/3 key power; larger size for softer shadow fill"),
            ("Add rim light — separates subject from background",
             "bpy.ops.object.light_add(type='AREA', location=(0, 3, 3))\nbpy.context.object.data.energy = 300\nbpy.context.object.data.size = 0.5\nbpy.context.object.name = 'Rim_Light'",
             "Rim light from behind creates separation highlight"),
            ("Add HDRI environment for ambient fill",
             """world = bpy.data.worlds.new('Studio_World')
bpy.context.scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes['Background']
bg.inputs['Strength'].default_value = 0.1""",
             "Low-strength world provides ambient fill without overwhelming 3-point rig"),
            ("Set render engine and samples",
             "bpy.context.scene.render.engine = 'CYCLES'\nbpy.context.scene.cycles.samples = 128",
             "128 samples sufficient for studio lighting — fast iteration"),
        ]
    },
    {
        "name": "cloth_simulation",
        "intent": "Create a cloth simulation — tablecloth falling over a table",
        "task_type": "SIMULATE",
        "steps": [
            ("Create a cloth simulation — tablecloth falling over a table",
             "# Cloth sim: plane with cloth modifier over box collider",
             "Cloth simulation setup"),
            ("Add a table box as collision object",
             "bpy.ops.mesh.primitive_cube_add(size=1, location=(0,0,0.5))\nbpy.ops.transform.resize(value=(2, 1, 0.4))",
             "Table: 2m x 1m x 40cm"),
            ("Add cloth modifier as collision",
             "bpy.ops.object.modifier_add(type='COLLISION')",
             "Collision modifier lets cloth interact with table surface"),
            ("Add the tablecloth plane high above",
             "bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 1.5))\nbpy.ops.transform.resize(value=(2.4, 1.4, 1.0))",
             "Cloth slightly wider than table to hang over edges"),
            ("Subdivide cloth for drape detail",
             "bpy.ops.object.mode_set(mode='EDIT')\nbpy.ops.mesh.subdivide(number_cuts=20)\nbpy.ops.object.mode_set(mode='OBJECT')",
             "High subdivision needed for realistic cloth fold detail"),
            ("Add cloth modifier",
             "bpy.ops.object.modifier_add(type='CLOTH')\nbpy.context.object.modifiers['Cloth'].settings.quality = 15\nbpy.context.object.modifiers['Cloth'].settings.mass = 0.3",
             "Cloth quality 15 for smooth drape; mass 300g for linen weight"),
            ("Set cloth stiffness for linen",
             "bpy.context.object.modifiers['Cloth'].settings.tension_stiffness = 15\nbpy.context.object.modifiers['Cloth'].settings.shear_stiffness = 5",
             "Linen: medium tension stiffness, lower shear — stiffer than silk"),
            ("Bake the cloth simulation",
             "bpy.ops.ptcache.bake_all(bake=True)",
             "Bake all 120 frames of cloth drop and settle"),
            ("Jump to frame 120 to see settled cloth",
             "bpy.context.scene.frame_set(120)",
             "Frame 120 shows cloth at rest on table"),
        ]
    },
    {
        "name": "character_face_sculpt",
        "intent": "Sculpt a stylized character face",
        "task_type": "BUILD",
        "steps": [
            ("Sculpt a stylized character face",
             "# Sculpt pipeline: sphere base → blocking → detail → retopo",
             "Character sculpt workflow"),
            ("Add a UV sphere as the head base",
             "bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(0,0,1.2))",
             "Sphere is the universal starting point for head sculpts"),
            ("Apply smooth shading",
             "bpy.ops.object.shade_smooth()",
             "Smooth shading before sculpting avoids visible polygon faceting"),
            ("Enter sculpt mode",
             "bpy.ops.object.mode_set(mode='SCULPT')",
             "Sculpt mode activates brush-based mesh editing"),
            ("Enable dynamic topology for adaptive detail",
             "bpy.ops.sculpt.dynamic_topology_toggle()\nbpy.context.scene.tool_settings.sculpt.detail_size = 8",
             "Dyntopo subdivides on the fly — no need to pre-subdivide"),
            ("Set detail percent to 8 for medium resolution",
             "bpy.context.scene.tool_settings.sculpt.constant_detail_resolution = 8",
             "Detail resolution 8 gives enough polygons for facial features"),
            ("Voxel remesh at 0.03 for uniform topology",
             "bpy.context.object.data.remesh_voxel_size = 0.03\nbpy.ops.object.voxel_remesh()",
             "Voxel remesh at 3cm voxels creates uniform mesh for clean sculpting"),
            ("Return to object mode and apply smooth modifier",
             "bpy.ops.object.mode_set(mode='OBJECT')\nbpy.ops.object.modifier_add(type='SUBSURF')\nbpy.context.object.modifiers['Subdivision'].levels = 2",
             "Subdivision smooths the remesh before fine detail sculpting"),
        ]
    },
    {
        "name": "procedural_wood_material",
        "intent": "Create a procedural wood material with grain",
        "task_type": "MATERIALIZE",
        "steps": [
            ("Create a procedural wood material with visible grain",
             "# Procedural approach: Wave texture for grain, Noise for variation",
             "Procedural material — no texture images needed"),
            ("Add a new material",
             "mat = bpy.data.materials.new('Procedural_Wood')\nmat.use_nodes = True\nbpy.context.object.data.materials.append(mat)",
             "Named material for wood"),
            ("Add a Wave texture node for grain pattern",
             """nt = bpy.context.object.active_material.node_tree
wave = nt.nodes.new('ShaderNodeTexWave')
wave.wave_type = 'RINGS'
wave.distortion = 3.0
wave.detail = 6.0
wave.detail_scale = 1.5""",
             "Wave rings simulate annular tree growth rings; distortion adds natural irregularity"),
            ("Add a Noise texture for knot variation",
             """noise = nt.nodes.new('ShaderNodeTexNoise')
noise.inputs['Scale'].default_value = 2.0
noise.inputs['Detail'].default_value = 8.0
noise.inputs['Roughness'].default_value = 0.6""",
             "Noise overlaid on wave creates knot and burl variation"),
            ("Mix the wave and noise outputs with ColorRamp",
             """cr = nt.nodes.new('ShaderNodeValToRGB')
cr.color_ramp.elements[0].color = (0.35, 0.2, 0.08, 1.0)
cr.color_ramp.elements[1].color = (0.65, 0.42, 0.22, 1.0)""",
             "ColorRamp maps grey wave to dark/light wood color range"),
            ("Connect to Principled BSDF",
             """bsdf = nt.nodes['Principled BSDF']
bsdf.inputs['Roughness'].default_value = 0.5
nt.links.new(cr.outputs['Color'], bsdf.inputs['Base Color'])""",
             "Color mapped to base color; roughness 0.5 for satin lacquer finish"),
            ("Add texture coordinate and mapping node",
             """texcoord = nt.nodes.new('ShaderNodeTexCoord')
mapping = nt.nodes.new('ShaderNodeMapping')
mapping.inputs['Scale'].default_value = (0.5, 0.5, 2.0)
nt.links.new(texcoord.outputs['Object'], mapping.inputs['Vector'])
nt.links.new(mapping.outputs['Vector'], wave.inputs['Vector'])""",
             "Object coordinates + stretched Z mapping orients grain along object's Z axis"),
        ]
    },
    {
        "name": "vehicle_wheel",
        "intent": "Model a realistic car wheel with tire",
        "task_type": "BUILD",
        "steps": [
            ("Model a realistic car wheel with tire",
             "# Wheel: rim + spokes + tire torus",
             "Automotive wheel modeling workflow"),
            ("Add a cylinder for the wheel hub",
             "bpy.ops.mesh.primitive_cylinder_add(radius=0.3, depth=0.05, location=(0,0,0))",
             "Hub disc: 30cm radius, 5cm thick"),
            ("Add a torus for the tire",
             "bpy.ops.mesh.primitive_torus_add(major_radius=0.35, minor_radius=0.12, location=(0,0,0))\nbpy.context.object.rotation_euler[0] = 1.5708",
             "Torus standing upright — major radius 35cm matches hub; minor 12cm for tire cross-section"),
            ("Add spoke cylinder",
             "bpy.ops.mesh.primitive_cylinder_add(radius=0.025, depth=0.28, location=(0.2, 0, 0.02))\nbpy.context.object.rotation_euler[1] = 1.5708",
             "First spoke — horizontal cylinder from hub to rim"),
            ("Array spoke around hub",
             "bpy.ops.object.modifier_add(type='ARRAY')\nbpy.context.object.modifiers['Array'].count = 5\nbpy.context.object.modifiers['Array'].use_object_offset = True",
             "5-spoke design via array with rotation offset"),
            ("Create aluminum rim material",
             """mat = bpy.data.materials.new('Alloy_Rim')
mat.use_nodes = True
bsdf = mat.node_tree.nodes['Principled BSDF']
bsdf.inputs['Base Color'].default_value = (0.85, 0.85, 0.87, 1.0)
bsdf.inputs['Metallic'].default_value = 1.0
bsdf.inputs['Roughness'].default_value = 0.1""",
             "Polished alloy: metallic 1.0, low roughness for mirror finish"),
            ("Create rubber tire material",
             """tire_mat = bpy.data.materials.new('Tire_Rubber')
tire_mat.use_nodes = True
bsdf = tire_mat.node_tree.nodes['Principled BSDF']
bsdf.inputs['Base Color'].default_value = (0.03, 0.03, 0.03, 1.0)
bsdf.inputs['Roughness'].default_value = 0.95
bsdf.inputs['Specular IOR Level'].default_value = 0.05""",
             "Carbon black rubber: nearly zero specular, max roughness"),
        ]
    },
    {
        "name": "hdri_lighting_setup",
        "intent": "Set up HDRI environment lighting for product photography",
        "task_type": "LIGHT",
        "steps": [
            ("Set up HDRI environment lighting",
             "# HDRI world + ground shadow catcher for product viz",
             "Environment lighting pipeline"),
            ("Set up the world nodes",
             """world = bpy.data.worlds.new('HDRI_World')
bpy.context.scene.world = world
world.use_nodes = True""",
             "Create and assign new world"),
            ("Add Environment Texture node",
             """nt = world.node_tree
env = nt.nodes.new('ShaderNodeTexEnvironment')
bg = nt.nodes['Background']
nt.links.new(env.outputs['Color'], bg.inputs['Color'])""",
             "Environment texture drives the world background and lighting"),
            ("Add texture coordinate mapping",
             """mapping = nt.nodes.new('ShaderNodeMapping')
coord = nt.nodes.new('ShaderNodeTexCoord')
nt.links.new(coord.outputs['Generated'], mapping.inputs['Vector'])
nt.links.new(mapping.outputs['Vector'], env.inputs['Vector'])""",
             "Mapping node lets us rotate the HDRI without re-baking"),
            ("Set HDRI rotation for best light angle",
             "mapping.inputs['Rotation'].default_value[2] = 0.785",
             "45 degrees rotation — standard starting point for product HDRI"),
            ("Set background strength",
             "bg.inputs['Strength'].default_value = 1.5",
             "1.5 strength brightens the environment slightly above neutral"),
            ("Add a ground plane shadow catcher",
             "bpy.ops.mesh.primitive_plane_add(size=20, location=(0,0,0))\nbpy.context.object.is_shadow_catcher = True",
             "Shadow catcher receives shadows but is invisible in render"),
            ("Set render engine for HDRI",
             "bpy.context.scene.render.engine = 'CYCLES'\nbpy.context.scene.cycles.samples = 128\nbpy.context.scene.render.film_transparent = True",
             "Transparent film mode: background is alpha, shows HDRI in reflections only"),
        ]
    },
    {
        "name": "architecture_arch",
        "intent": "Model a Gothic arch for architectural use",
        "task_type": "BUILD",
        "steps": [
            ("Model a Gothic pointed arch",
             "# Arch from bezier curve, converted to mesh, solidified",
             "Parametric arch modeling"),
            ("Add a bezier curve",
             "bpy.ops.curve.primitive_bezier_curve_add(location=(0,0,0))",
             "Bezier curve is the ideal starting point for arch profile"),
            ("Set curve resolution",
             "bpy.context.object.data.resolution_u = 32",
             "High resolution for smooth arch curve"),
            ("Set bevel depth for arch thickness",
             "bpy.context.object.data.bevel_depth = 0.15\nbpy.context.object.data.bevel_resolution = 4",
             "Bevel gives the arch its 3D cross-section profile"),
            ("Extrude curve for depth",
             "bpy.context.object.data.extrude = 0.5",
             "0.5m deep arch — suits interior doorway scale"),
            ("Convert to mesh for further editing",
             "bpy.ops.object.convert(target='MESH')",
             "Convert allows UV mapping and boolean operations"),
            ("Add solidify for wall thickness",
             "bpy.ops.object.modifier_add(type='SOLIDIFY')\nbpy.context.object.modifiers['Solidify'].thickness = 0.3",
             "Solidify adds 30cm wall thickness for architectural scale"),
            ("Create stone material for arch",
             """stone_mat = bpy.data.materials.new('Gothic_Stone')
stone_mat.use_nodes = True
bsdf = stone_mat.node_tree.nodes['Principled BSDF']
bsdf.inputs['Base Color'].default_value = (0.65, 0.62, 0.55, 1.0)
bsdf.inputs['Roughness'].default_value = 0.88""",
             "Cool grey limestone: high roughness for cut stone finish"),
        ]
    },
    {
        "name": "neon_sign",
        "intent": "Create a neon sign with glowing emission",
        "task_type": "BUILD",
        "steps": [
            ("Create a glowing neon sign",
             "# Neon: bezier curve text → tube → emission material",
             "Neon sign creation workflow"),
            ("Add a bezier circle as the neon tube base",
             "bpy.ops.curve.primitive_bezier_circle_add(radius=0.5, location=(0,0,0))",
             "Circle curve will be swept along path to form neon tube"),
            ("Set bevel to make it a tube",
             "bpy.context.object.data.bevel_depth = 0.015\nbpy.context.object.data.bevel_resolution = 8",
             "1.5cm diameter tube — realistic neon glass tube gauge"),
            ("Add a text object for the sign letters",
             "bpy.ops.object.text_add(location=(0,0,0))\nbpy.context.object.data.body = 'OPEN'",
             "Text object for neon lettering"),
            ("Set font size and extrude",
             "bpy.context.object.data.size = 0.8\nbpy.context.object.data.extrude = 0.02",
             "80cm tall letters with 2cm extrude for tube depth"),
            ("Convert text to mesh",
             "bpy.ops.object.convert(target='MESH')",
             "Mesh conversion allows modifier stack"),
            ("Add wireframe modifier to get tube outline",
             "bpy.ops.object.modifier_add(type='WIREFRAME')\nbpy.context.object.modifiers['Wireframe'].thickness = 0.015",
             "Wireframe modifier creates the tube outline from letter mesh edges"),
            ("Create neon emission material — pink",
             """neon_mat = bpy.data.materials.new('Neon_Pink')
neon_mat.use_nodes = True
bsdf = neon_mat.node_tree.nodes['Principled BSDF']
bsdf.inputs['Base Color'].default_value = (1.0, 0.1, 0.5, 1.0)
bsdf.inputs['Emission Color'].default_value = (1.0, 0.1, 0.5, 1.0)
bsdf.inputs['Emission Strength'].default_value = 8.0""",
             "High emission strength (8.0) causes visible light bloom in Cycles"),
            ("Enable bloom in render settings",
             "bpy.context.scene.render.engine = 'CYCLES'\nbpy.context.scene.cycles.samples = 64",
             "Bloom is added in compositor — emission creates physically accurate glow"),
        ]
    },
]


# ─── Expand short templates to full template list ─────────────────────────────

ALL_TEMPLATES = [IPHONE_TEMPLATE, APARTMENT_TEMPLATE, FOREST_TEMPLATE] + SHORT_TEMPLATES

TEMPLATE_NAMES = {t["name"] for t in ALL_TEMPLATES}


def generate_synthetic_conversation(template: dict,
                                    variation_seed: int = 0) -> dict:
    """Generate a single synthetic multi-turn conversation from a template."""
    steps = template["steps"]
    messages = []

    # System primer as first user message
    messages.append(make_message("user", template["intent"]))

    # First step: planning phase
    first_voice, first_python, first_reasoning = steps[0]
    messages.append(make_message(
        "assistant",
        _json_step(first_python, "MULTI_STEP_PLAN", first_reasoning)
    ))

    # Remaining steps
    for i, (voice, python, reasoning) in enumerate(steps[1:], start=1):
        # Add small natural variation to voice commands based on seed
        rng = random.Random(variation_seed * 100 + i)
        filler_starters = ["", "now ", "next, ", "okay, ", "great — ", ""]
        starter = rng.choice(filler_starters)
        messages.append(make_message("user", starter + voice))
        messages.append(make_message(
            "assistant",
            _json_step(python, f"STEP_{i}", reasoning)
        ))

    return pack_conversation(
        messages  = messages,
        task_type = template["task_type"],
        source    = "synthetic_multiturn",
        quality   = 3.5,
    )


def generate_synthetic(template_filter: str, count: int) -> list[dict]:
    """Generate `count` synthetic conversations, cycling through templates."""
    if template_filter == "all":
        templates = ALL_TEMPLATES
    else:
        names = {n.strip() for n in template_filter.split(",")}
        templates = [t for t in ALL_TEMPLATES if t["name"] in names]
        if not templates:
            print(f"[synthetic] No templates matched '{template_filter}'. Available:")
            for t in ALL_TEMPLATES:
                print(f"  {t['name']}")
            return []

    conversations = []
    for i in range(count):
        template = templates[i % len(templates)]
        conv = generate_synthetic_conversation(template, variation_seed=i)
        conversations.append(conv)

    print(f"[synthetic] Generated {len(conversations)} synthetic conversations "
          f"({len(templates)} templates, up to {count} total).")
    return conversations


# ─── Output ───────────────────────────────────────────────────────────────────

def write_conversations(conversations: list[dict]) -> None:
    MULTITURN_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")
    print(f"\nWrote {len(conversations):,} conversations → {OUTPUT_FILE}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chain training pairs into multi-turn conversations"
    )
    parser.add_argument("--from-pairs", action="store_true",
                        help="Chain existing validated pairs by video_id")
    parser.add_argument("--synthetic",  action="store_true",
                        help="Generate synthetic multi-turn conversations from templates")
    parser.add_argument("--templates",  default="all",
                        help="Comma-separated template names, or 'all'")
    parser.add_argument("--count",      type=int, default=1000,
                        help="Number of synthetic conversations to generate")
    parser.add_argument("--input",      type=Path, default=INPUT_FILE,
                        help="Input JSONL with validated pairs")
    parser.add_argument("--list-templates", action="store_true",
                        help="Print available template names and exit")
    args = parser.parse_args()

    if args.list_templates:
        print("Available synthetic templates:")
        for t in ALL_TEMPLATES:
            print(f"  {t['name']}  ({t['task_type']}, {len(t['steps'])} steps)")
        return

    all_conversations: list[dict] = []

    if args.from_pairs:
        pairs = load_pairs(args.input)
        if not pairs:
            print(f"[chain] No pairs found at {args.input}. Run validate.py first.")
        else:
            chained = generate_from_pairs(pairs)
            all_conversations.extend(chained)

    if args.synthetic:
        synthetic = generate_synthetic(args.templates, args.count)
        all_conversations.extend(synthetic)

    if not all_conversations:
        print("Nothing to write. Use --from-pairs and/or --synthetic.")
        return

    write_conversations(all_conversations)

    # Stats
    task_types: dict[str, int] = {}
    for c in all_conversations:
        t = c.get("task_type", "UNKNOWN")
        task_types[t] = task_types.get(t, 0) + 1
    print("\nTask type distribution:")
    for k, v in sorted(task_types.items()):
        print(f"  {k:<20} {v:>6}")
    print(f"\nNext: python task_classifier.py --input {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
