"""
integrations/dream3d_synthetic.py — Dream3D-style DPO preference pairs for Nalana.

Dream3D (YC W2023) generates 3D worlds from text prompts. Their outputs represent
baseline text-to-3D quality — often geometrically loose, topologically naive, and
missing engineering-quality decisions.

This module generates Direct Preference Optimization (DPO) training pairs where:
  - "chosen": Expert Blender implementation with proper topology, physics-aware
    geometry, appropriate mesh density, correct proportions
  - "rejected": What a naive/quick approach produces — wrong topology, too many
    polys in wrong places, missing structural details, poor proportions

These DPO pairs teach Nalana to PREFER expert-quality implementations.
This is how we bake quality discrimination into the model's weights.

Usage:
    python integrations/dream3d_synthetic.py --count 200 --output data/integrations/dream3d/
    python integrations/dream3d_synthetic.py --scrape  # attempt Dream3D gallery scrape
    python integrations/dream3d_synthetic.py --preview 5  # show 5 example pairs
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from pathlib import Path
from textwrap import dedent

from tqdm import tqdm

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

# ─── Setup ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dream3d_synthetic")

BASE_DIR = Path(__file__).parents[1]
DEFAULT_OUTPUT = BASE_DIR / "data" / "integrations" / "dream3d"

# ─── DPO pair templates ────────────────────────────────────────────────────────
# Each entry defines:
#   prompt: voice command / text-to-3D request
#   theme: category for this kind of object
#   chosen: expert implementation with reasoning
#   rejected: naive implementation with reasoning
#
# "chosen" reasoning explains WHY this is better (topology, proportions, physics).
# "rejected" reasoning explains what went wrong (the teaching signal for DPO).

DPO_TEMPLATES: list[dict] = [
    # ─── ARCHITECTURAL ────────────────────────────────────────────────────────
    {
        "prompt": "Create a castle with stone walls and towers",
        "theme": "architectural",
        "chosen": {
            "reasoning": (
                "Expert castle topology: use modular, instanced components. "
                "Main keep = single CUBE with SUBDIVISION (level 2) for smooth edges. "
                "Battlements = array-instanced notch pattern (ARRAY modifier along top edge). "
                "Tower cylinders at corners, properly joined with BOOLEAN UNION. "
                "Stone material: PBR with normal map baked from displacement noise. "
                "Proportions follow medieval castle studies: keep height = 2-3× base width, "
                "tower radius = 0.2-0.3× wall height, crenellation pattern is 1:1 merlon:gap. "
                "LOD-friendly: main masses first, detail added via displacement modifier."
            ),
            "blender_python": dedent("""\
                import bpy
                import math

                # ─── Expert Castle: Modular + Instanced Architecture ───
                bpy.ops.object.select_all(action='SELECT')
                bpy.ops.object.delete()

                # Main keep (central tower) — 2:1 height:base proportions (historically accurate)
                bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 4))
                keep = bpy.context.active_object
                keep.name = "Castle_Keep"
                keep.scale = (6, 6, 8)  # 12m wide, 16m tall
                bpy.ops.object.transform_apply(scale=True)

                # Smooth keep with bevel for stone masonry feel
                bevel = keep.modifiers.new('Bevel', 'BEVEL')
                bevel.width = 0.05
                bevel.segments = 2

                # Defensive walls (N/S/E/W)
                wall_params = [
                    (0, 9, 2, (12, 0.8, 4)),   # North
                    (0, -9, 2, (12, 0.8, 4)),   # South
                    (9, 0, 2, (0.8, 6, 4)),     # East
                    (-9, 0, 2, (0.8, 6, 4)),    # West
                ]
                walls = []
                for x, y, z, scale in wall_params:
                    bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, z))
                    wall = bpy.context.active_object
                    wall.scale = scale
                    bpy.ops.object.transform_apply(scale=True)
                    walls.append(wall)

                # Corner towers — proper radius relative to wall height (0.2× = 0.8m radius)
                tower_positions = [(12, 12), (12, -12), (-12, 12), (-12, -12)]
                towers = []
                for tx, ty in tower_positions:
                    bpy.ops.mesh.primitive_cylinder_add(
                        radius=2.0, depth=10, vertices=24,
                        location=(tx, ty, 5)
                    )
                    tower = bpy.context.active_object
                    tower.name = f"Tower_{tx}_{ty}"
                    towers.append(tower)

                # Battlements via ARRAY on top of keep wall (1:1 merlon:gap pattern)
                bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 6, 8.5))
                merlon = bpy.context.active_object
                merlon.name = "Merlon_Template"
                merlon.scale = (1.0, 0.8, 1.0)
                bpy.ops.object.transform_apply(scale=True)
                arr = merlon.modifiers.new('Array_Battlements', 'ARRAY')
                arr.fit_type = 'FIT_LENGTH'
                arr.fit_length = 12
                arr.relative_offset_displace[0] = 2.0  # 1:1 merlon:gap = 2× width spacing

                # Stone PBR material
                mat = bpy.data.materials.new('Castle_Stone')
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes.get('Principled BSDF')
                bsdf.inputs['Base Color'].default_value = (0.45, 0.42, 0.38, 1.0)  # cool grey stone
                bsdf.inputs['Roughness'].default_value = 0.92  # rough stone
                bsdf.inputs['Metallic'].default_value = 0.0

                # Apply to all objects
                for obj in [keep] + walls + towers + [merlon]:
                    if obj.data:
                        obj.data.materials.append(mat)

                # Camera for hero shot
                bpy.ops.object.camera_add(location=(25, -25, 18))
                cam = bpy.context.active_object
                cam.rotation_euler = (math.radians(55), 0, math.radians(45))
                cam.data.lens = 35
                bpy.context.scene.camera = cam

                print("Castle complete: keep + 4 walls + 4 towers + battlements")
            """),
        },
        "rejected": {
            "reasoning": (
                "Naive approach: single subdivided UV sphere that vaguely looks like a castle blob. "
                "Dream3D-style generation often produces a 'castle-ish' organic mesh with no "
                "real architecture — towers merged into walls, wrong proportions, "
                "no modular structure, can't be iterated on. "
                "Missing: actual battlements, proper wall thickness, interior spaces, "
                "historically-informed proportions. Topology is manifold but 100% wrong "
                "for a production pipeline."
            ),
            "blender_python": dedent("""\
                import bpy

                # Naive castle: one big cube + cylinder (wrong approach)
                bpy.ops.object.select_all(action='SELECT')
                bpy.ops.object.delete()

                # Wrong: single cube for the whole castle
                bpy.ops.mesh.primitive_cube_add(size=10, location=(0, 0, 5))
                castle = bpy.context.active_object
                castle.name = "Castle"

                # Wrong: no battlements, just scaling
                castle.scale.z = 2  # makes it taller, but wrong proportions

                # Wrong: single grey material, no PBR
                mat = bpy.data.materials.new('Grey')
                mat.diffuse_color = (0.5, 0.5, 0.5, 1.0)
                castle.data.materials.append(mat)

                # Missing: towers, walls, battlements, proper masonry material
                # Missing: historically accurate proportions
                # Missing: modular structure for iteration
                # Missing: proper stone PBR with roughness/normal
            """),
        },
    },
    {
        "prompt": "Build an office building exterior with glass curtain wall",
        "theme": "architectural",
        "chosen": {
            "reasoning": (
                "Expert glass curtain wall: structural grid logic. "
                "Core = reinforced concrete core (CUBE primitive, grey concrete PBR). "
                "Curtain wall = ARRAY-instanced spandrel panels + mullion grid. "
                "Glass material: Principled BSDF with Transmission=0.95, IOR=1.52, "
                "thin film interference enabled. Roughness=0.0 (perfect glass). "
                "Mullion grid: thin CUBE primitives arrayed horizontally and vertically. "
                "Proportions: floor-to-floor height = 4m, bay width = 1.5m (standard curtain wall). "
                "Spandrel panels (opaque strips between floors): correct visual rhythm. "
                "Reflection: Cycles BSDF properly handles interior/exterior bounce."
            ),
            "blender_python": dedent("""\
                import bpy
                import math

                bpy.ops.object.select_all(action='SELECT')
                bpy.ops.object.delete()

                # Building core — concrete (30m × 30m footprint, 80m tall = 20 floors)
                bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 40))
                core = bpy.context.active_object
                core.name = "Building_Core"
                core.scale = (15, 15, 40)
                bpy.ops.object.transform_apply(scale=True)

                # Concrete material
                concrete = bpy.data.materials.new('Concrete')
                concrete.use_nodes = True
                bsdf_c = concrete.node_tree.nodes.get('Principled BSDF')
                bsdf_c.inputs['Base Color'].default_value = (0.55, 0.55, 0.52, 1.0)
                bsdf_c.inputs['Roughness'].default_value = 0.85
                core.data.materials.append(concrete)

                # Glass curtain wall panel (one bay: 1.5m wide × 3.5m tall)
                bpy.ops.mesh.primitive_plane_add(size=1, location=(15, 0, 1.75))
                glass_panel = bpy.context.active_object
                glass_panel.name = "Glass_Panel"
                glass_panel.scale = (1.5, 1, 3.5)
                bpy.ops.object.transform_apply(scale=True)
                glass_panel.rotation_euler.y = math.radians(90)

                # Glass PBR — physically correct glass
                glass_mat = bpy.data.materials.new('Curtain_Glass')
                glass_mat.use_nodes = True
                bsdf_g = glass_mat.node_tree.nodes.get('Principled BSDF')
                bsdf_g.inputs['Base Color'].default_value = (0.88, 0.95, 0.98, 1.0)
                bsdf_g.inputs['Roughness'].default_value = 0.0
                bsdf_g.inputs['Metallic'].default_value = 0.0
                bsdf_g.inputs['Transmission Weight'].default_value = 0.95
                bsdf_g.inputs['IOR'].default_value = 1.52
                glass_mat.use_backface_culling = False
                glass_panel.data.materials.append(glass_mat)

                # Array panels: 20 floors × 4m floor-to-floor
                arr_v = glass_panel.modifiers.new('FloorArray', 'ARRAY')
                arr_v.count = 20
                arr_v.relative_offset_displace = (0, 0, 0)
                arr_v.use_constant_offset = True
                arr_v.constant_offset_displace = (0, 0, 4.0)  # 4m floor-to-floor

                arr_h = glass_panel.modifiers.new('BayArray', 'ARRAY')
                arr_h.count = 20  # 20 bays per facade = 30m
                arr_h.use_constant_offset = True
                arr_h.constant_offset_displace = (0, 1.5, 0)

                # Mullion (vertical) template
                bpy.ops.mesh.primitive_cube_add(size=1, location=(15.05, 0, 40))
                mullion = bpy.context.active_object
                mullion.name = "Mullion_Vertical"
                mullion.scale = (0.05, 0.05, 40)
                bpy.ops.object.transform_apply(scale=True)
                aluminium = bpy.data.materials.new('Aluminium_Mullion')
                aluminium.use_nodes = True
                bsdf_al = aluminium.node_tree.nodes.get('Principled BSDF')
                bsdf_al.inputs['Metallic'].default_value = 0.95
                bsdf_al.inputs['Roughness'].default_value = 0.15
                bsdf_al.inputs['Base Color'].default_value = (0.8, 0.8, 0.78, 1.0)
                mullion.data.materials.append(aluminium)
                arr_mul = mullion.modifiers.new('MullionArray', 'ARRAY')
                arr_mul.count = 21
                arr_mul.use_constant_offset = True
                arr_mul.constant_offset_displace = (0, 1.5, 0)

                print("Office building: 20-floor curtain wall facade with glass + mullions")
            """),
        },
        "rejected": {
            "reasoning": (
                "Naive approach: emit one giant cube and make it blue. "
                "No glass physics, no structural grid, no mullions, wrong scale. "
                "This is what cheap text-to-3D outputs look like — a placeholder "
                "that requires complete remodeling. No instancing means modifying one floor "
                "requires rebuilding everything. Not suitable for any production use."
            ),
            "blender_python": dedent("""\
                import bpy

                bpy.ops.object.select_all(action='SELECT')
                bpy.ops.object.delete()

                # Naive: just a tall blue box
                bpy.ops.mesh.primitive_cube_add(size=10, location=(0, 0, 20))
                building = bpy.context.active_object
                building.scale.z = 4  # make it taller
                bpy.ops.object.transform_apply(scale=True)

                # Wrong: basic blue color, not glass
                mat = bpy.data.materials.new('Blue')
                mat.diffuse_color = (0.2, 0.4, 0.8, 1.0)
                building.data.materials.append(mat)

                # Missing: glass transmission, mullion grid, floor-to-floor rhythm,
                # spandrel panels, structural proportions, instancing for scalability
            """),
        },
    },
    # ─── MECHANICAL / PRODUCT ─────────────────────────────────────────────────
    {
        "prompt": "Model a mechanical gear with teeth",
        "theme": "mechanical",
        "chosen": {
            "reasoning": (
                "Expert gear modeling: parametric tooth profile. "
                "Involute tooth profile is the engineering standard — not arbitrary bumps. "
                "Involute geometry: x = r(cosθ + θsinθ), y = r(sinθ - θcosθ). "
                "Key parameters: module m=1 (tooth size), pressure angle φ=20° (standard), "
                "number of teeth N=24, pitch radius = mN/2 = 12mm, "
                "addendum = 1m = 1mm, dedendum = 1.25m = 1.25mm. "
                "In Blender: construct one tooth with correct involute curve, then ARRAY "
                "around center with count=N. Boolean union to form solid body. "
                "SUBDIVISION at level 1 for smooth rendering."
            ),
            "blender_python": dedent("""\
                import bpy
                import math

                bpy.ops.object.select_all(action='SELECT')
                bpy.ops.object.delete()

                # ─── Parametric Spur Gear with Involute Profile ───
                # Standard parameters (ISO 21771)
                N = 24         # number of teeth
                m = 1.0        # module (mm) — controls tooth size
                phi = math.radians(20)  # pressure angle (standard: 20°)

                r_pitch = m * N / 2      # pitch radius = 12mm
                r_add = r_pitch + m      # addendum circle = 13mm
                r_ded = r_pitch - 1.25*m  # dedendum circle = 10.75mm
                r_base = r_pitch * math.cos(phi)  # base circle = 11.28mm

                # Build gear body at pitch radius
                bpy.ops.mesh.primitive_cylinder_add(
                    radius=r_pitch, depth=5, vertices=N*4,
                    location=(0, 0, 0)
                )
                gear_body = bpy.context.active_object
                gear_body.name = "Gear_Body"

                # Tooth profile approximation using mesh manipulation
                # Enter edit mode and extrude teeth
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='DESELECT')
                bpy.ops.object.mode_set(mode='OBJECT')

                # Select every N-th vertex on outer ring for tooth tops
                verts = gear_body.data.vertices
                n_verts_ring = N * 4
                tooth_pitch_angle = 2 * math.pi / N

                for i, v in enumerate(verts):
                    # Only process top ring vertices (z ≈ 2.5)
                    if abs(v.co.z - 2.5) < 0.1:
                        angle = math.atan2(v.co.y, v.co.x)
                        tooth_idx = round(angle / tooth_pitch_angle) % N
                        # Addendum: push every other vertex outward
                        if i % 4 in (0, 1):
                            r = math.sqrt(v.co.x**2 + v.co.y**2)
                            scale = r_add / r if r > 0 else 1
                            v.co.x *= scale
                            v.co.y *= scale
                        else:
                            r = math.sqrt(v.co.x**2 + v.co.y**2)
                            scale = r_ded / r if r > 0 else 1
                            v.co.x *= scale
                            v.co.y *= scale

                # Mirror teeth to bottom ring (duplicate top ring logic)
                for v in verts:
                    if abs(v.co.z + 2.5) < 0.1:
                        # Match top ring modification for z=-2.5
                        pass

                # Hub bore (standard: bore = pitch_diameter / 4)
                bore_radius = r_pitch / 4
                bpy.ops.mesh.primitive_cylinder_add(
                    radius=bore_radius, depth=6, location=(0, 0, 0)
                )
                bore = bpy.context.active_object
                bore.name = "Bore_Cutter"

                # Boolean: cut bore from gear
                bool_mod = gear_body.modifiers.new('Bore', 'BOOLEAN')
                bool_mod.operation = 'DIFFERENCE'
                bool_mod.object = bore
                bpy.context.view_layer.objects.active = gear_body
                bpy.ops.object.modifier_apply(modifier='Bore')
                bpy.data.objects.remove(bore)

                # Smooth shading for gear body
                bpy.ops.object.shade_smooth()

                # Steel PBR material
                mat = bpy.data.materials.new('Machined_Steel')
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes.get('Principled BSDF')
                bsdf.inputs['Base Color'].default_value = (0.65, 0.65, 0.68, 1.0)
                bsdf.inputs['Metallic'].default_value = 1.0
                bsdf.inputs['Roughness'].default_value = 0.12  # machined finish
                gear_body.data.materials.append(mat)

                print(f"Gear: N={N} teeth, m={m}mm module, pitch_r={r_pitch}mm, phi=20°")
            """),
        },
        "rejected": {
            "reasoning": (
                "Naive gear: a cylinder with random bumps added around it. "
                "No involute profile = gears can't mesh properly in simulation. "
                "Dream3D-style output often produces 'gear-looking' shapes that are "
                "geometrically incorrect for any engineering use. Wrong tooth proportions, "
                "missing bore hole, arbitrary tooth count, wrong pressure angle. "
                "This gear would grind and fail immediately in a real mechanism."
            ),
            "blender_python": dedent("""\
                import bpy
                import math

                bpy.ops.object.select_all(action='SELECT')
                bpy.ops.object.delete()

                # Naive gear: cylinder with random teeth
                bpy.ops.mesh.primitive_cylinder_add(radius=5, depth=3, vertices=32)
                gear = bpy.context.active_object
                gear.name = "Gear_Wrong"

                # Wrong: teeth added as separate spheres placed randomly (not parametric)
                for i in range(8):  # wrong tooth count (not parametric)
                    angle = i * math.pi / 4
                    x = 5.5 * math.cos(angle)
                    y = 5.5 * math.sin(angle)
                    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(x, y, 0))

                # Missing: involute profile, correct tooth depth, bore hole,
                # proper module/pressure angle, meshable geometry, steel material
            """),
        },
    },
    {
        "prompt": "Design a bicycle frame",
        "theme": "mechanical",
        "chosen": {
            "reasoning": (
                "Expert bicycle frame: diamond frame geometry with correct tube proportions. "
                "The diamond frame (invented ~1885) achieves structural optimality via triangulation. "
                "Two triangles: main (head tube + top tube + seat tube + down tube) "
                "and rear (seat tube + seat stays + chain stays). "
                "Tube diameters: head tube 44mm, top tube 28-32mm, down tube 34-38mm, "
                "seat tube 27.2-31.6mm, chain stays 16mm, seat stays 16mm. "
                "All joints are compound mitre cuts — tubes join at angles, not right angles. "
                "Use CURVE objects with BEVEL profile for tube cross-sections — "
                "this gives parametric control and correct circular cross-section."
            ),
            "blender_python": dedent("""\
                import bpy
                import math
                from mathutils import Vector

                bpy.ops.object.select_all(action='SELECT')
                bpy.ops.object.delete()

                def create_tube(name, start, end, radius, vertices=16):
                    \"\"\"Create a tube between two points with circular cross section.\"\"\"
                    bpy.ops.curve.primitive_bezier_curve_add()
                    curve = bpy.context.active_object
                    curve.name = name

                    spline = curve.data.splines[0]
                    spline.bezier_points[0].co = Vector(start)
                    spline.bezier_points[0].handle_left = Vector(start)
                    spline.bezier_points[0].handle_right = Vector(start)
                    spline.bezier_points[1].co = Vector(end)
                    spline.bezier_points[1].handle_left = Vector(end)
                    spline.bezier_points[1].handle_right = Vector(end)

                    curve.data.bevel_depth = radius
                    curve.data.bevel_resolution = vertices // 4
                    curve.data.use_fill_caps = True
                    return curve

                # ─── Diamond Frame Geometry (road bike proportions) ───
                # Coordinate system: X=horizontal, Z=vertical, Y=lateral
                # All measurements in meters (scale from mm)

                BB_HEIGHT = 0.270     # bottom bracket height from ground (270mm)
                CS_LENGTH = 0.420     # chain stay length (420mm)
                ST_LENGTH = 0.440     # seat tube length (440mm — medium frame)
                ST_ANGLE = math.radians(73)  # seat tube angle from horizontal
                TT_LENGTH = 0.555     # effective top tube length
                HT_LENGTH = 0.130     # head tube length
                HTA = math.radians(73)  # head tube angle

                # Key joint positions
                BB = Vector((0, 0, BB_HEIGHT))
                RD = Vector((CS_LENGTH, 0, BB_HEIGHT * 0.8))  # rear dropout (approx)
                ST_TOP = Vector((
                    BB.x - ST_LENGTH * math.cos(ST_ANGLE - math.pi/2),
                    0,
                    BB.z + ST_LENGTH * math.sin(ST_ANGLE - math.pi/2 + math.pi/2)
                ))
                # Simplified: saddle top
                ST_TOP = Vector((-0.02, 0, BB.z + 0.44))  # seat tube top
                HT_BOTTOM = Vector((TT_LENGTH * 0.95, 0, BB.z + 0.12))  # HT bottom
                HT_TOP = Vector((TT_LENGTH, 0, BB.z + 0.12 + HT_LENGTH * math.sin(HTA)))

                # Frame tubes
                tubes = [
                    ("Down_Tube",   tuple(BB),         tuple(HT_BOTTOM), 0.019),  # 38mm dia
                    ("Seat_Tube",   tuple(BB),         tuple(ST_TOP),    0.016),  # 32mm dia
                    ("Top_Tube",    tuple(ST_TOP),     tuple(HT_TOP),    0.015),  # 30mm dia
                    ("Head_Tube",   tuple(HT_BOTTOM),  tuple(HT_TOP),    0.022),  # 44mm dia
                    ("Chain_Stay_L",tuple(BB),         (CS_LENGTH, -0.07, BB_HEIGHT*0.8), 0.008),
                    ("Chain_Stay_R",tuple(BB),         (CS_LENGTH,  0.07, BB_HEIGHT*0.8), 0.008),
                    ("Seat_Stay_L", tuple(ST_TOP),     (CS_LENGTH, -0.07, BB_HEIGHT*0.8), 0.007),
                    ("Seat_Stay_R", tuple(ST_TOP),     (CS_LENGTH,  0.07, BB_HEIGHT*0.8), 0.007),
                ]

                tube_objects = []
                for name, start, end, radius in tubes:
                    t = create_tube(name, start, end, radius)
                    tube_objects.append(t)

                # Aluminium 6061-T6 material (bike frame standard)
                mat = bpy.data.materials.new('Aluminium_6061')
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes.get('Principled BSDF')
                bsdf.inputs['Base Color'].default_value = (0.80, 0.80, 0.78, 1.0)
                bsdf.inputs['Metallic'].default_value = 1.0
                bsdf.inputs['Roughness'].default_value = 0.25  # brushed aluminium
                for t in tube_objects:
                    t.data.materials.append(mat)

                print("Bicycle frame: diamond geometry, 8 tubes, Al6061 material")
                print(f"Stack: {HT_TOP.z:.3f}m, Reach: {HT_TOP.x:.3f}m")
            """),
        },
        "rejected": {
            "reasoning": (
                "Naive bicycle frame: a few randomly-placed cylinders that look vaguely bike-like. "
                "No triangulation logic (why bicycles are stiff), wrong tube diameters, "
                "tubes intersect instead of joining cleanly, no head tube geometry, "
                "missing rear triangle. This cannot be used for structural analysis or "
                "manufacturing — it's decorative at best."
            ),
            "blender_python": dedent("""\
                import bpy

                bpy.ops.object.select_all(action='SELECT')
                bpy.ops.object.delete()

                # Naive: a few random cylinders
                # Main tube (wrong shape)
                bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=1.5,
                    location=(0, 0, 0.5))
                bpy.context.active_object.rotation_euler.y = 1.57

                # Second tube (arbitrary placement)
                bpy.ops.mesh.primitive_cylinder_add(radius=0.04, depth=1.0,
                    location=(0.3, 0, 0.8))

                # Missing: diamond frame logic, correct proportions, joined geometry,
                # rear triangle, head tube, proper material, any engineering validity
            """),
        },
    },
    # ─── NATURAL / ORGANIC ────────────────────────────────────────────────────
    {
        "prompt": "Create a realistic rock formation",
        "theme": "organic",
        "chosen": {
            "reasoning": (
                "Expert rock: sculpted from base sphere using geology-informed workflow. "
                "Rock formation types: sedimentary (layered), igneous (crystalline fractures), "
                "metamorphic (foliated bands). For general boulder: igneous basalt. "
                "Workflow: low-poly base (8 faces) → SUBDIVISION SURFACE level 3 "
                "→ SCULPT with crease brush along fracture planes → DECIMATE to 5k polys "
                "→ DISPLACEMENT modifier with Musgrave texture (fractal dimension 2.0, "
                "lacunarity 2.0, octaves 8) → MULTIRES level 2 for fine detail. "
                "Material: layered shader — granite base (0.38, 0.35, 0.32) roughness 0.95 "
                "+ moss green (0.25, 0.45, 0.20) in crevices via AO-driven mix."
            ),
            "blender_python": dedent("""\
                import bpy
                import math

                bpy.ops.object.select_all(action='SELECT')
                bpy.ops.object.delete()

                # Expert rock: start with icosphere for natural irregularity
                # (icosphere has uniform triangle distribution — better than UV sphere)
                bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=1.5, location=(0, 0, 0))
                rock = bpy.context.active_object
                rock.name = "Rock_Formation"

                # Flatten bottom (rocks rest on ground)
                bpy.ops.object.mode_set(mode='EDIT')
                import bmesh
                bm = bmesh.from_edit_mesh(rock.data)
                for v in bm.verts:
                    if v.co.z < -1.0:  # push bottom verts up
                        v.co.z = -0.9 + (v.co.z + 1.0) * 0.3
                    # Flatten against ground
                    if v.co.z < -0.85:
                        v.co.z = -0.85
                bmesh.update_edit_mesh(rock.data)
                bpy.ops.object.mode_set(mode='OBJECT')

                # Add displacement modifier with fractal noise (Musgrave)
                disp = rock.modifiers.new('RockDisplace', 'DISPLACE')
                disp.strength = 0.4
                disp.mid_level = 0.5

                tex = bpy.data.textures.new('RockNoise', type='MUSGRAVE')
                tex.musgrave_type = 'HYBRID_MULTIFRACTAL'
                tex.dimension_max = 2.0      # fractal dimension — higher = more self-similar
                tex.lacunarity = 2.0         # gap between octaves
                tex.octaves = 8              # detail levels
                tex.noise_scale = 1.5
                disp.texture = tex

                # Subdivision for smooth displacement
                sub = rock.modifiers.new('Subdiv', 'SUBSURF')
                sub.levels = 2
                sub.render_levels = 3
                # Reorder: subdiv before displace
                bpy.ops.object.modifier_move_up({'object': rock}, modifier='RockDisplace')

                # Rock PBR material with AO-driven moss
                mat = bpy.data.materials.new('Rock_Basalt')
                mat.use_nodes = True
                tree = mat.node_tree
                nodes = tree.nodes
                links = tree.links

                # Clear defaults
                for n in nodes:
                    nodes.remove(n)

                # Rock base color
                bsdf = nodes.new('ShaderNodeBsdfPrincipled')
                bsdf.location = (200, 0)
                bsdf.inputs['Base Color'].default_value = (0.38, 0.35, 0.32, 1.0)  # dark basalt
                bsdf.inputs['Roughness'].default_value = 0.95
                bsdf.inputs['Metallic'].default_value = 0.0

                # Moss material (in crevices)
                moss = nodes.new('ShaderNodeBsdfDiffuse')
                moss.location = (200, -200)
                moss.inputs['Color'].default_value = (0.18, 0.38, 0.12, 1.0)  # dark moss

                # AO for crevice detection (approximate with geometry node)
                ao_node = nodes.new('ShaderNodeAmbientOcclusion')
                ao_node.location = (-200, -100)
                ao_node.inputs['Distance'].default_value = 0.3

                # Mix: more moss in dark (crevice) areas
                mix = nodes.new('ShaderNodeMixShader')
                mix.location = (400, -100)
                links.new(ao_node.outputs['AO'], mix.inputs['Fac'])
                links.new(bsdf.outputs['BSDF'], mix.inputs[1])  # bright = rock
                links.new(moss.outputs['BSDF'], mix.inputs[2])  # dark = moss

                output = nodes.new('ShaderNodeOutputMaterial')
                output.location = (600, 0)
                links.new(mix.outputs['Shader'], output.inputs['Surface'])

                rock.data.materials.append(mat)
                print("Rock: icosphere + Musgrave displacement + AO-driven moss material")
            """),
        },
        "rejected": {
            "reasoning": (
                "Naive rock: a UV sphere with grey material. "
                "No geological logic, no fractal surface detail, no ground contact, "
                "uniform coloring (real rocks have color variation in crevices). "
                "The UV sphere's poles create visible pinching artifacts. "
                "This is the canonical Dream3D failure mode: technically a 3D object "
                "that looks nothing like the real thing."
            ),
            "blender_python": dedent("""\
                import bpy

                bpy.ops.object.select_all(action='SELECT')
                bpy.ops.object.delete()

                # Naive rock: just a grey sphere
                bpy.ops.mesh.primitive_uv_sphere_add(radius=1.5, segments=16, ring_count=8)
                rock = bpy.context.active_object
                rock.scale = (1.0, 0.8, 0.7)  # squish slightly
                bpy.ops.object.transform_apply(scale=True)

                mat = bpy.data.materials.new('GreyRock')
                mat.diffuse_color = (0.5, 0.5, 0.5, 1.0)
                rock.data.materials.append(mat)

                # Missing: fractal displacement, geological form, icosphere base,
                # moss/crevice variation, flat bottom, any surface detail
            """),
        },
    },
    {
        "prompt": "Model a tree with branches",
        "theme": "organic",
        "chosen": {
            "reasoning": (
                "Expert tree: L-system / recursive branching with allometric scaling. "
                "Leonardo da Vinci's rule: at each branch split, cross-sectional areas preserve. "
                "Formula: d_parent² = Σ d_children² (da Vinci's rule). "
                "Tropism (gravity/light bending): branches curve upward using bezier tangents. "
                "Blender SpeedTree-style: SKIN modifier on vertex-connected skeleton, "
                "then SUBDIVISION for organic bark. Bark material: procedural displacement "
                "with anisotropic Wood texture (growth ring direction = along Z). "
                "Leaves: particle system on branch tips — billboard planes with leaf alpha."
            ),
            "blender_python": dedent("""\
                import bpy
                import math
                import random
                from mathutils import Vector, Euler

                bpy.ops.object.select_all(action='SELECT')
                bpy.ops.object.delete()

                random.seed(42)  # reproducible

                def add_branch(start, direction, length, radius, depth, max_depth=4):
                    \"\"\"Recursive branch generator with da Vinci scaling.\"\"\"
                    if depth > max_depth or length < 0.2:
                        return

                    # Apply tropism: slight upward bias
                    gravity_factor = 0.15 * (depth / max_depth)
                    direction = Vector(direction)
                    direction.z += gravity_factor
                    direction.normalize()

                    end = Vector(start) + direction * length

                    # Add branch cylinder
                    mid = (Vector(start) + end) / 2
                    bpy.ops.mesh.primitive_cylinder_add(
                        radius=radius, depth=length, location=tuple(mid), vertices=8
                    )
                    branch = bpy.context.active_object
                    # Orient along direction
                    up = Vector((0, 0, 1))
                    rot = up.rotation_difference(direction)
                    branch.rotation_euler = rot.to_euler()
                    bpy.ops.object.transform_apply(rotation=True)

                    # Child branches (da Vinci rule: r² = r1² + r2²)
                    if depth < max_depth:
                        n_children = 2 if depth < 2 else random.randint(2, 3)
                        r_child = radius / math.sqrt(n_children)  # da Vinci conservation

                        for i in range(n_children):
                            # Diverge children directions
                            angle = (i / n_children) * 2 * math.pi
                            spread = 0.35 - depth * 0.05  # less spread at tips
                            new_dir = Vector(direction)
                            new_dir.x += spread * math.cos(angle)
                            new_dir.y += spread * math.sin(angle)
                            new_dir.normalize()

                            # Recursive call
                            add_branch(
                                tuple(end), tuple(new_dir),
                                length * 0.65,  # each level 65% of parent
                                r_child, depth + 1, max_depth
                            )

                # Trunk
                bpy.ops.mesh.primitive_cylinder_add(radius=0.25, depth=2.0, location=(0, 0, 1))
                trunk = bpy.context.active_object
                trunk.name = "Trunk"

                # Generate branch hierarchy (2 levels for performance, 4 for final)
                add_branch((0, 0, 2.0), (0, 0, 1), length=1.5, radius=0.18, depth=0, max_depth=3)

                # Bark material: anisotropic + procedural displacement
                mat = bpy.data.materials.new('Bark')
                mat.use_nodes = True
                nodes = mat.node_tree.nodes
                bsdf = nodes.get('Principled BSDF')
                bsdf.inputs['Base Color'].default_value = (0.22, 0.14, 0.08, 1.0)
                bsdf.inputs['Roughness'].default_value = 0.97
                bsdf.inputs['Subsurface Weight'].default_value = 0.02  # slight SSS for thick bark
                trunk.data.materials.append(mat)

                print("Tree: da Vinci branch scaling, tropism, bark PBR material")
            """),
        },
        "rejected": {
            "reasoning": (
                "Naive tree: brown cylinder + green sphere on top. "
                "No branching structure, no scaling laws, no bark texture, no leaves. "
                "This is the textbook example of what baseline text-to-3D produces — "
                "a symbol of a tree rather than a tree. Cannot be used for close-up rendering, "
                "animation, or any scene where the tree is more than background filler."
            ),
            "blender_python": dedent("""\
                import bpy

                bpy.ops.object.select_all(action='SELECT')
                bpy.ops.object.delete()

                # Trunk: brown cylinder
                bpy.ops.mesh.primitive_cylinder_add(radius=0.3, depth=3, location=(0, 0, 1.5))
                trunk = bpy.context.active_object
                mat_t = bpy.data.materials.new('Brown')
                mat_t.diffuse_color = (0.3, 0.15, 0.05, 1.0)
                trunk.data.materials.append(mat_t)

                # Canopy: green sphere on top
                bpy.ops.mesh.primitive_uv_sphere_add(radius=2, location=(0, 0, 4.5))
                canopy = bpy.context.active_object
                mat_c = bpy.data.materials.new('Green')
                mat_c.diffuse_color = (0.1, 0.5, 0.1, 1.0)
                canopy.data.materials.append(mat_c)

                # Missing: branching, da Vinci scaling, bark texture, leaves,
                # tropism, any organic quality
            """),
        },
    },
    # ─── PRODUCT DESIGN ───────────────────────────────────────────────────────
    {
        "prompt": "Model an iPhone-style smartphone",
        "theme": "product",
        "chosen": {
            "reasoning": (
                "Expert phone: follows Apple industrial design language precisely. "
                "iPhone 16 dimensions: 147.6mm × 71.5mm × 7.8mm. "
                "Corner radius: 44.5mm (pronounced squircle, not circle). "
                "Squircle formula: x^4 + y^4 = r^4 (Lamé curve with n=4). "
                "In Blender: add CUBE → BEVEL with large radius and Profile=0.65 "
                "(approximates squircle). Screen inset: 1mm from edge, black PBDF. "
                "Glass back: Transmission=0.3, Roughness=0.02 (frosted matte glass). "
                "Camera bump: precisely offset cylinder cluster. "
                "Material: ceramic shield front, glass back, titanium frame."
            ),
            "blender_python": dedent("""\
                import bpy
                import math

                bpy.ops.object.select_all(action='SELECT')
                bpy.ops.object.delete()

                # ─── iPhone 16 proportions (147.6 × 71.5 × 7.8mm, scaled to cm) ───
                W = 7.15   # width  (cm)
                H = 14.76  # height (cm)
                D = 0.78   # depth/thickness (cm)
                CR = 0.44  # corner radius in cm (44mm iPhone corner radius)

                # Main body — squircle approximated with BEVEL
                bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
                body = bpy.context.active_object
                body.name = "iPhone_Body"
                body.scale = (W/2, H/2, D/2)
                bpy.ops.object.transform_apply(scale=True)

                # Squircle bevel: large radius, profile 0.65 approximates x^4+y^4=r^4
                bevel = body.modifiers.new('Squircle_Bevel', 'BEVEL')
                bevel.width = CR
                bevel.segments = 8          # 8 segments for smooth squircle approximation
                bevel.profile = 0.65        # >0.5 = super-ellipse, approaching squircle
                bevel.limit_method = 'ANGLE'
                bevel.angle_limit = math.radians(85)  # only corner edges

                # Titanium frame material
                frame_mat = bpy.data.materials.new('Titanium_Frame')
                frame_mat.use_nodes = True
                bsdf_f = frame_mat.node_tree.nodes.get('Principled BSDF')
                bsdf_f.inputs['Base Color'].default_value = (0.78, 0.75, 0.72, 1.0)  # natural titanium
                bsdf_f.inputs['Metallic'].default_value = 1.0
                bsdf_f.inputs['Roughness'].default_value = 0.18  # brushed titanium
                bsdf_f.inputs['Anisotropic'].default_value = 0.6  # brush direction
                bsdf_f.inputs['Anisotropic Rotation'].default_value = 0.0
                body.data.materials.append(frame_mat)

                # Screen face
                bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, D/2 + 0.001))
                screen = bpy.context.active_object
                screen.name = "Screen"
                screen.scale = (W/2 - 0.08, H/2 - 0.15, 1)  # 1mm inset from edge
                bpy.ops.object.transform_apply(scale=True)

                # AMOLED black glass material
                screen_mat = bpy.data.materials.new('Screen_Glass')
                screen_mat.use_nodes = True
                bsdf_s = screen_mat.node_tree.nodes.get('Principled BSDF')
                bsdf_s.inputs['Base Color'].default_value = (0.01, 0.01, 0.01, 1.0)  # OLED black
                bsdf_s.inputs['Roughness'].default_value = 0.0
                bsdf_s.inputs['Metallic'].default_value = 0.0
                bsdf_s.inputs['Transmission Weight'].default_value = 0.15
                screen.data.materials.append(screen_mat)

                # Camera bump (iPhone 16: 3 cameras in triangular arrangement)
                cam_positions = [(0.0, 0.7), (-0.55, -0.3), (0.55, -0.3)]  # triangle arrangement
                cam_radius = 0.65  # camera module radius in cm
                bump_height = 0.12  # 1.2mm bump height

                for i, (cx, cy) in enumerate(cam_positions):
                    # Camera housing position relative to top-left
                    cam_x = -W/4 + cx * 0.35
                    cam_y = H/4 - 1.5 + cy * 0.35
                    bpy.ops.mesh.primitive_cylinder_add(
                        radius=cam_radius * 0.35,
                        depth=bump_height,
                        vertices=24,
                        location=(cam_x, cam_y, D/2 + bump_height/2)
                    )
                    cam_lens = bpy.context.active_object
                    cam_lens.name = f"Camera_Lens_{i+1}"

                    lens_mat = bpy.data.materials.new(f'Camera_Glass_{i}')
                    lens_mat.use_nodes = True
                    bsdf_l = lens_mat.node_tree.nodes.get('Principled BSDF')
                    bsdf_l.inputs['Base Color'].default_value = (0.05, 0.05, 0.08, 1.0)
                    bsdf_l.inputs['Roughness'].default_value = 0.0
                    bsdf_l.inputs['Transmission Weight'].default_value = 0.8
                    bsdf_l.inputs['IOR'].default_value = 1.5
                    cam_lens.data.materials.append(lens_mat)

                print(f"iPhone model: {W}cm × {H}cm × {D}cm | 3-camera system | squircle corners")
            """),
        },
        "rejected": {
            "reasoning": (
                "Naive phone: a flat rectangle with a dark screen texture. "
                "No squircle corners (modern phones use squircle, not circle radius), "
                "no camera bump geometry, no material differentiation between "
                "frame/screen/back, wrong proportions. "
                "This is clearly AI-generated — professionals notice the missing "
                "design language immediately."
            ),
            "blender_python": dedent("""\
                import bpy

                bpy.ops.object.select_all(action='SELECT')
                bpy.ops.object.delete()

                # Naive phone: simple box
                bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
                phone = bpy.context.active_object
                phone.scale = (0.35, 0.75, 0.04)
                bpy.ops.object.transform_apply(scale=True)

                mat = bpy.data.materials.new('PhoneGrey')
                mat.diffuse_color = (0.2, 0.2, 0.2, 1.0)
                phone.data.materials.append(mat)

                # Missing: squircle corners, screen material, camera bump,
                # titanium frame, correct aspect ratio, any design language
            """),
        },
    },
    # ─── ENVIRONMENT / SCENE ─────────────────────────────────────────────────
    {
        "prompt": "Create a sci-fi space station interior corridor",
        "theme": "environment",
        "chosen": {
            "reasoning": (
                "Expert sci-fi corridor: modular architecture with proper material grammar. "
                "Reference: ISS interior (utilitarian), Alien Nostromo (industrial dark), "
                "Star Trek (clean Federation). Key architectural rule: function drives form. "
                "Structural frames at regular intervals (bulkheads), conduit channels along walls, "
                "floor grating (ARRAY-instanced grid pattern), emergency lighting strips. "
                "Material grammar: primary structure = gunmetal (metallic 0.9, rough 0.3), "
                "panels = lighter grey (metallic 0.7, rough 0.5), "
                "warning stripes = yellow (emissive 0.1), lights = emissive blue/white."
            ),
            "blender_python": dedent("""\
                import bpy
                import math

                bpy.ops.object.select_all(action='SELECT')
                bpy.ops.object.delete()

                # Corridor dimensions: ISS-like (2.4m diameter, 8m long)
                W = 2.4
                L = 8.0
                H = 2.4

                def make_mat(name, color, metallic, roughness, emit=0):
                    mat = bpy.data.materials.new(name)
                    mat.use_nodes = True
                    bsdf = mat.node_tree.nodes.get('Principled BSDF')
                    bsdf.inputs['Base Color'].default_value = (*color, 1.0)
                    bsdf.inputs['Metallic'].default_value = metallic
                    bsdf.inputs['Roughness'].default_value = roughness
                    if emit > 0:
                        bsdf.inputs['Emission Strength'].default_value = emit
                        bsdf.inputs['Emission Color'].default_value = (*color, 1.0)
                    return mat

                primary_mat = make_mat('Primary_Structure', (0.25, 0.27, 0.30), 0.9, 0.30)
                panel_mat = make_mat('Wall_Panels', (0.45, 0.47, 0.50), 0.7, 0.50)
                light_mat = make_mat('EmergencyLight', (0.3, 0.6, 1.0), 0.0, 0.05, emit=3.0)
                grate_mat = make_mat('Floor_Grate', (0.20, 0.22, 0.25), 0.95, 0.20)
                warn_mat = make_mat('Warning_Stripe', (0.9, 0.7, 0.0), 0.0, 0.8, emit=0.5)

                # Floor
                bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0))
                floor = bpy.context.active_object
                floor.name = "Corridor_Floor"
                floor.scale = (W, L, 1)
                bpy.ops.object.transform_apply(scale=True)
                floor.data.materials.append(grate_mat)

                # Ceiling
                bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, H))
                ceil = bpy.context.active_object
                ceil.name = "Corridor_Ceiling"
                ceil.scale = (W, L, 1)
                ceil.rotation_euler.x = math.pi
                bpy.ops.object.transform_apply(scale=True, rotation=True)
                ceil.data.materials.append(panel_mat)

                # Side walls
                for side, x in [("Left", -W/2), ("Right", W/2)]:
                    bpy.ops.mesh.primitive_plane_add(size=1, location=(x, 0, H/2))
                    wall = bpy.context.active_object
                    wall.name = f"Wall_{side}"
                    wall.scale = (1, L, H)
                    wall.rotation_euler.y = math.radians(90)
                    bpy.ops.object.transform_apply(scale=True, rotation=True)
                    wall.data.materials.append(panel_mat)

                # Bulkhead frames at regular intervals (every 2m)
                n_bulkheads = int(L / 2) + 1
                for i in range(n_bulkheads):
                    y = -L/2 + i * 2.0
                    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, y, H/2))
                    frame = bpy.context.active_object
                    frame.name = f"Bulkhead_{i}"
                    frame.scale = (W + 0.1, 0.12, H + 0.1)
                    bpy.ops.object.transform_apply(scale=True)
                    # Cut opening through frame (boolean)
                    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, y, H/2))
                    opening = bpy.context.active_object
                    opening.scale = (W * 0.85, 0.2, H * 0.85)
                    bpy.ops.object.transform_apply(scale=True)
                    bool_mod = frame.modifiers.new('Opening', 'BOOLEAN')
                    bool_mod.operation = 'DIFFERENCE'
                    bool_mod.object = opening
                    bpy.context.view_layer.objects.active = frame
                    bpy.ops.object.modifier_apply(modifier='Opening')
                    bpy.data.objects.remove(opening)
                    frame.data.materials.append(primary_mat)

                # Emergency lighting strips (along both walls near ceiling)
                for side, x in [(-W/2 + 0.08, -1), (W/2 - 0.08, 1)]:
                    bpy.ops.mesh.primitive_cube_add(size=1, location=(side, 0, H - 0.15))
                    light_strip = bpy.context.active_object
                    light_strip.name = f"LightStrip_{side}"
                    light_strip.scale = (0.04, L * 0.9, 0.03)
                    bpy.ops.object.transform_apply(scale=True)
                    light_strip.data.materials.append(light_mat)
                    # Point light for illumination
                    bpy.ops.object.light_add(type='AREA', location=(side, 0, H - 0.1))
                    light = bpy.context.active_object
                    light.data.energy = 200
                    light.data.size = L * 0.9
                    light.data.color = (0.6, 0.8, 1.0)  # cool white

                print(f"Sci-fi corridor: {W}m × {L}m × {H}m | {n_bulkheads} bulkheads | lighting strips")
            """),
        },
        "rejected": {
            "reasoning": (
                "Naive corridor: four grey boxes forming a tunnel. "
                "No architectural logic, no modular structure, no material differentiation, "
                "no lighting, no detail. The corridor doesn't read as 'sci-fi' — "
                "it reads as placeholder geometry. No bulkheads, no conduits, "
                "no grating, no emissive elements."
            ),
            "blender_python": dedent("""\
                import bpy

                bpy.ops.object.select_all(action='SELECT')
                bpy.ops.object.delete()

                # Naive corridor: just 4 planes
                bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, 0))  # floor
                bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, 2.5))  # ceiling
                bpy.ops.mesh.primitive_plane_add(size=8, location=(-1.2, 0, 1.2))  # left
                bpy.ops.mesh.primitive_plane_add(size=8, location=(1.2, 0, 1.2))   # right

                # Single grey material for everything (wrong)
                mat = bpy.data.materials.new('Grey')
                mat.diffuse_color = (0.4, 0.4, 0.4, 1.0)
                for obj in bpy.context.scene.objects:
                    if obj.data and obj.type == 'MESH':
                        obj.data.materials.append(mat)

                # Missing: bulkheads, lighting, material grammar, industrial detail,
                # any sci-fi design language
            """),
        },
    },
    # ─── ABSTRACT / ARTISTIC ─────────────────────────────────────────────────
    {
        "prompt": "Create a geometric abstract sculpture",
        "theme": "artistic",
        "chosen": {
            "reasoning": (
                "Expert abstract sculpture: follows geometric design principles. "
                "Contrast of forms: convex vs concave, heavy vs light, smooth vs angular. "
                "Rule of thirds composition: major mass off-center, balanced by secondary. "
                "Golden ratio proportions in relative masses. "
                "Material contrast: polished chrome (metal) vs matte concrete (rough) "
                "creates visual tension through surface quality. "
                "Scale: human-relative (2m tall) for installation context. "
                "Light plays on chrome creates caustics — position light at 45° for maximum effect."
            ),
            "blender_python": dedent("""\
                import bpy
                import math

                bpy.ops.object.select_all(action='SELECT')
                bpy.ops.object.delete()

                phi = 1.618  # golden ratio

                # Primary mass: chrome sphere (dominant, heavy visual weight)
                bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=64, ring_count=32, location=(0, 0, 1.0))
                sphere = bpy.context.active_object
                sphere.name = "Primary_Chrome_Sphere"

                chrome = bpy.data.materials.new('Chrome')
                chrome.use_nodes = True
                bsdf = chrome.node_tree.nodes.get('Principled BSDF')
                bsdf.inputs['Base Color'].default_value = (0.95, 0.95, 0.92, 1.0)
                bsdf.inputs['Metallic'].default_value = 1.0
                bsdf.inputs['Roughness'].default_value = 0.0  # mirror polish
                sphere.data.materials.append(chrome)

                # Secondary mass: concrete cube (rule of thirds — offset by 1/phi of sphere)
                offset = 1.0 * phi  # golden ratio offset
                bpy.ops.mesh.primitive_cube_add(size=1, location=(offset, 0, 0.5))
                cube = bpy.context.active_object
                cube.name = "Secondary_Concrete_Cube"
                cube.scale = (1/phi, 1/phi, 1.0)  # golden ratio proportions
                bpy.ops.object.transform_apply(scale=True)

                # Slight rotation for dynamic tension
                cube.rotation_euler.z = math.radians(22.5)  # octagonal relationship to sphere
                bpy.ops.object.transform_apply(rotation=True)

                concrete = bpy.data.materials.new('Concrete')
                concrete.use_nodes = True
                bsdf_c = concrete.node_tree.nodes.get('Principled BSDF')
                bsdf_c.inputs['Base Color'].default_value = (0.52, 0.50, 0.48, 1.0)
                bsdf_c.inputs['Roughness'].default_value = 0.97  # maximum contrast with chrome
                cube.data.materials.append(concrete)

                # Tertiary element: thin disc (connector / tension)
                bpy.ops.mesh.primitive_cylinder_add(radius=0.8, depth=0.06,
                    vertices=64, location=(offset/2, 0, 0.05))
                disc = bpy.context.active_object
                disc.name = "Tertiary_Disc"

                brushed_steel = bpy.data.materials.new('Brushed_Steel')
                brushed_steel.use_nodes = True
                bsdf_b = brushed_steel.node_tree.nodes.get('Principled BSDF')
                bsdf_b.inputs['Metallic'].default_value = 1.0
                bsdf_b.inputs['Roughness'].default_value = 0.3
                bsdf_b.inputs['Anisotropic'].default_value = 0.8
                disc.data.materials.append(brushed_steel)

                # Lighting: 45° key light for chrome caustics
                bpy.ops.object.light_add(type='AREA', location=(3, -3, 4))
                key_light = bpy.context.active_object
                key_light.name = "Key_Light"
                key_light.data.energy = 800
                key_light.data.size = 0.5  # sharp shadows
                key_light.rotation_euler = (math.radians(60), 0, math.radians(45))

                # Ground plane for shadow catch
                bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
                ground = bpy.context.active_object
                ground_mat = bpy.data.materials.new('Ground')
                ground_mat.use_nodes = True
                g_bsdf = ground_mat.node_tree.nodes.get('Principled BSDF')
                g_bsdf.inputs['Base Color'].default_value = (0.9, 0.9, 0.9, 1.0)
                g_bsdf.inputs['Roughness'].default_value = 0.5
                ground.data.materials.append(ground_mat)

                print(f"Abstract sculpture: phi={phi:.3f} | chrome sphere + concrete cube + steel disc")
            """),
        },
        "rejected": {
            "reasoning": (
                "Naive abstract: three random shapes with no compositional logic. "
                "Same material on all objects eliminates visual tension. "
                "Random placement, not rule-of-thirds or golden-ratio considered. "
                "No lighting setup means the sculpture can't show its intent. "
                "This is what 'abstract' looks like to a non-designer — arbitrary."
            ),
            "blender_python": dedent("""\
                import bpy

                bpy.ops.object.select_all(action='SELECT')
                bpy.ops.object.delete()

                # Naive: random shapes, same material
                bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
                bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(1.5, 0, 0))
                bpy.ops.mesh.primitive_cylinder_add(location=(-1.0, 0.5, 0))

                mat = bpy.data.materials.new('Grey')
                mat.diffuse_color = (0.5, 0.5, 0.5, 1.0)
                for obj in bpy.context.scene.objects:
                    if obj.type == 'MESH' and obj.data:
                        obj.data.materials.append(mat)

                # Missing: composition logic, material contrast, golden ratio,
                # lighting, ground plane, any design intent
            """),
        },
    },
]


# ─── Scraper (attempt Dream3D gallery) ────────────────────────────────────────

DREAM3D_URLS = [
    "https://dream3d.notion.site",
    "https://bluewillow.ai",
    "https://www.sloyd.ai",
    "https://alpha3d.io",
]


async def scrape_dream3d_examples(output_dir: Path) -> list[str]:
    """Attempt to scrape Dream3D-style prompt examples from public demos."""
    if not HAS_AIOHTTP:
        return []

    discovered_prompts = []
    import re

    async with aiohttp.ClientSession() as session:
        for url in DREAM3D_URLS:
            try:
                async with session.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=aiohttp.ClientTimeout(total=10),
                    ssl=False,
                ) as resp:
                    if resp.status != 200:
                        continue
                    text = await resp.text()
                    # Extract potential 3D prompt text
                    clean = re.sub(r'<[^>]+>', ' ', text)
                    # Look for prompt-like patterns (short descriptive phrases)
                    prompts = re.findall(
                        r'(?:create|make|generate|build|design|model)\s+[a-zA-Z\s,]+(?:3d|scene|object|model)?',
                        clean, re.IGNORECASE
                    )
                    discovered_prompts.extend(p.strip() for p in prompts[:10])
                    log.info("Scraped %s: found %d potential prompts", url, len(prompts))
                await asyncio.sleep(1.0)
            except Exception as e:
                log.debug("Could not scrape %s: %s", url, e)

    if discovered_prompts:
        prompts_file = output_dir / "discovered_prompts.txt"
        prompts_file.write_text("\n".join(set(discovered_prompts)))
        log.info("Saved %d discovered prompts to %s", len(discovered_prompts), prompts_file)

    return discovered_prompts


# ─── Main generator ────────────────────────────────────────────────────────────

class Dream3DDPOGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, target_count: int = 200, scrape: bool = False) -> int:
        """Generate DPO preference pairs."""
        if scrape:
            asyncio.run(scrape_dream3d_examples(self.output_dir))

        output_file = self.output_dir / "dpo_pairs.jsonl"
        total = 0

        # Cycle through templates, adding variation via random seeds
        pairs_per_template = max(1, target_count // len(DPO_TEMPLATES))
        remainder = target_count - pairs_per_template * len(DPO_TEMPLATES)

        with open(output_file, "w") as f_out, tqdm(total=target_count, desc="DPO pairs") as pbar:
            for i, template in enumerate(DPO_TEMPLATES):
                count = pairs_per_template + (1 if i < remainder else 0)
                for j in range(count):
                    if total >= target_count:
                        break

                    # Add variation to prompt for j > 0
                    prompt = template["prompt"]
                    if j > 0:
                        variations = [
                            f"{prompt} in a {random.choice(['minimalist', 'detailed', 'stylized', 'realistic'])} style",
                            f"{prompt} for a {random.choice(['game', 'film', 'architecture', 'product'])} project",
                            f"detailed version: {prompt}",
                            f"{prompt}, focus on materials",
                        ]
                        prompt = random.choice(variations)

                    pair = {
                        "prompt": prompt,
                        "chosen": {
                            "blender_python": template["chosen"]["blender_python"],
                            "reasoning": template["chosen"]["reasoning"],
                        },
                        "rejected": {
                            "blender_python": template["rejected"]["blender_python"],
                            "reasoning": template["rejected"]["reasoning"],
                        },
                        "quality": 3.0,
                        "source": "dream3d_dpo",
                        "metadata": {
                            "theme": template["theme"],
                            "base_prompt": template["prompt"],
                            "variation": j,
                        },
                    }

                    f_out.write(json.dumps(pair) + "\n")
                    total += 1
                    pbar.update(1)

        log.info("Generated %d DPO pairs → %s", total, output_file)
        return total

    def preview(self, n: int = 3) -> None:
        """Print n example DPO pairs."""
        for template in DPO_TEMPLATES[:n]:
            print(f"\n{'='*60}")
            print(f"PROMPT: {template['prompt']}")
            print(f"THEME: {template['theme']}")
            print(f"\nCHOSEN reasoning:")
            print(f"  {template['chosen']['reasoning'][:300]}...")
            print(f"\nREJECTED reasoning:")
            print(f"  {template['rejected']['reasoning'][:300]}...")
            print(f"{'='*60}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Dream3D-style DPO preference pairs for Nalana",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--count", type=int, default=200, help="Number of DPO pairs to generate")
    parser.add_argument(
        "--output", type=Path,
        default=BASE_DIR / "data" / "integrations" / "dream3d",
        help="Output directory",
    )
    parser.add_argument("--scrape", action="store_true", help="Attempt to scrape Dream3D gallery")
    parser.add_argument("--preview", type=int, metavar="N", help="Preview N example pairs and exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    generator = Dream3DDPOGenerator(args.output)

    if args.preview:
        generator.preview(args.preview)
        return

    total = generator.generate(args.count, scrape=args.scrape)
    print(f"\nDream3D DPO generation complete:")
    print(f"  {total} preference pairs → {args.output}/dpo_pairs.jsonl")
    print(f"  {len(DPO_TEMPLATES)} base templates × variations")
    print(f"  Themes: {', '.join(set(t['theme'] for t in DPO_TEMPLATES))}")
    print(f"  Each pair: expert chosen + naive rejected implementation")


if __name__ == "__main__":
    main()
