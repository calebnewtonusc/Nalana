"""
physics_sim.py — Nalana Physics Simulation Training Data Generator

Generates training pairs for physics-based tasks using the physics KB.
Uses GPU synthesis (not hardcoded templates) for deep physics reasoning.
Seed pairs bootstrap the process; GPU synthesis creates the real depth.
"""

from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import textwrap
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from knowledge.physics_kb import (
    GPU_SYNTHESIS_JOBS,
    PBR_PRESETS,
    SIMULATION_PRESETS,
    get_material_preset,
    get_simulation_preset,
)

# ─── 1. SEED SIMULATION PAIRS ─────────────────────────────────────────────────
# ~30 high-quality seeds. These bootstrap the training set.
# The GPU synthesis pipeline generates the real depth at scale.

SEED_SIMULATION_PAIRS: list[dict[str, str]] = [
    # ── Cloth ──────────────────────────────────────────────────────────────
    {
        "user": "Set up a silk cloth simulation for a flowing dress.",
        "assistant": textwrap.dedent("""\
            import bpy

            # Select the dress mesh object
            obj = bpy.context.active_object
            bpy.ops.object.modifier_add(type='CLOTH')
            cloth = obj.modifiers['Cloth'].settings

            # Silk preset — very low bending resistance, light weight
            cloth.quality               = 8
            cloth.mass                  = 0.3          # kg
            cloth.tension_stiffness     = 15.0
            cloth.compression_stiffness = 15.0
            cloth.shear_stiffness       = 0.05
            cloth.bending_stiffness     = 0.01         # near-zero: silk hangs freely
            cloth.air_damping           = 1.0
            cloth.bending_damping       = 0.5

            # Collision settings
            collision = obj.modifiers['Cloth'].collision_settings
            collision.collision_quality  = 4
            collision.distance_min       = 0.001
            collision.self_collision_distance = 0.002
            collision.use_self_collision = True

            # Cache for playback
            obj.modifiers['Cloth'].point_cache.frame_start = 1
            obj.modifiers['Cloth'].point_cache.frame_end   = 250
            print("Silk cloth simulation ready. Press Space to bake.")
        """),
    },
    {
        "user": "Make this fabric behave like stiff denim jeans.",
        "assistant": textwrap.dedent("""\
            import bpy

            obj = bpy.context.active_object
            bpy.ops.object.modifier_add(type='CLOTH')
            cloth = obj.modifiers['Cloth'].settings

            # Denim preset — heavy, high bending resistance
            cloth.quality               = 10
            cloth.mass                  = 1.2          # kg — heavy
            cloth.tension_stiffness     = 40.0
            cloth.compression_stiffness = 40.0
            cloth.shear_stiffness       = 5.0
            cloth.bending_stiffness     = 2.5          # high: denim resists creasing
            cloth.air_damping           = 0.3
            cloth.bending_damping       = 1.0

            collision = obj.modifiers['Cloth'].collision_settings
            collision.distance_min       = 0.002
            collision.use_self_collision = True
            collision.self_collision_distance = 0.003

            print("Denim cloth simulation configured.")
        """),
    },
    {
        "user": "Create a leather jacket cloth simulation.",
        "assistant": textwrap.dedent("""\
            import bpy

            obj = bpy.context.active_object
            bpy.ops.object.modifier_add(type='CLOTH')
            cloth = obj.modifiers['Cloth'].settings

            # Leather preset — very stiff, heavy
            cloth.quality               = 12
            cloth.mass                  = 2.5
            cloth.tension_stiffness     = 80.0
            cloth.compression_stiffness = 80.0
            cloth.shear_stiffness       = 12.0
            cloth.bending_stiffness     = 8.0           # stiff — leather barely drapes
            cloth.air_damping           = 0.1
            cloth.bending_damping       = 2.0

            collision = obj.modifiers['Cloth'].collision_settings
            collision.distance_min              = 0.003
            collision.use_self_collision        = False  # leather rarely self-collides
            collision.self_collision_distance   = 0.005

            print("Leather jacket cloth simulation ready.")
        """),
    },
    # ── Fluid ───────────────────────────────────────────────────────────────
    {
        "user": "Set up a water splash simulation inside a fluid domain.",
        "assistant": textwrap.dedent("""\
            import bpy

            # Assumes domain cube is active, flow object is named 'WaterFlow'
            domain_obj = bpy.context.active_object

            # Configure domain
            bpy.ops.object.modifier_add(type='FLUID')
            domain = domain_obj.modifiers['Fluid'].domain_settings
            domain.domain_type          = 'FLUID'
            domain.resolution_max       = 64
            domain.time_scale           = 1.0
            domain.use_spray            = True
            domain.use_foam             = True
            domain.use_bubbles          = True
            domain.surface_tension      = 0.0728        # N/m — water at 20°C

            # Flow object (the water source)
            flow_obj = bpy.data.objects['WaterFlow']
            flow_obj.select_set(True)
            bpy.context.view_layer.objects.active = flow_obj
            bpy.ops.object.modifier_add(type='FLUID')
            flow = flow_obj.modifiers['Fluid'].flow_settings
            flow.flow_type      = 'LIQUID'
            flow.flow_behavior  = 'INFLOW'
            flow.velocity_coord = (0.0, 0.0, -2.0)     # downward splash velocity

            print("Water splash simulation configured. Bake from domain object.")
        """),
    },
    {
        "user": "I want to simulate thick honey pouring slowly from a jar.",
        "assistant": textwrap.dedent("""\
            import bpy

            domain_obj = bpy.context.active_object
            bpy.ops.object.modifier_add(type='FLUID')
            domain = domain_obj.modifiers['Fluid'].domain_settings
            domain.domain_type      = 'FLUID'
            domain.resolution_max   = 48
            domain.time_scale       = 0.3               # slow-motion for viscous pour
            domain.use_spray        = False
            domain.use_foam         = False
            domain.use_bubbles      = False

            # Honey is ~2000x more viscous than water — FLIP solver handles this via time scale.
            # Increase substeps for accuracy with high viscosity.
            domain.use_adaptive_timestep = True
            domain.cfl_condition         = 2.0
            domain.timesteps_max         = 20

            flow_obj = bpy.data.objects.get('HoneySource')
            if flow_obj:
                bpy.context.view_layer.objects.active = flow_obj
                bpy.ops.object.modifier_add(type='FLUID')
                flow = flow_obj.modifiers['Fluid'].flow_settings
                flow.flow_type      = 'LIQUID'
                flow.flow_behavior  = 'INFLOW'
                flow.velocity_coord = (0.0, 0.0, -0.5)  # slow drip

            print("Honey fluid simulation configured.")
        """),
    },
    # ── Rigid Body ──────────────────────────────────────────────────────────
    {
        "user": "Make these objects fall and collide as rigid bodies. They should be rubber balls.",
        "assistant": textwrap.dedent("""\
            import bpy

            for obj in bpy.context.selected_objects:
                bpy.context.view_layer.objects.active = obj
                bpy.ops.rigidbody.object_add()
                rb = obj.rigid_body

                # Rubber ball properties
                rb.type                     = 'ACTIVE'
                rb.mass                     = 0.15       # kg — tennis ball ~0.057 kg
                rb.friction                 = 0.8
                rb.restitution              = 0.7        # bouncy
                rb.linear_damping           = 0.05
                rb.angular_damping          = 0.1
                rb.collision_shape          = 'SPHERE'   # most accurate for round objects
                rb.use_deactivation         = True
                rb.deactivate_linear_velocity  = 0.4
                rb.deactivate_angular_velocity = 0.5

            print(f"Configured {len(bpy.context.selected_objects)} objects as rubber rigid bodies.")
        """),
    },
    {
        "user": "Set up a metal crate that falls and doesn't bounce.",
        "assistant": textwrap.dedent("""\
            import bpy

            obj = bpy.context.active_object
            bpy.ops.rigidbody.object_add()
            rb = obj.rigid_body

            # Dense metal crate — steel ~7850 kg/m³
            rb.type             = 'ACTIVE'
            rb.mass             = 45.0           # kg for a small steel crate
            rb.friction         = 0.4
            rb.restitution      = 0.05           # almost no bounce — energy absorbed on impact
            rb.linear_damping   = 0.01
            rb.angular_damping  = 0.02
            rb.collision_shape  = 'BOX'          # box is most efficient for rectangular shapes

            rb.use_deactivation = True
            rb.deactivate_linear_velocity  = 0.1
            rb.deactivate_angular_velocity = 0.1

            print("Metal crate rigid body ready.")
        """),
    },
    {
        "user": "Create a rigid body constraint so this door swings on a hinge.",
        "assistant": textwrap.dedent("""\
            import bpy

            # Requires: 'Door' and 'DoorFrame' objects exist
            door  = bpy.data.objects['Door']
            frame = bpy.data.objects['DoorFrame']

            # Make frame a passive rigid body (fixed)
            bpy.context.view_layer.objects.active = frame
            bpy.ops.rigidbody.object_add()
            frame.rigid_body.type = 'PASSIVE'

            # Make door an active rigid body
            bpy.context.view_layer.objects.active = door
            bpy.ops.rigidbody.object_add()
            door.rigid_body.type        = 'ACTIVE'
            door.rigid_body.mass        = 8.0
            door.rigid_body.friction    = 0.5
            door.rigid_body.restitution = 0.1

            # Add empty at hinge pivot location
            bpy.ops.object.empty_add(type='ARROWS', location=(door.location.x + 0.5, door.location.y, door.location.z))
            hinge_empty = bpy.context.active_object
            hinge_empty.name = 'DoorHinge'

            # Add rigid body constraint
            bpy.context.view_layer.objects.active = door
            bpy.ops.rigidbody.constraint_add()
            constraint = door.rigid_body_constraint
            constraint.type             = 'HINGE'
            constraint.object1          = frame
            constraint.object2          = door
            constraint.use_limit_ang_z  = True
            constraint.limit_ang_z_lower = 0.0
            constraint.limit_ang_z_upper = 1.5708  # 90 degrees in radians

            print("Door hinge constraint configured.")
        """),
    },
    # ── Smoke & Fire ────────────────────────────────────────────────────────
    {
        "user": "Add campfire smoke coming from this object.",
        "assistant": textwrap.dedent("""\
            import bpy

            # Domain object must be a cube surrounding the fire source
            domain_obj = bpy.data.objects.get('SmokeDomain')
            source_obj = bpy.context.active_object

            # Configure domain
            bpy.context.view_layer.objects.active = domain_obj
            bpy.ops.object.modifier_add(type='FLUID')
            domain = domain_obj.modifiers['Fluid'].domain_settings
            domain.domain_type          = 'GAS'
            domain.resolution_max       = 64
            domain.use_noise            = True
            domain.noise_scale          = 2
            domain.temperature_max      = 1200.0        # K — campfire ~800-1200K
            domain.vorticity            = 0.2
            domain.use_dissolve         = True
            domain.dissolve_speed       = 25            # frames

            # Configure flow (fire source)
            bpy.context.view_layer.objects.active = source_obj
            bpy.ops.object.modifier_add(type='FLUID')
            flow = source_obj.modifiers['Fluid'].flow_settings
            flow.flow_type          = 'FIRE'
            flow.flow_behavior      = 'INFLOW'
            flow.temperature        = 1.0               # normalized heat
            flow.fuel_amount        = 1.0

            print("Campfire smoke/fire simulation configured.")
        """),
    },
    {
        "user": "Set up an explosion smoke effect.",
        "assistant": textwrap.dedent("""\
            import bpy

            domain_obj = bpy.data.objects.get('ExplosionDomain')
            source_obj = bpy.context.active_object

            bpy.context.view_layer.objects.active = domain_obj
            bpy.ops.object.modifier_add(type='FLUID')
            domain = domain_obj.modifiers['Fluid'].domain_settings
            domain.domain_type      = 'GAS'
            domain.resolution_max   = 128               # high res for hero explosion
            domain.use_noise        = True
            domain.noise_scale      = 4
            domain.temperature_max  = 6000.0            # K — explosion 3000-6000K
            domain.vorticity        = 1.0               # turbulent
            domain.use_dissolve     = True
            domain.dissolve_speed   = 40

            bpy.context.view_layer.objects.active = source_obj
            bpy.ops.object.modifier_add(type='FLUID')
            flow = source_obj.modifiers['Fluid'].flow_settings
            flow.flow_type      = 'FIRE'
            flow.flow_behavior  = 'GEOMETRY'            # instant volume injection
            flow.temperature    = 1.0
            flow.fuel_amount    = 2.0                   # large fuel burst

            print("Explosion simulation configured at 128 resolution.")
        """),
    },
    # ── Soft Body ───────────────────────────────────────────────────────────
    {
        "user": "Make this cube squash and stretch like jelly when it falls.",
        "assistant": textwrap.dedent("""\
            import bpy

            obj = bpy.context.active_object
            bpy.ops.object.modifier_add(type='SOFT_BODY')
            sb = obj.modifiers['Softbody'].settings

            # Jelly preset — low goal, low stiffness, bouncy
            sb.use_goal             = True
            sb.goal_strength        = 0.1               # weak shape retention
            sb.goal_damping         = 0.5
            sb.pull                 = 0.2               # edge tension (resist stretching)
            sb.push                 = 0.4               # edge compression resist
            sb.damping              = 0.2
            sb.mass                 = 1.0

            # Self-collision for blob-like behavior
            sb.use_self_collision   = True
            sb.ball_size            = 0.05
            sb.ball_stiffness       = 0.9
            sb.ball_damp            = 0.5

            # Gravity response
            bpy.context.scene.gravity = (0.0, 0.0, -9.81)

            print("Jelly soft body ready. Subdivide mesh for better simulation quality.")
        """),
    },
    {
        "user": "Add soft body to this character's belly so it jiggles.",
        "assistant": textwrap.dedent("""\
            import bpy

            # Best practice: soft body on a low-res proxy, driven to hi-res via shrinkwrap
            proxy = bpy.context.active_object
            bpy.ops.object.modifier_add(type='SOFT_BODY')
            sb = proxy.modifiers['Softbody'].settings

            # Foam/fat preset — moderate goal strength, retains shape but allows jiggle
            sb.use_goal             = True
            sb.goal_strength        = 0.3
            sb.goal_damping         = 0.3
            sb.pull                 = 0.5
            sb.push                 = 0.8
            sb.damping              = 0.1
            sb.mass                 = 0.05              # low mass = responsive to small motions

            sb.use_self_collision   = False             # proxy is simple, no self-collision needed

            # Pin non-belly vertices to goal (use vertex group)
            sb.vertex_group_goal = 'sb_pin'            # assign weight 1.0 to areas that should not jiggle
            sb.goal_default      = 0.3
            sb.goal_min          = 0.0
            sb.goal_max          = 1.0

            print("Belly jiggle soft body configured. Create 'sb_pin' vertex group and paint weights.")
        """),
    },
    # ── Particles ───────────────────────────────────────────────────────────
    {
        "user": "Create a particle system that emits sparks from this mesh.",
        "assistant": textwrap.dedent("""\
            import bpy

            obj = bpy.context.active_object
            bpy.ops.object.particle_system_add()
            ps = obj.particle_systems[-1]
            settings = ps.settings

            # Spark emitter — many small fast particles with gravity drag
            settings.name               = 'Sparks'
            settings.type               = 'EMITTER'
            settings.count              = 2000
            settings.frame_start        = 1
            settings.frame_end          = 50
            settings.lifetime           = 40
            settings.lifetime_random    = 0.3
            settings.emit_from          = 'FACE'
            settings.use_emit_random    = True

            # Velocity — burst outward
            settings.normal_factor      = 3.0
            settings.factor_random      = 1.0
            settings.tangent_factor     = 0.5

            # Physics — ember drag
            settings.physics_type       = 'NEWTON'
            settings.mass               = 0.001
            settings.drag_factor        = 0.05
            settings.brownian_factor    = 0.2           # random turbulence
            settings.use_size_deflect   = True

            # Render as tiny spheres or points
            settings.render_type        = 'HALO'
            settings.particle_size      = 0.02
            settings.size_random        = 0.5

            print("Spark particle system created. Add an emissive material for glow.")
        """),
    },
    {
        "user": "Make snow fall across the whole scene.",
        "assistant": textwrap.dedent("""\
            import bpy

            # Add a large emitter plane above the scene
            bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 8))
            emitter = bpy.context.active_object
            emitter.name = 'SnowEmitter'
            emitter.hide_render = True  # invisible in render

            bpy.ops.object.particle_system_add()
            settings = emitter.particle_systems[-1].settings

            settings.name           = 'Snow'
            settings.type           = 'EMITTER'
            settings.count          = 5000
            settings.frame_start    = 1
            settings.frame_end      = 200
            settings.lifetime       = 150
            settings.emit_from      = 'FACE'

            # Slow fall with wind drift
            settings.normal_factor  = -1.5             # fall downward
            settings.factor_random  = 0.2
            settings.physics_type   = 'NEWTON'
            settings.mass           = 0.0001            # nearly weightless
            settings.drag_factor    = 0.5               # air resistance slows fall
            settings.brownian_factor = 0.3              # wind turbulence

            settings.render_type    = 'OBJECT'
            settings.particle_size  = 0.05
            settings.size_random    = 0.6

            print("Snow particle system created. Set render object to a small flake mesh.")
        """),
    },
    # ── Force Fields ────────────────────────────────────────────────────────
    {
        "user": "Add a wind force that pushes the cloth to the right.",
        "assistant": textwrap.dedent("""\
            import bpy

            bpy.ops.object.effector_add(type='WIND', location=(0, -3, 1))
            wind = bpy.context.active_object
            wind.name = 'SceneWind'

            # Point wind along +X axis by rotating the empty
            wind.rotation_euler[2] = 1.5708  # 90 degrees

            field = wind.field
            field.strength      = 4.0
            field.noise         = 0.5        # turbulent gusts
            field.seed          = 42
            field.use_global_coords = True

            print("Wind force field added. The cloth simulation will pick it up automatically.")
        """),
    },
    {
        "user": "Add a vortex that pulls particles into a swirl.",
        "assistant": textwrap.dedent("""\
            import bpy

            bpy.ops.object.effector_add(type='VORTEX', location=(0, 0, 0))
            vortex = bpy.context.active_object
            vortex.name = 'SceneVortex'

            field = vortex.field
            field.strength          = 5.0
            field.inflow            = 1.0    # pull toward center
            field.use_max_distance  = True
            field.distance_max      = 6.0   # only affect particles within 6 meters

            print("Vortex force field added.")
        """),
    },
    # ── Constraint-Based Physics ────────────────────────────────────────────
    {
        "user": "Make this chain of objects simulate as a rope.",
        "assistant": textwrap.dedent("""\
            import bpy

            chain_objects = [obj for obj in bpy.context.selected_objects]

            # Add rigid body to each link
            for i, obj in enumerate(chain_objects):
                bpy.context.view_layer.objects.active = obj
                bpy.ops.rigidbody.object_add()
                rb = obj.rigid_body
                rb.type         = 'ACTIVE'
                rb.mass         = 0.1
                rb.friction     = 0.5
                rb.restitution  = 0.05
                rb.collision_shape = 'CONVEX_HULL'

                # Fixed first link to ceiling/anchor
                if i == 0:
                    rb.type = 'PASSIVE'

            # Add Generic Spring constraint between each consecutive pair
            for i in range(len(chain_objects) - 1):
                link_a = chain_objects[i]
                link_b = chain_objects[i + 1]

                # Empty at midpoint for constraint
                mid = (
                    (link_a.location.x + link_b.location.x) / 2,
                    (link_a.location.y + link_b.location.y) / 2,
                    (link_a.location.z + link_b.location.z) / 2,
                )
                bpy.ops.object.empty_add(type='ARROWS', location=mid)
                pivot = bpy.context.active_object
                pivot.name = f'ChainConstraint_{i}'

                bpy.ops.rigidbody.constraint_add()
                c = pivot.rigid_body_constraint
                c.type              = 'GENERIC_SPRING'
                c.object1           = link_a
                c.object2           = link_b
                c.use_spring_ang_x  = True
                c.spring_stiffness_ang_x = 10.0
                c.spring_damping_ang_x   = 0.5

            print(f"Chain rope simulation set up for {len(chain_objects)} links.")
        """),
    },
    # ── Material + Sim Combined ─────────────────────────────────────────────
    {
        "user": "Create a realistic ocean surface with foam and displacement.",
        "assistant": textwrap.dedent("""\
            import bpy

            # Add ocean modifier to a plane
            bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
            ocean_plane = bpy.context.active_object
            ocean_plane.name = 'Ocean'

            # Subdivide for detail
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.subdivide(number_cuts=10)
            bpy.ops.object.mode_set(mode='OBJECT')

            bpy.ops.object.modifier_add(type='OCEAN')
            ocean = ocean_plane.modifiers['Ocean']
            ocean.resolution            = 12
            ocean.wave_scale            = 1.5
            ocean.wave_scale_min        = 0.0
            ocean.choppiness            = 1.5
            ocean.wind_velocity         = 8.0           # m/s — moderate ocean wind
            ocean.wind_direction        = 0.0
            ocean.random_seed           = 0
            ocean.spatial_size          = 50
            ocean.use_foam              = True
            ocean.foam_coverage         = 0.9
            ocean.foam_layer_name       = 'foam'

            # Add Subdivision Surface for smooth geometry
            bpy.ops.object.modifier_add(type='SUBSURF')
            ocean_plane.modifiers['Subdivision'].render_levels = 3

            # Material — water IOR 1.333, transmission, blue-green tint
            mat = bpy.data.materials.new(name='OceanWater')
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            nodes.clear()

            output   = nodes.new('ShaderNodeOutputMaterial')
            principled = nodes.new('ShaderNodeBsdfPrincipled')
            output.location   = (400, 0)
            principled.location = (0, 0)

            principled.inputs['Base Color'].default_value = (0.01, 0.15, 0.25, 1.0)
            principled.inputs['Metallic'].default_value   = 0.0
            principled.inputs['Roughness'].default_value  = 0.0
            principled.inputs['IOR'].default_value        = 1.333
            principled.inputs['Transmission Weight'].default_value = 1.0

            links.new(principled.outputs['BSDF'], output.inputs['Surface'])
            ocean_plane.data.materials.append(mat)

            print("Ocean surface with foam ready. Bake foam to vertex color for shader use.")
        """),
    },
    {
        "user": "Set up a soft body simulation for a bouncing rubber ball using the correct material.",
        "assistant": textwrap.dedent("""\
            import bpy

            obj = bpy.context.active_object

            # Soft body for deformable rubber ball
            bpy.ops.object.modifier_add(type='SOFT_BODY')
            sb = obj.modifiers['Softbody'].settings
            sb.use_goal         = True
            sb.goal_strength    = 0.7               # rubber snaps back to shape
            sb.goal_damping     = 0.1
            sb.pull             = 0.9               # high tension stiffness
            sb.push             = 0.95
            sb.damping          = 0.05
            sb.mass             = 0.5
            sb.use_self_collision = False

            # Rubber material (Principled BSDF)
            mat = bpy.data.materials.new(name='Rubber_Red')
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            nodes.clear()

            output     = nodes.new('ShaderNodeOutputMaterial')
            principled = nodes.new('ShaderNodeBsdfPrincipled')
            output.location     = (400, 0)
            principled.location = (0, 0)

            principled.inputs['Base Color'].default_value  = (0.6, 0.015, 0.01, 1.0)
            principled.inputs['Roughness'].default_value   = 0.7
            principled.inputs['IOR'].default_value         = 1.52
            principled.inputs['Specular IOR Level'].default_value = 0.03

            links.new(principled.outputs['BSDF'], output.inputs['Surface'])
            obj.data.materials.append(mat)

            print("Red rubber ball with soft body configured.")
        """),
    },
    {
        "user": "Add a turbulence force field to make smoke look more chaotic.",
        "assistant": textwrap.dedent("""\
            import bpy

            bpy.ops.object.effector_add(type='TURBULENCE', location=(0, 0, 2))
            turb = bpy.context.active_object
            turb.name = 'SmokeTurbulence'

            field = turb.field
            field.strength  = 3.0
            field.size      = 1.0       # scale of turbulence features
            field.flow      = 0.5       # how quickly the pattern changes
            field.seed      = 17

            print("Turbulence force field added. It will affect smoke, particles, and cloth.")
        """),
    },
    {
        "user": "Create a quick rigid body demolition where a wall collapses.",
        "assistant": textwrap.dedent("""\
            import bpy

            # Use cell fracture on the wall mesh first (requires Cell Fracture add-on)
            bpy.ops.object.add_fracture_cell_objects(
                source={'PARTICLE_OWN'},
                source_limit=50,
                source_noise=0.1,
                cell_scale=(1.0, 1.0, 1.0),
                recursion=0,
                use_smooth_faces=False,
                use_data_match=True,
                margin=0.002,
            )

            # Add rigid body to all fracture pieces
            for obj in bpy.context.selected_objects:
                bpy.context.view_layer.objects.active = obj
                bpy.ops.rigidbody.object_add()
                rb = obj.rigid_body
                rb.type             = 'ACTIVE'
                rb.mass             = 20.0          # heavy concrete chunks
                rb.friction         = 0.6
                rb.restitution      = 0.1
                rb.collision_shape  = 'CONVEX_HULL'
                rb.linear_damping   = 0.03
                rb.angular_damping  = 0.08

                # Start as passive (wall intact), animated to active at frame 30
                rb.enabled          = True
                rb.use_deactivation = False

            print("Demolition rigid body setup complete. Animate 'enabled' toggle at frame 30.")
        """),
    },
    {
        "user": "Add a cloth curtain that reacts to a wind force.",
        "assistant": textwrap.dedent("""\
            import bpy

            # Add curtain plane
            bpy.ops.mesh.primitive_plane_add(size=3, location=(0, 0, 1.5))
            curtain = bpy.context.active_object
            curtain.name = 'Curtain'

            # Subdivide for simulation resolution
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.subdivide(number_cuts=15)
            bpy.ops.object.mode_set(mode='OBJECT')

            # Pin top edge using vertex group
            vg = curtain.vertex_groups.new(name='Pin')
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='DESELECT')
            # Select top row of vertices (high Z) — done via script for automation
            bpy.ops.object.mode_set(mode='OBJECT')
            top_verts = [v.index for v in curtain.data.vertices if v.co.z > 1.45]
            vg.add(top_verts, 1.0, 'REPLACE')

            # Cloth modifier — cotton curtain
            bpy.ops.object.modifier_add(type='CLOTH')
            cloth = curtain.modifiers['Cloth'].settings
            cloth.quality               = 10
            cloth.mass                  = 0.35
            cloth.tension_stiffness     = 20.0
            cloth.bending_stiffness     = 0.5
            cloth.shear_stiffness       = 1.0
            cloth.air_damping           = 0.5
            cloth.vertex_group_mass     = 'Pin'
            cloth.pin_stiffness         = 1.0

            # Wind force
            bpy.ops.object.effector_add(type='WIND', location=(0, -2, 1))
            wind = bpy.context.active_object
            wind.rotation_euler[0] = 1.5708
            wind.field.strength     = 2.5
            wind.field.noise        = 0.8

            print("Curtain cloth with wind configured.")
        """),
    },
    {
        "user": "Create a fluid simulation where water fills a glass.",
        "assistant": textwrap.dedent("""\
            import bpy

            # Assumes 'Glass' object exists as a mesh obstacle
            glass_obj = bpy.data.objects.get('Glass')

            # Domain — bounding box around the glass
            bpy.ops.mesh.primitive_cube_add(size=0.15, location=(0, 0, 0.075))
            domain_obj = bpy.context.active_object
            domain_obj.name = 'WaterDomain'

            bpy.ops.object.modifier_add(type='FLUID')
            domain = domain_obj.modifiers['Fluid'].domain_settings
            domain.domain_type      = 'FLUID'
            domain.resolution_max   = 96
            domain.time_scale       = 0.8
            domain.use_foam         = False
            domain.use_spray        = False

            # Glass as effector (obstacle)
            if glass_obj:
                bpy.context.view_layer.objects.active = glass_obj
                bpy.ops.object.modifier_add(type='FLUID')
                effector = glass_obj.modifiers['Fluid'].flow_settings
                glass_obj.modifiers['Fluid'].fluid_type = 'EFFECTOR'

            # Water flow source (above glass opening)
            bpy.ops.mesh.primitive_plane_add(size=0.06, location=(0, 0, 0.14))
            source_obj = bpy.context.active_object
            source_obj.name = 'WaterSource'

            bpy.ops.object.modifier_add(type='FLUID')
            flow = source_obj.modifiers['Fluid'].flow_settings
            flow.flow_type      = 'LIQUID'
            flow.flow_behavior  = 'INFLOW'
            flow.velocity_coord = (0.0, 0.0, -0.5)

            print("Water glass fill simulation configured. Bake from domain object.")
        """),
    },
]

# ─── 2. SEED PHYSICS REASONING PAIRS ─────────────────────────────────────────
# ~20 conceptual Q&A pairs explaining the physics behind visual phenomena.
# These train the model to reason about WHY, not just HOW.

SEED_PHYSICS_REASONING_PAIRS: list[dict[str, str]] = [
    {
        "user": "Why does velvet look so different from silk even though both are smooth fabrics?",
        "assistant": (
            "Velvet and silk have fundamentally different microstructures that cause opposite light-scattering behavior.\n\n"
            "Silk is a continuous filament with a smooth, triangular cross-section that acts like a tiny prism. "
            "Its surface reflects light in a single dominant direction — you get a narrow, bright specular highlight "
            "that shifts across the fabric as you move. In PBR terms, silk has very low roughness (0.1–0.2) and moderate specular.\n\n"
            "Velvet is woven with millions of tiny upright fibers (pile). Each fiber is essentially a cylinder standing on end. "
            "Several phenomena combine: (1) light bouncing between fibers scatters in nearly all directions, eliminating a clear "
            "highlight; (2) at grazing angles, you see the tips of the fibers illuminated against the dark cavity below them, "
            "which creates a bright rim or sheen; (3) light that enters the pile cavity undergoes multiple scatters before "
            "exiting, so the base color appears very dark and saturated when viewed straight-on.\n\n"
            "In Blender: velvet = roughness 1.0, sheen 1.0, sheen tint 1.0. The Sheen parameter is specifically designed "
            "to replicate the retroreflective, grazing-angle brightening of fiber-based materials. "
            "Silk = roughness 0.15, anisotropic 0.7, anisotropic rotation 0.0 (or driven by tangent map)."
        ),
    },
    {
        "user": "What IOR should I use for frosted glass, and why does frosting work differently from just being transparent?",
        "assistant": (
            "Use IOR 1.5 (same as clear glass — the IOR of the glass itself hasn't changed), but set roughness to 0.3–0.5 "
            "and transmission to 0.9.\n\n"
            "Clear glass is smooth at the micro scale, so all transmitted rays refract at the same angle and arrive "
            "at your eye from a single direction — you see a sharp image through it. Frosted glass has been acid-etched or "
            "sand-blasted, creating microscopic surface irregularities. Each transmitted ray refracts at a slightly different "
            "angle depending on the local micro-normal. Rays that entered close together diverge on exit, so instead of a "
            "sharp point of light from behind the glass, you see a blurred blob — this is volumetric scattering simulated "
            "by the roughness on the transmission lobe.\n\n"
            "The key insight: frosting is a surface phenomenon (roughness), not a bulk phenomenon (absorption or actual "
            "scattering inside the glass volume). The IOR stays 1.5 because you're still dealing with the same silica glass. "
            "If you want actual volumetric scattering (like frosted acrylic that glows internally), add a volume scatter "
            "node with a small density (0.01–0.1) and anisotropy around 0.5."
        ),
    },
    {
        "user": "Why does gold look yellow? I know it has something to do with physics but what exactly?",
        "assistant": (
            "Gold's yellow color comes directly from its complex index of refraction, which is a quantum mechanical "
            "consequence of gold's electron band structure.\n\n"
            "Like all metals, gold has free electrons that respond to incoming electromagnetic radiation (light). "
            "The critical parameter is the plasma frequency — the resonant frequency at which the electron gas oscillates. "
            "For most metals, the plasma frequency falls in the ultraviolet, meaning all visible light is reflected. "
            "This gives silver and aluminum their flat, spectrally neutral appearance.\n\n"
            "Gold is unusual: its plasma frequency falls in the visible range, near 500 nm (blue-green). "
            "Additionally, relativistic effects (gold is heavy — atomic number 79) cause its outer 6s electrons to "
            "contract, lowering the energy gap between the 5d and 6sp bands into the blue range (~2.4 eV, ~520 nm). "
            "This means gold absorbs blue and violet light strongly (the high extinction coefficient k for those wavelengths). "
            "Red and yellow light (600–700 nm) are reflected almost as efficiently as by silver. "
            "The result: gold's reflectance spectrum has low blue, rising sharply through green, and high red-yellow.\n\n"
            "In Blender: set base_color to (1.0, 0.766, 0.336) in linear space, metallic to 1.0. "
            "The Principled BSDF's metallic path uses the base color as the specular tint, which correctly reproduces "
            "the wavelength-dependent reflectance without needing to input actual n,k values."
        ),
    },
    {
        "user": "How do I make realistic ocean water in Blender? It always looks fake.",
        "assistant": (
            "Ocean water is optically complex — you need to get four separate phenomena right simultaneously:\n\n"
            "1. Fresnel reflection: at normal incidence, water reflects only ~2% of light (IOR 1.333, F0 ≈ 0.02). "
            "At grazing angles, it approaches 100% mirror reflection. In Principled BSDF: IOR=1.333, transmission=1.0. "
            "The Fresnel behavior is automatic — don't crank up specular manually or it breaks.\n\n"
            "2. Deep scattering (color): water absorbs red light faster than blue. At depth, transmitted light becomes "
            "cyan-to-deep-blue. Add a Volume Absorption node with color (1.0, 0.15, 0.02) and density ~0.05. "
            "The base_color should be nearly black or very dark blue (0.005, 0.07, 0.12) — water has almost no albedo.\n\n"
            "3. Surface foam/whitecaps: where wave crests break, air bubbles create thousands of micro-surfaces that "
            "scatter light diffusely — hence foam appears white. Drive a white, rough (roughness=0.9) diffuse shader "
            "with the Ocean modifier's foam vertex color attribute.\n\n"
            "4. Displacement and surface detail: use the Ocean modifier (Shift+A → Force Field → Ocean for newer versions, "
            "or add it as a modifier). Wave scale 1–2, choppiness 1–2, spatial size 50m. Add noise texture for small "
            "ripple detail on top. The displacement must be in 'True Displacement' mode in material settings.\n\n"
            "The most common mistake: making the base_color too light or too blue-green. Real open-ocean water seen "
            "from above is essentially black with a Fresnel mirror layer on top."
        ),
    },
    {
        "user": "What's the difference between roughness 0.0 and roughness 1.0 physically?",
        "assistant": (
            "Roughness in PBR describes the statistical distribution of microfacet normals — how many tiny facets "
            "on the surface point in directions other than the macro surface normal.\n\n"
            "Roughness 0.0: all microfacets are perfectly aligned with the surface normal. "
            "The GGX normal distribution function (NDF) becomes a Dirac delta — only exactly one angle reflects "
            "toward the eye. This is a perfect mirror: you see a point-perfect reflection of each light source. "
            "The alpha parameter in the GGX formula is roughness², so α=0 makes the distribution infinitely narrow.\n\n"
            "Roughness 1.0: microfacet normals are distributed across a very wide hemisphere. "
            "Light arriving from any direction gets scattered toward many output directions. "
            "The specular highlight becomes enormous and dim — so broad it's indistinguishable from diffuse. "
            "At roughness 1.0, a material looks like chalk or concrete: no perceivable highlight, flat matte appearance.\n\n"
            "Practical range by material: mirrors/chrome 0.0–0.05, polished metal 0.05–0.2, satin metal 0.2–0.4, "
            "brushed metal 0.3–0.6, matte plastic 0.7–0.9, chalk/concrete 0.9–1.0. "
            "Note: roughness is perceptually non-linear — the difference between 0.1 and 0.2 is visually larger "
            "than between 0.8 and 0.9. Some pipelines use roughness² internally (remapping) to linearize artist control."
        ),
    },
    {
        "user": "Why does subsurface scattering make skin look so much better? What's the physics?",
        "assistant": (
            "Without SSS, skin is treated as opaque: light hits the surface, reflects off, and that's it. "
            "But real skin is a layered translucent tissue. Light penetrates past the epidermis and scatters "
            "through the dermis (which contains hemoglobin, melanin, collagen, fat) before some fraction exits.\n\n"
            "The Jensen dipole model approximates this with absorption (σ_a) and scattering (σ_s) coefficients "
            "per wavelength. Red light (700 nm) has low absorption in tissue — it can penetrate up to a centimeter. "
            "Blue light (450 nm) is absorbed much faster — it barely makes it past the epidermis. "
            "This is why: thin skin areas (ears, fingertips) glow red when backlit (red photons passed through, "
            "blue ones were absorbed), and bruises look bluish-green (hemoglobin absorbs red in its deoxy state, "
            "blue scatters back from shallow dermis layers).\n\n"
            "In Blender Principled BSDF: subsurface 0.3–0.5, subsurface_radius (1.0, 0.2, 0.1) in meters "
            "(R scatters 1m, G 0.2m, B 0.1m — meaning red light exits much further from the entry point). "
            "This radius is in scene units — scale to ~(0.01, 0.002, 0.001) for a human head at real-world scale.\n\n"
            "Key visual result: contact shadows soften, ears glow red in backlight, lips appear semi-translucent, "
            "and the shadow terminator on skin looks soft and warm rather than hard and grey."
        ),
    },
    {
        "user": "Why does a diamond sparkle so much more than glass with the same shape?",
        "assistant": (
            "Diamond's extraordinary brilliance comes from three compounding physical advantages over glass:\n\n"
            "1. Higher IOR (2.417 vs 1.5): the Fresnel reflectance at normal incidence for diamond is "
            "F0 = ((2.417-1)/(2.417+1))² ≈ 0.172, versus glass at ~0.04. Diamond reflects over 4× more light "
            "straight-on, and at grazing angles it approaches ~100% faster.\n\n"
            "2. Much lower critical angle for total internal reflection: critical angle θ_c = arcsin(1/n). "
            "For diamond: arcsin(1/2.417) ≈ 24.4°. For glass: arcsin(1/1.5) ≈ 41.8°. "
            "This means light inside diamond undergoes total internal reflection (TIR) across a far wider range "
            "of angles — 90% of all light entering is trapped and redirected internally until it exits through "
            "the table or crown facets directly toward the viewer, rather than leaking out the sides.\n\n"
            "3. High dispersion (fire): diamond has unusually high dispersion — its IOR varies strongly by "
            "wavelength (1.458 nm → n=2.465; 768 nm → n=2.407). This spread of IOR by wavelength causes "
            "different colors to refract at different angles inside the stone, splitting white light into rainbow "
            "spectra visible as flashes of color ('fire').\n\n"
            "In Blender: set IOR=2.417, roughness=0.0, transmission=1.0. For dispersion, use the OSL diamond shader "
            "or approximate with a Color attribute driving a rainbow texture on the transmission color."
        ),
    },
    {
        "user": "Why do metals look different from plastic even when they have the same color?",
        "assistant": (
            "The fundamental difference is the quantum mechanical nature of their free electrons. "
            "This is exactly what the metallic slider in Principled BSDF encodes.\n\n"
            "Dielectrics (plastics, stone, wood): valence electrons are bound to atoms. "
            "Incoming photons interact only with bound electrons. Some light reflects at the surface via "
            "Fresnel (this reflection is always white/colorless — the specular is achromatic). "
            "The rest transmits into the bulk where it scatters off the molecular structure and re-exits "
            "as diffuse light. The color you see is the diffuse albedo — light that was selectively absorbed "
            "at certain wavelengths during its journey through the bulk.\n\n"
            "Metals: free electrons (the conduction band) respond to the full EM field. "
            "Almost all light is reflected at the surface — there's no diffuse component because light can't "
            "penetrate more than a few nanometers (the skin depth at optical frequencies). "
            "The reflection is not white: the complex IOR causes wavelength-dependent absorption. "
            "Gold absorbs blue, copper absorbs blue and green — so the reflected light is colored. "
            "The 'color' of a metal IS its specular color.\n\n"
            "Blender consequence: metallic=0 means the base_color feeds the diffuse lobe and specular is white. "
            "Metallic=1 means there IS no diffuse lobe (no light enters the bulk) and base_color directly "
            "tints the specular reflection — physically correct, because metals reflect colored specular light."
        ),
    },
    {
        "user": "How do I set up a realistic car paint material with the clear coat flakes?",
        "assistant": (
            "Automotive paint is a 3-layer system: primer (diffuse base), pigment layer (color), "
            "metallic flake layer (tiny aluminum particles suspended in the binder), and clear coat (lacquer). "
            "Each layer requires separate treatment.\n\n"
            "In Blender's Principled BSDF (which has a built-in clearcoat lobe):\n\n"
            "Base layer: set your car color as base_color, metallic=0, roughness=0.8 "
            "(this represents the pigment, which is diffuse). Keep this modest.\n\n"
            "Flake layer: add a separate Metallic Flake pass using a Noise Texture driving "
            "an Anisotropic BSDF at scale ~200 (tiny flakes). Mix it over the diffuse with a "
            "Mix Shader, factor ~0.3–0.5. Each flake is a mirror with random orientation, "
            "so use Vector Rotate with noise input to randomize the flake tangent.\n\n"
            "Clear coat: set clearcoat=1.0, clearcoat_roughness=0.03 (mirror-smooth lacquer). "
            "IOR of lacquer is ~1.5. This is the layer that makes paint shine with a "
            "distinct highlight separate from the underlying metallic flakes.\n\n"
            "The visual signature of metallic car paint: the diffuse color shifts at different angles "
            "(flakes catch light at some angles, not others), and there are two distinct specular "
            "highlights — one sharp from the clearcoat, one scattered and colorful from the flakes."
        ),
    },
    {
        "user": "Why does smoke look lighter at the edges and darker in the middle?",
        "assistant": (
            "This is Beer-Lambert law in action: T(d) = exp(-σ_a × d), where T is transmittance, "
            "σ_a is the absorption/scattering coefficient, and d is path length through the smoke.\n\n"
            "When you look at the center of a smoke plume, light must travel through the maximum thickness "
            "of participating medium before reaching your eye. Each micro-layer of smoke scatters some "
            "percentage of that light sideways, removing it from the line of sight. The transmittance "
            "falls exponentially — a path twice as long transmits the square of a single path's transmittance. "
            "Dense, thick smoke becomes nearly opaque in the interior.\n\n"
            "At the edges, you're looking through a thin sliver of smoke. Very little path length, "
            "very little scattering — most of the background sky/light passes through, "
            "making the edges appear bright and wispy.\n\n"
            "Additionally, smoke particles (soot, water droplets) scatter light forward preferentially "
            "(Mie scattering, high anisotropy g≈0.7). From the side or front, edge-lit smoke appears "
            "bright because forward-scattered light exits the edges toward you.\n\n"
            "In Blender's volume shader: set Volume Scatter anisotropy to 0.5–0.7 and density to 1.0–3.0. "
            "The density gradient from edge to center emerges naturally from the simulation's density field — "
            "you don't fake it. If smoke looks too uniform, increase the domain resolution or add a "
            "Noise Texture to modulate the density output in the volume material."
        ),
    },
    {
        "user": "What's the physically correct way to set up a candle flame material?",
        "assistant": (
            "Candle flame emission is blackbody radiation at roughly 1800–2000K, combined with "
            "chemiluminescence from excited CH and C2 radicals in the reaction zone.\n\n"
            "Blackbody at 1800K peaks in the infrared, with a long visible tail peaking toward orange-red. "
            "Use Blender's Blackbody node: connect a Value node set to 1800 into Blackbody, "
            "connect its Color output to Emission Color in the Principled BSDF "
            "(or directly to an Emission shader). Set Emission Strength to 3–8 depending on scene scale.\n\n"
            "The blue inner cone of a candle is hotter (2500–3000K) and has less yellow soot — "
            "it's slightly bluer. Layer a second emission shader with Blackbody at 2800K for the inner zone, "
            "masked by a gradient from the flame tip.\n\n"
            "Volume settings: the flame volume should have Volume Emission density 2–5, "
            "color driven by a Gradient Texture from yellow-orange (outer) to blue-white (inner). "
            "Add light Volume Scatter with density 0.1 and anisotropy 0.3 for the faint smoke haze above.\n\n"
            "Lighting contribution: candle flames are real area lights. The flame mesh should have a "
            "Point Light or Area Light parented to it, color temp 1800K, power 5–15 watts, "
            "to actually illuminate nearby objects — the emission shader on the mesh alone does not "
            "illuminate the scene in Eevee, and contributes only indirectly in Cycles."
        ),
    },
    {
        "user": "Why does marble appear slightly translucent even though it's rock?",
        "assistant": (
            "Marble is crystalline calcite (CaCO3) with a microcrystalline structure. "
            "Individual calcite crystals are highly transparent with IOR 1.486 — the crystal itself is "
            "as clear as glass. What makes marble opaque at large scales is the grain boundaries "
            "between millions of crystals oriented randomly.\n\n"
            "Each crystal boundary represents an abrupt change in IOR and orientation, causing a small "
            "refraction/reflection event. A photon entering the surface undergoes thousands of these "
            "micro-refractions, diffusing deeper into the stone — this is volumetric scattering. "
            "The mean free path (average distance between scattering events) in white marble is "
            "roughly 2–5mm. This is enough for light to travel a few millimeters below the surface "
            "and re-emerge slightly displaced — the defining signature of SSS.\n\n"
            "High-quality white marble like Carrara has such fine grain that light travels even further, "
            "giving it that distinctive glow. Darker marbles have more iron oxide and graphite impurities "
            "that increase absorption, reducing the SSS depth.\n\n"
            "Blender setup: subsurface 0.05–0.1, subsurface_radius (0.4, 0.35, 0.3) in scene units "
            "(slightly more red penetration than blue, consistent with calcite absorption spectrum). "
            "IOR 1.486, roughness 0.1–0.2, clearcoat 0.3 for the polished surface lacquer. "
            "The white veining is a different mineral intrusion (quartz, feldspar) with negligible SSS — "
            "drive subsurface weight with the vein mask so veins are slightly more opaque."
        ),
    },
    {
        "user": "How does anisotropy in a brushed aluminum material work physically?",
        "assistant": (
            "Anisotropy describes materials where the microfacet distribution is not rotationally symmetric. "
            "In brushed aluminum, the mechanical polishing process scratches the surface with parallel micro-grooves. "
            "These grooves act as tiny cylindrical mirrors, all aligned in the same direction.\n\n"
            "Light reflecting off a standard isotropic rough metal scatters in a roughly circular highlight. "
            "On brushed aluminum, the parallel grooves elongate the specular highlight perpendicular to the "
            "brushing direction — you see a long streak of light (like the streak you see on brushed metal "
            "in photography studios) that's narrow in the direction of brushing and wide across it.\n\n"
            "The GGX NDF is extended to handle this with two independent roughness parameters: α_x along "
            "the tangent and α_y along the bitangent. Brushed metal: α_x (along grooves) ≈ 0.05 (sharp), "
            "α_y (across grooves) ≈ 0.5 (broad). This produces the characteristic stretched highlight.\n\n"
            "In Blender: metallic=1.0, roughness=0.35, anisotropic=0.85, anisotropic_rotation=0.0. "
            "Critically, you need a tangent map to control the brushing direction — use a UV-based tangent "
            "so the streaks follow the surface correctly. For curved brushed metal (like a watch bezel), "
            "use a circular brushing pattern: drive anisotropic_rotation with atan2(UV.y, UV.x)."
        ),
    },
    {
        "user": "Why do bubbles show rainbow colors even though they're just soapy water?",
        "assistant": (
            "Soap bubbles display thin-film interference — a wave optics phenomenon that occurs when "
            "the film thickness is comparable to the wavelength of visible light (400–700 nm).\n\n"
            "A soap bubble is a thin liquid membrane (typically 100–500 nm thick). Light reflects off "
            "both the outer surface and the inner surface of the film. These two reflected beams travel "
            "slightly different path lengths (twice the film thickness for the inner reflection). "
            "When the path difference equals an integer multiple of the wavelength, constructive interference "
            "amplifies that color; when it equals a half-integer multiple, destructive interference "
            "cancels it. Since different wavelengths (colors) satisfy these conditions at different "
            "film thicknesses, you see rainbow bands corresponding to thickness variations across the bubble.\n\n"
            "As the bubble drains, the film at the top becomes very thin (<100 nm). "
            "Below about 100 nm, path difference is less than any visible wavelength, "
            "so all colors interfere destructively — the top appears black just before it pops.\n\n"
            "In Blender: thin-film interference is not built into Principled BSDF, but can be approximated "
            "using a Wavelength→IOR dispersion approach with OSL, or by using the Film Thickness parameter "
            "in some third-party PBR extensions. A simpler approximation: use a Rainbow texture with Fresnel "
            "factor driving the Hue rotation, mixed with clearcoat for the transparent outer surface."
        ),
    },
    {
        "user": "What render settings should I use for the most physically accurate output in Cycles?",
        "assistant": (
            "Physical accuracy in Cycles depends on correct light transport, color management, "
            "and sufficient sampling — in that order.\n\n"
            "Color management (most critical): set Color Space to Linear (in scene settings, Filmic, "
            "or ACES). All textures marked as data (normal maps, roughness, masks) must be set to "
            "Non-Color. Color textures must be sRGB. Mixing these up causes significant error. "
            "Exposure should be set per-scene in the Render Properties, not in post.\n\n"
            "Light paths: set Max Bounces — Diffuse 4–8, Specular 8–16, Transmission 16–32 "
            "(glass needs many bounces). Total 16–32. Caustics are physically correct but noisy — "
            "enable Refractive/Reflective caustics only if necessary and budget the sample count.\n\n"
            "Sampling: use Path Tracing with Adaptive Sampling, threshold 0.01. Start with 512 samples "
            "for test renders, 2048–4096 for final. Enable Denoising (OptiX or OpenImageDenoise) — "
            "it's mathematically valid when used after converged rendering, not as a crutch for low samples.\n\n"
            "Volume: set Volume Step Rate to 0.1 for accurate volumetrics; higher is faster but misses "
            "thin density variations. Homogeneous volumes can use step rate 1.0.\n\n"
            "Physically wrong defaults to fix: disable 'Clamp Indirect' (it removes bright caustics), "
            "keep 'Multiple Importance Sampling' on for all lights, and ensure all emission materials "
            "have their strength in watts/m² not arbitrary units."
        ),
    },
    {
        "user": "Why do wax candles and human skin need subsurface scattering but marble doesn't need as much?",
        "assistant": (
            "The amount of SSS needed depends on the mean free path of photons in the material "
            "relative to the features you're trying to render.\n\n"
            "Human skin: the subsurface_radius for red channel is ~1cm in real tissue. "
            "Skin is full of scattering structures — melanosomes, collagen fibers, fat globules. "
            "The visible consequence is dramatic: shadows soften significantly, ears glow red in backlight, "
            "and the shadow terminator on faces is warm-colored instead of cold.\n\n"
            "Wax: paraffin has even longer scattering paths (~1–2cm for red) because it's a nearly pure "
            "dielectric with few absorbers. Wax candles glow with a warm internal light because photons "
            "travel far before being absorbed — the flame heats the whole wax body and light leaks out everywhere.\n\n"
            "Marble: as described, calcite grain boundaries scatter light but the mean free path is shorter "
            "(2–5mm). The visual consequence — a faint glow and softening of contact shadows — is subtle "
            "but important for hero renders. For distant or secondary objects, marble can often skip SSS.\n\n"
            "Rule of thumb: if an object is thinner than its subsurface radius anywhere, SSS will be visible. "
            "A 5cm slab of marble barely shows SSS; a 3mm marble chip glows visibly. "
            "This is why fingers are more translucent than palms — the tissue is thin enough for "
            "red light to pass all the way through."
        ),
    },
    {
        "user": "How do I physically correctly light a scene to match midday outdoor sunlight?",
        "assistant": (
            "Midday direct sunlight has specific physical parameters you should match precisely:\n\n"
            "Sun lamp: color temperature 5500–5800K (use a Blackbody node or the Sun's temperature control). "
            "Strength: ~1000 W/m² at sea level (Cycles solar constant after atmospheric absorption). "
            "In Blender's Sun light, set strength to 3.0–6.0 depending on your scene scale and exposure. "
            "Angle: 0.526° (the actual angular diameter of the sun) — this controls shadow softness. "
            "Blender's Sun light 'angle' parameter in degrees: 0.526.\n\n"
            "Sky/ambient: use the Sky Texture node (Nishita sky model) for physically based atmospheric "
            "scattering. Set sun elevation to 80° (nearly overhead for midday), sun size 1.0, "
            "air density 1.0, dust density 0.0 for clear day. The sky dome automatically provides "
            "the correct blue ambient fill and horizon gradient.\n\n"
            "Combined: the Sun lamp provides the dominant direct light with hard shadows. "
            "The sky dome provides the diffuse fill (roughly 20–30% of total illuminance at midday). "
            "The sky is brightest at 90° from the sun — this is Rayleigh scattering at work.\n\n"
            "Exposure calibration: with Sun strength ~5.0 and sky at default, a 18% grey card in the scene "
            "should read ~0.18 linear in the render before color grading. Adjust exposure in Camera "
            "properties (EV +/- ) or via the Exposure slider in Render Properties."
        ),
    },
    {
        "user": "What causes the bright rim of light around objects when they're backlit?",
        "assistant": (
            "The backlit rim effect has two distinct physical causes depending on the material:\n\n"
            "For opaque diffuse objects (e.g., a person standing against a bright window): "
            "there is no 'rim' in the strict sense — what you see is the direct contribution of the "
            "bright background light source hitting the geometry that faces backward. "
            "The apparent rim is simply the narrow strip of surface whose normal is nearly aligned "
            "with the light direction. In the rendering equation, L_o includes L_i × cos(θ_i) — "
            "surfaces nearly face-on to the backlight contribute strongly.\n\n"
            "For translucent materials (wax, skin, leaves, thin cloth): the rim is genuinely different. "
            "Photons from behind penetrate the material and scatter forward, exiting through the "
            "front-facing surface you see. This is forward-scatter SSS — the dipole model's wide "
            "subsurface_radius means photons travel far from the entry point. "
            "The rim glows in a shifted color (warmer/redder for skin, greener for leaves) because "
            "long-path photons have been spectrally filtered by selective absorption along the way.\n\n"
            "For metallic objects: the rim is purely geometric — a Fresnel effect where the grazing-angle "
            "reflectance approaches 1.0, so the back-facing metal acts like a mirror and reflects the "
            "bright light source directly toward you in a narrow band.\n\n"
            "In Blender: for the SSS rim, increase subsurface and subsurface_radius appropriately. "
            "For artificial 3-point lighting rims in Eevee, add a Rim shader via Facing gradient × "
            "backlight color as an emission term — but note this is non-physical."
        ),
    },
    {
        "user": "Why does chrome look different from brushed stainless steel if both are metals with low roughness?",
        "assistant": (
            "Chrome and brushed stainless are both high-metallic materials, but they differ in three ways:\n\n"
            "1. Roughness and surface finish: electroplated chrome is deposited atom by atom into an extremely "
            "smooth surface — roughness 0.01–0.05 in PBR terms, approaching a true mirror. "
            "Brushed stainless has directional micro-scratches from the polishing process, "
            "giving roughness 0.2–0.4 in the perpendicular direction (anisotropy 0.8+).\n\n"
            "2. Base color (IOR): chrome (chromium) has IOR (n=3.18, k=3.31) — highly reflective "
            "across the entire visible spectrum, giving a neutral, cold silvery appearance. "
            "Stainless steel (iron with nickel/chromium alloy) has slightly different n,k values. "
            "In practice both appear near-neutral grey, but chrome has a very slightly cooler (bluer) tint "
            "while stainless can appear very slightly warm. Base color approximation: "
            "chrome (0.95, 0.95, 0.95), stainless (0.91, 0.91, 0.90).\n\n"
            "3. Highlight shape: chrome's near-perfect smoothness produces a point-sharp mirror image "
            "of every light source and environment. Brushed stainless's anisotropy produces the "
            "characteristic horizontal streak — narrow along the brush direction, extended across it.\n\n"
            "Blender: chrome = metallic 1.0, roughness 0.03, anisotropic 0.0, base_color (0.95, 0.95, 0.95). "
            "Brushed stainless = metallic 1.0, roughness 0.35, anisotropic 0.85, anisotropic_rotation 0.0, "
            "base_color (0.91, 0.91, 0.90). Use a tangent map to control brushing direction."
        ),
    },
    {
        "user": "How does the IOR value affect the look of a glass object in practice?",
        "assistant": (
            "IOR (index of refraction) controls three distinct visual properties of transparent materials:\n\n"
            "1. Refraction angle: higher IOR = more bending at the glass-air boundary. "
            "A glass ball with IOR 1.5 bends light moderately — the background is distorted but still readable. "
            "A diamond (IOR 2.417) bends light so severely that the background is completely scrambled. "
            "This is Snell's law: sin(θ_refract) = sin(θ_incident) / IOR.\n\n"
            "2. Fresnel reflectance: F0 = ((n-1)/(n+1))². For IOR 1.5 (glass): F0 ≈ 0.04, or 4% reflection at "
            "normal incidence. For IOR 1.333 (water): F0 ≈ 0.02. For IOR 2.417 (diamond): F0 ≈ 0.17, "
            "or 17% — nearly 5× more reflective than glass at straight-on view.\n\n"
            "3. Critical angle for total internal reflection: θ_critical = arcsin(1/n). "
            "Lower IOR = larger critical angle = light escapes more easily. "
            "Higher IOR = smaller critical angle = more light is trapped inside by TIR and redirected toward "
            "preferred exit faces (this is why diamond has been faceted to exploit TIR).\n\n"
            "Practical IOR ranges in Blender: water 1.333, glass crown 1.5, crystal/flint 1.6–1.9, "
            "gem stones 1.45–2.4, diamond 2.417. Using an incorrect IOR (e.g., 1.5 for water) "
            "subtly breaks realism — water caustics will be wrong, and thick water objects won't "
            "look quite right because the refraction distortion is off."
        ),
    },
]


# ─── 3. SIMULATION DATA GENERATOR ────────────────────────────────────────────


class SimulationDataGenerator:
    """
    Orchestrates generation of physics simulation training data for Nalana.

    Combines seed pairs (instant, offline), material QA generation (combinatorial),
    and GPU synthesis job queue creation (for offline GPU batch processing).
    """

    def __init__(
        self,
        output_dir: str = "data/physics",
        anthropic_api_key: str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.anthropic_api_key = anthropic_api_key or os.environ.get(
            "ANTHROPIC_API_KEY"
        )

    def _write_jsonl(self, pairs: list[dict], filename: str) -> Path:
        """Write a list of dicts to a JSONL file, one JSON object per line."""
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        print(f"Wrote {len(pairs)} pairs → {path}")
        return path

    def generate_from_seed_pairs(self) -> Path:
        """
        Save both seed pair lists to JSONL files.
        Returns path to the combined seed file.
        """
        sim_records = [
            {
                "id": f"sim_seed_{i:04d}",
                "source": "seed_simulation",
                "domain": "physics_simulation",
                "messages": [
                    {"role": "user", "content": p["user"]},
                    {"role": "assistant", "content": p["assistant"]},
                ],
            }
            for i, p in enumerate(SEED_SIMULATION_PAIRS)
        ]
        reasoning_records = [
            {
                "id": f"reasoning_seed_{i:04d}",
                "source": "seed_physics_reasoning",
                "domain": "physics_reasoning",
                "messages": [
                    {"role": "user", "content": p["user"]},
                    {"role": "assistant", "content": p["assistant"]},
                ],
            }
            for i, p in enumerate(SEED_PHYSICS_REASONING_PAIRS)
        ]

        all_seeds = sim_records + reasoning_records
        self._write_jsonl(sim_records, "seed_simulation_pairs.jsonl")
        self._write_jsonl(reasoning_records, "seed_reasoning_pairs.jsonl")
        return self._write_jsonl(all_seeds, "seed_all.jsonl")

    def generate_via_gpu_synthesis(self, synthesis_jobs: list[dict]) -> list[dict]:
        """
        Process GPU synthesis jobs that have status 'complete' (i.e., results already fetched).
        For jobs with status 'pending', just logs them and skips.

        In production: the GPU synthesis pipeline (separate script) populates each job's
        'results' field, then this method ingests and normalizes them into training records.

        Args:
            synthesis_jobs: List of job dicts from physics_kb.GPU_SYNTHESIS_JOBS or
                            loaded from a synthesis queue JSON file.

        Returns:
            List of normalized training records from completed jobs.
        """
        all_records: list[dict] = []

        for job in synthesis_jobs:
            job_id = job.get("job_id", "unknown")
            status = job.get("status", "pending")

            if status == "pending":
                print(f"  [PENDING]  {job_id} — not yet synthesized")
                continue

            if status == "running":
                print(f"  [RUNNING]  {job_id} — synthesis in progress")
                continue

            if status != "complete":
                print(f"  [SKIP]     {job_id} — unknown status '{status}'")
                continue

            results = job.get("results", [])
            if not results:
                print(f"  [EMPTY]    {job_id} — completed but no results")
                continue

            for i, raw in enumerate(results):
                record = {
                    "id": f"{job_id}_{i:04d}",
                    "source": f"gpu_synthesis:{job.get('source', 'unknown')}",
                    "domain": job.get("domain", "physics"),
                    "messages": [],
                }

                # Normalize QA pairs from various output formats
                if "user" in raw and "assistant" in raw:
                    record["messages"] = [
                        {"role": "user", "content": raw["user"]},
                        {"role": "assistant", "content": raw["assistant"]},
                    ]
                elif "question" in raw and "answer" in raw:
                    record["messages"] = [
                        {"role": "user", "content": raw["question"]},
                        {"role": "assistant", "content": raw["answer"]},
                    ]

                # Carry through any structured metadata
                for key in (
                    "physics_principle",
                    "blender_parameter",
                    "blender_node",
                    "material_type",
                    "principled_bsdf_params",
                    "reasoning",
                ):
                    if key in raw:
                        record[key] = raw[key]

                if record["messages"]:
                    all_records.append(record)

            print(f"  [COMPLETE] {job_id} — ingested {len(results)} synthesis pairs")

        if all_records:
            self._write_jsonl(all_records, "gpu_synthesis_results.jsonl")

        return all_records

    def generate_blender_sim_pairs(self, n: int = 100) -> list[dict]:
        """
        Generate n combinatorial simulation training pairs by combining
        PBR_PRESETS (materials) with SIMULATION_PRESETS (physics behaviors).

        Each pair shows: "how do I make X material with Y simulation behavior?"
        → correct Blender Python code using both the PBR and simulation presets.

        Args:
            n: Number of pairs to generate.

        Returns:
            List of training record dicts.
        """
        import itertools
        import random

        rng = random.Random(2026)

        # Build list of all (material, sim_category, sim_preset) combinations
        mat_names = list(PBR_PRESETS.keys())
        sim_combos: list[tuple[str, str]] = []
        for cat, presets in SIMULATION_PRESETS.items():
            for preset_name in presets:
                sim_combos.append((cat, preset_name))

        pairs = []
        pool = list(itertools.product(mat_names, sim_combos))
        rng.shuffle(pool)

        for i, (mat_name, (sim_cat, sim_preset)) in enumerate(pool[:n]):
            mat = get_material_preset(mat_name)
            sim = get_simulation_preset(sim_cat, sim_preset)
            code = create_blender_material_code(mat_name, mat)

            user_msg = (
                f"I want to create a {mat_name.replace('_', ' ')} material "
                f"and apply a {sim_preset} {sim_cat} simulation to it. "
                f"Give me the complete Blender Python setup."
            )

            assistant_msg = (
                f"Here's the complete setup for {mat_name.replace('_', ' ')} "
                f"with {sim_preset} {sim_cat} physics:\n\n"
                f"```python\nimport bpy\n\n"
                f"# ── Material Setup ──────────────────────────────────────\n"
                f"obj = bpy.context.active_object\n"
                f"{code}\n\n"
                f"# ── {sim_cat.replace('_', ' ').title()} Physics Setup ({sim_preset}) ──\n"
                + _format_sim_setup_code(sim_cat, sim_preset, sim)
                + "\n```"
            )

            pairs.append(
                {
                    "id": f"combined_{i:04d}",
                    "source": "combinatorial_generation",
                    "domain": "material_and_physics",
                    "material": mat_name,
                    "sim_category": sim_cat,
                    "sim_preset": sim_preset,
                    "messages": [
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": assistant_msg},
                    ],
                }
            )

        self._write_jsonl(pairs, f"combined_sim_material_{n}pairs.jsonl")
        return pairs

    def create_material_qa_pairs(self) -> list[dict]:
        """
        For each PBR preset in physics_kb.PBR_PRESETS, generate a Q&A pair that:
        1. Names the material and asks why those specific PBR values were chosen
        2. Explains the physical reasoning behind each parameter choice

        Returns:
            List of training record dicts.
        """
        pairs = []

        material_explanations: dict[str, str] = {
            "gold": (
                "Gold is set to metallic=1.0 because it is a conductor — light cannot penetrate the surface "
                "and all color comes from the specular reflectance. The yellowish base_color (1.0, 0.766, 0.336) "
                "reflects gold's real spectral reflectance: gold's complex IOR causes selective absorption of "
                "blue wavelengths (plasma frequency effect), leaving red-yellow photons to dominate the reflected spectrum. "
                "Low roughness (0.1) reflects polished gold's smooth crystalline surface."
            ),
            "glass_clear": (
                "Glass is set to transmission=1.0 because light passes through the bulk rather than reflecting off the surface. "
                "IOR=1.5 is standard crown glass, determining how much refraction (bending) occurs at the surface boundary — "
                "governed by Snell's law. Roughness=0.0 means a perfectly smooth surface, giving sharp, undistorted transmission. "
                "Metallic=0.0 because glass is a dielectric (no free electrons). The specular at 0.5 with IOR=1.5 gives "
                "F0≈0.04, meaning 4% reflection at normal incidence — correct for glass."
            ),
            "skin_caucasian": (
                "Skin requires subsurface=0.4 because skin is a multilayer translucent tissue. Light enters the surface, "
                "scatters through the dermis (collagen, blood, fat), and re-emerges displaced. "
                "The subsurface_radius (1.0, 0.2, 0.1) reflects real tissue absorption coefficients: "
                "red light has low absorption in hemoglobin and can travel ~1cm; blue has high absorption and only travels ~1mm. "
                "This is why ears glow red in backlight. Roughness=0.55 models skin's micro-pore texture. "
                "Sheen=0.05 adds the faint sheen from the skin's lipid surface layer at grazing angles."
            ),
            "fabric_velvet": (
                "Velvet is set to sheen=1.0 because velvet's pile fibers create a retroreflective grazing-angle brightening "
                "that is exactly what the Sheen parameter models. The Sheen lobe is an empirical fit to measurements of "
                "fiber-based materials where individual fibers scatter light back toward the viewer at grazing angles. "
                "Roughness=1.0 because the pile structure scatters light in all directions — no coherent specular highlight exists. "
                "Specular=0.0 because the fiber tips absorb or scatter all light before a Fresnel-style specular can form."
            ),
            "carbon_fiber": (
                "Carbon fiber has anisotropic=0.9 because the fiber tows are woven in a specific direction, creating "
                "directional micro-geometry similar to brushed metal. Each fiber bundle acts like a small cylinder, "
                "elongating the specular highlight along the fiber direction. Clearcoat=0.5 models the epoxy resin "
                "coating on top of the fiber layer — a smooth dielectric layer over a rough substrate, "
                "creating two distinct reflection components. Base_color nearly black because carbon fibers absorb "
                "essentially all light that enters them."
            ),
        }

        for mat_name, mat_data in PBR_PRESETS.items():
            # Use specific explanation if available, otherwise generate generic one
            if mat_name in material_explanations:
                explanation = material_explanations[mat_name]
            else:
                explanation = _generate_generic_material_explanation(mat_name, mat_data)

            code = create_blender_material_code(mat_name, mat_data)

            user_msg = (
                f"Why does the '{mat_name.replace('_', ' ')}' material use these specific PBR values? "
                f"And how do I set it up in Blender?"
            )

            assistant_msg = (
                f"Physical reasoning for {mat_name.replace('_', ' ')}:\n\n"
                f"{explanation}\n\n"
                f"Blender Python setup:\n\n"
                f"```python\nimport bpy\n\n"
                f"{code}\n```"
            )

            pairs.append(
                {
                    "id": f"material_qa_{mat_name}",
                    "source": "material_qa_generation",
                    "domain": "pbr_materials",
                    "material": mat_name,
                    "messages": [
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": assistant_msg},
                    ],
                }
            )

        self._write_jsonl(pairs, "material_qa_pairs.jsonl")
        return pairs

    def run_all(self) -> dict[str, int]:
        """
        Full pipeline orchestration. Runs all generation steps in sequence.

        Returns:
            Dict mapping step name → number of pairs generated.
        """
        print("=== Nalana Physics Training Data Generator ===\n")

        print("Step 1: Generating seed pairs...")
        self.generate_from_seed_pairs()
        seed_count = len(SEED_SIMULATION_PAIRS) + len(SEED_PHYSICS_REASONING_PAIRS)

        print("\nStep 2: Creating GPU synthesis job queue...")
        queue_path = self.output_dir / "synthesis_queue.json"
        generate_synthesis_job_queue(GPU_SYNTHESIS_JOBS, str(queue_path))

        print("\nStep 3: Generating combined sim+material pairs (n=100)...")
        combined = self.generate_blender_sim_pairs(n=100)

        print("\nStep 4: Generating material QA pairs...")
        mat_qa = self.create_material_qa_pairs()

        print("\nStep 5: Ingesting any completed synthesis jobs...")
        synth_results = self.generate_via_gpu_synthesis(GPU_SYNTHESIS_JOBS)

        summary = {
            "seed_pairs": seed_count,
            "combined_pairs": len(combined),
            "material_qa_pairs": len(mat_qa),
            "synthesis_pairs": len(synth_results),
            "total": seed_count + len(combined) + len(mat_qa) + len(synth_results),
        }

        print(f"\n{'=' * 50}")
        print("Generation complete. Summary:")
        for key, val in summary.items():
            print(f"  {key:20s}: {val}")
        print(f"{'=' * 50}")

        summary_path = self.output_dir / "generation_summary.json"
        with summary_path.open("w") as f:
            json.dump(
                {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "counts": summary,
                },
                f,
                indent=2,
            )

        return summary


# ─── 4. BLENDER MATERIAL CODE GENERATOR ───────────────────────────────────────


def create_blender_material_code(preset_name: str, preset: dict[str, Any]) -> str:
    """
    Generate Blender Python code string that creates a Principled BSDF material
    node tree matching the given preset dict.

    Args:
        preset_name: Human-readable name used to name the material.
        preset:      Dict of Principled BSDF parameters (as from PBR_PRESETS).

    Returns:
        Python code string (without 'import bpy' — caller adds that).
    """
    mat_var = "mat"
    display_name = preset_name.replace("_", " ").title()

    lines: list[str] = [
        f"# Create material: {display_name}",
        f"{mat_var} = bpy.data.materials.new(name={preset_name!r})",
        f"{mat_var}.use_nodes = True",
        f"nodes = {mat_var}.node_tree.nodes",
        f"links = {mat_var}.node_tree.links",
        "nodes.clear()",
        "",
        "output     = nodes.new('ShaderNodeOutputMaterial')",
        "principled = nodes.new('ShaderNodeBsdfPrincipled')",
        "output.location     = (400, 0)",
        "principled.location = (0, 0)",
        "",
    ]

    # Map preset keys to Principled BSDF input socket names
    socket_map: dict[str, str] = {
        "base_color": "Base Color",
        "metallic": "Metallic",
        "roughness": "Roughness",
        "ior": "IOR",
        "specular": "Specular IOR Level",
        "transmission": "Transmission Weight",
        "subsurface": "Subsurface Weight",
        "subsurface_radius": "Subsurface Radius",
        "anisotropic": "Anisotropic",
        "anisotropic_rotation": "Anisotropic Rotation",
        "clearcoat": "Coat Weight",
        "clearcoat_roughness": "Coat Roughness",
        "sheen": "Sheen Weight",
        "sheen_tint": "Sheen Tint",
        "emission_strength": "Emission Strength",
        "emission_color": "Emission Color",
    }

    for key, socket in socket_map.items():
        if key not in preset:
            continue
        value = preset[key]
        lines.append(f"principled.inputs[{socket!r}].default_value = {value!r}")

    lines.extend(
        [
            "",
            "links.new(principled.outputs['BSDF'], output.inputs['Surface'])",
            "",
            "# Assign to active object",
            "obj = bpy.context.active_object",
            "if obj and obj.type == 'MESH':",
            "    obj.data.materials.clear()",
            f"    obj.data.materials.append({mat_var})",
            f"    print(f'Applied {display_name!r} material to {{obj.name}}')",
        ]
    )

    return "\n".join(lines)


# ─── 5. SYNTHESIS JOB QUEUE WRITER ────────────────────────────────────────────


def generate_synthesis_job_queue(jobs: list[dict[str, Any]], output_path: str) -> None:
    """
    Write GPU synthesis jobs to a JSON queue file for offline GPU processing.

    Each job gets a unique run_id and timestamp. Status is preserved as-is
    (pending/running/complete) so incremental re-runs don't reset completed jobs.

    Args:
        jobs:        List of job dicts (typically GPU_SYNTHESIS_JOBS from physics_kb).
        output_path: File path to write the JSON queue.
    """
    queue: dict[str, Any] = {
        "queue_version": "1.0",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "total_jobs": len(jobs),
        "pending_count": sum(1 for j in jobs if j.get("status") == "pending"),
        "complete_count": sum(1 for j in jobs if j.get("status") == "complete"),
        "total_expected_pairs": sum(j.get("expected_pairs", 0) for j in jobs),
        "jobs": [],
    }

    for job in jobs:
        job_record = dict(job)
        # Add run metadata without overwriting existing fields
        if "run_id" not in job_record:
            job_record["run_id"] = str(uuid.uuid4())
        if "queued_at" not in job_record:
            job_record["queued_at"] = datetime.utcnow().isoformat() + "Z"
        if "completed_at" not in job_record:
            job_record["completed_at"] = None
        if "results" not in job_record:
            job_record["results"] = []
        if "error" not in job_record:
            job_record["error"] = None

        queue["jobs"].append(job_record)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(queue, f, indent=2, ensure_ascii=False)

    print(f"Synthesis queue written → {path}")
    print(f"  {queue['pending_count']} pending jobs")
    print(f"  {queue['complete_count']} completed jobs")
    print(f"  {queue['total_expected_pairs']} total expected pairs when complete")


# ─── INTERNAL HELPERS ─────────────────────────────────────────────────────────


def _format_sim_setup_code(
    category: str,
    preset_name: str,
    params: dict[str, Any],
) -> str:
    """
    Format Blender Python code for setting up a simulation modifier
    from a given SIMULATION_PRESETS entry.

    Args:
        category:    Simulation category (cloth, fluid, rigid_body, soft_body, smoke).
        preset_name: Human-readable preset name.
        params:      Dict of simulation parameters.

    Returns:
        Python code string (without import bpy).
    """
    lines: list[str] = [
        f"# {preset_name.replace('_', ' ').title()} {category.replace('_', ' ')} preset"
    ]

    if category == "cloth":
        lines += [
            "bpy.ops.object.modifier_add(type='CLOTH')",
            "cloth = obj.modifiers['Cloth'].settings",
            f"cloth.quality               = {params.get('quality_steps', 10)}",
            f"cloth.mass                  = {params.get('mass', 1.0)}",
            f"cloth.tension_stiffness     = {params.get('tension_stiffness', 15.0)}",
            f"cloth.compression_stiffness = {params.get('tension_stiffness', 15.0)}",
            f"cloth.shear_stiffness       = {params.get('shear_stiffness', 1.0)}",
            f"cloth.bending_stiffness     = {params.get('bending_stiffness', 0.5)}",
            f"cloth.air_damping           = {params.get('air_resistance', 1.0)}",
            "collision = obj.modifiers['Cloth'].collision_settings",
            f"collision.distance_min       = {params.get('collision_distance', 0.002)}",
            "collision.use_self_collision = True",
        ]

    elif category == "fluid":
        lines += [
            "bpy.ops.object.modifier_add(type='FLUID')",
            "domain = obj.modifiers['Fluid'].domain_settings",
            "domain.domain_type = 'FLUID'",
            f"domain.resolution_max  = {params.get('resolution', 64)}",
            f"domain.time_scale      = {params.get('time_scale', 1.0)}",
            f"domain.surface_tension = {params.get('surface_tension', 0.0728)}",
            f"domain.use_foam        = {params.get('use_foam', False)}",
            f"domain.use_spray       = {params.get('use_spray', False)}",
            f"domain.use_bubbles     = {params.get('use_bubbles', False)}",
        ]

    elif category == "rigid_body":
        lines += [
            "bpy.ops.rigidbody.object_add()",
            "rb = obj.rigid_body",
            "rb.type = 'ACTIVE'",
            f"rb.mass             = {params.get('mass', 1.0)}",
            f"rb.friction         = {params.get('friction', 0.5)}",
            f"rb.restitution      = {params.get('restitution', 0.2)}",
            f"rb.linear_damping   = {params.get('linear_damping', 0.04)}",
            f"rb.angular_damping  = {params.get('angular_damping', 0.1)}",
            f"rb.collision_shape  = '{params.get('collision_shape', 'CONVEX_HULL')}'",
            f"rb.use_deactivation = {params.get('use_deactivation', True)}",
        ]

    elif category == "soft_body":
        lines += [
            "bpy.ops.object.modifier_add(type='SOFT_BODY')",
            "sb = obj.modifiers['Softbody'].settings",
            f"sb.use_goal       = {params.get('use_goal', True)}",
            f"sb.goal_strength  = {params.get('goal_strength', 0.5)}",
            f"sb.goal_damping   = {params.get('goal_damping', 0.2)}",
            f"sb.pull           = {params.get('pull', 0.5)}",
            f"sb.push           = {params.get('push', 0.5)}",
            f"sb.mass           = {params.get('mass', 1.0)}",
            f"sb.use_self_collision = {params.get('self_collision', False)}",
        ]

    elif category == "smoke":
        lines += [
            "bpy.ops.object.modifier_add(type='FLUID')",
            "domain = obj.modifiers['Fluid'].domain_settings",
            "domain.domain_type = 'GAS'",
            f"domain.resolution_max  = {params.get('resolution', 64)}",
            f"domain.temperature_max = {params.get('temperature_max', params.get('temperature', 500.0))}",
            f"domain.vorticity       = {params.get('vorticity', 0.2)}",
            f"domain.use_dissolve    = {params.get('use_dissolve', True)}",
            f"domain.dissolve_speed  = {params.get('dissolve_speed', 25)}",
            "domain.use_noise       = True",
        ]

    else:
        lines.append(f"# Unknown category: {category}")

    return "\n".join(lines)


def _generate_generic_material_explanation(
    mat_name: str,
    mat_data: dict[str, Any],
) -> str:
    """
    Generate a generic physical explanation for a PBR material based on its parameter values.
    Used as fallback when no hand-written explanation exists.

    Args:
        mat_name: Material preset name.
        mat_data: Dict of PBR parameters.

    Returns:
        Paragraph explaining the physical reasoning for the parameter choices.
    """
    parts: list[str] = []

    metallic = mat_data.get("metallic", 0.0)
    roughness = mat_data.get("roughness", 0.5)
    transmission = mat_data.get("transmission", 0.0)
    subsurface = mat_data.get("subsurface", 0.0)
    emission = mat_data.get("emission_strength", 0.0)
    ior = mat_data.get("ior", 1.5)
    clearcoat = mat_data.get("clearcoat", 0.0)
    sheen = mat_data.get("sheen", 0.0)
    anisotropic = mat_data.get("anisotropic", 0.0)

    name_display = mat_name.replace("_", " ")

    if metallic > 0.5:
        parts.append(
            f"{name_display.title()} is a conductive material (metallic={metallic}), "
            f"meaning free electrons absorb and re-emit light at the surface. "
            f"There is no diffuse component — the base color directly tints the specular reflection."
        )
    elif transmission > 0.5:
        parts.append(
            f"{name_display.title()} is a transmissive dielectric (transmission={transmission}). "
            f"Light passes through the bulk, refracting at the surface boundary according to "
            f"Snell's law with IOR={ior}."
        )
    else:
        parts.append(
            f"{name_display.title()} is an opaque dielectric (metallic={metallic}). "
            f"Light scatters off molecular structures in the bulk and re-exits as diffuse color."
        )

    if roughness < 0.15:
        parts.append(
            f"Low roughness ({roughness}) indicates a polished surface where microfacet normals "
            f"are tightly clustered around the macro normal, producing a narrow specular highlight."
        )
    elif roughness > 0.7:
        parts.append(
            f"High roughness ({roughness}) indicates a rough or matte surface. "
            f"Microfacet normals are scattered across a wide hemisphere, diffusing the specular highlight "
            f"into a broad, dim glow indistinguishable from the diffuse lobe."
        )
    else:
        parts.append(
            f"Moderate roughness ({roughness}) represents a semi-polished surface with "
            f"visible but softened specular highlights."
        )

    if subsurface > 0.1:
        parts.append(
            f"Subsurface scattering (subsurface={subsurface}) accounts for the translucent nature "
            f"of the material — photons penetrate below the surface and scatter through the bulk "
            f"before re-emerging, softening shadows and adding interior glow in backlit areas."
        )

    if emission > 0:
        parts.append(
            f"Emission strength={emission} means the material actively emits light, "
            f"approximating thermal emission or luminescence."
        )

    if clearcoat > 0.2:
        parts.append(
            f"Clearcoat={clearcoat} models a smooth lacquer or protective coating over the base material — "
            f"a separate dielectric layer that adds a sharp specular highlight on top of the base material's shading."
        )

    if sheen > 0.2:
        parts.append(
            f"Sheen={sheen} models the grazing-angle brightening of fiber-based or felt-like materials "
            f"where micro-fibers retroreflect light back toward the viewer at extreme angles."
        )

    if anisotropic > 0.3:
        parts.append(
            f"Anisotropic={anisotropic} models a directionally scratched or woven surface "
            f"where the specular highlight is elongated along one axis (like brushed metal or woven fabric)."
        )

    return " ".join(parts)


# ─── CLI ENTRY POINT ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Nalana Physics Simulation Training Data Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Run everything:
              python physics_sim.py --output-dir data/physics --generate-seeds --create-synthesis-queue --n-material-pairs 200

              # Only write seed pairs:
              python physics_sim.py --generate-seeds

              # Only create synthesis queue for GPU batch processing:
              python physics_sim.py --create-synthesis-queue

              # Only generate material pairs:
              python physics_sim.py --n-material-pairs 50
        """),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/physics",
        help="Directory to write all output files (default: data/physics)",
    )
    parser.add_argument(
        "--generate-seeds",
        action="store_true",
        help="Write seed simulation and reasoning pairs to JSONL files",
    )
    parser.add_argument(
        "--create-synthesis-queue",
        action="store_true",
        help="Write GPU synthesis job queue JSON for offline batch processing",
    )
    parser.add_argument(
        "--n-material-pairs",
        type=int,
        default=0,
        metavar="N",
        help="Generate N combinatorial material+simulation pairs (0 = skip)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run full pipeline (equivalent to --generate-seeds --create-synthesis-queue --n-material-pairs 100)",
    )
    parser.add_argument(
        "--anthropic-api-key",
        type=str,
        default=None,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
    )

    args = parser.parse_args()

    generator = SimulationDataGenerator(
        output_dir=args.output_dir,
        anthropic_api_key=args.anthropic_api_key,
    )

    if args.all:
        generator.run_all()
    else:
        ran_something = False

        if args.generate_seeds:
            print("Generating seed pairs...")
            generator.generate_from_seed_pairs()
            generator.create_material_qa_pairs()
            ran_something = True

        if args.create_synthesis_queue:
            print("Creating synthesis queue...")
            queue_path = Path(args.output_dir) / "synthesis_queue.json"
            generate_synthesis_job_queue(GPU_SYNTHESIS_JOBS, str(queue_path))
            ran_something = True

        if args.n_material_pairs > 0:
            print(f"Generating {args.n_material_pairs} combined material+sim pairs...")
            generator.generate_blender_sim_pairs(n=args.n_material_pairs)
            ran_something = True

        if not ran_something:
            print(
                "No action specified. Use --all or specify --generate-seeds, --create-synthesis-queue, or --n-material-pairs N"
            )
            parser.print_help()
