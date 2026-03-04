"""
universal_dsl.py - The Universal 3D Operation Language.

THE KEY INNOVATION: Instead of generating Maya-specific or Blender-specific code,
Nalana thinks in a software-agnostic Universal DSL, then compiles to any target.

This means:
  - Train once → works in Blender, Maya, Cinema 4D, Houdini, Rhino, Unreal
  - Add a new software target without retraining
  - Cross-software knowledge transfer (what you learn from Maya tutorials
    makes Blender output better and vice versa)

DSL Structure:
  op     → canonical operation name (EXTRUDE, BEVEL, SUBDIVIDE, etc.)
  args   → software-agnostic parameters
  target → what the op acts on
  intent → human-readable description of what this achieves

Software Targets:
  blender, maya, cinema4d, houdini, rhino, sketchup, unreal, fusion360, zbrush
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any

# ─── Universal Operation Catalog ───────────────────────────────────────────────


@dataclass
class UniversalOp:
    op: str
    args: dict[str, Any] = field(default_factory=dict)
    target: str = "active_object"
    intent: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "UniversalOp":
        return cls(**d)


# ─── Software compilers ────────────────────────────────────────────────────────


class BlenderCompiler:
    def compile(self, op: UniversalOp) -> str:
        a = op.args
        match op.op:
            # ── Primitives ────────────────────────────────────────────────
            case "ADD_CUBE":
                s = a.get("size", 2)
                loc = a.get("location", (0, 0, 0))
                return (
                    f"bpy.ops.mesh.primitive_cube_add(size={s}, location={tuple(loc)})"
                )
            case "ADD_SPHERE":
                r = a.get("radius", 1)
                seg = a.get("segments", 32)
                return f"bpy.ops.mesh.primitive_uv_sphere_add(radius={r}, segments={seg}, ring_count=16)"
            case "ADD_CYLINDER":
                r = a.get("radius", 1)
                d = a.get("depth", 2)
                v = a.get("vertices", 32)
                return f"bpy.ops.mesh.primitive_cylinder_add(radius={r}, depth={d}, vertices={v})"
            case "ADD_PLANE":
                s = a.get("size", 2)
                return f"bpy.ops.mesh.primitive_plane_add(size={s})"
            case "ADD_TORUS":
                R = a.get("major_radius", 1)
                r = a.get("minor_radius", 0.25)
                return f"bpy.ops.mesh.primitive_torus_add(major_radius={R}, minor_radius={r})"
            case "ADD_CONE":
                r1 = a.get("radius1", 1)
                r2 = a.get("radius2", 0)
                d = a.get("depth", 2)
                return f"bpy.ops.mesh.primitive_cone_add(radius1={r1}, radius2={r2}, depth={d})"
            case "ADD_EMPTY":
                return "bpy.ops.object.empty_add(type='PLAIN_AXES')"
            case "ADD_ARMATURE":
                return "bpy.ops.object.armature_add()"
            case "ADD_CURVE":
                return "bpy.ops.curve.primitive_bezier_curve_add()"
            # ── Object transforms ─────────────────────────────────────────
            case "TRANSLATE":
                v = a.get("value", (0, 0, 0))
                axis = a.get("axis")
                if axis == "X":
                    return f"bpy.ops.transform.translate(value=({v[0]},0,0), constraint_axis=(True,False,False))"
                elif axis == "Y":
                    return f"bpy.ops.transform.translate(value=(0,{v[1]},0), constraint_axis=(False,True,False))"
                elif axis == "Z":
                    return f"bpy.ops.transform.translate(value=(0,0,{v[2]}), constraint_axis=(False,False,True))"
                return f"bpy.ops.transform.translate(value={tuple(v)})"
            case "ROTATE":
                angle = a.get("angle_deg", 90)
                axis = a.get("axis", "Z")
                import math

                rad = round(math.radians(angle), 6)
                return f"bpy.ops.transform.rotate(value={rad}, orient_axis='{axis}')"
            case "SCALE":
                v = a.get("value", (1, 1, 1))
                if isinstance(v, (int, float)):
                    v = (v, v, v)
                return f"bpy.ops.transform.resize(value={tuple(v)})"
            case "APPLY_TRANSFORMS":
                loc = a.get("location", True)
                rot = a.get("rotation", True)
                sc = a.get("scale", True)
                return f"bpy.ops.object.transform_apply(location={loc}, rotation={rot}, scale={sc})"
            case "SET_ORIGIN":
                t = a.get("type", "ORIGIN_GEOMETRY")
                return f"bpy.ops.object.origin_set(type='{t}', center='MEDIAN')"
            # ── Mode switching ─────────────────────────────────────────────
            case "ENTER_EDIT_MODE":
                return "bpy.ops.object.mode_set(mode='EDIT')"
            case "ENTER_OBJECT_MODE":
                return "bpy.ops.object.mode_set(mode='OBJECT')"
            case "ENTER_SCULPT_MODE":
                return "bpy.ops.object.mode_set(mode='SCULPT')"
            case "ENTER_WEIGHT_PAINT":
                return "bpy.ops.object.mode_set(mode='WEIGHT_PAINT')"
            # ── Edit mode ops ──────────────────────────────────────────────
            case "EXTRUDE":
                dist = a.get("amount", 1)
                return f"bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={{'value':(0,0,{dist})}})"
            case "INSET":
                t = a.get("thickness", 0.1)
                d = a.get("depth", 0)
                return f"bpy.ops.mesh.inset_faces(thickness={t}, depth={d})"
            case "BEVEL":
                offset = a.get("offset", 0.1)
                segs = a.get("segments", 2)
                return (
                    f"bpy.ops.mesh.bevel(offset={offset}, segments={segs}, profile=0.5)"
                )
            case "LOOP_CUT":
                n = a.get("cuts", 1)
                return f"bpy.ops.mesh.loop_cut(number_cuts={n}, smoothness=0)"
            case "SUBDIVIDE":
                n = a.get("cuts", 1)
                return f"bpy.ops.mesh.subdivide(number_cuts={n}, smoothness=0)"
            case "KNIFE":
                return "bpy.ops.mesh.knife_tool(use_occlude_geometry=True)"
            case "BRIDGE":
                return "bpy.ops.mesh.bridge_edge_loops()"
            case "FILL":
                return "bpy.ops.mesh.fill()"
            case "DISSOLVE_EDGES":
                return "bpy.ops.mesh.dissolve_edges()"
            case "DISSOLVE_FACES":
                return "bpy.ops.mesh.dissolve_faces()"
            case "MERGE_VERTICES":
                t = a.get("type", "CENTER")
                return f"bpy.ops.mesh.merge(type='{t}')"
            case "REMOVE_DOUBLES":
                thresh = a.get("threshold", 0.001)
                return f"bpy.ops.mesh.remove_doubles(threshold={thresh})"
            case "FLIP_NORMALS":
                return "bpy.ops.mesh.flip_normals()"
            case "RECALC_NORMALS":
                inside = a.get("inside", False)
                return f"bpy.ops.mesh.normals_make_consistent(inside={inside})"
            case "SELECT_ALL":
                return "bpy.ops.mesh.select_all(action='SELECT')"
            case "DESELECT_ALL":
                return "bpy.ops.mesh.select_all(action='DESELECT')"
            case "SEPARATE":
                return "bpy.ops.mesh.separate(type='SELECTED')"
            # ── Modifiers ─────────────────────────────────────────────────
            case "ADD_SUBDIVISION":
                lvl = a.get("levels", 2)
                return f"bpy.ops.object.modifier_add(type='SUBSURF')\nbpy.context.object.modifiers['Subdivision'].levels = {lvl}"
            case "ADD_MIRROR":
                axis = a.get("axis", "X")
                idx = {"X": 0, "Y": 1, "Z": 2}.get(axis.upper(), 0)
                return f"bpy.ops.object.modifier_add(type='MIRROR')\nbpy.context.object.modifiers['Mirror'].use_axis[{idx}] = True"
            case "ADD_SOLIDIFY":
                t = a.get("thickness", 0.1)
                return f"bpy.ops.object.modifier_add(type='SOLIDIFY')\nbpy.context.object.modifiers['Solidify'].thickness = {t}"
            case "ADD_BEVEL_MOD":
                w = a.get("width", 0.1)
                segs = a.get("segments", 2)
                return f"bpy.ops.object.modifier_add(type='BEVEL')\nbpy.context.object.modifiers['Bevel'].width = {w}\nbpy.context.object.modifiers['Bevel'].segments = {segs}"
            case "ADD_ARRAY":
                n = a.get("count", 3)
                return f"bpy.ops.object.modifier_add(type='ARRAY')\nbpy.context.object.modifiers['Array'].count = {n}"
            case "ADD_BOOLEAN":
                op_type = a.get("operation", "DIFFERENCE")
                return f"bpy.ops.object.modifier_add(type='BOOLEAN')\nbpy.context.object.modifiers['Boolean'].operation = '{op_type}'"
            case "ADD_SHRINKWRAP":
                return "bpy.ops.object.modifier_add(type='SHRINKWRAP')"
            case "ADD_DECIMATE":
                ratio = a.get("ratio", 0.5)
                return f"bpy.ops.object.modifier_add(type='DECIMATE')\nbpy.context.object.modifiers['Decimate'].ratio = {ratio}"
            case "ADD_REMESH":
                return "bpy.ops.object.modifier_add(type='REMESH')"
            case "APPLY_MODIFIER":
                name = a.get("name", "Subdivision")
                return f"bpy.ops.object.modifier_apply(modifier='{name}')"
            # ── Shading ───────────────────────────────────────────────────
            case "SHADE_SMOOTH":
                return "bpy.ops.object.shade_smooth()"
            case "SHADE_FLAT":
                return "bpy.ops.object.shade_flat()"
            # ── Object ops ────────────────────────────────────────────────
            case "DUPLICATE":
                return "bpy.ops.object.duplicate_move()"
            case "DELETE":
                return "bpy.ops.object.delete(use_global=False)"
            case "JOIN":
                return "bpy.ops.object.join()"
            case "PARENT_SET":
                return "bpy.ops.object.parent_set(type='OBJECT', keep_transform=False)"
            # ── UV / Materials ────────────────────────────────────────────
            case "UNWRAP_UV":
                return "bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.001)"
            case "SMART_UV_PROJECT":
                return "bpy.ops.uv.smart_project(angle_limit=66, island_margin=0.02)"
            case "ADD_MATERIAL":
                return "bpy.ops.object.material_slot_add()\nbpy.ops.material.new()"
            # ── Sculpt ────────────────────────────────────────────────────
            case "VOXEL_REMESH":
                size = a.get("voxel_size", 0.05)
                return f"bpy.context.object.data.remesh_voxel_size = {size}\nbpy.ops.object.voxel_remesh()"
            case "DYNAMIC_TOPOLOGY":
                return "bpy.ops.sculpt.dynamic_topology_toggle()"
            # ── Render ────────────────────────────────────────────────────
            case "RENDER":
                return "bpy.ops.render.render(write_still=True)"
            case "SET_RENDER_ENGINE":
                engine = a.get("engine", "CYCLES")
                return f"bpy.context.scene.render.engine = '{engine}'"
            # ── Lighting ──────────────────────────────────────────────────
            case "ADD_LIGHT":
                light_type = a.get("type", "AREA").upper()
                energy = a.get("energy", 500)
                loc = a.get("location", (4, -4, 6))
                return (
                    f"bpy.ops.object.light_add(type='{light_type}', location={tuple(loc)})\n"
                    f"bpy.context.object.data.energy = {energy}"
                )
            case "THREE_POINT_LIGHTING":
                return (
                    "bpy.ops.object.light_add(type='AREA', location=(4,-4,6))\n"
                    "bpy.context.object.data.energy = 800\n"
                    "bpy.ops.object.light_add(type='AREA', location=(-4,-2,4))\n"
                    "bpy.context.object.data.energy = 300\n"
                    "bpy.ops.object.light_add(type='AREA', location=(0,6,2))\n"
                    "bpy.context.object.data.energy = 400"
                )
            case "ADD_HDRI":
                path = a.get("path", "")
                return (
                    "world = bpy.data.worlds.new('World')\n"
                    "world.use_nodes = True\n"
                    "bg = world.node_tree.nodes['Background']\n"
                    f"env = world.node_tree.nodes.new('ShaderNodeTexEnvironment')\n"
                    f"env.image = bpy.data.images.load('{path}')\n"
                    "world.node_tree.links.new(env.outputs['Color'], bg.inputs['Color'])"
                )
            case _:
                raise NotImplementedError(f"Unsupported operation: {op.op}")


class MayaCompiler:
    def compile(self, op: UniversalOp) -> str:
        a = op.args
        match op.op:
            case "ADD_CUBE":
                s = a.get("size", 2) / 2
                return f"cmds.polyCube(w={s * 2}, h={s * 2}, d={s * 2})"
            case "ADD_SPHERE":
                r = a.get("radius", 1)
                return f"cmds.polySphere(r={r}, sx=32, sy=16)"
            case "ADD_CYLINDER":
                r = a.get("radius", 1)
                h = a.get("depth", 2)
                return f"cmds.polyCylinder(r={r}, h={h}, sx=32)"
            case "EXTRUDE":
                d = a.get("amount", 1)
                return f"cmds.polyExtrudeFacet(ltz={d})"
            case "BEVEL":
                offset = a.get("offset", 0.1)
                segs = a.get("segments", 2)
                return f"cmds.polyBevel3(offset={offset}, segments={segs})"
            case "SUBDIVIDE":
                return "cmds.polySubdivideFacet(dv=1)"
            case "ADD_SUBDIVISION":
                lvl = a.get("levels", 2)
                return f"cmds.displaySmoothness(polygonObject={lvl})"
            case "SHADE_SMOOTH":
                return "cmds.polySoftEdge(a=180)"
            case "SHADE_FLAT":
                return "cmds.polySoftEdge(a=0)"
            case "DUPLICATE":
                return "cmds.duplicate()"
            case "DELETE":
                return "cmds.delete()"
            case "TRANSLATE":
                v = a.get("value", (0, 0, 0))
                return f"cmds.move({v[0]}, {v[1]}, {v[2]}, r=True)"
            case "ROTATE":
                ang = a.get("angle_deg", 90)
                axis = a.get("axis", "Z").lower()
                kwargs = {f"r{axis}": ang}
                return f"cmds.rotate({', '.join(f'{k}={v}' for k, v in kwargs.items())}, r=True)"
            case "SCALE":
                v = a.get("value", (1, 1, 1))
                if isinstance(v, (int, float)):
                    v = (v, v, v)
                return f"cmds.scale({v[0]}, {v[1]}, {v[2]}, r=True)"
            case "RENDER":
                return "cmds.render()"
            case _:
                return f"# TODO: {op.op} not yet compiled for Maya"


class Cinema4DCompiler:
    def compile(self, op: UniversalOp) -> str:
        a = op.args
        match op.op:
            case "ADD_CUBE":
                s = a.get("size", 2) * 100  # C4D uses cm
                return (
                    f"obj = c4d.BaseObject(c4d.Ocube)\n"
                    f"obj[c4d.PRIM_CUBE_LEN] = c4d.Vector({s},{s},{s})\n"
                    f"doc.InsertObject(obj)\nc4d.EventAdd()"
                )
            case "ADD_SPHERE":
                r = a.get("radius", 1) * 100
                return (
                    f"obj = c4d.BaseObject(c4d.Osphere)\n"
                    f"obj[c4d.PRIM_SPHERE_RAD] = {r}\n"
                    f"doc.InsertObject(obj)\nc4d.EventAdd()"
                )
            case "ADD_SUBDIVISION":
                return (
                    "sds = c4d.BaseObject(c4d.Osds)\n"
                    "sds.InsertUnder(doc.GetActiveObject())\nc4d.EventAdd()"
                )
            case "ADD_MIRROR":
                return (
                    "sym = c4d.BaseObject(c4d.Osymmetry)\n"
                    "sym.InsertUnder(doc.GetActiveObject())\nc4d.EventAdd()"
                )
            case "SHADE_SMOOTH":
                return "c4d.CallCommand(12139)  # Phong tag"
            case _:
                return f"# TODO: {op.op} not yet compiled for Cinema 4D"


class HoudiniCompiler:
    def compile(self, op: UniversalOp) -> str:
        a = op.args
        match op.op:
            case "ADD_CUBE":
                s = a.get("size", 2)
                return (
                    f"geo = hou.node('/obj').createNode('geo')\n"
                    f"box = geo.createNode('box')\n"
                    f"box.parm('sizex').set({s})\nbox.parm('sizey').set({s})\nbox.parm('sizez').set({s})"
                )
            case "ADD_SPHERE":
                r = a.get("radius", 1)
                return (
                    f"geo = hou.node('/obj').createNode('geo')\n"
                    f"sphere = geo.createNode('sphere')\n"
                    f"sphere.parm('rad1').set({r})"
                )
            case "EXTRUDE":
                d = a.get("amount", 1)
                return f"polyextrude = geo.createNode('polyextrude::2.0')\npolyextrude.parm('dist').set({d})"
            case "SUBDIVIDE":
                return "subdiv = geo.createNode('subdivide')\nsubdiv.parm('iterations').set(1)"
            case _:
                return f"# TODO: {op.op} not yet compiled for Houdini"


class RhinoCompiler:
    def compile(self, op: UniversalOp) -> str:
        a = op.args
        match op.op:
            case "ADD_CUBE":
                s = a.get("size", 2)
                h = s / 2
                return f"rs.AddBox([[-{h},-{h},-{h}],[{h},-{h},-{h}],[{h},{h},-{h}],[-{h},{h},-{h}]], {s})"
            case "ADD_SPHERE":
                r = a.get("radius", 1)
                return f"rs.AddSphere([0,0,0], {r})"
            case "EXTRUDE":
                d = a.get("amount", 1)
                return f"rs.ExtrudeSurface(rs.GetObject(), rs.VectorCreate([0,0,{d}],[0,0,0]))"
            case "BEVEL":
                r = a.get("offset", 0.1)
                return f"rs.FilletEdge(rs.GetObject(), {r})"
            case _:
                return f"# TODO: {op.op} not yet compiled for Rhino"


# ─── Compiler registry ─────────────────────────────────────────────────────────

COMPILERS: dict[str, type] = {
    "blender": BlenderCompiler,
    "maya": MayaCompiler,
    "cinema4d": Cinema4DCompiler,
    "houdini": HoudiniCompiler,
    "rhino": RhinoCompiler,
}


def compile_op(op: UniversalOp, software: str) -> str:
    cls = COMPILERS.get(software.lower())
    if not cls:
        return f"# Software '{software}' not yet supported. Universal op: {op.op}"
    return cls().compile(op)


def compile_sequence(ops: list[UniversalOp], software: str) -> list[str]:
    return [compile_op(op, software) for op in ops]


def op_from_blender_python(blender_python: str, voice_command: str = "") -> UniversalOp:
    """
    Reverse-compile: infer Universal DSL from a bpy.ops call.
    Used during dataset normalization.
    """
    import re

    py = blender_python.strip()

    # Map common bpy.ops patterns to Universal ops
    patterns = [
        (r"mesh\.primitive_cube_add", "ADD_CUBE"),
        (r"mesh\.primitive_uv_sphere_add", "ADD_SPHERE"),
        (r"mesh\.primitive_cylinder_add", "ADD_CYLINDER"),
        (r"mesh\.primitive_plane_add", "ADD_PLANE"),
        (r"mesh\.primitive_torus_add", "ADD_TORUS"),
        (r"mesh\.primitive_cone_add", "ADD_CONE"),
        (r"mesh\.extrude_region_move", "EXTRUDE"),
        (r"mesh\.inset_faces", "INSET"),
        (r"mesh\.bevel", "BEVEL"),
        (r"mesh\.loop_cut", "LOOP_CUT"),
        (r"mesh\.subdivide", "SUBDIVIDE"),
        (r"mesh\.bridge_edge_loops", "BRIDGE"),
        (r"mesh\.dissolve_edges", "DISSOLVE_EDGES"),
        (r"mesh\.dissolve_faces", "DISSOLVE_FACES"),
        (r"mesh\.merge", "MERGE_VERTICES"),
        (r"mesh\.remove_doubles", "REMOVE_DOUBLES"),
        (r"mesh\.flip_normals", "FLIP_NORMALS"),
        (r"mesh\.normals_make_consistent", "RECALC_NORMALS"),
        (r"transform\.translate", "TRANSLATE"),
        (r"transform\.rotate", "ROTATE"),
        (r"transform\.resize", "SCALE"),
        (r"object\.shade_smooth", "SHADE_SMOOTH"),
        (r"object\.shade_flat", "SHADE_FLAT"),
        (r"object\.duplicate", "DUPLICATE"),
        (r"object\.delete", "DELETE"),
        (r"object\.join", "JOIN"),
        (r"modifier_add.*SUBSURF", "ADD_SUBDIVISION"),
        (r"modifier_add.*MIRROR", "ADD_MIRROR"),
        (r"modifier_add.*SOLIDIFY", "ADD_SOLIDIFY"),
        (r"modifier_add.*BEVEL", "ADD_BEVEL_MOD"),
        (r"modifier_add.*ARRAY", "ADD_ARRAY"),
        (r"modifier_add.*BOOLEAN", "ADD_BOOLEAN"),
        (r"object\.modifier_apply", "APPLY_MODIFIER"),
        (r"mode_set.*EDIT", "ENTER_EDIT_MODE"),
        (r"mode_set.*OBJECT", "ENTER_OBJECT_MODE"),
        (r"mode_set.*SCULPT", "ENTER_SCULPT_MODE"),
        (r"uv\.unwrap", "UNWRAP_UV"),
        (r"uv\.smart_project", "SMART_UV_PROJECT"),
        (r"object\.light_add.*AREA", "ADD_LIGHT"),
        (r"render\.render", "RENDER"),
        (r"voxel_remesh", "VOXEL_REMESH"),
        (r"dynamic_topology_toggle", "DYNAMIC_TOPOLOGY"),
    ]

    for pattern, universal_op in patterns:
        if re.search(pattern, py, re.IGNORECASE):
            return UniversalOp(op=universal_op, intent=voice_command)

    return UniversalOp(op="UNKNOWN", args={"raw": py}, intent=voice_command)


def normalize_training_pair(pair: dict) -> dict:
    """
    Add Universal DSL to an existing training pair.
    Call this during dataset preparation to enrich all pairs.
    """
    blender_python = pair.get("blender_python", "")
    voice_command = pair.get("voice_command", "")

    universal_op = op_from_blender_python(blender_python, voice_command)

    pair["universal_dsl"] = universal_op.to_dict()
    pair["software_implementations"] = {
        sw: compile_op(universal_op, sw) for sw in COMPILERS
    }
    return pair


if __name__ == "__main__":
    # Demo: build a cube, extrude a face, add subdivision — in every software
    ops = [
        UniversalOp("ADD_CUBE", {"size": 2}, intent="Start with a cube"),
        UniversalOp("ENTER_EDIT_MODE", intent="Go into edit mode"),
        UniversalOp("INSET", {"thickness": 0.3}, intent="Inset the top face"),
        UniversalOp("EXTRUDE", {"amount": 0.5}, intent="Extrude the face up"),
        UniversalOp("ENTER_OBJECT_MODE", intent="Return to object mode"),
        UniversalOp("ADD_SUBDIVISION", {"levels": 2}, intent="Smooth with subdivision"),
        UniversalOp("SHADE_SMOOTH", intent="Apply smooth shading"),
    ]

    for sw in COMPILERS:
        print(f"\n{'═' * 50}")
        print(f"  {sw.upper()}")
        print(f"{'═' * 50}")
        for code in compile_sequence(ops, sw):
            print(code)
