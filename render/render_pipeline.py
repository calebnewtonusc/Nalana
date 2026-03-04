"""
render_pipeline.py - Headless Blender multi-view rendering of 3D objects.

For each object in data/objaverse/metadata.jsonl:
  1. Spawns headless Blender
  2. Imports the GLB/OBJ file
  3. Renders from 8 standard views (front, back, left, right, top, bottom, front-iso, back-iso)
  4. Dumps topology stats (vertex/face count, manifold, bounding box)
  5. Saves renders to data/objaverse/renders/{uid}/

Usage:
    python render_pipeline.py --blender-path /path/to/blender
    python render_pipeline.py --workers 4 --limit 1000
    python render_pipeline.py --uid specific-uid-here

Blender paths (common):
    macOS:  /Applications/Blender.app/Contents/MacOS/Blender
    Linux:  /usr/bin/blender  (or wherever installed on the cluster)
    Azure:  blender (if in PATH after: apt-get install blender)
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

OBJAVERSE_DIR = Path(__file__).parents[1] / "data" / "objaverse"
RENDERS_DIR   = OBJAVERSE_DIR / "renders"

# The Blender script that actually does the rendering (runs inside Blender's Python)
BLENDER_RENDER_SCRIPT = """
import bpy
import sys
import json
import math
from pathlib import Path

argv = sys.argv
script_args_start = argv.index('--') + 1
args = argv[script_args_start:]
input_file  = args[0]
output_dir  = Path(args[1])
uid         = args[2]

output_dir.mkdir(parents=True, exist_ok=True)

# ── Reset scene ──────────────────────────────────────────────────────────────
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.samples = 64
bpy.context.scene.render.resolution_x = 512
bpy.context.scene.render.resolution_y = 512
bpy.context.scene.render.image_settings.file_format = 'PNG'

# ── World / lighting ─────────────────────────────────────────────────────────
bpy.ops.world.new()
world = bpy.data.worlds[0]
world.use_nodes = True
bg = world.node_tree.nodes['Background']
bg.inputs['Strength'].default_value = 1.5

# Three-point lighting
def add_light(name, energy, loc, light_type='AREA'):
    bpy.ops.object.light_add(type=light_type, location=loc)
    light = bpy.context.active_object
    light.name = name
    light.data.energy = energy
    return light

add_light('Key',   800, (4, -4, 6))
add_light('Fill',  300, (-4, -2, 4))
add_light('Rim',   400, (0,  6, 2))

# ── Import object ─────────────────────────────────────────────────────────────
ext = Path(input_file).suffix.lower()
if ext == '.glb' or ext == '.gltf':
    bpy.ops.import_scene.gltf(filepath=input_file)
elif ext == '.obj':
    bpy.ops.import_scene.obj(filepath=input_file)
elif ext == '.fbx':
    bpy.ops.import_scene.fbx(filepath=input_file)
elif ext == '.blend':
    bpy.ops.wm.open_mainfile(filepath=input_file)
else:
    print(f"Unsupported format: {ext}")
    sys.exit(1)

# ── Center and normalize ──────────────────────────────────────────────────────
# Only join mesh objects — joining non-mesh types (cameras, lights, armatures)
# causes bpy.ops.object.join() to fail with "cannot join different types".
bpy.ops.object.select_all(action='DESELECT')
for ob in bpy.context.scene.objects:
    if ob.type == 'MESH':
        ob.select_set(True)
mesh_objects = [ob for ob in bpy.context.scene.objects if ob.type == 'MESH']
if mesh_objects:
    bpy.context.view_layer.objects.active = mesh_objects[0]
    if len(mesh_objects) > 1:
        bpy.ops.object.join()
obj = bpy.context.active_object
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
obj.location = (0, 0, 0)

# Scale to fit in 2-unit cube
dims = obj.dimensions
max_dim = max(dims.x, dims.y, dims.z)
if max_dim > 0:
    scale = 2.0 / max_dim
    obj.scale = (scale, scale, scale)
    bpy.ops.object.transform_apply(scale=True)

# ── Shade smooth for nicer renders ───────────────────────────────────────────
bpy.ops.object.shade_smooth()

# ── Topology stats ────────────────────────────────────────────────────────────
mesh = obj.data
stats = {
    "uid": uid,
    "vertex_count": len(mesh.vertices),
    "face_count": len(mesh.polygons),
    "edge_count": len(mesh.edges),
    "dimensions": [round(d, 4) for d in obj.dimensions],
    "has_uv": len(mesh.uv_layers) > 0,
    "material_count": len(obj.material_slots),
}
(output_dir / "stats.json").write_text(json.dumps(stats, indent=2))

# ── Camera setup ──────────────────────────────────────────────────────────────
bpy.ops.object.camera_add()
cam = bpy.context.active_object
bpy.context.scene.camera = cam
cam.data.lens = 50

def render_view(name, loc, rot_euler_deg):
    import math
    cam.location = loc
    cam.rotation_euler = [math.radians(r) for r in rot_euler_deg]
    bpy.context.scene.render.filepath = str(output_dir / f"{name}.png")
    bpy.ops.render.render(write_still=True)

# 8 standard views
render_view('front',      (0, -4, 0),     (90, 0, 0))
render_view('back',       (0,  4, 0),     (90, 0, 180))
render_view('left',       (-4, 0, 0),     (90, 0, 90))
render_view('right',      (4,  0, 0),     (90, 0, -90))
render_view('top',        (0,  0, 4),     (0,  0, 0))
render_view('bottom',     (0,  0, -4),    (180,0, 0))
render_view('iso_front',  (3, -3, 2.5),   (63, 0, 45))
render_view('iso_back',   (-3, 3, 2.5),   (63, 0, 225))

print(f"Rendered {uid} -> {output_dir}")
"""


def get_blender_path(hint: str | None = None) -> str:
    candidates = [
        hint,
        os.environ.get("BLENDER_PATH"),
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "/usr/bin/blender",
        "/usr/local/bin/blender",
        "blender",
    ]
    for c in candidates:
        if c and (Path(c).exists() or c == "blender"):
            return c
    raise RuntimeError("Blender not found. Pass --blender-path or set BLENDER_PATH env var.")


def render_object(uid: str, input_file: str, blender_path: str) -> tuple[str, bool, str]:
    out_dir = RENDERS_DIR / uid
    if (out_dir / "front.png").exists():
        return uid, True, "cached"

    # Write the Blender script to a temp file using full uid to avoid collisions
    import tempfile
    script_path = RENDERS_DIR / f"_render_script_{uid}.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(BLENDER_RENDER_SCRIPT)

    try:
        result = subprocess.run(
            [blender_path, "--background", "--python", str(script_path),
             "--", input_file, str(out_dir), uid],
            capture_output=True, text=True, timeout=120,
        )
        script_path.unlink(missing_ok=True)
        if result.returncode == 0 and (out_dir / "front.png").exists():
            return uid, True, "ok"
        else:
            return uid, False, result.stderr[-500:] if result.stderr else "unknown error"
    except subprocess.TimeoutExpired:
        script_path.unlink(missing_ok=True)
        return uid, False, "timeout"
    except Exception as e:
        script_path.unlink(missing_ok=True)
        return uid, False, str(e)


def load_metadata() -> list[dict]:
    meta_path = OBJAVERSE_DIR / "metadata.jsonl"
    if not meta_path.exists():
        return []
    return [json.loads(l) for l in meta_path.read_text().splitlines() if l.strip()]


def main():
    parser = argparse.ArgumentParser(description="Render Objaverse objects via headless Blender")
    parser.add_argument("--blender-path", help="Path to Blender executable")
    parser.add_argument("--workers", type=int, default=4, help="Parallel Blender processes")
    parser.add_argument("--limit", type=int, help="Max objects to render")
    parser.add_argument("--uid", help="Render a single specific UID")
    args = parser.parse_args()

    blender = get_blender_path(args.blender_path)
    print(f"Blender: {blender}")

    objects = load_metadata()
    if not objects:
        print("No metadata. Run objaverse_prep.py first.")
        return

    if args.uid:
        objects = [o for o in objects if o["uid"] == args.uid]

    already_done = set(p.name for p in RENDERS_DIR.glob("*") if (RENDERS_DIR / p.name / "front.png").exists())
    pending = [o for o in objects if o["uid"] not in already_done]
    if args.limit:
        pending = pending[:args.limit]

    print(f"Total objects: {len(objects):,}")
    print(f"Already rendered: {len(already_done):,}")
    print(f"Pending: {len(pending):,}")
    print(f"Workers: {args.workers}")
    print()

    if not pending:
        print("All rendered.")
        return

    ok = err = 0
    pbar = tqdm(total=len(pending), unit="obj") if HAS_TQDM else None

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(render_object, o["uid"], o["local_path"], blender): o
            for o in pending
        }
        for future in as_completed(futures):
            uid, success, msg = future.result()
            if success:
                ok += 1
            else:
                err += 1
                if msg != "cached":
                    print(f"\n  [ERR] {uid}: {msg[:120]}")
            if pbar:
                pbar.set_postfix(ok=ok, err=err)
                pbar.update(1)

    if pbar:
        pbar.close()

    print(f"\nDone. OK: {ok:,}  Errors: {err:,}")
    print(f"Next step: python annotate_forms.py")


if __name__ == "__main__":
    main()
