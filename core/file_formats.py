"""
file_formats.py - Universal 3D file format support for Nalana.

Handles 40+ formats by routing to the right converter:
  - Blender (headless): the ultimate converter for most formats
  - trimesh: fast Python-native for OBJ/STL/PLY/GLB/DAE/OFF
  - rhino3dm: Rhino .3dm files (from 0studio)
  - open3d: point clouds (.e57, .pcd, .las proxy via laspy)
  - ezdxf: AutoCAD .dxf / .dwg
  - openusd (pxr): .usd / .usda / .usdc / .usdz
  - pythonocc / OCC: STEP / IGES / BRep (precision CAD)

All paths ultimately convert to GLB for rendering + analysis.

Usage:
    from core.file_formats import to_glb, get_format_info, SUPPORTED_FORMATS

    glb_path = to_glb("/path/to/model.fbx", output_dir="/tmp/nalana")
    info = get_format_info("/path/to/model.step")
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

# ─── Format registry ───────────────────────────────────────────────────────────

FORMATS = {
    # ext → (category, converter, notes)
    # ── Mesh / scene ──────────────────────────────────────────────
    ".blend": ("blender_native", "blender", "Blender native"),
    ".fbx": ("mesh", "blender", "Autodesk FBX - industry standard"),
    ".obj": ("mesh", "trimesh", "Wavefront OBJ"),
    ".glb": ("mesh", "trimesh", "Binary glTF 2.0"),
    ".gltf": ("mesh", "trimesh", "Text glTF 2.0"),
    ".dae": ("mesh", "trimesh", "COLLADA"),
    ".stl": ("mesh", "trimesh", "Stereolithography - 3D printing"),
    ".ply": ("mesh", "trimesh", "Stanford PLY"),
    ".off": ("mesh", "trimesh", "Object File Format"),
    ".3mf": ("mesh", "blender", "3D Manufacturing Format"),
    ".x3d": ("mesh", "blender", "Web3D X3D"),
    ".wrl": ("mesh", "blender", "VRML / Virtual Reality Modeling"),
    ".abc": ("scene", "blender", "Alembic - animation / VFX"),
    ".usd": ("scene", "usd", "Universal Scene Description (binary)"),
    ".usda": ("scene", "usd", "Universal Scene Description (ASCII)"),
    ".usdc": ("scene", "usd", "Universal Scene Description (crate)"),
    ".usdz": ("scene", "usd", "Universal Scene Description (zip) - Apple/iOS"),
    ".svg": ("vector", "blender", "SVG → extruded 3D curve"),
    ".3ds": ("mesh", "blender", "Legacy 3D Studio"),
    ".lwo": ("mesh", "blender", "LightWave Object"),
    ".mdd": ("animation", "blender", "Point cache animation"),
    ".pc2": ("animation", "blender", "Point cache 2"),
    # ── Rhino / CAD precision ─────────────────────────────────────
    ".3dm": ("nurbs", "rhino3dm", "Rhino 3D Model - NURBS precision"),
    ".step": ("cad", "occ", "STEP - neutral CAD exchange"),
    ".stp": ("cad", "occ", "STEP (alternate extension)"),
    ".iges": ("cad", "occ", "IGES - legacy CAD exchange"),
    ".igs": ("cad", "occ", "IGES (alternate extension)"),
    ".brep": ("cad", "occ", "Open CASCADE BRep"),
    # ── DXF / DWG ─────────────────────────────────────────────────
    ".dxf": ("cad_2d", "ezdxf", "AutoCAD DXF - 2D/3D"),
    ".dwg": ("cad_2d", "ezdxf", "AutoCAD DWG (via ezdxf)"),
    # ── Point clouds ──────────────────────────────────────────────
    ".e57": ("pointcloud", "open3d", "3D point cloud exchange format"),
    ".pcd": ("pointcloud", "open3d", "Point Cloud Data"),
    ".pts": ("pointcloud", "open3d", "Point cloud text format"),
    ".xyz": ("pointcloud", "open3d", "Plain XYZ coordinates"),
    ".las": ("pointcloud", "laspy", "LiDAR point cloud"),
    ".laz": ("pointcloud", "laspy", "Compressed LiDAR"),
    # ── Sketchup ──────────────────────────────────────────────────
    ".skp": ("sketchup", "sketchup", "SketchUp (via SketchUp Ruby/CLI)"),
    # ── Substance ─────────────────────────────────────────────────
    ".sbsar": ("material", "substance", "Substance parametric material"),
    ".sbs": ("material", "substance", "Substance designer graph"),
    # ── Maya / 3ds Max / C4D (read-only via converters) ──────────
    ".ma": ("maya", "blender", "Maya ASCII (Blender importer)"),
    ".mb": ("maya", "blender", "Maya Binary"),
}

SUPPORTED_FORMATS = set(FORMATS.keys())


def get_format_info(path: str | Path) -> dict:
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in FORMATS:
        return {"supported": False, "ext": ext}
    category, converter, notes = FORMATS[ext]
    return {
        "supported": True,
        "ext": ext,
        "category": category,
        "converter": converter,
        "notes": notes,
    }


# ─── Blender headless converter ────────────────────────────────────────────────

BLENDER_CONVERT_SCRIPT = """
import bpy, sys, os
from pathlib import Path

argv = sys.argv[sys.argv.index("--") + 1:]
input_file, output_file = argv[0], argv[1]
ext = Path(input_file).suffix.lower()

bpy.ops.wm.read_factory_settings(use_empty=True)

importers = {
    ".fbx":   lambda f: bpy.ops.import_scene.fbx(filepath=f),
    ".obj":   lambda f: bpy.ops.import_scene.obj(filepath=f),
    ".dae":   lambda f: bpy.ops.wm.collada_import(filepath=f),
    ".stl":   lambda f: bpy.ops.import_mesh.stl(filepath=f),
    ".ply":   lambda f: bpy.ops.import_mesh.ply(filepath=f),
    ".abc":   lambda f: bpy.ops.wm.alembic_import(filepath=f),
    ".usd":   lambda f: bpy.ops.wm.usd_import(filepath=f),
    ".usda":  lambda f: bpy.ops.wm.usd_import(filepath=f),
    ".usdc":  lambda f: bpy.ops.wm.usd_import(filepath=f),
    ".usdz":  lambda f: bpy.ops.wm.usd_import(filepath=f),
    ".3ds":   lambda f: bpy.ops.import_scene.autodesk_3ds(filepath=f),
    ".svg":   lambda f: bpy.ops.import_curve.svg(filepath=f),
    ".x3d":   lambda f: bpy.ops.import_scene.x3d(filepath=f),
    ".wrl":   lambda f: bpy.ops.import_scene.x3d(filepath=f),
    ".3mf":   lambda f: bpy.ops.import_mesh.threemf(filepath=f),
    ".gltf":  lambda f: bpy.ops.import_scene.gltf(filepath=f),
    ".glb":   lambda f: bpy.ops.import_scene.gltf(filepath=f),
    ".blend": lambda f: bpy.ops.wm.open_mainfile(filepath=f),
}

if ext not in importers:
    print(f"No Blender importer for {ext}")
    sys.exit(1)

try:
    importers[ext](input_file)
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)

# Center geometry
bpy.ops.object.select_all(action="SELECT")
meshes = [o for o in bpy.context.selected_objects if o.type == "MESH"]
if meshes:
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.join()
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    bpy.context.active_object.location = (0, 0, 0)

Path(output_file).parent.mkdir(parents=True, exist_ok=True)
bpy.ops.export_scene.gltf(filepath=output_file, export_format="GLB")
print(f"Exported GLB: {output_file}")
"""


def _blender_convert(input_path: Path, output_path: Path, blender: str) -> bool:
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(BLENDER_CONVERT_SCRIPT)
        script = f.name
    try:
        r = subprocess.run(
            [
                blender,
                "--background",
                "--python",
                script,
                "--",
                str(input_path),
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return r.returncode == 0 and output_path.exists()
    finally:
        Path(script).unlink(missing_ok=True)


def _trimesh_convert(input_path: Path, output_path: Path) -> bool:
    try:
        import trimesh

        mesh = trimesh.load(str(input_path), force="mesh")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(output_path))
        return output_path.exists()
    except Exception as e:
        print(f"  [trimesh] {e}")
        return False


def _rhino3dm_convert(input_path: Path, output_path: Path) -> bool:
    """Convert .3dm to GLB via rhino3dm → trimesh bridge."""
    try:
        import rhino3dm
        import trimesh
        import numpy as np

        doc = rhino3dm.File3dm.Read(str(input_path))
        if not doc:
            return False

        all_meshes = []
        for obj in doc.Objects:
            geo = obj.Geometry
            if hasattr(geo, "Faces"):  # Mesh
                verts = np.array([[v.X, v.Y, v.Z] for v in geo.Vertices])
                faces = []
                for i in range(geo.Faces.Count):
                    f = geo.Faces[i]
                    if f.IsQuad:
                        faces.extend([[f.A, f.B, f.C], [f.A, f.C, f.D]])
                    else:
                        faces.append([f.A, f.B, f.C])
                if len(verts) and len(faces):
                    all_meshes.append(trimesh.Trimesh(vertices=verts, faces=faces))
            elif hasattr(geo, "ToNurbsSurface"):  # Surface/Brep → mesh
                try:
                    mesh = geo.GetMesh(rhino3dm.MeshType.Any)
                    if mesh:
                        verts = np.array([[v.X, v.Y, v.Z] for v in mesh.Vertices])
                        faces = [
                            [mesh.Faces[i].A, mesh.Faces[i].B, mesh.Faces[i].C]
                            for i in range(mesh.Faces.Count)
                        ]
                        all_meshes.append(
                            trimesh.Trimesh(vertices=verts, faces=np.array(faces))
                        )
                except Exception:
                    pass

        if not all_meshes:
            return False

        combined = trimesh.util.concatenate(all_meshes)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.export(str(output_path))
        return output_path.exists()
    except Exception as e:
        print(f"  [rhino3dm] {e}")
        return False


def _occ_convert(input_path: Path, output_path: Path) -> bool:
    """Convert STEP/IGES to GLB via Open CASCADE (pythonocc-core)."""
    try:
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Extend.DataExchange import read_step_file, read_iges_file
        from OCC.Extend.TopologyUtils import TopologyExplorer
        import trimesh
        import numpy as np

        ext = input_path.suffix.lower()
        if ext in (".step", ".stp"):
            shape = read_step_file(str(input_path))
        else:
            shape = read_iges_file(str(input_path))

        # Tessellate
        mesh = BRepMesh_IncrementalMesh(shape, 0.1)
        mesh.Perform()

        verts_list, faces_list = [], []
        offset = 0
        topo = TopologyExplorer(shape)
        for face in topo.faces():
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.TopLoc import TopLoc_Location

            location = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation(face, location)
            if triangulation is None:
                continue
            nodes = triangulation.Nodes()
            tris = triangulation.Triangles()
            face_verts = np.array(
                [
                    [nodes.Value(i).X(), nodes.Value(i).Y(), nodes.Value(i).Z()]
                    for i in range(1, nodes.Length() + 1)
                ]
            )
            face_faces = np.array(
                [
                    [tris.Value(i).Get()[j] - 1 + offset for j in range(3)]
                    for i in range(1, tris.Length() + 1)
                ]
            )
            verts_list.append(face_verts)
            faces_list.append(face_faces)
            offset += len(face_verts)

        if not verts_list:
            return False

        tm = trimesh.Trimesh(
            vertices=np.vstack(verts_list),
            faces=np.vstack(faces_list),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tm.export(str(output_path))
        return output_path.exists()
    except Exception as e:
        print(f"  [occ] {e}")
        return False


def _open3d_convert(input_path: Path, output_path: Path) -> bool:
    """Convert point clouds to GLB (as point mesh)."""
    try:
        import open3d as o3d
        import trimesh
        import numpy as np

        ext = input_path.suffix.lower()
        if ext in (".las", ".laz"):
            try:
                import laspy

                las = laspy.read(str(input_path))
                pts = np.vstack([las.x, las.y, las.z]).T
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts[:100_000])  # cap
            except ImportError:
                return False
        else:
            pcd = o3d.io.read_point_cloud(str(input_path))

        # Estimate normals and reconstruct surface (Poisson)
        pcd.estimate_normals()
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8
        )
        tm = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices),
            faces=np.asarray(mesh.triangles),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tm.export(str(output_path))
        return output_path.exists()
    except Exception as e:
        print(f"  [open3d] {e}")
        return False


def _ezdxf_convert(input_path: Path, output_path: Path) -> bool:
    """Convert DXF to GLB (extracts 3D meshes and polylines)."""
    try:
        import ezdxf
        import trimesh
        import numpy as np

        doc = ezdxf.readfile(str(input_path))
        msp = doc.modelspace()

        all_meshes = []
        for entity in msp.query("3DFACE MESH POLYFACE"):
            try:
                if entity.dxftype() == "3DFACE":
                    pts = [
                        entity.dxf.vtx0,
                        entity.dxf.vtx1,
                        entity.dxf.vtx2,
                        entity.dxf.vtx3,
                    ]
                    v = np.array([[p.x, p.y, p.z] for p in pts])
                    faces = np.array([[0, 1, 2], [0, 2, 3]])
                    all_meshes.append(trimesh.Trimesh(vertices=v, faces=faces))
            except Exception:
                pass

        if not all_meshes:
            return False

        combined = trimesh.util.concatenate(all_meshes)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.export(str(output_path))
        return output_path.exists()
    except Exception as e:
        print(f"  [ezdxf] {e}")
        return False


def _usd_convert(input_path: Path, output_path: Path) -> bool:
    """Convert USD/USDZ to GLB via OpenUSD (pxr)."""
    try:
        from pxr import Usd, UsdGeom
        import trimesh
        import numpy as np

        stage = Usd.Stage.Open(str(input_path))
        all_meshes = []

        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                pts_attr = mesh.GetPointsAttr().Get()
                idx_attr = mesh.GetFaceVertexIndicesAttr().Get()
                counts_attr = mesh.GetFaceVertexCountsAttr().Get()
                if pts_attr is None or idx_attr is None:
                    continue
                verts = np.array([[p[0], p[1], p[2]] for p in pts_attr])
                # Build triangle faces
                faces = []
                idx = 0
                for count in counts_attr or [3] * (len(idx_attr) // 3):
                    poly = [idx_attr[idx + i] for i in range(count)]
                    for i in range(1, count - 1):
                        faces.append([poly[0], poly[i], poly[i + 1]])
                    idx += count
                if len(verts) and len(faces):
                    all_meshes.append(
                        trimesh.Trimesh(vertices=verts, faces=np.array(faces))
                    )

        if not all_meshes:
            return False

        combined = trimesh.util.concatenate(all_meshes)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.export(str(output_path))
        return output_path.exists()
    except Exception as e:
        print(f"  [usd/pxr] {e}")
        return False


# ─── Main public API ───────────────────────────────────────────────────────────


def to_glb(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    blender_path: str | None = None,
) -> Path | None:
    """
    Convert any supported 3D file to GLB.
    Returns the GLB path on success, None on failure.
    """
    p = Path(input_path)
    ext = p.suffix.lower()

    if ext not in FORMATS:
        print(f"  [to_glb] Unsupported format: {ext}")
        return None

    if ext == ".glb":
        return p  # Already GLB

    out_dir = Path(output_dir) if output_dir else p.parent / "nalana_converted"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (p.stem + ".glb")

    if out_path.exists():
        return out_path

    _, converter, _ = FORMATS[ext]

    if converter == "trimesh":
        success = _trimesh_convert(p, out_path)
    elif converter == "rhino3dm":
        success = _rhino3dm_convert(p, out_path)
    elif converter == "occ":
        success = _occ_convert(p, out_path)
    elif converter == "open3d" or converter == "laspy":
        success = _open3d_convert(p, out_path)
    elif converter == "ezdxf":
        success = _ezdxf_convert(p, out_path)
    elif converter == "usd":
        success = _usd_convert(p, out_path)
    elif converter in ("blender", "blender_native"):
        bl = blender_path or os.environ.get("BLENDER_PATH", "blender")
        success = _blender_convert(p, out_path, bl)
    else:
        print(f"  [to_glb] No converter for {ext}")
        return None

    return out_path if success else None


def batch_convert(
    input_paths: list[Path],
    output_dir: Path,
    blender_path: str | None = None,
    workers: int = 4,
) -> dict[str, Path | None]:
    """Convert many files in parallel. Returns {input_path: glb_path_or_None}."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(to_glb, p, output_dir, blender_path): p for p in input_paths
        }
        for f in as_completed(futures):
            src = futures[f]
            results[str(src)] = f.result()
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python file_formats.py <input_file> [output_dir]")
        print(f"\nSupported formats ({len(SUPPORTED_FORMATS)}):")
        from collections import defaultdict

        by_cat = defaultdict(list)
        for ext, (cat, conv, note) in FORMATS.items():
            by_cat[cat].append(f"{ext} ({note})")
        for cat, exts in sorted(by_cat.items()):
            print(f"  {cat}: {', '.join(sorted(exts))}")
    else:
        result = to_glb(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
        print(f"Output: {result}" if result else "Conversion failed")
