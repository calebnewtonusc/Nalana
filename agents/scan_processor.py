"""
scan_processor.py — Nalana 3D Scan Processing Agent

Generates training data for processing real-world 3D capture data:
- Photogrammetry cleanup (from Agisoft, RealityCapture, Meshroom)
- LiDAR point cloud processing
- NeRF/Gaussian Splatting to mesh conversion
- Mesh decimation while preserving detail
- Scan-to-BIM for architecture
- Medical scan to mesh (DICOM, CT/MRI)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

# ─── Output paths ──────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parents[1]
SCAN_DATA_DIR = BASE_DIR / "data" / "scan_processing"
SCAN_DATA_DIR.mkdir(parents=True, exist_ok=True)
PAIRS_OUTPUT = SCAN_DATA_DIR / "scan_processing_pairs.jsonl"

# ─── Scan Type Characteristics ─────────────────────────────────────────────────

SCAN_TYPES: dict[str, dict[str, Any]] = {
    "photogrammetry": {
        "description": (
            "Structure-from-Motion (SfM) reconstructions from overlapping photographs. "
            "Software: Agisoft Metashape, RealityCapture, Meshroom (AliceVision), COLMAP. "
            "Accuracy depends on photo count, overlap (>60%), and lighting consistency."
        ),
        "typical_polygon_count": (1_000_000, 20_000_000),
        "typical_texture_resolution": "8K-16K per atlas, multiple atlases common",
        "formats": [".obj", ".ply", ".fbx", ".abc", ".gltf"],
        "common_issues": [
            "Watertight holes in reflective or featureless surfaces (glass, sky, water)",
            "Texture stretching at UV seams from projection artifacts",
            "Noisy geometry in low-contrast regions",
            "Outlier floaters — disconnected geometry clusters in the background",
            "Incorrect scale — always requires a scale bar or known reference in the scene",
            "Camera shake or subject motion causes ghosting / blur artifacts in mesh",
            "Underside of objects missing if camera never shot from below",
        ],
        "workflow_origin": "SfM pipeline: sparse point cloud -> dense point cloud -> mesh -> texture",
        "triangle_quality": "Isotropic triangles (uniform size), poor topology for rigging",
        "accuracy_real_world": "0.1mm to 5mm depending on camera, distance, and calibration",
        "software_notes": {
            "agisoft_metashape": "Industry standard. Highest quality output. Export as OBJ+texture for Blender.",
            "realitycapture": "Fastest processing. Excellent for large-scale environments.",
            "meshroom": "Open source (AliceVision). Slower but free. Good for learning SfM pipeline.",
        },
    },
    "structured_light": {
        "description": (
            "Projects known light patterns (stripes, dot grids) onto the object and captures "
            "deformation with a camera. Systems: Artec Eva, Artec Leo, Shining3D, Creaform. "
            "Faster and more controlled than photogrammetry for small objects."
        ),
        "typical_polygon_count": (100_000, 5_000_000),
        "typical_texture_resolution": "No texture by default; color capture is optional add-on",
        "formats": [".stl", ".obj", ".ply", ".fbx"],
        "common_issues": [
            "Specular/shiny surface gaps — bright spots return invalid depth",
            "Thin feature loss — edges thinner than sensor resolution are rounded off",
            "Alignment errors between scan passes if tracking fails",
            "Missing data under overhangs (light cannot reach)",
            "Thermal expansion artifacts in precise engineering scans",
        ],
        "workflow_origin": "Hardware project-and-capture -> point cloud -> mesh reconstruction (Poisson/VCG)",
        "triangle_quality": "Dense, fairly uniform triangles; oversampled relative to true surface detail",
        "accuracy_real_world": "0.05mm to 0.5mm depending on scanner model and calibration",
        "software_notes": {
            "artec_studio": "Pairs with Artec Eva/Leo hardware. Excellent fusion and hole-filling.",
            "geomagic_wrap": "Industry standard for scan cleanup in manufacturing and dental.",
            "meshmixer": "Free Autodesk tool. Good for quick cleanup and repair.",
        },
    },
    "lidar_mobile": {
        "description": (
            "Mobile terrestrial LiDAR scanners (handheld or backpack-mounted). "
            "Systems: Leica BLK2GO, FARO Focus, NavVis M6/VLX, iPhone LiDAR. "
            "Captures entire room or building interior in minutes."
        ),
        "point_density": "1,000 to 50,000 points per square meter",
        "typical_polygon_count_after_meshing": (500_000, 50_000_000),
        "formats": [".las", ".laz", ".e57", ".ply", ".pts"],
        "common_issues": [
            "Dynamic object ghosting — people and vehicles that moved during scan appear as translucent blobs",
            "Glass and water surfaces return no valid returns (LiDAR passes through or scatters)",
            "Noisy returns on mirrors — infinite bounce returns false depth",
            "Registration drift in long corridors without enough distinctive features",
            "Ground plane and walls may have slight bow due to IMU drift",
            "Wiring and thin cables rarely resolved — too narrow for beam footprint",
        ],
        "coordinate_system": "Typically local LiDAR frame; georeferencing with GPS adds global coordinates",
        "scan_pattern": "Continuous rotation + vertical oscillation while walking",
        "software_notes": {
            "cloudcompare": "Open source, excellent for point cloud processing and registration.",
            "recap_pro": "Autodesk. Good for construction and architecture workflows.",
            "faro_scene": "Proprietary. Best for FARO hardware; powerful registration tools.",
            "leica_cyclone": "Best for Leica hardware; BIM integration via Cyclone REGISTER.",
        },
    },
    "lidar_aerial": {
        "description": (
            "Airborne LiDAR (ALS) mounted on aircraft, helicopter, or drone. "
            "Captures large terrain areas (hundreds of square kilometers). "
            "Used for urban mapping, forestry, corridor mapping, archaeology."
        ),
        "point_density": "5 to 100 points per square meter (drone: up to 500 pts/m2)",
        "typical_polygon_count_after_meshing": (1_000_000, 500_000_000),
        "formats": [".las", ".laz"],
        "common_issues": [
            "Vegetation penetration — first returns from canopy, last returns from ground",
            "Vertical face gaps — building facades nearly parallel to laser beam receive sparse returns",
            "Water surface returns false or no data (absorption or specular reflection)",
            "Power lines and thin wires rarely captured without very high density",
            "Urban canyon shadows — tall buildings mask neighboring areas from aircraft angles",
        ],
        "classification_standard": "ASPRS LAS: 0=unclassified, 2=ground, 5=vegetation, 6=building",
        "coordinate_system": "UTM or state plane with ellipsoidal height",
        "software_notes": {
            "lastools": "Industry standard for ALS processing. Proprietary but powerful.",
            "pdal": "Open source point cloud pipeline. Excellent for scripted batch processing.",
            "arcgis_pro": "Best for GIS workflows and urban digital twin pipelines.",
        },
    },
    "ct_medical": {
        "description": (
            "Computed Tomography reconstructs a 3D volume from X-ray projections at multiple angles. "
            "Medical CT: 0.5-2mm slice thickness. Industrial CT (micro-CT): 5-100 micron voxels. "
            "Outputs volumetric voxel data (DICOM); mesh extraction requires marching cubes."
        ),
        "voxel_size": "0.5mm to 2mm (medical); 0.005mm to 0.1mm (micro-CT)",
        "typical_polygon_count_after_meshing": (500_000, 10_000_000),
        "formats": [".dcm", ".nii", ".nifti", ".mhd", ".raw"],
        "common_issues": [
            "Beam hardening — dark bands between dense objects (metal implants cause severe artifacts)",
            "Partial volume effect — boundary voxels average adjacent tissues, softening edges",
            "Ring artifacts from faulty detector rows in older scanners",
            "Metal streak artifacts from hip replacements, dental work, surgical clips",
            "Patient motion during scan causes blurring (breathing, heartbeat)",
            "Noise in low-dose protocols — quantum noise makes small structure segmentation hard",
        ],
        "segmentation_thresholds": {
            "air": "< -900 HU",
            "soft_tissue": "-100 to 100 HU",
            "bone": "> 400 HU",
            "metal": "> 3000 HU",
        },
        "coordinate_system": "DICOM uses LPS (Left-Posterior-Superior); convert to RAS for most 3D tools",
        "software_notes": {
            "3d_slicer": "Free, open source. Best for medical segmentation and mesh extraction.",
            "mimics": "Industry standard in medical device manufacturing. Accurate segmentation.",
            "itk_snap": "Open source. Good for semi-automatic segmentation.",
        },
    },
    "nerf_instant": {
        "description": (
            "Neural Radiance Fields trained from photograph collections. "
            "Instant-NGP (NVIDIA) trains in minutes on a consumer GPU. "
            "Represents scene as a continuous volumetric radiance function, not explicit mesh. "
            "Mesh extraction requires density-threshold marching cubes."
        ),
        "typical_polygon_count_after_meshing": (200_000, 5_000_000),
        "formats": [".obj", ".ply", "via marching cubes"],
        "common_issues": [
            "Floaters — small density clusters where neural net failed to converge",
            "Boundary smoothing — the density field fades at edges, creating rounded boundaries",
            "No clean silhouettes — alpha/transparency edges are fuzzy in the neural volume",
            "Reflective and transparent surfaces often fail (NeRF assumes Lambertian surfaces)",
            "Mesh extracted via marching cubes has uniform triangle distribution (no topology)",
            "Texture baked from NeRF can show view-dependent lighting artifacts",
        ],
        "training_requirements": {
            "photos": "30-300 images, well-distributed viewing angles",
            "gpu_vram": "8GB+ for Instant-NGP; 24GB+ for large scenes",
            "training_time": "1-10 minutes (Instant-NGP), hours (vanilla NeRF)",
        },
        "software_notes": {
            "instant_ngp": "NVIDIA's fastest implementation. GUI + Python API.",
            "nerfstudio": "Modular NeRF framework. Multiple model types. Best for research.",
            "luma_ai": "Commercial cloud service. Upload photos, get NeRF + mesh output.",
        },
    },
    "gaussian_splatting": {
        "description": (
            "3D Gaussian Splatting (3DGS) represents scenes as millions of 3D Gaussians "
            "with color, opacity, and covariance. Real-time rendering at high quality. "
            "Training from the same photo inputs as NeRF, but rasterization-based output. "
            "No clean mesh by default — mesh extraction is an active research problem."
        ),
        "typical_gaussian_count": (500_000, 6_000_000),
        "formats": [".ply (splats format)", ".ksplat", ".splat"],
        "common_issues": [
            "No clean mesh without conversion (SuGaR, GOF, or marching cubes on opacity volume)",
            "Needle-like Gaussians in poorly constrained areas — visual but non-physical",
            "Background sky and ground often contain noisy Gaussians",
            "No physical collision geometry — splats are purely visual",
            "File sizes are large (100MB-1GB for scene splats)",
            "Real-time rendering requires WebGL or Vulkan-capable GPU",
        ],
        "mesh_extraction_methods": {
            "SuGaR": "Regularized Gaussian Splatting with surface alignment. Best quality mesh.",
            "GOF": "Gaussian Opacity Fields — density volume -> marching cubes mesh.",
            "opacity_threshold": "Extract mesh by thresholding Gaussian opacity; fast but jagged.",
        },
        "software_notes": {
            "3dgs_original": "Kerbl et al. 2023. Reference implementation in Python/CUDA.",
            "luma_ai": "Commercial. Upload photos -> 3DGS output + SuGaR mesh extraction.",
            "postshot": "Real-time 3DGS viewer with training. Windows native.",
            "nerfstudio_splatfacto": "3DGS in the Nerfstudio framework.",
        },
    },
}

# ─── Decimation Targets ────────────────────────────────────────────────────────

DECIMATION_TARGETS: dict[str, dict[str, Any]] = {
    "game_hero_prop": {
        "polygon_range": (5_000, 25_000),
        "description": "Hero prop seen close-up in a game. Detailed silhouette, baked normal map.",
        "texture_budget": "2K-4K diffuse + normal + ORM",
        "lod_chain": True,
        "notes": (
            "Retopology often manual for hero props. Decimate modifier only as a starting point. "
            "Normal map from high-poly bake recovers surface detail lost in decimation."
        ),
    },
    "game_background": {
        "polygon_range": (500, 5_000),
        "description": "Background prop rarely seen closely. Aggressive decimation acceptable.",
        "texture_budget": "512-1K shared atlas",
        "lod_chain": True,
        "notes": "Instanced background props can be as low as 100-500 polygons.",
    },
    "film_hero": {
        "polygon_range": (100_000, 500_000),
        "description": "Hero asset for film/VFX. Close-up shots, subdivision-ready, UDIM textures.",
        "texture_budget": "4K-16K UDIM tiles (multiple)",
        "lod_chain": False,
        "notes": (
            "Film assets are rendered via subdivision. 100K quads = base mesh; "
            "subdiv level 2 = 1.6M polygons at render time. Topology must be clean quads."
        ),
    },
    "real_time_preview": {
        "polygon_range": (1_000, 10_000),
        "description": "Interactive preview in viewport or configurator. Balance speed and quality.",
        "texture_budget": "1K-2K",
        "lod_chain": False,
        "notes": "Used in AR configurators, e-commerce 3D viewers (glTF).",
    },
    "archviz_hero": {
        "polygon_range": (50_000, 200_000),
        "description": "Hero architectural element for arch-viz rendering.",
        "texture_budget": "4K PBR, sometimes UDIM for complex pieces",
        "lod_chain": False,
        "notes": "Archviz renders allow higher poly budgets than games.",
    },
    "archviz_background": {
        "polygon_range": (1_000, 20_000),
        "description": "Background furniture, distant building, street clutter in arch-viz.",
        "texture_budget": "1K-2K atlas",
        "lod_chain": False,
        "notes": "Background archviz assets can share texture atlases.",
    },
    "3d_print": {
        "polygon_range": (50_000, 2_000_000),
        "description": "Export-ready for FDM, SLA, or SLS 3D printing. Must be manifold and watertight.",
        "texture_budget": "None — printers don't use texture maps",
        "lod_chain": False,
        "notes": (
            "3D print meshes need high polygon count to preserve smooth curves. "
            "Use STL or 3MF format. Minimum wall thickness: 0.8mm for FDM, 0.5mm for SLA."
        ),
    },
    "xr_mobile": {
        "polygon_range": (500, 3_000),
        "description": "AR/VR on mobile (iOS ARKit, Android ARCore). Extreme polygon budget pressure.",
        "texture_budget": "512-1K, compressed ETC2/ASTC",
        "lod_chain": True,
        "notes": (
            "Mobile XR targets 60fps at 1440p. Keep assets under 3K polygons. "
            "Batch multiple assets into one draw call via atlased textures. "
            "glTF/glb with draco compression for delivery."
        ),
    },
    "digital_twin": {
        "polygon_range": (10_000, 100_000),
        "description": "Asset in a digital twin or BIM model. Geometry must match real-world accurately.",
        "texture_budget": "2K-4K or procedural",
        "lod_chain": True,
        "notes": (
            "Digital twin assets balance fidelity with performance for real-time simulation. "
            "Scan data is often used directly if under 100K polygons post-decimation."
        ),
    },
}

# ─── Mesh Repair Operations ────────────────────────────────────────────────────

MESH_REPAIR_OPERATIONS: list[dict[str, Any]] = [
    {
        "step": 1,
        "name": "fill_holes",
        "description": (
            "Identify and fill boundary edge loops (open holes in the mesh surface). "
            "Small holes: use flat or fan fill. Large holes: use Poisson surface reconstruction "
            "or manual bridge-edge reconstruction to match surrounding surface curvature."
        ),
        "blender_api": "bmesh.ops.holes_fill(bm, edges=boundary_edges, sides=4)",
        "when_to_use": "Always — open meshes cannot be 3D printed and cause shading artifacts",
        "caution": (
            "Large hole fills can introduce flat triangles that don't match surrounding curvature. "
            "Smooth or subdivide after filling."
        ),
    },
    {
        "step": 2,
        "name": "remove_isolated_vertices",
        "description": (
            "Delete vertices that are not connected to any edge or face. "
            "Common in scan data where sparse point cloud regions were meshed but produced "
            "disconnected vertex islands."
        ),
        "blender_api": "bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_edges], context='VERTS')",
        "when_to_use": "Always — isolated vertices inflate vert count and confuse downstream tools",
        "caution": "None — pure cleanup, no visual impact.",
    },
    {
        "step": 3,
        "name": "fix_non_manifold_edges",
        "description": (
            "Non-manifold edges are shared by more than 2 faces (interior duplicate faces) "
            "or fewer than 1 face (boundary). Fix by deleting interior duplicate faces, "
            "merging nearly-identical faces, or splitting multi-face edge loops."
        ),
        "blender_api": "Select non-manifold with bm.edges[i].is_manifold; delete or dissolve offending geometry",
        "when_to_use": "Before any boolean operation, 3D print export, or subdivision",
        "caution": "Aggressive non-manifold removal can delete valid thin-shell geometry. Inspect visually.",
    },
    {
        "step": 4,
        "name": "merge_duplicate_vertices",
        "description": (
            "Merge vertices that occupy the same or nearly the same position. "
            "Common in scan data where overlapping scan passes produce double-layer surfaces. "
            "Use a threshold of 0.1-1mm depending on scan accuracy."
        ),
        "blender_api": "bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.001)",
        "when_to_use": "After importing scan data, especially from aligned multiple scan passes",
        "caution": "Threshold too large -> legitimate geometry collapsed. Too small -> nothing merged.",
    },
    {
        "step": 5,
        "name": "recalculate_normals",
        "description": (
            "Ensure all face normals point outward consistently. Scan data often has mixed normal "
            "directions depending on scan angle and reconstruction algorithm. "
            "Blender's Recalculate Outside uses winding order and topology to determine "
            "the correct outward direction."
        ),
        "blender_api": "bmesh.ops.recalc_face_normals(bm, faces=bm.faces)",
        "when_to_use": "After any topology changes; before baking, rendering, or export",
        "caution": "Complex topologies with interior geometry may recalculate incorrectly. Inspect thin walls.",
    },
    {
        "step": 6,
        "name": "remove_interior_geometry",
        "description": (
            "Delete faces and volumes that are inside the mesh surface and not visible. "
            "Common in CT scan meshes where bone marrow cavities are fully modeled, "
            "or in photogrammetry of complex architecture where interior rooms are captured "
            "even though only the exterior is needed."
        ),
        "blender_api": "Select interior faces by normal direction or cavity selection; delete with bmesh.ops.delete",
        "when_to_use": "Before 3D print, game export, or any real-time use where interior wastes memory",
        "caution": "Do not remove interior geometry if the asset will be seen from the inside.",
    },
    {
        "step": 7,
        "name": "smooth_noise",
        "description": (
            "Apply Laplacian or HC Laplacian smoothing to reduce scan noise while preserving "
            "sharp feature edges. 1-5 iterations of Laplacian smooth is typical. "
            "Avoid over-smoothing — it rounds sharp edges and shrinks the mesh volume."
        ),
        "blender_api": "bmesh.ops.smooth_laplacian_vert(bm, verts=bm.verts, lambda_factor=0.5, preserve_volume=True)",
        "when_to_use": "Photogrammetry with visible bumpy noise; LiDAR with high-frequency noise",
        "caution": "Over-smoothing causes volume shrinkage and feature loss. Limit to 2-5 iterations.",
    },
    {
        "step": 8,
        "name": "decimation",
        "description": (
            "Reduce polygon count while preserving visual shape. Algorithms: "
            "Quadric Error Metrics (QEM, Garland-Heckbert 1997) — best quality/speed tradeoff. "
            "Used in Blender's Decimate modifier (Collapse mode). "
            "Adaptive variants weight by curvature — flat areas decimated more aggressively."
        ),
        "blender_api": "decimate_modifier = obj.modifiers.new('Decimate', 'DECIMATE'); decimate_modifier.ratio = 0.1",
        "when_to_use": "After all cleanup steps — never decimate first or you lose context for repairs",
        "caution": (
            "QEM decimation does not guarantee manifold output. Check non-manifold edges after. "
            "Sharp features require crease marks before decimation."
        ),
    },
    {
        "step": 9,
        "name": "retopology",
        "description": (
            "Manual or automatic creation of a new low-poly mesh that conforms to the high-poly scan. "
            "Manual: Blender Shrinkwrap + snap workflow, RetopoFlow add-on, or ZBrush ZRemesher. "
            "Automatic: ZRemesher, Instant Meshes (open source), Quadremesher (Blender add-on). "
            "The new topology then has normals baked from the high-poly scan."
        ),
        "blender_api": "Use Shrinkwrap modifier; final bake via bpy.ops.object.bake(type='NORMAL')",
        "when_to_use": "When scan data needs animation rigging or clean subdivision topology",
        "caution": "Time-consuming for complex organic shapes. Automatic retopology rarely matches manual quality.",
    },
]

# ─── Scan Processing Training Pairs ───────────────────────────────────────────

SCAN_PROCESSING_TRAINING_PAIRS: list[dict[str, str]] = [
    {
        "input": "I have a 15 million polygon photogrammetry scan. Make it game-ready.",
        "output": """import bpy
import bmesh

def merge_doubles(obj, dist=0.001):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    before = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=dist)
    bm.to_mesh(obj.data)
    bm.free()
    return before - len(obj.data.vertices)

def fix_normals(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()

def remove_isolated(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    isolated = [v for v in bm.verts if not v.link_edges]
    bmesh.ops.delete(bm, geom=isolated, context='VERTS')
    bm.to_mesh(obj.data)
    bm.free()
    return len(isolated)

def decimate_to_target(obj, target_polys=10_000):
    current = len(obj.data.polygons)
    if current <= target_polys:
        print(f"Already at target: {current} polys")
        return
    ratio = target_polys / current
    dec = obj.modifiers.new("Decimate_GameReady", type='DECIMATE')
    dec.decimate_type = 'COLLAPSE'
    dec.ratio = ratio
    dec.use_collapse_triangulate = True
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier="Decimate_GameReady")
    print(f"Decimated: {current:,} -> {len(obj.data.polygons):,} polygons")

def auto_unwrap(obj):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=66.0, margin_method='SCALED', island_margin=0.02)
    bpy.ops.object.mode_set(mode='OBJECT')

source_obj_name = "Photogrammetry_Scan"
obj = bpy.data.objects.get(source_obj_name)
if obj:
    print("=== Photogrammetry -> Game-Ready Pipeline ===")
    print(f"Source: {len(obj.data.polygons):,} polygons")
    removed_isolated = remove_isolated(obj)
    print(f"Removed {removed_isolated} isolated vertices")
    merged = merge_doubles(obj, dist=0.0005)
    print(f"Merged {merged} duplicate vertices")
    fix_normals(obj)
    print("Normals recalculated")
    decimate_to_target(obj, target_polys=10_000)
    auto_unwrap(obj)
    print("UV unwrap complete")
    print("Next: bake normal map from original 15M scan onto this 10K mesh in Cycles")
else:
    print(f"Object '{source_obj_name}' not found.")""",
    },
    {
        "input": "Clean up the holes in my LiDAR scan of a building",
        "output": """import bpy
import bmesh

def fill_scan_holes(obj_name: str, max_hole_verts: int = 100):
    obj = bpy.data.objects.get(obj_name)
    if obj is None or obj.type != 'MESH':
        print(f"Mesh object '{obj_name}' not found.")
        return

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.edges.ensure_lookup_table()

    boundary_edges = [e for e in bm.edges if e.is_boundary]
    print(f"Found {len(boundary_edges)} boundary edges before repair")

    # Fill holes — limit size to avoid filling intentional openings (doors, windows)
    result = bmesh.ops.holes_fill(bm, edges=boundary_edges, sides=max_hole_verts)
    filled_faces = result.get('faces', [])
    print(f"Filled {len(filled_faces)} hole face(s)")

    # Smooth the filled areas to blend with surrounding geometry
    if filled_faces:
        bmesh.ops.smooth_vert(
            bm,
            verts=list({v for f in filled_faces for v in f.verts}),
            factor=0.5,
            use_axis_x=True, use_axis_y=True, use_axis_z=True,
            mirror_clip_x=False, mirror_clip_y=False, mirror_clip_z=False,
            clip_dist=0.0,
        )

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    obj.data.update()
    bm.free()
    print("Hole fill complete.")
    print("NOTE: Large gaps (windows, doors) must be filled manually to preserve architectural intent.")

fill_scan_holes("LiDAR_Building_Scan", max_hole_verts=50)""",
    },
    {
        "input": "Convert my NeRF point cloud output to a clean mesh",
        "output": """import bpy
import bmesh

def import_and_clean_nerf_mesh(ply_path: str):
    bpy.ops.wm.ply_import(filepath=ply_path)
    obj = bpy.context.active_object
    if obj is None:
        print("Import failed.")
        return

    obj.name = "NeRF_Mesh_Cleaned"
    print(f"Imported: {len(obj.data.polygons):,} polygons")

    bm = bmesh.new()
    bm.from_mesh(obj.data)

    # Remove isolated vertices (NeRF floaters produce disconnected verts)
    isolated = [v for v in bm.verts if not v.link_edges]
    bmesh.ops.delete(bm, geom=isolated, context='VERTS')
    print(f"Removed {len(isolated)} isolated vertices (floaters)")

    # Merge near-duplicate vertices from marching cubes grid boundary
    before = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)
    print(f"Merged {before - len(bm.verts)} near-duplicate vertices")

    # Remove small disconnected components (NeRF background floaters)
    bm.faces.ensure_lookup_table()
    visited: set[int] = set()
    components: list[list] = []
    for start_face in bm.faces:
        if start_face.index in visited:
            continue
        comp: list = []
        stack = [start_face]
        while stack:
            face = stack.pop()
            if face.index in visited:
                continue
            visited.add(face.index)
            comp.append(face)
            for edge in face.edges:
                for lf in edge.link_faces:
                    if lf.index not in visited:
                        stack.append(lf)
        components.append(comp)

    min_component_faces = 500
    small_faces = [f for comp in components if len(comp) < min_component_faces for f in comp]
    if small_faces:
        bmesh.ops.delete(bm, geom=small_faces, context='FACES')
        print(f"Removed {len(small_faces)} small component faces")

    # Laplacian smooth to reduce marching cubes staircase artifacts
    bm.verts.ensure_lookup_table()
    bmesh.ops.smooth_laplacian_vert(
        bm, verts=bm.verts,
        lambda_factor=0.5, lambda_border=0.1,
        use_x=True, use_y=True, use_z=True,
        preserve_volume=True,
    )
    print("Applied Laplacian smoothing (1 pass)")

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    obj.data.update()
    bm.free()

    print(f"Clean mesh: {len(obj.data.polygons):,} polygons")
    print("Recommended next: decimate to target use case, then UV unwrap and texture bake.")

import_and_clean_nerf_mesh("/path/to/nerf_output.ply")""",
    },
    {
        "input": "My photogrammetry mesh has texture stretching. Fix it.",
        "output": """import bpy

def fix_texture_stretching(obj_name: str):
    obj = bpy.data.objects.get(obj_name)
    if obj is None or obj.type != 'MESH':
        print(f"Mesh object '{obj_name}' not found.")
        return

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Smart UV Project — better seam placement than the original photogrammetry UV
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(
        angle_limit=66.0,
        margin_method='SCALED',
        island_margin=0.02,
        area_weight=0.0,
        correct_aspect=True,
        scale_to_bounds=False,
    )
    print("Smart UV Project applied — stretching resolved by new seam placement.")
    bpy.ops.object.mode_set(mode='OBJECT')

    # Prepare bake target for re-projecting the texture
    if len(obj.material_slots) > 0 and obj.material_slots[0].material:
        mat = obj.material_slots[0].material
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        existing_img_node = next((n for n in nodes if n.type == 'TEX_IMAGE'), None)
        if existing_img_node and existing_img_node.image:
            old_img = existing_img_node.image
            new_img = bpy.data.images.new(
                f"{obj_name}_rebaked",
                width=old_img.size[0],
                height=old_img.size[1],
                alpha=True,
            )
            bake_node = nodes.new('ShaderNodeTexImage')
            bake_node.image = new_img
            bake_node.location = (existing_img_node.location.x, existing_img_node.location.y - 300)
            nodes.active = bake_node
            print(f"Bake target created: {new_img.name} ({new_img.size[0]}x{new_img.size[1]})")
            print("In Cycles: Bake Type=Diffuse, uncheck Direct+Indirect, check Color.")

fix_texture_stretching("Photogrammetry_Object")""",
    },
    {
        "input": "Decimate this medical CT scan to 100K polygons without losing surface detail",
        "output": """import bpy
import bmesh

def adaptive_decimate_ct_scan(obj_name: str, target_polys: int = 100_000):
    obj = bpy.data.objects.get(obj_name)
    if obj is None or obj.type != 'MESH':
        print(f"Mesh object '{obj_name}' not found.")
        return

    current_polys = len(obj.data.polygons)
    print(f"Source: {current_polys:,} polygons")

    if current_polys <= target_polys:
        print(f"Already at or below target {target_polys:,}")
        return

    # Step 1: Smooth voxel staircase artifacts before decimation
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    for _ in range(3):
        bmesh.ops.smooth_laplacian_vert(
            bm, verts=bm.verts,
            lambda_factor=0.3, lambda_border=0.1,
            use_x=True, use_y=True, use_z=True,
            preserve_volume=True,
        )
    bm.to_mesh(obj.data)
    bm.free()
    print("Laplacian smoothing (3 passes) applied to remove voxel staircase")

    # Step 2: Planar dissolve first — removes coplanar faces without touching curved regions
    dec_planar = obj.modifiers.new("Decimate_Planar", type='DECIMATE')
    dec_planar.decimate_type = 'DISSOLVE'
    dec_planar.angle_limit = 0.017453  # 1 degree
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier="Decimate_Planar")
    print(f"After planar dissolve: {len(obj.data.polygons):,} polygons")

    # Step 3: QEM collapse to final target
    if len(obj.data.polygons) > target_polys:
        ratio = target_polys / len(obj.data.polygons)
        dec_collapse = obj.modifiers.new("Decimate_QEM", type='DECIMATE')
        dec_collapse.decimate_type = 'COLLAPSE'
        dec_collapse.ratio = ratio
        bpy.ops.object.modifier_apply(modifier="Decimate_QEM")

    final_polys = len(obj.data.polygons)
    print(f"Final: {final_polys:,} polygons (target: {target_polys:,})")

    # Step 4: Recalculate normals after decimation
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    print(f"Reduction: {current_polys:,} -> {final_polys:,} ({100*(1 - final_polys/current_polys):.1f}%)")

adaptive_decimate_ct_scan("CT_Bone_Scan", target_polys=100_000)""",
    },
    {
        "input": "Extract the floor plan from my LiDAR scan of a house",
        "output": """import bpy
import bmesh

def extract_floor_plan(obj_name: str, floor_height: float = 0.5, slice_thickness: float = 0.2):
    obj = bpy.data.objects.get(obj_name)
    if obj is None or obj.type != 'MESH':
        print(f"Mesh object '{obj_name}' not found.")
        return

    mesh = obj.data
    verts_world = [obj.matrix_world @ v.co for v in mesh.vertices]
    z_values = [v.z for v in verts_world]
    min_z = min(z_values)
    slice_z_min = min_z + floor_height
    slice_z_max = slice_z_min + slice_thickness
    print(f"Slicing at Z={slice_z_min:.3f} to {slice_z_max:.3f}")

    bm = bmesh.new()
    bm.from_mesh(mesh)

    floor_verts_xy = []
    for edge in bm.edges:
        v1 = obj.matrix_world @ edge.verts[0].co
        v2 = obj.matrix_world @ edge.verts[1].co
        z1, z2 = v1.z, v2.z
        if (z1 <= slice_z_max and z2 >= slice_z_min) or (z2 <= slice_z_max and z1 >= slice_z_min):
            if abs(z2 - z1) > 1e-6:
                t = (slice_z_min - z1) / (z2 - z1)
                floor_verts_xy.append((v1.x + t * (v2.x - v1.x), v1.y + t * (v2.y - v1.y)))
    bm.free()

    if not floor_verts_xy:
        print("No geometry at this height. Adjust floor_height parameter.")
        return

    outline_mesh = bpy.data.meshes.new("FloorPlan_Outline")
    outline_obj = bpy.data.objects.new("FloorPlan_Outline", outline_mesh)
    bpy.context.scene.collection.objects.link(outline_obj)

    bm2 = bmesh.new()
    for x, y in floor_verts_xy:
        bm2.verts.new((x, y, 0.0))
    bm2.to_mesh(outline_mesh)
    bm2.free()

    print(f"Floor plan outline: {len(floor_verts_xy)} vertices")
    print("Export as DXF: File > Export > AutoCAD DXF (.dxf)")

extract_floor_plan("LiDAR_House_Scan", floor_height=0.3, slice_thickness=0.15)""",
    },
    {
        "input": "Make this scan watertight for 3D printing",
        "output": """import bpy
import bmesh

def make_watertight_for_printing(obj_name: str, wall_thickness_mm: float = 1.2):
    obj = bpy.data.objects.get(obj_name)
    if obj is None or obj.type != 'MESH':
        print(f"Object '{obj_name}' not found.")
        return

    print(f"Before: {len(obj.data.vertices):,} verts, {len(obj.data.polygons):,} polys")
    bm = bmesh.new()
    bm.from_mesh(obj.data)

    # Merge duplicates
    before = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.001)
    print(f"Merged {before - len(bm.verts)} duplicate vertices")

    # Recalculate normals — must all point outward for printing
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    # Fill all holes
    bm.edges.ensure_lookup_table()
    boundary_edges = [e for e in bm.edges if e.is_boundary]
    print(f"Found {len(boundary_edges)} boundary edges")
    if boundary_edges:
        result = bmesh.ops.holes_fill(bm, edges=boundary_edges, sides=0)
        print(f"Filled {len(result.get('faces', []))} holes")

    bm.to_mesh(obj.data)
    obj.data.update()
    bm.free()

    # Solidify to ensure minimum wall thickness
    solidify = obj.modifiers.new("Solidify_Print", type='SOLIDIFY')
    solidify.thickness = wall_thickness_mm / 1000.0  # mm -> meters (Blender units)
    solidify.offset = -1.0   # grow inward
    solidify.use_even_offset = True
    solidify.fill_rim = True
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier="Solidify_Print")
    print(f"Applied {wall_thickness_mm}mm minimum wall thickness")

    # Final manifold check
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    non_manifold = [e for e in bm.edges if not e.is_manifold]
    bm.free()

    if non_manifold:
        print(f"WARNING: {len(non_manifold)} non-manifold edges remain — manual cleanup needed")
    else:
        print("Manifold check: PASSED — mesh is watertight and ready for 3D printing")
    print("Export as STL or 3MF for slicing.")

make_watertight_for_printing("Scan_Object", wall_thickness_mm=1.2)""",
    },
    {
        "input": "Remove the background from this object scan",
        "output": """import bpy
import bmesh

def remove_scan_background(obj_name: str):
    obj = bpy.data.objects.get(obj_name)
    if obj is None or obj.type != 'MESH':
        print(f"Object '{obj_name}' not found.")
        return

    print(f"Processing: {obj_name} ({len(obj.data.polygons):,} polygons)")
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    visited: set[int] = set()
    components: list[list] = []
    for start in bm.faces:
        if start.index in visited:
            continue
        comp: list = []
        stack = [start]
        while stack:
            f = stack.pop()
            if f.index in visited:
                continue
            visited.add(f.index)
            comp.append(f)
            for e in f.edges:
                for lf in e.link_faces:
                    if lf.index not in visited:
                        stack.append(lf)
        components.append(comp)

    if not components:
        bm.free()
        return

    largest = max(components, key=len)
    to_delete = [f for comp in components if comp is not largest for f in comp]
    if to_delete:
        bmesh.ops.delete(bm, geom=to_delete, context='FACES')
        print(f"Removed {len(components) - 1} background components")
    else:
        print("Only one component — no background to remove")

    isolated = [v for v in bm.verts if not v.link_edges]
    if isolated:
        bmesh.ops.delete(bm, geom=isolated, context='VERTS')

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    obj.data.update()
    bm.free()
    print(f"Result: {len(obj.data.polygons):,} polygons")

remove_scan_background("Object_Scan_With_Background")""",
    },
    {
        "input": "Align two scans of the same object taken from different angles",
        "output": """import bpy
import mathutils

def manual_coarse_align(source_name: str, target_name: str):
    source = bpy.data.objects.get(source_name)
    target = bpy.data.objects.get(target_name)
    if source is None or target is None:
        print("Source or target object not found.")
        return

    def bbox_center(obj):
        corners = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
        return sum(corners, mathutils.Vector()) / 8

    source_center = bbox_center(source)
    target_center = bbox_center(target)
    offset = target_center - source_center
    source.location += offset
    print(f"Coarse alignment: translated source by {tuple(round(v, 3) for v in offset)}")

    def bbox_size(obj):
        corners = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
        return (max(c.x for c in corners) - min(c.x for c in corners),
                max(c.y for c in corners) - min(c.y for c in corners),
                max(c.z for c in corners) - min(c.z for c in corners))

    src_size = bbox_size(source)
    tgt_size = bbox_size(target)
    scale_factors = tuple(t / s if s > 1e-6 else 1.0 for s, t in zip(src_size, tgt_size))
    avg_scale = sum(scale_factors) / 3
    source.scale = (source.scale.x * avg_scale,
                    source.scale.y * avg_scale,
                    source.scale.z * avg_scale)
    print(f"Scale correction applied: {avg_scale:.4f}x")

    print("\\nFor fine ICP alignment, use CloudCompare:")
    print("  Tools > Registration > Fine Registration (ICP)")
    print("Or use Open3D Python:")
    print("  import open3d as o3d")
    print("  result = o3d.pipelines.registration.registration_icp(source_pc, target_pc, 0.01)")
    print("  # Apply result.transformation (4x4 matrix) to source in Blender")

manual_coarse_align("Scan_Angle_A", "Scan_Angle_B")""",
    },
    {
        "input": "Generate a LOD chain from this high-res photogrammetry scan",
        "output": """import bpy

LOD_RATIOS = {'LOD0': 1.0, 'LOD1': 0.5, 'LOD2': 0.25, 'LOD3': 0.05, 'LOD4': 0.01}

def generate_scan_lod_chain(source_obj_name: str, hero_target: int = 20_000):
    source = bpy.data.objects.get(source_obj_name)
    if source is None or source.type != 'MESH':
        print(f"Object '{source_obj_name}' not found.")
        return

    original_polys = len(source.data.polygons)
    print(f"Source: {source_obj_name} ({original_polys:,} polygons)")

    lod_col_name = f"{source_obj_name}_LOD_Chain"
    lod_col = bpy.data.collections.get(lod_col_name)
    if lod_col is None:
        lod_col = bpy.data.collections.new(lod_col_name)
        bpy.context.scene.collection.children.link(lod_col)

    for lod_name, ratio in LOD_RATIOS.items():
        new_mesh = source.data.copy()
        new_obj = bpy.data.objects.new(f"{source_obj_name}_{lod_name}", new_mesh)
        lod_col.objects.link(new_obj)
        target_polys = int(hero_target * ratio)
        if target_polys < original_polys:
            dec = new_obj.modifiers.new(name="Decimate", type='DECIMATE')
            dec.decimate_type = 'COLLAPSE'
            dec.ratio = target_polys / original_polys
            dec.use_collapse_triangulate = True
            bpy.context.view_layer.objects.active = new_obj
            bpy.ops.object.modifier_apply(modifier="Decimate")
        result_polys = len(new_obj.data.polygons)
        print(f"  {lod_name}: {result_polys:,} polygons")

    print(f"LOD chain in collection '{lod_col_name}'")
    print("Note: LOD0 from photogrammetry is usually too dense. Manual retopo recommended.")

generate_scan_lod_chain("Photogrammetry_Hero_Object", hero_target=15_000)""",
    },
    {
        "input": "Process my drone photogrammetry of a construction site for a digital twin",
        "output": """import bpy
import bmesh

def process_construction_site_scan(obj_name: str, target_polys: int = 500_000):
    obj = bpy.data.objects.get(obj_name)
    if obj is None or obj.type != 'MESH':
        print(f"Object '{obj_name}' not found.")
        return

    original_polys = len(obj.data.polygons)
    print(f"=== Construction Site Digital Twin Pipeline ===")
    print(f"Source: {original_polys:,} polygons")

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    # Find connected components and remove small debris/vegetation
    visited: set[int] = set()
    components: list[list] = []
    for start in bm.faces:
        if start.index in visited:
            continue
        comp: list = []
        stack = [start]
        while stack:
            f = stack.pop()
            if f.index in visited:
                continue
            visited.add(f.index)
            comp.append(f)
            for e in f.edges:
                for lf in e.link_faces:
                    if lf.index not in visited:
                        stack.append(lf)
        components.append(comp)

    small_faces = [f for comp in components if len(comp) < 1000 for f in comp]
    if small_faces:
        bmesh.ops.delete(bm, geom=small_faces, context='FACES')
        print(f"Removed {len(small_faces)} small component faces (debris/vegetation)")

    isolated = [v for v in bm.verts if not v.link_edges]
    if isolated:
        bmesh.ops.delete(bm, geom=isolated, context='VERTS')

    before = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.005)
    print(f"Merged {before - len(bm.verts)} near-duplicate vertices")

    bm.edges.ensure_lookup_table()
    boundary = [e for e in bm.edges if e.is_boundary]
    if boundary:
        bmesh.ops.holes_fill(bm, edges=boundary, sides=20)

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    obj.data.update()
    bm.free()

    current = len(obj.data.polygons)
    if current > target_polys:
        ratio = target_polys / current
        dec = obj.modifiers.new("Decimate_DigitalTwin", type='DECIMATE')
        dec.decimate_type = 'COLLAPSE'
        dec.ratio = ratio
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier="Decimate_DigitalTwin")

    final = len(obj.data.polygons)
    print(f"Final: {final:,} polygons")
    print(f"Reduction: {original_polys:,} -> {final:,} ({100*(1-final/original_polys):.1f}%)")
    print("Next: Export as GLB for web viewer or USD for BIM integration.")

process_construction_site_scan("Drone_Survey_Mesh", target_polys=500_000)""",
    },
    {
        "input": "Extract a clean mesh from my Gaussian Splatting capture for use in Blender",
        "output": """import bpy
import bmesh

def clean_gaussian_splat_mesh(ply_path: str):
    bpy.ops.wm.ply_import(filepath=ply_path)
    obj = bpy.context.active_object
    if obj is None:
        print("PLY import failed.")
        return

    obj.name = "GaussianSplat_Mesh"
    print(f"Imported: {len(obj.data.polygons):,} polygons")

    bm = bmesh.new()
    bm.from_mesh(obj.data)

    # Remove isolated vertices (Gaussians that became disconnected verts)
    isolated = [v for v in bm.verts if not v.link_edges]
    bmesh.ops.delete(bm, geom=isolated, context='VERTS')
    print(f"Removed {len(isolated)} isolated vertices")

    before = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)
    print(f"Merged {before - len(bm.verts)} near-duplicate vertices")

    # Remove tiny disconnected components (needle Gaussian artifacts)
    bm.faces.ensure_lookup_table()
    visited: set[int] = set()
    components: list[list] = []
    for start in bm.faces:
        if start.index in visited:
            continue
        comp: list = []
        stack = [start]
        while stack:
            f = stack.pop()
            if f.index in visited:
                continue
            visited.add(f.index)
            comp.append(f)
            for e in f.edges:
                for lf in e.link_faces:
                    if lf.index not in visited:
                        stack.append(lf)
        components.append(comp)

    needle_faces = [f for comp in components if len(comp) < 100 for f in comp]
    if needle_faces:
        bmesh.ops.delete(bm, geom=needle_faces, context='FACES')
        print(f"Removed {len(needle_faces)} needle artifact faces")

    bm.edges.ensure_lookup_table()
    boundary = [e for e in bm.edges if e.is_boundary]
    if boundary:
        bmesh.ops.holes_fill(bm, edges=boundary, sides=30)

    bm.verts.ensure_lookup_table()
    bmesh.ops.smooth_laplacian_vert(
        bm, verts=bm.verts,
        lambda_factor=0.5, lambda_border=0.1,
        use_x=True, use_y=True, use_z=True,
        preserve_volume=True,
    )
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    obj.data.update()
    bm.free()

    print(f"Clean mesh: {len(obj.data.polygons):,} polygons")
    print("Next: Decimate, UV unwrap, bake texture from Gaussian Splat renderer output.")

clean_gaussian_splat_mesh("/path/to/sugar_mesh.ply")""",
    },
    {
        "input": "My LiDAR scan has ghost objects from people walking through. Remove them.",
        "output": """import bpy
import bmesh

def remove_dynamic_ghosts(obj_name: str, min_component_size: int = 5000):
    obj = bpy.data.objects.get(obj_name)
    if obj is None or obj.type != 'MESH':
        print(f"Object '{obj_name}' not found.")
        return

    print(f"Removing LiDAR dynamic object ghosts from: {obj_name}")
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    visited: set[int] = set()
    components: list[list] = []
    for start in bm.faces:
        if start.index in visited:
            continue
        comp: list = []
        stack = [start]
        while stack:
            f = stack.pop()
            if f.index in visited:
                continue
            visited.add(f.index)
            comp.append(f)
            for e in f.edges:
                for lf in e.link_faces:
                    if lf.index not in visited:
                        stack.append(lf)
        components.append(comp)

    print(f"Found {len(components)} connected component(s)")
    sorted_comps = sorted(components, key=len, reverse=True)

    ghost_faces = [f for comp in sorted_comps[1:] if len(comp) < min_component_size for f in comp]
    if ghost_faces:
        bmesh.ops.delete(bm, geom=ghost_faces, context='FACES')
        print(f"Removed {len(ghost_faces)} ghost faces")
    else:
        print(f"No ghosts found below threshold {min_component_size}. Try lowering the threshold.")

    isolated = [v for v in bm.verts if not v.link_edges]
    if isolated:
        bmesh.ops.delete(bm, geom=isolated, context='VERTS')

    bm.edges.ensure_lookup_table()
    boundary = [e for e in bm.edges if e.is_boundary]
    if boundary:
        bmesh.ops.holes_fill(bm, edges=boundary, sides=50)

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    obj.data.update()
    bm.free()
    print(f"After: {len(obj.data.polygons):,} polygons")

remove_dynamic_ghosts("LiDAR_Office_Scan", min_component_size=3000)""",
    },
    {
        "input": "Convert a DICOM CT scan of a bone to a 3D mesh for surgical planning",
        "output": """# DICOM -> 3D mesh workflow requires 3D Slicer before Blender post-processing.
#
# --- IN 3D SLICER ---
# 1. File > Add Data > Choose DICOM directory
# 2. Segment Editor > Add segment > Threshold 400-3000 HU (bone range)
# 3. Islands > Keep Largest Island (remove small fragments)
# 4. Smoothing > Gaussian, sigma=0.5mm (reduce voxel staircase)
# 5. Segmentations > Export as STL (coordinate system: RAS)
# Save as: bone_segmentation.stl

import bpy
import bmesh

def process_bone_mesh(stl_path: str, target_polys: int = 200_000):
    bpy.ops.wm.stl_import(filepath=stl_path)
    obj = bpy.context.active_object
    if obj is None:
        print("STL import failed.")
        return

    obj.name = "Bone_Surgical_Model"
    # 3D Slicer exports in RAS coordinates; correct to Blender Z-up
    obj.rotation_euler = (1.5707963, 0, 0)
    bpy.ops.object.transform_apply(rotation=True)
    print(f"Bone mesh imported: {len(obj.data.polygons):,} polygons")

    bm = bmesh.new()
    bm.from_mesh(obj.data)

    # STL stores per-face verts (no shared verts) — merge them
    before = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.01)
    print(f"Merged {before - len(bm.verts)} STL boundary vertices")

    # Remove small fragments (calcifications, bone chips in scan)
    bm.faces.ensure_lookup_table()
    visited: set[int] = set()
    comps: list[list] = []
    for start in bm.faces:
        if start.index in visited:
            continue
        comp: list = []
        stack = [start]
        while stack:
            f = stack.pop()
            if f.index in visited:
                continue
            visited.add(f.index)
            comp.append(f)
            for e in f.edges:
                for lf in e.link_faces:
                    if lf.index not in visited:
                        stack.append(lf)
        comps.append(comp)

    fragments = [f for c in comps if len(c) < 500 for f in c]
    if fragments:
        bmesh.ops.delete(bm, geom=fragments, context='FACES')
        print(f"Removed {len(fragments)} small fragment faces")

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    obj.data.update()
    bm.free()

    current = len(obj.data.polygons)
    if current > target_polys:
        ratio = target_polys / current
        dec = obj.modifiers.new("Decimate", type='DECIMATE')
        dec.decimate_type = 'COLLAPSE'
        dec.ratio = ratio
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier="Decimate")

    print(f"Final: {len(obj.data.polygons):,} polygons")
    print("Export as STL for 3D-printed surgical guide, GLTF for web viewer, OBJ for Mimics.")

process_bone_mesh("/path/to/bone_segmentation.stl", target_polys=150_000)""",
    },
]

# ─── Class ─────────────────────────────────────────────────────────────────────


class ScanProcessor:
    """
    3D scan and photogrammetry processing domain agent for Nalana.

    Generates training data for processing real-world 3D capture workflows
    including photogrammetry cleanup, LiDAR processing, NeRF/GS mesh extraction,
    decimation for various use cases, and medical scan processing.
    """

    def __init__(self, output_dir: str = "data/scan_processing") -> None:
        self.output_dir = BASE_DIR / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scan_types = SCAN_TYPES
        self.decimation_targets = DECIMATION_TARGETS
        self.repair_ops = MESH_REPAIR_OPERATIONS
        self._rng = random.Random(42)

    # ── Training Pair Generation ───────────────────────────────────────────────

    def generate_training_pairs(self, n: int = 150) -> list[dict[str, Any]]:
        """
        Generate n training pairs for scan processing tasks.
        Mixes all curated static pairs with dynamically generated variants.
        """
        pairs: list[dict[str, Any]] = []

        for pair in SCAN_PROCESSING_TRAINING_PAIRS:
            pairs.append({
                "input": pair["input"],
                "output": pair["output"].strip(),
                "domain": "scan_processing",
                "source": "curated",
                "task_type": self._classify_task_type(pair["input"]),
            })

        photogrammetry_scenarios = [
            "outdoor building facade with glass windows",
            "indoor museum artifact under controlled lighting",
            "vehicle exterior from 300 photos",
            "food product for e-commerce 3D viewer",
            "archaeological site with rough terrain",
            "industrial machine component",
            "person face for VFX previz",
            "shoe for product visualization",
            "historical monument exterior",
            "interior room with furniture",
        ]

        lidar_scan_types = ["mobile", "aerial"]
        lidar_tasks = [
            "extract walls and floor for BIM",
            "measure room dimensions accurately",
            "detect structural deformation over time",
            "create as-built documentation",
            "remove vegetation to reveal building footprint",
            "generate a terrain mesh for game development",
        ]

        decimation_use_cases = list(self.decimation_targets.keys())

        while len(pairs) < n:
            roll = self._rng.random()
            if roll < 0.35:
                scenario = self._rng.choice(photogrammetry_scenarios)
                pairs.append(self.generate_photogrammetry_pair(scenario))
            elif roll < 0.55:
                scan_type = self._rng.choice(lidar_scan_types)
                task = self._rng.choice(lidar_tasks)
                pairs.append(self.generate_lidar_pair(scan_type, task))
            elif roll < 0.80:
                source_polys = self._rng.choice([
                    1_000_000, 5_000_000, 15_000_000, 500_000, 2_000_000
                ])
                use_case = self._rng.choice(decimation_use_cases)
                pairs.append(self.generate_decimation_pair(source_polys, use_case))
            else:
                ops = self._rng.sample(
                    [op["name"] for op in self.repair_ops],
                    k=self._rng.randint(2, 5)
                )
                script = self.create_blender_cleanup_script(ops)
                human_ops = ", ".join(o.replace("_", " ") for o in ops)
                pairs.append({
                    "input": f"Clean up this scan: {human_ops}",
                    "output": script,
                    "domain": "scan_processing",
                    "source": "generated",
                    "task_type": "mesh_cleanup",
                    "operations": ops,
                })

        return pairs[:n]

    def generate_photogrammetry_pair(self, scenario: str) -> dict[str, Any]:
        """
        Generate a photogrammetry processing training pair for the given scenario.
        Returns a dict with 'input', 'output' (Blender Python), and metadata.
        """
        input_text = f"Clean up my photogrammetry scan of a {scenario} for production use"

        issues: list[str] = []
        extra_steps: list[str] = []

        if any(k in scenario for k in ["glass", "window", "water", "mirror"]):
            issues.append("reflective surface holes")
            extra_steps.append("fill_holes")

        if any(k in scenario for k in ["outdoor", "terrain", "building", "vehicle", "monument"]):
            issues.append("background floaters")
            extra_steps.append("remove_isolated_vertices")

        if any(k in scenario for k in ["indoor", "artifact", "face", "shoe", "product"]):
            issues.append("scan noise")
            extra_steps.append("smooth_noise")

        # Build ordered step list with deduplication
        base_steps = [
            "remove_isolated_vertices",
            "merge_duplicate_vertices",
            "fill_holes",
            "smooth_noise",
            "recalculate_normals",
            "decimation",
        ]
        all_steps = list(dict.fromkeys(base_steps + extra_steps))[:6]

        output_code = self.create_blender_cleanup_script(all_steps)
        output_code += f"\n\n# Scenario: {scenario}"
        if issues:
            output_code += f"\n# Expected issues: {', '.join(issues)}"

        return {
            "input": input_text,
            "output": output_code,
            "domain": "scan_processing",
            "source": "generated",
            "task_type": "photogrammetry_cleanup",
            "scenario": scenario,
            "expected_issues": issues,
            "operations": all_steps,
        }

    def generate_lidar_pair(self, scan_type: str, task: str) -> dict[str, Any]:
        """
        Generate a LiDAR processing training pair for the given scan type and task.
        Returns a dict with 'input', 'output' (Blender Python), and metadata.
        """
        scan_key = f"lidar_{scan_type}"
        scan_info = self.scan_types.get(scan_key, self.scan_types["lidar_mobile"])
        input_text = f"Process my {scan_type} LiDAR scan to {task}"

        if "bim" in task or "wall" in task or "floor" in task or "room" in task:
            task_code = '''import bpy
import bmesh

def classify_architectural_elements(obj_name: str):
    obj = bpy.data.objects.get(obj_name)
    if obj is None or obj.type != 'MESH':
        return
    import mathutils
    up = mathutils.Vector((0, 0, 1))
    walls, floors, ceilings = [], [], []
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    for face in bm.faces:
        dot = face.normal.dot(up)
        if dot > 0.8:
            floors.append(face.index)
        elif dot < -0.8:
            ceilings.append(face.index)
        else:
            walls.append(face.index)
    bm.free()
    print(f"Floors: {len(floors)} | Walls: {len(walls)} | Ceilings: {len(ceilings)}")

classify_architectural_elements("LiDAR_Scan")'''
        elif "vegetation" in task or "foliage" in task or "tree" in task:
            task_code = '''import bpy
import bmesh

def remove_vegetation_by_height(obj_name: str, canopy_ratio: float = 0.7):
    obj = bpy.data.objects.get(obj_name)
    if obj is None or obj.type != 'MESH':
        return
    zvals = [v.co.z for v in obj.data.vertices]
    max_z, min_z = max(zvals), min(zvals)
    threshold_z = min_z + (max_z - min_z) * canopy_ratio
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    veg = [f for f in bm.faces if all(v.co.z > threshold_z for v in f.verts)]
    bmesh.ops.delete(bm, geom=veg, context='FACES')
    bm.to_mesh(obj.data)
    bm.free()
    print(f"Removed {len(veg)} vegetation faces above Z={threshold_z:.2f}")

remove_vegetation_by_height("Aerial_LiDAR_Scan")'''
        else:
            task_code = self.create_blender_cleanup_script([
                "remove_isolated_vertices",
                "merge_duplicate_vertices",
                "fill_holes",
                "recalculate_normals",
            ])

        formats_str = ", ".join(scan_info.get("formats", []))
        issues_str = "; ".join(scan_info.get("common_issues", [])[:2])
        output_code = (
            f"# LiDAR {scan_type} scan processing: {task}\n"
            f"# Typical formats: {formats_str}\n"
            f"# Common issues: {issues_str}\n\n"
            + task_code
        )

        return {
            "input": input_text,
            "output": output_code,
            "domain": "scan_processing",
            "source": "generated",
            "task_type": "lidar_processing",
            "scan_type": scan_type,
            "task": task,
        }

    def generate_decimation_pair(
        self, source_polys: int, target_use_case: str
    ) -> dict[str, Any]:
        """
        Generate a decimation training pair for a given source polygon count and use case.
        Returns a dict with 'input', 'output' (Blender Python), and metadata.
        """
        target_info = self.decimation_targets.get(
            target_use_case, self.decimation_targets["game_hero_prop"]
        )
        target_min, target_max = target_info["polygon_range"]
        target_mid = (target_min + target_max) // 2
        ratio = max(0.001, min(1.0, target_mid / max(source_polys, 1)))
        use_case_display = target_use_case.replace("_", " ")
        fn_name = f"decimate_for_{target_use_case}"

        input_text = (
            f"Decimate my {source_polys:,}-polygon scan to {use_case_display} quality"
        )

        collapse_triangulate = "False" if "film" in target_use_case else "True"

        output_code = (
            f"import bpy\n\n"
            f"# Decimation: {source_polys:,} -> {use_case_display} ({target_min:,}-{target_max:,} polys)\n"
            f"# {target_info['description']}\n\n"
            f"def {fn_name}(obj_name: str):\n"
            f"    obj = bpy.data.objects.get(obj_name)\n"
            f"    if obj is None or obj.type != 'MESH':\n"
            f"        print(f\"Object '{{obj_name}}' not found.\")\n"
            f"        return\n"
            f"\n"
            f"    source_polys = len(obj.data.polygons)\n"
            f"    target_polys = {target_mid}\n"
            f"    print(f'Decimating: {{source_polys:,}} -> {{target_polys:,}} polygons')\n"
            f"\n"
            f"    if source_polys <= target_polys:\n"
            f"        print('Already at or below target.')\n"
            f"        return\n"
            f"\n"
            f"    ratio = target_polys / source_polys\n"
            f"    dec = obj.modifiers.new('{fn_name}', type='DECIMATE')\n"
            f"    dec.decimate_type = 'COLLAPSE'\n"
            f"    dec.ratio = ratio\n"
            f"    dec.use_collapse_triangulate = {collapse_triangulate}\n"
            f"    bpy.context.view_layer.objects.active = obj\n"
            f"    bpy.ops.object.modifier_apply(modifier='{fn_name}')\n"
            f"\n"
            f"    final = len(obj.data.polygons)\n"
            f"    print(f'Result: {{final:,}} polygons')\n"
            "    print('Texture budget: " + target_info["texture_budget"] + "')\n"
            "    print('LOD chain needed: " + str(target_info["lod_chain"]) + "')\n"
            f"\n"
            f"    import bmesh\n"
            f"    bm = bmesh.new()\n"
            f"    bm.from_mesh(obj.data)\n"
            f"    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)\n"
            f"    bm.to_mesh(obj.data)\n"
            f"    bm.free()\n"
            f"\n"
            f"{fn_name}('Scan_Object')"
        )

        return {
            "input": input_text,
            "output": output_code,
            "domain": "scan_processing",
            "source": "generated",
            "task_type": "decimation",
            "source_poly_count": source_polys,
            "target_use_case": target_use_case,
            "target_poly_range": target_info["polygon_range"],
            "target_mid_polys": target_mid,
        }

    def create_blender_cleanup_script(self, operations: list[str]) -> str:
        """
        Generate a complete Blender Python script that performs the specified mesh repair
        operations in the given order. Each operation name is matched against
        MESH_REPAIR_OPERATIONS.

        Supported operation names:
          fill_holes, remove_isolated_vertices, fix_non_manifold_edges,
          merge_duplicate_vertices, recalculate_normals, remove_interior_geometry,
          smooth_noise, decimation, retopology
        """
        op_lookup = {op["name"]: op for op in self.repair_ops}

        lines: list[str] = [
            "import bpy",
            "import bmesh",
            "",
            "",
            "def run_scan_cleanup(obj_name: str):",
            '    """Auto-generated scan cleanup pipeline."""',
            "    obj = bpy.data.objects.get(obj_name)",
            "    if obj is None or obj.type != 'MESH':",
            "        print(f\"Object '{obj_name}' not found.\")",
            "        return",
            "",
            "    print(f'=== Scan Cleanup: {obj_name} ===')",
            "    print(f'Before: {len(obj.data.polygons):,} polygons, {len(obj.data.vertices):,} vertices')",
            "    bm = bmesh.new()",
            "    bm.from_mesh(obj.data)",
            "",
        ]

        for op_name in operations:
            op = op_lookup.get(op_name)
            if op:
                desc = op["description"].split(".")[0]
            else:
                desc = op_name.replace("_", " ").capitalize()
            lines.append(f"    # --- {desc} ---")

            if op_name == "remove_isolated_vertices":
                lines += [
                    "    isolated = [v for v in bm.verts if not v.link_edges]",
                    "    bmesh.ops.delete(bm, geom=isolated, context='VERTS')",
                    "    print(f'  Removed {len(isolated)} isolated vertices')",
                    "",
                ]
            elif op_name == "merge_duplicate_vertices":
                lines += [
                    "    _before = len(bm.verts)",
                    "    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.001)",
                    "    print(f'  Merged {_before - len(bm.verts)} duplicate vertices')",
                    "",
                ]
            elif op_name == "fill_holes":
                lines += [
                    "    bm.edges.ensure_lookup_table()",
                    "    _boundary = [e for e in bm.edges if e.is_boundary]",
                    "    if _boundary:",
                    "        _result = bmesh.ops.holes_fill(bm, edges=_boundary, sides=0)",
                    "        print(f'  Filled {len(_result.get(\"faces\", []))} holes')",
                    "    else:",
                    "        print('  No holes found')",
                    "",
                ]
            elif op_name == "fix_non_manifold_edges":
                lines += [
                    "    bm.edges.ensure_lookup_table()",
                    "    _nm = [e for e in bm.edges if not e.is_manifold and not e.is_boundary]",
                    "    if _nm:",
                    "        bmesh.ops.dissolve_edges(bm, edges=_nm, use_verts=True, use_face_split=False)",
                    "        print(f'  Dissolved {len(_nm)} non-manifold edges')",
                    "    else:",
                    "        print('  No non-manifold edges found')",
                    "",
                ]
            elif op_name == "recalculate_normals":
                lines += [
                    "    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)",
                    "    print('  Normals recalculated')",
                    "",
                ]
            elif op_name == "remove_interior_geometry":
                lines += [
                    "    import statistics as _stats",
                    "    _z_vals = [v.co.z for v in bm.verts]",
                    "    _median_z = _stats.median(_z_vals) if _z_vals else 0",
                    "    _interior = [f for f in bm.faces if all(v.co.z < _median_z for v in f.verts)]",
                    "    if _interior:",
                    "        bmesh.ops.delete(bm, geom=_interior, context='FACES')",
                    "        print(f'  Removed {len(_interior)} potential interior faces')",
                    "    else:",
                    "        print('  No interior geometry detected')",
                    "",
                ]
            elif op_name == "smooth_noise":
                lines += [
                    "    bm.verts.ensure_lookup_table()",
                    "    bmesh.ops.smooth_laplacian_vert(",
                    "        bm, verts=bm.verts,",
                    "        lambda_factor=0.5, lambda_border=0.1,",
                    "        use_x=True, use_y=True, use_z=True,",
                    "        preserve_volume=True,",
                    "    )",
                    "    print('  Laplacian smoothing applied (1 pass)')",
                    "",
                ]
            elif op_name == "decimation":
                lines += [
                    "    bm.to_mesh(obj.data)",
                    "    bm.free()",
                    "    _current = len(obj.data.polygons)",
                    "    _target = max(1, _current // 10)  # Default: 10% of current",
                    "    _dec = obj.modifiers.new('Decimate_Auto', type='DECIMATE')",
                    "    _dec.decimate_type = 'COLLAPSE'",
                    "    _dec.ratio = _target / _current",
                    "    _dec.use_collapse_triangulate = True",
                    "    bpy.context.view_layer.objects.active = obj",
                    "    bpy.ops.object.modifier_apply(modifier='Decimate_Auto')",
                    "    print(f'  Decimated: {_current:,} -> {len(obj.data.polygons):,} polygons')",
                    "    bm = bmesh.new()",
                    "    bm.from_mesh(obj.data)",
                    "",
                ]
            elif op_name == "retopology":
                lines += [
                    "    bm.to_mesh(obj.data)",
                    "    bm.free()",
                    "    _remesh = obj.modifiers.new('Remesh_Voxel', type='REMESH')",
                    "    _remesh.mode = 'VOXEL'",
                    "    _remesh.voxel_size = 0.005  # 5mm — adjust to detail level",
                    "    bpy.context.view_layer.objects.active = obj",
                    "    bpy.ops.object.modifier_apply(modifier='Remesh_Voxel')",
                    "    print(f'  Voxel remesh: {len(obj.data.polygons):,} polygons')",
                    "    bm = bmesh.new()",
                    "    bm.from_mesh(obj.data)",
                    "",
                ]
            else:
                lines += [
                    f"    # {op_name}: implement custom logic here",
                    "    pass",
                    "",
                ]

        lines += [
            "    bm.to_mesh(obj.data)",
            "    obj.data.update()",
            "    bm.free()",
            "    print(f'After: {len(obj.data.polygons):,} polygons, {len(obj.data.vertices):,} vertices')",
            "    print('Cleanup complete.')",
            "",
            "",
            'run_scan_cleanup("Scan_Object")',
        ]

        return "\n".join(lines)

    @staticmethod
    def _classify_task_type(query: str) -> str:
        """Classify a scan processing query into a task type string."""
        q = query.lower()
        if any(k in q for k in ["photogrammetry", "photo", "sfm", "meshroom", "realitycapture", "agisoft"]):
            return "photogrammetry_cleanup"
        if any(k in q for k in ["lidar", "point cloud", "las", "laz", "e57"]):
            return "lidar_processing"
        if any(k in q for k in ["nerf", "neural radiance"]):
            return "nerf_to_mesh"
        if any(k in q for k in ["gaussian", "splat", "3dgs", "sugar"]):
            return "gaussian_splat_processing"
        if any(k in q for k in ["ct scan", "dicom", "mri", "medical", "bone", "surgical"]):
            return "medical_scan"
        if any(k in q for k in ["decimate", "reduce", "polygon count", "lod", "optimize", "optimize"]):
            return "decimation"
        if any(k in q for k in ["watertight", "print", "3d print", "manifold", "printing"]):
            return "print_preparation"
        if any(k in q for k in ["floor plan", "bim", "architectural", "wall", "room", "building"]):
            return "scan_to_bim"
        if any(k in q for k in ["align", "register", "icp", "two scans", "different angle"]):
            return "scan_registration"
        return "mesh_cleanup"


# ─── Entry Point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nalana 3D Scan Processing Agent — Training Data Generator"
    )
    parser.add_argument(
        "--generate-pairs",
        action="store_true",
        help="Generate training pairs and save to data/scan_processing/",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=150,
        help="Number of training pairs to generate (default: 150)",
    )
    parser.add_argument(
        "--list-scan-types",
        action="store_true",
        help="Print all supported scan types and their characteristics",
    )
    parser.add_argument(
        "--list-targets",
        action="store_true",
        help="Print all decimation targets with polygon budgets",
    )
    parser.add_argument(
        "--demo-cleanup-script",
        action="store_true",
        help="Print a demo cleanup script with all repair operations",
    )
    args = parser.parse_args()

    processor = ScanProcessor()

    if args.generate_pairs:
        print(f"Generating {args.n_pairs} scan processing training pairs...")
        pairs = processor.generate_training_pairs(n=args.n_pairs)
        with open(PAIRS_OUTPUT, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        print(f"Saved {len(pairs)} pairs to: {PAIRS_OUTPUT}")
        json_out = SCAN_DATA_DIR / "scan_processing_pairs_readable.json"
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)
        print(f"Formatted version: {json_out}")

    if args.list_scan_types:
        print("\nSupported Scan Types:")
        for name, info in SCAN_TYPES.items():
            print(f"\n  {name.upper()}")
            print(f"    {info['description'][:120]}...")
            print(f"    Formats: {', '.join(info.get('formats', []))}")

    if args.list_targets:
        print("\nDecimation Targets:")
        for name, info in DECIMATION_TARGETS.items():
            lo, hi = info["polygon_range"]
            print(f"  {name:<22} {lo:>8,} - {hi:>10,} polys   {info['description'][:55]}")

    if args.demo_cleanup_script:
        all_ops = [op["name"] for op in MESH_REPAIR_OPERATIONS]
        script = processor.create_blender_cleanup_script(all_ops)
        print(script)

    if not any([args.generate_pairs, args.list_scan_types, args.list_targets, args.demo_cleanup_script]):
        parser.print_help()


if __name__ == "__main__":
    main()
