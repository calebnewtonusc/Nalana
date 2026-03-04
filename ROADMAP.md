# Nalana Roadmap

**"You speak. Nalana builds."**

This roadmap covers the full trajectory of Nalana from v1 (current: voice-to-workflow execution) through v3 (autonomous 3D studio intelligence). Each phase builds on the previous. The goal is to be the operating system of professional 3D creation.

---

## Phase 1 — v1: GENERATE (Current Build)

**Status:** In training. Target release: Q2 2026.

**What it does:** Any voice or text command maps to executable workflows in any supported 3D software. The model reasons about operations, their physical validity, and topological consequences before generating code.

**Scope:**
- Modeling, materials, lighting, physics simulation, rigging, basic animation
- Single operations ("bevel these edges") and full multi-step builds ("create a brutalist apartment block")
- Multi-turn creative dialogue ("actually, make it more weathered")
- Cross-software: Blender, Maya, Cinema 4D, Houdini, Rhino, Unreal Engine, Unity, Substance Painter, Web

**Key capabilities shipping in v1:**
- [ ] Universal 3D DSL with software compilers for 9 platforms
- [ ] Expert synthesis pipeline (10,000+ hours of tutorial content)
- [ ] 3-stage training: SFT → Execution-Verified RL → DPO
- [ ] NalanaBench v1.0 (500 prompts, 8 categories) — published as community standard
- [ ] Blender plugin (N-panel + voice hotkey)
- [ ] REST API + WebSocket streaming
- [ ] Docker one-command deploy

**Target metrics (v1):**
| Metric | Target |
|---|---|
| NalanaBench overall | >85 |
| Execution success rate | >85% |
| Voice command accuracy | >80% |
| API latency p50 | <500ms |
| Supported software | 9 |
| Supported file formats | 40+ |

---

## Phase 2 — v1.5: MAKE PRODUCTION-READY (Q3 2026)

**What it adds:** Takes any asset from artistic to shippable. The model gains the full production pipeline.

### Retopology Intelligence

Nalana-v1.5 can retopologize any mesh on command. Input: dense sculpt or scan. Output: clean quad-dominant mesh with proper edge flow, correct pole placement, subdivision-friendly topology. The artist specifies target face count and priority (game-ready vs. film-ready vs. 3D print).

- Target: 80% of AI-generated retopo within 10% of target face count
- NalanaBench-Retopo: 100 meshes tested against professional artist baseline

### UV Intelligence

Smart UV unwrapping across all workflows. Nalana understands seam placement strategy — minimizing distortion in high-visibility areas, maintaining texel density consistency, UDIM layout for characters and environments.

- Seam placement: learned from 10,000+ professional UV maps in training data
- Integration with Substance Painter UDIM export pipeline
- NalanaBench-UV: measured by avg stretch and texel density variance

### Normal Map Baking

High-to-low-poly bake automation. Nalana handles cage setup, ray distance calculation, and artifact identification. Outputs bake configurations ready for Marmoset, Substance, or Blender.

- NalanaBench-Bake: PSNR vs. ground truth normal maps target >30dB

### LOD Chain Generation

One command: "generate LOD chain at 50%, 25%, 10% poly count." Nalana manages decimation strategy, preserves silhouette, and outputs LOD0-LOD3 naming conventions for each target engine.

### Collision Mesh Generation

Voice command → convex hull, box, capsule, or hand-crafted convex decomposition collision mesh. Understands game engine collision pipeline requirements (Unreal, Unity, Godot).

### v1.5 Domain Expansion

| Domain | Capability Added |
|---|---|
| Game Development | LOD chains, collision meshes, draw call optimization |
| VFX / Film | Production retopo, bake-ready UV, UDIM pipeline |
| 3D Printing | Wall thickness validation, printability analysis, support optimization |
| Architecture | Revit/ArchiCAD IFC round-trip, LOD 100-400 mesh generation |

---

## Phase 3 — v2: ANALYZE & CRITIQUE (Q4 2026)

**What it adds:** Nalana becomes a technical reviewer, compliance checker, and quality auditor — not just an executor.

### Architecture Mode (ARCH_ANALYZE)

Nalana reads floorplans and spatial layouts and performs:
- IBC (International Building Code) compliance checks
- ADA accessibility verification (door widths, turning radii, ramp grades)
- Daylight factor computation via simplified Radiosity (IES standard)
- Egress path analysis — minimum exit widths, travel distance to exit
- Structural logic review — load-bearing wall identification, span ratio validation

**Input:** Blender model, Revit IFC export, or DXF drawing
**Output:** Compliance report + annotated Blender scene + remediation code

- NalanaBench-Arch: 30 floorplan scenarios, >90% IBC compliance accuracy

### CAD/Engineering Mode (CAD_OPTIMIZE)

- Design For Manufacturability (DFM) analysis: draft angle violations, undercuts, thin wall detection
- Topology optimization: identify material that can be removed while maintaining structural requirements
- Material selection guidance from mechanical properties (tensile strength, thermal conductivity, weight)
- Integration with FreeCAD and Fusion 360 API for engineering round-trips

- NalanaBench-CAD: 40 CAD parts, DFM issue detection recall >85%

### Scene QA Mode (QA_LINT)

Audits any 3D scene before delivery:
- **Naming**: enforces studio naming conventions (user-definable)
- **Transforms**: flags unapplied transforms that will break engine import
- **UVs**: detects overlapping, flipped, and zero-area UV islands
- **Topology**: identifies N-gons, poles in deformation zones, non-manifold geometry
- **Materials**: detects missing/default materials, texture resolution inconsistencies
- **Scene scale**: flags objects at non-real-world scales

- NalanaBench-QA: 60 scenes with planted errors, F1 score >0.90

### v2 Verticals

| Vertical | Added Capability |
|---|---|
| Architecture / BIM | Code compliance, daylight analysis, IFC export |
| Engineering / Manufacturing | DFM, topology optimization, materials selection |
| Game Development | Scene QA, naming conventions, LOD validation |
| VFX / Film | Pre-delivery scene audit, naming standards, UV validation |
| 3D Printing | Printability analysis, wall thickness, slice preview |

---

## Phase 4 — v3: SHIP (2027)

**What it adds:** Nalana integrates into the full studio pipeline at scale.

### Asset Intelligence at Scale

- Search and retrieve across 10,000+ asset libraries by semantic description ("find me a weathered concrete wall with realistic wear")
- Deduplication: identify near-identical assets across formats and variations
- Batch re-texturing: "re-texture all assets in this folder using the new brand palette"
- Automatic LOD and collision generation for entire library batches
- NalanaBench-Assets: asset dedup precision >95%, batch re-texture quality score

### Digital Twin Mode

- Ingest Matterport scans, LiDAR point clouds, NeRF captures
- Extract clean editable meshes from scans with intelligent hole-filling
- Maintain bidirectional sync: physical sensor updates → 3D model updates
- Applications: building management, factory floor planning, retail visualization
- Integration: Matterport API, ARKit, Azure Spatial Anchors

### Scan Processing Pipeline (SCAN_PROCESS)

- NeRF → clean mesh conversion (instant-ngp, nerfstudio output cleanup)
- Photogrammetry cleanup (RealityCapture, Metashape output optimization)
- LiDAR segmentation and object extraction
- Output: production-ready asset with UVs, materials, LODs

### Multi-Agent Studio Orchestration

Nalana v3 can spawn and coordinate multiple specialized instances:
- Modeling agent: generates base geometry
- Materials agent: creates physically-accurate surface shaders
- Rigging agent: auto-rigs characters to Mixamo/UE5 skeleton standards
- Lighting agent: sets up environment and key lighting
- QA agent: audits the result before delivery

One voice command: "Create a game-ready orc warrior character" → multi-agent pipeline runs in parallel → delivers a complete, shippable asset.

---

## Domain Coverage Roadmap

| Domain | v1 | v1.5 | v2 | v3 |
|---|---|---|---|---|
| Blender (all tools) | Full | Full | Full | Full |
| Maya (film/TV) | 78% | 90% | Full | Full |
| Houdini (VFX/sim) | 71% | 85% | 90% | Full |
| Cinema 4D (motion graphics) | 68% | 80% | 90% | Full |
| Rhino / Grasshopper | 55% | 70% | 85% | Full |
| Unreal Engine 5 | 60% | 75% | 85% | Full |
| Unity | 50% | 65% | 80% | Full |
| ZBrush (sculpting) | Conceptual | Indirect | Blender bridge | Direct |
| Substance Painter/Designer | 65% | 80% | Full | Full |
| Fusion 360 / FreeCAD | — | 40% | 70% | Full |
| Revit / ArchiCAD | — | 25% | 70% | Full |
| SketchUp | — | 30% | 60% | Full |
| Marvelous Designer | — | 25% | 50% | Full |
| Plasticity / CAD tools | — | 20% | 60% | Full |

---

## Knowledge Domain Training Roadmap

The model's capabilities are bounded by what its weights contain. This is the full knowledge domain expansion plan:

### v1 (Current)
- Blender operations (all modules)
- Basic Maya, Houdini, C4D, Rhino operations
- PBR material physics (Fresnel, IOR, SSS, microfacet)
- Topology principles (edge flow, poles, subdivision)
- Design theory (form language, proportion, composition)
- Lighting theory (three-point, HDRI, product, cinematic)
- Rigid body, cloth, fluid, smoke/fire simulation

### v1.5 Additions
- Retopology techniques (character, hard surface, environments)
- UV theory (seam placement, texel density, UDIM)
- Normal baking methodology (cage, ray distance, bias)
- LOD strategy (game engine, film, realtime)
- 3D printing constraints (FDM, SLA, SLS parameters)
- Rigging fundamentals (joint hierarchy, weight painting, IK/FK)

### v2 Additions
- Building codes: IBC, ADA, Eurocodes, ASCE 7 (wind/seismic)
- Structural engineering: span calculations, load paths, material stress
- DFM principles: injection molding, CNC, sheet metal, die casting
- Topology optimization theory (SIMP method, density-based)
- Materials engineering: metals, polymers, composites, ceramics
- Daylight simulation (climate-based daylight modeling, IES standards)

### v3 Additions
- Digital twin synchronization protocols
- NeRF and Gaussian splatting reconstruction techniques
- Photogrammetry best practices (camera placement, lighting for capture)
- Multi-agent coordination protocols
- Asset library management at scale
- Full game engine integration (shaders, Blueprint, C# patterns)
- XR spatial computing (ARKit, Quest, HoloLens anchor systems)

---

## NalanaBench Expansion Roadmap

| Version | Prompts | New Categories |
|---|---|---|
| v1.0 (current) | 500 | BASIC_OPS, OBJECT_BUILD, MATERIAL, SIMULATION, LIGHTING, TOPOLOGY, MULTI_STEP, REASONING |
| v1.5 | 800 | + RETOPO, UV_UNWRAP, BAKE, LOD, COLLISION, 3D_PRINT |
| v2.0 | 1,200 | + ARCH_ANALYZE, CAD_OPTIMIZE, QA_LINT, COMPLIANCE |
| v3.0 | 2,000 | + SCAN_PROCESS, ASSET_MANAGE, MULTI_AGENT, ANIMATION |

NalanaBench is published and maintained separately from the model to ensure it functions as a neutral industry standard. All scores from Nalana, GPT-4o, and community models are tracked on a public leaderboard at `nalana.ai/benchmark`.

---

## Company Integration Roadmap

| Integration | v1 | v1.5 | v2 | v3 |
|---|---|---|---|---|
| Spline (3D web) | One-way DSL export | Bidirectional | Full sync | Full |
| Matterport (digital twins) | Scan import | Extract + clean | Digital twin mode | Full |
| Topological (DFM/sim) | — | Sim feedback | Full CAD mode | Full |
| Dream3D (generative) | Mesh import | Post-process | Full pipeline | Full |
| Adam (parametric design) | — | Design → Nalana handoff | Full integration | Full |
| One Robot (physical AI) | — | Collision mesh export | Sensor sync | Digital twin |
| Sketchfab (asset marketplace) | Import only | Import + tag | Full search | Full |
| Polyhaven (CC0 assets) | Import | Smart import | Contextual suggest | Auto-use |
| Adobe Substance 3D | Material only | Full PBR | Library integration | Full |

---

## Research Publications Plan

| Paper | Target Venue | Expected |
|---|---|---|
| Nalana v1 — System Paper | SIGGRAPH 2026 Emerging Tech | Jun 2026 |
| NalanaBench — Benchmark Paper | ICCV 2026 / NeurIPS 2026 | Sep 2026 |
| Execution-Verified RL for 3D | NeurIPS 2026 | Dec 2026 |
| Universal 3D DSL Design | ACM TOG 2026/2027 | 2027 |
| Physics-Grounded 3D Materials | CVPR 2027 | Jun 2027 |

---

*This roadmap reflects current planning. Priorities shift based on user feedback, benchmark results, and new research directions. The core thesis — that 3D workflow intelligence requires expert reasoning, verifiable execution, and cross-software generalization — does not change.*
