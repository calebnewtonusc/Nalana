# Nalana Training Data Sources

Complete catalog of every data source type the pipeline uses, organized by training stream.
For the automated source hunter, see `data_discovery.py`.

---

## Stream 1 — YouTube Tutorial Transcripts

**What it is**: Professional 3D software tutorial content extracted as timestamped transcript chunks, then synthesized into `(voice_command, scene_context, reasoning, executable_code)` training pairs.

**Why it's the primary signal**: Tutorial creators narrate their reasoning while working — "I'm adding support loops here because subdivision needs geometry to hold the sharpness." This tacit expert knowledge is the core training signal for Nalana's reasoning capability.

**Collection method**: `discovery/discover_v2.py` + `discovery/fetch_bulk.py`

### YouTube Channels by Software

#### Blender (core — ~65% of Stream 1)
| Channel | Content Focus | Priority |
|---------|--------------|----------|
| Blender Guru | Fundamentals, photorealism, architecture | High |
| Grant Abbitt | Stylized characters, fundamentals | High |
| CGFastTrack | Speed modeling, hard surface | High |
| Default Cube | Geometry nodes, procedural | High |
| Ducky 3D | Motion, shaders, stylized | High |
| CG Geek | VFX, environments, sci-fi | High |
| Josh Gambrell | Hard surface, vehicles, weapons | High |
| Polygon Runway | Motion graphics, stylized | High |
| YanSculpts | Digital sculpting, character | High |
| SouthernShotty 3D | Hard surface, product design | High |
| Derek Elliott | Materials, lighting, composition | Medium |
| Remington Creative | Architecture, arch viz | Medium |

#### Maya
| Channel | Content Focus | Priority |
|---------|--------------|----------|
| Autodesk Maya (official) | All workflows, official docs | High |
| Alex Cheparev | Character rigging, animation | High |
| CGCircuit | VFX, simulations, production | High |
| Arrimus 3D | Hard surface, game assets | High |

#### Houdini
| Channel | Content Focus | Priority |
|---------|--------------|----------|
| SideFX (official) | All Houdini workflows | High |
| Tim van Helsdingen | FX, pyro, simulations | High |
| Houdini Tutorial | Node-based workflows | High |
| Indie-Pixel | Procedural modeling | Medium |

#### Cinema 4D
| Channel | Content Focus | Priority |
|---------|--------------|----------|
| Maxon (official) | C4D all workflows | High |
| School of Motion | Motion graphics, MoGraph | High |
| Greyscalegorilla | Lighting, materials, production | High |

#### Other Software
| Channel | Software | Priority |
|---------|---------|----------|
| Adobe Substance 3D | Substance | High |
| McNeel | Rhino / Grasshopper | High |
| Parametric Architecture | Rhino / Grasshopper | Medium |
| Unreal Engine (official) | UE5 | High |
| William Faucher | UE5 lighting, film | High |

#### Cross-discipline / Industry
| Channel | Focus | Priority |
|---------|-------|----------|
| Gnomon Workshop | Professional film/game workflows | High |
| FlippedNormals | Topology, game art, multi-software | High |
| Corridor Crew | VFX breakdowns, film technique | Medium |
| GDC | Game developer conference talks | Medium |
| ACM SIGGRAPH | Academic rendering, novel techniques | Medium |

---

## Stream 2 — 3D Geometry Datasets

**What it is**: 3D object files (GLB, OBJ, USDZ, FBX) processed through VLM-based form analysis to produce `(object, build_sequence)` training pairs. Provides object vocabulary, construction awareness, and shape priors.

**Collection method**: `discovery/objaverse_prep.py`, `discovery/api_harvest.py`, `render/render_pipeline.py`

### Geometry Datasets

| Dataset | Size | License | Content | Priority |
|---------|------|---------|---------|----------|
| Objaverse | ~800k objects | CC-BY 4.0 | General 3D objects, diverse | **High** |
| Objaverse-XL | ~10M objects | CC-BY 4.0 | Extended Objaverse | High |
| ShapeNet (Core) | ~55k models | Custom (research) | 55 semantic categories | **High** |
| ABC Dataset | ~1M models | MIT | CAD models with exact geometry | **High** |
| ABO (Amazon Berkeley) | ~8k objects | CC BY-NC 4.0 | Consumer products, multi-view | High |
| 3D-FUTURE | ~16k objects | Custom | Furniture with CAD geometry | Medium |
| Thingi10K | ~10k models | CC-BY 4.0 | Real 3D printing designs | Medium |
| ScanNet | ~1.5k scenes | Custom (academic) | Indoor RGB-D scans | Medium |
| NASA 3D Resources | ~300 models | Public domain | Spacecraft, planets | Medium |
| NIH 3D Print Exchange | ~3k models | CC / Public | Medical, anatomical | Medium |
| Smithsonian 3D | ~300 objects | CC0 | Museum artifacts, high quality | Medium |
| Printables | Millions | CC | Community 3D printing | Low |

### Model APIs (real-time access)

| API | Content | Rate Limits |
|-----|---------|------------|
| Sketchfab Data API v3 | CC-licensed 3D, all categories | 100 req/min (free) |
| Polyhaven API | CC0 HDRIs, textures, meshes | Unlimited |
| Thingiverse API | 3D printing community | OAuth required |
| GrabCAD Library | Engineering CAD | Login required |

---

## Stream 3 — Physics and Optics Knowledge Base

**What it is**: Q&A pairs synthesized from physically based rendering literature, optics references, and simulation theory. Provides physical grounding for material parameter choices and simulation configuration.

**Collection method**: `integrations/collect_design_physics.py`, `knowledge/physics_kb.py`

### Rendering & Optics

| Source | Content | Access |
|--------|---------|--------|
| pbr-book.org | Physically Based Rendering (Pharr, Jakob, Humphreys) | Free online |
| MERL BRDF Database | Measured BRDFs for 100 real materials | Free download |
| refractiveindex.info | IOR database — glass, crystals, metals, polymers | Free API |
| Cook-Torrance 1982 | Microfacet reflectance model | ACM DL |
| GGX / Walter 2007 | Improved microfacet distribution | PDF available |
| Donner & Jensen 2005 | Measured human skin SSS data | PDF available |
| ShaderToy | Community GLSL procedural shaders | Free scrape |

### Physics Simulation

| Source | Content | Access |
|--------|---------|--------|
| OpenFOAM Documentation | CFD solver theory + tutorials | Free |
| Navier-Stokes refs | Fluid dynamics for smoke/fire | Academic |
| Bullet Physics docs | Rigid body simulation | Free |
| MuJoCo docs | Contact dynamics, cloth | Free |

### Design Theory

| Source | Content | Access |
|--------|---------|--------|
| ArchDaily | Architecture visualization references | Free scrape |
| IFC Open BIM Models | Building information models | Free download |
| CyArk | Photogrammetry of heritage sites | Free (registered) |
| Feynman Lectures (EM) | Electromagnetism, light-matter interaction | Free online |

---

## Stream 4 — Multi-Turn Conversation Sequences

**What it is**: Simulated expert-student dialogues covering topology debugging, render artifact explanation, and scene optimization. Provides dialogue coherence for interactive use.

**Collection method**: `synthesis/multi_turn.py`, community forum scraping

### Community Sources

| Source | Content | Priority |
|--------|---------|----------|
| BlenderArtists.org | Q&A threads with expert replies, critique threads | **High** |
| Polycount Forum | Game art feedback, topology critique, industry tips | **High** |
| SideFX Forum | Houdini Q&A, simulation debugging | High |
| r/blender | Beginner → expert range, common errors | Medium |
| r/3Dmodeling | Cross-software, workflow questions | Medium |
| CGSociety | Professional VFX community | Medium |
| ArtStation | Process posts with step-by-step breakdowns | Low |

**Scraping policy**: Only public posts. Respect robots.txt. No user account data collected. Content is used to train model to answer questions in the style of expert replies, not to reproduce specific user content.

---

## Stream 5 — Cross-Software Integration Pairs

**What it is**: Training pairs for cross-software workflows, file format bridges, and integration pipelines. Covers how the same operation is expressed in different software APIs.

**Collection method**: `integrations/` scripts, software documentation

### Integration Targets

| Integration | Description | Priority |
|------------|-------------|----------|
| Blender ↔ Substance Painter | PBR texture roundtrip | High |
| Blender ↔ Unreal Engine | Asset pipeline, nanite prep | High |
| Maya ↔ Houdini | FX pipeline (Maya rig + Houdini sims) | High |
| Rhino → Unreal | Architecture viz pipeline | Medium |
| Any ↔ Three.js | Web export via glTF | Medium |
| Matterport scans → Blender | 3D scan import and cleanup | Medium |
| Spline → Three.js | Web 3D workflow | Medium |

### File Format References

| Format | Spec | Use |
|--------|------|-----|
| glTF 2.0 | Khronos Group | Primary web + cross-DCC transfer |
| USD / USDA / USDZ | Pixar / Apple | Film/game pipeline standard |
| FBX | Autodesk | Legacy game/film interchange |
| OBJ / MTL | Wavefront | Universal, simple, no animation |
| IFC | BuildingSmart | Architecture BIM |
| STEP / IGES | ISO | Engineering CAD interchange |
| ABC (Alembic) | ILM | Animated geometry (cloth, sims) |

### Motion Capture

| Dataset | Content | License | Priority |
|---------|---------|---------|----------|
| AMASS | 50h+ human motion, 300+ subjects | Academic | **High** |
| CMU MoCap | 2500+ motion sequences | Free | Medium |
| SFU Motion Capture | Diverse sports, activities | Research | Low |

---

## Data Ethics and Licensing

### YouTube Content
- Collected via YouTube Transcript API (public endpoint, no scraping)
- Transcripts are processed into synthesis inputs — model does not reproduce verbatim transcripts
- Tutorial creator attribution maintained in internal dataset metadata
- Fair use justification: transformation (synthesis → training pairs), no commercial exploitation of transcripts themselves

### 3D Model Datasets
- Objaverse / Objaverse-XL: CC-BY 4.0 — attribution maintained in metadata
- ShapeNet: Research license — not for commercial redistribution of raw models; training outputs are permitted
- ABC Dataset: MIT license

### Community Forums
- BlenderArtists, Polycount, Reddit: Public posts under site terms
- No user PII collected; content used only for conversational pattern training
- Model is trained to respond helpfully, not to attribute or reproduce specific users' posts

### Physics Literature
- All referenced papers are used for knowledge extraction only
- Full text Q&A synthesis from PBR book uses fair use for educational/research AI training

---

## Source Counts (pipeline targets)

| Stream | Source Type | Target Volume |
|--------|------------|--------------|
| 1 | Tutorial pairs (Blender) | 300k+ |
| 1 | Tutorial pairs (cross-software) | 100k+ |
| 2 | Geometry build sequences | 150k+ |
| 3 | Physics/material Q&A pairs | 95k+ |
| 4 | Multi-turn conversations | 100k+ |
| 5 | Cross-software integration pairs | 50k+ |
| **Total** | | **~800k+ pairs** |

These are targets for the full pipeline run. Current pipeline capacity with 18× A6000s and 4 vLLM synthesis instances: ~10,000 videos in 8–12 hours, ~50,000 Objaverse objects in 4–6 hours.
