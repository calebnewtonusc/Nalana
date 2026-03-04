# Nalana — Full System Architecture
## "The world's first universal voice-to-3D AI"

---

## THE VISION

User says: "Create a photorealistic iPhone 16 with reflective glass"
Nalana: understands form → plans build sequence → executes in ANY 3D software → done.

Not just Blender. Maya. Cinema 4D. Houdini. ZBrush. Rhino. Unreal. All of them.

---

## 4-PHASE PRODUCT VISION

```
Phase 1 (v1):   GENERATE          voice/text → executable 3D workflow    ← CURRENT BUILD
Phase 2 (v1.5): MAKE PRODUCTION-READY   retopo, UV, bake, LOD, collision
Phase 3 (v2):   ANALYZE & CRITIQUE      architecture compliance, QA, CAD optimization
Phase 4 (v3):   SHIP              asset management, digital twins, bidirectional editing
```

### Phase 1 — v1: GENERATE (Current)

The foundation. Any voice or text command maps to an executable 3D workflow in any supported software. Covers modeling, materials, lighting, physics simulation, rigging, and animation. Handles single operations ("bevel these edges") and full multi-step builds ("create a brutalist apartment").

### Phase 2 — v1.5: MAKE PRODUCTION-READY

Takes any asset from artistic to shippable. The model gains the full production pipeline: retopology, UV unwrapping, normal baking, LOD chain generation, and collision mesh creation. An artist imports a ZBrush sculpt, speaks to Nalana, and the asset is game-engine ready in minutes — not days.

### Phase 3 — v2: ANALYZE & CRITIQUE

Nalana becomes a technical reviewer, not just an executor. Architecture mode: reads floorplans, runs IBC/ADA compliance checks, computes daylight factors, flags egress failures. CAD mode: runs DFM analysis, finds draft angle violations, suggests topology optimization paths. QA mode: audits any scene for naming, transform, UV, and topology errors before export.

### Phase 4 — v3: SHIP

Nalana integrates into the full studio pipeline. Asset management at scale: search, tag, deduplicate, batch re-texture entire libraries. Digital twin mode: bidirectional sync between physical sensor data and 3D models. The model can receive geometry edits from Blender and push updates back to Matterport, Spline, or CAD systems.

---

## TARGET METRICS

| Version | Task | Success Rate | Latency | Key Benchmark |
|---------|------|-------------|---------|---------------|
| v1 | Op execution | >85% | <500ms | NalanaBench-Ops |
| v1 | Voice → correct op | >80% | — | NalanaBench-Voice |
| v1 | Multi-step coherence | >75% | — | NalanaBench-Build |
| v1.5 | Retopo face count within 10% of target | >80% | <2s | NalanaBench-Retopo |
| v1.5 | UV distortion < 5% avg stretch | >85% | — | NalanaBench-UV |
| v1.5 | Normal bake PSNR vs ground truth | >30dB | — | NalanaBench-Bake |
| v2 | IBC compliance check accuracy | >90% | <1s | NalanaBench-Arch |
| v2 | DFM issue detection recall | >85% | — | NalanaBench-CAD |
| v2 | Scene QA false positive rate | <5% | — | NalanaBench-QA |
| v3 | Asset dedup precision | >95% | — | NalanaBench-Assets |

---

## 7 TECHNICAL DIFFERENTIATORS

### 1. Free Verifiable Reward Signal (headless Blender)

Every generated `blender_python` snippet is tested in a headless Blender instance. The execution result — success, no-op, or error — is a free, binary, ground-truth signal. This enables Stage 2 RL training without human labelers. No other 3D AI model has a verifiable execution reward loop. The signal scales to millions of samples at near-zero cost.

### 2. Cross-Software via Universal DSL

All operations are normalized to a Universal 3D Operation Language before being compiled to software-specific APIs. Train once on the DSL, execute in Blender today, Maya tomorrow, Houdini next month. The DSL is the abstraction layer that makes Nalana software-agnostic. Every other 3D AI is tied to one software stack.

### 3. Expert Reasoning from Tutorial Synthesis (the data moat)

Nalana's training data is extracted from thousands of hours of expert 3D artists explaining not just what they do, but why — topology rationale, proportion decisions, lighting intent, material physics. This expert reasoning is not in any pre-existing dataset. Competitors scraping text or 3D models get the what; Nalana gets the why. This is the data moat.

### 4. Topology-Aware Generation

When Nalana generates geometry, it reasons about edge flow, pole placement, subdivision behavior, and deformation zones — not just vertex positions. The model has internalized the difference between topology that will render cleanly, animate correctly, and UV unwrap efficiently versus topology that looks fine at rest but breaks in production.

### 5. Physics-Grounded Materials

Material parameters are derived from real physical quantities: IOR from optical physics, roughness from microfacet theory, SSS radius from measured photon mean free paths in biological tissue. When a user says "make this look like aged copper," Nalana reasons from electron band structure and oxide layer physics to exact Principled BSDF values — not from pattern-matching to "copper-looking" training examples.

### 6. 3-Stage Training (no other 3D model does Stage 2 or 3)

Stage 1: supervised fine-tuning on expert demonstration data.
Stage 2: reinforcement learning with execution reward (headless Blender).
Stage 3: DPO on conversation quality and when-to-ask-questions behavior.
No other 3D AI model runs Stage 2 or Stage 3. Stage 2 is the reason Nalana's code actually executes. Stage 3 is the reason Nalana knows when to act and when to ask.

### 7. NalanaBench Defines the Evaluation Standard

There is no standard benchmark for voice-controlled 3D AI. NalanaBench fills that gap, defining evaluation across ops execution, voice naturalness, multi-step coherence, production readiness, and architecture compliance. Publishing NalanaBench positions Nalana as the standard against which all future 3D AI models are measured — regardless of who builds them.

---

## ALL 14 TASK TYPES

| Task Type | Description | Source Module |
|-----------|-------------|---------------|
| EXECUTE | Single Blender op from voice command | Tutorial transcripts (Stream 1) |
| BUILD | Multi-step construction sequence | Form analysis + intent decomp (Stream 2) |
| MATERIALIZE | Physics-accurate PBR material setup | Physics reasoning (Stream 3) |
| SIMULATE | Rigid body, cloth, fluid, particles | Physics reasoning (Stream 3) |
| LIGHT | Lighting design with physical rationale | Tutorial transcripts + synthesis |
| UNDERSTAND | Expert explanation of design/physics/topology | Multi-turn conversations |
| CROSS_SOFTWARE | Universal DSL translation across software | Cross-software normalization |
| RETOPO | Retopology — high-poly → clean quad mesh | Production module (v1.5) |
| UV_UNWRAP | UV map creation, seams, texel density, UDIM | Production module (v1.5) |
| BAKE | Normal/AO/curvature transfer high-poly → low-poly | Production module (v1.5) |
| LOD | LOD chain generation (LOD0-LOD3) | Production module (v1.5) |
| COLLISION | Convex hull / box / capsule collision mesh | Production module (v1.5) |
| ARCH_GENERATE | Architecture generation from constraints | Architecture module (v2) |
| ARCH_ANALYZE | IBC/ADA compliance, daylight, pros/cons | Architecture module (v2) |
| CAD_OPTIMIZE | Topology optimization, DFM, materials selection | CAD module (v2) |
| QA_LINT | Scene audit — naming, transforms, UVs, topology | QA module (v2) |
| ASSET_MANAGE | Search, tag, dedup, batch operations | Asset module (v3) |
| SCAN_PROCESS | NeRF/LiDAR/photogrammetry cleanup and extraction | Asset module (v3) |

---

## 3-STAGE TRAINING PIPELINE

### Stage 1 — Supervised Fine-Tuning: train.py

Teaches the model what correct behavior looks like. Training data covers all 5 synthesis streams: tutorial ops, spatial form analysis, physics knowledge, multi-turn conversations, and cross-software integration pairs.

```
Input:  (voice_command, scene_context, optional_image)
Output: (Universal DSL JSON + software-specific Python)

Data mix:
  35% Stream 1: Tutorial procedural pairs (voice → op)
  25% Stream 2: 3D spatial pairs (form analysis + build sequences)
  15% Stream 3: Physics knowledge pairs (materials, simulation)
  20% Stream 4: Multi-turn sequences ("Create an iPhone" → 25 steps)
   5% Stream 5: Cross-software integration pairs

Target dataset: ~500k pairs after quality filter
Base model: Qwen2.5-Coder-7B-Instruct
Training: DeepSpeed ZeRO-3, LoRA rank 64, 3 epochs, ~6 hours on 18×A6000
```

### Stage 2 — Execution RL: train_rl.py

Uses headless Blender as a free, ground-truth reward signal. The model generates Blender Python; Blender executes it; the result (success / no-op / error) is the reward. No human labelers required. This is what makes Nalana's code actually run — and what no other 3D model does.

```
Reward function:
  +1.0  code executes, scene state changes correctly
  +0.5  code executes, no error, but scene unchanged
  -1.0  Python exception raised
  -0.5  code runs but produces invalid geometry

Algorithm: GRPO with KL penalty against Stage 1 checkpoint
Parallelism: 12 headless Blender workers (GPUs 16-17)
Target: execution success rate >85% on held-out test set
```

### Stage 3 — DPO on Conversation Quality: train_dpo.py

Teaches the model when to ask versus when to act, how to ask targeted expert questions, and how to match the user's expertise level. Uses preference pairs generated by: (a) human raters on multi-turn conversations, (b) automated preference from `generate_dpo_pairs.py`.

```
Preference signal sources:
  - Human: "this response asked unnecessary questions" vs. "this executed and explained"
  - Automated: "this question is too vague" vs. "this question is specific and expert"
  - Rule-based: penalize >1 clarifying question in a single response

Algorithm: Direct Preference Optimization (DPO)
Data: ~50k preference pairs
Teaches: smart defaults, one focused question, insight volunteering
```

---

## WHAT MAKES THIS DIFFERENT

| Tool | What it does | What it lacks |
|------|-------------|---------------|
| GET3D (NVIDIA) | Generates geometry from images | No design reasoning, no software control |
| DreamFusion | Text → NeRF blob | Not editable, not production-ready |
| Shap-E | Text → 3D mesh | Low quality, no workflow understanding |
| ChatGPT w/ code | Generates Blender Python | No 3D spatial understanding, no RL loop |
| **Nalana** | Voice → expert 3D workflow in any software | **Nothing like this exists** |

The moat: Nalana learns from thousands of hours of expert 3D artists explaining WHY they make each decision — topology, proportions, form language, lighting rationale. That knowledge is not in any dataset. We're building it. Combined with the Stage 2 execution RL loop and Stage 3 DPO, the result is a model that reasons, executes, and communicates at the level of a senior 3D artist.

---

## HARDWARE

```
18x A6000 (48GB VRAM each) = 864GB total VRAM
$25k Azure credits for storage, networking, deployment
Timeline: ~2-3 days total (data overnight, training overnight, deploy day 3)
```

### GPU Allocation During Synthesis

```
GPUs  0-3:  vLLM Qwen2.5-72B   (text synthesis, port 8001)
GPUs  4-7:  vLLM Qwen2.5-72B   (text synthesis, port 8002)
GPUs  8-11: vLLM Qwen2-VL-72B  (vision form analysis, port 8003)
GPUs 12-15: vLLM Qwen2-VL-72B  (vision form analysis, port 8004)
GPUs 16-17: Headless Blender   (validation + RL reward workers)
```

### GPU Allocation During Training

```
All 18 A6000s: DeepSpeed ZeRO-3 (Stage 1 SFT)
  Model: 7B × bf16 = ~14GB
  Per GPU: 14GB model + 30GB activations/optimizer = ~44GB (fits with ZeRO-3)
  Batch: 4 per GPU × 18 GPUs × 4 grad_accum = 288 effective batch size

GPUs 16-17 idle during Stage 2 RL: Blender workers
All 18 for DPO (Stage 3): same config as Stage 1, smaller dataset
```

---

## FILE STRUCTURE

```
nalana-dataset/
│
├── DATA COLLECTION
│   └── discovery/
│       ├── discover.py              Channel crawl (legacy, kept for reference)
│       ├── discover_v2.py           Channel crawl + keyword search for all 3D software
│       ├── fetch.py                 Single YouTube transcript fetch
│       ├── fetch_bulk.py            Parallel transcript fetch, 30 workers
│       ├── api_harvest.py           Sketchfab + Polyhaven + Thingiverse mass download
│       ├── objaverse_prep.py        Download + filter Objaverse (50k objects)
│       └── urls.txt                 Curated URL list
│
├── SYNTHESIS
│   └── synthesis/
│       ├── prompts.py               All system prompts — canonical, import from here
│       ├── synthesize.py            Single-video synthesis via Claude
│       ├── synthesize_bulk.py       Async multi-video (vLLM or Claude backend)
│       ├── annotate_forms.py        VLM form analysis + build sequences
│       ├── multi_turn.py            Multi-turn conversation chain generation
│       └── generate_dpo_pairs.py    Preference pair generation for Stage 3 DPO
│
├── CORE (cross-software)
│   └── core/
│       ├── universal_dsl.py         Universal 3D DSL + software-specific compilers
│       └── file_formats.py          40+ format support (trimesh/OCC/USD/rhino3dm)
│
├── 3D RENDERING
│   └── render/
│       └── render_pipeline.py       Headless Blender 8-view rendering for form analysis
│
├── VALIDATION
│   └── validation/
│       ├── validate.py              Quality filtering, dedup, score thresholding
│       └── validate_blender.py      Headless Blender execution validator (RL reward source)
│
├── TRAINING
│   └── training/
│       ├── train_prep.py            Merge all streams → HuggingFace dataset format
│       ├── train.py                 Stage 1: DeepSpeed + LoRA SFT (Qwen2.5-Coder-7B)
│       ├── train_rl.py              Stage 2: GRPO execution RL with Blender reward
│       ├── train_dpo.py             Stage 3: DPO on conversation quality preferences
│       ├── ds_config.json           DeepSpeed ZeRO-3 config (Stage 1 SFT + Stage 3 DPO)
│       └── ds_config_rl.json        DeepSpeed ZeRO-2 config for Stage 2 GRPO RL
│
├── EVALUATION
│   └── evaluation/
│       ├── eval.py                  Post-training model evaluation (NalanaBench)
│       └── task_classifier.py       Classify incoming commands by task type
│
├── DOMAIN AGENTS
│   └── agents/
│       ├── arch_agent.py            Architecture domain training pair generation
│       ├── cad_agent.py             CAD domain training pair generation
│       ├── animation_agent.py       Animation domain training pair generation
│       ├── production_agent.py      Production (retopo/UV/bake) training pairs
│       └── qa_agent.py              QA / scene audit training pairs
│
├── INTEGRATIONS
│   └── integrations/
│       ├── spline_scraper.py        Spline web 3D connector
│       ├── matterport_harvest.py    Matterport digital twin connector
│       ├── topological_signal.py    Topological DFM/simulation connector
│       ├── dream3d_synthetic.py     Dream3D generative connector
│       └── collect_design_physics.py  Physics + design theory KB collector
│
├── DEPLOYMENT
│   └── deploy/
│       ├── docker-compose.yml       One-command deploy: model + API + nginx + Redis
│       ├── Dockerfile               Multi-stage CUDA container
│       ├── api_server.py            FastAPI REST + WebSocket server
│       └── nginx.conf               Reverse proxy + SSL template
│
├── SCRIPTS
│   └── scripts/
│       ├── run_all.sh               Master 21-step pipeline
│       ├── start_vllm.sh            Launch vLLM servers on GPUs 0-15
│       ├── check_env.sh             Verify CUDA, Blender, Python deps
│       └── health_check.py          Post-deploy smoke test
│
├── PLUGINS
│   └── plugins/                     Software-native plugins (one folder per platform)
│       ├── blender/                 Blender Python add-on (zip to install)
│       ├── maya/                    Maya MEL + Python plugin
│       ├── cinema4d/                Cinema 4D Python plugin
│       ├── houdini/                 Houdini Python SOP
│       ├── unreal/                  Unreal Engine Python plugin
│       ├── rhino/                   Rhino Python plugin
│       ├── substance/               Substance Painter Python plugin
│       ├── unity/                   Unity C# Editor extension
│       └── web/                     Three.js / Babylon.js NPM module
│
├── DATA (gitignored)
│   └── data/
│       ├── raw/                     Downloaded transcripts, model files
│       ├── processed/               Stream 1 synthesized tutorial pairs
│       ├── spatial/                 Stream 2 Objaverse form analysis pairs
│       ├── physics/                 Stream 3 physics KB pairs
│       ├── multiturn/               Stream 4 multi-turn conversations
│       ├── integrations/            Stream 5 cross-software pairs
│       ├── validated/               Quality-filtered pairs (score ≥ 0.7)
│       ├── train/                   Final HuggingFace dataset splits
│       └── dpo_pairs/               Preference pairs for Stage 3
│
├── PAPER
│   └── paper/nalana_paper.md        SIGGRAPH/ICCV submission draft
│
├── RESULTS
│   └── results/                     Eval results, training curves, benchmark outputs
│
├── ROOT-LEVEL SCRIPTS
│   ├── data_discovery.py            LLM-guided autonomous source hunter
│   └── pipeline.py                  Lightweight pipeline runner / entry point
│
└── REQUIREMENTS
    └── requirements.txt             Python dependencies
```

---

## COMPLETE RUN ORDER

```bash
# ── STEP 1: Verify environment ────────────────────────────────────────────────
bash scripts/check_env.sh
# Checks: CUDA 12+, Blender 4+, Python 3.11+, all pip deps

# ── STEP 2: Fill environment variables ───────────────────────────────────────
cp .env.example .env && nano .env
# Required: YOUTUBE_API_KEY, SKETCHFAB_API_TOKEN, HF_TOKEN, WANDB_API_KEY
# Already available: ANTHROPIC_API_KEY, AZURE_SUBSCRIPTION_ID

# ── STEP 3: Discover tutorial videos (all 3D software) ───────────────────────
python discovery/discover_v2.py --api-key $YOUTUBE_API_KEY --channels --search --all-software
# Output: data/video_ids.txt with 10,000+ video IDs across all supported software

# ── STEP 4: Harvest 3D model datasets ────────────────────────────────────────
python discovery/api_harvest.py --all
# Downloads metadata + download URLs from Sketchfab, Polyhaven, Thingiverse

# ── STEP 5: Fetch transcripts in parallel ────────────────────────────────────
python discovery/fetch_bulk.py --workers 30
# ~30 minutes for 10,000 videos. Output: data/raw/transcripts/

# ── STEP 6: Download and prep Objaverse ──────────────────────────────────────
python discovery/objaverse_prep.py --limit 50000
# Downloads 50k objects, filters by quality, converts to GLB

# ── STEP 7: Start vLLM synthesis servers ─────────────────────────────────────
bash scripts/start_vllm.sh
# Launches 4 vLLM servers on GPUs 0-15 (2 text, 2 vision)

# ── STEP 8: Synthesize tutorial pairs (Stream 1) ─────────────────────────────
python synthesis/synthesize_bulk.py --backend vllm \
  --vllm-urls http://localhost:8001 http://localhost:8002
# ~8 hours overnight. Output: data/processed/

# ── STEP 9: Form analysis on 3D models (Stream 2) ────────────────────────────
python render/render_pipeline.py --input data/raw/models/ --output data/raw/renders/
python synthesis/annotate_forms.py --backend vllm --vllm-url http://localhost:8003
# Renders 8 views per model, then runs VLM form analysis

# ── STEP 10: Generate multi-turn conversations (Stream 4) ────────────────────
python synthesis/multi_turn.py --backend vllm --vllm-url http://localhost:8001
# Chains individual ops into full build conversations

# ── STEP 11: Generate DPO preference pairs (Stage 3 prep) ────────────────────
python synthesis/generate_dpo_pairs.py --backend vllm
# Creates chosen/rejected preference pairs for conversation quality training

# ── STEP 12: Validate all synthesized data ───────────────────────────────────
python validation/validate.py
# Quality scoring, dedup, threshold filter (keep score ≥ 0.7)

python validation/validate_blender.py
# Headless Blender execution test on all blender_python snippets
# Only pairs with exec_success pass to training

# ── STEP 13: Prepare training datasets ───────────────────────────────────────
python training/train_prep.py
# Merges all streams, formats as HuggingFace Dataset, creates splits
# Output: data/train/ (train/val/test splits)

# ── STEP 14: Stage 1 — Supervised Fine-Tuning ────────────────────────────────
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 \
deepspeed --num_gpus=18 training/train.py \
  --deepspeed training/ds_config.json \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --data-dir data/train \
  --output-dir checkpoints/nalana-sft \
  --epochs 3
# ~6 hours. Checkpoint saved to checkpoints/nalana-sft/final

# ── STEP 15: Stage 2 — Execution RL ──────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 \
deepspeed --num_gpus=18 training/train_rl.py \
  --deepspeed training/ds_config_rl.json \
  --base-model checkpoints/nalana-sft/final \
  --output-dir checkpoints/nalana-rl \
  --blender-workers 12
# ~4 hours. Reward: headless Blender GRPO execution success

# ── STEP 16: Stage 3 — DPO Conversation Quality ──────────────────────────────
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 \
deepspeed --num_gpus=18 training/train_dpo.py \
  --deepspeed training/ds_config.json \
  --base-model checkpoints/nalana-rl/final \
  --dpo-data data/dpo_pairs \
  --output-dir checkpoints/nalana-final
# ~2 hours. Final checkpoint is production model.

# ── STEP 17: Evaluate (NalanaBench) ──────────────────────────────────────────
python evaluation/eval.py --model checkpoints/nalana-final
# Runs full NalanaBench: ops, voice, multi-step, format coverage
# Output: results/nalana-final-benchmark.json

# ── STEP 18: Review results ───────────────────────────────────────────────────
# Check results/nalana-final-benchmark.json
# Target: >85% op execution, >80% voice accuracy, <500ms p50 latency

# ── STEP 19: Deploy ───────────────────────────────────────────────────────────
cd deploy && docker compose up -d
# Spins up: vLLM model server + FastAPI + WebSocket + Nginx + Redis
# Nalana is live at https://api.nalana.ai

# ── STEP 20: Smoke test ───────────────────────────────────────────────────────
curl -X POST https://api.nalana.ai/v1/command \
  -H "Content-Type: application/json" \
  -d '{"voice": "add a cube to the center", "software": "blender"}'
# Expected: {"blender_python": "bpy.ops.mesh.primitive_cube_add()", ...}
```

---

## DATA COLLECTION SOURCES

### Stream 1: Tutorial Transcripts (ALL 3D SOFTWARE)

YouTube transcripts scraped for every major 3D tool:

| Software | What we learn | Python API for execution |
|----------|--------------|--------------------------|
| Blender | Full workflow, modeling/sculpt/rig/render | `bpy.ops.*` |
| Maya | Film/TV industry standard | `maya.cmds.*`, `pymel.*` |
| Cinema 4D | Motion graphics, MoGraph | `c4d.*` |
| Houdini | Procedural VFX, simulations | `hou.*` |
| ZBrush | Character sculpting | ZScript / ZBrush Bridge |
| Substance Painter | PBR texturing | `substance_painter.*` |
| Substance Designer | Material creation | `sd.*` |
| Rhino 3D | NURBS, architecture, product | `rhinoscriptsyntax.*` |
| Grasshopper | Parametric/generative | GH Python |
| SketchUp | Architecture visualization | `sketchup.*` Ruby/Python |
| 3ds Max | VFX, architecture | `MaxPlus.*`, pymxs |
| Fusion 360 | Engineering CAD | `adsk.*` API |
| Unreal Engine 5 | Game/film environments | Python Editor API |
| Unity | Game environments | C# / Python editor |
| Marvelous Designer | Cloth simulation | MD Python |
| World Creator / Gaea | Terrain generation | Python API |

**Target: 10,000+ tutorial videos → 500,000+ procedural training pairs**

### Stream 2: 3D Model Datasets (GEOMETRIC UNDERSTANDING)

| Source | Count | License | Format | What we learn |
|--------|-------|---------|--------|---------------|
| Objaverse | 800k objects | CC | GLB | Form, proportions, topology |
| Objaverse-XL | 10M objects | CC | GLB | Scale of 3D world |
| ShapeNet v2 | 51k clean | CC-BY | OBJ | Annotated categories |
| ABO (Amazon) | 8k products | CC | GLB | Product design proportions |
| GSO (Google) | 1k scanned | CC | OBJ | Real-world scan quality |
| Sketchfab CC | 400k+ | CC | GLB | Diverse real-world models |
| Adobe Substance 3D | 6k+ | Licensed | FBX/GLB | Production-ready models + PBR |
| Polyhaven | 500+ | CC0 | BLEND | High quality reference |
| Thingiverse | 2M+ | CC | STL | Printable designs |
| NIH 3D Print | 10k+ | Public | STL/OBJ | Scientific/medical |
| Smithsonian 3D | 4k+ | CC0 | GLB | Cultural artifacts |

**Target: 100,000 models rendered + annotated → 2,000,000+ spatial training pairs**

### Stream 3: Multi-Turn Conversations (WORKFLOW UNDERSTANDING)

Full build conversations chained from individual op pairs. "Create an iPhone 16" becomes a 25-step structured conversation with scene state propagated across turns.

### Stream 4: Design Knowledge (EXPERT REASONING)

- Topology guides (edge flow, poles, subdivision-friendly topology)
- Lighting theory (three-point, HDRI, cinematic, product lighting)
- PBR material theory (roughness, metallic, subsurface)
- Composition principles (rule of thirds, camera angles, focal length)
- Animation principles (12 Disney principles translated to 3D)
- Color theory in 3D (value, saturation, temperature in renders)
- Form language (primary/secondary/tertiary shapes, silhouette design)

Source: books, articles, design school content → synthesized into training pairs

---

## THE UNIVERSAL DSL

All software-specific operations are normalized to a Universal 3D Operation Language before compilation to platform APIs:

```json
{
  "op": "EXTRUDE",
  "target": "selected_faces",
  "args": { "amount": 0.5, "direction": "normal" },
  "software_implementations": {
    "blender":   "bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={'value':(0,0,0.5)})",
    "maya":      "cmds.extrude(et=1, d=[0,0,0.5])",
    "cinema4d":  "c4d.CallCommand(12238)",
    "houdini":   "node.parm('dist').set(0.5); node.cook()",
    "unreal":    "# Modeling tools extrude"
  }
}
```

This means:
1. Train on Universal DSL → model understands the operation conceptually
2. Compile DSL → any software at inference time
3. Nalana works in Blender today, Maya tomorrow, Houdini next month

---

## VALIDATION PIPELINE

### Headless Blender Execution Test

For every generated `blender_python` snippet:
1. Spawn headless Blender instance
2. Execute the snippet in a fresh or staged scene
3. Check: no Python exception, scene state changed, geometry is valid
4. Score: 1.0 (success + change) / 0.5 (ran, no error, no change) / 0.0 (exception)
5. Filter: only pairs with score ≥ 0.5 enter training

### Quality Scoring

Each training pair receives a composite quality score (0-1):
- `op_valid`: bpy.ops name is in known registry
- `exec_success`: headless Blender test passed
- `voice_natural`: voice command passes naturalness heuristics (3-15 words, imperative)
- `no_duplicate`: not a near-duplicate of existing pair (MinHash similarity < 0.9)

**Only pairs with composite quality ≥ 0.7 enter Stage 1 training.**

---

## DEPLOYMENT

### One Command Deploy

```bash
docker compose up -d
```

Spins up:
- vLLM serving the trained Nalana model (GPU-backed)
- FastAPI REST + WebSocket server
- Nginx reverse proxy + SSL termination
- Redis for session state (multi-turn conversation history)

### API Surface

```
POST /v1/command
  { "voice": "add a cube", "scene_context": {...}, "software": "blender" }
  → { "blender_python": "...", "universal_dsl": {...}, "reasoning": "..." }

WS  /v1/stream
  Real-time streaming for live voice → execution loop

POST /v1/plan
  { "intent": "create an iPhone 16", "software": "blender" }
  → { "build_plan": [...25 steps...], "estimated_time": "45min" }

POST /v1/qa
  { "scene_context": {...} }
  → { "score": 87, "critical_errors": [...], "fix_code": "..." }

POST /v1/arch
  { "constraints": {...}, "task_type": "ARCH_GENERATE" }
  → { "scheme": "...", "blender_python": "...", "code_compliance": {...} }
```

### Infrastructure

```
Azure VM: Standard_NC48ads_A100_v4 (or equivalent A6000)
  → vLLM serving trained model (2-4 GPUs for inference)
  → ~100ms p50 latency per command

Azure Container Registry → Docker image storage
Azure CDN → Global distribution
Azure Monitor → Logging + alerting
```

---

## COMPANY INTEGRATIONS

### Spline (3D web design)

Bidirectional: Nalana generates geometry, Spline renders it in browser. Users build in Spline's visual editor, Nalana adds intelligence — topology checks, material suggestions, animation. Export path: Nalana → Universal DSL → Spline scene format.

### Matterport (digital twins / spatial data)

Matterport scans become editable 3D scenes. Nalana ingests Matterport point clouds, classifies objects (walls, floors, furniture), extracts clean meshes. Phase 4 use case: "update this room's furniture" → Nalana edits the digital twin → syncs back.

### Topological (simulation-driven design)

Topological's simulation outputs feed Nalana's CAD module. Stress analysis results → Nalana recommends material removal, identifies DFM violations, generates topology-optimized alternatives with full Blender Python implementation.

### Dream3D (generative 3D)

Dream3D handles latent-space generation; Nalana handles production pipeline. A Dream3D-generated mesh → Nalana retopo → UV → bake → LOD → export. Integration connector handles the handoff format (GLB/FBX).

### Adam (AI design assistant)

Adam focuses on parametric design exploration; Nalana handles execution and production. Users iterate designs with Adam's generative system, then speak to Nalana to make the chosen design game-ready or film-ready.

### One Robot (robotics / physical AI)

One Robot's physical AI systems need accurate 3D representations of real-world objects. Nalana's SCAN_PROCESS task type handles NeRF/LiDAR/photogrammetry input from One Robot sensors, extracts clean meshes, and generates collision geometry for robot planning.

---

## EVALUATION: NalanaBench

NalanaBench is the evaluation standard Nalana defines for voice-controlled 3D AI. Published alongside model release to establish the field benchmark.

### Benchmark Suites

```
NalanaBench-Ops        500 single-operation test cases, measured by execution success
NalanaBench-Voice      200 human-eval samples, voice naturalness score 1-5
NalanaBench-Build      50 full build conversations, evaluated by scene diff scoring
NalanaBench-Retopo     100 meshes, measured by face count accuracy and quad ratio
NalanaBench-UV         100 meshes, measured by distortion and texel density consistency
NalanaBench-Bake       50 bake tasks, measured by PSNR vs ground truth normal maps
NalanaBench-Arch       30 floorplan scenarios, measured by IBC compliance accuracy
NalanaBench-CAD        40 CAD parts, measured by DFM issue detection recall
NalanaBench-QA         60 scenes with planted errors, measured by detection F1 score
```

### v1 Baseline Targets (for model release)

| Benchmark | Target | Notes |
|-----------|--------|-------|
| NalanaBench-Ops | >85% | Execution success rate |
| NalanaBench-Voice | >4.0/5.0 | Human naturalness rating |
| NalanaBench-Build | >75% | Scene diff coherence |
| Inference latency | <500ms p50 | vLLM single-GPU |
| Supported formats | 40+ | file_formats.py |
| Supported software | 15+ | universal_dsl.py |

---

## API KEYS NEEDED

```bash
# .env file — fill these in before running anything

# YouTube (tutorial discovery)
YOUTUBE_API_KEY=...          # console.cloud.google.com — free, 10k quota/day

# Sketchfab (3D model downloads)
SKETCHFAB_API_TOKEN=...      # sketchfab.com/settings/password — free account

# Adobe Substance 3D (production models + materials)
ADOBE_CLIENT_ID=...          # developer.adobe.com — requires approval
ADOBE_CLIENT_SECRET=...

# Hugging Face (Objaverse, model weights)
HF_TOKEN=...                 # huggingface.co/settings/tokens — free

# Anthropic (synthesis fallback + form analysis)
ANTHROPIC_API_KEY=...        # already configured

# Weights & Biases (training monitoring)
WANDB_API_KEY=...            # wandb.ai — free

# Azure (deployment)
AZURE_SUBSCRIPTION_ID=...    # already configured ($25k credits)
AZURE_RESOURCE_GROUP=...

# Thingiverse (printable models)
THINGIVERSE_TOKEN=...        # thingiverse.com/apps — free

# Polyhaven — NO KEY NEEDED (free CDN, CC0 license)
```

---

## STATUS

All pipeline components are implemented and tested. See `ROADMAP.md` for future version (v1.5, v2, v3) plans.
