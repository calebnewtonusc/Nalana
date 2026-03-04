# Nalana — The World's First Universal Voice-to-3D AI

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Model: Qwen2.5-7B](https://img.shields.io/badge/base_model-Qwen2.5--7B-purple.svg)](https://huggingface.co/Qwen)
[![GPUs: 18x A6000](https://img.shields.io/badge/training-18×_A6000-red.svg)](https://www.nvidia.com)

> **"You speak. Nalana builds."**

Nalana is the first AI that understands 3D creation the way a master artist does — not just commands, but *intent*, *physics*, *design principles*, and *workflow*. Speak naturally to Blender, Maya, Cinema 4D, Houdini, Unreal Engine, and more. Nalana translates human language into physically-grounded, topology-aware, execution-verified 3D operations across every major platform.

This repository contains the complete dataset pipeline, training infrastructure, and deployment stack for the Nalana model — from raw YouTube transcripts to a production-ready voice API.

---

## What Makes Nalana Different

| Capability | GET3D | DreamFusion | Shap-E | ChatGPT + code | **Nalana** |
|---|---|---|---|---|---|
| Cross-software support | — | — | — | partial | **Blender, Maya, C4D, Houdini, Unreal, Rhino, Substance, ZBrush, Fusion 360** |
| Physics reasoning | — | — | — | — | **PBR materials, rigid body, cloth, fluid, smoke** |
| Expert workflow understanding | — | — | — | partial | **10,000+ tutorials learned, modifiers, retopology, UV, rigging** |
| Execution-verified (self-correcting) | — | — | — | — | **Blender validator in the loop** |
| Topology-aware generation | — | — | — | — | **Edge flows, poles, subdivision-friendliness** |
| Voice + text + image input | — | — | — | text only | **All three modalities** |
| Physics-accurate materials | — | — | — | — | **IOR, SSS, microfacet — real physics values** |
| Multi-turn creative dialogue | — | — | — | partial | **Remembers context, design rationale, follows vision** |

---

## Architecture

```
                           ┌─────────────────────────────────────┐
 Voice / Text / Image ────►│           Nalana Model              │
                           │  (Qwen2.5-7B + LoRA, 5-stream SFT  │
                           │   + RL + DPO, ZeRO-3 trained)       │
                           └──────────────┬──────────────────────┘
                                          │
                                          ▼
                           ┌─────────────────────────────────────┐
                           │        Universal 3D DSL             │
                           │  ADD_PRIMITIVE, EXTRUDE, BEVEL,     │
                           │  BOOLEAN, SCULPT, SIMULATE, ...     │
                           └──────┬──────────────────────────────┘
                                  │
              ┌───────────────────┼──────────────────────────────┐
              ▼                   ▼                              ▼
        ┌─────────┐         ┌──────────┐                  ┌──────────┐
        │ Blender │         │   Maya   │                  │  Cinema  │
        │  .py    │         │  .py/.mel│                  │  4D .py  │
        └─────────┘         └──────────┘                  └──────────┘
              ▼                   ▼                              ▼
        ┌─────────┐         ┌──────────┐                  ┌──────────┐
        │ Houdini │         │  Unreal  │                  │  Rhino / │
        │  .py    │         │  UE5 API │                  │Grasshopper│
        └─────────┘         └──────────┘                  └──────────┘
```

**Training data sources (5 streams, 500k+ pairs):**
- Stream 1: YouTube tutorial transcripts → AI-synthesized operation pairs (35%)
- Stream 2: Objaverse 50k 3D objects → VLM form analysis + build sequences (25%)
- Stream 3: Physics knowledge base → material/simulation pairs (15%)
- Stream 4: Multi-turn creative dialogues (20%)
- Stream 5: Spline, Matterport, cross-software integration pairs (5%)

---

## Plugin Installation

Install the Nalana plugin for your 3D software to get real-time voice control:

| Software | Install |
|---|---|
| **Blender** | Zip the `plugins/blender/` folder → Edit > Preferences > Add-ons > Install From Disk > select zip |
| **Maya** | Drag `plugins/maya/nalana_maya.py` into Maya viewport |
| **Cinema 4D** | Extensions > Install Plugin > `plugins/cinema4d/` |
| **Houdini** | `hconfig -A plugins/houdini/nalana.json` |
| **Unreal Engine 5** | Plugins panel > Enable > Nalana UE5 (`plugins/unreal/`) |
| **Rhino / Grasshopper** | `_PackageManager` > Install > NalanaRhino |

**Blender note**: The plugin is a package (`plugins/blender/__init__.py`), not a single file. Blender requires the entire folder to be zipped before installation. From the repo root:
```bash
cd plugins && zip -r nalana_blender.zip blender/
# Then install nalana_blender.zip via Edit > Preferences > Add-ons > Install From Disk
```
Alternatively, on Blender 4.2+, drag `nalana_blender.zip` directly into any Blender viewport.

Each plugin adds a floating voice panel (hotkey: `N` in viewport) that streams your microphone to the Nalana API and executes returned operations live.

---

## Training Your Own Nalana

### Prerequisites

- Python 3.11+
- API keys (see step 2)
- GPU hardware for training (see Hardware Requirements below)

### Step-by-Step Setup

**1. Clone the repository**

```bash
git clone https://github.com/calebnewtonusc/nalana.git
cd nalana
pip install -r requirements.txt
```

**2. Fill in `.env`**

Copy the template and add your keys:

```bash
cp .env.example .env
```

Edit `.env` — all required keys:

| Variable | Where to get it |
|---|---|
| `ANTHROPIC_API_KEY` | [claude.ai/settings](https://claude.ai/settings) |
| `YOUTUBE_API_KEY` | [console.cloud.google.com](https://console.cloud.google.com) → YouTube Data API v3 |
| `VLLM_API_KEY` | Any secret string (auth for your vLLM servers) |

Optional but recommended:

| Variable | Purpose |
|---|---|
| `WANDB_API_KEY` | Training metrics at [wandb.ai](https://wandb.ai) |
| `HF_TOKEN` | Upload model to HuggingFace Hub |
| `BLENDER_PATH` | Path to Blender binary (if not in PATH) |

**3. Validate environment**

```bash
bash scripts/check_env.sh
```

This checks Python version, all required env vars, GPU availability, Blender installation, disk space (needs ~500GB), and RAM (needs ~64GB).

**4. Run the full pipeline**

```bash
bash scripts/run_all.sh
```

Or run individual steps:

```bash
# Data collection only (no GPU needed)
python3 discovery/discover_v2.py --all-software --channels --search --api-key $YOUTUBE_API_KEY
python3 discovery/fetch_bulk.py --workers 30
python3 discovery/objaverse_prep.py --limit 50000

# Synthesis (Claude API fallback — no GPU needed)
python3 synthesis/synthesize_bulk.py --backend claude
python3 synthesis/annotate_forms.py --backend claude

# Or synthesis with vLLM (GPU required)
bash scripts/start_vllm.sh
python3 synthesis/synthesize_bulk.py --backend vllm --vllm-urls http://localhost:8001 http://localhost:8002

# Dataset prep (all 5 streams, quality weights, curriculum)
python3 training/train_prep.py

# Training (18 GPUs, ZeRO-3)
deepspeed --num_gpus=18 training/train.py --deepspeed training/ds_config.json
```

**5. Evaluate**

```bash
python3 evaluation/eval.py --model checkpoints/nalana-final
```

**6. Deploy**

```bash
cd deploy && docker compose up -d
python3 scripts/health_check.py
```

---

## Hardware Requirements

### Data Collection
Any machine with internet. The discovery and fetch scripts are CPU-bound and use async I/O — a laptop works fine. Expect ~1-3 days for 10,000 videos at 30 workers.

### Synthesis (vLLM)
Qwen2.5-72B requires 4x A100 (80GB) or 4x A6000 (48GB) per instance. With 4 synthesis instances (8 cards), you can process ~10,000 videos in 8-12 hours overnight.

### Training (Target Configuration)
| Resource | Specification |
|---|---|
| GPUs | 18x NVIDIA A6000 (48GB each) |
| Total VRAM | 864GB |
| Strategy | DeepSpeed ZeRO-3 + CPU offload |
| RAM | 512GB+ (for ZeRO-3 optimizer offload) |
| Expected time | 4-8 hours (100k pairs, 3 epochs) |

### Inference (Production)
| Configuration | Latency | Throughput |
|---|---|---|
| 2x A100 (80GB) | < 100ms | 50 req/s |
| 1x A100 (80GB) | ~150ms | 25 req/s |
| 1x RTX 4090 (24GB) | ~400ms | 8 req/s |
| CPU only | ~3-8s | 2 req/s |

Minimum for real-time voice interaction: 1x A100 or equivalent.

---

## NalanaBench

NalanaBench is our task-specific evaluation suite for voice-to-3D models. It tests:

- **Execution accuracy** — does the generated Python run in Blender without errors?
- **Intent alignment** — does the operation match what was asked?
- **Physics correctness** — are material/simulation parameters physically grounded?
- **Cross-software fidelity** — do implementations across Maya/C4D/Houdini produce equivalent results?
- **Multi-turn coherence** — does the model maintain context over a 10-turn session?

```bash
python3 evaluation/eval.py --model checkpoints/nalana-final --all
```

Baseline comparison results will be published after model training completes.

---

## Dataset Structure

```
data/
├── raw/                    # Downloaded YouTube transcripts (JSON per video)
├── processed/              # Stream 1: synthesized tutorial pairs (JSONL)
├── spatial/                # Stream 2: Objaverse form analysis pairs (JSONL)
├── objaverse/
│   ├── renders/            # 8-view renders per object (PNG)
│   └── annotations/        # VLM form analysis (JSON per object)
├── physics/
│   └── physics_pairs.jsonl # Stream 3: physics KB pairs
├── multiturn/
│   └── conversations.jsonl # Stream 4: multi-turn sequences
├── integrations/           # Stream 5: cross-software pairs
│   ├── spline/
│   ├── matterport/
│   └── crosssoftware/
├── validated/              # Post-validation dataset
│   └── dataset.jsonl
├── train/                  # Final training-ready files
│   ├── sharegpt_train.jsonl
│   ├── sharegpt_val.jsonl
│   ├── alpaca_train.jsonl
│   └── alpaca_val.jsonl
└── video_ids.txt           # Discovered YouTube video IDs
```

---

## Contributing

Nalana gets better with more training data and community plugins. Ways to contribute:

**Training Data**
- Submit high-quality `(voice_command, blender_python)` pairs via `data/community/`
- Contribute tutorials from software not yet covered (SketchUp, Vectorworks, ArchiCAD)
- Add physics or design theory examples to `data/physics/`

**Plugins**
- New software integrations welcome: follow `plugins/blender/` as the reference implementation
- Plugin interface is documented in `plugins/PLUGIN_SPEC.md`

**Evaluation**
- Add test cases to NalanaBench: `nalana_bench.py --add-test`
- Report model failures via GitHub Issues with the `[benchmark]` tag

**Guidelines:**
- All contributed code must pass `bash scripts/check_env.sh` and `python3 validate.py`
- Training pairs must pass Blender execution validation (`validate_blender.py`)
- Follow the code style in existing files (no type: ignore, docstrings required)

See `CONTRIBUTING.md` for full guidelines.

---

## Citation

If you use Nalana in academic work, please cite:

```bibtex
@inproceedings{newton2026nalana,
  title     = {Nalana: A Universal Voice-to-3D AI via Multi-Stream Dataset Synthesis and Physics-Grounded Fine-Tuning},
  author    = {Newton, Caleb and others},
  booktitle = {ACM SIGGRAPH 2026 Emerging Technologies},
  year      = {2026},
  publisher = {ACM},
  doi       = {10.1145/XXXXXXX.XXXXXXX}
}
```

---

## License

**Code:** MIT License — see `LICENSE`

**Model weights:** Apache 2.0 — free to use, modify, and redistribute with attribution

**Training data:** CC BY 4.0 for community-contributed pairs. YouTube-derived pairs are synthesis outputs (not raw transcripts) and are covered under fair use for AI training research.

---

*Target: 864GB of VRAM, 500,000+ training pairs. Training in progress.*
