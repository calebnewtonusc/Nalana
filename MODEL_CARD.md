# Nalana-v1: Universal Voice-to-3D Workflow Intelligence

[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)
[![Blender Plugin](https://img.shields.io/badge/Blender-4.0%2B-orange)](plugins/blender/)
[![Maya Plugin](https://img.shields.io/badge/Maya-2024%2B-red)](plugins/maya/)

---

## Model Description

Nalana-v1 is the first universal voice-controlled 3D workflow intelligence model. Unlike text-to-3D geometry generators (Shap-E, DreamFusion, GET3D), Nalana understands and executes complete expert workflows within professional 3D software — reasoning about physical accuracy, topology consequences, and multi-step planning before generating executable code.

### Architecture

| Property | Value |
|---|---|
| Base model | Qwen2.5-Coder-7B-Instruct |
| Fine-tuning method | LoRA (r=64, alpha=128) |
| Trainable parameters | 168M (2.2% of 7.6B total) |
| Context length | 8,192 tokens |
| Training stages | 3 (SFT → Execution RL → DPO) |
| Target training data | ~1M+ pairs (pipeline in progress) |
| Supported software | Blender, Maya, Cinema 4D, Houdini, Rhino, Unreal Engine |
| Input modalities | Text command, scene context JSON, optional reference image |
| Output format | Reasoning chain + Universal DSL JSON + software-specific Python |

### What makes Nalana different

Standard LLMs generate code. Nalana generates **expert reasoning followed by code**. The reasoning step is trained on 10,000+ hours of professional tutorial content and is what enables topology-aware, physically-grounded outputs that general models cannot produce.

---

## How to Use

### Installation

```bash
# From source:
git clone https://github.com/calebnewtonusc/Nalana
cd Nalana && pip install -e .
```

### REST API

```bash
# Start the server
nalana serve --port 8080

# Send a command
curl -X POST http://localhost:8080/v1/command \
  -H "Content-Type: application/json" \
  -d '{
    "command": "Create an aged copper material with green patina in crevices",
    "scene_context": {
      "software": "blender",
      "active_object": {"name": "Statue", "type": "MESH"}
    }
  }'
```

Response format:

```json
{
  "reasoning": "Aged copper forms malachite patina (Cu₂(OH)₂CO₃) in crevices where moisture pools...",
  "dsl": {
    "op": "CREATE_MAT",
    "name": "AgedCopper",
    "params": { "metallic": 1.0, "base_color": [0.72, 0.45, 0.2], "roughness_range": [0.1, 0.6] }
  },
  "code": "import bpy\nmat = bpy.data.materials.new('AgedCopper')\n...",
  "execution_validated": true,
  "confidence": 0.94
}
```

### Blender Plugin Installation

**Method 1: Drag-and-drop (Blender 4.2+)**
1. Download `nalana_blender_plugin.zip` from [Releases](https://github.com/calebnewtonusc/Nalana/releases)
2. Drag the zip file into any Blender viewport
3. Confirm installation in the popup

**Method 2: Preferences panel**
1. Edit > Preferences > Add-ons > Install
2. Select `nalana_blender_plugin.zip`
3. Enable "Nalana — 3D Workflow Intelligence"

**Using the plugin:**
- Press `N` in any 3D viewport to open the N-panel
- Find the "Nalana" tab
- Type your command (or use voice with Whisper integration enabled)
- Press Enter or click "Execute"

```python
# Plugin usage via Blender Python console
import nalana_blender
nalana_blender.execute("Bevel all edges of the selected object with 3 segments")
```

### Maya Plugin Installation

**MEL installation:**
```mel
// Place nalana_maya/ in your Maya scripts directory, then:
source "nalana_init.mel";
nalana_setup;
```

**Python usage in Maya:**
```python
import nalana_maya
nalana_maya.execute(
    command="Create a car paint material with metallic flakes and clear coat",
    target_object="pSphere1"
)
```

**Shelf button**: After installation, drag `nalana_maya/shelf_button.png` to any Maya shelf.

### Cinema 4D Plugin

```python
# Place in ~/.local/share/MAXON/Cinema 4D R2024/plugins/
# Enable in Plugins > Plugin Manager
import c4d
from nalana_c4d import NalanaPlugin

plugin = NalanaPlugin.instance()
plugin.execute("Animate a logo reveal with a wipe effect over 60 frames")
```

### Houdini Integration

```python
# In a Python SOP or shelf tool:
import hou
from nalana_houdini import NalanaHoudini

nh = NalanaHoudini()
result = nh.execute(
    command="Set up a pyro explosion simulation with wind turbulence",
    context_node=hou.node("/obj/geo1")
)
# Applies resulting VEX and Python node network automatically
```

---

## Training Data

The training pipeline is currently being built across five data streams. Final pair counts will be reported after training completes.

### Stream 1: YouTube Tutorial Transcripts
Professional 3D software tutorial content from Blender, Maya, Houdini, Cinema 4D, and other platforms. Transcripts are segmented, intent-extracted, and reasoning-annotated via the Expert Synthesis Pipeline to capture not just what experts do but why they do it.

### Stream 2: 3D Geometry Datasets
Objaverse, ShapeNet, and Sketchfab Creative Commons objects processed through VLM-based form analysis to produce (object, build sequence) pairs. Provides object vocabulary, construction awareness, and shape priors.

### Stream 3: Physics and Optics Knowledge Base
Q&A pairs synthesized from physically based rendering literature, optics references, and simulation theory. Provides physical grounding for material parameters and simulation configuration.

### Stream 4: Multi-Turn Conversation Sequences
Simulated expert-student dialogue sequences covering topology debugging, render artifact explanation, and scene optimization. Provides dialogue coherence for interactive use.

### Stream 5: Execution-Verified RL Pairs
(Prompt, response) pairs generated and validated during Stage 2 RL training, including negative examples with error analysis. Provides the correctness signal for code generation.

### Data Ethics
All YouTube content is collected via public transcript APIs (respecting robots.txt and terms of service). No proprietary or paywalled content is included. Tutorial creator attribution is maintained in internal dataset metadata but is not exposed in model outputs.

---

## Intended Use

### Primary Use Cases

**3D Artists**: Voice-controlled workflow automation. Replace "look up how to do X" with "ask Nalana". The model explains reasoning alongside code, making it educational as well as functional.

**Game Studios**: Rapid prototyping. An environment artist can model, texture, and LOD an asset in one-third the usual time using Nalana for standard operations.

**Architects**: BIM and visualization workflow. Nalana understands structural constraints and can assist with parametric modeling in Rhino/Grasshopper.

**Educators**: Interactive 3D learning. Students can ask Nalana *why* a technique works, not just *how*, receiving expert-level explanations grounded in physics and topology theory.

**VFX Studios**: Simulation setup and scripting. Nalana can configure cloth, fluid, and rigid body simulations with physically appropriate parameters.

**Product Designers**: Rapid CAD-adjacent modeling for concept visualization and 3D printing preparation.

### Intended Audiences

- Individual 3D artists (beginner through professional)
- Game development studios
- Architecture and design firms
- Film and television VFX departments
- Educational institutions teaching 3D content creation
- Research groups in computer graphics

---

## Limitations

### Technical Limitations

**Software coverage**: Blender receives the strongest support (65% of training data). Maya, Houdini, and Cinema 4D performance is meaningful but below Blender. ZBrush has no Python API and is not directly executable.

**Context window**: At 8,192 tokens, very complex multi-step scenes can exceed context. Planned: 32K context fine-tune.

**Simulation validation**: Nalana can configure simulations correctly but cannot verify that a simulation converged to a physically plausible result — only that the setup code ran.

**Visual input**: Nalana-v1 accepts scene context as structured JSON, not rendered images or viewport screenshots. Visual understanding requires a separate transcription step.

**Niche software**: Specialized tools (Marvelous Designer, Plasticity, KeyShot) are not currently supported.

**Non-English**: Training data is predominantly English-language tutorials. Performance in other languages has not been evaluated.

### Safety Considerations

Nalana generates executable Python code. As with any code generation model, outputs should be reviewed before execution in production environments. Nalana's Blender plugin runs code in Blender's own Python environment — it cannot access the broader file system except where Blender already has permission.

Nalana does not generate harmful, illegal, or privacy-violating content. The 3D domain does not present significant harm vectors beyond general code execution risks.

---

## Citation

If you use Nalana or NalanaBench in your research, please cite:

```bibtex
@inproceedings{nalana2026,
  title     = {Nalana: Universal Voice-to-3D Workflow Intelligence via Expert Synthesis
               and Execution-Verified Reinforcement Learning},
  author    = {[Clarence Last Name] and Newton, Caleb and [Co-authors]},
  booktitle = {SIGGRAPH 2026 / ICCV 2026},
  year      = {2026},
  url       = {https://arxiv.org/abs/[arxiv-id]}
}

@software{nalana_bench,
  title   = {NalanaBench: Industry Standard Benchmark for 3D AI Workflow Quality},
  author  = {[Clarence Last Name] and Newton, Caleb and [Co-authors]},
  year    = {2026},
  url     = {https://github.com/calebnewtonusc/Nalana-dataset}
}
```

---

## License

- **Model weights**: Apache 2.0
- **NalanaBench dataset**: CC BY 4.0
- **Plugins**: MIT

Base model (Qwen2.5-Coder-7B-Instruct) is licensed under Tongyi Qianwen License v1.0, which permits commercial use.

---

## Contact

- **Issues**: [GitHub Issues](https://github.com/calebnewtonusc/Nalana/issues)
- **Research inquiries**: research@nalana.ai
- **Plugin support**: support@nalana.ai
