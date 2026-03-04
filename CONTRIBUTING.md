# Contributing to Nalana

Thank you for contributing. Nalana gets better through community contributions of training data, plugins, evaluations, and integrations. This guide explains how.

---

## What We Need Most

In rough priority order:

1. **Training pairs** — (voice command, blender_python) pairs that run clean in Blender
2. **NalanaBench test cases** — prompts with expected behavior for evaluating new models
3. **Software plugins** — integrations for platforms not yet covered
4. **Physics / domain knowledge** — expert Q&A pairs for materials, simulation, architecture, CAD
5. **Bug reports** — wrong outputs, failed executions, reasoning errors

---

## Training Data Contributions

### Format

All training pairs must follow this JSON structure:

```json
{
  "voice_command": "bevel the selected edges with 3 segments and 0.05 width",
  "scene_context": {
    "software": "blender",
    "mode": "EDIT",
    "active_object": {"name": "Cube", "type": "MESH"}
  },
  "reasoning": "Beveling edges with support loops close to the edge prevents subdivision surface pinching. 3 segments creates two tight support loops flanking the original edge, producing a controlled bevel that stays sharp under Catmull-Clark subdivision.",
  "blender_python": "import bpy\nbpy.ops.object.mode_set(mode='EDIT')\nbpy.ops.mesh.select_all(action='SELECT')\nbpy.ops.mesh.bevel(offset=0.05, segments=3, affect='EDGES')\nbpy.ops.object.mode_set(mode='OBJECT')",
  "universal_dsl": {
    "op": "BEVEL",
    "target": {"type": "selection", "filter": "edges"},
    "params": {"width": 0.05, "segments": 3}
  },
  "task_type": "EXECUTE",
  "quality_tier": "gold",
  "source": "community"
}
```

Required fields: `voice_command`, `blender_python`, `task_type`

Optional but valuable: `reasoning`, `scene_context`, `universal_dsl`

### Validation Requirement

**Every training pair must pass headless Blender execution before being accepted.**

Run before submitting:

```bash
python validation/validate_blender.py --input path/to/your/pairs.jsonl
```

Any pair that fails execution will be rejected. This is non-negotiable — bad data degrades the model.

### Where to Submit

1. Place your validated `.jsonl` file in `data/community/`
2. Open a Pull Request with title `[data] <brief description>`
3. Include the output of `validate_blender.py` in the PR description

### Quality Tiers

| Tier | Weight | Requirements |
|---|---|---|
| Gold | 5x | Handcrafted with expert reasoning + execution validated |
| Silver | 2x | Execution validated, minimal reasoning |
| Bronze | 1x | Seed pair, no reasoning, needs validation |

Gold contributions have 5x the impact on training. Take time to write good reasoning.

### Software Coverage We Need Most

In priority order:
- **Houdini** (VEX + Python, all simulation types)
- **Maya** (rigging, animation, MEL + Python)
- **Cinema 4D** (MoGraph, effectors, animation)
- **ZBrush** (workflow explanations translated to Blender Sculpt equivalents)
- **Substance Designer** (node graphs, material authoring)
- **Fusion 360** (parametric modeling, assemblies)
- **SketchUp** (architecture, visualization)
- **Rhino / Grasshopper** (parametric, NURBS, computational design)

---

## NalanaBench Contributions

### Adding Test Cases

NalanaBench needs more coverage, especially in:
- Non-Blender software (Maya, Houdini, C4D)
- Architecture and engineering workflows
- Animation and rigging
- Advanced simulation (cloth destruction, fluid-rigid coupling)
- Multi-step workflows (>15 steps)

To add a test case:

```bash
python nalana_bench.py --add-test \
  --prompt "Configure a cloth simulation for a flag with wind force and gravity" \
  --category SIMULATION \
  --difficulty hard \
  --expected-behavior "Should configure cloth modifier, add wind effector, set material parameters"
```

Or edit `nalana_bench.py` directly and open a PR.

### Reporting Model Failures

If you find a prompt where Nalana gives a wrong, unsafe, or nonsensical response:
1. Open a GitHub Issue with tag `[benchmark]`
2. Include: the exact command, your scene context, what Nalana returned, what the correct behavior should be
3. We will add it to NalanaBench and use it to improve training

---

## Plugin Contributions

### New Software Integrations

Each plugin is a thin compiler from Universal 3D DSL to native software API. Reference implementation: [plugins/blender/__init__.py](plugins/blender/__init__.py)

A minimal plugin must implement:

```python
class NalanaPlugin:
    def __init__(self, api_url: str, api_key: str):
        ...

    def send_command(self, command: str, scene_context: dict) -> dict:
        """Send voice command to Nalana API and return result."""
        ...

    def execute_result(self, result: dict) -> bool:
        """Execute the returned DSL/code in the host software."""
        ...

    def get_scene_context(self) -> dict:
        """Extract current scene state for context."""
        ...
```

See [plugins/PLUGIN_SPEC.md](plugins/PLUGIN_SPEC.md) for the full plugin interface specification.

### Plugin PR Checklist

- [ ] Implements the full plugin interface (send_command, execute_result, get_scene_context)
- [ ] Has error handling — never crashes the host application
- [ ] Tested in the actual software (not just unit tests)
- [ ] Includes install instructions in the plugin's own README
- [ ] No hardcoded API keys or URLs

---

## Domain Knowledge Contributions

### Physics and Materials

We want more (question, answer) pairs grounded in real physics. Format:

```json
{
  "question": "Why does gold appear warm/yellow rather than the white-silver color of most metals?",
  "answer": "Gold's characteristic color arises from relativistic contraction of its 5d and 6s electron orbitals, which moves an interband electronic transition into the visible range (~450-500nm). Photons at blue wavelengths are absorbed by this transition, and the complementary warm yellow-orange color is reflected. In PBR terms: gold has a strongly colored F0 value of approximately (1.0, 0.78, 0.34) — unlike silver or aluminum which have nearly achromatic (white) F0 values.",
  "domain": "physics_optics",
  "tags": ["materials", "metals", "PBR", "IOR", "F0"]
}
```

Domains we want: `physics_optics`, `physics_mechanics`, `physics_fluid`, `topology_theory`, `design_principles`, `architecture_theory`, `engineering_manufacturing`, `animation_principles`, `color_theory`, `lighting_theory`

### Architecture and Engineering

Contributions from licensed architects and structural engineers are especially valuable for v2 features. If you have professional knowledge in:
- Building code compliance (IBC, Eurocodes, local codes)
- Structural analysis
- DFM (Design for Manufacturability)
- HVAC, MEP systems

Please reach out at research@nalana.ai — we want to work with domain experts.

---

## Code Contributions

### Setup

```bash
git clone https://github.com/calebnewtonusc/nalana.git
cd nalana
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill in your keys
```

### Running Checks

```bash
bash scripts/check_env.sh
python validation/validate.py --sample 100    # Validate 100 random data pairs
python validation/validate_blender.py --dry-run  # Check Blender is accessible
```

### Code Standards

- Python 3.11+. No type: ignore comments.
- Docstrings on all public functions.
- No hardcoded paths or API keys anywhere. Use environment variables.
- All new data synthesis prompts go in `prompts.py` — centralized, versioned, documented.
- All new training pairs must pass `validate_blender.py` before being accepted.
- No `print()` in production code — use Python `logging` module.

### Pull Request Process

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature-name`
3. Make changes. Write tests if applicable.
4. Run `bash scripts/check_env.sh` — must pass.
5. Open PR with a clear title and description
6. Link any related Issues

---

## Issue Reports

### Good Bug Report

```
**What I did:** Said "add a metal material" to Blender with the Nalana plugin

**Scene context:** Single cube in Object Mode, no materials assigned

**Expected:** A Principled BSDF material with metallic=1.0 assigned to the cube

**Got:** Python error — bpy.ops.MATERIAL.new() not found

**Nalana version:** v1.0 (Docker image sha: abc123)

**Blender version:** 4.1.0
```

### Tags to Use

| Tag | Use for |
|---|---|
| `[bug]` | Wrong output, crash, error |
| `[benchmark]` | Model failure to add to NalanaBench |
| `[data]` | Training data issues |
| `[plugin]` | Plugin-specific bugs |
| `[docs]` | Documentation errors |
| `[feature]` | New capability requests |
| `[research]` | Academic / paper discussion |

---

## Community

- **GitHub Issues**: [github.com/calebnewtonusc/nalana/issues](https://github.com/calebnewtonusc/nalana/issues) — bugs, features, discussion
- **Research inquiries**: research@nalana.ai
- **Plugin support**: support@nalana.ai

---

*Every training pair, every test case, every bug report makes Nalana better. The model is only as good as the knowledge we put into it.*
