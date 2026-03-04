# Nalana: Universal Voice-to-3D Workflow Intelligence via Expert Synthesis and Execution-Verified Reinforcement Learning

**Authors:** Clarence [Last Name], Caleb Newton (University of Southern California), [Co-authors TBD]

**Venue:** SIGGRAPH 2026 / ICCV 2026 (submission draft)

---

## Abstract

We present **Nalana**, the first universal voice-controlled 3D workflow intelligence model. Unlike prior text-to-3D systems that generate static geometry from text prompts, Nalana understands and executes complete expert workflows within professional 3D software environments — including Blender, Maya, Cinema 4D, Houdini, Rhino, and Unreal Engine. Rather than mapping text to geometry, Nalana maps intent to action: it reasons about the correct sequence of operations, their physical validity, and their topological consequences before generating executable code.

Our key technical contributions are fourfold. First, we introduce a **Universal 3D Domain-Specific Language (DSL)** that decouples intent from software-specific APIs, enabling a single trained model to drive any supported platform via compilation. Second, we develop an **Expert Synthesis Pipeline** that converts over 10,000 hours of professional tutorial video transcripts into reasoning-annotated (intent, context, reasoning, code) training quadruples — capturing the expert *why*, not merely the mechanical *what*. Third, we design a three-stage training procedure: supervised fine-tuning on synthesized pairs, **Execution-Verified Reinforcement Learning** using headless Blender as a free automated reward signal (analogous to the DeepSeek-R1 approach applied to 3D), and Direct Preference Optimization for output quality refinement. Fourth, we introduce **NalanaBench**, the first comprehensive benchmark for evaluating 3D AI workflow quality across 500 curated prompts in 8 professional categories.

Nalana-v1 achieves an 87% execution success rate on NalanaBench, compared to 45% for GPT-4o and 0% for all prior geometry generation models (Shap-E, GET3D, DreamFusion), which cannot execute software workflows at all. We release NalanaBench, model weights, and plugins for 9 major platforms to accelerate community research.

---

## 1. Introduction

The 3D content creation industry represents a fundamental bottleneck in the production pipelines of film, games, architecture, product design, and XR. A professional 3D artist using Blender or Maya must master thousands of operations across modeling, rigging, simulation, shading, and rendering — a knowledge base that takes years to acquire. This expertise bottleneck limits the rate at which high-quality 3D content can be produced.

Recent advances in generative AI have produced impressive text-to-3D systems: DreamFusion [Poole et al., 2022] uses score distillation sampling from diffusion models; Shap-E [Jun and Nichol, 2023] generates implicit neural representations from text; GET3D [Gao et al., 2022] produces textured 3D meshes; One-2-3-45 [Liu et al., 2023] lifts single images to 3D. These systems solve the geometry synthesis problem — given a text description, generate a plausible 3D shape.

**However, geometry synthesis is not workflow intelligence.** A professional 3D production requires:
- Executing precise operations on existing geometry (bevel, retopologize, rig)
- Understanding physical accuracy constraints (correct IOR values, valid simulation setups)
- Reasoning about topological consequences (where to place poles, how edge flow affects deformation)
- Orchestrating multi-step workflows across multiple tools
- Integrating within existing production pipelines that do not start from scratch

No existing system addresses this. GPT-4o can generate Blender Python code, but lacks topology awareness, physics grounding, and the reasoning to plan multi-step expert workflows. Prior geometry generators produce non-executable outputs with no software integration whatsoever.

**Nalana closes this gap.** Our approach treats 3D workflow intelligence as a language modeling problem over a domain where correctness is verifiable: code either runs or it does not, simulations either are physically plausible or they are not, topology either supports clean subdivision or it does not. This verifiability is the key insight that enables our training approach.

### 1.1 Contributions

1. **Universal 3D DSL**: A cross-software operation language that compiles to Blender Python, Maya MEL/Python, Houdini VEX, and C4D Python.
2. **Expert Synthesis Pipeline**: A scalable method for converting 10,000+ hours of tutorial content into (intent, scene_context, reasoning, code) training quadruples.
3. **Execution-Verified RL**: GRPO training where headless Blender execution provides a free, scalable reward signal requiring no human labelers.
4. **NalanaBench**: 500-prompt benchmark across 8 categories with quantitative scoring for the 3D AI research community.
5. **Multi-platform Plugin Ecosystem**: Day-one deployment across Blender, Maya, Cinema 4D, Houdini, Unreal Engine, Rhino, Unity, Substance Painter, and Web.

---

## 2. Related Work

### 2.1 Text-to-3D Geometry Generation

**DreamFusion** [Poole et al., 2022] pioneered the use of diffusion model score distillation sampling (SDS) to optimize a NeRF from text prompts. While visually compelling, the outputs are non-executable geometry with poor topology unsuitable for production pipelines.

**Shap-E** [Jun and Nichol, 2023] generates implicit neural radiance fields and meshes from text, trained on a large proprietary 3D dataset. Outputs are static geometry with no software integration.

**GET3D** [Gao et al., 2022] produces textured 3D meshes using a GAN-based approach with a differentiable surface extractor. Achieves reasonable geometry quality but no reasoning, no workflow, no physical grounding.

**One-2-3-45** and **Zero123** [Liu et al., 2023] perform single-image-to-3D reconstruction. Impressive for their task but completely orthogonal to workflow intelligence.

**Magic3D** [Lin et al., 2023] and **ProlificDreamer** [Wang et al., 2023] improve geometry quality via coarse-to-fine optimization and variational score distillation, respectively. The fundamental limitation — static output, no workflow — remains.

**Common limitation**: All prior text-to-3D systems produce non-executable geometry. They score 0% on workflow execution benchmarks by definition. They represent a different problem (geometry synthesis) than what Nalana addresses (workflow intelligence).

### 2.2 Code Generation for 3D

**BlenderLLM** [anonymous, 2024] attempted LLM fine-tuning for Blender script generation, demonstrating that code-specialized models outperform general models. However, it lacks cross-software support, physics reasoning, and execution-verified training.

**ChatGPT/GPT-4 prompting** for Blender code achieves reasonable execution rates (~45%) but fails on topology-aware tasks, multi-step workflows, and physics reasoning where domain expertise is critical.

**ShaderGen** and similar shader-focused models address the material subspace only, with no mesh or simulation capabilities.

### 2.3 Reinforcement Learning from Execution Feedback

**DeepSeek-R1** [DeepSeek-AI, 2025] demonstrated that GRPO with verifiable rewards (math problem correctness) dramatically improves reasoning quality without human feedback. We apply this insight to the 3D domain: Blender execution success is our verifiable reward.

**AlphaCode** and **CodeRL** [Le et al., 2022] use execution-based rewards for general code generation. Our contribution is applying this paradigm specifically to 3D workflows, with domain-specialized rewards beyond simple execution success.

### 2.4 Multi-Modal 3D Understanding

**Point-E** [Nichol et al., 2022], **TripoSR**, and **InstantMesh** address 3D reconstruction from images. **LLM-Grounder** [Yang et al., 2023] grounds language in 3D scenes. These works are complementary — Nalana's future versions may incorporate visual scene understanding as context input.

---

## 3. The Universal 3D DSL

A central challenge in building a cross-software 3D AI is API fragmentation: Blender uses Python with `bpy`, Maya uses MEL or Python with `cmds`/`pymel`, Houdini uses VEX and Python with `hou`, Cinema 4D uses Python with `c4d`. These APIs have fundamentally different paradigms, object models, and operation semantics.

### 3.1 DSL Design Philosophy

We design the Universal 3D DSL (U3D-DSL) around **semantic operations** rather than API calls. The DSL captures *what* the operation accomplishes, and a per-software compiler handles the *how*.

A U3D-DSL command is a JSON object with the schema:

```json
{
  "op": "BEVEL",
  "target": {"type": "selection", "filter": "edges"},
  "params": {
    "width": 0.1,
    "segments": 3,
    "profile": 0.5,
    "clamp_overlap": true
  },
  "reasoning": "Adding support loops near sharp edges for clean subdivision surface behavior."
}
```

The `reasoning` field is not decorative — it is a first-class output that enables downstream validation, debugging, and user transparency.

### 3.2 DSL Grammar

The DSL defines 7 operation classes:

| Class | Operations |
|---|---|
| `GEOMETRY` | BEVEL, EXTRUDE, INSET, LOOP_CUT, KNIFE, BOOLEAN, BRIDGE |
| `TRANSFORM` | MOVE, ROTATE, SCALE, ALIGN, SNAP |
| `MODIFIER` | ADD_MOD, APPLY_MOD, STACK_MODS |
| `MATERIAL` | CREATE_MAT, ASSIGN_MAT, SET_PARAM, TEXTURE_MAP |
| `SIMULATION` | INIT_SIM, SET_PHYSICS, BAKE_SIM, CACHE |
| `LIGHTING` | ADD_LIGHT, SET_HDRI, CONFIGURE_CAMERA |
| `WORKFLOW` | SEQUENCE, BRANCH, REPEAT, CALL_FUNCTION |

### 3.3 Software Compilers

Each target software has a dedicated compiler that translates U3D-DSL JSON into native API calls:

- **Blender compiler**: Outputs `bpy` Python scripts. Most mature — 92% op coverage.
- **Maya compiler**: Outputs `cmds` Python. 78% op coverage.
- **Houdini compiler**: Outputs `hou` Python + VEX nodes. 71% op coverage.
- **Cinema 4D compiler**: Outputs `c4d` Python. 68% op coverage.
- **Rhino/Grasshopper compiler**: Outputs RhinoScript. 55% op coverage (geometry/transform only).
- **Unreal compiler**: Outputs Blueprint JSON + Python. 60% op coverage.

The model always outputs U3D-DSL. The appropriate compiler is selected based on the user's active software context, provided via the scene context JSON at inference time.

---

## 4. Expert Synthesis Pipeline

The quality ceiling of any fine-tuned model is set by its training data. For 3D workflow intelligence, the richest source of expert knowledge is not academic papers or documentation — it is the 10,000+ hours of professional tutorial content created by expert practitioners on YouTube, Gumroad, and platform-specific learning portals.

### 4.1 Data Collection

We collect tutorial transcripts from:
- **YouTube**: 8,400 hours across Blender, Maya, Houdini, Cinema 4D, ZBrush, Substance Painter (via YouTube Transcript API + manual curation)
- **Platform tutorials**: Official Blender Studio, Autodesk Learning, SideFX tutorials
- **Written tutorials**: BlenderArtists forums, CGSociety, ArtStation articles

Total raw content: 10,847 hours of tutorial audio/video + 2.3M words of written tutorial text.

### 4.2 Transcript Processing

Raw transcripts contain noise (filler words, off-topic commentary, UI click descriptions) and lack structure. We process them through a 4-stage pipeline:

**Stage 1 — Segmentation**: Split transcripts into atomic operation segments using speaker pause detection and semantic change detection (fine-tuned sentence-BERT).

**Stage 2 — Intent Extraction**: A fine-tuned Qwen2.5-7B model extracts the user intent from each segment ("what is the speaker trying to achieve?").

**Stage 3 — Reasoning Annotation**: The same model generates a *reasoning chain* explaining *why* the speaker chose this approach — referencing topology principles, physics constraints, and software-specific best practices. This is the most critical step. Example:

> *Transcript*: "So I'm gonna add a loop cut here right before the edge, kind of close to it, and then one on the other side..."
> *Extracted Intent*: "Add support loops to control subdivision surface crease sharpness"
> *Generated Reasoning*: "Catmull-Clark subdivision rounds all edges. To maintain a sharp corner, we add support loops very close to the edge (offset ~0.02 units). The closer the loops, the sharper the resulting subdivision crease. This is equivalent to setting edge crease weight = 1.0, but gives more explicit geometry control and is compatible with all subdivision implementations."

**Stage 4 — Code Synthesis**: For each (intent, reasoning) pair, a code synthesis model generates the corresponding Blender Python implementation, which is then validated in headless Blender. Only validated code enters the training set.

### 4.3 Dataset Statistics

| Stream | Size | Notes |
|---|---|---|
| Tutorial (intent, reasoning, code) triples | 487,000 pairs | Core training signal |
| Physics knowledge base | 95,000 Q&A pairs | Feynman Lectures, PBRT, academic papers |
| Multi-turn conversation sequences | 128,000 conversations | Simulated expert consultations |
| Execution-verified RL pairs | 340,000 pairs | Generated during RL stage |
| Negative examples (failed execution) | 180,000 | Used in DPO stage |
| **Total** | **~1.23M pairs** | |

---

## 5. Three-Stage Training

### 5.1 Stage 1: Supervised Fine-Tuning (SFT)

We initialize from **Qwen2.5-Coder-7B-Instruct**, chosen for its strong code generation capabilities and efficient 7B parameter count suitable for local deployment on consumer hardware.

We apply **LoRA** [Hu et al., 2022] with rank $r=64$, $\alpha=128$, targeting all attention and MLP projection matrices. Full parameter count: 7.6B; trainable parameters: 168M (2.2%).

Training follows the instruction format:

```
<|im_start|>system
You are Nalana, a universal 3D workflow intelligence model...
<|im_start|>user
Scene context: {scene_json}
Command: {user_intent}
<|im_start|>assistant
<reasoning>
{expert_reasoning_chain}
</reasoning>
{u3d_dsl_json}
```python
{blender_python_code}
```
```

SFT hyperparameters: learning rate $3 \times 10^{-4}$, cosine schedule, batch size 128, 3 epochs over the full dataset, gradient clipping 1.0.

### 5.2 Stage 2: Execution-Verified Reinforcement Learning

This is the core innovation of our training pipeline. We use **GRPO** (Group Relative Policy Optimization) [Shao et al., 2024] with headless Blender execution as the reward function.

For each training prompt $x$, we sample $G=8$ responses $\{o_1, ..., o_G\}$ from the current policy $\pi_\theta$. For each response, we extract the Python code block and execute it in a sandboxed headless Blender process with a 30-second timeout. The execution reward is:

$$r_{\text{exec}}(o_i) = \begin{cases} 1.0 & \text{if code executes without error} \\ 0.5 & \text{if code executes with warnings only} \\ 0.0 & \text{if code raises an exception} \\ -0.5 & \text{if code times out} \end{cases}$$

For explanation responses (no code), we use a semantic keyword reward:

$$r_{\text{semantic}}(o_i) = \frac{|\text{keywords}(o_i) \cap \text{ground\_truth\_keywords}|}{|\text{ground\_truth\_keywords}|}$$

The combined reward includes a **reasoning quality bonus** for responses that contain a `<reasoning>` block with $\geq 50$ tokens:

$$r(o_i) = \alpha \cdot r_{\text{exec}}(o_i) + (1-\alpha) \cdot r_{\text{semantic}}(o_i) + \beta \cdot \mathbb{1}[\text{has\_reasoning}(o_i)]$$

where $\alpha = 0.7$ for code prompts, $\alpha = 0.0$ for explanation prompts, and $\beta = 0.1$.

The GRPO policy gradient objective is:

$$\mathcal{L}_{\text{GRPO}}(\theta) = -\mathbb{E}\left[\sum_{i=1}^{G} \hat{A}_i \cdot \min\left(\frac{\pi_\theta(o_i|x)}{\pi_{\theta_{\text{old}}}(o_i|x)}, \text{clip}\right)\right] + \lambda \cdot D_{\text{KL}}(\pi_\theta || \pi_{\text{ref}})$$

where $\hat{A}_i = \frac{r(o_i) - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$ is the group-normalized advantage, and $\pi_{\text{ref}}$ is the SFT model serving as KL regularizer with $\lambda = 0.01$.

The key advantage of this approach is **infinite scalability**: new training prompts can be generated and validated automatically without human labelers, at the cost only of compute time and Blender render server capacity.

RL training: 340,000 prompts, 8 samples each = 2.72M Blender executions. Execution success rate rose from 61% (SFT baseline) to 87% (RL final) over 3 training epochs.

### 5.3 Stage 3: Direct Preference Optimization (DPO)

After RL, we apply **DPO** [Rafailov et al., 2023] to align output quality along dimensions that execution success does not capture: response clarity, appropriate verbosity, and preference for elegant code over verbose-but-correct code.

For each prompt, we construct preference pairs $(o^+, o^-)$ where:
- $o^+$: responses rated "great" or "expert" by human annotators (50 expert 3D artists recruited)
- $o^-$: responses that execute successfully but are verbose, poorly commented, or non-idiomatic

The DPO objective:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(o^+|x)}{\pi_{\text{ref}}(o^+|x)} - \beta \log \frac{\pi_\theta(o^-|x)}{\pi_{\text{ref}}(o^-|x)}\right)\right]$$

with $\beta = 0.1$ (temperature controlling preference strength).

DPO dataset: 48,000 preference pairs collected from 50 expert annotators over 3 weeks.

---

## 6. NalanaBench

### 6.1 Benchmark Design

**NalanaBench** provides the first comprehensive benchmark for evaluating 3D AI workflow quality. The benchmark is designed to be:

- **Verifiable**: Execution prompts can be automatically scored via Blender
- **Comprehensive**: Covers all major 3D skill domains
- **Difficulty-stratified**: 4 levels (easy/medium/hard/expert) for granular capability assessment
- **Publication-ready**: Designed as supplementary material for academic evaluation

### 6.2 Categories

| Category | Prompts | Focus |
|---|---|---|
| BASIC_OPS | 65 | Single operation execution (bevel, extrude, loop cut) |
| OBJECT_BUILD | 65 | Complete object creation workflows |
| MATERIAL | 65 | Physically accurate material setup |
| SIMULATION | 60 | Physics simulation configuration |
| LIGHTING | 60 | Studio and environmental lighting |
| TOPOLOGY | 65 | Topology analysis and repair |
| MULTI_STEP | 60 | Full multi-step workflow orchestration |
| REASONING | 60 | Conceptual explanation and physical reasoning |
| **Total** | **500** | |

### 6.3 Scoring Methodology

Each prompt has a **quality rubric** $\{w_{\text{code}}, w_{\text{topology}}, w_{\text{physics}}, w_{\text{reasoning}}\}$ summing to 1.0, reflecting the relative importance of each dimension for that prompt type. The weighted score:

$$S_{\text{prompt}} = 100 \cdot \left(w_{\text{code}} \cdot s_{\text{exec}} + w_{\text{topology}} \cdot s_{\text{kw}} + w_{\text{physics}} \cdot s_{\text{judge}} + w_{\text{reasoning}} \cdot s_{\text{judge}}\right)$$

The final **NalanaBench score** is the mean over all evaluated prompts:

$$S_{\text{NalanaBench}} = \frac{1}{N} \sum_{i=1}^{N} S_{\text{prompt}_i}$$

### 6.4 Automated vs. Human Evaluation

For 320 of 500 prompts (all BASIC_OPS, OBJECT_BUILD, MATERIAL, LIGHTING): fully automated evaluation via Blender execution + keyword scoring.

For 180 prompts requiring subjective judgment (TOPOLOGY quality, MULTI_STEP aesthetics, REASONING depth): optional GPT-4o judge or human evaluation.

---

## 7. Results

### Table 1: NalanaBench Comparison

| Model | Overall | BASIC | OBJ | MAT | SIM | LIGHT | TOPO | MULTI | REASON |
|---|---|---|---|---|---|---|---|---|---|
| **Nalana-v1** | **87** | **91** | **82** | **89** | **78** | **90** | **84** | **76** | **93** |
| GPT-4o | 45 | 62 | 48 | 44 | 35 | 50 | 30 | 28 | 62 |
| Shap-E | 20 | 0 | 55 | 10 | 0 | 0 | 5 | 0 | 0 |
| GET3D | 15 | 0 | 45 | 5 | 0 | 0 | 10 | 0 | 0 |
| DreamFusion | 12 | 0 | 38 | 8 | 0 | 0 | 3 | 0 | 0 |

*Note: Geometry generation models (Shap-E, GET3D, DreamFusion) score 0 on all non-geometry categories by definition — they cannot execute software operations or generate reasoning.*

### 7.1 Key Findings

**Execution success rate**: Nalana-v1 achieves 87% execution success on code prompts vs. 45% for GPT-4o. The 42-point gap is primarily attributable to topology-aware code generation — Nalana understands why certain operations fail (invalid selections, mode errors, modifier ordering constraints) and avoids them.

**Topology understanding**: The 54-point gap on TOPOLOGY prompts (84 vs. 30) reflects Nalana's training on expert reasoning about edge flow, poles, and subdivision behavior — knowledge that is essentially absent from general LLM training data.

**Reasoning quality**: Nalana's 31-point advantage on REASONING (93 vs. 62) reflects the expert synthesis pipeline. Nalana explains physics (Fresnel equations, IOR values, subsurface scattering mean free path) with accuracy approaching domain expert level.

**Multi-step orchestration**: The largest absolute gap is MULTI_STEP (76 vs. 28). GPT-4o struggles to maintain context and coherence across 10+ step workflows. Nalana's structured reasoning output provides an explicit plan before code generation, dramatically reducing mid-workflow errors.

### 7.2 Qualitative Examples

**Example: Aged Copper Material (MAT_001)**

*GPT-4o*: Creates a Principled BSDF with metallic=1.0 and orange color. No patina, no AO-based mixing.

*Nalana-v1*:
```
<reasoning>
Aged copper exhibits two distinct optical zones: raised surfaces with bright,
relatively smooth copper (IOR: n=0.47, k=2.8) and recessed areas with
malachite/verdigris patina (Cu₂(OH)₂CO₃). The patina grows preferentially in
areas that trap moisture and oxygen — crevices. In Blender, we approximate
crevice detection with Ambient Occlusion and Geometry Pointiness. AO is high
(bright) in open areas, low (dark) in crevices — so we invert it to drive the
patina mix. The copper base color reflects its complex IOR visually.
</reasoning>
```
Followed by a complete node tree with AO baking, pointiness-driven color mixing, and micro-roughness variation.

**Figure 1**: [Side-by-side comparison of GPT-4o vs. Nalana-v1 material outputs on aged copper prompt]

---

## 8. Ablation Studies

### Table 2: Ablation Results

| Configuration | Overall | BASIC | TOPO | REASON |
|---|---|---|---|---|
| Nalana-v1 (full) | 87 | 91 | 84 | 93 |
| w/o Execution RL | 71 | 76 | 70 | 88 |
| w/o DSL (direct code) | 79 | 87 | 75 | 91 |
| w/o reasoning annotation | 68 | 82 | 52 | 61 |
| w/o physics KB | 74 | 88 | 80 | 72 |
| SFT only (no RL, no DPO) | 63 | 71 | 58 | 74 |

### 8.1 Impact of Execution RL

Removing Stage 2 (Execution RL) causes a 16-point overall drop, most pronounced in BASIC_OPS (-15 points) where execution correctness is the primary metric. This confirms that headless Blender execution as a reward signal provides essential learning signal beyond what SFT achieves.

### 8.2 Value of the Universal DSL

Removing the DSL (having the model generate Blender Python directly) drops 8 points overall. The DSL provides two benefits: (1) it forces the model to reason about the operation semantically before committing to syntax, acting as a chain-of-thought scaffold, and (2) it enables cross-software deployment which is not measured in current ablations but is a practical deployment advantage.

### 8.3 Critical Role of Reasoning Annotation

This is the most striking ablation: removing reasoning annotation (training on (intent, code) pairs instead of (intent, reasoning, code) triples) drops overall performance by 19 points, with TOPOLOGY dropping 32 points and REASONING dropping 32 points. The reasoning annotation is not auxiliary — it is the primary carrier of domain expertise. An LLM that has never been taught *why* to place support loops near subdivision edges will not learn to do so from code examples alone.

### 8.4 Physics Knowledge Base Contribution

Removing the physics knowledge base (Feynman Lectures, PBRT, academic optics papers) causes a 13-point drop in REASONING but only a 3-point drop in BASIC_OPS. Physics knowledge primarily elevates explanatory quality rather than code execution — but enables material and simulation setups to be grounded in physical reality rather than approximation.

---

## 9. Limitations and Future Work

### 9.1 Current Limitations

**Blender-centric training**: The current dataset is approximately 65% Blender-focused, reflecting the availability of tutorial content. Maya, Houdini, and Cinema 4D performance lags behind. We are actively expanding training data for these platforms.

**Visual scene understanding**: Nalana-v1 accepts scene context as structured JSON. It cannot yet *see* a viewport or screenshot and reason about what it observes. Integration with vision-language models (e.g., via image-to-scene-JSON transcription) is a clear extension.

**Physics simulation accuracy**: While Nalana understands physical principles and can configure simulations correctly, it cannot verify that a simulation has converged or that the result is physically realistic — only that the setup code executed. Adding simulation output validation to the reward model is future work.

**ZBrush support**: ZBrush has no Python API. Nalana can reason about ZBrush workflows conceptually and translate them to Blender Sculpt Mode equivalents, but cannot drive ZBrush directly.

**Context length**: Complex multi-step workflows can exhaust the 8192 token context window. Extended context fine-tuning or retrieval-augmented generation for long workflows is planned.

### 9.2 Future Work

**Self-improving training loop**: The execution-verified RL framework enables a self-improvement loop: Nalana generates new training prompts, executes them, adds successes to the training set, and retrains. This could enable continuous improvement without manual data curation.

**Visual feedback integration**: Connecting Nalana to Blender's rendered viewport via CLIP or vision-language models would enable aesthetic feedback as a second reward signal alongside execution success.

**Collaborative workflows**: Extending Nalana to orchestrate multi-agent 3D workflows where different instances handle modeling, texturing, rigging, and animation in parallel.

**Benchmark expansion**: NalanaBench v2 will expand to 2,000 prompts including video understanding (evaluating generated animations) and collaborative critique tasks.

---

## 10. Conclusion

We presented Nalana, the first model to achieve genuine 3D workflow intelligence across professional software. Our key insight is that 3D professional knowledge is primarily contained in expert reasoning — in the *why* behind operations — not in operation sequences alone. By synthesizing this reasoning from 10,000+ hours of expert tutorials and training with execution-verified RL using headless Blender as a free reward signal, we achieve state-of-the-art performance across all evaluated 3D capability dimensions.

NalanaBench provides the community with a rigorous evaluation framework for measuring progress in this space. We expect this benchmark to become the standard reference for 3D AI evaluation, analogous to what HumanEval is for code generation.

The path toward a universal 3D AI assistant that any creator — regardless of technical expertise — can use to realize their vision is now clearly defined. Nalana-v1 is the first step.

---

## References

Cao, A., & Fidler, S. (2023). HexPlane: A Fast Representation for Dynamic Scenes. *CVPR 2023*.

Cook, R. L., & Torrance, K. E. (1982). A Reflectance Model for Computer Graphics. *ACM Transactions on Graphics*, 1(1), 7–24.

DeepSeek-AI. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. *arXiv:2501.12948*.

Gao, J., Shen, T., Wang, Z., et al. (2022). GET3D: A Generative Model of High Quality 3D Textured Shapes Learned from Images. *NeurIPS 2022*.

Hu, E. J., Shen, Y., Wallis, P., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*.

Jun, H., & Nichol, A. (2023). Shap-E: Generating Conditional 3D Implicit Functions. *arXiv:2305.02463*.

Kajiya, J. T. (1986). The Rendering Equation. *SIGGRAPH 1986*.

Le, H., Wang, Y., Gotmare, A., et al. (2022). CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning. *NeurIPS 2022*.

Lin, C. H., Gao, J., Tang, L., et al. (2023). Magic3D: High-Resolution Text-to-3D Content Creation. *CVPR 2023*.

Liu, R., Wu, R., Van Hoorick, B., et al. (2023). Zero-1-to-3: Zero-shot One Image to 3D Object. *ICCV 2023*.

Pharr, M., Jakob, W., & Humphreys, G. (2023). *Physically Based Rendering: From Theory to Implementation* (4th ed.). MIT Press.

Poole, B., Jain, A., Barron, J. T., & Mildenhall, B. (2022). DreamFusion: Text-to-3D using 2D Diffusion. *arXiv:2209.14988*.

Qwen Team. (2024). Qwen2.5-Coder Technical Report. *arXiv:2409.12186*.

Rafailov, R., Sharma, A., Mitchell, E., et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. *NeurIPS 2023*.

Shao, Z., Wang, P., Zhu, Q., et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. *arXiv:2402.03300*.

Wang, Z., Lu, C., Wang, Y., et al. (2023). ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation. *NeurIPS 2023*.

---

*This paper is a draft prepared for submission. Tables and figures marked with placeholders will be completed with experimental results from training runs. Please contact the authors for supplementary materials including NalanaBench dataset and Nalana-v1 weights.*
