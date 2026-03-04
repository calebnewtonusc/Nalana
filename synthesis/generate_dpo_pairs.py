"""
generate_dpo_pairs.py - Self-improving DPO pair generation

Uses the CURRENT Nalana model to generate DPO preference pairs automatically.

Algorithm:
  1. Take a prompt from the SFT training set
  2. Generate 4 responses at temperature=1.0 (diverse)
  3. Execute all 4 in headless Blender
  4. Rank by: execution_score * 0.7 + reasoning_quality * 0.3
  5. Pair the best (chosen) vs. worst (rejected)
  6. Save as DPO pairs for the next round of train_dpo.py

Self-improvement loop:
  train.py (SFT)
    → train_rl.py (RL, execution reward)
      → train_dpo.py (DPO, conversation quality)
        → generate_dpo_pairs.py (new pairs from improved model)
          → train_dpo.py (DPO round 2, with better pairs)
            → ... (iterate until performance plateaus)

Why this works:
  As the model improves, the gap between its best and worst outputs grows —
  better chosen responses, worse rejected responses = stronger training signal.
  This is the core insight of iterative DPO / self-play.

Ranking formula:
  score = execution_score * EXEC_WEIGHT + reasoning_quality * REASONING_WEIGHT
  where:
    execution_score:    0.0 / 0.5 / 1.0 from headless Blender
    reasoning_quality:  0.0 - 1.0 heuristic (see score_reasoning)
    EXEC_WEIGHT:        0.7  (code correctness is primary signal)
    REASONING_WEIGHT:   0.3  (explanation quality is secondary)

Also generates synthetic conversation quality pairs using Claude (optional).
These teach Nalana WHEN to ask questions vs. just execute.

Usage:
  # Basic: generate execution-based pairs from current model
  python generate_dpo_pairs.py --model checkpoints/nalana-rl/final

  # Full: execution pairs + conversation quality pairs via Claude
  python generate_dpo_pairs.py \\
    --model checkpoints/nalana-rl/final \\
    --n-prompts 500 \\
    --n-candidates 4 \\
    --output-dir data/dpo \\
    --gen-conversation-pairs \\
    --anthropic-api-key $ANTHROPIC_API_KEY

  # Resume interrupted run
  python generate_dpo_pairs.py --model checkpoints/nalana-dpo/final \\
    --resume --output-dir data/dpo/round2
"""

import argparse
import json
import logging
import os
import sys
import random
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("nalana-dpo-gen")

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

# Ranking weights
EXEC_WEIGHT      = 0.7
REASONING_WEIGHT = 0.3

# ─── Reasoning quality heuristics ────────────────────────────────────────────

# Keywords that indicate design/physics reasoning in a response
REASONING_KEYWORDS = {
    # Physics / material
    "ior", "index of refraction", "roughness", "metallic", "subsurface",
    "scattering", "fresnel", "pbr", "physically based",
    # Topology
    "topology", "edge loop", "edge flow", "pole", "ngon", "subdivision",
    "quad", "triangulate", "manifold", "non-manifold",
    # Form language
    "primary shape", "secondary shape", "silhouette", "form language",
    "chamfer", "bevel", "fillet", "radius",
    # Lighting
    "three-point", "hdri", "area light", "rim light", "fill light",
    "key light", "cinematic", "exposure", "color temperature",
    # Workflow reasoning
    "because", "this will", "the reason", "in order to", "which means",
    "note that", "be aware", "alternatively", "consider",
}

QUESTION_INDICATORS = {
    "?", "could you clarify", "what style", "how many", "what size",
    "would you like", "do you want", "can you specify", "what do you mean",
}

VAGUE_QUESTION_PENALTY_PHRASES = {
    "what exactly", "can you be more specific", "i need more information",
    "could you elaborate", "what do you mean by",
}


def score_reasoning(text: str) -> float:
    """
    Heuristic scoring of response reasoning quality.

    Rewards:
      - Uses domain-specific terminology (IOR, topology, etc.)
      - Asks ONE focused clarifying question (not zero, not five)
      - Gives rationale before or after code

    Penalizes:
      - Raw code dump with no explanation
      - Multiple vague questions
      - Very short responses (likely incomplete)

    Returns a score in [0.0, 1.0].
    """
    text_lower = text.lower()
    score = 0.0

    # Count reasoning keywords
    keyword_hits = sum(1 for kw in REASONING_KEYWORDS if kw in text_lower)
    score += min(keyword_hits * 0.08, 0.4)  # cap at 0.4 for keyword density

    # Check for exactly one clarifying question (good behavior)
    question_count = text.count("?")
    if question_count == 1:
        score += 0.2   # One focused question = ideal
    elif question_count == 0:
        score += 0.05  # No question = just executes (acceptable)
    elif question_count == 2:
        score += 0.05  # Two questions = slightly verbose
    else:
        score -= 0.1   # Many questions = bad UX (interrogating the user)

    # Code presence (good — we want code)
    if "bpy." in text or "import bpy" in text:
        score += 0.2

    # Has explanation around the code (not just raw code)
    has_text_around_code = (
        "```" in text and
        len(text.split("```")[0].strip()) > 20  # text before code block
    )
    if has_text_around_code:
        score += 0.1

    # Penalize vague questions
    vague_count = sum(1 for phrase in VAGUE_QUESTION_PENALTY_PHRASES if phrase in text_lower)
    score -= vague_count * 0.05

    # Penalize very short responses (likely truncated or unhelpful)
    if len(text.strip()) < 50:
        score -= 0.3

    return max(0.0, min(1.0, score))


def rank_candidates(
    candidates: list[str],
    execution_scores: list[float],
) -> list[tuple[int, float]]:
    """
    Rank candidates by combined execution + reasoning quality score.

    Returns list of (index, combined_score) sorted best-first.
    """
    ranked = []
    for i, (text, exec_score) in enumerate(zip(candidates, execution_scores)):
        reasoning_score = score_reasoning(text)
        combined = exec_score * EXEC_WEIGHT + reasoning_score * REASONING_WEIGHT
        ranked.append((i, combined))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


# ─── Conversation quality pair generation via Claude ─────────────────────────

CONVERSATION_PAIRS_SYSTEM = """You are a dataset engineer building training data for Nalana — a voice-to-Blender AI.

You will generate DPO (Direct Preference Optimization) pairs that teach Nalana how to have GREAT conversations with 3D artists.

For each Blender task prompt I give you, generate TWO assistant responses:
1. CHOSEN: The ideal response — shows expertise, gives ONE focused clarifying question if needed, explains the physics/design reasoning, then gives clean Blender Python
2. REJECTED: A bad response — either floods the user with 5 vague questions, or dumps raw code with zero explanation, or gives a wrong explanation

Output ONLY valid JSON in this exact format:
{
  "prompt": "<the user's request>",
  "chosen": "<ideal response>",
  "rejected": "<bad response>",
  "source": "synthetic_conv",
  "rejection_type": "too_many_questions|no_explanation|wrong_physics|code_dump"
}

Rules for CHOSEN responses:
- If the request is clear (e.g. "add a cube"), just execute it with brief expert commentary
- If ambiguous (e.g. "make it look realistic"), ask ONE specific question: "What material? Glass, metal, or plastic?"
- Always include: why you're doing what you're doing (topology reason, physics reason, etc.)
- Code should be clean bpy.ops calls with a brief comment

Rules for REJECTED responses:
- Vary the failure mode: sometimes flood with questions, sometimes give code-only dump, sometimes explain wrong physics
- Make it realistic — something a naive model would actually generate"""

CONVERSATION_PROMPTS = [
    "Make this object look like brushed aluminum.",
    "Add a subdivision surface to smooth it out.",
    "Create a glass wine glass.",
    "Make the material look like wet skin.",
    "Add a rim light to make the silhouette pop.",
    "Extrude these faces outward by 0.3 meters.",
    "Make the water look more realistic.",
    "Add a camera aimed at the object from 45 degrees.",
    "Give this the topology of a production-ready character mesh.",
    "Create a procedural wood texture.",
    "Make the render look cinematic.",
    "Add a bevel to these hard edges.",
    "Create a simple rigging setup for this character arm.",
    "Make this look like carbon fiber.",
    "Set up a three-point lighting rig.",
    "Give this object a worn leather material.",
    "Create a normal map bake setup.",
    "Make the fur simulation look realistic.",
    "Add a cloth simulation to this plane.",
    "Create an HDRI-based outdoor lighting setup.",
]


def generate_conversation_pairs(
    api_key: str,
    n_pairs: int = 100,
    output_file: Path = None,
) -> list[dict]:
    """
    Use Claude to generate synthetic conversation quality DPO pairs.

    These teach Nalana the difference between:
      - One focused clarifying question (good)
      - Flooding with 5 vague questions (bad)
      - Physics explanation + code (good)
      - Raw code dump (bad)
    """
    if not HAS_ANTHROPIC:
        log.error("anthropic package not installed. Run: pip install anthropic")
        return []

    client = anthropic.Anthropic(api_key=api_key)
    pairs = []
    prompt_pool = CONVERSATION_PROMPTS * (n_pairs // len(CONVERSATION_PROMPTS) + 1)
    random.shuffle(prompt_pool)

    pbar = tqdm(total=n_pairs, desc="Generating conv pairs") if HAS_TQDM else None

    for prompt_text in prompt_pool[:n_pairs]:
        try:
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=CONVERSATION_PAIRS_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": f"Generate a DPO pair for this Blender request:\n\"{prompt_text}\""
                }],
            )

            content = msg.content[0].text.strip()

            # Try to extract JSON from response
            # Claude might wrap it in ```json blocks
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()

            pair = json.loads(content)
            if all(k in pair for k in ("prompt", "chosen", "rejected")):
                pair["source"] = "synthetic_conv"
                pairs.append(pair)

                if output_file:
                    with output_file.open("a") as f:
                        f.write(json.dumps(pair) + "\n")

        except (json.JSONDecodeError, KeyError, Exception) as e:
            log.debug(f"Skipped pair (parse error): {e}")
            continue

        if pbar:
            pbar.update(1)

        # Rate limit: Claude API has limits
        time.sleep(0.1)

    if pbar:
        pbar.close()

    log.info(f"Generated {len(pairs):,} synthetic conversation pairs")
    return pairs


# ─── Physics reasoning pair generation ───────────────────────────────────────

PHYSICS_PAIRS = [
    {
        "prompt": "Make this material look like glass.",
        "chosen": (
            "For realistic glass, the key parameters are IOR (Index of Refraction) and roughness. "
            "Glass has an IOR of ~1.45-1.52 — this controls how much the material bends light. "
            "Setting roughness to 0.0 gives perfectly clear glass; add a tiny amount (0.02) for "
            "slight surface imperfections on older glass.\n\n"
            "```python\n"
            "import bpy\n"
            "obj = bpy.context.active_object\n"
            "mat = bpy.data.materials.new(name='Glass')\n"
            "mat.use_nodes = True\n"
            "nodes = mat.node_tree.nodes\n"
            "nodes.clear()\n"
            "bsdf = nodes.new('ShaderNodeBsdfGlass')\n"
            "bsdf.inputs['IOR'].default_value = 1.45\n"
            "bsdf.inputs['Roughness'].default_value = 0.02\n"
            "output = nodes.new('ShaderNodeOutputMaterial')\n"
            "mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])\n"
            "obj.data.materials.append(mat)\n"
            "```\n\n"
            "Enable Cycles with GPU for accurate caustics."
        ),
        "rejected": (
            "Here's the glass material:\n"
            "```python\n"
            "bpy.ops.material.new()\n"
            "```"
        ),
        "source": "physics",
        "rejection_type": "code_dump",
    },
    {
        "prompt": "What roughness value should I use for polished metal?",
        "chosen": (
            "Polished metal roughness depends on how perfect the surface is:\n"
            "- Mirror polish (chrome, fresh aluminum): 0.0 - 0.05\n"
            "- Brushed metal (directional scratches): 0.2 - 0.4\n"
            "- Worn/matte metal: 0.6 - 0.8\n\n"
            "For polished aluminum specifically, use roughness=0.05 and metallic=1.0. "
            "The IOR for aluminum is ~1.1 (but in Cycles, metallic=1.0 handles this automatically). "
            "Add a subtle Noise texture (scale=50, strength=0.02) to the roughness to break up "
            "the perfect reflection and make it look real.\n\n"
            "```python\n"
            "mat = bpy.context.active_object.active_material\n"
            "bsdf = mat.node_tree.nodes['Principled BSDF']\n"
            "bsdf.inputs['Roughness'].default_value = 0.05\n"
            "bsdf.inputs['Metallic'].default_value = 1.0\n"
            "```"
        ),
        "rejected": (
            "What kind of metal? What's the object? What renderer are you using? "
            "What is the intended use case? Is this for games or film? "
            "What style are you going for? Can you be more specific?"
        ),
        "source": "physics",
        "rejection_type": "too_many_questions",
    },
    {
        "prompt": "The skin material looks waxy. How do I fix it?",
        "chosen": (
            "Waxy skin is caused by missing subsurface scattering (SSS). Real skin transmits "
            "light through its layers — red light penetrates deepest (veins), green is mid-range, "
            "blue bounces near the surface. Without SSS, skin looks like painted plastic.\n\n"
            "In the Principled BSDF, increase Subsurface to 0.1-0.2 and set the "
            "Subsurface Radius to (1.0, 0.2, 0.1) — red channel highest because it penetrates deepest.\n\n"
            "```python\n"
            "bsdf = bpy.context.active_object.active_material.node_tree.nodes['Principled BSDF']\n"
            "bsdf.inputs['Subsurface'].default_value = 0.15\n"
            "bsdf.inputs['Subsurface Radius'].default_value = (1.0, 0.2, 0.1)\n"
            "bsdf.inputs['Subsurface Color'].default_value = (0.9, 0.6, 0.5, 1.0)\n"
            "```\n\n"
            "Also check: make sure your mesh normals are correct — inverted normals can "
            "make SSS look wrong."
        ),
        "rejected": (
            "Try increasing the subsurface value.\n"
            "```python\n"
            "bsdf.inputs['Subsurface'].default_value = 0.5\n"
            "```"
        ),
        "source": "physics",
        "rejection_type": "no_explanation",
    },
]


# ─── Execution-based pair generation ─────────────────────────────────────────

@torch.no_grad()
def generate_execution_pairs(
    model,
    tokenizer,
    prompts: list[dict],
    reward_fn,
    n_candidates: int,
    max_new_tokens: int,
    temperature: float,
    output_file: Path,
    resume_offset: int = 0,
    device: str = "cuda",
) -> list[dict]:
    """
    For each prompt:
      1. Generate n_candidates completions
      2. Execute all in Blender
      3. Rank by execution_score * 0.7 + reasoning_quality * 0.3
      4. Pair best (chosen) vs. worst (rejected) if gap > MIN_GAP
      5. Save pairs to JSONL

    We only create pairs where best - worst > MIN_GAP to ensure
    the preference signal is meaningful (not just noise).
    """
    from training.train_rl import (
        build_prompt_text,
        extract_blender_code,
        generate_completions as gen_completions,
    )

    MIN_GAP = 0.2  # Minimum score difference to create a pair
    pairs = []

    pbar = tqdm(total=len(prompts), desc="Generating execution pairs") if HAS_TQDM else None

    for i, prompt_data in enumerate(prompts):
        if i < resume_offset:
            if pbar:
                pbar.update(1)
            continue

        prompt_text = build_prompt_text(prompt_data, tokenizer)

        # Generate N candidates
        try:
            completions = gen_completions(
                model, tokenizer, prompt_text,
                n=n_candidates,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                device=device,
            )
        except Exception as e:
            log.debug(f"Generation failed for prompt {i}: {e}")
            if pbar:
                pbar.update(1)
            continue

        # Extract Blender code from each completion
        codes = [extract_blender_code(c) for c in completions]

        # Score all candidates via Blender execution
        exec_scores = reward_fn.score_batch(codes)

        # Rank by combined score
        ranked = rank_candidates(completions, exec_scores)

        if len(ranked) < 2:
            if pbar:
                pbar.update(1)
            continue

        best_idx, best_score = ranked[0]
        worst_idx, worst_score = ranked[-1]

        # Only create a pair if there's a meaningful gap
        if best_score - worst_score < MIN_GAP:
            if pbar:
                pbar.update(1)
            continue

        # The human turn from the conversation is the prompt
        human_turns = [
            c for c in prompt_data.get("conversations", [])
            if c.get("from") == "human"
        ]
        prompt_str = human_turns[0]["value"] if human_turns else prompt_text

        pair = {
            "prompt":       prompt_str,
            "chosen":       completions[best_idx],
            "rejected":     completions[worst_idx],
            "source":       "execution",
            "chosen_score": best_score,
            "rejected_score": worst_score,
            "score_gap":    best_score - worst_score,
            "exec_scores":  exec_scores,
        }

        pairs.append(pair)

        # Write immediately so we can resume interrupted runs
        with output_file.open("a") as f:
            f.write(json.dumps(pair) + "\n")

        if pbar:
            pbar.update(1)
            pbar.set_postfix(
                pairs=len(pairs),
                avg_gap=f"{sum(p['score_gap'] for p in pairs)/len(pairs):.2f}"
            )

    if pbar:
        pbar.close()

    return pairs


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate DPO preference pairs for Nalana self-improvement loop"
    )

    # Model
    parser.add_argument("--model", required=True,
                        help="Path to current Nalana model (RL or DPO checkpoint)")
    parser.add_argument("--flash-attn", action="store_true", default=False)

    # Data
    parser.add_argument("--sft-data-dir", default="data/train",
                        help="SFT training data dir (source of prompts)")
    parser.add_argument("--n-prompts", type=int, default=1000,
                        help="Number of prompts to generate pairs for")
    parser.add_argument("--output-dir", default="data/dpo",
                        help="Output directory for DPO pair files")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an interrupted run (append to existing files)")

    # Generation
    parser.add_argument("--n-candidates", type=int, default=4,
                        help="Number of candidates to generate per prompt")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Generation temperature (1.0 = diverse, needed for good pairs)")

    # Blender
    parser.add_argument("--blender-path", type=str, default=None)
    parser.add_argument("--blender-workers", type=int, default=4)
    parser.add_argument("--blender-timeout", type=int, default=30)

    # Conversation pair generation via Claude
    parser.add_argument("--gen-conversation-pairs", action="store_true",
                        help="Generate synthetic conversation quality pairs via Claude")
    parser.add_argument("--n-conversation-pairs", type=int, default=200)
    parser.add_argument("--anthropic-api-key", default=None,
                        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")

    # Physics pairs
    parser.add_argument("--include-physics-pairs", action="store_true", default=True,
                        help="Include the built-in physics reasoning pairs")

    # Misc
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    log.info(f"Loading model: {args.model}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, **model_kwargs
    )
    model.eval()
    log.info("Model loaded.")

    # ── Reward function ───────────────────────────────────────────────────────
    import sys
    sys.path.insert(0, str(ROOT))
    from training.train_rl import NalanaRewardFunction, load_rl_prompts

    reward_fn = NalanaRewardFunction(
        blender_path=args.blender_path,
        timeout=args.blender_timeout,
        workers=args.blender_workers,
    )

    # ── Load prompts ──────────────────────────────────────────────────────────
    log.info(f"Loading prompts from {args.sft_data_dir}...")
    prompts = load_rl_prompts(Path(args.sft_data_dir), limit=args.n_prompts)
    random.shuffle(prompts)
    log.info(f"Prompts loaded: {len(prompts):,}")

    stats = {"execution": 0, "synthetic_conv": 0, "physics": 0}

    # ── Generate execution-based pairs ────────────────────────────────────────
    exec_output = output_dir / "execution_pairs.jsonl"
    resume_offset = 0

    if args.resume and exec_output.exists():
        existing = sum(1 for l in exec_output.read_text().splitlines() if l.strip())
        resume_offset = existing
        log.info(f"Resuming: {existing} pairs already written to {exec_output}")

    log.info(f"\nGenerating execution-based DPO pairs ({len(prompts):,} prompts)...")
    exec_pairs = generate_execution_pairs(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        reward_fn=reward_fn,
        n_candidates=args.n_candidates,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        output_file=exec_output,
        resume_offset=resume_offset,
        device=device,
    )
    stats["execution"] = len(exec_pairs)
    log.info(f"Execution pairs generated: {len(exec_pairs):,} → {exec_output}")

    # ── Generate conversation quality pairs via Claude ─────────────────────────
    if args.gen_conversation_pairs:
        api_key = args.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            log.warning(
                "No Anthropic API key found. Skipping conversation pair generation.\n"
                "Set --anthropic-api-key or ANTHROPIC_API_KEY env var."
            )
        elif not HAS_ANTHROPIC:
            log.warning("anthropic package not installed. Skipping. Run: pip install anthropic")
        else:
            conv_output = output_dir / "synthetic_conv_pairs.jsonl"
            if not (args.resume and conv_output.exists()):
                conv_output.write_text("")  # Clear if not resuming

            log.info(f"\nGenerating conversation quality pairs via Claude ({args.n_conversation_pairs} pairs)...")
            conv_pairs = generate_conversation_pairs(
                api_key=api_key,
                n_pairs=args.n_conversation_pairs,
                output_file=conv_output,
            )
            stats["synthetic_conv"] = len(conv_pairs)
            log.info(f"Conversation pairs: {len(conv_pairs):,} → {conv_output}")

    # ── Write physics reasoning pairs ─────────────────────────────────────────
    if args.include_physics_pairs:
        physics_output = output_dir / "physics_reasoning_pairs.jsonl"
        with physics_output.open("w") as f:
            for pair in PHYSICS_PAIRS:
                f.write(json.dumps(pair) + "\n")
        stats["physics"] = len(PHYSICS_PAIRS)
        log.info(f"Physics pairs: {len(PHYSICS_PAIRS)} → {physics_output}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total = sum(stats.values())
    log.info(f"\n{'='*50}")
    log.info(f"DPO pair generation complete")
    log.info(f"  Execution pairs:   {stats['execution']:,}")
    log.info(f"  Conv quality pairs:{stats['synthetic_conv']:,}")
    log.info(f"  Physics pairs:     {stats['physics']:,}")
    log.info(f"  Total:             {total:,}")
    log.info(f"  Output dir:        {output_dir}")
    log.info(f"\nNext: train_dpo.py with fresh pairs:")
    log.info(f"  python train_dpo.py \\")
    log.info(f"    --base-model {args.model} \\")
    log.info(f"    --data-dir {output_dir} \\")
    log.info(f"    --output-dir checkpoints/nalana-dpo-round2")


if __name__ == "__main__":
    main()
