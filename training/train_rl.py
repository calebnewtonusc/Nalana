import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
train_rl.py - Stage 2: Execution-reward RL for Nalana

Reward signal: headless Blender execution (validate_blender.py)
Method: GRPO (Group Relative Policy Optimization) — same as DeepSeek-R1
Base: The SFT checkpoint from train.py

For each prompt:
  1. Generate N=4 candidate responses (Blender Python code)
  2. Execute all 4 in headless Blender via validate_blender.py
  3. Score each: 1.0 (success+change) / 0.5 (ran/no change) / 0.0 (error)
  4. Compute advantages: score - mean(scores) = relative quality
  5. GRPO loss: encourage high-advantage responses, penalize low-advantage

Why GRPO over PPO:
  - No separate value/critic model needed
  - Group normalization is more stable
  - Perfect for verifiable code correctness

Hardware target: 18x A6000 (48GB each)
GPU allocation for RL:
  - GPUs 0-15: Policy model training (DeepSpeed ZeRO-2)
  - GPUs 16-17: Generation inference (or use same process with offload)
  - Blender workers: CPU-side parallel processes (4 workers per node)

Launch commands:
  # Single node, 16 GPUs:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \\
  deepspeed --num_gpus=16 train_rl.py \\
    --base-model checkpoints/nalana-sft/final \\
    --output-dir checkpoints/nalana-rl \\
    --deepspeed ds_config_rl.json

  # Quick test (2 GPUs, small batch):
  CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 train_rl.py \\
    --base-model checkpoints/nalana-sft/final \\
    --num-samples 2 --max-steps 100 --no-wandb

Usage:
  python train_rl.py --base-model checkpoints/nalana-sft/final --output-dir checkpoints/nalana-rl
  python train_rl.py --num-samples 4 --generations-per-prompt 4 --max-steps 2000
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    get_cosine_schedule_with_warmup,
)

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ─── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parents[1]
DATA_DIR = ROOT / "data" / "train"
VALIDATE_SCRIPT = ROOT / "validation" / "validate_blender.py"

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("nalana-rl")


# ─── Blender reward harness (embedded, mirrors validate_blender.py logic) ─────

BLENDER_HARNESS = """
import bpy, sys, json, traceback

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
    bpy.ops.object.select_all(action="DESELECT")

def scene_fingerprint():
    data = bpy.data
    return {
        "object_count": len(data.objects),
        "mesh_names": sorted(o.name for o in data.objects if o.type == "MESH"),
        "total_verts": sum(len(o.data.vertices) for o in data.objects if o.type == "MESH"),
        "total_faces": sum(len(o.data.polygons) for o in data.objects if o.type == "MESH"),
        "modifier_count": sum(len(o.modifiers) for o in data.objects),
        "material_count": len(data.materials),
        "has_armature": any(o.type == "ARMATURE" for o in data.objects),
        "has_light": any(o.type == "LIGHT" for o in data.objects),
        "has_camera": any(o.type == "CAMERA" for o in data.objects),
    }

argv = sys.argv
try:
    payload_file = argv[argv.index("--") + 1]
except (ValueError, IndexError):
    print(json.dumps({"error": "no payload file"}))
    sys.exit(1)

with open(payload_file) as f:
    payload = json.load(f)

python_code = payload.get("blender_python", "")
mode = payload.get("mode", "object")

reset_scene()

if mode in ("edit", "EDIT"):
    bpy.ops.object.select_all(action="SELECT")
    bpy.context.view_layer.objects.active = bpy.data.objects[0]
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")

before = scene_fingerprint()
exec_success = False
exec_error = None

try:
    exec(python_code, {"bpy": bpy, "__builtins__": __builtins__})
    exec_success = True
except Exception as e:
    exec_error = f"{type(e).__name__}: {str(e)}"

try:
    bpy.ops.object.mode_set(mode="OBJECT")
except Exception:
    pass

after = scene_fingerprint()
scene_changed = before != after

if not exec_success:
    score = 0.0
elif scene_changed:
    score = 1.0
else:
    score = 0.5

print("NALANA_RESULT:" + json.dumps({
    "score": score,
    "exec_success": exec_success,
    "exec_error": exec_error,
    "scene_changed": scene_changed,
}))
"""


# ─── Reward function ──────────────────────────────────────────────────────────


class NalanaRewardFunction:
    """
    Wraps headless Blender execution as a reward signal.

    Score:
      1.0  — script executed successfully AND scene state changed
      0.5  — script executed but no measurable scene change (e.g. a hide op)
      0.0  — Python/syntax error, timeout, or Blender crash

    This is a FREE reward signal — no human labelers, no reward model.
    The ground truth is: does the code actually work in Blender?
    """

    def __init__(
        self,
        blender_path: Optional[str] = None,
        timeout: int = 30,
        workers: int = 4,
    ):
        self.blender_path = self._resolve_blender(blender_path)
        self.timeout = timeout
        self.workers = workers
        log.info(
            f"Reward function: Blender={self.blender_path}, workers={workers}, timeout={timeout}s"
        )

    @staticmethod
    def _resolve_blender(hint: Optional[str]) -> str:
        candidates = [
            hint,
            os.environ.get("BLENDER_PATH"),
            "/Applications/Blender.app/Contents/MacOS/Blender",
            "/usr/bin/blender",
            "/usr/local/bin/blender",
            "blender",
        ]
        for c in candidates:
            if c and (Path(c).exists() or c == "blender"):
                return c
        raise RuntimeError(
            "Blender not found. Set BLENDER_PATH env var or pass --blender-path."
        )

    @staticmethod
    def _detect_mode(code: str) -> str:
        code_lower = code.strip().lower()
        for prefix in ("bpy.ops.mesh.", "bpy.ops.curve.", "bpy.ops.uv."):
            if prefix in code_lower:
                return "edit"
        return "object"

    def _execute_one(self, code: str) -> float:
        """Run one Blender script, return its score."""
        if not code or not code.strip():
            return 0.0

        payload = {"blender_python": code, "mode": self._detect_mode(code)}

        payload_path = None
        script_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as pf:
                json.dump(payload, pf)
                payload_path = pf.name

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as sf:
                sf.write(BLENDER_HARNESS)
                script_path = sf.name

            result = subprocess.run(
                [
                    self.blender_path,
                    "--background",
                    "--python",
                    script_path,
                    "--",
                    payload_path,
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            for line in result.stdout.splitlines():
                if line.startswith("NALANA_RESULT:"):
                    data = json.loads(line[len("NALANA_RESULT:") :])
                    return float(data.get("score", 0.0))

            # Blender ran but no result marker — crash or unexpected output
            return 0.0

        except subprocess.TimeoutExpired:
            return 0.0
        except Exception:
            return 0.0
        finally:
            if payload_path:
                Path(payload_path).unlink(missing_ok=True)
            if script_path:
                Path(script_path).unlink(missing_ok=True)

    def score_batch(self, codes: list[str]) -> list[float]:
        """
        Score a batch of generated code strings in parallel.
        Uses ProcessPoolExecutor to run multiple headless Blender instances.
        Returns list of floats in same order as input.
        """
        if not codes:
            return []

        workers = min(self.workers, len(codes))
        scores = [0.0] * len(codes)

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._execute_one, code): i for i, code in enumerate(codes)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    scores[idx] = future.result()
                except Exception:
                    scores[idx] = 0.0

        return scores

    def score_groups(self, groups: list[list[str]]) -> list[list[float]]:
        """
        Score groups of candidate codes.
        groups: [[code1, code2, code3, code4], [code1, ...], ...]
        Returns matching list of score lists.
        """
        # Flatten for parallel execution
        flat_codes = []
        group_sizes = []
        for group in groups:
            group_sizes.append(len(group))
            flat_codes.extend(group)

        flat_scores = self.score_batch(flat_codes)

        # Reconstruct group structure
        result = []
        offset = 0
        for size in group_sizes:
            result.append(flat_scores[offset : offset + size])
            offset += size
        return result


# ─── Dataset ──────────────────────────────────────────────────────────────────


def load_rl_prompts(data_dir: Path, limit: Optional[int] = None) -> list[dict]:
    """
    Load training prompts from the SFT dataset (ShareGPT format).
    We only use the prompt (human turn) — the model will generate completions.
    """
    train_file = data_dir / "sharegpt_train.jsonl"
    if not train_file.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_file}. Run train_prep.py first."
        )

    prompts = []
    for line in train_file.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        convs = record.get("conversations", [])
        if not convs:
            continue

        # Extract human turns as prompts (we'll generate assistant completions)
        human_turns = [c for c in convs if c.get("from") == "human"]
        if human_turns:
            prompts.append(
                {
                    "prompt_text": human_turns[0]["value"],
                    "conversations": convs,
                }
            )

    if limit:
        prompts = prompts[:limit]

    log.info(f"Loaded {len(prompts):,} prompts from {train_file}")
    return prompts


def build_prompt_text(example: dict, tokenizer) -> str:
    """
    Format a prompt into the model's expected input format.
    Uses the chat template with only the human turn (no assistant response).
    """
    convs = example.get("conversations", [])
    messages = []

    system_msgs = [c for c in convs if c.get("from") == "system"]
    if system_msgs:
        messages.append({"role": "system", "content": system_msgs[0]["value"]})

    human_msgs = [c for c in convs if c.get("from") == "human"]
    if human_msgs:
        messages.append({"role": "user", "content": human_msgs[0]["value"]})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # Adds the assistant prefix so model generates next
    )


def extract_blender_code(text: str) -> str:
    """
    Extract Blender Python code from model output.
    Handles: raw code, ```python blocks, ```blender blocks.
    """
    text = text.strip()

    # Try to extract from code fences first
    for fence_lang in ("```python", "```blender", "```"):
        if fence_lang in text:
            start = text.find(fence_lang) + len(fence_lang)
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

    # If no code fence, look for bpy.ops lines
    lines = text.splitlines()
    bpy_lines = [l for l in lines if "bpy." in l]
    if bpy_lines:
        return "\n".join(bpy_lines)

    return text


# ─── GRPO Loss ────────────────────────────────────────────────────────────────


def compute_grpo_loss(
    model,
    ref_log_probs: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    advantages: torch.Tensor,
    kl_coeff: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    GRPO (Group Relative Policy Optimization) loss.

    This is the core RL update. For each token in the response:
      1. Compute log prob under current policy
      2. Compute KL penalty vs reference model (SFT checkpoint)
      3. Scale by advantage (how good was this completion vs group average)

    Loss = -mean(advantage * log_prob) + kl_coeff * KL(policy || ref)

    Why not PPO:
      - PPO needs a critic/value network (another 7B model just for value estimation)
      - GRPO normalizes within the group: advantage = reward - group_mean
      - No clipping ratio needed because group normalization provides stability

    Args:
        model: Current policy model
        ref_log_probs: Log probs from the frozen reference model, shape [B, T]
        input_ids: Token IDs, shape [B, T]
        attention_mask: Attention mask, shape [B, T]
        response_mask: 1 for response tokens, 0 for prompt tokens, shape [B, T]
        advantages: Per-sample scalar advantages, shape [B]
        kl_coeff: Weight of KL penalty term

    Returns:
        (loss, metrics_dict)
    """
    # Forward pass through current policy
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [B, T, V]

    # Shift for autoregressive: predict token t+1 from token t
    shift_logits = logits[:, :-1, :]  # [B, T-1, V]
    shift_labels = input_ids[:, 1:]  # [B, T-1]
    shift_mask = response_mask[:, 1:]  # [B, T-1]

    # Per-token log probs under current policy
    log_probs = F.log_softmax(shift_logits, dim=-1)  # [B, T-1, V]
    token_log_probs = log_probs.gather(
        dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)  # [B, T-1]

    # Mask to response tokens only
    token_log_probs = token_log_probs * shift_mask  # zero out prompt tokens

    # Per-sequence log prob sum (over response tokens)
    seq_log_probs = token_log_probs.sum(dim=-1)  # [B]
    n_response_tokens = shift_mask.sum(dim=-1).clamp(min=1)
    seq_log_probs_mean = seq_log_probs / n_response_tokens  # normalize by length

    # KL divergence penalty vs reference (per token)
    # ref_log_probs is already token-level, shifted and masked
    kl_per_token = token_log_probs - ref_log_probs  # [B, T-1]
    kl_per_seq = kl_per_token.sum(dim=-1) / n_response_tokens  # [B]

    # GRPO policy gradient loss
    # advantages are normalized at group level: A_i = r_i - mean(r)
    # We want to MAXIMIZE: advantage * log_prob → MINIMIZE: -advantage * log_prob
    pg_loss = -(advantages.detach() * seq_log_probs_mean).mean()
    kl_loss = kl_per_seq.mean()

    total_loss = pg_loss + kl_coeff * kl_loss

    metrics = {
        "loss/total": total_loss.item(),
        "loss/pg": pg_loss.item(),
        "loss/kl": kl_loss.item(),
        "policy/mean_log_prob": seq_log_probs_mean.mean().item(),
        "policy/mean_advantage": advantages.mean().item(),
    }

    return total_loss, metrics


def compute_advantages(scores: list[list[float]]) -> list[list[float]]:
    """
    GRPO advantage computation.

    For each group of N scores (one per generated completion):
      advantage_i = score_i - mean(scores_in_group)

    This means:
      - Completions better than the group average get positive advantage (reinforced)
      - Completions worse than average get negative advantage (discouraged)
      - If all completions score the same: all advantages are 0 (no update needed)

    Returns advantages in the same shape as scores.
    """
    advantages = []
    for group in scores:
        group_mean = sum(group) / len(group)
        group_std = (sum((s - group_mean) ** 2 for s in group) / len(group)) ** 0.5
        # Normalize by std for stability (avoid tiny/huge gradients)
        normed = [(s - group_mean) / (group_std + 1e-8) for s in group]
        advantages.append(normed)
    return advantages


# ─── Reference model log probs ────────────────────────────────────────────────


@torch.no_grad()
def compute_ref_log_probs(
    ref_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-token log probs from the frozen reference model.
    Used for the KL divergence penalty in GRPO loss.

    The reference model is the SFT checkpoint — we don't want to drift too far
    from it (policy collapse prevention).
    """
    outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    mask = response_mask[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    return token_log_probs * mask  # [B, T-1]


# ─── Tokenization helpers ──────────────────────────────────────────────────────


def tokenize_prompt_and_response(
    prompt: str,
    response: str,
    tokenizer,
    max_length: int = 1024,
) -> dict:
    """
    Tokenize a (prompt, response) pair and return a response_mask
    that's 1 only for the response tokens (the part the model generated).
    """
    full_text = prompt + response

    encoding = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
    )

    prompt_encoding = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
    )

    prompt_len = prompt_encoding["input_ids"].shape[1]
    total_len = encoding["input_ids"].shape[1]

    # Response mask: 0 for prompt tokens, 1 for response tokens
    response_mask = torch.zeros(total_len, dtype=torch.long)
    response_mask[prompt_len:] = 1

    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
        "response_mask": response_mask,
    }


def pad_batch(items: list[dict], tokenizer, max_length: int) -> dict:
    """
    Pad a list of tokenized items to the same length for batched processing.
    """
    max_len = min(max(item["input_ids"].shape[0] for item in items), max_length)

    input_ids_list = []
    attention_mask_list = []
    response_mask_list = []

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    for item in items:
        ids = item["input_ids"]
        attn = item["attention_mask"]
        resp = item["response_mask"]

        pad_len = max_len - ids.shape[0]
        if pad_len > 0:
            ids = torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
            attn = torch.cat([attn, torch.zeros(pad_len, dtype=torch.long)])
            resp = torch.cat([resp, torch.zeros(pad_len, dtype=torch.long)])
        else:
            ids = ids[:max_len]
            attn = attn[:max_len]
            resp = resp[:max_len]

        input_ids_list.append(ids)
        attention_mask_list.append(attn)
        response_mask_list.append(resp)

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "response_mask": torch.stack(response_mask_list),
    }


# ─── Generation ───────────────────────────────────────────────────────────────


@torch.no_grad()
def generate_completions(
    model,
    tokenizer,
    prompt_text: str,
    n: int,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 0.95,
    device: str = "cuda",
) -> list[str]:
    """
    Generate N candidate completions for a single prompt.

    Temperature=1.0 is intentional: we want diversity in the group.
    If temperature were low, all N completions would be nearly identical
    and the group-relative advantages would all be ~0 → no learning signal.
    """
    input_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).input_ids.to(device)

    # Repeat prompt N times for batch generation
    input_ids = input_ids.repeat(n, 1)

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    output_ids = model.generate(input_ids, generation_config=gen_config)

    # Decode only the new tokens (not the prompt)
    prompt_len = input_ids.shape[1]
    completions = []
    for i in range(n):
        new_tokens = output_ids[i, prompt_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        completions.append(text)

    return completions


# ─── GRPO Trainer ─────────────────────────────────────────────────────────────


class GRPOTrainer:
    """
    Group Relative Policy Optimization trainer for Nalana.

    Training loop:
      For each batch of prompts:
        1. Generate N completions per prompt using current policy
        2. Execute all completions in headless Blender (parallel)
        3. Compute group-relative advantages
        4. Compute GRPO loss (policy gradient + KL penalty)
        5. Backprop and update

    The key insight from DeepSeek-R1: when the reward signal is VERIFIABLE
    (code either works or it doesn't), you don't need a learned reward model.
    The execution result IS the reward.
    """

    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        reward_fn: NalanaRewardFunction,
        optimizer,
        scheduler,
        args: argparse.Namespace,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args

        self.device = next(model.parameters()).device
        self.global_step = 0
        self.best_avg_reward = -float("inf")

        # Stats tracking
        self.running_rewards = []
        self.running_losses = []

    def train_step(self, prompt_batch: list[dict]) -> dict:
        """
        One GRPO training step over a batch of prompts.
        Returns metrics dict.
        """
        self.model.eval()  # Eval mode during generation (no dropout)

        # 1. Format prompts
        prompt_texts = [build_prompt_text(p, self.tokenizer) for p in prompt_batch]

        # 2. Generate N completions per prompt
        all_completions = []  # List of lists: [[comp1..compN], [comp1..compN], ...]
        for prompt_text in prompt_texts:
            completions = generate_completions(
                self.model,
                self.tokenizer,
                prompt_text,
                n=self.args.num_samples,
                max_new_tokens=self.args.max_new_tokens,
                temperature=self.args.temperature,
                device=str(self.device),
            )
            all_completions.append(completions)

        # 3. Extract Blender code and score via execution
        all_codes = [
            [extract_blender_code(c) for c in group] for group in all_completions
        ]
        raw_scores = self.reward_fn.score_groups(all_codes)  # [[float, ...], ...]

        # 4. Compute GRPO advantages (group-relative normalization)
        advantages_groups = compute_advantages(raw_scores)  # same shape as scores

        # 5. Build training batch (flatten prompt+completion pairs)
        train_items = []
        flat_advantages = []

        for i, (prompt_text, completions) in enumerate(
            zip(prompt_texts, all_completions)
        ):
            for j, completion in enumerate(completions):
                tok = tokenize_prompt_and_response(
                    prompt_text,
                    completion,
                    self.tokenizer,
                    max_length=self.args.max_length,
                )
                train_items.append(tok)
                flat_advantages.append(advantages_groups[i][j])

        if not train_items:
            return {}

        # 6. Pad and move to device
        batch = pad_batch(train_items, self.tokenizer, self.args.max_length)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        response_mask = batch["response_mask"].to(self.device)
        advantages_t = torch.tensor(flat_advantages, dtype=torch.float32).to(
            self.device
        )

        # 7. Compute reference log probs (frozen SFT model)
        ref_log_probs = compute_ref_log_probs(
            self.ref_model, input_ids, attention_mask, response_mask
        )

        # 8. Forward + GRPO loss
        self.model.train()
        loss, metrics = compute_grpo_loss(
            self.model,
            ref_log_probs,
            input_ids,
            attention_mask,
            response_mask,
            advantages_t,
            kl_coeff=self.args.kl_coeff,
        )

        # 9. Backward + optimizer step (with gradient accumulation)
        grad_accum = getattr(self.args, "grad_accum_steps", 1) or 1
        (loss / grad_accum).backward()

        if (self.global_step + 1) % grad_accum == 0:
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.grad_clip
                )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        # Add reward stats to metrics
        flat_scores = [s for group in raw_scores for s in group]
        metrics["reward/mean"] = sum(flat_scores) / len(flat_scores)
        metrics["reward/success"] = sum(1 for s in flat_scores if s == 1.0) / len(
            flat_scores
        )
        metrics["reward/partial"] = sum(1 for s in flat_scores if s == 0.5) / len(
            flat_scores
        )
        metrics["reward/failure"] = sum(1 for s in flat_scores if s == 0.0) / len(
            flat_scores
        )
        metrics["train/lr"] = self.scheduler.get_last_lr()[0]
        metrics["train/step"] = self.global_step

        self.running_rewards.extend(flat_scores)
        self.running_losses.append(metrics["loss/total"])
        self.global_step += 1

        return metrics

    def save_checkpoint(self, output_dir: Path, tag: str = ""):
        """Save model checkpoint and training state."""
        ckpt_dir = (
            output_dir / f"checkpoint-{self.global_step}{'-' + tag if tag else ''}"
        )
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model (LoRA adapter weights or full model)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(str(ckpt_dir))
        else:
            torch.save(self.model.state_dict(), str(ckpt_dir / "model.pt"))

        self.tokenizer.save_pretrained(str(ckpt_dir))

        # Save training state
        state = {
            "global_step": self.global_step,
            "best_avg_reward": self.best_avg_reward,
            "optimizer_state": self.optimizer.state_dict(),
        }
        torch.save(state, str(ckpt_dir / "trainer_state.pt"))

        log.info(f"Checkpoint saved: {ckpt_dir}")
        return ckpt_dir

    def log_metrics(self, metrics: dict):
        """Log metrics to console and wandb."""
        if not metrics:
            return

        step = self.global_step
        reward = metrics.get("reward/mean", 0)
        loss = metrics.get("loss/total", 0)
        kl = metrics.get("loss/kl", 0)
        lr = metrics.get("train/lr", 0)
        succ = metrics.get("reward/success", 0)

        log.info(
            f"step {step:>5} | loss {loss:.4f} | kl {kl:.4f} | "
            f"reward {reward:.3f} | success {succ:.1%} | lr {lr:.2e}"
        )

        if HAS_WANDB and not self.args.no_wandb:
            wandb.log(metrics, step=step)


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Stage 2: GRPO RL training for Nalana")

    # Model args
    parser.add_argument(
        "--base-model",
        required=True,
        help="Path to SFT checkpoint (from train.py), or HF model ID",
    )
    parser.add_argument("--output-dir", default="checkpoints/nalana-rl")
    parser.add_argument(
        "--ref-model",
        default=None,
        help="Reference model path (defaults to --base-model)",
    )
    parser.add_argument("--deepspeed", type=str, help="Path to DeepSpeed config JSON")
    parser.add_argument("--flash-attn", action="store_true", default=False)
    parser.add_argument(
        "--lora-r",
        type=int,
        default=32,
        help="LoRA rank for RL (lower than SFT, more stable)",
    )

    # Data args
    parser.add_argument("--data-dir", default="data/train")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="Limit number of training prompts (for debugging)",
    )

    # RL args
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of completions to generate per prompt (N in GRPO)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=2000, help="Total RL training steps"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of prompts per step (effective = batch * num-samples completions)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-6,
        help="Learning rate (lower than SFT — RL is sensitive)",
    )
    parser.add_argument(
        "--kl-coeff",
        type=float,
        default=0.1,
        help="KL divergence penalty weight (prevents policy collapse)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Generation temperature (1.0 = diverse completions)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max tokens to generate per completion",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Max total sequence length (prompt + response)",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch size = batch-size × grad-accum-steps)",
    )
    parser.add_argument("--warmup-steps", type=int, default=50)

    # Blender reward args
    parser.add_argument("--blender-path", type=str, default=None)
    parser.add_argument(
        "--blender-workers",
        type=int,
        default=4,
        help="Parallel Blender processes for reward evaluation",
    )
    parser.add_argument(
        "--blender-timeout",
        type=int,
        default=30,
        help="Timeout per Blender execution (seconds)",
    )

    # Checkpoint args
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=5)
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume training from a checkpoint directory",
    )

    # Logging
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="nalana-rl")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--log-steps", type=int, default=10)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── W&B init ──────────────────────────────────────────────────────────────
    if HAS_WANDB and not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"nalana-rl-{time.strftime('%Y%m%d-%H%M%S')}",
            config=vars(args),
        )
        log.info(f"W&B initialized: {wandb.run.url}")
    else:
        log.info("W&B disabled (pass --no-wandb to suppress this message)")

    # ── Device setup ──────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpus = torch.cuda.device_count()
        log.info(f"Using {n_gpus} GPU(s)")
    else:
        device = torch.device("cpu")
        log.warning("No GPU found — RL training on CPU will be extremely slow")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    log.info(f"Loading tokenizer from: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Load policy model ─────────────────────────────────────────────────────
    log.info(f"Loading policy model: {args.base_model}")
    model_kwargs = dict(torch_dtype=torch.bfloat16, device_map=None)
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True, **model_kwargs
    )
    model.enable_input_require_grads()

    # Apply LoRA for parameter-efficient RL
    # Smaller rank than SFT — RL updates should be conservative
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.0,  # No dropout during RL (we need stable gradients)
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model = model.to(device)

    # ── Load reference model (frozen SFT checkpoint) ──────────────────────────
    ref_model_path = args.ref_model or args.base_model
    log.info(f"Loading reference model (frozen): {ref_model_path}")
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_model_path, trust_remote_code=True, **model_kwargs
    )
    ref_model = ref_model.to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    # ── Reward function ───────────────────────────────────────────────────────
    reward_fn = NalanaRewardFunction(
        blender_path=args.blender_path,
        timeout=args.blender_timeout,
        workers=args.blender_workers,
    )

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.0,  # No weight decay for RL
        betas=(0.9, 0.95),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    # ── Load training prompts ─────────────────────────────────────────────────
    prompts = load_rl_prompts(Path(args.data_dir), limit=args.num_prompts)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
    )

    # Resume from checkpoint if specified
    if args.resume_from:
        state_file = Path(args.resume_from) / "trainer_state.pt"
        if state_file.exists():
            state = torch.load(str(state_file))
            trainer.global_step = state["global_step"]
            trainer.best_avg_reward = state["best_avg_reward"]
            optimizer.load_state_dict(state["optimizer_state"])
            log.info(f"Resumed from step {trainer.global_step}")

    # ── Training loop ─────────────────────────────────────────────────────────
    log.info("\nStarting GRPO RL training")
    log.info(f"  Policy model: {args.base_model}")
    log.info(f"  Prompts: {len(prompts):,}")
    log.info(f"  Steps: {args.max_steps:,}")
    log.info(f"  Samples per prompt: {args.num_samples}")
    log.info(f"  LR: {args.learning_rate:.2e}")
    log.info(f"  KL coeff: {args.kl_coeff}")
    log.info(f"  Output: {output_dir}\n")

    # Save args for reproducibility
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    prompt_idx = 0
    kept_checkpoints = []

    while trainer.global_step < args.max_steps:
        # Sample batch of prompts (cyclic)
        batch = []
        for _ in range(args.batch_size):
            batch.append(prompts[prompt_idx % len(prompts)])
            prompt_idx += 1

        metrics = trainer.train_step(batch)

        if trainer.global_step % args.log_steps == 0:
            trainer.log_metrics(metrics)

        # Save checkpoint
        if trainer.global_step % args.save_steps == 0:
            ckpt_path = trainer.save_checkpoint(output_dir)
            kept_checkpoints.append(ckpt_path)

            # Trim old checkpoints
            while len(kept_checkpoints) > args.save_total_limit:
                old = kept_checkpoints.pop(0)
                if old.exists():
                    import shutil

                    shutil.rmtree(str(old))
                    log.info(f"Removed old checkpoint: {old}")

            # Track best reward
            if trainer.running_rewards:
                recent_reward = sum(trainer.running_rewards[-100:]) / min(
                    100, len(trainer.running_rewards)
                )
                if recent_reward > trainer.best_avg_reward:
                    trainer.best_avg_reward = recent_reward
                    trainer.save_checkpoint(output_dir, tag="best")
                    log.info(f"New best avg reward: {recent_reward:.3f}")

    # ── Final save ────────────────────────────────────────────────────────────
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"\nMerging LoRA weights → {final_dir}")
    try:
        merged = model.merge_and_unload()
        merged.save_pretrained(str(final_dir))
    except Exception:
        # If merge fails (e.g., not a LoRA model), just save as-is
        model.save_pretrained(str(final_dir))

    tokenizer.save_pretrained(str(final_dir))

    if HAS_WANDB and not args.no_wandb:
        wandb.finish()

    log.info("\nRL training complete.")
    log.info(f"  Final model: {final_dir}")
    log.info(f"  Best reward: {trainer.best_avg_reward:.3f}")
    log.info(
        f"\nNext: python train_dpo.py --base-model {final_dir} --output-dir checkpoints/nalana-dpo"
    )


if __name__ == "__main__":
    main()
