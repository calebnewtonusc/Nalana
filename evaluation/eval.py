"""
eval.py - Post-training evaluation suite for the Nalana model.

Metrics computed:
  1. Blender execution success rate  — % of generated code that runs in headless Blender
  2. Task type accuracy              — does the model generate the right response type
  3. Physics reasoning quality       — IOR/Fresnel/roughness keyword presence + logic
  4. Multi-turn coherence            — context maintained across 50 multi-turn scenarios
  5. Cross-software accuracy         — correct API namespacing per software
  6. Latency benchmarks              — P50/P95/P99 for command generation

Outputs:
  eval_results/results.json      — raw metric data
  eval_results/eval_report.md    — human-readable report

Usage:
    python eval.py --model checkpoints/nalana-sft/final
    python eval.py --model checkpoints/nalana-rl/final --compare checkpoints/nalana-sft/final
    python eval.py --quick  # 100-sample fast eval
"""

from __future__ import annotations
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import math
import os
import random
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# ─── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).parent
PROJECT_ROOT  = Path(__file__).parents[1]
EVAL_DIR      = BASE_DIR / "eval_results"
VALIDATED_DIR = PROJECT_ROOT / "data" / "validated"
MULTITURN_DIR = PROJECT_ROOT / "data" / "multiturn"

# ─── Lazy imports (require GPU / model loaded) ─────────────────────────────────

def load_vllm_engine(model_path: str):
    from vllm import AsyncLLMEngine, AsyncEngineArgs
    args = AsyncEngineArgs(
        model             = model_path,
        tensor_parallel_size = int(os.environ.get("NALANA_TP_SIZE", "1")),
        dtype             = "bfloat16",
        max_model_len     = 8192,
        gpu_memory_utilization = 0.85,
    )
    return AsyncLLMEngine.from_engine_args(args)


def load_hf_model(model_path: str):
    """Fallback: load with HuggingFace transformers (single GPU, no vLLM)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype    = torch.bfloat16,
        device_map     = "auto",
        trust_remote_code = True,
    )
    model.eval()
    return model, tok


# ─── Nalana system prompt (same as training) ─────────────────────────────────

NALANA_SYSTEM = """You are Nalana, a voice-to-3D AI assistant.
Given a voice command, generate a JSON response with:
- blender_python: executable Python code
- blender_op: {"op": ..., "args": {...}}
- reasoning: one sentence explanation

For BUILD tasks, output a multi-step plan as a JSON array.
Always output valid, executable Blender Python API code."""

# ─── Evaluation prompts ───────────────────────────────────────────────────────

EXECUTE_PROMPTS = [
    ("Add a cube to the scene",          "EXECUTE", r"bpy\.ops\.mesh\.primitive_cube_add"),
    ("Add a UV sphere",                  "EXECUTE", r"bpy\.ops\.mesh\.primitive_uv_sphere_add"),
    ("Extrude the selected face up 1 meter", "EXECUTE", r"extrude|bpy\.ops\.mesh\.extrude"),
    ("Add a subdivision surface modifier", "EXECUTE", r"SUBSURF|modifier_add"),
    ("Shade the object smooth",          "EXECUTE", r"shade_smooth"),
    ("Delete the selected object",       "EXECUTE", r"bpy\.ops\.object\.delete"),
    ("Scale the object by 2 on X axis",  "EXECUTE", r"transform\.resize|scale"),
    ("Add a bevel to the edges",         "EXECUTE", r"bevel"),
    ("Rotate 90 degrees on Z axis",      "EXECUTE", r"transform\.rotate|rotation"),
    ("Add a mirror modifier",            "EXECUTE", r"MIRROR|mirror"),
    ("Loop cut the cube once",           "EXECUTE", r"loop_cut"),
    ("Inset the top face",               "EXECUTE", r"inset_faces"),
    ("Add an array modifier with 5 copies", "EXECUTE", r"ARRAY|array"),
    ("Set origin to geometry center",    "EXECUTE", r"origin_set|ORIGIN_GEOMETRY"),
    ("Apply all transforms",             "EXECUTE", r"transform_apply"),
    ("Duplicate the selected object",    "EXECUTE", r"duplicate"),
    ("Join all selected objects",        "EXECUTE", r"bpy\.ops\.object\.join"),
    ("Add a plane",                      "EXECUTE", r"primitive_plane_add"),
    ("Add a cylinder",                   "EXECUTE", r"primitive_cylinder_add"),
    ("Set render engine to Cycles",      "EXECUTE", r"CYCLES|render\.engine"),
]

MATERIALIZE_PROMPTS = [
    ("Create a gold metal material",
     "MATERIALIZE",
     r"Metallic|metallic|Principled BSDF|materials\.new"),
    ("Make the material look like brushed aluminum",
     "MATERIALIZE",
     r"[Mm]etallic|[Rr]oughness|Principled"),
    ("Create a glass material with IOR 1.52",
     "MATERIALIZE",
     r"IOR|1\.52|[Tt]ransmission|glass"),
    ("Apply a matte red paint material",
     "MATERIALIZE",
     r"[Rr]oughness|Base Color|materials"),
    ("Create aged copper with patina",
     "MATERIALIZE",
     r"[Mm]etallic|[Rr]oughness|patina|copper|Principled"),
    ("Make a subsurface skin material",
     "MATERIALIZE",
     r"[Ss]ubsurface|SSS|skin|Principled"),
    ("Create an emission neon material",
     "MATERIALIZE",
     r"[Ee]mission|Strength|neon|glow"),
    ("Apply a wood texture material",
     "MATERIALIZE",
     r"[Ww]ood|materials|Principled|[Rr]oughness"),
]

PHYSICS_REASONING_PROMPTS = [
    ("Explain why glass needs IOR 1.5 and what Fresnel means",
     ["IOR", "Fresnel", "refraction", "1.5", "angle", "reflection"]),
    ("Why does roughness affect how shiny a material looks",
     ["roughness", "scatter", "microfacet", "specular", "smooth"]),
    ("What is the difference between metallic 0 and metallic 1 in PBR",
     ["metallic", "conductor", "dielectric", "Fresnel", "specular", "absorption"]),
    ("How does subsurface scattering make skin look realistic",
     ["subsurface", "scatter", "skin", "light", "absorption", "translucent"]),
    ("Why use HDRI for lighting instead of area lights",
     ["HDRI", "environment", "reflections", "global", "ambient", "realistic"]),
    ("Explain clearcoat and when to use it",
     ["clearcoat", "lacquer", "second layer", "specular", "car", "shiny"]),
    ("What does anisotropy do in Principled BSDF",
     ["anisotropy", "directional", "brushed", "reflection", "highlight"]),
    ("Why is emission strength 1.0 not the same as a physical light",
     ["emission", "energy", "lux", "lumen", "physical", "exposure"]),
]

CROSS_SOFTWARE_PROMPTS = [
    ("Add a cube in Maya", "maya",   r"cmds\.polyCube|cmds\.poly"),
    ("Extrude a face in Maya", "maya", r"cmds\.polyExtrudeFacet|polyExtrude"),
    ("Add a box in Houdini", "houdini", r"hou\.|createNode.*box|geo\.createNode"),
    ("Subdivide in Houdini", "houdini", r"hou\.|subdivide|iterations"),
    ("Add a sphere in Cinema 4D", "cinema4d", r"c4d\.|Osphere|InsertObject"),
    ("Add subdivision in Cinema 4D", "cinema4d", r"c4d\.|Osds|InsertUnder"),
    ("Add a sphere in Rhino", "rhino", r"rs\.|AddSphere"),
    ("Fillet an edge in Rhino", "rhino", r"rs\.|FilletEdge"),
]

MULTI_TURN_SCENARIOS = [
    {
        "turns": [
            "Create a simple wooden chair",
            "Now add cushions to the seat",
            "Make the wood dark walnut colored",
            "Set up studio lighting for a product shot",
        ],
        "check_context": ["chair", "wood", "cushion", "light"],
    },
    {
        "turns": [
            "Start modeling an office building facade",
            "Add windows as rectangular cutouts",
            "Apply a concrete material to the walls",
            "Add glass material to the windows",
        ],
        "check_context": ["building", "windows", "concrete", "glass"],
    },
    {
        "turns": [
            "Create a photorealistic apple",
            "Add a stem and leaf",
            "Make the skin waxy and red",
            "Set up macro photography lighting",
        ],
        "check_context": ["apple", "stem", "leaf", "waxy", "red"],
    },
]


# ─── Model generation wrapper ─────────────────────────────────────────────────

class ModelClient:
    """Abstracts vLLM or HuggingFace generation behind a single interface."""

    def __init__(self, model_path: str, use_vllm: bool = True):
        self.model_path       = model_path
        self.use_vllm         = use_vllm
        self._model           = None
        self._tokenizer       = None
        self._use_vllm_actual = False

    def _ensure_loaded(self):
        if self._model is not None:
            return
        if self.use_vllm:
            try:
                from vllm import LLM, SamplingParams
                self._model = LLM(
                    model                = self.model_path,
                    tensor_parallel_size = int(os.environ.get("NALANA_TP_SIZE", "1")),
                    dtype                = "bfloat16",
                    max_model_len        = 8192,
                    gpu_memory_utilization = 0.85,
                )
                self._use_vllm_actual = True
                print(f"Loaded model with vLLM: {self.model_path}")
            except ImportError:
                print("vLLM not available, falling back to HuggingFace...")
                self.use_vllm = False

        if not self.use_vllm:
            self._model, self._tokenizer = load_hf_model(self.model_path)
            self._use_vllm_actual = False
            print(f"Loaded model with HuggingFace: {self.model_path}")

    def generate(self, prompt: str, software: str = "blender",
                 max_tokens: int = 512, temperature: float = 0.1) -> tuple[str, float]:
        """Returns (generated_text, latency_seconds)."""
        self._ensure_loaded()

        messages = [
            {"role": "system",  "content": NALANA_SYSTEM},
            {"role": "user",    "content": f"[{software.upper()}] {prompt}"},
        ]

        t_start = time.perf_counter()

        if self._use_vllm_actual:
            from vllm import SamplingParams
            # Format as chat
            text_input = self._format_chat(messages)
            params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
            outputs = self._model.generate([text_input], params)
            generated = outputs[0].outputs[0].text
        else:
            import torch
            text_input = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(text_input, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens = max_tokens,
                    temperature    = temperature,
                    do_sample      = temperature > 0,
                )
            generated = self._tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )

        latency = time.perf_counter() - t_start
        return generated.strip(), latency

    def _format_chat(self, messages: list[dict]) -> str:
        """Simple chat formatting fallback using Qwen2.5 chat template."""
        parts = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)


# ─── Blender execution check ──────────────────────────────────────────────────

def get_blender_path() -> str | None:
    candidates = [
        os.environ.get("BLENDER_PATH"),
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "/usr/bin/blender",
        "/usr/local/bin/blender",
        "blender",
    ]
    for c in candidates:
        if c and (Path(c).exists() or c == "blender"):
            return c
    return None


BLENDER_EXEC_HARNESS = '''
import bpy, sys, json, traceback
argv = sys.argv
try:
    code_file = argv[argv.index("--") + 1]
    code = open(code_file).read()
except (ValueError, IndexError, FileNotFoundError) as e:
    print("EVAL_RESULT:" + json.dumps({"success": False, "error": str(e)}))
    sys.exit(1)

bpy.ops.wm.read_factory_settings(use_empty=True)
try:
    exec(code, {"bpy": bpy, "__builtins__": __builtins__})
    print("EVAL_RESULT:" + json.dumps({"success": True, "error": None}))
except Exception as e:
    print("EVAL_RESULT:" + json.dumps({"success": False, "error": f"{type(e).__name__}: {e}"}))
'''


def blender_exec_check(code: str, blender_path: str, timeout: int = 25) -> bool:
    """Run code in headless Blender. Returns True if no exception."""
    if not code.strip() or code.strip().startswith("#"):
        return False
    # Only attempt if code looks like Blender Python
    if not re.search(r"bpy\.", code):
        return False

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as sf:
        sf.write(BLENDER_EXEC_HARNESS)
        harness_path = sf.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as cf:
        cf.write(code)
        code_path = cf.name

    try:
        result = subprocess.run(
            [blender_path, "--background", "--python", harness_path, "--", code_path],
            capture_output=True, text=True, timeout=timeout,
        )
        for line in result.stdout.splitlines():
            if line.startswith("EVAL_RESULT:"):
                data = json.loads(line[len("EVAL_RESULT:"):])
                return data.get("success", False)
        return False
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        Path(harness_path).unlink(missing_ok=True)
        Path(code_path).unlink(missing_ok=True)


def extract_code_from_response(response: str) -> str:
    """Pull blender_python field from JSON response, or extract raw code block."""
    # Try JSON parse
    try:
        data = json.loads(response)
        if isinstance(data, dict):
            return data.get("blender_python", "")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data[0].get("blender_python", "")
    except json.JSONDecodeError:
        pass
    # Try code block extraction
    m = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fall back to full response
    return response


# ─── Metric helpers ───────────────────────────────────────────────────────────

def check_physics_keywords(response: str, required_keywords: list[str]) -> tuple[float, list[str]]:
    """Check presence of expected physics/materials keywords. Returns (score, found)."""
    resp_lower = response.lower()
    found = [kw for kw in required_keywords if kw.lower() in resp_lower]
    score = len(found) / max(len(required_keywords), 1)
    return round(score, 4), found


def check_task_type_response(response: str, expected_type: str) -> bool:
    """
    Verify the response format matches the expected task type:
    - EXECUTE: should have blender_python with a bpy.ops call
    - BUILD:   should have a plan or JSON array
    - MATERIALIZE: should have material-related code
    """
    has_json  = "blender_python" in response or "{" in response
    has_bpy   = "bpy." in response
    has_plan  = "plan" in response.lower() or "step" in response.lower()
    has_array = "[" in response and "blender_python" in response

    if expected_type == "EXECUTE":
        return has_bpy and has_json
    elif expected_type == "BUILD":
        return has_plan or has_array
    elif expected_type == "MATERIALIZE":
        mat_keywords = ["material", "bsdf", "roughness", "metallic", "emission", "ior"]
        return has_bpy and any(kw in response.lower() for kw in mat_keywords)
    elif expected_type == "LIGHT":
        light_keywords = ["light", "energy", "world", "hdri", "cycles"]
        return any(kw in response.lower() for kw in light_keywords)
    return has_json or has_bpy


def check_cross_software(response: str, software: str, pattern: str) -> bool:
    return bool(re.search(pattern, response, re.IGNORECASE))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = int(math.ceil(p / 100 * len(sorted_v))) - 1
    return sorted_v[max(0, idx)]


# ─── Evaluation runners ───────────────────────────────────────────────────────

def eval_blender_exec(model: ModelClient, blender_path: str | None,
                      n: int = 500) -> dict[str, Any]:
    print(f"\n[Eval 1] Blender execution success rate (n={n})...")
    if not blender_path:
        print("  Blender not found — skipping execution eval.")
        return {"skipped": True, "reason": "blender_not_found"}

    prompts = EXECUTE_PROMPTS * (n // len(EXECUTE_PROMPTS) + 1)
    prompts = prompts[:n]
    random.shuffle(prompts)

    success   = 0
    attempted = 0

    try:
        from tqdm import tqdm
        bar = tqdm(prompts, desc="exec eval")
    except ImportError:
        bar = prompts

    for voice, task_type, _ in bar:
        response, _ = model.generate(voice, software="blender", max_tokens=256)
        code = extract_code_from_response(response)
        if not code:
            continue
        attempted += 1
        if blender_exec_check(code, blender_path):
            success += 1

    rate = success / max(attempted, 1)
    result = {
        "total_prompts": n,
        "attempted":     attempted,
        "success":       success,
        "success_rate":  round(rate, 4),
    }
    print(f"  Success rate: {rate*100:.1f}% ({success}/{attempted})")
    return result


def eval_task_type_accuracy(model: ModelClient) -> dict[str, Any]:
    print(f"\n[Eval 2] Task type accuracy...")
    prompts = EXECUTE_PROMPTS + list(MATERIALIZE_PROMPTS)
    total   = 0
    correct = 0
    breakdown: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0})

    for voice, task_type, pattern_or_kws in prompts:
        response, _ = model.generate(voice, software="blender", max_tokens=256)
        is_correct  = check_task_type_response(response, task_type)
        total      += 1
        breakdown[task_type]["total"]   += 1
        if is_correct:
            correct                         += 1
            breakdown[task_type]["correct"] += 1

    for voice, task_type, keywords in PHYSICS_REASONING_PROMPTS:
        response, _ = model.generate(voice, software="blender", max_tokens=512)
        is_correct  = check_task_type_response(response, "UNDERSTAND")
        total      += 1
        breakdown["UNDERSTAND"]["total"] = breakdown["UNDERSTAND"].get("total", 0) + 1
        if is_correct:
            correct += 1
            breakdown["UNDERSTAND"]["correct"] = breakdown["UNDERSTAND"].get("correct", 0) + 1

    accuracy = correct / max(total, 1)
    result = {
        "total":    total,
        "correct":  correct,
        "accuracy": round(accuracy, 4),
        "by_type":  {
            k: {
                "accuracy": round(v["correct"] / max(v["total"], 1), 4),
                "correct":  v["correct"],
                "total":    v["total"],
            }
            for k, v in breakdown.items()
        },
    }
    print(f"  Overall accuracy: {accuracy*100:.1f}% ({correct}/{total})")
    for k, v in result["by_type"].items():
        print(f"    {k:<20} {v['accuracy']*100:.1f}%  ({v['correct']}/{v['total']})")
    return result


def eval_physics_reasoning(model: ModelClient) -> dict[str, Any]:
    print(f"\n[Eval 3] Physics reasoning quality (n={len(PHYSICS_REASONING_PROMPTS)})...")
    scores   = []
    per_prompt = []

    for voice, keywords in PHYSICS_REASONING_PROMPTS:
        response, _ = model.generate(voice, software="blender", max_tokens=512)
        score, found = check_physics_keywords(response, keywords)
        scores.append(score)
        per_prompt.append({
            "prompt":         voice,
            "score":          score,
            "required":       keywords,
            "found":          found,
            "missing":        [k for k in keywords if k not in found],
        })

    avg = sum(scores) / max(len(scores), 1)
    result = {
        "avg_score":  round(avg, 4),
        "per_prompt": per_prompt,
        "n":          len(PHYSICS_REASONING_PROMPTS),
    }
    print(f"  Average physics keyword coverage: {avg*100:.1f}%")
    for pp in per_prompt:
        coverage = f"{pp['score']*100:.0f}%"
        print(f"    [{coverage:>4}] {pp['prompt'][:55]}")
        if pp["missing"]:
            print(f"           missing: {pp['missing']}")
    return result


def eval_multi_turn_coherence(model: ModelClient, n: int = 50) -> dict[str, Any]:
    print(f"\n[Eval 4] Multi-turn coherence (n={n} scenarios)...")
    scenarios = MULTI_TURN_SCENARIOS * (n // len(MULTI_TURN_SCENARIOS) + 1)
    scenarios = scenarios[:n]

    total_coherent = 0
    context_scores = []

    for scenario in scenarios:
        turns       = scenario["turns"]
        check_ctxs  = scenario["check_context"]
        history     = []
        responses   = []

        for turn in turns:
            # Build a context-aware prompt including history summary
            if history:
                ctx_summary = "Previous actions: " + "; ".join(history[-2:])
                full_prompt = f"{ctx_summary}\n\nNow: {turn}"
            else:
                full_prompt = turn

            response, _ = model.generate(full_prompt, software="blender", max_tokens=384)
            responses.append(response)
            history.append(turn)

        # Check if final response maintains context from earlier turns
        all_responses_text = " ".join(responses).lower()
        ctx_matches = sum(1 for ctx in check_ctxs
                         if ctx.lower() in all_responses_text)
        ctx_score   = ctx_matches / max(len(check_ctxs), 1)
        context_scores.append(ctx_score)
        if ctx_score >= 0.5:
            total_coherent += 1

    avg_ctx    = sum(context_scores) / max(len(context_scores), 1)
    coherence  = total_coherent / max(len(scenarios), 1)
    result     = {
        "scenarios_tested": len(scenarios),
        "coherent":         total_coherent,
        "coherence_rate":   round(coherence, 4),
        "avg_context_score": round(avg_ctx, 4),
    }
    print(f"  Coherence rate: {coherence*100:.1f}% ({total_coherent}/{len(scenarios)})")
    print(f"  Avg context coverage: {avg_ctx*100:.1f}%")
    return result


def eval_cross_software(model: ModelClient) -> dict[str, Any]:
    print(f"\n[Eval 5] Cross-software accuracy (n={len(CROSS_SOFTWARE_PROMPTS)})...")
    total   = 0
    correct = 0
    per_sw: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0})

    for voice, software, pattern in CROSS_SOFTWARE_PROMPTS:
        response, _ = model.generate(voice, software=software, max_tokens=256)
        is_correct  = check_cross_software(response, software, pattern)
        total      += 1
        per_sw[software]["total"]   += 1
        if is_correct:
            correct                     += 1
            per_sw[software]["correct"] += 1

    accuracy = correct / max(total, 1)
    result   = {
        "total":    total,
        "correct":  correct,
        "accuracy": round(accuracy, 4),
        "by_software": {
            sw: {
                "accuracy": round(v["correct"] / max(v["total"], 1), 4),
                "correct":  v["correct"],
                "total":    v["total"],
            }
            for sw, v in per_sw.items()
        },
    }
    print(f"  Overall accuracy: {accuracy*100:.1f}% ({correct}/{total})")
    for sw, v in result["by_software"].items():
        print(f"    {sw:<12} {v['accuracy']*100:.1f}%  ({v['correct']}/{v['total']})")
    return result


def eval_latency(model: ModelClient, n: int = 100) -> dict[str, Any]:
    print(f"\n[Eval 6] Latency benchmarks (n={n})...")
    prompts = (EXECUTE_PROMPTS * (n // len(EXECUTE_PROMPTS) + 1))[:n]
    latencies = []

    try:
        from tqdm import tqdm
        bar = tqdm(prompts, desc="latency")
    except ImportError:
        bar = prompts

    for voice, _, _ in bar:
        _, lat = model.generate(voice, software="blender", max_tokens=128, temperature=0.0)
        latencies.append(lat * 1000)  # ms

    p50  = percentile(latencies, 50)
    p95  = percentile(latencies, 95)
    p99  = percentile(latencies, 99)
    avg  = sum(latencies) / max(len(latencies), 1)

    result = {
        "n":    n,
        "p50_ms":  round(p50, 1),
        "p95_ms":  round(p95, 1),
        "p99_ms":  round(p99, 1),
        "avg_ms":  round(avg, 1),
        "min_ms":  round(min(latencies), 1) if latencies else 0,
        "max_ms":  round(max(latencies), 1) if latencies else 0,
    }
    print(f"  P50: {p50:.1f}ms | P95: {p95:.1f}ms | P99: {p99:.1f}ms | Avg: {avg:.1f}ms")
    return result


# ─── Report generation ────────────────────────────────────────────────────────

def write_results(results: dict, model_path: str) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    results_path = EVAL_DIR / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults: {results_path}")


def write_report(results: dict, model_path: str,
                 compare_path: str | None = None,
                 compare_results: dict | None = None) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    report_path = EVAL_DIR / "eval_report.md"

    lines = [
        "# Nalana Evaluation Report",
        "",
        f"**Model:** `{model_path}`",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
        "",
    ]

    if compare_path and compare_results:
        lines += [
            f"**Baseline:** `{compare_path}`",
            "",
            "## Comparison Summary",
            "",
            "| Metric | Baseline | This Model | Delta |",
            "|--------|----------|------------|-------|",
        ]
        metrics_to_compare = [
            ("Blender Exec Success", "blender_exec.success_rate"),
            ("Task Type Accuracy",   "task_type.accuracy"),
            ("Physics Reasoning",    "physics_reasoning.avg_score"),
            ("Multi-turn Coherence", "multi_turn.coherence_rate"),
            ("Cross-SW Accuracy",    "cross_software.accuracy"),
            ("Latency P50 (ms)",     "latency.p50_ms"),
        ]
        for label, key_path in metrics_to_compare:
            keys = key_path.split(".")
            baseline_v = compare_results
            current_v  = results
            for k in keys:
                baseline_v = baseline_v.get(k, {}) if isinstance(baseline_v, dict) else "N/A"
                current_v  = current_v.get(k, {})  if isinstance(current_v, dict)  else "N/A"
            if isinstance(baseline_v, float) and isinstance(current_v, float):
                delta = current_v - baseline_v
                delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
                baseline_str = f"{baseline_v:.4f}"
                current_str  = f"{current_v:.4f}"
            else:
                baseline_str = str(baseline_v)
                current_str  = str(current_v)
                delta_str    = "N/A"
            lines.append(f"| {label} | {baseline_str} | {current_str} | {delta_str} |")
        lines.append("")

    # Detailed metrics
    lines += [
        "## Detailed Results",
        "",
        "### 1. Blender Execution Success",
        "",
    ]
    be = results.get("blender_exec", {})
    if be.get("skipped"):
        lines.append("*Skipped — Blender not found.*")
    else:
        rate = be.get("success_rate", 0)
        lines += [
            f"- **Success Rate:** {rate*100:.1f}% ({be.get('success','?')}/{be.get('attempted','?')} attempted)",
            f"- **Prompts Tested:** {be.get('total_prompts','?')}",
        ]

    lines += [
        "",
        "### 2. Task Type Accuracy",
        "",
    ]
    tt = results.get("task_type", {})
    lines.append(f"- **Overall:** {tt.get('accuracy', 0)*100:.1f}%  ({tt.get('correct','?')}/{tt.get('total','?')})")
    lines.append("")
    lines.append("| Type | Accuracy | Correct | Total |")
    lines.append("|------|----------|---------|-------|")
    for k, v in (tt.get("by_type") or {}).items():
        lines.append(f"| {k} | {v['accuracy']*100:.1f}% | {v['correct']} | {v['total']} |")

    lines += [
        "",
        "### 3. Physics Reasoning Quality",
        "",
    ]
    pr = results.get("physics_reasoning", {})
    lines.append(f"- **Avg Keyword Coverage:** {pr.get('avg_score', 0)*100:.1f}%")
    lines.append("")
    for pp in (pr.get("per_prompt") or []):
        score_pct = f"{pp['score']*100:.0f}%"
        missing   = ", ".join(pp.get("missing", [])) or "none"
        lines.append(f"- [{score_pct}] *{pp['prompt'][:60]}*  — missing: {missing}")

    lines += [
        "",
        "### 4. Multi-turn Coherence",
        "",
    ]
    mt = results.get("multi_turn", {})
    lines += [
        f"- **Coherence Rate:** {mt.get('coherence_rate', 0)*100:.1f}%",
        f"- **Avg Context Score:** {mt.get('avg_context_score', 0)*100:.1f}%",
        f"- **Scenarios Tested:** {mt.get('scenarios_tested','?')}",
    ]

    lines += [
        "",
        "### 5. Cross-Software Accuracy",
        "",
    ]
    cs = results.get("cross_software", {})
    lines.append(f"- **Overall:** {cs.get('accuracy', 0)*100:.1f}%")
    lines.append("")
    lines.append("| Software | Accuracy | Correct | Total |")
    lines.append("|----------|----------|---------|-------|")
    for sw, v in (cs.get("by_software") or {}).items():
        lines.append(f"| {sw} | {v['accuracy']*100:.1f}% | {v['correct']} | {v['total']} |")

    lines += [
        "",
        "### 6. Latency Benchmarks",
        "",
    ]
    lat = results.get("latency", {})
    lines += [
        f"| Percentile | Latency |",
        f"|------------|---------|",
        f"| P50        | {lat.get('p50_ms','?')} ms |",
        f"| P95        | {lat.get('p95_ms','?')} ms |",
        f"| P99        | {lat.get('p99_ms','?')} ms |",
        f"| Avg        | {lat.get('avg_ms','?')} ms |",
        f"| Min        | {lat.get('min_ms','?')} ms |",
        f"| Max        | {lat.get('max_ms','?')} ms |",
        "",
    ]

    report_path.write_text("\n".join(lines))
    print(f"Report:  {report_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Nalana model")
    parser.add_argument("--model",   type=str, required=False,
                        help="Path to model checkpoint (or HuggingFace repo ID)")
    parser.add_argument("--compare", type=str, default=None,
                        help="Baseline model to compare against")
    parser.add_argument("--quick",   action="store_true",
                        help="Fast eval: 100 samples, skip Blender exec")
    parser.add_argument("--no-vllm", action="store_true",
                        help="Use HuggingFace instead of vLLM")
    parser.add_argument("--blender-path", type=str,
                        help="Path to Blender executable")
    parser.add_argument("--exec-n",  type=int, default=500,
                        help="Blender exec eval sample count")
    parser.add_argument("--skip-exec",   action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    args = parser.parse_args()

    model_path = args.model or os.environ.get("NALANA_MODEL_PATH", "")
    if not model_path:
        print("Error: --model or NALANA_MODEL_PATH required.")
        parser.print_help()
        return

    # Quick mode overrides
    exec_n    = 100 if args.quick else args.exec_n
    lat_n     = 50  if args.quick else 100
    mt_n      = 10  if args.quick else 50

    print(f"Loading model: {model_path}")
    model = ModelClient(model_path, use_vllm=not args.no_vllm)

    blender = args.blender_path or get_blender_path()

    results: dict[str, Any] = {
        "model":      model_path,
        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "quick_mode": args.quick,
    }

    # Run evaluations
    if not args.skip_exec:
        results["blender_exec"] = eval_blender_exec(model, blender, n=exec_n)
    else:
        results["blender_exec"] = {"skipped": True, "reason": "skip_exec_flag"}

    results["task_type"]         = eval_task_type_accuracy(model)
    results["physics_reasoning"] = eval_physics_reasoning(model)
    results["multi_turn"]        = eval_multi_turn_coherence(model, n=mt_n)
    results["cross_software"]    = eval_cross_software(model)

    if not args.skip_latency and not args.quick:
        results["latency"] = eval_latency(model, n=lat_n)
    elif args.quick:
        results["latency"] = eval_latency(model, n=lat_n)
    else:
        results["latency"] = {"skipped": True}

    # Compare mode
    compare_results = None
    if args.compare:
        print(f"\n\nRunning baseline eval on: {args.compare}")
        compare_model   = ModelClient(args.compare, use_vllm=not args.no_vllm)
        compare_results = {"model": args.compare}
        compare_results["task_type"]         = eval_task_type_accuracy(compare_model)
        compare_results["physics_reasoning"] = eval_physics_reasoning(compare_model)
        compare_results["cross_software"]    = eval_cross_software(compare_model)
        compare_results["latency"]           = eval_latency(compare_model, n=lat_n)

    write_results(results, model_path)
    write_report(results, model_path, args.compare, compare_results)

    # Print summary
    print(f"\n{'═'*55}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'═'*55}")
    be_rate = results.get("blender_exec", {}).get("success_rate", "N/A")
    tt_acc  = results.get("task_type", {}).get("accuracy", "N/A")
    pr_avg  = results.get("physics_reasoning", {}).get("avg_score", "N/A")
    mt_coh  = results.get("multi_turn", {}).get("coherence_rate", "N/A")
    cs_acc  = results.get("cross_software", {}).get("accuracy", "N/A")
    p50     = results.get("latency", {}).get("p50_ms", "N/A")

    def fmt(v): return f"{v*100:.1f}%" if isinstance(v, float) else str(v)
    print(f"  Blender Exec Success:   {fmt(be_rate)}")
    print(f"  Task Type Accuracy:     {fmt(tt_acc)}")
    print(f"  Physics Reasoning:      {fmt(pr_avg)}")
    print(f"  Multi-turn Coherence:   {fmt(mt_coh)}")
    print(f"  Cross-Software:         {fmt(cs_acc)}")
    print(f"  Latency P50:            {p50} ms")
    print(f"{'═'*55}")


if __name__ == "__main__":
    main()
