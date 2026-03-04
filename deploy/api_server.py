"""
deploy/api_server.py - FastAPI server for the Nalana voice-to-3D model.

Endpoints:
  POST /v1/command      Single command → Blender Python + Universal DSL
  WS   /v1/stream       Streaming command execution (WebSocket)
  POST /v1/plan         Full multi-step build plan from intent
  POST /v1/materialize  Named material → PBR code + physics explanation
  GET  /v1/health       Health check with GPU info

Infrastructure:
  - vLLM AsyncEngine for high-throughput GPU inference
  - Redis for multi-turn session state (conversation history)
  - Structured JSON logging
  - Per-API-key rate limiting
  - CORS for browser plugins (Spline-style)

Environment variables:
  NALANA_MODEL_PATH     Path to model checkpoint directory
  NALANA_TP_SIZE        Tensor parallel size (default: 4)
  NALANA_MAX_MODEL_LEN  Max context tokens (default: 8192)
  REDIS_URL             Redis connection URL (default: redis://redis:6379)
  NALANA_API_KEYS       Comma-separated valid API keys (default: allow all)
  LOG_LEVEL             Logging level (default: INFO)

Usage:
  uvicorn deploy.api_server:app --host 0.0.0.0 --port 8000 --workers 1
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import redis.asyncio as aioredis
import torch
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

# ─── Logging setup ────────────────────────────────────────────────────────────

class StructuredJSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ"),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
        }
        if hasattr(record, "extra"):
            log_record.update(record.extra)
        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


def setup_logging() -> None:
    level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredJSONFormatter())
    logging.basicConfig(level=level, handlers=[handler])
    # Suppress noisy vLLM/transformers logs at DEBUG level
    logging.getLogger("vllm").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)


setup_logging()
logger = logging.getLogger("nalana.api")

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_PATH    = os.environ.get("NALANA_MODEL_PATH", "/model")
TP_SIZE       = int(os.environ.get("NALANA_TP_SIZE", "4"))
MAX_MODEL_LEN = int(os.environ.get("NALANA_MAX_MODEL_LEN", "8192"))
REDIS_URL     = os.environ.get("REDIS_URL", "redis://redis:6379")

# API key authorization — if not set, all requests are allowed (dev mode)
_raw_keys     = os.environ.get("NALANA_API_KEYS", "")
VALID_API_KEYS: set[str] = set(k.strip() for k in _raw_keys.split(",") if k.strip())

# Rate limits per API key
RATE_LIMIT_RPM = int(os.environ.get("NALANA_RATE_LIMIT_RPM", "60"))

GPU_COUNT = torch.cuda.device_count()

# ─── System prompt ────────────────────────────────────────────────────────────

NALANA_SYSTEM = """You are Nalana, an expert voice-to-3D AI assistant.

Given a voice command, always respond with a valid JSON object containing:
{
  "blender_python": "<executable Python using bpy API>",
  "universal_dsl": {"op": "<OP_NAME>", "args": {...}, "target": "active_object", "intent": "..."},
  "reasoning": "<one sentence: why this op achieves the intent>"
}

Rules:
1. blender_python must be valid, executable Python using the bpy API.
2. Use real bpy.ops.* function names — never invent ops.
3. For BUILD tasks (create X from scratch), output a multi-step JSON array.
4. For MATERIALIZE tasks, include Principled BSDF node setup with physically correct values.
5. For LIGHT tasks, include energy values in watts and use physically based units.
6. For CROSS_SOFTWARE tasks, use the correct API for the specified software.
7. Never output explanatory text outside the JSON structure.
"""


# ─── Pydantic models ──────────────────────────────────────────────────────────

class CommandRequest(BaseModel):
    # Accept both "voice" (canonical) and "voice_command" (plugin legacy) as aliases.
    voice:         str | None       = Field(default=None, min_length=2, max_length=1000,
                                           description="The voice command (canonical field)")
    voice_command: str | None       = Field(default=None, min_length=2, max_length=1000,
                                           description="Alias for 'voice' (plugin compatibility)")
    scene_context: dict[str, Any]  = Field(default_factory=dict,
                                           description="Current scene state from the DCC plugin")
    software:      str              = Field(default="blender",
                                           description="Target software: blender, maya, cinema4d, houdini, rhino")
    session_id:    str | None       = Field(default=None,
                                           description="Session ID for multi-turn context")
    max_tokens:    int              = Field(default=512, ge=64, le=4096)
    temperature:   float            = Field(default=0.1, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def resolve_voice_field(self) -> "CommandRequest":
        """Merge voice_command alias into the canonical voice field."""
        self.voice = self.voice or self.voice_command
        if not self.voice:
            raise ValueError("Either 'voice' or 'voice_command' must be provided")
        return self


class CommandResponse(BaseModel):
    # Primary code field for the requested software (always present)
    code:            str
    # Per-software code dict (blender_python, maya_python, etc. for the requested software)
    code_by_software: dict[str, str]
    # Legacy field: always contains Blender Python for backwards compatibility
    blender_python:  str
    universal_dsl:   dict[str, Any]
    reasoning:       str
    request_id:      str
    model:           str
    latency_ms:      float
    session_id:      str | None


class PlanRequest(BaseModel):
    intent:       str   = Field(...,   min_length=5, max_length=500,
                               description="High-level intent, e.g. 'create an iPhone 16'")
    software:     str   = Field(default="blender")
    session_id:   str | None = Field(default=None)
    max_tokens:   int   = Field(default=2048, ge=256, le=4096)


class PlanStep(BaseModel):
    step:           int
    voice_command:  str
    blender_python: str
    reasoning:      str


class PlanResponse(BaseModel):
    intent:         str
    build_plan:     list[PlanStep]
    total_steps:    int
    estimated_time: str
    request_id:     str
    latency_ms:     float


class MaterializeRequest(BaseModel):
    material_name: str  = Field(...,   min_length=2, max_length=200,
                               description="Material name, e.g. 'aged copper' or 'frosted glass'")
    software:      str  = Field(default="blender")
    session_id:    str | None = Field(default=None)
    max_tokens:    int  = Field(default=768, ge=128, le=2048)


class MaterializeResponse(BaseModel):
    material_name:       str
    blender_python:      str
    physics_explanation: str
    universal_dsl:       dict[str, Any]
    request_id:          str
    latency_ms:          float


class QARequest(BaseModel):
    scene_context: dict[str, Any]  = Field(default_factory=dict,
                                           description="Scene state from the DCC plugin")
    blender_python: str | None     = Field(default=None,
                                           description="Optional code snippet to pre-evaluate before the full audit")
    profile:       str              = Field(default="game_pc",
                                           description="Platform profile: mobile, game_pc, cinematics, print_3d, arch_viz")
    auto_fix:      bool             = Field(default=False,
                                           description="If true, include fix_command code in the response")


class QAResponse(BaseModel):
    passed:         bool
    score:          float
    issue_count:    int
    warning_count:  int
    issues:         list[dict[str, Any]]
    warnings:       list[dict[str, Any]]
    info:           list[str]
    stats:          dict[str, Any]
    fix_code:       str             # Concatenated auto-fix Python (empty if auto_fix=False)
    request_id:     str
    latency_ms:     float


class HealthResponse(BaseModel):
    status:       str
    model:        str
    gpu_count:    int
    gpu_names:    list[str]
    version:      str = "nalana-v1"


# ─── Global state ─────────────────────────────────────────────────────────────

class AppState:
    engine:      Any = None   # vLLM AsyncLLMEngine
    tokenizer:   Any = None
    redis:       Any = None   # aioredis client
    model_name:  str = "nalana-v1"
    startup_time: float = 0.0

state = AppState()


# ─── vLLM engine ──────────────────────────────────────────────────────────────

async def init_vllm() -> None:
    """Initialize the vLLM async engine. Logs progress at each step."""
    logger.info("Initializing vLLM engine", extra={"model": MODEL_PATH, "tp_size": TP_SIZE})
    try:
        from vllm import AsyncLLMEngine, AsyncEngineArgs
        engine_args = AsyncEngineArgs(
            model                  = MODEL_PATH,
            tensor_parallel_size   = TP_SIZE,
            dtype                  = "bfloat16",
            max_model_len          = MAX_MODEL_LEN,
            gpu_memory_utilization = 0.90,
            enforce_eager          = False,
            disable_log_requests   = True,
        )
        state.engine = AsyncLLMEngine.from_engine_args(engine_args)
        # Extract tokenizer for chat template formatting
        state.tokenizer = await state.engine.get_tokenizer()
        logger.info("vLLM engine ready", extra={"model": MODEL_PATH})
    except ImportError:
        logger.warning("vLLM not installed — falling back to HuggingFace (slow)")
        await init_hf_fallback()
    except Exception as e:
        logger.error(f"vLLM init failed: {e}")
        raise


async def init_hf_fallback() -> None:
    """HuggingFace transformers fallback (development only)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    loop = asyncio.get_event_loop()

    def _load():
        tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype    = torch.bfloat16,
            device_map     = "auto",
            trust_remote_code = True,
        )
        model.eval()
        return model, tok

    hf_model, hf_tok = await loop.run_in_executor(None, _load)
    state.tokenizer = hf_tok
    # Store HF model under engine with a flag
    state.engine = ("hf", hf_model, hf_tok)
    logger.info("HuggingFace fallback model loaded")


async def init_redis() -> None:
    try:
        state.redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        await state.redis.ping()
        logger.info("Redis connected", extra={"url": REDIS_URL})
    except Exception as e:
        logger.warning(f"Redis unavailable ({e}) — session state disabled")
        state.redis = None


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    state.startup_time = time.time()
    logger.info("Nalana API starting up...", extra={"gpu_count": GPU_COUNT})
    await asyncio.gather(init_vllm(), init_redis())
    logger.info("Nalana API ready.")
    yield
    # Cleanup
    if state.redis:
        await state.redis.aclose()
    logger.info("Nalana API shut down.")


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Nalana Voice-to-3D API",
    description = "Transform voice commands into executable 3D software operations.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],           # Restrict in production via env
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─── Rate limiting ────────────────────────────────────────────────────────────

async def check_rate_limit(api_key: str) -> None:
    """Sliding window rate limit using Redis. Raises 429 if exceeded."""
    if not state.redis:
        return  # No Redis = no rate limiting (dev mode)

    window = 60  # 1-minute window
    key    = f"ratelimit:{api_key}"
    now    = time.time()
    window_start = now - window

    pipe = state.redis.pipeline()
    pipe.zremrangebyscore(key, 0, window_start)
    pipe.zadd(key, {str(uuid.uuid4()): now})
    pipe.zcard(key)
    pipe.expire(key, window * 2)
    results = await pipe.execute()
    request_count = results[2]

    if request_count > RATE_LIMIT_RPM:
        raise HTTPException(
            status_code = status.HTTP_429_TOO_MANY_REQUESTS,
            detail      = f"Rate limit exceeded: {RATE_LIMIT_RPM} requests/minute",
            headers     = {"Retry-After": str(window)},
        )


# ─── Auth ─────────────────────────────────────────────────────────────────────

async def get_api_key(x_api_key: str | None = Header(default=None)) -> str:
    """Validate API key. Returns "anonymous" in dev mode (no keys configured)."""
    if not VALID_API_KEYS:
        return "anonymous"  # Dev mode
    if not x_api_key or x_api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Invalid or missing X-API-Key header",
        )
    return x_api_key


# ─── Session state helpers ────────────────────────────────────────────────────

SESSION_TTL = 3600  # 1 hour


async def get_session_history(session_id: str | None) -> list[dict]:
    if not session_id or not state.redis:
        return []
    raw = await state.redis.get(f"session:{session_id}")
    if not raw:
        return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


async def save_session_history(session_id: str, history: list[dict]) -> None:
    if not session_id or not state.redis:
        return
    await state.redis.setex(
        f"session:{session_id}",
        SESSION_TTL,
        json.dumps(history[-20:]),  # Keep last 20 turns
    )


def build_session_id() -> str:
    return str(uuid.uuid4())


# ─── Inference helpers ────────────────────────────────────────────────────────

def build_messages(voice: str, software: str, scene_context: dict,
                   history: list[dict]) -> list[dict]:
    """Construct the chat message list for the model."""
    messages = [{"role": "system", "content": NALANA_SYSTEM}]

    # Inject recent history (last 4 turns)
    for turn in history[-4:]:
        messages.append({"role": "user",      "content": turn.get("user", "")})
        messages.append({"role": "assistant",  "content": turn.get("assistant", "")})

    # Build user prompt
    scene_str = ""
    if scene_context:
        scene_str = f"\nScene: {json.dumps(scene_context, separators=(',', ':'))}"

    user_content = f"[{software.upper()}]{scene_str}\n{voice}"
    messages.append({"role": "user", "content": user_content})
    return messages


async def generate_vllm(messages: list[dict],
                         max_tokens: int,
                         temperature: float,
                         request_id: str) -> tuple[str, float]:
    """Generate with vLLM AsyncEngine. Returns (text, latency_ms)."""
    from vllm import SamplingParams

    # Format messages using the tokenizer's chat template
    if hasattr(state.tokenizer, "apply_chat_template"):
        text_input = state.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Fallback manual formatting using Qwen2.5 chat template
        text_input = "\n".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages
        ) + "\n<|im_start|>assistant\n"

    params = SamplingParams(
        temperature  = temperature,
        max_tokens   = max_tokens,
        stop         = ["<|im_end|>"],
    )

    t0 = time.perf_counter()
    collected = []
    async for output in state.engine.generate(text_input, params, request_id=request_id):
        if output.outputs:
            collected = output.outputs
    t1 = time.perf_counter()

    if not collected:
        return "", (t1 - t0) * 1000

    return collected[0].text.strip(), (t1 - t0) * 1000


async def generate_hf(messages: list[dict],
                       max_tokens: int,
                       temperature: float) -> tuple[str, float]:
    """HF transformers generation fallback (blocking, runs in thread pool)."""
    import torch
    _, model, tok = state.engine

    text_input = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    loop = asyncio.get_event_loop()

    def _generate():
        inputs = tok(text_input, return_tensors="pt").to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens = max_tokens,
                temperature    = max(temperature, 1e-6),
                do_sample      = temperature > 0.05,
            )
        t1 = time.perf_counter()
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip(), (t1 - t0) * 1000

    return await loop.run_in_executor(None, _generate)


async def generate(messages: list[dict], max_tokens: int, temperature: float,
                   request_id: str) -> tuple[str, float]:
    """Dispatch to vLLM or HF fallback."""
    if isinstance(state.engine, tuple) and state.engine[0] == "hf":
        return await generate_hf(messages, max_tokens, temperature)
    return await generate_vllm(messages, max_tokens, temperature, request_id)


def parse_model_output(raw: str) -> dict[str, Any]:
    """
    Parse the model's raw output into structured fields.
    Handles JSON, JSON in markdown code blocks, and free-text fallback.
    """
    # Strip markdown code fences
    clean = raw.strip()
    if "```" in clean:
        import re
        m = re.search(r"```(?:json)?\n?(.*?)```", clean, re.DOTALL)
        if m:
            clean = m.group(1).strip()

    try:
        data = json.loads(clean)
        if isinstance(data, dict):
            return {
                "blender_python": data.get("blender_python", ""),
                "universal_dsl":  data.get("universal_dsl", {}),
                "reasoning":      data.get("reasoning", ""),
            }
        if isinstance(data, list):
            return {
                "blender_python": json.dumps(data),
                "universal_dsl":  {},
                "reasoning":      "Multi-step plan returned as array.",
            }
    except json.JSONDecodeError:
        pass

    # Fallback: treat entire output as blender_python
    return {
        "blender_python": clean,
        "universal_dsl":  {},
        "reasoning":      "Raw output (JSON parse failed).",
    }


# ─── Endpoints ────────────────────────────────────────────────────────────────

def _build_code_by_software(parsed: dict, software: str) -> dict[str, str]:
    """
    Compile Universal DSL to all supported software targets.
    Returns a dict like {"blender_python": "...", "maya_python": "...", ...}.
    Falls back gracefully if universal_dsl module is unavailable.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    blender_code = parsed.get("blender_python", "")
    result: dict[str, str] = {"blender_python": blender_code}

    dsl_data = parsed.get("universal_dsl", {})
    if not dsl_data or not dsl_data.get("op"):
        return result

    try:
        from core.universal_dsl import UniversalOp, COMPILERS, compile_op
        op = UniversalOp(
            op=dsl_data.get("op", "UNKNOWN"),
            args=dsl_data.get("args", {}),
            target=dsl_data.get("target", "active_object"),
            intent=dsl_data.get("intent", ""),
        )
        for sw_name in COMPILERS:
            key = f"{sw_name}_python" if sw_name != "rhino" else "rhino_python"
            result[f"{sw_name}_python"] = compile_op(op, sw_name)
    except Exception as e:
        logger.warning(f"DSL compilation failed: {e}")

    return result


@app.post("/v1/command", response_model=CommandResponse)
async def command(
    req:     CommandRequest,
    api_key: str = Depends(get_api_key),
) -> CommandResponse:
    """Convert a single voice command to executable 3D code."""
    request_id = str(uuid.uuid4())
    await check_rate_limit(api_key)

    session_id = req.session_id or build_session_id()
    history    = await get_session_history(req.session_id)
    voice      = req.voice  # resolved by model_validator (accepts voice or voice_command)

    logger.info("command request", extra={
        "request_id": request_id,
        "voice":      voice[:80],
        "software":   req.software,
        "session_id": session_id,
    })

    messages = build_messages(voice, req.software, req.scene_context, history)
    raw, latency = await generate(messages, req.max_tokens, req.temperature, request_id)
    parsed = parse_model_output(raw)

    # Retry once if JSON parse failed (no blender_python in output)
    if not parsed.get("blender_python") and not parsed.get("universal_dsl"):
        logger.warning("JSON parse failed on first attempt — retrying with stricter prompt",
                       extra={"request_id": request_id})
        retry_messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content":
             "Your previous response was not valid JSON. "
             "Reply with ONLY a JSON object: "
             "{\"blender_python\": \"...\", \"universal_dsl\": {...}, \"reasoning\": \"...\"}. "
             "No markdown fences, no extra text."},
        ]
        raw2, latency2 = await generate(retry_messages, req.max_tokens, 0.05, request_id + "-retry")
        parsed2 = parse_model_output(raw2)
        if parsed2.get("blender_python"):
            parsed = parsed2
            latency += latency2
            raw = raw2

    # Compile DSL to all supported software targets
    code_by_software = _build_code_by_software(parsed, req.software)
    primary_code = code_by_software.get(f"{req.software}_python", parsed.get("blender_python", ""))

    # Update session history
    new_history = history + [{"user": voice, "assistant": raw}]
    await save_session_history(session_id, new_history)

    logger.info("command response", extra={
        "request_id": request_id,
        "latency_ms": round(latency, 1),
        "has_code":   bool(primary_code),
        "software":   req.software,
    })

    return CommandResponse(
        code             = primary_code,
        code_by_software = code_by_software,
        blender_python   = parsed.get("blender_python", ""),
        universal_dsl    = parsed.get("universal_dsl", {}),
        reasoning        = parsed.get("reasoning", ""),
        request_id       = request_id,
        model            = state.model_name,
        latency_ms       = round(latency, 1),
        session_id       = session_id,
    )


@app.websocket("/v1/stream")
async def stream(ws: WebSocket) -> None:
    """
    Real-time streaming command execution via WebSocket.

    Client sends: {"voice": "...", "software": "blender", "session_id": "..."}
    Server streams: {"token": "..."} ... {"done": true, "latency_ms": ...}
    """
    await ws.accept()
    request_id = str(uuid.uuid4())
    logger.info("WebSocket connected", extra={"request_id": request_id})

    try:
        while True:
            data = await ws.receive_json()
            voice      = data.get("voice", "")
            software   = data.get("software", "blender")
            session_id = data.get("session_id")
            max_tokens = int(data.get("max_tokens", 512))
            temperature = float(data.get("temperature", 0.1))

            if not voice:
                await ws.send_json({"error": "voice field required"})
                continue

            history  = await get_session_history(session_id)
            messages = build_messages(voice, software, {}, history)

            # Stream tokens
            if isinstance(state.engine, tuple) and state.engine[0] == "hf":
                # HF doesn't stream — generate full then send
                raw, latency = await generate_hf(messages, max_tokens, temperature)
                await ws.send_json({"token": raw})
                await ws.send_json({"done": True, "latency_ms": round(latency, 1)})
            else:
                from vllm import SamplingParams
                if hasattr(state.tokenizer, "apply_chat_template"):
                    text_input = state.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    # Qwen2.5 chat format
                    formatted = "<|im_start|>system\nYou are Nalana, an expert 3D design AI assistant.<|im_end|>\n"
                    for msg in messages:
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
                    formatted += "<|im_start|>assistant\n"
                    text_input = formatted

                params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
                t0 = time.perf_counter()
                prev_len = 0
                async for output in state.engine.generate(
                    text_input, params, request_id=f"{request_id}-stream"
                ):
                    if output.outputs:
                        current_text = output.outputs[0].text
                        new_token    = current_text[prev_len:]
                        prev_len     = len(current_text)
                        if new_token:
                            await ws.send_json({"token": new_token})

                latency = (time.perf_counter() - t0) * 1000
                await ws.send_json({"done": True, "latency_ms": round(latency, 1)})

            # Update session
            if session_id:
                new_history = history + [{"user": voice, "assistant": ""}]
                await save_session_history(session_id, new_history)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected", extra={"request_id": request_id})
    except Exception as e:
        logger.error(f"WebSocket error: {e}", extra={"request_id": request_id})
        try:
            await ws.send_json({"error": str(e)})
            await ws.close()
        except Exception:
            pass


@app.post("/v1/plan", response_model=PlanResponse)
async def plan(
    req:     PlanRequest,
    api_key: str = Depends(get_api_key),
) -> PlanResponse:
    """
    Generate a complete multi-step build plan for a complex intent.
    Returns a structured list of steps with individual Blender Python code.
    """
    request_id = str(uuid.uuid4())
    await check_rate_limit(api_key)

    logger.info("plan request", extra={
        "request_id": request_id,
        "intent":     req.intent[:80],
        "software":   req.software,
    })

    plan_system = NALANA_SYSTEM + (
        "\n\nFor PLAN requests, output a JSON array of steps. Each step:"
        "\n{\"step\": N, \"voice_command\": \"...\", \"blender_python\": \"...\", \"reasoning\": \"...\"}"
        "\nRespond ONLY with the JSON array."
    )
    messages = [
        {"role": "system", "content": plan_system},
        {"role": "user",   "content": f"[{req.software.upper()}] {req.intent}"},
    ]

    raw, latency = await generate(messages, req.max_tokens, 0.1, request_id)

    # Parse plan JSON
    steps_data: list[dict] = []
    try:
        clean = raw.strip()
        if "```" in clean:
            import re
            m = re.search(r"```(?:json)?\n?(.*?)```", clean, re.DOTALL)
            if m:
                clean = m.group(1).strip()
        parsed = json.loads(clean)
        if isinstance(parsed, list):
            steps_data = parsed
        elif isinstance(parsed, dict) and "steps" in parsed:
            steps_data = parsed["steps"]
    except json.JSONDecodeError:
        # Fall back: treat as single step
        steps_data = [{"step": 1, "voice_command": req.intent,
                        "blender_python": raw, "reasoning": "Parsed as single step"}]

    plan_steps = []
    for i, s in enumerate(steps_data):
        plan_steps.append(PlanStep(
            step           = s.get("step", i + 1),
            voice_command  = s.get("voice_command", ""),
            blender_python = s.get("blender_python", ""),
            reasoning      = s.get("reasoning", ""),
        ))

    # Rough time estimate (3 min/step for complex models)
    est_min = len(plan_steps) * 3
    est_str = f"{est_min}min" if est_min < 60 else f"{est_min//60}h {est_min%60}min"

    return PlanResponse(
        intent         = req.intent,
        build_plan     = plan_steps,
        total_steps    = len(plan_steps),
        estimated_time = est_str,
        request_id     = request_id,
        latency_ms     = round(latency, 1),
    )


@app.post("/v1/materialize", response_model=MaterializeResponse)
async def materialize(
    req:     MaterializeRequest,
    api_key: str = Depends(get_api_key),
) -> MaterializeResponse:
    """
    Generate a PBR material setup for a named material.
    Also returns a physics_explanation explaining the material properties.
    """
    request_id = str(uuid.uuid4())
    await check_rate_limit(api_key)

    logger.info("materialize request", extra={
        "request_id":    request_id,
        "material_name": req.material_name,
        "software":      req.software,
    })

    mat_system = NALANA_SYSTEM + (
        "\n\nFor MATERIALIZE requests, output JSON with:"
        "\n{\"blender_python\": \"<full material setup code>\","
        "\n \"universal_dsl\": {\"op\": \"ADD_MATERIAL\", \"args\": {...}},"
        "\n \"reasoning\": \"<one sentence>\","
        "\n \"physics_explanation\": \"<2-3 sentences explaining IOR, Fresnel, roughness, etc. as they apply to this material>\"}"
    )
    messages = [
        {"role": "system", "content": mat_system},
        {"role": "user",   "content": f"[{req.software.upper()}] Create a PBR material: {req.material_name}"},
    ]

    raw, latency = await generate(messages, req.max_tokens, 0.15, request_id)
    parsed = parse_model_output(raw)

    # Extract physics_explanation if present in raw JSON
    physics_exp = ""
    try:
        raw_data = json.loads(raw.strip())
        physics_exp = raw_data.get("physics_explanation", "")
    except (json.JSONDecodeError, AttributeError):
        pass

    if not physics_exp:
        physics_exp = parsed.get("reasoning", "")

    return MaterializeResponse(
        material_name       = req.material_name,
        blender_python      = parsed["blender_python"],
        physics_explanation = physics_exp,
        universal_dsl       = parsed["universal_dsl"],
        request_id          = request_id,
        latency_ms          = round(latency, 1),
    )


@app.post("/v1/qa", response_model=QAResponse)
async def qa(
    req:     QARequest,
    api_key: str = Depends(get_api_key),
) -> QAResponse:
    """
    Scene QA audit. Accepts scene_context JSON from a DCC plugin and returns a
    structured report with issues, warnings, score, and optional fix code.

    Full Blender-based topology/UV checks require a .blend file (run qa_agent.py
    locally). This endpoint provides rule-based checks on the scene_context dict
    that work without a running Blender instance.
    """
    request_id = str(uuid.uuid4())
    await check_rate_limit(api_key)
    t0 = time.perf_counter()

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    issues: list[dict] = []
    warnings: list[dict] = []
    info: list[str] = []
    fix_parts: list[str] = []

    ctx = req.scene_context

    # ── Check 1: unapplied transforms ────────────────────────────────────────
    # SceneContext v1: prefer pre-computed has_unapplied_transform flag (catches
    # both scale AND rotation). Fall back to manual scale check for older clients.
    objects = ctx.get("objects", [])
    for obj in objects:
        name = obj.get("name", "unknown")
        # v1 path: use pre-computed flag
        if obj.get("has_unapplied_transform"):
            scale = obj.get("scale", [1, 1, 1])
            rot   = obj.get("rotation", [0, 0, 0])
            scale_bad = isinstance(scale, list) and any(abs(s - 1.0) > 1e-4 for s in scale)
            rot_bad   = isinstance(rot,   list) and any(abs(r)       > 1e-4 for r in rot)
            what = []
            if scale_bad: what.append("scale")
            if rot_bad:   what.append("rotation")
            issues.append({
                "category":    "transforms",
                "severity":    "error",
                "object_name": name,
                "description": f"Unapplied {'+'.join(what) or 'transform'} on '{name}' — will break FBX/glTF export and physics simulations.",
                "fix_command": (
                    f"bpy.context.view_layer.objects.active = bpy.data.objects[{name!r}]\n"
                    f"bpy.ops.object.transform_apply(location=False, rotation={str(rot_bad).capitalize()}, scale={str(scale_bad).capitalize()})"
                ),
                "auto_fixable": True,
            })
        else:
            # Legacy fallback for older plugins that don't send has_unapplied_transform
            scale = obj.get("scale", [1, 1, 1])
            if isinstance(scale, list) and any(abs(s - 1.0) > 1e-4 for s in scale):
                issues.append({
                    "category":    "transforms",
                    "severity":    "error",
                    "object_name": name,
                    "description": f"Unapplied scale {scale} on '{name}' — will break FBX/glTF export and physics.",
                    "fix_command": (
                        f"bpy.context.view_layer.objects.active = bpy.data.objects[{name!r}]\n"
                        "bpy.ops.object.transform_apply(scale=True)"
                    ),
                    "auto_fixable": True,
                })

    # ── Check 2: default names ────────────────────────────────────────────────
    import re
    DEFAULT_NAME = re.compile(
        r"^(Cube|Sphere|Cylinder|Cone|Torus|Plane|Camera|Light|Armature|Empty|Text)(\.\d+)?$",
        re.IGNORECASE,
    )
    for obj in objects:
        name = obj.get("name", "")
        if DEFAULT_NAME.match(name):
            warnings.append({
                "category": "naming",
                "severity": "warning",
                "object_name": name,
                "description": f"'{name}' is a default Blender name. Rename it before export.",
                "fix_command": "",
                "auto_fixable": False,
            })

    # ── Check 3: missing materials on mesh objects ────────────────────────────
    for obj in objects:
        if obj.get("type") == "MESH" and not obj.get("materials"):
            warnings.append({
                "category": "materials",
                "severity": "warning",
                "object_name": obj.get("name", "unknown"),
                "description": "Mesh has no material assigned. Will export as white/default.",
                "fix_command": "",
                "auto_fixable": False,
            })

    # ── Check 4: scene scale sanity ───────────────────────────────────────────
    # SceneContext v1: units.scale_length — fallback to legacy unit_scale field
    units = ctx.get("units", {})
    unit_scale = (
        units.get("scale_length")
        if isinstance(units, dict)
        else ctx.get("unit_scale", 1.0)
    )
    if unit_scale is None:
        unit_scale = 1.0
    if isinstance(unit_scale, (int, float)) and unit_scale not in (0.01, 0.001, 1.0):
        info.append(
            f"Scene unit scale is {unit_scale} (units.scale_length). "
            "Standard values: 1.0 (meters), 0.01 (centimeters), 0.001 (millimeters). "
            "Ensure this matches your target engine's expected scale."
        )

    # ── Check 5: render engine ────────────────────────────────────────────────
    render_engine = ctx.get("render_engine", "")
    if render_engine == "BLENDER_EEVEE":
        info.append("Using EEVEE — switch to Cycles for physically accurate renders and bakes.")

    # ── Check 6: evaluate the blender_python snippet if provided ─────────────
    if req.blender_python:
        snippet = req.blender_python.strip()
        # Static analysis: flag use of deprecated ops
        deprecated = ["bpy.ops.mesh.remove_doubles"]
        for dep in deprecated:
            if dep in snippet:
                warnings.append({
                    "category": "code",
                    "severity": "warning",
                    "object_name": "code_snippet",
                    "description": f"'{dep}' is deprecated in Blender 4.x. Use bpy.ops.mesh.merge_by_distance() instead.",
                    "fix_command": snippet.replace(dep, "bpy.ops.mesh.merge_by_distance"),
                    "auto_fixable": True,
                })
        # Check for hardcoded object names (brittleness warning)
        if re.search(r'bpy\.data\.objects\["[^"]+"\]', snippet):
            info.append("Code uses hardcoded object names — may fail if scene naming changes.")

    # ── Compile fix code ──────────────────────────────────────────────────────
    fix_code = ""
    if req.auto_fix:
        fix_parts = [
            i["fix_command"] for i in issues + warnings
            if i.get("auto_fixable") and i.get("fix_command")
        ]
        fix_code = "\n".join(fix_parts)

    # ── Scoring ───────────────────────────────────────────────────────────────
    error_count   = len(issues)
    warning_count = len(warnings)
    score = max(0.0, 100.0 - (error_count * 15) - (warning_count * 5))
    passed = error_count == 0

    latency = (time.perf_counter() - t0) * 1000

    logger.info("qa response", extra={
        "request_id":  request_id,
        "score":       round(score, 1),
        "errors":      error_count,
        "warnings":    warning_count,
        "latency_ms":  round(latency, 1),
    })

    return QAResponse(
        passed        = passed,
        score         = round(score, 2),
        issue_count   = error_count,
        warning_count = warning_count,
        issues        = issues,
        warnings      = warnings,
        info          = info,
        stats         = {
            "object_count":   len(objects),
            "profile":        req.profile,
        },
        fix_code      = fix_code,
        request_id    = request_id,
        latency_ms    = round(latency, 1),
    )


@app.get("/v1/capabilities")
async def capabilities() -> dict:
    """
    Returns supported DSL operations per compiler target and QA checks per mode.

    Intended use: plugin UI can grey out unsupported ops; CI can verify coverage.

    Response schema:
    {
      "schema_version": "1",
      "compilers": {
        "<software>": {
          "supported_ops": [str, ...],
          "status": "full | partial | stub"
        }
      },
      "qa_checks": [str, ...],
      "modes": [str, ...]
    }
    """
    # Ops supported by each compiler (derived from universal_dsl.py match blocks)
    BLENDER_OPS = [
        "ADD_CUBE", "ADD_SPHERE", "ADD_CYLINDER", "ADD_PLANE", "ADD_TORUS", "ADD_CONE",
        "ADD_EMPTY", "ADD_ARMATURE", "ADD_CURVE",
        "TRANSLATE", "ROTATE", "SCALE", "APPLY_TRANSFORMS", "SET_ORIGIN",
        "ENTER_EDIT_MODE", "ENTER_OBJECT_MODE", "ENTER_SCULPT_MODE", "ENTER_WEIGHT_PAINT",
        "EXTRUDE", "INSET", "BEVEL", "LOOP_CUT", "SUBDIVIDE", "KNIFE", "BRIDGE",
        "FILL", "DISSOLVE_EDGES", "DISSOLVE_FACES", "MERGE_VERTICES", "REMOVE_DOUBLES",
        "FLIP_NORMALS", "RECALC_NORMALS", "SELECT_ALL", "DESELECT_ALL", "SEPARATE",
        "ADD_SUBDIVISION", "ADD_MIRROR", "ADD_SOLIDIFY", "ADD_BEVEL_MOD", "ADD_ARRAY",
        "ADD_BOOLEAN", "ADD_SHRINKWRAP", "ADD_DECIMATE", "ADD_REMESH", "APPLY_MODIFIER",
        "SHADE_SMOOTH", "SHADE_FLAT", "DUPLICATE", "DELETE", "JOIN", "PARENT_SET",
        "UNWRAP_UV", "SMART_UV_PROJECT", "ADD_MATERIAL",
        "VOXEL_REMESH", "DYNAMIC_TOPOLOGY", "RENDER", "SET_RENDER_ENGINE",
        "ADD_LIGHT", "THREE_POINT_LIGHTING", "ADD_HDRI",
    ]
    MAYA_OPS = [
        "ADD_CUBE", "ADD_SPHERE", "ADD_CYLINDER", "EXTRUDE", "BEVEL",
        "SUBDIVIDE", "ADD_SUBDIVISION", "SHADE_SMOOTH", "SHADE_FLAT",
    ]
    HOUDINI_OPS = [
        "ADD_CUBE", "ADD_SPHERE", "ADD_CYLINDER", "EXTRUDE", "SUBDIVIDE",
        "ADD_SUBDIVISION", "ADD_BEVEL_MOD",
    ]
    C4D_OPS = [
        "ADD_CUBE", "ADD_SPHERE", "ADD_CYLINDER", "EXTRUDE", "SUBDIVIDE",
    ]
    RHINO_OPS = [
        "ADD_CUBE", "ADD_SPHERE", "ADD_CYLINDER", "EXTRUDE",
    ]

    return {
        "schema_version": "1",
        "compilers": {
            "blender": {
                "supported_ops": BLENDER_OPS,
                "op_count": len(BLENDER_OPS),
                "status": "full",
            },
            "maya": {
                "supported_ops": MAYA_OPS,
                "op_count": len(MAYA_OPS),
                "status": "partial",
                "note": "Core modeling ops. Simulation/rigging ops planned for v2.",
            },
            "houdini": {
                "supported_ops": HOUDINI_OPS,
                "op_count": len(HOUDINI_OPS),
                "status": "partial",
                "note": "Core ops. VEX/SOP network generation planned for v2.",
            },
            "cinema4d": {
                "supported_ops": C4D_OPS,
                "op_count": len(C4D_OPS),
                "status": "partial",
                "note": "Core ops. MoGraph and dynamics planned for v2.",
            },
            "rhino": {
                "supported_ops": RHINO_OPS,
                "op_count": len(RHINO_OPS),
                "status": "partial",
                "note": "Core ops. Grasshopper node generation planned for v2.",
            },
        },
        "qa_checks": [
            "unapplied_transforms",
            "default_names",
            "missing_materials",
            "unit_scale_sanity",
            "render_engine",
            "deprecated_ops_static_analysis",
        ],
        "modes": [
            "game_pc",
            "mobile",
            "cinematics",
            "arch_viz",
            "print_3d",
            "cad",
        ],
        "scene_context_schema_version": "1",
        "endpoints": [
            "POST /v1/command",
            "POST /v1/plan",
            "POST /v1/materialize",
            "POST /v1/qa",
            "GET  /v1/capabilities",
            "GET  /v1/health",
        ],
    }


@app.get("/v1/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check. Returns model name, GPU count, and status."""
    gpu_names = []
    try:
        for i in range(GPU_COUNT):
            gpu_names.append(torch.cuda.get_device_name(i))
    except Exception:
        pass

    return HealthResponse(
        status    = "ok",
        model     = state.model_name,
        gpu_count = GPU_COUNT,
        gpu_names = gpu_names,
    )


# ─── Global exception handler ─────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = str(uuid.uuid4())
    logger.error(f"Unhandled exception: {exc}", extra={
        "request_id": request_id,
        "path":       request.url.path,
        "exc_type":   type(exc).__name__,
    }, exc_info=True)
    return JSONResponse(
        status_code = 500,
        content     = {
            "error":      "Internal server error",
            "request_id": request_id,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code = exc.status_code,
        content     = {"error": exc.detail},
        headers     = exc.headers or {},
    )


# ─── Request logging middleware ───────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    latency  = (time.perf_counter() - t0) * 1000
    logger.info("http request", extra={
        "method":     request.method,
        "path":       request.url.path,
        "status":     response.status_code,
        "latency_ms": round(latency, 1),
    })
    return response
