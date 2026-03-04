"""
stb_integration.py - Drop-in Nalana replacement for STB's GPT-4o tier

HOW TO USE: In Clarence's voice_to_blender.py, replace the gpt_to_json() function:

    # OLD (expensive, generic):
    result = gpt_to_json(transcript)

    # NEW (free, 3D-specialized):
    from stb_integration import nalana_to_json
    result = nalana_to_json(transcript, scene_context=get_scene_context())

This preserves ALL of Clarence's existing plumbing:
  - faster-whisper STT ✅ (unchanged)
  - I/O rules tier ✅ (unchanged)
  - Local regex tier ✅ (unchanged)
  - XML-RPC bridge ✅ (unchanged)
  - Safety gate ✅ (unchanged)
  - Task queue ✅ (unchanged)
  - Meshy provider ✅ (unchanged)

Only the "brain" (GPT-4o → Nalana-v1) changes.
"""

import json
import urllib.request
import urllib.error
import os
import time
from typing import Optional

# ─── Configuration ────────────────────────────────────────────────────────────

NALANA_API_URL = os.environ.get("NALANA_API_URL", "http://localhost:8000")
NALANA_API_KEY = os.environ.get("NALANA_API_KEY", "")
NALANA_TIMEOUT = int(os.environ.get("NALANA_TIMEOUT", "30"))
NALANA_FALLBACK_TO_GPT = os.environ.get("NALANA_FALLBACK_TO_GPT", "true").lower() == "true"

# ─── Nalana API Client ────────────────────────────────────────────────────────

def nalana_to_json(
    transcript: str,
    scene_context: Optional[dict] = None,
    software: str = "blender",
    conversation_history: Optional[list] = None,
    openai_api_key: Optional[str] = None,
) -> dict:
    """
    Drop-in replacement for STB's gpt_to_json().

    Args:
        transcript:           The transcribed voice command (same as gpt_to_json input)
        scene_context:        Scene state dict from Blender (objects, active, mode, etc.)
        software:             Target software ("blender", "maya", "cinema4d", etc.)
        conversation_history: Prior turns for multi-turn conversation mode
        openai_api_key:       Fallback GPT-4o key if Nalana is unreachable

    Returns:
        Dict compatible with STB's existing command format:
        {"op": "mesh.primitive_cube_add", "kwargs": {"size": 2}}
        OR for multi-step:
        [{"op": "...", "kwargs": {}}, ...]
        PLUS Nalana extras:
        {"reasoning": "...", "task_type": "...", "physics_notes": "..."}
    """
    payload = {
        "voice_command": transcript,
        "scene_context": scene_context or {},
        "software": software,
        "conversation_history": conversation_history or [],
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if NALANA_API_KEY:
        headers["Authorization"] = f"Bearer {NALANA_API_KEY}"

    endpoint = NALANA_API_URL.rstrip("/") + "/v1/command"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=NALANA_TIMEOUT) as resp:
            nalana_response = json.loads(resp.read().decode("utf-8"))
            return _convert_nalana_to_stb_format(nalana_response)

    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"[Nalana] Server unreachable: {e}")

        if NALANA_FALLBACK_TO_GPT and openai_api_key:
            print("[Nalana] Falling back to GPT-4o...")
            return _gpt4o_fallback(transcript, scene_context, openai_api_key)

        # Last resort: return empty command so Blender doesn't crash
        return {"op": None, "kwargs": {}, "error": f"Nalana unreachable: {e}"}


def _convert_nalana_to_stb_format(nalana_response: dict) -> dict:
    """
    Convert Nalana's rich response format to STB's expected command format.

    Nalana returns:
    {
      "blender_python": "bpy.ops.mesh.primitive_cube_add(size=2)",
      "universal_dsl": {"op": "add_primitive", "args": {"type": "cube"}},
      "reasoning": "...",
      "task_type": "EXECUTE",
      "physics_notes": "...",
      "build_plan": [...],  # for BUILD tasks
    }

    STB expects:
    {"op": "mesh.primitive_cube_add", "kwargs": {"size": 2}}
    OR array of such dicts for multi-step.
    """
    task_type = nalana_response.get("task_type", "EXECUTE")

    # For BUILD tasks, return a plan as multiple ops
    if task_type == "BUILD" and nalana_response.get("build_plan"):
        plan = nalana_response["build_plan"]
        steps = []
        for step in plan:
            if isinstance(step, dict) and step.get("blender_python"):
                ops = _python_to_ops(step["blender_python"])
                steps.extend(ops)
        if steps:
            return steps

    # For all other tasks, extract from blender_python
    blender_python = nalana_response.get("blender_python", "")
    if blender_python:
        ops = _python_to_ops(blender_python)
        if len(ops) == 1:
            result = ops[0]
        elif len(ops) > 1:
            result = ops
        else:
            result = {"op": None, "kwargs": {}}

        # Attach Nalana's extra context for logging/UI display
        if isinstance(result, dict):
            result["_nalana_reasoning"] = nalana_response.get("reasoning", "")
            result["_nalana_task_type"] = task_type
            result["_nalana_physics"] = nalana_response.get("physics_notes", "")

        return result

    # Nalana returned an UNDERSTAND/CONVERSATION response (no code)
    return {
        "op": None,
        "kwargs": {},
        "_nalana_explanation": nalana_response.get("explanation", ""),
        "_nalana_task_type": task_type,
    }


def _python_to_ops(blender_python: str) -> list:
    """
    Parse blender_python code string into STB op/kwargs dicts.

    Handles the most common patterns:
      bpy.ops.mesh.primitive_cube_add(size=2)
      bpy.ops.transform.resize(value=(1, 1, 0.5))
    """
    import re

    ops = []
    # Match bpy.ops.MODULE.FUNCTION(ARGS) calls.
    # Use a balanced-paren approach to handle args containing nested parens
    # like value=(0, 0, 0.2) which the simpler [^)]* pattern would truncate.
    for m in re.finditer(r"bpy\.ops\.(\w+\.\w+)\(", blender_python):
        op_name = m.group(1)
        start = m.end()  # index just after opening '('
        depth = 1
        pos = start
        while pos < len(blender_python) and depth > 0:
            if blender_python[pos] == '(':
                depth += 1
            elif blender_python[pos] == ')':
                depth -= 1
            pos += 1
        args_str = blender_python[start:pos - 1]  # content between outer parens
        kwargs = _parse_kwargs(args_str)
        ops.append({"op": op_name, "kwargs": kwargs})
        op_name = match.group(1)  # e.g. "mesh.primitive_cube_add"
        args_str = match.group(2)  # e.g. "size=2, location=(0,0,0)"
        kwargs = _parse_kwargs(args_str)
        ops.append({"op": op_name, "kwargs": kwargs})

    return ops if ops else []


def _parse_kwargs(args_str: str) -> dict:
    """Parse 'size=2, location=(0,0,0)' into {'size': 2, 'location': (0,0,0)}"""
    import ast

    kwargs = {}
    if not args_str.strip():
        return kwargs

    # Wrap in a dict and parse
    try:
        result = ast.literal_eval(f"dict({args_str})")
        kwargs = result
    except Exception:
        # Fallback: split by commas and try key=value
        for part in args_str.split(","):
            part = part.strip()
            if "=" in part:
                key, _, val = part.partition("=")
                key = key.strip()
                val = val.strip()
                try:
                    kwargs[key] = ast.literal_eval(val)
                except Exception:
                    kwargs[key] = val  # keep as string

    return kwargs


def _gpt4o_fallback(transcript: str, scene_context: Optional[dict], api_key: str) -> dict:
    """
    Exact replica of STB's original gpt_to_json() for fallback.
    Only called when Nalana server is unreachable.
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        system_prompt = (
            "You are a Blender automation agent.\n"
            "Output ONLY raw JSON (no prose, no code fences).\n"
            "Each command must be of the form: {\"op\":\"<module.op>\",\"kwargs\":{}}.\n"
            "If multiple steps are implied, output a JSON array of such dicts.\n"
            "Prefer creative operators (object/mesh/curve/transform/material/node/render).\n"
            "Never use file/quit/addon/script/image.save operators."
        )

        user_content = transcript
        if scene_context:
            user_content = f"Scene: {json.dumps(scene_context)}\n\nCommand: {transcript}"

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()

        # Clean code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        return json.loads(raw)

    except Exception as e:
        print(f"[Nalana] GPT-4o fallback also failed: {e}")
        return {"op": None, "kwargs": {}, "error": str(e)}


# ─── Scene Context Extraction ─────────────────────────────────────────────────
# Call this inside Blender's Python context and pass result to nalana_to_json()

def get_nalana_scene_context() -> dict:
    """
    Extract rich scene context from Blender.
    Compatible with both STB's existing context AND Nalana's richer format.

    Call this from within Blender (where bpy is available).
    """
    try:
        import bpy

        active = bpy.context.active_object
        selected = bpy.context.selected_objects

        scene_context = {
            "software": "blender",
            "mode": bpy.context.mode,
            "active_object": None,
            "selected_objects": [],
            "object_count": len(bpy.data.objects),
            "scene_name": bpy.context.scene.name,
            "frame_current": bpy.context.scene.frame_current,
            "units": bpy.context.scene.unit_settings.system,
        }

        if active:
            scene_context["active_object"] = {
                "name": active.name,
                "type": active.type,
                "location": list(active.location),
                "scale": list(active.scale),
                "rotation_euler": list(active.rotation_euler),
                "dimensions": list(active.dimensions),
                "vertex_count": len(active.data.vertices) if hasattr(active.data, "vertices") else 0,
                "face_count": len(active.data.polygons) if hasattr(active.data, "polygons") else 0,
                "material_count": len(active.data.materials) if hasattr(active.data, "materials") else 0,
            }

        scene_context["selected_objects"] = [
            {
                "name": obj.name,
                "type": obj.type,
                "location": list(obj.location),
                "dimensions": list(obj.dimensions),
            }
            for obj in selected[:10]  # cap at 10 to avoid huge payloads
        ]

        return scene_context

    except ImportError:
        # Running outside Blender (testing)
        return {"software": "blender", "mode": "OBJECT", "note": "Called outside Blender"}


# ─── Multi-turn Session State ─────────────────────────────────────────────────
# Enables Nalana's conversation memory within a Blender session.

class NalanaSession:
    """
    Maintains conversation history for multi-turn interactions.

    Usage in voice_to_blender.py:
        session = NalanaSession()
        result = session.send("bevel these edges")
        result = session.send("now add a subdivision surface")  # Nalana remembers context
    """

    def __init__(self, max_history: int = 20):
        self.history: list = []
        self.max_history = max_history
        self._start_time = time.time()

    def send(
        self,
        transcript: str,
        scene_context: Optional[dict] = None,
        software: str = "blender",
        openai_api_key: Optional[str] = None,
    ) -> dict:
        """Send a command with full conversation history."""
        result = nalana_to_json(
            transcript=transcript,
            scene_context=scene_context,
            software=software,
            conversation_history=self.history[-self.max_history:],
            openai_api_key=openai_api_key,
        )

        # Append to history
        self.history.append({
            "role": "user",
            "content": transcript,
            "scene_context": scene_context,
        })
        self.history.append({
            "role": "assistant",
            "content": json.dumps(result),
        })

        return result

    def clear(self):
        """Clear conversation history (new task / new scene)."""
        self.history = []

    def summary(self) -> dict:
        """Return session stats."""
        return {
            "turns": len(self.history) // 2,
            "session_duration_s": int(time.time() - self._start_time),
        }


# ─── Health Check ─────────────────────────────────────────────────────────────

def is_nalana_running() -> bool:
    """Check if the Nalana API server is reachable."""
    try:
        url = NALANA_API_URL.rstrip("/") + "/v1/health"
        with urllib.request.urlopen(url, timeout=2) as resp:
            data = json.loads(resp.read())
            return data.get("status") == "ok"
    except Exception:
        return False


def nalana_model_info() -> dict:
    """Return info about the loaded Nalana model."""
    try:
        url = NALANA_API_URL.rstrip("/") + "/v1/health"
        with urllib.request.urlopen(url, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e), "nalana_running": False}


# ─── CLI Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print(f"Nalana server running: {is_nalana_running()}")

    if not is_nalana_running():
        print("Start the Nalana server first: cd deploy && docker compose up -d")
        print("Or locally: python deploy/api_server.py")
        sys.exit(1)

    print(f"Model info: {json.dumps(nalana_model_info(), indent=2)}")

    # Test commands
    test_commands = [
        "add a cube in the center",
        "bevel the selected edges with 3 segments",
        "make this look like aged copper",
        "why does gold look warm?",
        "set up studio lighting for a product shot",
    ]

    session = NalanaSession()
    for cmd in test_commands:
        print(f"\nCommand: {cmd}")
        result = session.send(cmd)
        print(f"Result: {json.dumps(result, indent=2)[:200]}")
