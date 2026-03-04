"""
Nalana Blender 4.x Addon
Natural language / voice command interface for Blender via the Nalana API.
Falls back to Claude (Anthropic) if the Nalana backend is unreachable.
"""

bl_info = {
    "name": "Nalana",
    "author": "Nalana Team",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Nalana",
    "description": "Voice and text command interface powered by Nalana AI",
    "category": "3D View",
}

import bpy
import json
import threading
import traceback
import urllib.request
import urllib.error
from bpy.props import StringProperty, EnumProperty, IntProperty
from bpy.types import AddonPreferences, Operator, Panel, PropertyGroup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NALANA_HISTORY_PROP = "nalana_history_json"
MAX_HISTORY = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_scene_context() -> dict:
    """
    Collect Blender scene data conforming to SceneContext v1.

    Schema (same structure expected by /v1/command and /v1/qa):
    {
      "schema_version": "1",
      "software": "blender",
      "software_version": "4.2.0",
      "active_object": str | null,
      "selected_objects": [str, ...],
      "mode": str,           // OBJECT | EDIT_MESH | SCULPT | WEIGHT_PAINT | ...
      "frame_current": int,
      "render_engine": str,  // CYCLES | BLENDER_EEVEE_NEXT | BLENDER_WORKBENCH
      "units": {"system": str, "scale_length": float},
      "objects": [
        {
          "name": str,
          "type": str,       // MESH | CURVE | ARMATURE | LIGHT | CAMERA | EMPTY | ...
          "location": [x, y, z],
          "rotation": [x, y, z],   // Euler XYZ in radians
          "scale": [x, y, z],
          "dimensions": [x, y, z],
          "material_count": int,
          "materials": [str, ...],
          "vertex_count": int | null,   // null for non-mesh objects
          "face_count": int | null,
          "has_unapplied_transform": bool,
          "visible": bool
        },
        ...
      ]
    }
    """
    scene = bpy.context.scene
    active = bpy.context.active_object
    selected_names = [obj.name for obj in bpy.context.selected_objects]

    # Per-object details
    objects_info = []
    for obj in scene.objects:
        mat_names = [slot.material.name for slot in obj.material_slots if slot.material]
        mesh_verts = mesh_faces = None
        if obj.type == "MESH" and obj.data:
            mesh_verts = len(obj.data.vertices)
            mesh_faces = len(obj.data.polygons)

        # An object has an "unapplied transform" if scale or rotation is non-identity
        import math

        sx, sy, sz = obj.scale
        rx, ry, rz = obj.rotation_euler
        has_unapplied = (
            abs(sx - 1.0) > 1e-4
            or abs(sy - 1.0) > 1e-4
            or abs(sz - 1.0) > 1e-4
            or abs(rx) > 1e-4
            or abs(ry) > 1e-4
            or abs(rz) > 1e-4
        )

        objects_info.append(
            {
                "name": obj.name,
                "type": obj.type,
                "location": [round(v, 4) for v in obj.location],
                "rotation": [round(v, 4) for v in obj.rotation_euler],
                "scale": [round(v, 4) for v in obj.scale],
                "dimensions": [round(v, 4) for v in obj.dimensions],
                "material_count": len(mat_names),
                "materials": mat_names,
                "vertex_count": mesh_verts,
                "face_count": mesh_faces,
                "has_unapplied_transform": has_unapplied,
                "visible": obj.visible_get(),
            }
        )

    return {
        "schema_version": "1",
        "software": "blender",
        "software_version": ".".join(str(v) for v in bpy.app.version),
        "active_object": active.name if active else None,
        "selected_objects": selected_names,
        "mode": bpy.context.mode,
        "frame_current": scene.frame_current,
        "render_engine": scene.render.engine,
        "units": {
            "system": scene.unit_settings.system,
            "scale_length": scene.unit_settings.scale_length,
        },
        "objects": objects_info,
    }


def execute_code_safely(code: str, operator=None) -> bool:
    """
    Execute arbitrary Python code returned by the Nalana API inside Blender.
    Reports errors via the operator report mechanism or print fallback.
    Returns True on success, False on failure.
    """
    try:
        exec(code, {"bpy": bpy, "__builtins__": __builtins__})  # noqa: S102
        return True
    except Exception:
        msg = traceback.format_exc()
        if operator:
            operator.report({"ERROR"}, f"Nalana code execution failed:\n{msg}")
        else:
            print(f"[Nalana] Code execution failed:\n{msg}")
        return False


def get_history(scene) -> list:
    """Load command history stored as JSON in a scene custom property."""
    raw = scene.get(NALANA_HISTORY_PROP, "[]")
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []


def push_history(scene, command: str):
    """Append a command to the scene history (capped at MAX_HISTORY)."""
    history = get_history(scene)
    history.append(command)
    history = history[-MAX_HISTORY:]
    scene[NALANA_HISTORY_PROP] = json.dumps(history)


def call_nalana_api(
    api_url: str, api_key: str, voice_command: str, scene_context: dict, software: str
) -> dict:
    """
    POST to the Nalana API and return the parsed JSON response.
    Raises urllib.error.URLError or ValueError on failure.
    """
    endpoint = api_url.rstrip("/") + "/v1/command"
    payload = json.dumps(
        {
            "voice_command": voice_command,
            "scene_context": scene_context,
            "software": software,
        }
    ).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_claude_fallback(
    anthropic_key: str, voice_command: str, scene_context: dict
) -> dict:
    """
    Fallback: send the command directly to Claude (claude-sonnet-4-6) when the
    Nalana backend is not reachable. Returns a dict with a 'blender_python' key.
    Requires the anthropic package to be installed in Blender's Python.
    """
    try:
        import anthropic  # type: ignore
    except ImportError:
        raise RuntimeError(
            "The 'anthropic' package is not installed in Blender's Python. "
            "Install it via: <blender_python> -m pip install anthropic"
        )

    client = anthropic.Anthropic(api_key=anthropic_key)
    system_prompt = (
        "You are a Blender 4.x Python expert. "
        "The user will describe a 3D operation in plain English. "
        "Reply with ONLY a JSON object containing one key: 'blender_python' "
        "whose value is executable bpy Python code that performs the requested operation. "
        "Do not include markdown fences. Scene context is provided as a JSON object."
    )
    user_msg = f"Scene context: {json.dumps(scene_context)}\n\nCommand: {voice_command}"
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": user_msg}],
        system=system_prompt,
    )
    raw = message.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Wrap plain code in expected structure
        return {
            "blender_python": raw,
            "reasoning": "Claude fallback (raw code)",
            "task_type": "unknown",
        }


# ---------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------


class NalanaPreferences(AddonPreferences):
    bl_idname = __name__

    api_url: StringProperty(
        name="API URL",
        description="Base URL of the Nalana API server",
        default="http://localhost:8000",
    )

    api_key: StringProperty(
        name="API Key",
        description="Bearer token for authenticating with the Nalana API",
        default="",
        subtype="PASSWORD",
    )

    software_target: EnumProperty(
        name="Software Target",
        description="3D software to target when generating code",
        items=[
            ("blender", "Blender", "Generate bpy Python code"),
            ("maya", "Maya", "Generate Maya cmds code"),
        ],
        default="blender",
    )

    whisper_path: StringProperty(
        name="Whisper CLI Path",
        description="Path to the whisper executable or 'whisper' if on PATH",
        default="whisper",
        subtype="FILE_PATH",
    )

    anthropic_key: StringProperty(
        name="Anthropic API Key (Fallback)",
        description="Used when the Nalana server is unreachable",
        default="",
        subtype="PASSWORD",
    )

    show_settings: bpy.props.BoolProperty(
        name="Show Settings",
        default=False,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "api_url")
        layout.prop(self, "api_key")
        layout.prop(self, "software_target")
        layout.prop(self, "whisper_path")
        layout.separator()
        layout.label(text="Fallback (Claude API):")
        layout.prop(self, "anthropic_key")


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


class NALANA_OT_Execute(Operator):
    """Send the current command text to Nalana and execute the returned code."""

    bl_idname = "nalana.execute"
    bl_label = "Execute Command"
    bl_description = "Send the command to Nalana AI and execute the result"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        prefs = context.preferences.addons[__name__].preferences
        scene = context.scene
        command = scene.nalana_command.strip()

        if not command:
            self.report({"WARNING"}, "No command entered.")
            return {"CANCELLED"}

        scene_ctx = get_scene_context()

        # Attempt Nalana API, fall back to Claude.
        response = None
        try:
            response = call_nalana_api(
                prefs.api_url,
                prefs.api_key,
                command,
                scene_ctx,
                prefs.software_target,
            )
            self.report({"INFO"}, "Nalana: received response from server.")
        except Exception as api_err:
            self.report(
                {"WARNING"},
                f"Nalana API unreachable ({api_err}), trying Claude fallback…",
            )
            try:
                response = call_claude_fallback(
                    prefs.anthropic_key,
                    command,
                    scene_ctx,
                )
            except Exception as claude_err:
                self.report(
                    {"ERROR"},
                    f"Both Nalana API and Claude fallback failed: {claude_err}",
                )
                return {"CANCELLED"}

        # Execute the code block returned by the API.
        code = response.get("blender_python") or response.get("code") or ""
        if not code:
            self.report({"WARNING"}, "Nalana returned no executable code.")
            return {"CANCELLED"}

        success = execute_code_safely(code, operator=self)
        if success:
            push_history(scene, command)
            scene.nalana_command = ""
            self.report({"INFO"}, f"Nalana: executed '{command}' successfully.")
            return {"FINISHED"}

        return {"CANCELLED"}


class NALANA_OT_Record(Operator):
    """Record audio from the microphone and transcribe it using Whisper."""

    bl_idname = "nalana.record"
    bl_label = "Record Voice"
    bl_description = "Record a voice command and transcribe it"
    bl_options = {"REGISTER"}

    _SAMPLE_RATE = 16000
    _DURATION = 5  # seconds

    def execute(self, context):
        prefs = context.preferences.addons[__name__].preferences
        scene = context.scene

        import tempfile
        import os
        import subprocess

        wav_path = os.path.join(tempfile.gettempdir(), "nalana_record.wav")

        # Try sounddevice first, then fall back to pyaudio.
        recorded = False
        try:
            import sounddevice as sd  # type: ignore
            import numpy as np  # type: ignore
            import wave

            self.report({"INFO"}, f"Recording for {self._DURATION}s (sounddevice)…")
            audio = sd.rec(
                int(self._DURATION * self._SAMPLE_RATE),
                samplerate=self._SAMPLE_RATE,
                channels=1,
                dtype="int16",
            )
            sd.wait()
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self._SAMPLE_RATE)
                wf.writeframes(audio.tobytes())
            recorded = True
        except ImportError:
            pass

        if not recorded:
            try:
                import pyaudio  # type: ignore
                import wave

                self.report({"INFO"}, f"Recording for {self._DURATION}s (pyaudio)…")
                pa = pyaudio.PyAudio()
                stream = pa.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self._SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=1024,
                )
                frames = []
                for _ in range(0, int(self._SAMPLE_RATE / 1024 * self._DURATION)):
                    frames.append(stream.read(1024))
                stream.stop_stream()
                stream.close()
                pa.terminate()
                with wave.open(wav_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(self._SAMPLE_RATE)
                    wf.writeframes(b"".join(frames))
                recorded = True
            except ImportError:
                pass

        if not recorded:
            self.report(
                {"ERROR"},
                "Neither sounddevice nor pyaudio is available. Cannot record audio.",
            )
            return {"CANCELLED"}

        # Transcribe with whisper.
        try:
            import whisper as whisper_lib  # type: ignore

            model = whisper_lib.load_model("base")
            result = model.transcribe(wav_path)
            transcript = result.get("text", "").strip()
        except ImportError:
            # Fall back to whisper CLI.
            try:
                result = subprocess.run(
                    [
                        prefs.whisper_path,
                        wav_path,
                        "--output_format",
                        "txt",
                        "--output_dir",
                        tempfile.gettempdir(),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                txt_path = wav_path.replace(".wav", ".txt")
                if os.path.exists(txt_path):
                    with open(txt_path) as f:
                        transcript = f.read().strip()
                else:
                    transcript = result.stdout.strip()
            except Exception as e:
                self.report({"ERROR"}, f"Whisper transcription failed: {e}")
                return {"CANCELLED"}

        if transcript:
            scene.nalana_command = transcript
            self.report({"INFO"}, f"Transcribed: {transcript}")
        else:
            self.report({"WARNING"}, "Whisper returned an empty transcript.")

        return {"FINISHED"}


class NALANA_OT_ClearHistory(Operator):
    """Clear the Nalana command history for the current scene."""

    bl_idname = "nalana.clear_history"
    bl_label = "Clear History"
    bl_description = "Clear the Nalana command history"
    bl_options = {"REGISTER"}

    def execute(self, context):
        context.scene[NALANA_HISTORY_PROP] = "[]"
        self.report({"INFO"}, "Nalana history cleared.")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


class NALANA_PT_Panel(Panel):
    """Main Nalana sidebar panel in the 3D Viewport."""

    bl_label = "Nalana"
    bl_idname = "NALANA_PT_Panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Nalana"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        prefs = context.preferences.addons[__name__].preferences

        # Command input row
        col = layout.column(align=True)
        col.label(text="Command:")
        row = col.row(align=True)
        row.prop(scene, "nalana_command", text="")

        # Execute + mic buttons
        row2 = col.row(align=True)
        row2.operator("nalana.execute", text="Execute", icon="PLAY")
        row2.operator("nalana.record", text="", icon="MICROPHONE")

        layout.separator()

        # History
        history = get_history(scene)
        if history:
            box = layout.box()
            box.label(text="Recent Commands:", icon="TIME")
            for cmd in reversed(history):
                row = box.row()
                row.label(text=cmd, icon="RIGHTARROW_THIN")
            box.operator("nalana.clear_history", text="Clear History", icon="X")
        else:
            layout.label(text="No history yet.", icon="INFO")

        layout.separator()

        # Settings toggle
        row = layout.row()
        icon = "TRIA_DOWN" if prefs.show_settings else "TRIA_RIGHT"
        row.prop(prefs, "show_settings", icon=icon, emboss=False, text="Settings")

        if prefs.show_settings:
            box = layout.box()
            box.prop(prefs, "api_url")
            box.prop(prefs, "api_key")
            box.prop(prefs, "software_target")
            box.prop(prefs, "whisper_path")
            box.prop(prefs, "anthropic_key")


# ---------------------------------------------------------------------------
# Scene Properties
# ---------------------------------------------------------------------------


def register_properties():
    bpy.types.Scene.nalana_command = StringProperty(
        name="Nalana Command",
        description="Natural language command to send to Nalana",
        default="",
    )


def unregister_properties():
    try:
        del bpy.types.Scene.nalana_command
    except AttributeError:
        pass


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_CLASSES = [
    NalanaPreferences,
    NALANA_OT_Execute,
    NALANA_OT_Record,
    NALANA_OT_ClearHistory,
    NALANA_PT_Panel,
]


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    register_properties()


def unregister():
    unregister_properties()
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
