"""
Nalana Cinema 4D Plugin
Natural language / voice command interface for Maxon Cinema 4D via the Nalana API.

Installation:
  Copy this file (or a ZIP containing it) to your Cinema 4D plugins folder.
  Extensions → Plugin Manager → install → restart C4D.
  The plugin registers as a command plugin and appears in Extensions menu.

Plugin ID: 1000001 (placeholder — register a real ID at developer.maxon.net)
"""

import json
import traceback
import urllib.request
import urllib.error

# Cinema 4D imports — available only when running inside C4D.
try:
    import c4d
    import c4d.gui
    import c4d.plugins
    import c4d.documents

    _IN_C4D = True
except ImportError:
    _IN_C4D = False

    # Provide stubs so the module is importable for syntax checking.
    class _Stub:
        pass

    c4d = _Stub()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLUGIN_ID = 1000001  # Replace with your registered Maxon plugin ID.
PLUGIN_NAME = "Nalana"
PLUGIN_HELP = "AI-powered natural language interface for Cinema 4D"
PLUGIN_VERSION = "1.0.0"

# Dialog widget IDs
ID_CMD_EDITTEXT = 10001
ID_SEND_BUTTON = 10002
ID_HISTORY_LISTVIEW = 10003
ID_STATUS_TEXT = 10004
ID_API_URL_EDITTEXT = 10005
ID_API_KEY_EDITTEXT = 10006
ID_CLEAR_BUTTON = 10007

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CONFIG = {
    "api_url": "http://localhost:8000",
    "api_key": "",
    "anthropic_key": "",
}

# Global dialog reference to prevent GC.
_dialog_ref = None

# ---------------------------------------------------------------------------
# Context helper
# ---------------------------------------------------------------------------


def get_c4d_context() -> dict:
    """Return a snapshot of the current C4D scene state."""
    if not _IN_C4D:
        return {}

    doc = c4d.documents.GetActiveDocument()
    if doc is None:
        return {"error": "No active document"}

    active_objects = doc.GetActiveObjects(c4d.GETACTIVEOBJECTFLAGS_CHILDREN)
    active_names = [obj.GetName() for obj in active_objects] if active_objects else []

    # Active object (first selected)
    active_obj = doc.GetActiveObject()
    active_name = active_obj.GetName() if active_obj else None

    # Count all objects in the scene
    def count_objects(root):
        count = 0
        obj = root
        while obj:
            count += 1
            child = obj.GetDown()
            if child:
                count += count_objects(child)
            obj = obj.GetNext()
        return count

    total_objects = count_objects(doc.GetFirstObject()) if doc.GetFirstObject() else 0

    # Current tool
    try:
        current_tool = c4d.GetCommandName(c4d.GetActiveToolData()["tool"])
    except Exception:
        current_tool = "unknown"

    return {
        "active_object": active_name,
        "selected_objects": active_names,
        "object_count": total_objects,
        "current_tool": current_tool,
        "document_name": doc.GetDocumentName(),
        "frame_current": doc.GetTime().GetFrame(doc.GetFps()),
        "fps": doc.GetFps(),
    }


# ---------------------------------------------------------------------------
# API communication
# ---------------------------------------------------------------------------


def call_nalana_api(voice_command: str, scene_context: dict) -> dict:
    """POST to the Nalana API and return the parsed JSON response."""
    endpoint = _CONFIG["api_url"].rstrip("/") + "/v1/command"
    payload = json.dumps(
        {
            "voice_command": voice_command,
            "scene_context": scene_context,
            "software": "cinema4d",
        }
    ).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if _CONFIG["api_key"]:
        headers["Authorization"] = f"Bearer {_CONFIG['api_key']}"

    req = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_claude_fallback(voice_command: str, scene_context: dict) -> dict:
    """Fall back to Claude claude-sonnet-4-6 when the Nalana server is unreachable."""
    try:
        import anthropic  # type: ignore
    except ImportError:
        raise RuntimeError("'anthropic' package not available.")

    client = anthropic.Anthropic(api_key=_CONFIG["anthropic_key"])
    system_prompt = (
        "You are a Cinema 4D Python API expert. "
        "Reply with ONLY a JSON object with key 'c4d_python' whose value is "
        "executable c4d Python code for the requested operation. "
        "No markdown. Scene context provided as JSON."
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
        return {
            "c4d_python": raw,
            "reasoning": "Claude fallback",
            "task_type": "unknown",
        }


def send_command(command: str) -> tuple:
    """
    Send a command to Nalana (or Claude fallback), execute returned code.
    Returns (success: bool, message: str).
    """
    scene_ctx = get_c4d_context()

    response = None
    try:
        response = call_nalana_api(command, scene_ctx)
    except Exception:
        try:
            response = call_claude_fallback(command, scene_ctx)
        except Exception as claude_err:
            return False, f"All APIs failed: {claude_err}"

    code = response.get("c4d_python") or response.get("code") or ""
    if not code:
        return False, "API returned no executable code."

    try:
        exec(code, {"c4d": c4d, "__builtins__": __builtins__})  # noqa: S102
        return True, f"Executed: {command}"
    except Exception:
        return False, traceback.format_exc()


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


if _IN_C4D:

    class NalanaDialog(c4d.gui.GeDialog):
        """Main Nalana dialog window for Cinema 4D."""

        def __init__(self):
            super().__init__()
            self._history: list = []

        # ------------------------------------------------------------------
        # GeDialog overrides
        # ------------------------------------------------------------------

        def CreateLayout(self):
            self.SetTitle("Nalana — C4D AI Assistant")

            # Settings group
            self.GroupBegin(0, c4d.BFH_SCALEFIT, cols=2, title="Settings")
            self.AddStaticText(0, c4d.BFH_LEFT, name="API URL:")
            self.AddEditText(ID_API_URL_EDITTEXT, c4d.BFH_SCALEFIT, initw=260)
            self.AddStaticText(0, c4d.BFH_LEFT, name="API Key:")
            self.AddEditText(ID_API_KEY_EDITTEXT, c4d.BFH_SCALEFIT, initw=260)
            self.GroupEnd()

            self.AddSeparatorH(0)

            # Command input
            self.GroupBegin(0, c4d.BFH_SCALEFIT, cols=1, title="Command")
            self.AddEditText(ID_CMD_EDITTEXT, c4d.BFH_SCALEFIT, initw=380, inith=24)
            self.GroupBegin(0, c4d.BFH_SCALEFIT, cols=2)
            self.AddButton(ID_SEND_BUTTON, c4d.BFH_SCALEFIT, name="Send to Nalana")
            self.AddButton(ID_CLEAR_BUTTON, c4d.BFH_SCALEFIT, name="Clear History")
            self.GroupEnd()
            self.GroupEnd()

            self.AddSeparatorH(0)

            # Status
            self.AddStaticText(ID_STATUS_TEXT, c4d.BFH_SCALEFIT, name="Ready.")

            self.AddSeparatorH(0)

            # History list
            self.GroupBegin(
                0,
                c4d.BFH_SCALEFIT | c4d.BFV_SCALEFIT,
                cols=1,
                title="History (last 10)",
            )
            self.AddListView(
                ID_HISTORY_LISTVIEW,
                c4d.BFH_SCALEFIT | c4d.BFV_SCALEFIT,
                initw=380,
                inith=160,
            )
            self.GroupEnd()

            return True

        def InitValues(self):
            self.SetString(ID_API_URL_EDITTEXT, _CONFIG["api_url"])
            self.SetString(ID_API_KEY_EDITTEXT, _CONFIG["api_key"])
            self.SetString(ID_STATUS_TEXT, "Ready.")
            return True

        def Command(self, id, msg):
            if id == ID_SEND_BUTTON:
                self._on_send()
                return True
            if id == ID_CLEAR_BUTTON:
                self._clear_history()
                return True
            return True

        # ------------------------------------------------------------------
        # Internal helpers
        # ------------------------------------------------------------------

        def _on_send(self):
            """Handle the Send button press."""
            _CONFIG["api_url"] = (
                self.GetString(ID_API_URL_EDITTEXT).strip() or _CONFIG["api_url"]
            )
            _CONFIG["api_key"] = self.GetString(ID_API_KEY_EDITTEXT).strip()

            command = self.GetString(ID_CMD_EDITTEXT).strip()
            if not command:
                self.SetString(ID_STATUS_TEXT, "Enter a command first.")
                return

            self.SetString(ID_STATUS_TEXT, "Sending to Nalana…")
            success, message = send_command(command)

            if success:
                self._push_history(command)
                self.SetString(ID_CMD_EDITTEXT, "")
                self.SetString(ID_STATUS_TEXT, message)
            else:
                self.SetString(ID_STATUS_TEXT, f"Error: {message[:120]}")
                print(f"[Nalana] {message}")

        def _push_history(self, command: str):
            self._history.append(command)
            self._history = self._history[-10:]
            self._refresh_list()

        def _clear_history(self):
            self._history.clear()
            self._refresh_list()
            self.SetString(ID_STATUS_TEXT, "History cleared.")

        def _refresh_list(self):
            lv = self.FindCustomGui(ID_HISTORY_LISTVIEW, c4d.gui.CustomGuiListView)
            if lv is None:
                return
            lv.SetLayout(1, [c4d.LV_USER])
            for i, cmd in enumerate(reversed(self._history)):
                lv.SetItem(i, 0, cmd)
            lv.DataChanged()

    # -----------------------------------------------------------------------
    # Plugin command
    # -----------------------------------------------------------------------

    class NalanaPlugin(c4d.plugins.CommandPlugin):
        """Cinema 4D CommandPlugin that opens the Nalana dialog."""

        def Execute(self, doc):
            global _dialog_ref
            if _dialog_ref is None:
                _dialog_ref = NalanaDialog()
            _dialog_ref.Open(
                dlgtype=c4d.DLG_TYPE_ASYNC,
                pluginid=PLUGIN_ID,
                xpos=-2,
                ypos=-2,
                defaultw=420,
                defaulth=480,
            )
            return True

    # -----------------------------------------------------------------------
    # Plugin registration (executed at module import time inside C4D)
    # -----------------------------------------------------------------------

    if __name__ == "__main__":
        # Direct execution from Script Manager — open dialog immediately.
        _dialog_ref = NalanaDialog()
        _dialog_ref.Open(
            dlgtype=c4d.DLG_TYPE_ASYNC,
            pluginid=PLUGIN_ID,
            xpos=-2,
            ypos=-2,
            defaultw=420,
            defaulth=480,
        )
    else:
        # Loaded as a plugin — register the command.
        c4d.plugins.RegisterCommandPlugin(
            id=PLUGIN_ID,
            str=PLUGIN_NAME,
            info=0,
            icon=None,
            help=PLUGIN_HELP,
            dat=NalanaPlugin(),
        )
        print(
            f"[Nalana] Cinema 4D plugin v{PLUGIN_VERSION} registered (ID={PLUGIN_ID})."
        )

else:
    # Running outside C4D — define stubs so the file is importable.
    class NalanaDialog:  # noqa: F811
        pass

    class NalanaPlugin:  # noqa: F811
        pass
