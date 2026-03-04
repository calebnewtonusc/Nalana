"""
Nalana Rhino Plugin
Natural language / voice command interface for McNeel Rhino 3D via the Nalana API.

Rhino runs on .NET and ships IronPython (Rhino 7) or CPython (Rhino 8).
This script uses System.Windows.Forms for the UI, which is always available in Rhino.

Installation:
  Tools → PythonScript → Run → select this file
  The command "Nalana" becomes available in the command bar.
  Rhino 8 (CPython): drop this file in the Rhino scripts folder and it auto-registers.
"""

import sys
import json
import traceback
import urllib.request
import urllib.error

# Rhino / RhinoCommon imports.
try:
    import rhinoscriptsyntax as rs
    import Rhino
    import Rhino.Geometry as rg
    import scriptcontext
    _IN_RHINO = True
except ImportError:
    _IN_RHINO = False

# System.Windows.Forms — always available in Rhino's .NET runtime.
try:
    import System
    import System.Windows.Forms as WinForms
    import System.Drawing as Drawing
    _HAS_WINFORMS = True
except ImportError:
    _HAS_WINFORMS = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CONFIG = {
    "api_url": "http://localhost:8000",
    "api_key": "",
    "anthropic_key": "",
}

# Global form reference.
_form_ref = None

# ---------------------------------------------------------------------------
# Context helper
# ---------------------------------------------------------------------------


def get_rhino_context() -> dict:
    """Return a snapshot of the current Rhino document state."""
    if not _IN_RHINO:
        return {}

    try:
        selected_ids = rs.SelectedObjects() or []
        selected_names = []
        for obj_id in selected_ids:
            name = rs.ObjectName(obj_id)
            selected_names.append(name if name else str(obj_id))
    except Exception:
        selected_ids = []
        selected_names = []

    try:
        current_layer = rs.CurrentLayer()
    except Exception:
        current_layer = "unknown"

    try:
        doc = Rhino.RhinoDoc.ActiveDoc
        total_objects = doc.Objects.Count if doc else 0
    except Exception:
        total_objects = 0

    try:
        active_view = Rhino.RhinoDoc.ActiveDoc.Views.ActiveView
        viewport_name = active_view.ActiveViewport.Name if active_view else "unknown"
    except Exception:
        viewport_name = "unknown"

    return {
        "selected_objects": selected_names,
        "selected_count": len(selected_ids),
        "current_layer": current_layer,
        "object_count": total_objects,
        "active_viewport": viewport_name,
    }


# ---------------------------------------------------------------------------
# API communication
# ---------------------------------------------------------------------------


def call_nalana_api(voice_command: str, scene_context: dict) -> dict:
    """POST to the Nalana API and return the parsed JSON response."""
    endpoint = _CONFIG["api_url"].rstrip("/") + "/v1/command"
    payload = json.dumps({
        "voice_command": voice_command,
        "scene_context": scene_context,
        "software": "rhino",
    }).encode("utf-8")

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
        "You are a Rhino Python (rhinoscriptsyntax) expert. "
        "Reply with ONLY a JSON object with key 'rhino_python' whose value is "
        "executable rhinoscriptsyntax Python code for the requested operation. "
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
        return {"rhino_python": raw, "reasoning": "Claude fallback", "task_type": "unknown"}


def execute_code_safely(code: str) -> tuple:
    """
    Execute Rhino Python code.
    Returns (success: bool, error_message: str).
    """
    exec_globals: dict = {"__builtins__": __builtins__}
    if _IN_RHINO:
        exec_globals.update({"rs": rs, "Rhino": Rhino, "rg": rg, "scriptcontext": scriptcontext})
    try:
        exec(code, exec_globals)  # noqa: S102
        return True, ""
    except Exception:
        return False, traceback.format_exc()


def send_command(command: str) -> tuple:
    """Orchestrate API call + code execution. Returns (success, message)."""
    scene_ctx = get_rhino_context()

    response = None
    try:
        response = call_nalana_api(command, scene_ctx)
    except Exception as api_err:
        try:
            response = call_claude_fallback(command, scene_ctx)
        except Exception as claude_err:
            return False, f"All APIs failed: {claude_err}"

    code = response.get("rhino_python") or response.get("code") or ""
    if not code:
        return False, "API returned no executable code."

    return execute_code_safely(code)


# ---------------------------------------------------------------------------
# Windows Forms UI
# ---------------------------------------------------------------------------


if _HAS_WINFORMS:

    class NalanaForm(WinForms.Form):
        """Windows Forms dialog for Nalana in Rhino."""

        def __init__(self):
            super().__init__()
            self._history: list = []
            self._init_form()

        def _init_form(self):
            self.Text = "Nalana — Rhino AI Assistant"
            self.Width = 480
            self.Height = 540
            self.StartPosition = WinForms.FormStartPosition.CenterScreen
            self.FormBorderStyle = WinForms.FormBorderStyle.Sizable
            self.TopMost = True

            y = 10

            # Settings group
            settings_box = WinForms.GroupBox()
            settings_box.Text = "Settings"
            settings_box.Left = 10
            settings_box.Top = y
            settings_box.Width = 440
            settings_box.Height = 80
            self.Controls.Add(settings_box)

            lbl_url = WinForms.Label()
            lbl_url.Text = "API URL:"
            lbl_url.Left = 8
            lbl_url.Top = 22
            lbl_url.Width = 60
            settings_box.Controls.Add(lbl_url)

            self.api_url_box = WinForms.TextBox()
            self.api_url_box.Text = _CONFIG["api_url"]
            self.api_url_box.Left = 70
            self.api_url_box.Top = 20
            self.api_url_box.Width = 350
            settings_box.Controls.Add(self.api_url_box)

            lbl_key = WinForms.Label()
            lbl_key.Text = "API Key:"
            lbl_key.Left = 8
            lbl_key.Top = 50
            lbl_key.Width = 60
            settings_box.Controls.Add(lbl_key)

            self.api_key_box = WinForms.TextBox()
            self.api_key_box.PasswordChar = "*"
            self.api_key_box.Text = _CONFIG["api_key"]
            self.api_key_box.Left = 70
            self.api_key_box.Top = 48
            self.api_key_box.Width = 350
            settings_box.Controls.Add(self.api_key_box)

            y += 90

            # Command input
            cmd_box = WinForms.GroupBox()
            cmd_box.Text = "Command"
            cmd_box.Left = 10
            cmd_box.Top = y
            cmd_box.Width = 440
            cmd_box.Height = 80
            self.Controls.Add(cmd_box)

            self.cmd_textbox = WinForms.TextBox()
            self.cmd_textbox.Left = 8
            self.cmd_textbox.Top = 22
            self.cmd_textbox.Width = 418
            self.cmd_textbox.KeyDown += self._on_key_down
            cmd_box.Controls.Add(self.cmd_textbox)

            self.send_btn = WinForms.Button()
            self.send_btn.Text = "Send to Nalana"
            self.send_btn.Left = 8
            self.send_btn.Top = 50
            self.send_btn.Width = 130
            self.send_btn.Click += self._on_send
            cmd_box.Controls.Add(self.send_btn)

            self.clear_btn = WinForms.Button()
            self.clear_btn.Text = "Clear History"
            self.clear_btn.Left = 150
            self.clear_btn.Top = 50
            self.clear_btn.Width = 110
            self.clear_btn.Click += self._on_clear
            cmd_box.Controls.Add(self.clear_btn)

            y += 90

            # Status label
            self.status_label = WinForms.Label()
            self.status_label.Text = "Ready."
            self.status_label.Left = 12
            self.status_label.Top = y
            self.status_label.Width = 440
            self.status_label.ForeColor = Drawing.Color.Gray
            self.Controls.Add(self.status_label)

            y += 24

            # History listbox
            history_group = WinForms.GroupBox()
            history_group.Text = "Command History (last 10)"
            history_group.Left = 10
            history_group.Top = y
            history_group.Width = 440
            history_group.Height = 220
            self.Controls.Add(history_group)

            self.history_listbox = WinForms.ListBox()
            self.history_listbox.Left = 8
            self.history_listbox.Top = 20
            self.history_listbox.Width = 420
            self.history_listbox.Height = 185
            history_group.Controls.Add(self.history_listbox)

        # ------------------------------------------------------------------
        # Event handlers
        # ------------------------------------------------------------------

        def _on_key_down(self, sender, e):
            if e.KeyCode == WinForms.Keys.Return:
                self._on_send(sender, e)

        def _on_send(self, sender, e):
            _CONFIG["api_url"] = self.api_url_box.Text.strip() or _CONFIG["api_url"]
            _CONFIG["api_key"] = self.api_key_box.Text.strip()

            command = self.cmd_textbox.Text.strip()
            if not command:
                self.status_label.Text = "Enter a command first."
                return

            self.status_label.Text = "Sending to Nalana…"
            WinForms.Application.DoEvents()

            success, message = send_command(command)

            if success:
                self._push_history(command)
                self.cmd_textbox.Text = ""
                self.status_label.Text = f"Done: {command}"
            else:
                self.status_label.Text = "Error — see Rhino console for details."
                print(f"[Nalana] {message}")

        def _on_clear(self, sender, e):
            self._history.clear()
            self.history_listbox.Items.Clear()
            self.status_label.Text = "History cleared."

        def _push_history(self, command: str):
            self._history.append(command)
            self._history = self._history[-10:]
            self.history_listbox.Items.Clear()
            for cmd in reversed(self._history):
                self.history_listbox.Items.Add(cmd)

else:
    # Fallback stub when WinForms is not available.
    class NalanaForm:  # noqa: F811
        def __init__(self):
            pass


# ---------------------------------------------------------------------------
# Public API / Rhino command entry point
# ---------------------------------------------------------------------------


def RunCommand():
    """
    Entry point called when the user types 'Nalana' in the Rhino command bar.
    Also works as a standalone function for testing.
    """
    global _form_ref

    if not _HAS_WINFORMS:
        print("[Nalana] System.Windows.Forms not available — cannot display UI.")
        if _IN_RHINO:
            Rhino.UI.Dialogs.ShowMessage(
                "PySide2/WinForms not available — cannot display Nalana UI.",
                "Nalana",
            )
        return

    if _form_ref is None or _form_ref.IsDisposed:
        _form_ref = NalanaForm()

    _form_ref.Show()
    _form_ref.BringToFront()


# ---------------------------------------------------------------------------
# Rhino command registration (Rhino 8 CPython approach)
# ---------------------------------------------------------------------------

if _IN_RHINO:
    try:
        import rhinoscript.userinterface  # type: ignore

        class NalanaCommand(Rhino.Commands.Command):
            """Rhino command class — registers 'Nalana' in the command bar."""

            @property
            def EnglishName(self):
                return "Nalana"

            def RunCommand(self, doc, mode):
                RunCommand()
                return Rhino.Commands.Result.Success

    except Exception:
        # Rhino 7 / IronPython — command registration handled differently.
        pass

# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if _HAS_WINFORMS:
        form = NalanaForm()
        WinForms.Application.Run(form)
    else:
        # Fallback: simple CLI interaction.
        print("[Nalana] Running in CLI mode (WinForms not available).")
        while True:
            command = input("Nalana> ").strip()
            if command.lower() in ("exit", "quit"):
                break
            success, message = send_command(command)
            print("OK" if success else f"Error: {message}")
