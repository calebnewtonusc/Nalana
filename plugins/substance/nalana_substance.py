"""
Nalana Substance Painter Plugin
Natural language / voice command interface for Adobe Substance Painter via the Nalana API.

Installation:
  Python → Plugin Manager → install → select this file (or a ZIP containing it) → restart.
  The Nalana panel appears as a dock widget in Substance Painter.

Substance Painter ships with PySide2, which this plugin uses for its UI.
"""

import sys
import json
import traceback
import urllib.request
import urllib.error

# Substance Painter imports.
try:
    import substance_painter as sp
    import substance_painter.ui
    import substance_painter.project
    import substance_painter.textureset
    import substance_painter.logging as sp_log

    _IN_SP = True
except ImportError:
    _IN_SP = False

# PySide2 — ships with Substance Painter.
try:
    from PySide2 import QtWidgets, QtCore, QtGui

    _HAS_PYSIDE = True
except ImportError:
    try:
        from PySide6 import QtWidgets, QtCore, QtGui  # type: ignore

        _HAS_PYSIDE = True
    except ImportError:
        _HAS_PYSIDE = False

# ---------------------------------------------------------------------------
# Plugin metadata — required by Substance Painter's plugin system.
# ---------------------------------------------------------------------------

PLUGIN_METADATA = {
    "name": "Nalana",
    "version": "1.0.0",
    "author": "Nalana Team",
    "description": "AI-powered natural language command interface for Substance Painter",
    "url": "https://github.com/nalana-team/nalana",
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CONFIG = {
    "api_url": "http://localhost:8000",
    "api_key": "",
    "anthropic_key": "",
}

# Global references.
_dock_widget = None
_panel_widget = None

# ---------------------------------------------------------------------------
# Context helper
# ---------------------------------------------------------------------------


def get_substance_context() -> dict:
    """Return a snapshot of the current Substance Painter project state."""
    if not _IN_SP:
        return {}

    ctx: dict = {}

    # Project name
    try:
        ctx["project_name"] = sp.project.name()
    except Exception:
        ctx["project_name"] = "unknown"

    # Texture sets
    try:
        all_sets = sp.textureset.all_texture_sets()
        ctx["texture_sets"] = [ts.name() for ts in all_sets]
        ctx["texture_set_count"] = len(all_sets)
    except Exception:
        ctx["texture_sets"] = []
        ctx["texture_set_count"] = 0

    # Active texture set
    try:
        active_ts = sp.textureset.get_active_stack()
        ctx["active_texture_set"] = active_ts.name() if active_ts else None
    except Exception:
        ctx["active_texture_set"] = None

    # Current layer (best-effort — API varies by SP version)
    try:
        stack = sp.textureset.get_active_stack()
        if stack:
            layer_stack = sp.layerstack.get_root_layer_nodes(stack)
            ctx["layer_count"] = len(layer_stack) if layer_stack else 0
        else:
            ctx["layer_count"] = 0
    except Exception:
        ctx["layer_count"] = 0

    return ctx


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
            "software": "substance",
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
        "You are a Substance Painter Python API expert. "
        "Reply with ONLY a JSON object with key 'substance_python' whose value is "
        "executable substance_painter Python code for the requested operation. "
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
            "substance_python": raw,
            "reasoning": "Claude fallback",
            "task_type": "unknown",
        }


def execute_code_safely(code: str) -> tuple:
    """
    Execute Substance Painter Python code.
    Returns (success: bool, error_message: str).
    """
    exec_globals: dict = {"__builtins__": __builtins__}
    if _IN_SP:
        exec_globals["sp"] = sp
    try:
        exec(code, exec_globals)  # noqa: S102
        return True, ""
    except Exception:
        return False, traceback.format_exc()


def send_command(command: str) -> tuple:
    """Orchestrate API call + code execution. Returns (success, message)."""
    scene_ctx = get_substance_context()

    response = None
    try:
        response = call_nalana_api(command, scene_ctx)
    except Exception:
        try:
            response = call_claude_fallback(command, scene_ctx)
        except Exception as claude_err:
            return False, f"All APIs failed: {claude_err}"

    code = response.get("substance_python") or response.get("code") or ""
    if not code:
        return False, "API returned no executable code."

    return execute_code_safely(code)


# ---------------------------------------------------------------------------
# Qt UI Panel
# ---------------------------------------------------------------------------


if _HAS_PYSIDE:

    class NalanaPanel(QtWidgets.QWidget):
        """Main Nalana dock panel for Substance Painter."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self._history: list = []
            self._build_ui()

        def _build_ui(self):
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(8, 8, 8, 8)

            # Settings group
            settings_group = QtWidgets.QGroupBox("Settings")
            settings_form = QtWidgets.QFormLayout()
            self.api_url_edit = QtWidgets.QLineEdit(_CONFIG["api_url"])
            self.api_key_edit = QtWidgets.QLineEdit(_CONFIG["api_key"])
            self.api_key_edit.setEchoMode(QtWidgets.QLineEdit.Password)
            settings_form.addRow("API URL:", self.api_url_edit)
            settings_form.addRow("API Key:", self.api_key_edit)
            settings_group.setLayout(settings_form)
            layout.addWidget(settings_group)

            # Command group
            cmd_group = QtWidgets.QGroupBox("Command")
            cmd_layout = QtWidgets.QVBoxLayout()
            self.command_edit = QtWidgets.QLineEdit()
            self.command_edit.setPlaceholderText(
                "Describe a Substance Painter operation…"
            )
            self.command_edit.returnPressed.connect(self._on_send)
            cmd_layout.addWidget(self.command_edit)

            btn_row = QtWidgets.QHBoxLayout()
            self.send_btn = QtWidgets.QPushButton("Send to Nalana")
            self.send_btn.clicked.connect(self._on_send)
            btn_row.addWidget(self.send_btn)
            self.clear_btn = QtWidgets.QPushButton("Clear History")
            self.clear_btn.clicked.connect(self._clear_history)
            btn_row.addWidget(self.clear_btn)
            cmd_layout.addLayout(btn_row)
            cmd_group.setLayout(cmd_layout)
            layout.addWidget(cmd_group)

            # Status
            self.status_label = QtWidgets.QLabel("Ready.")
            self.status_label.setStyleSheet("color: gray; font-style: italic;")
            layout.addWidget(self.status_label)

            # History
            history_group = QtWidgets.QGroupBox("Command History (last 10)")
            history_layout = QtWidgets.QVBoxLayout()
            self.history_list = QtWidgets.QListWidget()
            self.history_list.setMaximumHeight(160)
            history_layout.addWidget(self.history_list)
            history_group.setLayout(history_layout)
            layout.addWidget(history_group)

            layout.addStretch()

        def _on_send(self):
            _CONFIG["api_url"] = self.api_url_edit.text().strip() or _CONFIG["api_url"]
            _CONFIG["api_key"] = self.api_key_edit.text().strip()

            command = self.command_edit.text().strip()
            if not command:
                self.status_label.setText("Enter a command first.")
                return

            self.status_label.setText("Sending to Nalana…")
            QtWidgets.QApplication.processEvents()

            success, message = send_command(command)

            if success:
                self._push_history(command)
                self.command_edit.clear()
                self.status_label.setText(f"Done: {command}")
                if _IN_SP:
                    sp_log.log(sp_log.INFO, "Nalana", f"Executed: {command}")
            else:
                self.status_label.setText("Error — see console for details.")
                print(f"[Nalana] {message}")
                if _IN_SP:
                    sp_log.log(sp_log.ERROR, "Nalana", message[:500])

        def _push_history(self, command: str):
            self._history.append(command)
            self._history = self._history[-10:]
            self.history_list.clear()
            for cmd in reversed(self._history):
                self.history_list.addItem(cmd)

        def _clear_history(self):
            self._history.clear()
            self.history_list.clear()
            self.status_label.setText("History cleared.")

else:

    class NalanaPanel:  # noqa: F811
        """Stub when PySide2 is not available."""

        pass


# ---------------------------------------------------------------------------
# Plugin lifecycle
# ---------------------------------------------------------------------------


def start_plugin():
    """
    Called by Substance Painter when the plugin is loaded.
    Registers the Nalana dock widget.
    """
    global _dock_widget, _panel_widget

    if not _HAS_PYSIDE or not _IN_SP:
        print(
            "[Nalana] PySide2 or Substance Painter not available — plugin not started."
        )
        return

    _panel_widget = NalanaPanel()
    _dock_widget = sp.ui.add_dock_widget(_panel_widget)

    sp_log.log(sp_log.INFO, "Nalana", f"Plugin v{PLUGIN_METADATA['version']} started.")


def close_plugin():
    """
    Called by Substance Painter when the plugin is unloaded.
    Removes the dock widget and cleans up references.
    """
    global _dock_widget, _panel_widget

    if _dock_widget is not None and _IN_SP and _HAS_PYSIDE:
        try:
            sp.ui.delete_ui_element(_dock_widget)
        except Exception as e:
            print(f"[Nalana] Error removing dock widget: {e}")
        _dock_widget = None

    _panel_widget = None
    if _IN_SP:
        sp_log.log(sp_log.INFO, "Nalana", "Plugin closed.")


# ---------------------------------------------------------------------------
# Standalone execution (outside Substance Painter — for testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not _HAS_PYSIDE:
        print("[Nalana] PySide2 not available — cannot run standalone UI.")
        sys.exit(1)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    panel = NalanaPanel()
    panel.setWindowTitle("Nalana — Substance Painter (Standalone Test)")
    panel.resize(440, 520)
    panel.show()
    sys.exit(app.exec_())
