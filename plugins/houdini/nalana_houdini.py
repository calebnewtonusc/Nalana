"""
Nalana Houdini Plugin
Natural language / voice command interface for SideFX Houdini via the Nalana API.

Usage (shelf tool):
  1. Copy to $HOUDINI_USER_PREF_DIR/scripts/python/nalana_houdini.py
  2. Create a Houdini shelf tool with the following script:
       exec(open('/path/to/nalana_houdini.py').read()); show_panel()
  OR call create_shelf_tool() once to auto-create the shelf entry.

Direct Python console usage:
  exec(open('nalana_houdini.py').read()); show_panel()
"""

import sys
import json
import traceback
import urllib.request
import urllib.error

# Houdini imports — available only inside Houdini.
try:
    import hou
    _IN_HOUDINI = True
except ImportError:
    _IN_HOUDINI = False

# PySide2 — ships with Houdini.
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
# Configuration
# ---------------------------------------------------------------------------

_CONFIG = {
    "api_url": "http://localhost:8000",
    "api_key": "",
    "anthropic_key": "",
}

_SHELF_NAME = "nalana_shelf"
_TOOL_NAME = "nalana_open"

# Global reference to prevent GC.
_panel_ref = None

# ---------------------------------------------------------------------------
# Context helper
# ---------------------------------------------------------------------------


def get_houdini_context() -> dict:
    """Return a snapshot of the current Houdini session state."""
    if not _IN_HOUDINI:
        return {}

    selected_nodes = hou.selectedNodes()
    selected_names = [node.path() for node in selected_nodes]

    # Determine current network editor context (obj, sop, dop, rop, …)
    panes = hou.ui.paneTabs() if hasattr(hou.ui, "paneTabs") else []
    network_type = "unknown"
    for pane in panes:
        if isinstance(pane, hou.NetworkEditor):
            cwd = pane.currentNode()
            if cwd:
                network_type = cwd.childTypeCategory().name()
            break

    frame = hou.frame() if hasattr(hou, "frame") else 0
    fps = hou.fps() if hasattr(hou, "fps") else 24

    return {
        "selected_nodes": selected_names,
        "node_count": len(selected_nodes),
        "network_type": network_type,
        "frame_current": frame,
        "fps": fps,
        "hip_name": hou.hipFile.basename() if hasattr(hou, "hipFile") else "untitled",
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
        "software": "houdini",
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
        "You are a Houdini Python (hou module) expert. "
        "Reply with ONLY a JSON object with key 'houdini_python' whose value is "
        "executable hou Python code for the requested operation. "
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
        return {"houdini_python": raw, "reasoning": "Claude fallback", "task_type": "unknown"}


def execute_code_safely(code: str) -> tuple:
    """
    Execute Houdini Python code in the hou context.
    Returns (success: bool, error_message: str).
    """
    exec_globals = {"hou": hou, "__builtins__": __builtins__} if _IN_HOUDINI else {}
    try:
        exec(code, exec_globals)  # noqa: S102
        return True, ""
    except Exception:
        return False, traceback.format_exc()


def send_command(command: str) -> tuple:
    """Orchestrate API call + code execution. Returns (success, message)."""
    scene_ctx = get_houdini_context()

    response = None
    try:
        response = call_nalana_api(command, scene_ctx)
    except Exception as api_err:
        try:
            response = call_claude_fallback(command, scene_ctx)
        except Exception as claude_err:
            return False, f"All APIs failed: {claude_err}"

    code = response.get("houdini_python") or response.get("code") or ""
    if not code:
        return False, "API returned no executable code."

    return execute_code_safely(code)


# ---------------------------------------------------------------------------
# Qt UI Panel
# ---------------------------------------------------------------------------


class NalanaPanel(QtWidgets.QDialog):
    """Main Nalana UI dialog for Houdini."""

    def __init__(self, parent=None):
        if parent is None and _IN_HOUDINI and _HAS_PYSIDE:
            try:
                parent = hou.qt.mainWindow()
            except Exception:
                pass
        super().__init__(parent)
        self.setWindowTitle("Nalana — Houdini AI Assistant")
        self.setMinimumWidth(440)
        self._history: list = []
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Settings
        settings_group = QtWidgets.QGroupBox("Settings")
        settings_form = QtWidgets.QFormLayout()
        self.api_url_edit = QtWidgets.QLineEdit(_CONFIG["api_url"])
        self.api_key_edit = QtWidgets.QLineEdit(_CONFIG["api_key"])
        self.api_key_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        settings_form.addRow("API URL:", self.api_url_edit)
        settings_form.addRow("API Key:", self.api_key_edit)
        settings_group.setLayout(settings_form)
        layout.addWidget(settings_group)

        # Command
        cmd_group = QtWidgets.QGroupBox("Command")
        cmd_layout = QtWidgets.QVBoxLayout()
        self.command_edit = QtWidgets.QLineEdit()
        self.command_edit.setPlaceholderText("Describe the Houdini operation…")
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

    def _on_send(self):
        _CONFIG["api_url"] = self.api_url_edit.text().strip() or _CONFIG["api_url"]
        _CONFIG["api_key"] = self.api_key_edit.text().strip()

        command = self.command_edit.text().strip()
        if not command:
            self.status_label.setText("Enter a command first.")
            return

        self.status_label.setText("Sending…")
        QtWidgets.QApplication.processEvents()

        success, message = send_command(command)

        if success:
            self._push_history(command)
            self.command_edit.clear()
            self.status_label.setText(f"Done: {command}")
        else:
            self.status_label.setText(f"Error — see console for details.")
            print(f"[Nalana] {message}")

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def show_panel():
    """Create (or raise) the Nalana panel."""
    global _panel_ref
    if not _HAS_PYSIDE:
        if _IN_HOUDINI:
            hou.ui.displayMessage(
                "PySide2 not available — cannot display Nalana UI.",
                title="Nalana",
                severity=hou.severityType.Warning,
            )
        else:
            print("[Nalana] PySide2 not available.")
        return

    if _panel_ref is None or not _panel_ref.isVisible():
        _panel_ref = NalanaPanel()

    _panel_ref.show()
    _panel_ref.raise_()
    _panel_ref.activateWindow()


def create_shelf_tool():
    """
    Programmatically create a 'Nalana' shelf and tool in Houdini.
    Call this once from the Python Shell to set up the shelf permanently.
    """
    if not _IN_HOUDINI:
        print("[Nalana] Not running inside Houdini — cannot create shelf.")
        return

    script = (
        "import sys\n"
        "sys.path.insert(0, '/path/to/nalana/plugins/houdini')\n"
        "import nalana_houdini\n"
        "nalana_houdini.show_panel()\n"
    )

    # Check if the shelf already exists; if not, create it.
    existing_shelves = hou.shelves.shelves()
    shelf = existing_shelves.get(_SHELF_NAME)
    if shelf is None:
        shelf = hou.shelves.newShelf(file_path=None, name=_SHELF_NAME, label="Nalana")

    # Create or replace the tool.
    existing_tools = hou.shelves.tools()
    tool = existing_tools.get(_TOOL_NAME)
    if tool is None:
        tool = hou.shelves.newTool(
            file_path=None,
            name=_TOOL_NAME,
            label="Nalana",
            script=script,
            help="Open the Nalana AI assistant panel",
            icon="PLASMA_App",
        )

    shelf.setTools(list(shelf.tools()) + [tool])
    print("[Nalana] Shelf tool created. Look for the 'Nalana' shelf in your toolbar.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Allow direct execution: python nalana_houdini.py (outside Houdini for testing)
    if not _HAS_PYSIDE:
        print("[Nalana] PySide2 not available in this environment.")
        sys.exit(1)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    panel = NalanaPanel()
    panel.show()
    sys.exit(app.exec_())
