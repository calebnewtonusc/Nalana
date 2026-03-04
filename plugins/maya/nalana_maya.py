"""
Nalana Maya Plugin
Natural language / voice command interface for Autodesk Maya via the Nalana API.

Installation:
  Window → Settings/Preferences → Plugin Manager → Browse → select this file → Load

The plugin registers a "Nalana" menu in Maya's main menu bar.
"""

import sys
import json
import traceback
import urllib.request
import urllib.error

# Maya imports — these are available only when running inside Maya.
try:
    import maya.cmds as cmds
    import maya.mel as mel
    import maya.api.OpenMaya as om
    _IN_MAYA = True
except ImportError:
    _IN_MAYA = False

# PySide2 — ships with Maya 2022+.
try:
    from PySide2 import QtWidgets, QtCore, QtGui
    from shiboken2 import wrapInstance
    _HAS_PYSIDE = True
except ImportError:
    _HAS_PYSIDE = False

# ---------------------------------------------------------------------------
# Maya plugin metadata (required by Maya plugin API)
# ---------------------------------------------------------------------------

PLUGIN_VENDOR = "Nalana Team"
PLUGIN_VERSION = "1.0.0"


def maya_useNewAPI():
    """Declare that this plugin uses the Maya Python API 2.0."""
    pass


# ---------------------------------------------------------------------------
# Configuration (defaults — override via the UI or environment)
# ---------------------------------------------------------------------------

_CONFIG = {
    "api_url": "http://localhost:8000",
    "api_key": "",
    "anthropic_key": "",
}

# Global reference to the window so it is not garbage-collected.
_nalana_window_ref = None
_menu_name = "NalanaMenu"


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------


def get_maya_context() -> dict:
    """Return a snapshot of the current Maya scene state."""
    if not _IN_MAYA:
        return {}
    selected = cmds.ls(sl=True) or []
    active = (cmds.ls(sl=True, head=1) or [None])[0]
    current_ctx = cmds.currentCtx() if cmds.currentCtx else "unknown"
    all_objects = cmds.ls(type="transform") or []
    all_meshes = cmds.ls(type="mesh") or []
    current_frame = cmds.currentTime(query=True)

    return {
        "selected_objects": selected,
        "active": active,
        "mode": current_ctx,
        "object_count": len(all_objects),
        "mesh_count": len(all_meshes),
        "frame_current": current_frame,
        "scene_name": cmds.file(query=True, sceneName=True, shortName=True) or "untitled",
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
        "software": "maya",
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
        raise RuntimeError("'anthropic' package not available. Run: pip install anthropic")

    client = anthropic.Anthropic(api_key=_CONFIG["anthropic_key"])
    system_prompt = (
        "You are a Maya Python (maya.cmds) expert. "
        "Reply with ONLY a JSON object with key 'maya_python' whose value is "
        "executable maya.cmds Python code for the requested operation. "
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
        return {"maya_python": raw, "reasoning": "Claude fallback", "task_type": "unknown"}


def execute_code_safely(code: str) -> tuple:
    """
    Execute Maya Python code returned by the API.
    Returns (success: bool, error_message: str).
    """
    try:
        exec(code, {"cmds": cmds, "mel": mel, "__builtins__": __builtins__})  # noqa: S102
        return True, ""
    except Exception:
        return False, traceback.format_exc()


# ---------------------------------------------------------------------------
# Qt UI
# ---------------------------------------------------------------------------


def get_maya_main_window():
    """Return Maya's main QMainWindow instance wrapped as a QWidget."""
    if not _IN_MAYA or not _HAS_PYSIDE:
        return None
    try:
        import maya.OpenMayaUI as omui
        import ctypes
        ptr = omui.MQtUtil.mainWindow()
        if ptr is not None:
            return wrapInstance(int(ptr), QtWidgets.QWidget)
    except Exception:
        pass
    return None


class NalanaMayaWindow(QtWidgets.QDialog):
    """Main Nalana UI dialog for Maya."""

    def __init__(self, parent=None):
        super().__init__(parent or get_maya_main_window())
        self.setWindowTitle("Nalana — Maya AI Assistant")
        self.setMinimumWidth(420)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        self._history: list = []
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Settings group
        settings_group = QtWidgets.QGroupBox("Settings")
        settings_layout = QtWidgets.QFormLayout()
        self.api_url_edit = QtWidgets.QLineEdit(_CONFIG["api_url"])
        self.api_key_edit = QtWidgets.QLineEdit(_CONFIG["api_key"])
        self.api_key_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        settings_layout.addRow("API URL:", self.api_url_edit)
        settings_layout.addRow("API Key:", self.api_key_edit)
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Command input
        cmd_group = QtWidgets.QGroupBox("Command")
        cmd_layout = QtWidgets.QVBoxLayout()
        self.command_edit = QtWidgets.QLineEdit()
        self.command_edit.setPlaceholderText("Type a natural language command…")
        self.command_edit.returnPressed.connect(self.send_command)
        cmd_layout.addWidget(self.command_edit)

        btn_row = QtWidgets.QHBoxLayout()
        self.send_btn = QtWidgets.QPushButton("Send to Nalana")
        self.send_btn.setDefault(True)
        self.send_btn.clicked.connect(self.send_command)
        btn_row.addWidget(self.send_btn)
        self.clear_btn = QtWidgets.QPushButton("Clear History")
        self.clear_btn.clicked.connect(self.clear_history)
        btn_row.addWidget(self.clear_btn)
        cmd_layout.addLayout(btn_row)
        cmd_group.setLayout(cmd_layout)
        layout.addWidget(cmd_group)

        # Status label
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

    # ------------------------------------------------------------------
    # Logic
    # ------------------------------------------------------------------

    def _update_config(self):
        _CONFIG["api_url"] = self.api_url_edit.text().strip() or _CONFIG["api_url"]
        _CONFIG["api_key"] = self.api_key_edit.text().strip()

    def send_command(self):
        """Read command, call API, execute code, update UI."""
        self._update_config()
        command = self.command_edit.text().strip()
        if not command:
            self.status_label.setText("Enter a command first.")
            return

        self.status_label.setText("Sending to Nalana…")
        QtWidgets.QApplication.processEvents()

        scene_ctx = get_maya_context()

        response = None
        try:
            response = call_nalana_api(command, scene_ctx)
            self.status_label.setText("Received response from Nalana server.")
        except Exception as api_err:
            self.status_label.setText(f"Nalana unreachable, trying Claude… ({api_err})")
            QtWidgets.QApplication.processEvents()
            try:
                response = call_claude_fallback(command, scene_ctx)
            except Exception as claude_err:
                self.status_label.setText(f"Both APIs failed: {claude_err}")
                if _IN_MAYA:
                    cmds.warning(f"[Nalana] {claude_err}")
                return

        code = response.get("maya_python") or response.get("code") or ""
        if not code:
            self.status_label.setText("API returned no executable code.")
            return

        success, error_msg = execute_code_safely(code)
        if success:
            self._push_history(command)
            self.command_edit.clear()
            self.status_label.setText(f"Executed: {command}")
        else:
            self.status_label.setText("Execution failed — see Script Editor for details.")
            if _IN_MAYA:
                cmds.warning(f"[Nalana] Execution error:\n{error_msg}")
            print(f"[Nalana] Execution error:\n{error_msg}")

    def _push_history(self, command: str):
        self._history.append(command)
        self._history = self._history[-10:]
        self.history_list.clear()
        for cmd in reversed(self._history):
            self.history_list.addItem(cmd)

    def clear_history(self):
        self._history.clear()
        self.history_list.clear()
        self.status_label.setText("History cleared.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def show_nalana_window():
    """Create (or raise) the Nalana dialog window."""
    global _nalana_window_ref
    if not _HAS_PYSIDE:
        if _IN_MAYA:
            cmds.confirmDialog(
                title="Nalana",
                message="PySide2 not available — cannot display UI.",
                button=["OK"],
            )
        return

    if _nalana_window_ref is None or not _nalana_window_ref.isVisible():
        _nalana_window_ref = NalanaMayaWindow()

    _nalana_window_ref.show()
    _nalana_window_ref.raise_()
    _nalana_window_ref.activateWindow()


# ---------------------------------------------------------------------------
# Menu registration
# ---------------------------------------------------------------------------


def _create_menu():
    """Add a 'Nalana' item to Maya's main menu bar."""
    if not _IN_MAYA:
        return
    if cmds.menu(_menu_name, exists=True):
        cmds.deleteUI(_menu_name)
    maya_window = mel.eval("$tmpVar=$gMainWindow")
    cmds.menu(_menu_name, label="Nalana", parent=maya_window, tearOff=True)
    cmds.menuItem(label="Open Nalana Panel", command=lambda *_: show_nalana_window())
    cmds.menuItem(divider=True)
    cmds.menuItem(label="About Nalana", command=lambda *_: cmds.confirmDialog(
        title="About Nalana",
        message=f"Nalana Maya Plugin v{PLUGIN_VERSION}\nAI-powered 3D operations.",
        button=["OK"],
    ))


def _remove_menu():
    """Remove the Nalana menu from Maya's main menu bar."""
    if _IN_MAYA and cmds.menu(_menu_name, exists=True):
        cmds.deleteUI(_menu_name)


# ---------------------------------------------------------------------------
# Maya plugin entry points
# ---------------------------------------------------------------------------


def initializePlugin(plugin):
    """Called by Maya when the plugin is loaded."""
    fn_plugin = om.MFnPlugin(plugin, PLUGIN_VENDOR, PLUGIN_VERSION)
    try:
        _create_menu()
        print(f"[Nalana] Plugin v{PLUGIN_VERSION} loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Nalana: Failed to initialize plugin — {e}")


def uninitializePlugin(plugin):
    """Called by Maya when the plugin is unloaded."""
    fn_plugin = om.MFnPlugin(plugin)
    try:
        _remove_menu()
        global _nalana_window_ref
        if _nalana_window_ref is not None:
            _nalana_window_ref.close()
            _nalana_window_ref = None
        print("[Nalana] Plugin unloaded.")
    except Exception as e:
        raise RuntimeError(f"Nalana: Failed to uninitialize plugin — {e}")


# ---------------------------------------------------------------------------
# Standalone / direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    show_nalana_window()
