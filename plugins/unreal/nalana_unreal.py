"""
Nalana Unreal Engine 5 Python Editor Script
Natural language / voice command interface for Unreal Engine 5 via the Nalana API.

Installation:
  Project Settings → Plugins → Python Script Plugin → enable
  Project Settings → Python → Startup Scripts → add this file's path
  OR: run directly from the Python console in the UE5 editor.

Usage:
  import nalana_unreal
  nalana_unreal.show_nalana_dialog()
"""

import json
import traceback
import urllib.request
import urllib.error
import tkinter as tk
from tkinter import ttk

# Unreal Engine imports — available only inside the UE5 editor.
try:
    import unreal

    _IN_UNREAL = True
except ImportError:
    _IN_UNREAL = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CONFIG = {
    "api_url": "http://localhost:8000",
    "api_key": "",
    "anthropic_key": "",
}

# Global references.
_dialog_ref = None
_toolbar_name = "NalanaToolbar"
_toolbar_section = "Nalana"
_menu_entry_name = "NalanaOpen"

# ---------------------------------------------------------------------------
# Context helper
# ---------------------------------------------------------------------------


def get_unreal_context() -> dict:
    """Return a snapshot of the current UE5 editor state."""
    if not _IN_UNREAL:
        return {}

    try:
        selected_actors = unreal.EditorLevelLibrary.get_selected_level_actors()
        selected_names = [a.get_name() for a in selected_actors]
        active_name = selected_names[0] if selected_names else None
    except Exception:
        selected_names = []
        active_name = None

    try:
        current_level = unreal.EditorLevelLibrary.get_editor_world().get_name()
    except Exception:
        current_level = "unknown"

    try:
        all_actors = unreal.EditorLevelLibrary.get_all_level_actors()
        actor_count = len(all_actors)
    except Exception:
        actor_count = 0

    try:
        viewport = unreal.UnrealEditorSubsystem().get_level_viewport_client()
        camera_location = str(viewport.get_view_location()) if viewport else "unknown"
    except Exception:
        camera_location = "unknown"

    return {
        "active_object": active_name,
        "selected_objects": selected_names,
        "actor_count": actor_count,
        "current_level": current_level,
        "camera_location": camera_location,
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
            "software": "unreal",
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
        "You are an Unreal Engine 5 Python (unreal module) expert. "
        "Reply with ONLY a JSON object with key 'unreal_python' whose value is "
        "executable unreal Python code for the requested operation. "
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
            "unreal_python": raw,
            "reasoning": "Claude fallback",
            "task_type": "unknown",
        }


def execute_code_safely(code: str) -> tuple:
    """
    Execute Unreal Python code.
    Returns (success: bool, error_message: str).
    """
    exec_globals = (
        {"unreal": unreal, "__builtins__": __builtins__} if _IN_UNREAL else {}
    )
    try:
        exec(code, exec_globals)  # noqa: S102
        return True, ""
    except Exception:
        return False, traceback.format_exc()


def send_command(command: str) -> tuple:
    """Orchestrate API call + code execution. Returns (success, message)."""
    scene_ctx = get_unreal_context()

    response = None
    try:
        response = call_nalana_api(command, scene_ctx)
    except Exception:
        try:
            response = call_claude_fallback(command, scene_ctx)
        except Exception as claude_err:
            return False, f"All APIs failed: {claude_err}"

    code = response.get("unreal_python") or response.get("code") or ""
    if not code:
        return False, "API returned no executable code."

    return execute_code_safely(code)


# ---------------------------------------------------------------------------
# Tkinter UI (Unreal does not ship Qt; tkinter ships with CPython)
# ---------------------------------------------------------------------------


class NalanaDialog:
    """Tkinter-based dialog window for Nalana in Unreal Engine."""

    def __init__(self):
        self._history: list = []
        self._root = tk.Tk()
        self._root.title("Nalana — Unreal Engine 5 AI Assistant")
        self._root.resizable(True, True)
        self._root.geometry("480x520")
        self._build_ui()

    def _build_ui(self):
        root = self._root

        # Settings frame
        settings_frame = ttk.LabelFrame(root, text="Settings", padding=6)
        settings_frame.pack(fill="x", padx=8, pady=4)

        ttk.Label(settings_frame, text="API URL:").grid(row=0, column=0, sticky="w")
        self.api_url_var = tk.StringVar(value=_CONFIG["api_url"])
        ttk.Entry(settings_frame, textvariable=self.api_url_var, width=38).grid(
            row=0, column=1, padx=4
        )

        ttk.Label(settings_frame, text="API Key:").grid(row=1, column=0, sticky="w")
        self.api_key_var = tk.StringVar(value=_CONFIG["api_key"])
        ttk.Entry(
            settings_frame, textvariable=self.api_key_var, width=38, show="*"
        ).grid(row=1, column=1, padx=4)

        # Command frame
        cmd_frame = ttk.LabelFrame(root, text="Command", padding=6)
        cmd_frame.pack(fill="x", padx=8, pady=4)

        self.cmd_var = tk.StringVar()
        cmd_entry = ttk.Entry(cmd_frame, textvariable=self.cmd_var, width=48)
        cmd_entry.pack(fill="x")
        cmd_entry.bind("<Return>", lambda _: self._on_send())

        btn_frame = tk.Frame(cmd_frame)
        btn_frame.pack(fill="x", pady=4)
        ttk.Button(btn_frame, text="Send to Nalana", command=self._on_send).pack(
            side="left", padx=2
        )
        ttk.Button(btn_frame, text="Clear History", command=self._clear_history).pack(
            side="left", padx=2
        )

        # Status
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(
            root,
            textvariable=self.status_var,
            foreground="gray",
            font=("TkDefaultFont", 9, "italic"),
        ).pack(anchor="w", padx=10)

        # History
        history_frame = ttk.LabelFrame(
            root, text="Command History (last 10)", padding=6
        )
        history_frame.pack(fill="both", expand=True, padx=8, pady=4)

        self.history_listbox = tk.Listbox(history_frame, height=10)
        scrollbar = ttk.Scrollbar(
            history_frame, orient="vertical", command=self.history_listbox.yview
        )
        self.history_listbox.configure(yscrollcommand=scrollbar.set)
        self.history_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _on_send(self):
        _CONFIG["api_url"] = self.api_url_var.get().strip() or _CONFIG["api_url"]
        _CONFIG["api_key"] = self.api_key_var.get().strip()

        command = self.cmd_var.get().strip()
        if not command:
            self.status_var.set("Enter a command first.")
            return

        self.status_var.set("Sending to Nalana…")
        self._root.update()

        success, message = send_command(command)

        if success:
            self._push_history(command)
            self.cmd_var.set("")
            self.status_var.set(f"Done: {command}")
        else:
            self.status_var.set("Error — see Output Log for details.")
            print(f"[Nalana] {message}")
            if _IN_UNREAL:
                unreal.log_warning(f"[Nalana] {message[:500]}")

    def _push_history(self, command: str):
        self._history.append(command)
        self._history = self._history[-10:]
        self.history_listbox.delete(0, tk.END)
        for cmd in reversed(self._history):
            self.history_listbox.insert(tk.END, cmd)

    def _clear_history(self):
        self._history.clear()
        self.history_listbox.delete(0, tk.END)
        self.status_var.set("History cleared.")

    def show(self):
        self._root.deiconify()
        self._root.lift()
        self._root.focus_force()

    def mainloop(self):
        self._root.mainloop()


# ---------------------------------------------------------------------------
# Toolbar / menu registration (Unreal Engine 5)
# ---------------------------------------------------------------------------


def _register_toolbar():
    """Register a Nalana button in the UE5 editor toolbar via unreal.ToolMenus."""
    if not _IN_UNREAL:
        return

    try:
        menus = unreal.ToolMenus.get()
        main_menu = menus.find_menu("LevelEditor.MainMenu")
        if main_menu:
            section = main_menu.find_section(_toolbar_section)
            if section is None:
                main_menu.add_section(_toolbar_section, _toolbar_section)
            entry = unreal.ToolMenuEntry(
                name=_menu_entry_name,
                type=unreal.MultiBlockType.MENU_ENTRY,
            )
            entry.set_label("Open Nalana")
            entry.set_tool_tip("Open the Nalana AI assistant panel")
            main_menu.add_menu_entry(_toolbar_section, entry)
            menus.refresh_all_widgets()
            print("[Nalana] Toolbar entry registered in Level Editor main menu.")
    except Exception as e:
        print(f"[Nalana] Could not register toolbar: {e}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def show_nalana_dialog():
    """Create (or raise) the Nalana dialog. Safe to call from UE5 Python console."""
    global _dialog_ref
    if _dialog_ref is None:
        _dialog_ref = NalanaDialog()
    _dialog_ref.show()
    # In UE5 the tkinter mainloop must not block the editor event loop;
    # call update in a loop or rely on the editor's Python async execution.
    try:
        _dialog_ref._root.mainloop()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Auto-register when imported inside UE5
# ---------------------------------------------------------------------------

if _IN_UNREAL:
    _register_toolbar()
    print(
        "[Nalana] Unreal Engine 5 plugin loaded. Call nalana_unreal.show_nalana_dialog() to open."
    )


# ---------------------------------------------------------------------------
# Standalone execution (outside UE5 — for testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dialog = NalanaDialog()
    dialog.mainloop()
