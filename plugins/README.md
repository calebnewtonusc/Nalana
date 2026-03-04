# Nalana Plugins — Installation Guide

Nalana provides AI-powered natural language and voice command interfaces for every major 3D and creative software. Each plugin sends your command to the Nalana API (or falls back to Claude directly) and executes the returned code inside the host application.

---

## Quick Start

1. Start the Nalana API server: `uvicorn nalana.api:app --host 0.0.0.0 --port 8000`
2. Install the plugin for your software (see sections below).
3. Set the API URL to `http://localhost:8000` in the plugin settings.
4. Type or speak a command — Nalana executes it.

If the Nalana backend is not running, each plugin falls back to calling Claude (claude-sonnet-4-6) directly using an Anthropic API key you provide in the settings.

---

## Blender

**File:** `plugins/blender/__init__.py`

### Installation

The Blender plugin is a package (folder), not a single file. You must zip the folder before installing.

From the repo root:
```bash
cd plugins && zip -r nalana_blender.zip blender/
```

1. Open Blender.
2. Go to **Edit → Preferences → Add-ons**.
3. Click **Install…** and select `nalana_blender.zip`.
4. Enable the **Nalana** addon by checking the checkbox.

On Blender 4.2+, you can drag `nalana_blender.zip` directly into any Blender viewport instead of steps 2–3.
5. In the addon preferences, set:
   - **API URL** — default `http://localhost:8000`
   - **API Key** — leave blank for local use
   - **Anthropic API Key (Fallback)** — your `sk-ant-...` key for offline use
   - **Whisper CLI Path** — path to the `whisper` executable (for voice commands)

### Using the Panel

1. Open the **3D Viewport** sidebar (press **N**).
2. Click the **Nalana** tab.
3. Type a command (e.g. "add a red sphere at the origin") and click **Execute**, or click the microphone icon to record voice.
4. The last 5 commands appear in the **Recent Commands** section.

### API URL Configuration

Preferences → Add-ons → Nalana → API URL field.

---

## Maya

**File:** `plugins/maya/nalana_maya.py`

### Installation

1. Open Maya.
2. Go to **Window → Settings/Preferences → Plugin Manager**.
3. Click **Browse** and select `plugins/maya/nalana_maya.py`.
4. Check **Loaded** (and optionally **Auto load**).
5. A **Nalana** menu appears in Maya's main menu bar.

### Using the Panel

1. Click **Nalana → Open Nalana Panel**.
2. In the panel, set the **API URL** and **API Key**.
3. Type a command in the text field and press **Send to Nalana** or hit Enter.

### API URL Configuration

The API URL field is at the top of the Nalana panel. Changes persist within the session; for permanent configuration, edit `_CONFIG` at the top of `nalana_maya.py`.

---

## Cinema 4D

**File:** `plugins/cinema4d/nalana_c4d.py`

### Installation

**Method A — Plugin folder:**
1. Copy `nalana_c4d.py` to your Cinema 4D plugins folder:
   - macOS: `~/Library/Preferences/Maxon/Cinema 4D R.../plugins/`
   - Windows: `%APPDATA%\Maxon\Cinema 4D R...\plugins\`
2. Restart Cinema 4D.
3. The plugin appears in **Extensions** menu.

**Method B — Script Manager:**
1. Go to **Script → Script Manager**.
2. Paste or open `nalana_c4d.py`.
3. Click **Execute**.

### Using the Dialog

1. Go to **Extensions → Nalana** (or execute from Script Manager).
2. Set the **API URL** and **API Key** at the top of the dialog.
3. Type a command and click **Send to Nalana**.

### API URL Configuration

The API URL field is at the top of the Nalana dialog.

> **Note:** The plugin uses placeholder ID `1000001`. For production releases, register a real plugin ID at [developer.maxon.net](https://developer.maxon.net).

---

## Houdini

**File:** `plugins/houdini/nalana_houdini.py`

### Installation

**Method A — Copy to scripts folder:**
1. Copy `nalana_houdini.py` to `$HOUDINI_USER_PREF_DIR/scripts/python/`.
   - macOS/Linux: `~/houdini<version>/scripts/python/`
   - Windows: `C:\Users\<name>\Documents\houdini<version>\scripts\python\`

**Method B — Create a shelf tool:**
1. In Houdini, open the **Python Shell** (Windows → Python Shell).
2. Run:
   ```python
   exec(open('/path/to/plugins/houdini/nalana_houdini.py').read())
   create_shelf_tool()
   ```
3. A **Nalana** shelf with an **Open Nalana** tool is created in your toolbar.

**Direct usage from Python console:**
```python
exec(open('/path/to/plugins/houdini/nalana_houdini.py').read())
show_panel()
```

### Using the Panel

1. Click the **Nalana** shelf tool (or call `show_panel()`).
2. Set **API URL** and **API Key**.
3. Type a command and click **Send to Nalana** or press Enter.

### API URL Configuration

The API URL field is at the top of the Nalana panel. Edit `_CONFIG` in the script for permanent defaults.

---

## Unreal Engine 5

**File:** `plugins/unreal/nalana_unreal.py`

### Installation

1. Enable the Python Script Plugin:
   - **Edit → Plugins → Scripting → Python Script Plugin** → enable → restart.
2. Add the plugin path:
   - **Edit → Project Settings → Python → Startup Scripts** → click **+** → select `nalana_unreal.py`.
3. Alternatively, run once from the Python console:
   ```python
   import importlib.util, sys
   spec = importlib.util.spec_from_file_location("nalana_unreal", "/path/to/nalana_unreal.py")
   mod = importlib.util.module_from_spec(spec)
   sys.modules["nalana_unreal"] = mod
   spec.loader.exec_module(mod)
   mod.show_nalana_dialog()
   ```

### Using the Dialog

1. Go to **Level Editor main menu → Nalana → Open Nalana** (registered automatically on startup).
2. Or call from the Output Log Python REPL: `import nalana_unreal; nalana_unreal.show_nalana_dialog()`
3. Set **API URL** and **API Key** in the Settings section of the tkinter window.
4. Type a command and click **Send to Nalana**.

### API URL Configuration

The API URL field is at the top of the Nalana dialog. Edit `_CONFIG` in the script for permanent defaults.

---

## Rhino

**File:** `plugins/rhino/nalana_rhino.py`

### Installation

**Method A — Run from script:**
1. In Rhino, go to **Tools → PythonScript → Run**.
2. Select `plugins/rhino/nalana_rhino.py`.
3. The Nalana dialog appears.

**Method B — Add as a command (Rhino 8 CPython):**
1. Copy `nalana_rhino.py` to your Rhino Python scripts folder.
2. In Rhino, type `EditPythonScript` and open the file.
3. Run `RunCommand()` or type `Nalana` in the Rhino command bar.

### Using the Form

1. The Nalana Windows Forms dialog appears when you run the script.
2. Set **API URL** and **API Key** in the Settings section.
3. Type a command and click **Send to Nalana** or press Enter.

### API URL Configuration

The API URL field is in the Settings group at the top of the form.

---

## Substance Painter

**File:** `plugins/substance/nalana_substance.py`

### Installation

1. In Substance Painter, go to **Python → Plugin Manager**.
2. Click **Install Plugin** and select `nalana_substance.py` (or a ZIP containing it).
3. Enable the plugin — the **Nalana** panel appears as a dock widget.

Alternatively, for immediate use:
1. Go to **Python → Script Editor**.
2. Open and run `nalana_substance.py`.

### Using the Panel

1. The Nalana panel docks in the Substance Painter workspace.
2. Set **API URL** and **API Key**.
3. Type a command (e.g. "add a metal roughness layer") and click **Send to Nalana**.

### API URL Configuration

The API URL field is at the top of the Nalana dock panel. Edit `_CONFIG` in the script for permanent defaults.

---

## Web (Three.js / Babylon.js)

**File:** `plugins/web/nalana_web.js`

### Installation — ES Module

```html
<script type="module">
  import { NalanaClient, NalanaUI, NalanaVoiceInput } from './nalana_web.js';

  const client = new NalanaClient('http://localhost:8000', {
    apiKey: 'your-key',      // optional
    software: 'threejs',     // or 'babylonjs'
  });

  // Attach floating UI panel to the Three.js canvas
  const ui = new NalanaUI(client, {
    canvas: renderer.domElement,
    framework: 'threejs',
  });
  ui.mount({ scene, camera, renderer });
</script>
```

### Installation — Script Tag (window.Nalana global)

```html
<script src="nalana_web.js"></script>
<script>
  const client = new window.Nalana.NalanaClient('http://localhost:8000');
  const context = client.getThreeJSContext(scene, camera, renderer);
  client.sendCommand('add a blue torus', context).then((res) => {
    client.executeThreeJS(res.code, scene, camera, renderer);
  });
</script>
```

### Three.js Quick Start

```javascript
import { NalanaClient } from './nalana_web.js';

const nalana = new NalanaClient('http://localhost:8000');

// Get scene state
const ctx = nalana.getThreeJSContext(scene, camera, renderer);

// Send a command
const result = await nalana.sendCommand('rotate all meshes 45 degrees on Y', ctx);

// Execute the returned code
nalana.executeThreeJS(result.code, scene, camera, renderer);
```

### Babylon.js Quick Start

```javascript
import { NalanaClient } from './nalana_web.js';

const nalana = new NalanaClient('http://localhost:8000', { software: 'babylonjs' });

const ctx = nalana.getBabylonContext(scene);
const result = await nalana.sendCommand('make all meshes wireframe', ctx);
nalana.executeBabylon(result.code, scene);
```

### Voice Input

```javascript
import { NalanaVoiceInput, NalanaClient } from './nalana_web.js';

const voice = new NalanaVoiceInput();
const client = new NalanaClient('http://localhost:8000');

voice.onResult = async (transcript) => {
  const result = await client.sendCommand(transcript, {});
  client.executeThreeJS(result.code, scene, camera, renderer);
};

document.getElementById('mic-btn').onclick = () => voice.start();
```

### API URL Configuration

Pass the URL as the first argument to `new NalanaClient(url)`. To change it dynamically, create a new client instance.

---

## Unity

**File:** `plugins/unity/NalanaUnity.cs`

### Installation

1. Copy `NalanaUnity.cs` to `Assets/Editor/` in your Unity project.
   - If the `Editor` folder does not exist, create it.
2. Unity will automatically compile the script.
3. Open the window via **Tools → Nalana** in the menu bar.

### Dependencies

- Unity 2021.3+ (LTS recommended)
- For complex code execution, optionally add **Newtonsoft.Json** via the Package Manager:
  `com.unity.nuget.newtonsoft-json`

### Using the Window

1. Go to **Tools → Nalana**.
2. Expand **Settings** to set the **API URL** and **API Key**.
   - Settings are persisted via `EditorPrefs`.
3. Type a command and click **Send to Nalana** or press Enter.
4. The plugin queues the returned C# code as a temporary Editor script that executes after recompilation.

### Right-Click Menu

Right-click any GameObject in the Hierarchy and select **Send to Nalana** to open the window with that object's details pre-populated.

### API URL Configuration

Expand the **Settings** foldout in the Nalana window. The URL is stored in `EditorPrefs` and persists across sessions. Optionally store your Anthropic fallback key in `EditorPrefs` with key `Nalana_AnthropicKey`.

---

## Fallback Mode (Claude API)

Every plugin works without a running Nalana server by falling back to Claude (claude-sonnet-4-6) directly. To enable fallback:

1. Obtain an Anthropic API key from [console.anthropic.com](https://console.anthropic.com).
2. Enter the key in the plugin's settings:
   - **Blender**: Preferences → Nalana → Anthropic API Key (Fallback)
   - **Maya / Houdini / Cinema 4D / Substance / Rhino / Unreal**: edit `_CONFIG["anthropic_key"]` in the script, or expose it via the UI settings form
   - **Unity**: set `EditorPrefs` key `Nalana_AnthropicKey`
   - **Web**: the fallback is handled server-side; for client-side fallback, call the Anthropic REST API directly

When the Nalana API is unreachable, the plugin automatically retries with Claude and shows a "using Claude fallback" status message.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Connection refused" on send | Ensure the Nalana server is running on the configured port |
| Empty code returned | Check the Nalana server logs; the model may not have a handler for this command type |
| Blender: addon not visible | Ensure Blender 4.0+ is installed; check the system console for import errors |
| Maya: Nalana menu missing | Confirm the plugin loaded (no red X in Plugin Manager) |
| Unreal: tkinter not found | Unreal ships CPython — install tkinter: `pip install tk` in the Unreal Python env |
| Rhino: WinForms import fails | Rhino 7 uses IronPython which has WinForms built in; Rhino 8 CPython needs `pythonnet` |
| Unity: code not executing | Check the Console for compilation errors in the `_NalanaTemp` folder |
| Web: CORS error | Add `http://localhost:<port>` to the Nalana API's CORS allowed origins |
