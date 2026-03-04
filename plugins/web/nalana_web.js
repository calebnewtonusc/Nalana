/**
 * Nalana Web Plugin
 * Browser JavaScript integration for Three.js and Babylon.js scenes.
 *
 * @module nalana_web
 *
 * @example
 * // ES module usage with Three.js
 * import { NalanaClient, NalanaUI, NalanaVoiceInput } from './nalana_web.js';
 *
 * const client = new NalanaClient('http://localhost:8000', { software: 'threejs' });
 * const ui = new NalanaUI(client, { canvas: renderer.domElement, framework: 'threejs' });
 * ui.mount({ scene, camera, renderer });
 *
 * @example
 * // Global (script tag) usage
 * const client = new window.Nalana.NalanaClient('http://localhost:8000');
 * const result = await client.sendCommand('add a red sphere', client.getThreeJSContext(scene, camera, renderer));
 */

// ---------------------------------------------------------------------------
// NalanaClient
// ---------------------------------------------------------------------------

/**
 * Core API client for the Nalana backend.
 */
class NalanaClient {
  /**
   * @param {string} apiUrl   - Base URL of the Nalana API server, e.g. 'http://localhost:8000'
   * @param {object} [options]
   * @param {string} [options.apiKey]       - Bearer token for authenticated endpoints
   * @param {string} [options.software]     - Target software hint ('threejs' | 'babylonjs')
   * @param {number} [options.timeoutMs]    - Fetch timeout in milliseconds (default: 30000)
   */
  constructor(apiUrl, options = {}) {
    this.apiUrl = apiUrl.replace(/\/$/, '');
    this.apiKey = options.apiKey || '';
    this.software = options.software || 'threejs';
    this.timeoutMs = options.timeoutMs || 30000;
  }

  /**
   * Build default request headers.
   * @private
   */
  _headers() {
    const headers = {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    };
    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }
    return headers;
  }

  /**
   * Send a command to the Nalana API and return the structured response.
   *
   * @param {string} voiceText    - Natural language command
   * @param {object} sceneContext - Scene state object (use getThreeJSContext or getBabylonContext)
   * @returns {Promise<{code: string, reasoning: string, task_type: string}>}
   */
  async sendCommand(voiceText, sceneContext = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const response = await fetch(`${this.apiUrl}/v1/command`, {
        method: 'POST',
        headers: this._headers(),
        signal: controller.signal,
        body: JSON.stringify({
          voice_command: voiceText,
          scene_context: sceneContext,
          software: this.software,
        }),
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`Nalana API error ${response.status}: ${text}`);
      }

      return await response.json();
    } finally {
      clearTimeout(timer);
    }
  }

  /**
   * Stream a command response from the Nalana API using Server-Sent Events.
   * The callback is invoked once for each SSE event received.
   *
   * @param {string}   voiceText    - Natural language command
   * @param {object}   sceneContext - Scene state object
   * @param {function} callback     - Called with (eventData: object) for each SSE chunk
   * @returns {EventSource}         - The underlying EventSource; call .close() to stop.
   */
  streamCommand(voiceText, sceneContext = {}, callback) {
    const params = new URLSearchParams({
      voice_command: voiceText,
      scene_context: JSON.stringify(sceneContext),
      software: this.software,
    });

    const url = `${this.apiUrl}/v1/command/stream?${params.toString()}`;
    const headers = this._headers();

    // EventSource does not support custom headers natively.
    // For authenticated streaming, use a POST-first approach or include the
    // API key as a query param if the server supports it.
    const eventSource = new EventSource(url);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        callback(data);
      } catch {
        callback({ raw: event.data });
      }
    };

    eventSource.onerror = (err) => {
      console.error('[Nalana] SSE error:', err);
      eventSource.close();
    };

    return eventSource;
  }

  // -------------------------------------------------------------------------
  // Context serializers
  // -------------------------------------------------------------------------

  /**
   * Extract a serializable scene context from a Three.js scene.
   *
   * @param {THREE.Scene}    scene
   * @param {THREE.Camera}   camera
   * @param {THREE.WebGLRenderer} renderer
   * @returns {object}
   */
  getThreeJSContext(scene, camera, renderer) {
    if (!scene) return {};

    const objects = [];
    scene.traverse((obj) => {
      if (obj.isMesh || obj.isLight || obj.isCamera) {
        objects.push({
          name: obj.name || obj.uuid,
          type: obj.type,
          position: obj.position ? obj.position.toArray() : null,
          visible: obj.visible,
        });
      }
    });

    return {
      framework: 'threejs',
      object_count: objects.length,
      objects: objects.slice(0, 50), // cap to avoid huge payloads
      camera_position: camera ? camera.position.toArray() : null,
      camera_type: camera ? camera.type : null,
      renderer_size: renderer
        ? { width: renderer.domElement.width, height: renderer.domElement.height }
        : null,
    };
  }

  /**
   * Extract a serializable scene context from a Babylon.js scene.
   *
   * @param {BABYLON.Scene} scene
   * @returns {object}
   */
  getBabylonContext(scene) {
    if (!scene) return {};

    const meshes = (scene.meshes || []).slice(0, 50).map((m) => ({
      name: m.name,
      id: m.id,
      position: m.position ? [m.position.x, m.position.y, m.position.z] : null,
      isVisible: m.isVisible,
    }));

    const camera = scene.activeCamera;

    return {
      framework: 'babylonjs',
      mesh_count: (scene.meshes || []).length,
      meshes,
      camera_position: camera
        ? [camera.position.x, camera.position.y, camera.position.z]
        : null,
      camera_type: camera ? camera.getClassName() : null,
    };
  }

  // -------------------------------------------------------------------------
  // Code execution
  // -------------------------------------------------------------------------

  /**
   * Evaluate Three.js code returned by the API.
   * The code runs with `scene`, `camera`, `renderer`, and `THREE` in scope.
   *
   * @param {string} code
   * @param {THREE.Scene}         scene
   * @param {THREE.Camera}        camera
   * @param {THREE.WebGLRenderer} renderer
   * @returns {{ success: boolean, error: string|null }}
   */
  executeThreeJS(code, scene, camera, renderer) {
    try {
      // eslint-disable-next-line no-new-func
      const fn = new Function('scene', 'camera', 'renderer', 'THREE', code);
      fn(scene, camera, renderer, window.THREE || undefined);
      return { success: true, error: null };
    } catch (err) {
      console.error('[Nalana] Three.js execution error:', err);
      return { success: false, error: err.message };
    }
  }

  /**
   * Evaluate Babylon.js code returned by the API.
   * The code runs with `scene` and `BABYLON` in scope.
   *
   * @param {string}        code
   * @param {BABYLON.Scene} scene
   * @returns {{ success: boolean, error: string|null }}
   */
  executeBabylon(code, scene) {
    try {
      // eslint-disable-next-line no-new-func
      const fn = new Function('scene', 'BABYLON', code);
      fn(scene, window.BABYLON || undefined);
      return { success: true, error: null };
    } catch (err) {
      console.error('[Nalana] Babylon.js execution error:', err);
      return { success: false, error: err.message };
    }
  }
}

// ---------------------------------------------------------------------------
// NalanaVoiceInput
// ---------------------------------------------------------------------------

/**
 * Web Speech API wrapper for voice input.
 * Supported in Chrome, Edge, and Safari (partial).
 *
 * @example
 * const voice = new NalanaVoiceInput();
 * voice.onResult = (transcript) => console.log(transcript);
 * voice.start();
 * // user speaks...
 * voice.stop();
 */
class NalanaVoiceInput {
  constructor() {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition || null;

    if (!SpeechRecognition) {
      console.warn('[Nalana] Web Speech API not supported in this browser.');
      this._supported = false;
      this._recognition = null;
      return;
    }

    this._supported = true;
    this._recognition = new SpeechRecognition();
    this._recognition.continuous = false;
    this._recognition.interimResults = false;
    this._recognition.lang = 'en-US';
    this._isRecording = false;

    this._recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      if (typeof this.onResult === 'function') {
        this.onResult(transcript);
      }
    };

    this._recognition.onerror = (event) => {
      console.error('[Nalana] Speech recognition error:', event.error);
      this._isRecording = false;
      if (typeof this.onError === 'function') {
        this.onError(event.error);
      }
    };

    this._recognition.onend = () => {
      this._isRecording = false;
      if (typeof this.onEnd === 'function') {
        this.onEnd();
      }
    };
  }

  /** @type {boolean} */
  get supported() {
    return this._supported;
  }

  /** @type {boolean} */
  get isRecording() {
    return this._isRecording;
  }

  /**
   * Start listening for voice input.
   */
  start() {
    if (!this._supported || this._isRecording) return;
    this._recognition.start();
    this._isRecording = true;
  }

  /**
   * Stop listening.
   */
  stop() {
    if (!this._supported || !this._isRecording) return;
    this._recognition.stop();
    this._isRecording = false;
  }

  /**
   * Callback invoked when a transcript is ready.
   * @type {function(string): void}
   */
  onResult = null;

  /**
   * Callback invoked on error.
   * @type {function(string): void}
   */
  onError = null;

  /**
   * Callback invoked when recognition ends.
   * @type {function(): void}
   */
  onEnd = null;
}

// ---------------------------------------------------------------------------
// NalanaUI
// ---------------------------------------------------------------------------

/**
 * Floating draggable UI panel that overlays any Three.js or Babylon.js canvas.
 *
 * @example
 * const client = new NalanaClient('http://localhost:8000');
 * const ui = new NalanaUI(client, { canvas: renderer.domElement, framework: 'threejs' });
 * ui.mount({ scene, camera, renderer });
 */
class NalanaUI {
  /**
   * @param {NalanaClient} client
   * @param {object}       options
   * @param {HTMLCanvasElement} [options.canvas]    - Canvas element to position the panel near
   * @param {string}           [options.framework] - 'threejs' | 'babylonjs'
   * @param {string}           [options.position]  - CSS position string, e.g. 'top:10px;right:10px'
   */
  constructor(client, options = {}) {
    this._client = client;
    this._canvas = options.canvas || null;
    this._framework = options.framework || 'threejs';
    this._position = options.position || 'top:10px;right:10px';
    this._history = [];
    this._scene = null;
    this._camera = null;
    this._renderer = null;
    this._voice = new NalanaVoiceInput();
    this._panel = null;
    this._collapsed = false;
    this._dragging = false;
    this._dragOffsetX = 0;
    this._dragOffsetY = 0;
  }

  /**
   * Inject the panel into the document and bind scene references.
   *
   * @param {object} [refs]
   * @param {*} [refs.scene]
   * @param {*} [refs.camera]
   * @param {*} [refs.renderer]
   */
  mount(refs = {}) {
    this._scene = refs.scene || null;
    this._camera = refs.camera || null;
    this._renderer = refs.renderer || null;
    this._createPanel();
    this._bindVoice();
  }

  /**
   * Remove the panel from the document.
   */
  unmount() {
    if (this._panel && this._panel.parentNode) {
      this._panel.parentNode.removeChild(this._panel);
    }
    this._panel = null;
  }

  // -------------------------------------------------------------------------
  // Panel construction
  // -------------------------------------------------------------------------

  /** @private */
  _createPanel() {
    const style = `
      position:fixed;
      ${this._position}
      width:320px;
      background:rgba(20,20,30,0.95);
      color:#e0e0e0;
      border:1px solid #444;
      border-radius:8px;
      box-shadow:0 4px 24px rgba(0,0,0,0.6);
      font-family:system-ui,sans-serif;
      font-size:13px;
      z-index:99999;
      user-select:none;
    `;

    const panel = document.createElement('div');
    panel.id = 'nalana-panel';
    panel.setAttribute('style', style);

    panel.innerHTML = this._panelHTML();
    document.body.appendChild(panel);
    this._panel = panel;

    // Bind events
    panel.querySelector('#nalana-header').addEventListener('mousedown', this._onDragStart.bind(this));
    panel.querySelector('#nalana-collapse-btn').addEventListener('click', this._toggleCollapse.bind(this));
    panel.querySelector('#nalana-send-btn').addEventListener('click', this._onSend.bind(this));
    panel.querySelector('#nalana-mic-btn').addEventListener('click', this._onMic.bind(this));
    panel.querySelector('#nalana-input').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') this._onSend();
    });

    document.addEventListener('mousemove', this._onDragMove.bind(this));
    document.addEventListener('mouseup', this._onDragEnd.bind(this));
  }

  /** @private */
  _panelHTML() {
    return `
      <div id="nalana-header" style="
        padding:8px 12px;
        background:rgba(80,60,200,0.7);
        border-radius:8px 8px 0 0;
        cursor:move;
        display:flex;
        justify-content:space-between;
        align-items:center;
      ">
        <span style="font-weight:600;letter-spacing:.5px;">Nalana AI</span>
        <button id="nalana-collapse-btn" style="
          background:none;border:none;color:#e0e0e0;cursor:pointer;font-size:16px;line-height:1;
        ">&#8213;</button>
      </div>
      <div id="nalana-body" style="padding:10px;">
        <div style="display:flex;gap:6px;margin-bottom:8px;">
          <input id="nalana-input" type="text" placeholder="Describe a 3D operation…" style="
            flex:1;padding:6px 8px;background:#1e1e2e;color:#e0e0e0;
            border:1px solid #555;border-radius:4px;font-size:12px;outline:none;
          "/>
          <button id="nalana-mic-btn" title="Voice input" style="
            padding:6px 8px;background:#2a2a3e;border:1px solid #555;
            border-radius:4px;color:#e0e0e0;cursor:pointer;font-size:14px;
          ">&#127908;</button>
        </div>
        <button id="nalana-send-btn" style="
          width:100%;padding:7px;background:rgba(80,60,200,0.85);
          border:none;border-radius:4px;color:#fff;cursor:pointer;font-size:13px;
          font-weight:500;
        ">Send to Nalana</button>
        <div id="nalana-status" style="
          margin-top:6px;color:#999;font-style:italic;font-size:11px;min-height:16px;
        ">Ready.</div>
        <div id="nalana-history" style="margin-top:8px;"></div>
      </div>
    `;
  }

  /** @private */
  _toggleCollapse() {
    this._collapsed = !this._collapsed;
    const body = this._panel.querySelector('#nalana-body');
    const btn = this._panel.querySelector('#nalana-collapse-btn');
    if (this._collapsed) {
      body.style.display = 'none';
      btn.innerHTML = '&#43;';
    } else {
      body.style.display = '';
      btn.innerHTML = '&#8213;';
    }
  }

  // -------------------------------------------------------------------------
  // Dragging
  // -------------------------------------------------------------------------

  /** @private */
  _onDragStart(e) {
    this._dragging = true;
    const rect = this._panel.getBoundingClientRect();
    this._dragOffsetX = e.clientX - rect.left;
    this._dragOffsetY = e.clientY - rect.top;
  }

  /** @private */
  _onDragMove(e) {
    if (!this._dragging) return;
    const x = e.clientX - this._dragOffsetX;
    const y = e.clientY - this._dragOffsetY;
    this._panel.style.left = `${x}px`;
    this._panel.style.top = `${y}px`;
    this._panel.style.right = 'auto';
    this._panel.style.bottom = 'auto';
  }

  /** @private */
  _onDragEnd() {
    this._dragging = false;
  }

  // -------------------------------------------------------------------------
  // Command send
  // -------------------------------------------------------------------------

  /** @private */
  async _onSend() {
    const input = this._panel.querySelector('#nalana-input');
    const command = input.value.trim();
    if (!command) {
      this._setStatus('Enter a command first.');
      return;
    }

    this._setStatus('Sending…');
    this._setSendDisabled(true);

    let sceneContext = {};
    try {
      if (this._framework === 'threejs' && this._scene) {
        sceneContext = this._client.getThreeJSContext(this._scene, this._camera, this._renderer);
      } else if (this._framework === 'babylonjs' && this._scene) {
        sceneContext = this._client.getBabylonContext(this._scene);
      }
    } catch (ctxErr) {
      console.warn('[Nalana] Could not build scene context:', ctxErr);
    }

    let response;
    try {
      response = await this._client.sendCommand(command, sceneContext);
    } catch (apiErr) {
      this._setStatus(`API error: ${apiErr.message}`);
      this._setSendDisabled(false);
      return;
    }

    const code = response.code || response.threejs_javascript || response.babylonjs_javascript || '';
    if (!code) {
      this._setStatus('No executable code returned.');
      this._setSendDisabled(false);
      return;
    }

    let result;
    if (this._framework === 'threejs') {
      result = this._client.executeThreeJS(code, this._scene, this._camera, this._renderer);
    } else if (this._framework === 'babylonjs') {
      result = this._client.executeBabylon(code, this._scene);
    } else {
      try {
        // eslint-disable-next-line no-new-func
        new Function(code)();
        result = { success: true, error: null };
      } catch (err) {
        result = { success: false, error: err.message };
      }
    }

    if (result.success) {
      this._pushHistory(command);
      input.value = '';
      this._setStatus(`Done: ${command}`);
    } else {
      this._setStatus(`Error: ${result.error}`);
      console.error('[Nalana] Execution error:', result.error);
    }

    this._setSendDisabled(false);
  }

  // -------------------------------------------------------------------------
  // Voice input
  // -------------------------------------------------------------------------

  /** @private */
  _bindVoice() {
    this._voice.onResult = (transcript) => {
      const input = this._panel && this._panel.querySelector('#nalana-input');
      if (input) input.value = transcript;
      this._setStatus(`Heard: "${transcript}"`);
      this._updateMicButton(false);
    };
    this._voice.onEnd = () => this._updateMicButton(false);
    this._voice.onError = (err) => {
      this._setStatus(`Mic error: ${err}`);
      this._updateMicButton(false);
    };
  }

  /** @private */
  _onMic() {
    if (!this._voice.supported) {
      this._setStatus('Speech recognition not supported in this browser.');
      return;
    }

    if (this._voice.isRecording) {
      this._voice.stop();
      this._updateMicButton(false);
    } else {
      this._voice.start();
      this._updateMicButton(true);
      this._setStatus('Listening…');
    }
  }

  /** @private */
  _updateMicButton(active) {
    const btn = this._panel && this._panel.querySelector('#nalana-mic-btn');
    if (btn) {
      btn.style.background = active ? 'rgba(200,60,60,0.8)' : '#2a2a3e';
    }
  }

  // -------------------------------------------------------------------------
  // History
  // -------------------------------------------------------------------------

  /** @private */
  _pushHistory(command) {
    this._history.push(command);
    this._history = this._history.slice(-10);
    const container = this._panel.querySelector('#nalana-history');
    if (!container) return;

    container.innerHTML = `
      <div style="font-size:11px;color:#666;margin-bottom:4px;">Recent commands:</div>
      ${[...this._history].reverse().map((cmd) => `
        <div style="
          font-size:11px;padding:3px 6px;margin:2px 0;
          background:#1e1e2e;border-radius:3px;color:#aaa;
          white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
        " title="${cmd.replace(/"/g, '&quot;')}">${this._escapeHtml(cmd)}</div>
      `).join('')}
    `;
  }

  // -------------------------------------------------------------------------
  // Utilities
  // -------------------------------------------------------------------------

  /** @private */
  _setStatus(msg) {
    const el = this._panel && this._panel.querySelector('#nalana-status');
    if (el) el.textContent = msg;
  }

  /** @private */
  _setSendDisabled(disabled) {
    const btn = this._panel && this._panel.querySelector('#nalana-send-btn');
    if (btn) {
      btn.disabled = disabled;
      btn.style.opacity = disabled ? '0.6' : '1';
    }
  }

  /** @private */
  _escapeHtml(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }
}

// ---------------------------------------------------------------------------
// ES module exports
// ---------------------------------------------------------------------------

export { NalanaClient, NalanaVoiceInput, NalanaUI };

// ---------------------------------------------------------------------------
// Browser global (window.Nalana) for script-tag usage
// ---------------------------------------------------------------------------

if (typeof window !== 'undefined') {
  window.Nalana = { NalanaClient, NalanaVoiceInput, NalanaUI };
}
