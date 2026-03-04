// NalanaUnity.cs
// Unity Editor Window for the Nalana AI assistant.
// Provides a natural language command interface for Unity scenes.
//
// Requirements:
//   - Unity 2021.3+ (LTS recommended)
//   - Newtonsoft.Json (com.unity.nuget.newtonsoft-json or manual import)
//
// Installation:
//   Copy this file to Assets/Editor/ in your Unity project.
//   Open via: Tools -> Nalana

using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

#if UNITY_EDITOR

// ---------------------------------------------------------------------------
// JSON helpers (minimal — avoids hard Newtonsoft dependency for basic cases)
// ---------------------------------------------------------------------------

namespace Nalana
{
    /// <summary>
    /// Minimal JSON serializer for Nalana payloads.
    /// Replace with Newtonsoft.Json or Unity's JsonUtility for complex objects.
    /// </summary>
    internal static class NalanaJson
    {
        /// <summary>Serialize a dictionary to a JSON object string.</summary>
        public static string Serialize(Dictionary<string, object> data)
        {
            var sb = new StringBuilder("{");
            bool first = true;
            foreach (var kv in data)
            {
                if (!first) sb.Append(',');
                first = false;
                sb.Append('"');
                sb.Append(Escape(kv.Key));
                sb.Append("\":");
                sb.Append(ValueToJson(kv.Value));
            }
            sb.Append('}');
            return sb.ToString();
        }

        private static string ValueToJson(object value)
        {
            if (value == null) return "null";
            if (value is bool b) return b ? "true" : "false";
            if (value is int i) return i.ToString();
            if (value is float f) return f.ToString("G");
            if (value is double d) return d.ToString("G");
            if (value is string s) return "\"" + Escape(s) + "\"";
            return "\"" + Escape(value.ToString()) + "\"";
        }

        private static string Escape(string s)
        {
            return s
                .Replace("\\", "\\\\")
                .Replace("\"", "\\\"")
                .Replace("\n", "\\n")
                .Replace("\r", "\\r")
                .Replace("\t", "\\t");
        }

        /// <summary>
        /// Very simple JSON value extractor — looks for "key":"value" or "key":value patterns.
        /// For production use, replace with Newtonsoft.Json.JsonConvert.
        /// </summary>
        public static string ExtractStringField(string json, string key)
        {
            // Match "key": "value" (string)
            string searchKey = "\"" + key + "\"";
            int keyIndex = json.IndexOf(searchKey, StringComparison.Ordinal);
            if (keyIndex < 0) return null;

            int colonIndex = json.IndexOf(':', keyIndex + searchKey.Length);
            if (colonIndex < 0) return null;

            int valueStart = colonIndex + 1;
            while (valueStart < json.Length && json[valueStart] == ' ') valueStart++;

            if (valueStart >= json.Length) return null;

            if (json[valueStart] == '"')
            {
                // Quoted string value
                int end = json.IndexOf('"', valueStart + 1);
                if (end < 0) return null;
                return json.Substring(valueStart + 1, end - valueStart - 1)
                           .Replace("\\n", "\n")
                           .Replace("\\\"", "\"")
                           .Replace("\\\\", "\\");
            }
            else
            {
                // Unquoted value (number, bool, null)
                int end = valueStart;
                while (end < json.Length && json[end] != ',' && json[end] != '}') end++;
                return json.Substring(valueStart, end - valueStart).Trim();
            }
        }
    }

    // -----------------------------------------------------------------------
    // Scene context
    // -----------------------------------------------------------------------

    /// <summary>
    /// Collects scene state for inclusion in Nalana API requests.
    /// </summary>
    internal static class NalanaContext
    {
        public static Dictionary<string, object> GetSceneContext()
        {
            string activeObjectName = Selection.activeGameObject != null
                ? Selection.activeGameObject.name
                : "(none)";

            string currentSceneName = EditorSceneManager.GetActiveScene().name;
            int selectionCount = Selection.count;

            var allObjects = UnityEngine.Object.FindObjectsOfType<GameObject>();
            int totalObjects = allObjects != null ? allObjects.Length : 0;

            return new Dictionary<string, object>
            {
                { "active_object",  activeObjectName  },
                { "selection_count", selectionCount   },
                { "scene_name",     currentSceneName  },
                { "object_count",   totalObjects      },
                { "software",       "unity"           },
            };
        }
    }

    // -----------------------------------------------------------------------
    // Editor Window
    // -----------------------------------------------------------------------

    /// <summary>
    /// Main Nalana Unity Editor Window.
    /// Open via Tools > Nalana.
    /// </summary>
    public class NalanaWindow : EditorWindow
    {
        // --------------------------------------------------------------------
        // Fields
        // --------------------------------------------------------------------

        private string _commandText  = string.Empty;
        private string _apiUrl       = "http://localhost:8000";
        private string _apiKey       = string.Empty;
        private string _statusMessage = "Ready.";
        private bool   _isSending    = false;
        private bool   _showSettings = false;

        private readonly List<string> _history = new List<string>();
        private Vector2 _historyScrollPos = Vector2.zero;

        private static readonly HttpClient _httpClient = new HttpClient { Timeout = TimeSpan.FromSeconds(30) };

        // --------------------------------------------------------------------
        // EditorPrefs keys
        // --------------------------------------------------------------------

        private const string PrefApiUrl = "Nalana_ApiUrl";
        private const string PrefApiKey = "Nalana_ApiKey";

        // --------------------------------------------------------------------
        // Menu item
        // --------------------------------------------------------------------

        [MenuItem("Tools/Nalana")]
        public static void OpenWindow()
        {
            NalanaWindow window = GetWindow<NalanaWindow>(title: "Nalana");
            window.minSize = new Vector2(360, 480);
            window.Show();
        }

        // --------------------------------------------------------------------
        // Lifecycle
        // --------------------------------------------------------------------

        private void OnEnable()
        {
            _apiUrl = EditorPrefs.GetString(PrefApiUrl, "http://localhost:8000");
            _apiKey = EditorPrefs.GetString(PrefApiKey, string.Empty);
        }

        private void OnDisable()
        {
            EditorPrefs.SetString(PrefApiUrl, _apiUrl);
            EditorPrefs.SetString(PrefApiKey, _apiKey);
        }

        // --------------------------------------------------------------------
        // GUI
        // --------------------------------------------------------------------

        private void OnGUI()
        {
            GUILayout.Label("Nalana AI Assistant", EditorStyles.boldLabel);
            EditorGUILayout.Space(4);

            // Settings foldout
            _showSettings = EditorGUILayout.Foldout(_showSettings, "Settings", true);
            if (_showSettings)
            {
                EditorGUI.indentLevel++;
                _apiUrl = EditorGUILayout.TextField("API URL", _apiUrl);
                _apiKey = EditorGUILayout.PasswordField("API Key", _apiKey);
                EditorGUI.indentLevel--;
                EditorGUILayout.Space(4);
            }

            // Command input
            GUILayout.Label("Command:", EditorStyles.label);
            GUI.SetNextControlName("NalanaCommandField");
            _commandText = EditorGUILayout.TextField(_commandText);

            EditorGUI.BeginDisabledGroup(_isSending);
            if (GUILayout.Button("Send to Nalana") || (Event.current.type == EventType.KeyDown &&
                Event.current.keyCode == KeyCode.Return &&
                GUI.GetNameOfFocusedControl() == "NalanaCommandField"))
            {
                GUI.FocusControl(null);
                if (!string.IsNullOrWhiteSpace(_commandText))
                {
                    _isSending = true;
                    _statusMessage = "Sending…";
                    Repaint();
                    string cmd = _commandText;
                    _ = SendCommandAsync(cmd);
                }
                else
                {
                    _statusMessage = "Enter a command first.";
                }
            }
            EditorGUI.EndDisabledGroup();

            EditorGUILayout.Space(4);

            // Status
            EditorGUILayout.HelpBox(_statusMessage, MessageType.None);

            EditorGUILayout.Space(4);

            // History
            if (_history.Count > 0)
            {
                GUILayout.Label("Command History:", EditorStyles.boldLabel);
                _historyScrollPos = EditorGUILayout.BeginScrollView(_historyScrollPos, GUILayout.MaxHeight(160));
                for (int i = _history.Count - 1; i >= 0; i--)
                {
                    EditorGUILayout.LabelField($"- {_history[i]}", EditorStyles.wordWrappedLabel);
                }
                EditorGUILayout.EndScrollView();

                if (GUILayout.Button("Clear History"))
                {
                    _history.Clear();
                    _statusMessage = "History cleared.";
                }
            }
        }

        // --------------------------------------------------------------------
        // API communication
        // --------------------------------------------------------------------

        private async Task SendCommandAsync(string command)
        {
            string sceneContextJson = NalanaJson.Serialize(NalanaContext.GetSceneContext());

            string payload = "{" +
                "\"voice_command\":\"" + command.Replace("\"", "\\\"") + "\"," +
                "\"scene_context\":" + sceneContextJson + "," +
                "\"software\":\"unity\"" +
                "}";

            string responseJson = null;
            try
            {
                using (var request = new HttpRequestMessage(HttpMethod.Post, $"{_apiUrl.TrimEnd('/')}/v1/command"))
                {
                    request.Content = new StringContent(payload, Encoding.UTF8, "application/json");
                    if (!string.IsNullOrEmpty(_apiKey))
                        request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);

                    using (var response = await _httpClient.SendAsync(request))
                    {
                        response.EnsureSuccessStatusCode();
                        responseJson = await response.Content.ReadAsStringAsync();
                    }
                }
            }
            catch (Exception apiEx)
            {
                // Attempt Claude fallback
                responseJson = await CallClaudeFallbackAsync(command, sceneContextJson, apiEx.Message);
            }

            if (!string.IsNullOrEmpty(responseJson))
            {
                string code = NalanaJson.ExtractStringField(responseJson, "unity_csharp")
                           ?? NalanaJson.ExtractStringField(responseJson, "code")
                           ?? string.Empty;

                if (!string.IsNullOrWhiteSpace(code))
                {
                    EditorApplication.delayCall += () => ExecuteCode(code, command);
                }
                else
                {
                    EditorApplication.delayCall += () =>
                    {
                        _statusMessage = "API returned no executable code.";
                        _isSending = false;
                        Repaint();
                    };
                }
            }
            else
            {
                EditorApplication.delayCall += () =>
                {
                    _statusMessage = "No response received.";
                    _isSending = false;
                    Repaint();
                };
            }
        }

        private async Task<string> CallClaudeFallbackAsync(string command, string sceneContextJson, string apiError)
        {
            // Claude fallback via HTTP (avoids requiring Anthropic SDK in Unity)
            string anthropicKey = EditorPrefs.GetString("Nalana_AnthropicKey", string.Empty);
            if (string.IsNullOrEmpty(anthropicKey))
            {
                EditorApplication.delayCall += () =>
                {
                    _statusMessage = $"Nalana API failed ({apiError}) and no Anthropic key set.";
                    _isSending = false;
                    Repaint();
                };
                return null;
            }

            const string model = "claude-sonnet-4-6";
            const string apiEndpoint = "https://api.anthropic.com/v1/messages";

            string systemPrompt = "You are a Unity C# / UnityEditor API expert. Reply with ONLY a JSON object with key 'code' whose value is executable C#-via-EditorUtility code for the requested operation. No markdown.";
            string userMsg = $"Scene context: {sceneContextJson}\\n\\nCommand: {command}";

            string msgPayload = "{" +
                "\"model\":\"" + model + "\"," +
                "\"max_tokens\":1024," +
                "\"system\":\"" + systemPrompt.Replace("\"", "\\\"") + "\"," +
                "\"messages\":[{\"role\":\"user\",\"content\":\"" + userMsg.Replace("\"", "\\\"") + "\"}]" +
                "}";

            try
            {
                using (var request = new HttpRequestMessage(HttpMethod.Post, apiEndpoint))
                {
                    request.Headers.Add("x-api-key", anthropicKey);
                    request.Headers.Add("anthropic-version", "2023-06-01");
                    request.Content = new StringContent(msgPayload, Encoding.UTF8, "application/json");

                    using (var response = await _httpClient.SendAsync(request))
                    {
                        response.EnsureSuccessStatusCode();
                        string raw = await response.Content.ReadAsStringAsync();
                        // Extract text from Anthropic response (content[0].text)
                        string text = NalanaJson.ExtractStringField(raw, "text");
                        return text ?? raw;
                    }
                }
            }
            catch (Exception ex)
            {
                EditorApplication.delayCall += () =>
                {
                    _statusMessage = $"Both Nalana and Claude failed: {ex.Message}";
                    _isSending = false;
                    Repaint();
                };
                return null;
            }
        }

        // --------------------------------------------------------------------
        // Code execution
        // --------------------------------------------------------------------

        /// <summary>
        /// Execute C# code returned by the Nalana API.
        ///
        /// Unity does not provide a built-in eval/REPL, so this method uses
        /// EditorApplication.ExecuteMenuItem for simple menu commands, and for
        /// arbitrary code it writes a temporary .cs file and triggers a recompile.
        ///
        /// For complex automation, consider using the Roslyn scripting API
        /// (Microsoft.CodeAnalysis.CSharp.Scripting) added as a UPM package.
        /// </summary>
        private void ExecuteCode(string code, string command)
        {
            // Detect simple menu-based commands (e.g. "File/Save Project")
            if (code.TrimStart().StartsWith("EditorApplication.ExecuteMenuItem"))
            {
                try
                {
                    // Extract the menu path string
                    int openParen = code.IndexOf('(');
                    int closeParen = code.IndexOf(')');
                    if (openParen >= 0 && closeParen > openParen)
                    {
                        string menuPath = code.Substring(openParen + 1, closeParen - openParen - 1)
                                             .Trim('"', '\'', ' ');
                        EditorApplication.ExecuteMenuItem(menuPath);
                        _OnSuccess(command);
                        return;
                    }
                }
                catch { /* Fall through to script compilation approach */ }
            }

            // Write to a temporary ScriptableObject method and execute via compilation.
            // NOTE: This triggers a domain reload — suitable for automation only.
            try
            {
                string tempDir = Path.Combine(Application.dataPath, "Editor", "_NalanaTemp");
                Directory.CreateDirectory(tempDir);

                string scriptPath = Path.Combine(tempDir, "NalanaTempScript.cs");
                string scriptContent = $@"
// AUTO-GENERATED by Nalana — delete after use.
using UnityEngine;
using UnityEditor;

[InitializeOnLoad]
public static class NalanaTempRunner
{{
    static NalanaTempRunner()
    {{
        EditorApplication.delayCall += Run;
    }}

    static void Run()
    {{
        try
        {{
{IndentCode(code, 12)}
            Debug.Log(""[Nalana] Temp script executed: {EscapeForString(command)}"");
        }}
        catch (System.Exception ex)
        {{
            Debug.LogError(""[Nalana] Execution error: "" + ex.Message);
        }}
        finally
        {{
            // Self-destruct after execution
            string path = ""{EscapeForString(scriptPath.Replace("\\", "/"))}"";
            if (System.IO.File.Exists(path))
            {{
                System.IO.File.Delete(path);
                AssetDatabase.Refresh();
            }}
        }}
    }}
}}
";
                File.WriteAllText(scriptPath, scriptContent);
                AssetDatabase.Refresh();
                _statusMessage = $"Queued: {command} (recompiling…)";
                _history.Add(command);
                _history.RemoveRange(0, Mathf.Max(0, _history.Count - 10));
                _commandText = string.Empty;
            }
            catch (Exception ex)
            {
                _statusMessage = $"Execution error: {ex.Message}";
                Debug.LogError($"[Nalana] {ex}");
            }

            _isSending = false;
            Repaint();
        }

        private void _OnSuccess(string command)
        {
            _history.Add(command);
            _history.RemoveRange(0, Mathf.Max(0, _history.Count - 10));
            _commandText = string.Empty;
            _statusMessage = $"Done: {command}";
            _isSending = false;
            Repaint();
        }

        // --------------------------------------------------------------------
        // Utilities
        // --------------------------------------------------------------------

        private static string IndentCode(string code, int spaces)
        {
            string indent = new string(' ', spaces);
            return indent + code.Replace("\n", "\n" + indent);
        }

        private static string EscapeForString(string s)
        {
            return s.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\n");
        }
    }

    // -----------------------------------------------------------------------
    // Context menu on GameObjects
    // -----------------------------------------------------------------------

    /// <summary>
    /// Adds a "Send to Nalana" right-click option on GameObjects in the Hierarchy.
    /// </summary>
    public static class NalanaContextMenu
    {
        [MenuItem("GameObject/Send to Nalana", false, 49)]
        public static void SendSelectedToNalana()
        {
            GameObject go = Selection.activeGameObject;
            if (go == null)
            {
                EditorUtility.DisplayDialog("Nalana", "No GameObject selected.", "OK");
                return;
            }

            string info =
                $"Name: {go.name}\n" +
                $"Tag: {go.tag}\n" +
                $"Layer: {LayerMask.LayerToName(go.layer)}\n" +
                $"Components: {string.Join(", ", GetComponentNames(go))}\n" +
                $"Position: {go.transform.position}";

            NalanaWindow.OpenWindow();
            EditorUtility.DisplayDialog("Nalana — Object Info", info, "OK");
        }

        [MenuItem("GameObject/Send to Nalana", true)]
        public static bool ValidateSendSelectedToNalana()
        {
            return Selection.activeGameObject != null;
        }

        private static string[] GetComponentNames(GameObject go)
        {
            var comps = go.GetComponents<Component>();
            var names = new List<string>();
            foreach (var c in comps)
            {
                if (c != null) names.Add(c.GetType().Name);
            }
            return names.ToArray();
        }
    }

} // namespace Nalana

#endif // UNITY_EDITOR
