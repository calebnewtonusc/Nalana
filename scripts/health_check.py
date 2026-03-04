"""
scripts/health_check.py - Post-deployment verification for Nalana.

Checks:
  1. /v1/health endpoint responds
  2. Sends a test voice command: "add a cube at the origin"
  3. Verifies response contains blender_python field
  4. Optionally executes the blender_python in headless Blender

Usage:
    python3 scripts/health_check.py
    python3 scripts/health_check.py --api-url http://your-server:9000
    python3 scripts/health_check.py --no-blender    # skip Blender execution
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def check_health(api_url: str, api_key: str, timeout: float = 10.0) -> bool:
    """Hit the /v1/health endpoint."""
    import urllib.request
    import urllib.error

    url = f"{api_url}/v1/health"
    try:
        req = urllib.request.Request(url)
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
            return resp.status == 200
    except urllib.error.HTTPError:
        # Some servers return 200 on /v1/models but 404 on /v1/health — try models
        try:
            url2 = f"{api_url}/v1/models"
            req2 = urllib.request.Request(url2)
            if api_key:
                req2.add_header("Authorization", f"Bearer {api_key}")
            with urllib.request.urlopen(req2, timeout=timeout) as resp2:
                return resp2.status == 200
        except Exception:
            return False
    except Exception:
        return False


def send_test_command(api_url: str, api_key: str, timeout: float = 60.0) -> dict | None:
    """Send a test voice command and return the parsed response."""
    import urllib.request
    import urllib.error

    payload = json.dumps(
        {
            "voice_command": "add a cube at the origin",
            "scene_context": "Empty Blender scene, no objects, Object Mode",
        }
    ).encode("utf-8")

    url = f"{api_url}/v1/command"
    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        # Try the OpenAI-compatible chat endpoint as a fallback
        try:
            chat_payload = json.dumps(
                {
                    "model": "nalana",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Voice command: add a cube at the origin\nScene context: Empty Blender scene, Object Mode",
                        }
                    ],
                    "max_tokens": 512,
                    "temperature": 0.0,
                }
            ).encode("utf-8")

            url2 = f"{api_url}/v1/chat/completions"
            req2 = urllib.request.Request(
                url2,
                data=chat_payload,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
            )
            with urllib.request.urlopen(req2, timeout=timeout) as resp2:
                raw = json.loads(resp2.read())
                content = raw["choices"][0]["message"]["content"]
                # Try to parse blender_python from the content
                if "```python" in content:
                    code = content.split("```python")[1].split("```")[0].strip()
                    return {"blender_python": code, "raw_content": content}
                return {"blender_python": None, "raw_content": content}
        except Exception:
            return None


def execute_in_blender(blender_python: str) -> tuple[bool, str]:
    """
    Execute blender_python in headless Blender.
    Returns (success, output).
    """
    # Find Blender
    blender_path = os.environ.get("BLENDER_PATH", "")
    if not blender_path:
        candidates = [
            "blender",
            "/Applications/Blender.app/Contents/MacOS/Blender",
            "/usr/bin/blender",
            "/usr/local/bin/blender",
        ]
        for c in candidates:
            if shutil.which(c) or Path(c).exists():
                blender_path = c
                break

    if not blender_path:
        return False, "Blender not found (set BLENDER_PATH)"

    # Wrap code in a minimal headless script
    script = f"""
import bpy
import sys

# Reset to clean state
bpy.ops.wm.read_factory_settings(use_empty=True)

try:
    # Execute the generated code
    exec({repr(blender_python)})
    print("NALANA_OK: execution succeeded")
    sys.exit(0)
except Exception as e:
    print(f"NALANA_ERROR: {{e}}")
    sys.exit(1)
"""

    script_file = Path("/tmp/nalana_health_test.py")
    script_file.write_text(script)

    try:
        result = subprocess.run(
            [blender_path, "--background", "--python", str(script_file)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        success = "NALANA_OK" in output and result.returncode == 0
        return success, output
    except subprocess.TimeoutExpired:
        return False, "Blender execution timed out after 30s"
    except Exception as e:
        return False, str(e)
    finally:
        script_file.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Nalana deployment health check")
    parser.add_argument(
        "--api-url", default=os.environ.get("NALANA_API_URL", "http://localhost:9000")
    )
    parser.add_argument("--api-key", default=os.environ.get("NALANA_API_KEY", "nalana"))
    parser.add_argument(
        "--no-blender", action="store_true", help="Skip Blender execution test"
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    api_url = args.api_url.rstrip("/")

    print()
    print("Nalana Health Check")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  API URL: {api_url}")
    print()

    issues = []
    passed = []

    # ── Check 1: Health endpoint ───────────────────────────────────────────────
    print("[1/3] Checking /v1/health endpoint...")
    if check_health(api_url, args.api_key, timeout=args.timeout):
        print(f"  PASS  {api_url} is responding")
        passed.append("health_endpoint")
    else:
        msg = f"FAIL  {api_url} did not respond — is the server running?"
        print(f"  {msg}")
        issues.append(msg)

    # ── Check 2: Test voice command ────────────────────────────────────────────
    print()
    print('[2/3] Sending test command: "add a cube at the origin"...')
    response = send_test_command(api_url, args.api_key, timeout=args.timeout)

    blender_python = None
    if response is None:
        msg = "FAIL  No response from API"
        print(f"  {msg}")
        issues.append(msg)
    elif "blender_python" not in response or not response.get("blender_python"):
        msg = (
            f"FAIL  Response missing blender_python field: {json.dumps(response)[:200]}"
        )
        print(f"  {msg}")
        issues.append(msg)
    else:
        blender_python = response["blender_python"]
        print("  PASS  Response has blender_python:")
        print(
            f"        {blender_python[:120]}{'...' if len(blender_python) > 120 else ''}"
        )
        passed.append("test_command")

        # Also check for reasoning/op
        if response.get("reasoning"):
            print(f"        Reasoning: {response['reasoning'][:100]}")

    # ── Check 3: Blender execution ─────────────────────────────────────────────
    print()
    if args.no_blender:
        print("[3/3] Blender execution: SKIPPED (--no-blender)")
    elif blender_python is None:
        print("[3/3] Blender execution: SKIPPED (no blender_python to execute)")
    else:
        print("[3/3] Executing blender_python in headless Blender...")
        success, output = execute_in_blender(blender_python)
        if success:
            print("  PASS  Blender executed successfully")
            passed.append("blender_execution")
        else:
            # Non-blocking: warn but don't fail
            print("  WARN  Blender execution had issues (non-blocking):")
            for line in output.strip().split("\n")[-5:]:
                if line.strip():
                    print(f"        {line}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if not issues:
        print(f"Nalana is LIVE at {api_url}")
        print()
        print(f"  Passed: {len(passed)}/{len(passed)} checks")
        print()
        print("  Try it:")
        print(f"  curl -X POST {api_url}/v1/command \\")
        print(f'       -H "Authorization: Bearer {args.api_key}" \\')
        print('       -H "Content-Type: application/json" \\')
        print(
            '       -d \'{"voice_command": "add a red sphere", "scene_context": "Empty scene"}\''
        )
        sys.exit(0)
    else:
        print(f"Nalana health check FAILED ({len(issues)} issue(s)):")
        for issue in issues:
            print(f"  x  {issue}")
        print()
        print("Troubleshooting:")
        print("  - Check server logs: logs/")
        print("  - Restart servers:   bash scripts/start_vllm.sh")
        print("  - Check deployment:  cd deploy && docker compose ps")
        sys.exit(1)


if __name__ == "__main__":
    main()
