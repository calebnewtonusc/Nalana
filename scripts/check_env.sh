#!/usr/bin/env bash
# scripts/check_env.sh - Validate environment before running the Nalana pipeline.
#
# Usage: bash scripts/check_env.sh
# Exit code: 0 if environment is ready, 1 if there are blocking issues.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

ISSUES=()
WARNINGS=()

# ─── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BOLD='\033[1m'
NC='\033[0m'

pass()  { echo -e "  ${GREEN}[PASS]${NC}  $1"; }
warn()  { echo -e "  ${YELLOW}[WARN]${NC}  $1"; WARNINGS+=("$1"); }
fail()  { echo -e "  ${RED}[FAIL]${NC}  $1"; ISSUES+=("$1"); }

echo ""
echo -e "${BOLD}Nalana Environment Check${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ─── 1. Python version ────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}[1/6] Python Version${NC}"
if command -v python3 &>/dev/null; then
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
    MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
    if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ]; then
        pass "Python $PYTHON_VERSION >= 3.10"
    else
        fail "Python $PYTHON_VERSION < 3.10 — upgrade required (3.11 recommended)"
    fi
else
    fail "python3 not found in PATH"
fi

# ─── 2. Required environment variables ───────────────────────────────────────
echo ""
echo -e "${BOLD}[2/6] Environment Variables (.env)${NC}"

# Load .env if present
if [ -f "$ROOT_DIR/.env" ]; then
    set -o allexport
    # shellcheck disable=SC1091
    source "$ROOT_DIR/.env" 2>/dev/null || true
    set +o allexport
    pass ".env file found and loaded"
else
    warn ".env file not found — expecting variables to be set in shell environment"
fi

REQUIRED_VARS=(
    "ANTHROPIC_API_KEY:Anthropic API (claude.ai/settings)"
    "YOUTUBE_API_KEY:YouTube Data API v3 (console.cloud.google.com)"
    "VLLM_API_KEY:vLLM server auth token (set to any secret string)"
)

OPTIONAL_VARS=(
    "WANDB_API_KEY:Weights & Biases tracking (wandb.ai)"
    "HF_TOKEN:HuggingFace Hub token (huggingface.co/settings/tokens)"
    "OPENAI_API_KEY:OpenAI API (optional fallback)"
)

for entry in "${REQUIRED_VARS[@]}"; do
    VAR="${entry%%:*}"
    DESC="${entry#*:}"
    VAL="${!VAR:-}"
    if [ -n "$VAL" ]; then
        # Show first 8 chars only
        PREVIEW="${VAL:0:8}..."
        pass "$VAR is set ($PREVIEW)"
    else
        fail "$VAR is not set — get it at: $DESC"
    fi
done

for entry in "${OPTIONAL_VARS[@]}"; do
    VAR="${entry%%:*}"
    DESC="${entry#*:}"
    VAL="${!VAR:-}"
    if [ -n "$VAL" ]; then
        pass "$VAR is set (optional)"
    else
        warn "$VAR not set (optional) — $DESC"
    fi
done

# ─── 3. GPU availability ──────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}[3/6] GPU Availability${NC}"

if command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
    GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -3 | tr '\n' ', ')
    TOTAL_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s}')
    TOTAL_VRAM_GB=$((TOTAL_VRAM_MB / 1024))

    pass "nvidia-smi found: $GPU_COUNT GPU(s) detected"
    pass "GPUs: $GPU_NAMES"
    pass "Total VRAM: ~${TOTAL_VRAM_GB}GB"

    if [ "$GPU_COUNT" -lt 8 ]; then
        warn "Only $GPU_COUNT GPU(s) found — full training pipeline requires 18x A6000 (864GB VRAM)"
        warn "For single-node training: use CUDA_VISIBLE_DEVICES to select GPUs"
    elif [ "$GPU_COUNT" -ge 18 ]; then
        pass "18+ GPUs available — ready for full ZeRO-3 distributed training"
    fi

    if [ "$TOTAL_VRAM_GB" -lt 80 ]; then
        warn "Total VRAM ${TOTAL_VRAM_GB}GB < 80GB — synthesis will work, but training may OOM"
        warn "Reduce batch size in ds_config.json or use more GPUs"
    fi
else
    warn "nvidia-smi not found — GPU training and vLLM synthesis will not be available"
    warn "Data collection and synthesis (Claude API fallback) will still work"
fi

# ─── 4. Blender installation ──────────────────────────────────────────────────
echo ""
echo -e "${BOLD}[4/6] Blender Installation${NC}"

BLENDER_CMD=""
for candidate in blender /Applications/Blender.app/Contents/MacOS/Blender /usr/bin/blender /usr/local/bin/blender; do
    if command -v "$candidate" &>/dev/null 2>&1 || [ -f "$candidate" ]; then
        BLENDER_CMD="$candidate"
        break
    fi
done

if [ -n "$BLENDER_CMD" ]; then
    BLENDER_VER=$("$BLENDER_CMD" --version 2>/dev/null | head -1 || echo "unknown")
    pass "Blender found: $BLENDER_CMD"
    pass "Version: $BLENDER_VER"
else
    warn "Blender not found in PATH — needed for validate_blender.py and plugins"
    warn "Install from: https://www.blender.org/download/"
    warn "Or set BLENDER_PATH in .env"
fi

# Check BLENDER_PATH env override
if [ -n "${BLENDER_PATH:-}" ] && [ -f "$BLENDER_PATH" ]; then
    pass "BLENDER_PATH override set: $BLENDER_PATH"
fi

# ─── 5. Disk space ────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}[5/6] Disk Space${NC}"

# Check space on the project drive
DISK_AVAIL_GB=$(df -BG "$ROOT_DIR" 2>/dev/null | awk 'NR==2 {gsub(/G/, "", $4); print $4}' || echo "0")

if [ "$DISK_AVAIL_GB" -ge 500 ]; then
    pass "${DISK_AVAIL_GB}GB available — sufficient for full dataset (~300GB)"
elif [ "$DISK_AVAIL_GB" -ge 100 ]; then
    warn "${DISK_AVAIL_GB}GB available — enough for partial dataset, but full pipeline needs ~500GB"
    warn "Consider mounting additional storage or using --limit flags"
else
    fail "${DISK_AVAIL_GB}GB available — insufficient. Full dataset requires ~500GB free"
    fail "Free up disk space or change data directory (set DATA_DIR in .env)"
fi

# ─── 6. RAM ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}[6/6] System RAM${NC}"

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    RAM_GB=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
elif [[ "$OSTYPE" == "darwin"* ]]; then
    RAM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
    RAM_GB=$((RAM_BYTES / 1024 / 1024 / 1024))
else
    RAM_GB=0
fi

if [ "$RAM_GB" -ge 64 ]; then
    pass "${RAM_GB}GB RAM — sufficient for ZeRO-3 CPU offloading"
elif [ "$RAM_GB" -ge 32 ]; then
    warn "${RAM_GB}GB RAM — minimum for CPU offload; 64GB+ recommended for ZeRO-3"
elif [ "$RAM_GB" -gt 0 ]; then
    warn "${RAM_GB}GB RAM — below recommended 64GB; data collection will work, training may be slow"
else
    warn "Could not detect RAM size"
fi

# ─── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

N_ISSUES=${#ISSUES[@]}
N_WARNS=${#WARNINGS[@]}

if [ "$N_ISSUES" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}Environment: READY${NC}"
    if [ "$N_WARNS" -gt 0 ]; then
        echo -e "${YELLOW}  ($N_WARNS warning(s) — see above)${NC}"
    fi
    echo ""
    echo "  Run the full pipeline: bash scripts/run_all.sh"
    exit 0
else
    echo -e "${RED}${BOLD}Environment: NOT READY ($N_ISSUES blocking issue(s))${NC}"
    echo ""
    echo "  Blocking issues:"
    for issue in "${ISSUES[@]}"; do
        echo -e "    ${RED}x${NC} $issue"
    done
    if [ "$N_WARNS" -gt 0 ]; then
        echo ""
        echo "  Warnings ($N_WARNS):"
        for w in "${WARNINGS[@]}"; do
            echo -e "    ${YELLOW}!${NC} $w"
        done
    fi
    echo ""
    exit 1
fi
