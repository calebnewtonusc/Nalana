#!/usr/bin/env bash
# scripts/start_vllm.sh - Start vLLM synthesis servers on GPUs 0-15.
#
# GPU allocation:
#   GPUs 0-3:   Qwen2.5-72B text synthesis  (port 8001)
#   GPUs 4-7:   Qwen2.5-72B text synthesis  (port 8002)
#   GPUs 8-11:  Qwen2-VL-72B vision synth   (port 8003)
#   GPUs 12-15: Qwen2-VL-72B vision synth   (port 8004)
#   GPUs 16-17: reserved for Blender validation workers
#
# Usage:
#   bash scripts/start_vllm.sh              # start all 4 servers
#   bash scripts/start_vllm.sh --text-only  # only text servers (ports 8001-8002)
#   bash scripts/start_vllm.sh --vision-only # only vision servers (ports 8003-8004)
#   VLLM_API_KEY=mykey bash scripts/start_vllm.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

# ─── Configuration ────────────────────────────────────────────────────────────
TEXT_MODEL="${VLLM_TEXT_MODEL:-Qwen/Qwen2.5-72B-Instruct}"
VISION_MODEL="${VLLM_VISION_MODEL:-Qwen/Qwen2-VL-72B-Instruct}"
API_KEY="${VLLM_API_KEY:-nalana}"
HEALTH_TIMEOUT="${VLLM_HEALTH_TIMEOUT:-300}"  # seconds to wait for each server

TEXT_ONLY=false
VISION_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --text-only)   TEXT_ONLY=true ;;
        --vision-only) VISION_ONLY=true ;;
    esac
done

# ─── Colors ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

log()  { echo -e "$(date '+%H:%M:%S')  $1"; }
pass() { echo -e "$(date '+%H:%M:%S')  ${GREEN}[UP]${NC}    $1"; }
fail() { echo -e "$(date '+%H:%M:%S')  ${RED}[FAIL]${NC}  $1"; }
info() { echo -e "$(date '+%H:%M:%S')  ${YELLOW}[INFO]${NC}  $1"; }

# ─── Check vLLM is installed ──────────────────────────────────────────────────
if ! command -v vllm &>/dev/null; then
    if ! python3 -c "import vllm" &>/dev/null 2>&1; then
        echo -e "${RED}ERROR: vLLM is not installed.${NC}"
        echo "Install with: pip install vllm"
        exit 1
    fi
fi

# ─── Helper: start one vLLM server ────────────────────────────────────────────
start_server() {
    local name="$1"
    local gpus="$2"
    local model="$3"
    local port="$4"
    local tp_size="$5"        # tensor parallel size
    local extra_args="${6:-}"

    local log_file="$LOG_DIR/vllm_${port}.log"

    log "Starting $name (GPUs $gpus, port $port)..."
    log "  Model: $model"
    log "  Log:   $log_file"

    CUDA_VISIBLE_DEVICES="$gpus" \
    nohup python3 -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --tensor-parallel-size "$tp_size" \
        --port "$port" \
        --api-key "$API_KEY" \
        --host 0.0.0.0 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.92 \
        --trust-remote-code \
        --disable-log-requests \
        $extra_args \
        > "$log_file" 2>&1 &

    echo $! > "$LOG_DIR/vllm_${port}.pid"
    log "  PID: $(cat "$LOG_DIR/vllm_${port}.pid")"
}

# ─── Helper: wait for server health ───────────────────────────────────────────
wait_healthy() {
    local name="$1"
    local port="$2"
    local timeout="$HEALTH_TIMEOUT"
    local elapsed=0
    local interval=5

    log "Waiting for $name (port $port) to be healthy..."

    while [ "$elapsed" -lt "$timeout" ]; do
        if curl -sf "http://localhost:${port}/v1/models" \
            -H "Authorization: Bearer $API_KEY" \
            > /dev/null 2>&1; then
            pass "$name is UP at http://localhost:${port}"
            return 0
        fi
        sleep "$interval"
        elapsed=$((elapsed + interval))
        if [ $((elapsed % 30)) -eq 0 ]; then
            info "$name still starting... ($elapsed/${timeout}s)"
        fi
    done

    fail "$name failed to start within ${timeout}s"
    fail "Check logs: $LOG_DIR/vllm_${port}.log"
    return 1
}

# ─── Start servers ────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Nalana vLLM Server Startup${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
info "API key: ${API_KEY:0:8}..."
info "Text model:   $TEXT_MODEL"
info "Vision model: $VISION_MODEL"
echo ""

PIDS=()

if [ "$VISION_ONLY" = false ]; then
    # Text synthesis servers (Qwen2.5-72B, 4 GPUs each, tensor-parallel=4)
    start_server "text-synth-1" "0,1,2,3"   "$TEXT_MODEL" 8001 4
    PIDS+=("8001")

    start_server "text-synth-2" "4,5,6,7"   "$TEXT_MODEL" 8002 4
    PIDS+=("8002")
fi

if [ "$TEXT_ONLY" = false ]; then
    # Vision synthesis servers (Qwen2-VL-72B, 4 GPUs each, tensor-parallel=4)
    start_server "vision-synth-1" "8,9,10,11"   "$VISION_MODEL" 8003 4 "--max-num-seqs 8"
    PIDS+=("8003")

    start_server "vision-synth-2" "12,13,14,15" "$VISION_MODEL" 8004 4 "--max-num-seqs 8"
    PIDS+=("8004")
fi

# GPUs 16-17 are intentionally left free for Blender validation workers

echo ""
echo -e "${BOLD}Waiting for all servers to be healthy...${NC}"
echo "(This takes 3-8 minutes for 72B models to load)"
echo ""

FAILED=()
for port in "${PIDS[@]}"; do
    case "$port" in
        8001) name="text-synth-1" ;;
        8002) name="text-synth-2" ;;
        8003) name="vision-synth-1" ;;
        8004) name="vision-synth-2" ;;
        *) name="server-$port" ;;
    esac

    if ! wait_healthy "$name" "$port"; then
        FAILED+=("$name:$port")
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ${#FAILED[@]} -eq 0 ]; then
    echo -e "${GREEN}${BOLD}All vLLM servers are LIVE${NC}"
    echo ""
    echo "  Text synthesis:   http://localhost:8001 http://localhost:8002"
    if [ "$TEXT_ONLY" = false ]; then
        echo "  Vision synthesis: http://localhost:8003 http://localhost:8004"
    fi
    echo "  GPUs 16-17:       reserved for Blender validation"
    echo ""
    echo "  Next: python synthesize_bulk.py --backend vllm \\"
    echo "           --vllm-urls http://localhost:8001 http://localhost:8002"
else
    echo -e "${RED}${BOLD}${#FAILED[@]} server(s) failed to start:${NC}"
    for s in "${FAILED[@]}"; do
        echo -e "  ${RED}x${NC} $s"
    done
    echo ""
    echo "Check logs in $LOG_DIR/"
    exit 1
fi
