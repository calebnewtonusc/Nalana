#!/usr/bin/env bash
# scripts/run_all.sh - Master pipeline: complete Nalana training run.
#
# Usage:
#   bash scripts/run_all.sh                    # full pipeline
#   bash scripts/run_all.sh --from-step 6      # resume from step 6
#   bash scripts/run_all.sh --skip-training    # data + synthesis only
#   bash scripts/run_all.sh --skip-discovery   # skip YouTube/API discovery
#   DRY_RUN=1 bash scripts/run_all.sh          # print commands, don't run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

[ -f "$ROOT/.env" ] && {
	set -o allexport
	source "$ROOT/.env"
	set +o allexport
}

FROM_STEP=1
SKIP_TRAINING=false
SKIP_DISCOVERY=false
DRY_RUN="${DRY_RUN:-0}"
for arg in "$@"; do
	case "$arg" in
	--from-step=*) FROM_STEP="${arg#*=}" ;;
	--from-step)
		shift
		FROM_STEP="$1"
		;;
	--skip-training) SKIP_TRAINING=true ;;
	--skip-discovery) SKIP_DISCOVERY=true ;;
	--dry-run) DRY_RUN=1 ;;
	esac
done

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'
STEP=0
TIMINGS=()
START=$(date +%s)

step() {
	STEP=$((STEP + 1))
	[ "$STEP" -lt "$FROM_STEP" ] && {
		echo -e "  ${YELLOW}[SKIP]${NC} Step $STEP: $1"
		return 0
	}
	echo -e "\n${CYAN}━━━ Step $STEP: $1 ━━━${NC}"
	STEP_START=$(date +%s)
	return 1
}
run() {
	echo -e "  ${CYAN}>>>${NC} $*"
	[ "$DRY_RUN" = "1" ] && return 0
	eval "$*"
}
done_step() {
	local e=$(($(date +%s) - STEP_START))
	echo -e "  ${GREEN}[DONE]${NC} $((e / 60))m$((e % 60))s"
	TIMINGS+=("Step $STEP: $((e / 60))m$((e % 60))s")
}

cd "$ROOT"
echo -e "\n${BOLD}Nalana Pipeline — 18x A6000 (864GB VRAM)${NC}\nStarted: $(date)\n"
[ "$DRY_RUN" = "1" ] && echo -e "${YELLOW}DRY RUN${NC}\n"

# ── Step 1: Environment check ─────────────────────────────────────────────────
if ! step "Environment validation"; then
	run "bash scripts/check_env.sh"
	done_step
fi

# ── Step 2: Autonomous dataset discovery (NEW — runs first, finds everything) ─
if ! step "Autonomous dataset discovery (LLM-guided, all sources)"; then
	if [ "$SKIP_DISCOVERY" = true ]; then echo "  [SKIP]"; else
		run "python3 data_discovery.py --all --output data/discovered_sources.json"
	fi
	done_step
fi

# ── Step 3: YouTube + channel discovery ──────────────────────────────────────
if ! step "YouTube video discovery (all categories, not just tutorials)"; then
	if [ "$SKIP_DISCOVERY" = true ]; then echo "  [SKIP]"; else
		run "python3 discovery/discover_v2.py \
            --sources data/discovered_sources.json \
            --all-software --all-categories \
            --channels --search \
            --api-key \"${YOUTUBE_API_KEY:-}\""
	fi
	done_step
fi

# ── Step 4: API harvest (3D model repos) ─────────────────────────────────────
if ! step "API harvest (Sketchfab, Polyhaven, Thingiverse, GrabCAD, NASA...)"; then
	if [ "$SKIP_DISCOVERY" = true ]; then echo "  [SKIP]"; else
		run "python3 discovery/api_harvest.py --all"
	fi
	done_step
fi

# ── Step 5: Bulk transcript fetch ─────────────────────────────────────────────
if ! step "Bulk transcript fetch (30 workers)"; then
	run "python3 discovery/fetch_bulk.py --workers 30"
	done_step
fi

# ── Step 6: Objaverse prep ────────────────────────────────────────────────────
if ! step "Objaverse 3D dataset prep (50k objects)"; then
	run "python3 discovery/objaverse_prep.py --limit 50000"
	done_step
fi

# ── Step 7: Start vLLM servers ───────────────────────────────────────────────
if ! step "Launch vLLM servers (GPUs 0-15)"; then
	if [ "$SKIP_TRAINING" = true ]; then echo "  [SKIP]"; else
		run "bash scripts/start_vllm.sh"
	fi
	done_step
fi

# ── Step 8: Synthesize training pairs (text) ─────────────────────────────────
if ! step "Synthesize training pairs — all software, all domains (Qwen2.5-72B)"; then
	if [ "$SKIP_TRAINING" = true ]; then
		run "python3 synthesis/synthesize_bulk.py --backend claude"
	else
		run "python3 synthesis/synthesize_bulk.py \
            --backend vllm \
            --vllm-urls http://localhost:8001 http://localhost:8002 \
            --sources data/discovered_sources.json"
	fi
	done_step
fi

# ── Step 9: Form annotation (vision) ─────────────────────────────────────────
if ! step "Annotate 3D forms — VLM (Qwen2-VL-72B)"; then
	if [ "$SKIP_TRAINING" = true ]; then
		run "python3 synthesis/annotate_forms.py --backend claude"
	else
		run "python3 synthesis/annotate_forms.py \
            --backend vllm --vllm-url http://localhost:8003"
	fi
	done_step
fi

# ── Step 10: Cross-software integration data ──────────────────────────────────
if ! step "Collect cross-software and integration data"; then
	run "python3 integrations/collect_design_physics.py --all"
	done_step
fi

# ── Step 11: Generate agent training pairs ────────────────────────────────────
if ! step "Generate domain agent training pairs (arch, cad, animation, QA...)"; then
	run "python3 agents/arch_agent.py --generate-pairs"
	run "python3 agents/cad_agent.py --generate-pairs"
	run "python3 agents/animation_agent.py --generate-pairs"
	run "python3 agents/production_agent.py --generate-pairs"
	run "python3 agents/qa_agent.py --generate-pairs"
	done_step
fi

# ── Step 12: Validate dataset ─────────────────────────────────────────────────
if ! step "Quality validate + dedup all synthesized data"; then
	run "python3 validation/validate.py"
	done_step
fi

# ── Step 13: Blender execution validation ────────────────────────────────────
if ! step "Blender headless execution validation (6 workers)"; then
	run "python3 validation/validate_blender.py --workers 6"
	done_step
fi

# ── Step 14: Multi-turn + DPO pair generation ────────────────────────────────
if ! step "Generate multi-turn conversations + DPO preference pairs"; then
	run "python3 synthesis/multi_turn.py --from-pairs --synthetic"
	run "python3 synthesis/generate_dpo_pairs.py --model checkpoints/nalana-sft/final"
	done_step
fi

# ── Step 15: Training data prep ───────────────────────────────────────────────
if ! step "Training data prep — all streams, quality weights, curriculum order"; then
	run "python3 training/train_prep.py"
	done_step
fi

if [ "$SKIP_TRAINING" = true ]; then
	echo -e "\n${YELLOW}━━━ Skipping training (--skip-training) ━━━${NC}"
	STEP=$((STEP + 4))
else

	# ── Step 16: Stage 1 — SFT ────────────────────────────────────────────────────
	if ! step "Stage 1 — Supervised Fine-Tuning (ZeRO-3, 18 GPUs, ~6h)"; then
		run "deepspeed --num_gpus=18 training/train.py \
        --deepspeed training/ds_config.json \
        --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --data-dir data/train \
        --output-dir checkpoints/nalana-sft \
        --epochs 3"
		done_step
	fi

	# ── Step 17: Stage 2 — Execution RL ──────────────────────────────────────────
	if ! step "Stage 2 — Execution RL (GRPO + headless Blender reward, ~4h)"; then
		run "deepspeed --num_gpus=18 training/train_rl.py \
        --deepspeed training/ds_config_rl.json \
        --base-model checkpoints/nalana-sft/final \
        --output-dir checkpoints/nalana-rl \
        --blender-workers 12"
		done_step
	fi

	# ── Step 18: Stage 3 — DPO ───────────────────────────────────────────────────
	if ! step "Stage 3 — DPO conversation quality (~2h)"; then
		run "deepspeed --num_gpus=18 training/train_dpo.py \
        --deepspeed training/ds_config.json \
        --base-model checkpoints/nalana-rl/final \
        --output-dir checkpoints/nalana-final"
		done_step
	fi

	# ── Step 19: NalanaBench evaluation ──────────────────────────────────────────
	if ! step "NalanaBench evaluation (all 500 prompts, 8 categories)"; then
		run "python3 evaluation/eval.py --model checkpoints/nalana-final"
		done_step
	fi

fi # end skip-training

# ── Step 20: Deploy ───────────────────────────────────────────────────────────
if ! step "Deploy serving stack (docker compose up -d)"; then
	run "cd deploy && docker compose up -d && cd \"$ROOT\""
	done_step
fi

# ── Step 21: Health check ─────────────────────────────────────────────────────
if ! step "Health check + smoke test"; then
	run "python3 scripts/health_check.py"
	done_step
fi

# ── Summary ───────────────────────────────────────────────────────────────────
ELAPSED=$(($(date +%s) - START))
echo -e "\n${GREEN}${BOLD}Pipeline complete!${NC}"
echo "  Total: $((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m"
echo "  Model: checkpoints/nalana-final"
echo "  API:   cd deploy && docker compose ps"
for t in "${TIMINGS[@]}"; do echo "    $t"; done
