# GPU Cluster Setup Guide

Complete, exact steps to go from SSH access to a running Nalana training run.
Assumes 18× NVIDIA A6000 (48GB each) or equivalent A100s on a Linux cluster.

---

## 1. SSH and initial orientation

```bash
ssh user@gpu-cluster-host

# Verify GPU inventory
nvidia-smi --list-gpus
# Expect: 18 GPUs listed

# Check CUDA version
nvcc --version
# Need: CUDA 12.1+

# Check disk space — pipeline needs ~500GB
df -h /workspace
# Need: ≥ 500GB free

# Check RAM — ZeRO-3 optimizer offload needs ≥ 256GB, ideally 512GB
free -h
```

---

## 2. Clone the repository

```bash
cd /workspace
git clone https://github.com/calebnewtonusc/Nalana.git nalana
cd nalana
```

---

## 3. Python environment

```bash
# Create isolated environment (Python 3.11 recommended)
conda create -n nalana python=3.11 -y
conda activate nalana

# Verify version
python --version  # should print Python 3.11.x
```

---

## 4. Install dependencies

```bash
# Core dependencies (works on CPU too)
pip install -r requirements.txt

# GPU-specific: flash attention (compile takes 5-10 min, skip if short on time)
pip install flash-attn --no-build-isolation

# vLLM for synthesis servers (GPU only)
pip install vllm

# Verify torch sees GPUs
python -c "import torch; print(torch.cuda.device_count(), 'GPUs')"
# Expect: 18 GPUs
```

---

## 5. Configure environment variables

```bash
cp .env.example .env
nano .env  # or vim .env
```

Fill in these required values:

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-api03-...    # from claude.ai/settings
YOUTUBE_API_KEY=AIza...               # from console.cloud.google.com → YouTube Data API v3
VLLM_API_KEY=nalana                   # any secret string, used to auth vLLM endpoints

# Recommended
WANDB_API_KEY=...                     # from wandb.ai — training metrics dashboard
HF_TOKEN=hf_...                       # from huggingface.co/settings — upload final model
BLENDER_PATH=/usr/bin/blender         # path to Blender binary (needed for RL validation)

# Optional inference deployment
REDIS_URL=redis://localhost:6379/0
SESSION_SECRET=...                    # any long random string for session tokens
```

Verify:
```bash
bash scripts/check_env.sh
# Should pass all checks and print: ✓ Environment ready
```

---

## 6. Install Blender (for RL validation)

The execution RL stage and Blender validation harness require headless Blender.

```bash
# Check if already installed
blender --version 2>/dev/null || echo "not found"

# Install Blender 4.2 LTS (stable, long-term support)
wget https://download.blender.org/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz
tar -xf blender-4.2.0-linux-x64.tar.xz -C /opt/
ln -s /opt/blender-4.2.0-linux-x64/blender /usr/local/bin/blender

# Add to .env if not in PATH
echo 'BLENDER_PATH=/usr/local/bin/blender' >> .env

# Test headless execution
blender --background --python-expr "import bpy; print('Blender OK')"
# Expect: "Blender OK" in output
```

---

## 7. Launch vLLM synthesis servers

These run on GPUs 0–15 (4 GPUs per instance, 4 instances = 16 GPUs).
GPUs 16–17 are reserved for training.

```bash
# Start all 4 vLLM synthesis servers in background
bash scripts/start_vllm.sh

# Verify all 4 are up (wait ~60s for model load)
for port in 8001 8002 8003 8004; do
    curl -s http://localhost:$port/health && echo " :$port OK" || echo " :$port FAIL"
done
```

If `scripts/start_vllm.sh` isn't available, start manually:
```bash
# Instance 1 (GPUs 0-3)
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup vllm serve Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 --port 8001 --api-key nalana \
    --gpu-memory-utilization 0.9 --max-model-len 8192 \
    > logs/vllm_8001.log 2>&1 &

# Instance 2 (GPUs 4-7)
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup vllm serve Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 --port 8002 --api-key nalana \
    --gpu-memory-utilization 0.9 --max-model-len 8192 \
    > logs/vllm_8002.log 2>&1 &

# Instance 3 (GPUs 8-11) — vision model for form annotation
CUDA_VISIBLE_DEVICES=8,9,10,11 nohup vllm serve Qwen/Qwen2-VL-72B-Instruct \
    --tensor-parallel-size 4 --port 8003 --api-key nalana \
    --gpu-memory-utilization 0.9 --max-model-len 4096 \
    > logs/vllm_8003.log 2>&1 &

# Instance 4 (GPUs 12-15) — vision model for form annotation
CUDA_VISIBLE_DEVICES=12,13,14,15 nohup vllm serve Qwen/Qwen2-VL-72B-Instruct \
    --tensor-parallel-size 4 --port 8004 --api-key nalana \
    --gpu-memory-utilization 0.9 --max-model-len 4096 \
    > logs/vllm_8004.log 2>&1 &

echo "vLLM servers starting — tail logs/vllm_*.log to monitor"
```

---

## 8. Run the full pipeline

```bash
# Full pipeline (all 21 steps, ~24h total wall time)
bash scripts/run_all.sh

# Or run individual steps — useful for resuming after failure:
bash scripts/run_all.sh --from-step 5    # resume from step 5

# Skip training (data collection + synthesis only, no GPU needed)
bash scripts/run_all.sh --skip-training

# Dry run — prints all commands without executing
DRY_RUN=1 bash scripts/run_all.sh
```

### Step-by-step breakdown

| Step | Command | Time | GPUs |
|------|---------|------|------|
| 1 | Environment check | <1 min | 0 |
| 2 | Autonomous source discovery | ~10 min | 0 |
| 3 | YouTube channel + search discovery | ~2h | 0 |
| 4 | API harvest (Sketchfab, Polyhaven, etc.) | ~30 min | 0 |
| 5 | Bulk transcript fetch (30 workers) | ~3h | 0 |
| 6 | Objaverse 3D dataset prep (50k objects) | ~1h | 0 |
| 7 | Launch vLLM synthesis servers | ~5 min (wait 60s) | 0-15 |
| 8 | Synthesize training pairs (Qwen2.5-72B) | ~8h | 0-15 |
| 9 | Form annotation VLM (Qwen2-VL-72B) | ~4h | 12-15 |
| 10 | Cross-software integration data | ~30 min | 0 |
| 11 | Agent training pair generation | ~1h | 0 |
| 12 | Quality validation + dedup | ~30 min | 0 |
| 13 | Blender execution validation (6 workers) | ~2h | 0 |
| 14 | Multi-turn + DPO pair generation | ~1h | 0 |
| 15 | Training data prep (all 5 streams) | ~15 min | 0 |
| 16 | Stage 1: SFT (ZeRO-3, 18 GPUs) | ~6h | 0-17 |
| 17 | Stage 2: Execution RL (GRPO) | ~4h | 0-17 |
| 18 | Stage 3: DPO | ~2h | 0-17 |
| 19 | NalanaBench evaluation | ~1h | 0-17 |
| 20 | Deploy (docker compose up) | ~5 min | 0 |
| 21 | Health check + smoke test | ~5 min | 0 |

---

## 9. Monitor training

```bash
# GPU utilization (run in a second terminal)
watch -n5 nvidia-smi

# Training loss (if W&B is configured)
# Visit https://wandb.ai/[your-username]/nalana

# DeepSpeed log
tail -f logs/train_sft.log

# Synthesis progress
tail -f logs/synthesize.log
```

---

## 10. After training completes

```bash
# Model is at:
ls checkpoints/nalana-final/

# Run evaluation
python3 evaluation/eval.py --model checkpoints/nalana-final

# Upload to HuggingFace (optional)
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='checkpoints/nalana-final',
    repo_id='[your-org]/nalana-v1',
    repo_type='model'
)
"

# Generate SSL DH parameters (one-time, ~30s)
openssl dhparam -out deploy/ssl/dhparam.pem 2048

# Deploy inference stack
cd deploy && docker compose up -d

# Test live
curl -X POST http://localhost:8080/v1/command \
  -H 'Content-Type: application/json' \
  -d '{"voice_command": "add a UV sphere with 32 segments", "scene_context": {"software": "blender"}}'
```

---

## Troubleshooting

### OOM during training
```bash
# Reduce micro batch size in training/ds_config.json:
# "train_micro_batch_size_per_gpu": 1  (default 2)
# Increase gradient accumulation to compensate:
# "gradient_accumulation_steps": 8
```

### vLLM fails to start
```bash
# Check logs
tail -100 logs/vllm_8001.log

# Common fix: model not downloaded
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-72B-Instruct')"
# This downloads ~140GB — run once, cached for future runs
```

### Blender headless fails
```bash
# Test directly
blender --background --python-expr "import bpy; print(bpy.app.version_string)"

# If Blender isn't in PATH
export BLENDER_PATH=/path/to/blender
# Add to .env permanently
```

### YouTube quota exhausted
```bash
# Discovery uses ~10,000 quota units/day on full run
# YouTube Data API v3 daily limit: 10,000 units (free), 1M units (paid)
# Run channel crawl first (cheapest: 1 unit per 50 videos)
python3 discovery/discover_v2.py --api-key $YOUTUBE_API_KEY --channels

# Then run searches on separate days to stay within quota
python3 discovery/discover_v2.py --api-key $YOUTUBE_API_KEY --search
python3 discovery/discover_v2.py --api-key $YOUTUBE_API_KEY --all-software
```
