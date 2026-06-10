#!/bin/bash
# Overnight chain: toy-e2e retry (closes Phase 0) -> Phase 2 (caption bake-off) -> Phase 3 (embeddings).
# Run DETACHED: setsid nohup bash src/run_overnight.sh > artifacts/overnight.log 2>&1 < /dev/null &
set -uo pipefail
cd /home/b3ali/projects/stamp26
export PATH="/home/b3ali/projects/stamp26/venvs/vllm/bin:$PATH"
VLLM=venvs/vllm/bin/vllm
PY=venvs/main/bin/python
rm -f artifacts/overnight.done artifacts/overnight.failed
log() { echo "[overnight $(date +%H:%M:%S)] $*"; }

kill_servers() {
  pkill -f "vllm serv[e]" 2>/dev/null
  for i in $(seq 1 36); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
    [ "$used" -lt 2000 ] && return 0
    sleep 5
  done
}

log "STEP 0: toy e2e retry (namer TP=4, CPU text embedder)"
kill_servers
NCCL_P2P_DISABLE=1 $VLLM serve Qwen/Qwen3.6-35B-A3B-FP8 --port 8000 \
  --served-model-name namer --tensor-parallel-size 4 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --language-model-only --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-prefix-caching > artifacts/smoke/namer_retry.log 2>&1 &
ready=0
for i in $(seq 1 240); do
  curl -s -o /dev/null http://localhost:8000/v1/models && { ready=1; break; }
  sleep 5
done
if [ "$ready" -eq 1 ]; then
  $PY src/smoke/toy_e2e.py > artifacts/smoke/toy_e2e_retry.log 2>&1 \
    && log "toy e2e: PASS" || log "toy e2e: FAIL (see artifacts/smoke/toy_e2e_retry.log)"
else
  log "namer did not come up for toy retry — skipping"
fi
kill_servers

log "STEP 1: Phase 2 (captioning bake-off)"
bash src/phase2/run_phase2.sh || log "phase 2 reported failure — continuing to phase 3 (V/T tracks do not need captions)"

log "STEP 2: Phase 3 (embeddings + detections)"
bash src/phase3/run_phase3.sh || true

if [ -f artifacts/phase2.done ] && [ -f artifacts/phase3.done ]; then
  log "overnight chain complete"
  touch artifacts/overnight.done
else
  log "overnight chain finished WITH FAILURES (phase2: $([ -f artifacts/phase2.done ] && echo ok || echo FAIL), phase3: $([ -f artifacts/phase3.done ] && echo ok || echo FAIL))"
  touch artifacts/overnight.failed
fi
