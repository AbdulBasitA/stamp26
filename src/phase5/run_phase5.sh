#!/bin/bash
# Phase 5 runner — namer TP=2 on GPUs 0-1 (embedder on GPU 2); falls back to TP=4 + CPU embedder.
# DETACHED-safe. Markers: artifacts/phase5.done / artifacts/phase5.failed
set -uo pipefail
cd /home/b3ali/projects/stamp26
export PATH="/home/b3ali/projects/stamp26/venvs/vllm/bin:$PATH"
VLLM=venvs/vllm/bin/vllm
PY=venvs/main/bin/python
mkdir -p artifacts/phase5 artifacts/naming
rm -f artifacts/phase5.done artifacts/phase5.failed
log() { echo "[phase5 $(date +%H:%M:%S)] $*"; }

kill_servers() {
  pkill -f "vllm serv[e]" 2>/dev/null
  for i in $(seq 1 36); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
    [ "$used" -lt 2000 ] && return 0
    sleep 5
  done
}

start_namer() { # tp_size gpus
  NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=$2 $VLLM serve Qwen/Qwen3.6-35B-A3B-FP8 --port 8000 \
    --served-model-name namer --tensor-parallel-size $1 --max-model-len 16384 \
    --gpu-memory-utilization 0.94 --language-model-only --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --enable-prefix-caching > artifacts/phase5/namer.log 2>&1 &
}

wait_ready() {
  for i in $(seq 1 $(($1 / 5))); do
    curl -s -o /dev/null http://localhost:8000/v1/models && return 0
    sleep 5
  done
  return 1
}

kill_servers
log "starting namer TP=2 on GPUs 0,1"
start_namer 2 0,1
EMBED=cuda
if ! wait_ready 900; then
  log "TP=2 failed to start — falling back to TP=4 + CPU embedder"
  kill_servers
  start_namer 4 0,1,2,3
  EMBED=cpu
  wait_ready 1200 || { log "namer failed entirely"; touch artifacts/phase5.failed; exit 1; }
fi

FAIL=0
for TRACK in captions siglip2; do
  log "naming track: $TRACK"
  if [ "$EMBED" = cuda ]; then
    CUDA_VISIBLE_DEVICES=2 EMBED_DEVICE=cuda $PY src/phase5/name_topics.py --track $TRACK \
      > artifacts/phase5/naming_$TRACK.log 2>&1 || { log "$TRACK naming FAILED"; FAIL=1; }
  else
    EMBED_DEVICE=cpu $PY src/phase5/name_topics.py --track $TRACK \
      > artifacts/phase5/naming_$TRACK.log 2>&1 || { log "$TRACK naming FAILED"; FAIL=1; }
  fi
  tail -3 artifacts/phase5/naming_$TRACK.log
done
kill_servers

log "verify"
$PY src/phase5/verify_phase5.py | tee artifacts/phase5/verify.log
if [ "$FAIL" -eq 0 ] && grep -q PHASE5_VERIFY_PASS artifacts/phase5/verify.log; then
  touch artifacts/phase5.done
  log "phase 5 complete"
else
  touch artifacts/phase5.failed
  exit 1
fi
