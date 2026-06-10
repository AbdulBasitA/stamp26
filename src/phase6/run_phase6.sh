#!/bin/bash
# Phase 6 runner: neutral embeddings (GPU) -> namer up -> [judge + stability reruns] while
# eval (a) runs on CPU in parallel -> summarize. DETACHED. Markers: phase6.done/.failed
set -uo pipefail
cd /home/b3ali/projects/stamp26
export PATH="/home/b3ali/projects/stamp26/venvs/vllm/bin:$PATH"
VLLM=venvs/vllm/bin/vllm
PY=venvs/main/bin/python
mkdir -p artifacts/eval artifacts/phase6
rm -f artifacts/phase6.done artifacts/phase6.failed
log() { echo "[phase6 $(date +%H:%M:%S)] $*"; }

kill_servers() {
  pkill -f "vllm serv[e]" 2>/dev/null
  for i in $(seq 1 36); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
    [ "$used" -lt 2000 ] && return 0
    sleep 5
  done
}

log "neutral caption embeddings (GPU)"
kill_servers
CUDA_VISIBLE_DEVICES=0 $PY src/phase6/embed_captions_neutral.py > artifacts/phase6/neutral.log 2>&1 \
  || { log "neutral embed FAILED"; touch artifacts/phase6.failed; exit 1; }

log "launching eval (a) on CPU (clip then window) in background"
( $PY src/phase6/outlier_eval.py --granularity clip --workers 10 > artifacts/phase6/outlier_clip.log 2>&1 \
  && $PY src/phase6/outlier_eval.py --granularity window --workers 8 > artifacts/phase6/outlier_window.log 2>&1 \
  && touch artifacts/phase6/.eval_a_done || touch artifacts/phase6/.eval_a_failed ) &
EVAL_A_PID=$!

log "starting namer TP=4"
NCCL_P2P_DISABLE=1 $VLLM serve Qwen/Qwen3.6-35B-A3B-FP8 --port 8000 \
  --served-model-name namer --tensor-parallel-size 4 --max-model-len 16384 \
  --gpu-memory-utilization 0.90 --language-model-only --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-prefix-caching > artifacts/phase6/namer.log 2>&1 &
ready=0
for i in $(seq 1 240); do
  curl -s -o /dev/null http://localhost:8000/v1/models && { ready=1; break; }
  sleep 5
done
[ "$ready" -eq 1 ] || { log "namer failed"; touch artifacts/phase6.failed; exit 1; }

log "judge (held-out members + metadata consistency)"
$PY src/phase6/judge_names.py > artifacts/phase6/judge.log 2>&1 || log "judge FAILED (continuing)"
tail -4 artifacts/phase6/judge.log

log "stability reruns (2 extra namings per track)"
for TRACK in captions siglip2; do
  for SUF in _run2 _run3; do
    EMBED_DEVICE=cpu $PY src/phase5/name_topics.py --track $TRACK --suffix $SUF \
      > artifacts/phase6/naming_${TRACK}${SUF}.log 2>&1 || log "rerun ${TRACK}${SUF} FAILED"
  done
done
kill_servers

$PY src/phase6/stability.py > artifacts/phase6/stability.log 2>&1 || log "stability FAILED"
tail -8 artifacts/phase6/stability.log

log "waiting for eval (a)"
wait $EVAL_A_PID 2>/dev/null
[ -f artifacts/phase6/.eval_a_done ] || { log "eval (a) FAILED"; touch artifacts/phase6.failed; exit 1; }

log "summarize + verify"
$PY src/phase6/summarize_eval.py | tee artifacts/phase6/verify.log
if grep -q PHASE6_VERIFY_PASS artifacts/phase6/verify.log; then
  touch artifacts/phase6.done
  log "phase 6 complete"
else
  touch artifacts/phase6.failed
  exit 1
fi
