#!/bin/bash
# Phase 2 runner — captions everything with BOTH captioner candidates (D1 bake-off).
# Designed to run DETACHED. Markers: artifacts/phase2.done / artifacts/phase2.failed
set -uo pipefail
cd /home/b3ali/projects/stamp26
export PATH="/home/b3ali/projects/stamp26/venvs/vllm/bin:$PATH"
VLLM=venvs/vllm/bin/vllm
PY=venvs/main/bin/python
mkdir -p artifacts/phase2 artifacts/captions
rm -f artifacts/phase2.done artifacts/phase2.failed

MODEL_A="QuantTrio/Qwen3.5-9B-AWQ"
MODEL_B="Qwen/Qwen3-VL-8B-Instruct-FP8"

log() { echo "[phase2 $(date +%H:%M:%S)] $*"; }

wait_ready_all() { # timeout_s
  for i in $(seq 1 $(($1 / 5))); do
    ok=0
    for p in 8001 8002 8003 8004; do
      curl -s -o /dev/null "http://localhost:$p/v1/models" && ok=$((ok+1))
    done
    [ "$ok" -eq 4 ] && return 0
    sleep 5
  done
  return 1
}

kill_servers() {
  pkill -f "vllm serv[e]" 2>/dev/null
  for i in $(seq 1 36); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
    [ "$used" -lt 2000 ] && return 0
    sleep 5
  done
  log "WARN: GPUs did not free after kill"
}

start_workers() { # model tag
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i $VLLM serve "$1" --port 800$((i+1)) \
      --max-model-len 32768 --gpu-memory-utilization 0.92 \
      --allowed-local-media-path /home/b3ali/projects/stamp26 \
      --mm-processor-cache-type shm > "artifacts/phase2/server_$2_gpu$i.log" 2>&1 &
  done
}

log "cutting event-window mp4s"
$PY src/phase2/cut_event_windows.py || { touch artifacts/phase2.failed; exit 1; }

for TAG in a b; do
  MODEL=$([ "$TAG" = a ] && echo "$MODEL_A" || echo "$MODEL_B")
  log "=== captioner $TAG: $MODEL ==="
  kill_servers
  start_workers "$MODEL" "$TAG"
  if ! wait_ready_all 1200; then
    log "captioner $TAG: workers failed to start — skipping this model"
    kill_servers
    continue
  fi
  if ! $PY src/phase2/caption_clips.py --tag $TAG --targets clips --pilot 20; then
    log "captioner $TAG: PILOT FAILED — skipping this model"
    kill_servers
    continue
  fi
  $PY src/phase2/caption_clips.py --tag $TAG --targets clips || log "captioner $TAG: clips run below gate"
  $PY src/phase2/caption_clips.py --tag $TAG --targets windows || log "captioner $TAG: windows run below gate"
  kill_servers
done

# success = at least one model produced full caption sets above gate
$PY - <<'EOF'
import json, sys
from pathlib import Path
ok_models = []
for tag in ["a", "b"]:
    f = Path(f"artifacts/captions/clips_{tag}.jsonl")
    if not f.exists():
        continue
    recs = [json.loads(l) for l in f.open()]
    done = {r["clip_id"] for r in recs if r.get("ok")}
    if len(done) >= 1485:  # 99% of 1500
        ok_models.append(tag)
print(f"complete caption sets: {ok_models}")
sys.exit(0 if ok_models else 1)
EOF
if [ $? -eq 0 ]; then
  log "phase 2 complete"
  touch artifacts/phase2.done
else
  log "phase 2 FAILED — no complete caption set"
  touch artifacts/phase2.failed
  exit 1
fi
