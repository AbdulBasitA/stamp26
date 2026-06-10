#!/bin/bash
# Phase 3 runner — windows, embeddings (parallel across GPUs), detections, caption embeddings, verify.
# Designed to run DETACHED. Markers: artifacts/phase3.done / artifacts/phase3.failed
set -uo pipefail
cd /home/b3ali/projects/stamp26
PY=venvs/main/bin/python
mkdir -p artifacts/phase3
rm -f artifacts/phase3.done artifacts/phase3.failed
log() { echo "[phase3 $(date +%H:%M:%S)] $*"; }

log "building windows"
$PY src/phase3/build_windows.py || { touch artifacts/phase3.failed; exit 1; }

log "launching GPU jobs (vjepa2 win/clip, siglip2 win/clip, detections)"
CUDA_VISIBLE_DEVICES=0 $PY src/phase3/embed_vjepa2.py --what windows > artifacts/phase3/vjepa2_windows.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 bash -c "$PY src/phase3/embed_vjepa2.py --what clips && $PY src/phase3/embed_siglip2.py --what clips" > artifacts/phase3/gpu1.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=2 $PY src/phase3/embed_siglip2.py --what windows > artifacts/phase3/siglip2_windows.log 2>&1 &
P2=$!
CUDA_VISIBLE_DEVICES=3 $PY src/phase3/detect_tokens.py > artifacts/phase3/detect.log 2>&1 &
P3=$!
FAIL=0
wait $P0 || { log "vjepa2 windows FAILED"; FAIL=1; }
wait $P1 || { log "gpu1 chain (vjepa2/siglip2 clips) FAILED"; FAIL=1; }
wait $P2 || { log "siglip2 windows FAILED"; FAIL=1; }
wait $P3 || { log "detections FAILED"; FAIL=1; }

log "caption embeddings (tags with complete sets)"
CUDA_VISIBLE_DEVICES=0 $PY src/phase3/embed_captions.py > artifacts/phase3/embed_captions.log 2>&1 \
  || { log "caption embeddings FAILED"; FAIL=1; }

log "verify"
$PY src/phase3/verify_phase3.py | tee artifacts/phase3/verify.log

if [ "$FAIL" -eq 0 ] && grep -q PHASE3_VERIFY_PASS artifacts/phase3/verify.log; then
  log "phase 3 complete"
  touch artifacts/phase3.done
else
  log "phase 3 had failures"
  touch artifacts/phase3.failed
  exit 1
fi
