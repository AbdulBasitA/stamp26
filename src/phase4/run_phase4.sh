#!/bin/bash
# Phase 4 runner (CPU-only). Detached-safe with markers.
set -uo pipefail
cd /home/b3ali/projects/stamp26
PY=venvs/main/bin/python
rm -f artifacts/phase4.done artifacts/phase4.failed
mkdir -p artifacts/phase4
log() { echo "[phase4 $(date +%H:%M:%S)] $*"; }

log "building maps + clustering"
$PY src/phase4/build_maps.py || { touch artifacts/phase4.failed; exit 1; }
log "rendering cluster cards"
$PY src/phase4/cluster_cards.py || { touch artifacts/phase4.failed; exit 1; }
log "verify"
$PY src/phase4/verify_phase4.py | tee artifacts/phase4/verify.log
if grep -q PHASE4_VERIFY_PASS artifacts/phase4/verify.log; then
  touch artifacts/phase4.done
  log "phase 4 complete"
else
  touch artifacts/phase4.failed
  exit 1
fi
