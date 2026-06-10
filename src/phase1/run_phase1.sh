#!/bin/bash
# Phase 1 runner — designed to be launched DETACHED (setsid nohup) so it survives terminal close.
set -uo pipefail
cd /home/b3ali/projects/stamp26
PY=venvs/main/bin/python
LOG_PREFIX="[phase1 $(date +%H:%M:%S)]"
echo "$LOG_PREFIX starting"

rm -f artifacts/phase1.done artifacts/phase1.failed

$PY src/phase1/build_manifest.py || { echo "$LOG_PREFIX manifest FAILED"; touch artifacts/phase1.failed; exit 1; }
$PY src/phase1/extract_frames.py || { echo "$LOG_PREFIX extraction FAILED"; touch artifacts/phase1.failed; exit 1; }
$PY src/phase1/verify_phase1.py
grep -q PHASE1_VERIFY_PASS <(tail -5 artifacts/phase1.log 2>/dev/null) 2>/dev/null  # informational only

echo "$LOG_PREFIX done at $(date)"
touch artifacts/phase1.done
