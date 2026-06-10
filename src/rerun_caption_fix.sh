#!/bin/bash
# Re-run chain after the enum-artifact schema fix (v2): re-caption (model A only) ->
# snow gate -> re-embed captions -> rebuild captions-track maps -> re-name both tracks.
# DETACHED. Markers: artifacts/rerun.done / artifacts/rerun.failed
set -uo pipefail
cd /home/b3ali/projects/stamp26
export PATH="/home/b3ali/projects/stamp26/venvs/vllm/bin:$PATH"
VLLM=venvs/vllm/bin/vllm
PY=venvs/main/bin/python
rm -f artifacts/rerun.done artifacts/rerun.failed
log() { echo "[rerun $(date +%H:%M:%S)] $*"; }

kill_servers() {
  pkill -f "vllm serv[e]" 2>/dev/null
  for i in $(seq 1 36); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
    [ "$used" -lt 2000 ] && return 0
    sleep 5
  done
}

log "archiving v1 (enum-bug) captions"
mkdir -p artifacts/captions/enumbug
for f in clips_a windows_a; do
  [ -f artifacts/captions/$f.jsonl ] && mv artifacts/captions/$f.jsonl artifacts/captions/enumbug/$f.jsonl
done

log "starting 4 captioner workers (model A)"
kill_servers
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i $VLLM serve QuantTrio/Qwen3.5-9B-AWQ --port 800$((i+1)) \
    --max-model-len 32768 --gpu-memory-utilization 0.92 \
    --allowed-local-media-path /home/b3ali/projects/stamp26 \
    --mm-processor-cache-type shm > "artifacts/phase2/server_v2_gpu$i.log" 2>&1 &
done
ok=0
for i in $(seq 1 240); do
  n=0
  for p in 8001 8002 8003 8004; do curl -s -o /dev/null "http://localhost:$p/v1/models" && n=$((n+1)); done
  [ "$n" -eq 4 ] && { ok=1; break; }
  sleep 5
done
[ "$ok" -eq 1 ] || { log "workers failed"; touch artifacts/rerun.failed; exit 1; }

$PY src/phase2/caption_clips.py --tag a --targets clips --pilot 20 || { log "pilot FAILED"; touch artifacts/rerun.failed; kill_servers; exit 1; }
$PY src/phase2/caption_clips.py --tag a --targets clips || { log "clips below gate"; touch artifacts/rerun.failed; kill_servers; exit 1; }
$PY src/phase2/caption_clips.py --tag a --targets windows || log "windows below gate (continuing)"
kill_servers

log "snow gate: corrupted-weather rate must collapse"
$PY - <<'EOF'
import json, sys
n = snow = sunny = 0
for l in open("artifacts/captions/clips_a.jsonl"):
    r = json.loads(l)
    if r.get("ok"):
        n += 1
        snow += r.get("weather") == "snow"
        sunny += r.get("weather") == "sunny"
print(f"v2 captions: {n} ok, weather=snow {snow} ({snow/n:.1%}), weather=sunny {sunny} ({sunny/n:.1%})")
sys.exit(0 if snow <= n * 0.02 else 1)
EOF
[ $? -eq 0 ] || { log "SNOW GATE FAILED"; touch artifacts/rerun.failed; exit 1; }

log "re-embedding captions"
CUDA_VISIBLE_DEVICES=0 $PY src/phase3/embed_captions.py > artifacts/phase3/embed_captions_v2.log 2>&1 \
  || { touch artifacts/rerun.failed; exit 1; }

log "rebuilding captions-track maps (other tracks reuse saved UMAPs)"
rm -f artifacts/maps/captions__5d.npy artifacts/maps/captions__2d.npy
bash src/phase4/run_phase4.sh || { touch artifacts/rerun.failed; exit 1; }

log "re-naming both tracks"
bash src/phase5/run_phase5.sh || { touch artifacts/rerun.failed; exit 1; }

log "final check: snowy names"
grep -ci "snow" <(grep -hE '^\s+\[' artifacts/phase5/naming_captions.log artifacts/phase5/naming_siglip2.log) \
  | xargs -I{} echo "names mentioning snow: {}"
touch artifacts/rerun.done
log "rerun chain complete"
