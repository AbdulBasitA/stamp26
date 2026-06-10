#!/bin/bash
# Phase 0 GPU smoke tests, sequential. Logs to artifacts/smoke/.
cd /home/b3ali/projects/stamp26
mkdir -p artifacts/smoke
VLLM=venvs/vllm/bin/vllm
PY=venvs/main/bin/python
# venv bin on PATH so flashinfer JIT subprocesses find ninja
export PATH="/home/b3ali/projects/stamp26/venvs/vllm/bin:$PATH"

wait_ready() { # port, timeout_s
  for i in $(seq 1 $(($2 / 5))); do
    curl -s -o /dev/null "http://localhost:$1/v1/models" && return 0
    sleep 5
  done
  return 1
}
kill_server() {
  pkill -f "vllm serve" 2>/dev/null
  for i in $(seq 1 24); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
    [ "$used" -lt 2000 ] && return 0
    sleep 5
  done
}

echo "=== SMOKE 1: embedder load test (GPU0) ==="
CUDA_VISIBLE_DEVICES=0 $PY src/smoke/embed_load_test.py 2>&1 | grep -Ev "^(Down|Fetch|Load)" | tail -6

echo "=== SMOKE 2: captioner A — QuantTrio/Qwen3.5-9B-AWQ (GPU0) ==="
CUDA_VISIBLE_DEVICES=0 $VLLM serve QuantTrio/Qwen3.5-9B-AWQ --port 8001 \
  --max-model-len 32768 --gpu-memory-utilization 0.92 \
  --allowed-local-media-path /home/b3ali/projects/stamp26/data/nexar \
  --mm-processor-cache-type shm > artifacts/smoke/captioner_a.log 2>&1 &
if wait_ready 8001 900; then
  $PY src/smoke/probe_vlm.py 8001 2>&1 | tail -4
else
  echo "CAPTIONER_A_FAILED_TO_START (see artifacts/smoke/captioner_a.log)"
fi
kill_server

echo "=== SMOKE 3: captioner B — Qwen/Qwen3-VL-8B-Instruct-FP8 (GPU0) ==="
CUDA_VISIBLE_DEVICES=0 $VLLM serve Qwen/Qwen3-VL-8B-Instruct-FP8 --port 8001 \
  --max-model-len 32768 --gpu-memory-utilization 0.92 \
  --allowed-local-media-path /home/b3ali/projects/stamp26/data/nexar \
  --mm-processor-cache-type shm > artifacts/smoke/captioner_b.log 2>&1 &
if wait_ready 8001 900; then
  $PY src/smoke/probe_vlm.py 8001 2>&1 | tail -4
else
  echo "CAPTIONER_B_FAILED_TO_START (see artifacts/smoke/captioner_b.log)"
fi
kill_server

echo "=== SMOKE 4: namer TP=4 — Qwen/Qwen3.6-35B-A3B-FP8 + toy end-to-end ==="
NCCL_P2P_DISABLE=1 $VLLM serve Qwen/Qwen3.6-35B-A3B-FP8 --port 8000 \
  --served-model-name namer --tensor-parallel-size 4 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --language-model-only --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-prefix-caching > artifacts/smoke/namer.log 2>&1 &
if wait_ready 8000 1200; then
  echo "--- direct json_object + max_tokens=128 probe (Toponymy's exact call shape) ---"
  curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "namer", "max_tokens": 128, "response_format": {"type": "json_object"},
    "messages": [{"role": "user", "content": "Name this topic as JSON with keys topic_name and topic_specificity (0-1): clips of rear-end collisions in rain."}]
  }' | $PY -c "import sys,json; r=json.load(sys.stdin); print('namer probe:', r['choices'][0]['message']['content'][:200])"
  echo "--- toy end-to-end ---"
  $PY src/smoke/toy_e2e.py 2>&1 | tail -10
else
  echo "NAMER_FAILED_TO_START (see artifacts/smoke/namer.log)"
fi
kill_server
echo "ALL_SMOKES_DONE"
touch artifacts/smoke/.suite_done
