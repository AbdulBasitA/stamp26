# Local LLM for cluster NAMING + LLM-AS-JUDGE (4x RTX 4090, no NVLink) — research note

Date: 2026-06-09. Workload: ~50–200 naming calls (short JSON outputs over keyphrases/exemplar captions) + ~500–2000 short judge calls. Served via vLLM OpenAI-compatible API; Toponymy `OpenAINamer`/`AsyncOpenAINamer` points at it via `base_url`.

**Bottom line: a 70B dense model is NOT needed. Primary pick: `Qwen/Qwen3.6-35B-A3B-FP8` (Apr 2026, Apache-2.0, MoE 35B total / 3B active, official FP8), TP=4 (or TP=2) on vLLM 0.22.1 with thinking disabled server-side. Fallback: `Qwen/Qwen3-32B-FP8` (TP=2) or `Qwen/Qwen3-32B-AWQ` (single GPU).**

---

## 0. Toponymy integration constraints (read first — affects everything)

From `/home/b3ali/projects/stamp26/repos/toponymy/toponymy/llm_wrappers.py`:

- `OpenAINamer` / `AsyncOpenAINamer` accept `base_url` (and `api_key`, `model`) → point at vLLM directly. `AsyncOpenAINamer(max_concurrent_requests=10)` default.
- **Hard-coded `response_format={"type": "json_object"}`** in `_call_llm` and `_call_llm_with_system_prompt` (lines ~2693, ~2716). The vLLM server must support OpenAI JSON mode → vLLM structured outputs (xgrammar backend, on by default in 0.19+). Works with Qwen3.x. Riskier with gpt-oss (harmony channels + structured output interplay).
- **`max_tokens=128` default for topic-name generation** (line 414; cluster-name routine uses 1024). A thinking-mode model burns `<think>` tokens against `max_tokens` → empty/truncated JSON. **Thinking MUST be disabled for naming** (Qwen3.5/3.6 think by default).
- The wrapper does NOT pass `extra_body`, so per-request `chat_template_kwargs` is impossible from Toponymy → disable thinking **server-side** with `--default-chat-template-kwargs '{"enable_thinking": false}'` (verified vLLM flag; request-level `extra_body` can still re-enable it for our own judge scripts).
- Toponymy issue #155 (open, 2026-06-03): "Detect and surface malformed/empty LLM responses instead of silently returning bad output" — i.e., truncation/thinking failure modes are currently *silent*. Another reason to disable thinking and sanity-check outputs.
- No Toponymy issue/doc demands a large model. The `OpenAINamer` docstring is explicit: default `gpt-4o-mini` is "sufficiently powerful ... more advanced models ... have **diminishing returns for this task**". gpt-4o-mini is a small-model tier → a modern 30B-class local model is comfortably above the bar Toponymy was designed for.

## 1. Landscape as of June 2026 (things moved a lot since the plan was drafted)

The plan's candidate list (Qwen3-32B, Qwen3-30B-A3B-2507, Llama-3.3-70B, Qwen2.5-72B, GLM-4.5-Air) is now one to two generations old:

- **Qwen3.5** released 2026-02-16+: `Qwen/Qwen3.5-397B-A17B`, `Qwen3.5-122B-A10B`, `Qwen3.5-35B-A3B`, `Qwen3.5-27B` (+9B/4B/2B/0.8B). Native vision-language models, hybrid Gated-DeltaNet linear attention + sparse MoE, 262,144 native context (→1M YaRN), 201 languages. ([deeplearning.ai](https://www.deeplearning.ai/the-batch/qwen-announces-new-open-weights-flagship-update), HF cards)
- **Qwen3.6** released April 2026 ([github.com/QwenLM/Qwen3.6](https://github.com/QwenLM/Qwen3.6)): `Qwen/Qwen3.6-35B-A3B` (2026-04-16) and `Qwen/Qwen3.6-27B` dense (2026-04-22). Official FP8 repos exist: **`Qwen/Qwen3.6-35B-A3B-FP8`**, **`Qwen/Qwen3.6-27B-FP8`** ("fine-grained fp8 quantization with block size of 128"). Apache 2.0. vLLM >= 0.19.0 required (qwen3_5/3_6 arch not in older stables).
- **GLM-5 / GLM-5.1**: 744–745B MoE (40B active) — far too big for 96GB; GLM-4.6 = 357B too big; GLM-4.5-Air (106B-A12B, AWQ ~60GB) fits TP=4 but is now clearly outclassed by Qwen3.6-35B at a fraction of the footprint. Drop.
- **DeepSeek V4 / Kimi K2.6**: 1T+ class, irrelevant locally.
- **vLLM current stable: 0.22.1 (released 2026-06-05)**; 0.20.0 Apr 27, 0.21.0 May 15, 0.22.0 May 29 (PyPI).

### Qwen3.6-35B-A3B quality datapoints (HF model card)
SWE-bench Verified 73.4 (Qwen3.5-35B-A3B: 70.0), GPQA 86.0 (84.2), MMLU-Pro 85.2. This is *well* above Llama-3.3-70B / Qwen2.5-72B territory while activating only 3B params/token → fast even over PCIe TP. For a task whose ceiling is "gpt-4o-mini is fine", it is overkill in the right direction.

## 2. Candidate table

| Model (HF ID) | Type | Quant | Weights VRAM | Fits | vLLM | Notes |
|---|---|---|---|---|---|---|
| **`Qwen/Qwen3.6-35B-A3B-FP8`** (primary) | MoE 35B/A3B, hybrid GDN, native VLM | official FP8 (block-128) | ~36 GB | TP=2 (tight) or TP=4 (comfy) | >=0.19; use 0.22.1 | 262K ctx; thinking default ON → disable; `--language-model-only` skips vision tower; A3B → high tok/s |
| `Qwen/Qwen3.6-27B-FP8` | dense 27B | official FP8 | ~28 GB | TP=2 | >=0.19 | dense alternative if MoE+GDN misbehaves on Ada |
| `Qwen/Qwen3.5-122B-A10B` | MoE 122B/A10B | FP8 = ~122 GB **does not fit**; community 4-bit (~62 GB) only | ~62 GB @4bit | TP=4 only | >=0.19 | highest quality that could fit; AWQ/GPTQ support for hybrid-GDN MoE is community-grade → risky for a 2–3 day workshop. Skip unless curious. |
| **`Qwen/Qwen3-32B-FP8`** (fallback) | dense 32B, hybrid thinking | official FP8 | ~33 GB | TP=2 | any 2025+ stable | battle-tested, boring, works. `enable_thinking` toggle same as 3.6 |
| `Qwen/Qwen3-32B-AWQ` (fallback-min) | dense 32B | official AWQ 4-bit | ~18 GB | **1 GPU** | any | single-4090 option; frees GPUs 1–3 for embeddings/judge |
| `Qwen/Qwen3-30B-A3B-Instruct-2507(-FP8)` | MoE, non-thinking by design | official FP8 | ~31 GB | TP=2 | any | fine, but strictly dominated by Qwen3.6-35B-A3B now |
| `openai/gpt-oss-120b` | MoE 117B/A5.1B, MXFP4 | native MXFP4 | ~63 GB | TP=4 | recipe still says Ada support "actively working" | harmony format + Toponymy's hard-coded `json_object` = risk; reasoning-always-on eats max_tokens=128. **Not primary.** |
| `openai/gpt-oss-20b` | MoE 21B/A3.6B | native MXFP4 | ~13 GB | 1 GPU | same caveat | useful as a *different-family secondary judge* (bias control), called from our own scripts where we control max_tokens |
| `meta-llama/Llama-3.3-70B-Instruct` AWQ / `Qwen/Qwen2.5-72B-Instruct-AWQ` | dense 70B+ | AWQ ~38–40 GB | TP=4, all GPUs | any | Dec-2024-era quality, slower per token on PCIe TP=4, monopolizes the box. **Obsolete — drop from plan.** |
| GLM-4.5-Air AWQ / GLM-4.6 / GLM-5 | MoE | community AWQ / none | 60 GB / >170 GB | TP=4 / no | — | superseded or doesn't fit. Drop. |

FP8 on Ada (compute 8.9): vLLM FP8 W8A8 is supported natively on CC >= 8.9 (Ada, Hopper); on < 8.9 it falls back to weight-only W8A16 via Marlin ([vLLM FP8 docs](https://docs.vllm.ai/en/latest/features/quantization/fp8/)). 4090 = 8.9 → native path; worst case for the block-128 fine-grained scheme is the Marlin W8A16 fallback, which is correct just slower. A real-world guide ([pentagi vllm-qwen35-27b-fp8.md](https://github.com/vxcontrol/pentagi/blob/main/examples/guides/vllm-qwen35-27b-fp8.md)) serves `Qwen/Qwen3.5-27B-FP8` with `NCCL_P2P_DISABLE=1 ... --tensor-parallel-size 4 --block-size 128 --reasoning-parser qwen3` on 4x consumer GPUs (5090s), ~650 tok/s generation.

## 3. Is 70B needed for naming + judging? No.

- **Toponymy's own position**: default model gpt-4o-mini; docstring says bigger models have "diminishing returns for this task". Naming input = keyphrases + exemplar captions + (our) metadata histograms; output = a short JSON topic name. This is a summarization/labeling task, not reasoning.
- **Judge literature**: "Judge's Verdict" ([arXiv:2510.09738](https://arxiv.org/abs/2510.09738), 54 models incl. 43 open 1B–405B) finds judge–human agreement is **not monotone in model size**; training/design dominates, and Qwen3-30B-class models already show human-like judgment patterns. "An Empirical Study of LLM-as-a-Judge" ([arXiv:2506.13639](https://arxiv.org/abs/2506.13639)) likewise finds prompt/rubric design choices impact reliability more than scale.
- **Our judge task is easy-mode**: binary/Likert "does this cluster name describe this held-out member's caption?" — grounded, verifiable, short context. Rerun-stability (same judge, temperature 0, prompt-order shuffles) matters more than parameter count.
- **Where size could matter**: NHTSA pre-crash-typology alignment requires taxonomy knowledge in the prompt anyway (we inject the typology into the Jinja template), so model world-knowledge is not the bottleneck.
- **Bias control instead of size**: captions will likely come from a Qwen-family VLM and the namer is Qwen → a Qwen judge has family/self-preference risk. Cheap mitigation: run `openai/gpt-oss-20b` on one spare 4090 as a *second* judge for an agreement check on a subsample (~200 calls). Disagreement rate is itself a trustworthiness statistic for the writeup.

## 4. TP on 4x 4090 without P2P/NVLink — what actually happens

- Consumer 4090s have driver-disabled P2P; NCCL transparently falls back to shared-memory/host transport. vLLM detects missing P2P and disables its custom all-reduce automatically. Common practice (and used in the pentagi guide) is to export `NCCL_P2P_DISABLE=1` explicitly to avoid NCCL probing hangs/`invalid usage` errors ([vllm#7201](https://github.com/vllm-project/vllm/issues/7201)).
- Throughput: TP over PCIe is sublinear but positive. Databasemart benchmarks: 2x4090 TP=2 = 5,479 tok/s vs 3,965 tok/s on 1 GPU for a 7B model (+38%); their rule: "Always use --tensor-parallel-size=N on multi-GPU boxes" ([guide](https://www.databasemart.com/blog/vllm-distributed-inference-optimization-guide)). 4-GPU TP=4 on consumer cards serving a 27B FP8: ~650 tok/s generation (pentagi). The geohot/tinygrad P2P driver patch gives +10–30% but is NOT worth the risk on a workshop box ([smcleod.net writeup](https://smcleod.net/2026/02/patching-nvidias-driver-and-vllm-to-enable-p2p-on-consumer-gpus/)).
- Our volume: 200 naming calls × ~300 out-tokens + 2000 judge calls × ~100 out-tokens ≈ 260K output tokens → **minutes**, not hours, at even 100 tok/s effective. TP efficiency is a non-issue; pick TP for *fit and headroom*, not speed.

## 5. Operational design / serving schedule

**Facts**: one vLLM server = one model (no multi-model hot-swap). vLLM *does* have sleep mode (`--enable-sleep-mode` + `VLLM_SERVER_DEV_MODE=1`; `POST /sleep?level=2` frees ~90%+ VRAM, `POST /wake_up`, `GET /is_sleeping`) ([docs](https://docs.vllm.ai/en/latest/features/sleep_mode/)) — usable to park the namer while embedding jobs need VRAM, but for a 2–3 day project **sequential phases are simpler and safer**.

**Recommended schedule** (phases are naturally sequential in the pipeline anyway):

- **Phase A — captioning**: VLM captioner owns all 4 GPUs (data-parallel: 4 single-GPU vLLM instances on `CUDA_VISIBLE_DEVICES=0..3`, or one `-dp 4` server). 1,500 clips + sub-windows. Kill it when captions are on disk (captions are cached artifacts; never re-run).
- **Phase B — embeddings/UMAP/HDBSCAN**: GPUs mostly free; no LLM server needed.
- **Phase C — naming + judging**: launch the namer on all 4 GPUs, TP=4 (GPUs are idle now; TP=4 gives huge KV headroom and best latency):

```bash
# Primary namer/judge — Phase C, all four GPUs
export NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
  --port 8000 \
  --served-model-name namer \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-prefix-caching
```

- 32K context is generous for naming prompts (keyphrases + ~20 exemplar captions + metadata histogram ≈ 3–6K tokens); raise to 65536 if cluster-layer prompts get fat — fits easily at TP=4.
- `--language-model-only` skips the (unneeded) vision tower of Qwen3.6 → less VRAM.
- `--enable-prefix-caching`: judge calls share a long rubric prefix → big win.

**Co-existence variant** (if you want the namer up while GPUs 2–3 do embedding work): same command with `CUDA_VISIBLE_DEVICES=0,1 --tensor-parallel-size 2 --max-model-len 16384 --gpu-memory-utilization 0.94`. ~18 GB/GPU weights + small GDN cache; tight but workable. If it OOMs, drop to the single-GPU fallback below.

```bash
# Fallback A: Qwen3-32B FP8 on two GPUs (older stable vLLM also fine)
CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 vllm serve Qwen/Qwen3-32B-FP8 \
  --port 8000 --served-model-name namer --tensor-parallel-size 2 \
  --max-model-len 16384 --gpu-memory-utilization 0.93 \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}'

# Fallback B: single-GPU AWQ (frees 3 GPUs entirely)
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-32B-AWQ \
  --port 8000 --served-model-name namer \
  --max-model-len 8192 --gpu-memory-utilization 0.95 \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}'

# Optional: second-family judge for bias check, one spare GPU
CUDA_VISIBLE_DEVICES=3 vllm serve openai/gpt-oss-20b --port 8001 --served-model-name judge2
```

**Toponymy hookup**:
```python
from toponymy.llm_wrappers import AsyncOpenAINamer
namer = AsyncOpenAINamer(
    api_key="EMPTY",
    model="namer",                       # matches --served-model-name
    base_url="http://localhost:8000/v1",
    max_concurrent_requests=16,
)
```

**Judge calls** (our own scripts, plain `openai` client at the same `base_url`): temperature 0 (or 0.7/top_p 0.8 per Qwen3.6 non-thinking recommendation if sampling for stability estimates), `max_tokens=256`, optionally per-request thinking:
```python
client.chat.completions.create(
    model="namer", messages=msgs, max_tokens=2048, temperature=1.0, top_p=0.95,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},  # overrides server default
)
```

## 6. Thinking mode: use it or not?

- **Naming via Toponymy: NO.** `max_tokens=128` + hard-coded JSON mode makes thinking actively harmful (truncated/empty JSON, silently — issue #155). Disable server-side as above. Qwen3.6 non-thinking recommended sampling: temperature 0.7, top_p 0.8.
- **Judging: optional, per-request.** Latency is irrelevant (~2000 short calls). Qwen3.6 thinking sampling: temperature 1.0, top_p 0.95; budget max_tokens >= 2048. With `--reasoning-parser qwen3`, reasoning lands in `message.reasoning_content` and `message.content` stays clean; vLLM applies structured-output grammar after the think block. Suggested experiment: run judge both ways on 100 items; if agreement > 95%, use non-thinking for the full run (cleaner, cheaper, more reproducible at temp 0).
- Toggle mechanics (verified): server default `--default-chat-template-kwargs '{"enable_thinking": false}'`; per-request override `extra_body={"chat_template_kwargs": {"enable_thinking": true/false}}`; request-level wins ([vLLM reasoning docs](https://docs.vllm.ai/en/latest/features/reasoning_outputs/), [Qwen vLLM docs](https://qwen.readthedocs.io/en/latest/deployment/vllm.html)).

## 7. Version pins & pre-downloads (air-gap prep)

- `vllm==0.22.1` (2026-06-05; Qwen3.6 needs >=0.19.0). Python 3.12 OK (vllm supports 3.10–3.14). Keep the LLM-serving venv separate from the pipeline venv (`uv venv`); they only talk over HTTP.
- Pre-download before the workshop (`hf download <id>`): `Qwen/Qwen3.6-35B-A3B-FP8` (~37 GB), `Qwen/Qwen3-32B-FP8` (~33 GB), `Qwen/Qwen3-32B-AWQ` (~19 GB), optional `openai/gpt-oss-20b` (~13 GB). Total < 110 GB of the 1.7 TB.
- Smoke test on arrival: one naming-shaped request with `response_format={"type":"json_object"}` and `max_tokens=128` through the *actual* Toponymy wrapper, not curl.

## 8. Risks / contradictions with the project plan

1. **Plan's candidate list is stale**: Llama-3.3-70B-AWQ and Qwen2.5-72B-AWQ (TP=4) are dominated on quality AND footprint by Qwen3.6-35B-A3B-FP8; GLM-4.6/GLM-5 don't fit 96 GB; GLM-4.5-Air superseded. gpt-oss-120b fits (~63 GB MXFP4, TP=4) but the official vLLM recipe still lists Ada support as in-progress and harmony-format + Toponymy's hard-coded `json_object`/`max_tokens=128` make it a poor namer here.
2. **Toponymy wrapper constraints** (json_object + max_tokens=128 + no extra_body) force server-side thinking disable; a thinking-enabled server default would fail *silently* (issue #155).
3. Qwen3.6 = new hybrid GDN+MoE arch: needs vLLM >= 0.19; if the workshop box must run an older vLLM for the captioning VLM, run two venvs or fall back to Qwen3-32B-FP8 (supported since mid-2025 stables).
4. FP8 block-128 on Ada: native CC-8.9 path expected; if a kernel gap appears in 0.22.x, the Marlin W8A16 fallback or `Qwen/Qwen3-32B-AWQ` covers it.
5. Judge family bias: Qwen judging Qwen-captioned/Qwen-named output; mitigate with grounded rubric + optional gpt-oss-20b second judge.
6. Qwen3.5/3.6 are *native VLMs* — potential simplification: the captioning agent should evaluate whether one Qwen3.6-35B-A3B server (without `--language-model-only`) can do BOTH captioning and naming, collapsing two phases into one model. Not load-bearing for this note's recommendation.

## Sources

- [QwenLM/Qwen3.6 GitHub](https://github.com/QwenLM/Qwen3.6) · [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) · [Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8) · [Qwen/Qwen3.6-27B-FP8](https://huggingface.co/Qwen/Qwen3.6-27B-FP8) · [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- [vLLM Qwen3.5/3.6 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html) · [vLLM GPT-OSS recipe](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html) · [vLLM FP8 docs](https://docs.vllm.ai/en/latest/features/quantization/fp8/) · [vLLM sleep mode](https://docs.vllm.ai/en/latest/features/sleep_mode/) · [vLLM reasoning outputs](https://docs.vllm.ai/en/latest/features/reasoning_outputs/) · [vllm on PyPI](https://pypi.org/project/vllm/)
- [pentagi Qwen3.5-27B-FP8 multi-GPU guide](https://github.com/vxcontrol/pentagi/blob/main/examples/guides/vllm-qwen35-27b-fp8.md) · [databasemart multi-GPU vLLM benchmarks](https://www.databasemart.com/blog/vllm-distributed-inference-optimization-guide) · [smcleod P2P driver patching](https://smcleod.net/2026/02/patching-nvidias-driver-and-vllm-to-enable-p2p-on-consumer-gpus/) · [vllm#7201 NCCL TP error](https://github.com/vllm-project/vllm/issues/7201)
- [Judge's Verdict, arXiv:2510.09738](https://arxiv.org/abs/2510.09738) · [LLM-as-a-Judge design choices, arXiv:2506.13639](https://arxiv.org/abs/2506.13639) · [openai/gpt-oss-120b card](https://huggingface.co/openai/gpt-oss-120b) · [deeplearning.ai on Qwen3.5](https://www.deeplearning.ai/the-batch/qwen-announces-new-open-weights-flagship-update)
- Local: `/home/b3ali/projects/stamp26/repos/toponymy/toponymy/llm_wrappers.py` (OpenAINamer ~line 2610; max_tokens defaults lines 414/471); `gh issue list -R TutteInstitute/toponymy` (issues #155, #96, #56)
