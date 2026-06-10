# Local VLM for structured captioning of Nexar dashcam clips (4x RTX 4090, vLLM)

Research date: 2026-06-09. Target: caption ~1,500 clips (~40 s, 1280x720@30fps) + event-centered sub-windows, one model instance per GPU, 4 data-parallel vLLM workers, fully local / air-gap-safe, structured JSON output.

**Bottom line: Qwen2.5-VL-7B (the prior plan) is two generations out of date.** The current best fit is the Qwen3.5 unified-multimodal generation (Feb/Mar 2026), with Qwen3-VL-8B (Oct 2025) as the battle-tested fallback. Both have first-class video support in vLLM (current stable vLLM 0.22.1, released 2026-06-05).

---

## 1. Qwen family status (June 2026)

### Qwen3-VL (Sep–Nov 2025) — dedicated VLM line
- Sizes: dense **2B / 4B / 8B / 32B**, MoE **30B-A3B / 235B-A22B**. Every size in both **Instruct** and **Thinking** editions (shared tokenizer/weights base; deployable side by side). Tech report: arXiv:2511.21631.
- Native **256K context (expandable to 1M)**, explicit video features: timestamp-grounded event localization ("Text–Timestamp Alignment"), hours-long video.
- Official quantized checkpoints: **FP8 and AWQ for 2B/4B/8B/32B/30B-A3B/235B** (e.g. `Qwen/Qwen3-VL-8B-Instruct-FP8`). License Apache-2.0.
- VRAM (8B): FP8 weights ~8 GB, ~9.2–9.6 GB at runtime → very comfortable on 24 GB with large KV headroom for video tokens. BF16 ~17–18 GB → workable but tight with video.
- Video-MME (w/o subs): **Qwen3-VL-8B-Instruct ≈ 71.4** (Thinking ≈ −0.6 on this benchmark).
- Official recommended sampling for Instruct (important — counters known repetition loops): `temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, repetition_penalty=1.0`.
- qwen-vl-utils video knobs: `fps` (default 2), `num_frames` (overrides fps), `max_pixels` per frame, `total_pixels` budget — recommended `total_pixels ≤ 24576 × 32 × 32` ⇒ ~24.5 K vision tokens per video (1 token = 32×32 px after patch-16 + 2×2 merge).
- Cookbook: https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/video_understanding.ipynb

### Qwen3.5 (Feb 16 – Mar 2, 2026) — NEWER than Qwen3-VL, natively multimodal incl. VIDEO
- The user asked whether something newer than Qwen3-VL exists: **yes.** Qwen3.5 is an early-fusion, natively multimodal series ("Towards Native Multimodal Agents"); Qwen states it **outperforms Qwen3-VL across visual-understanding benchmarks**. There is **no separate "Qwen3.5-VL"** — vision+video is built into the main models.
- Open-weight sizes: **0.8B / 2B / 4B / 9B (dense)**, **27B (dense)**, **35B-A3B / 122B-A10B (MoE)**, plus 397B-A17B. Apache-2.0.
- **`Qwen/Qwen3.5-9B`**: 9–10 B params incl. vision encoder, image+video+text input, native 262,144-token context (YaRN → 1.01 M), hybrid gated-delta-net (linear-attention) layers ⇒ much smaller KV cache — helpful for long video token streams.
- Reported **Video-MME (with subs) = 84.5 for 9B, 83.5 for 4B** (vendor numbers; not directly comparable to Qwen3-VL's 71.4 w/o-subs figure, but the generational gain is consistent across benches).
- Thinking: hybrid. For the small models reasoning is **off by default**; toggle explicitly via `chat_template_kwargs: {"enable_thinking": false}` (do this for the caption pass — 1,500–6,000 calls). Sampling (non-thinking): `temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5`.
- Official quantizations exist for 9B/27B/35B-A3B: **FP8 (fine-grained, block-128; attention + shared expert kept 16-bit)** and INT4; community AWQ (`QuantTrio/Qwen3.5-9B-AWQ`, cyankiwi) and Intel AutoRound INT4. Kaitchup evals: FP8 ≈ identical to BF16.
- vLLM: day-1 support; dedicated recipe page (https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html). Default video sampling fps=2 with dynamic sampling; override via `mm_processor_kwargs`.
- 24 GB fit: BF16 ≈ 20 GB weights → too tight for video serving. **Use `Qwen/Qwen3.5-9B-FP8`** (≈10–11 GB) or AWQ INT4.

### Qwen3.6 (Apr 2026)
- Only **27B dense** and **35B-A3B MoE** so far; multimodal (image+text) but **video support not documented**; neither fits 24 GB without INT4. Focused on agentic coding. **Not a candidate.**

## 2. Alternatives (late 2025 / early 2026)

| Model | HF ID | Size | Video | vLLM | 24 GB fit | Notes |
|---|---|---|---|---|---|---|
| **Molmo 2** (Ai2, 2025-12-16) | `allenai/Molmo2-8B` (also 4B, O-7B, 32B-Thinking) | 9B (Qwen3-8B + SigLIP2) | ≤128 frames, ≤2 fps (`max_fps=8` for tracking), emits **timestamps + pointing coords** | official **vLLM ≥ 0.15.0**; `trust_remote_code`; `molmo-utils` for offline preprocessing; FP8 vision backbone NOT supported in stock vLLM | BF16 ~18–19 GB: tight but OK | Best-in-class **temporal grounding/tracking** ("beats Gemini 3 Pro on tracking"; 4B claimed > Qwen3-VL-8B on video QA). Apache-2.0, but trained on some non-commercial datasets (fine for a workshop). |
| **MiniCPM-V 4.5** (OpenBMB, Aug 2025) | `openbmb/MiniCPM-V-4_5` | 8B (Qwen3-8B + SigLIP2) | 3D-Resampler: 6 frames → 64 tokens (**96× compression**), up to 10 fps | yes (vLLM + SGLang) | int4 ~9 GB; BF16 ~18 GB | SOTA <30B on Video-MME at release; **46.7% GPU memory and 8.7% inference time of Qwen2.5-VL-7B** on video — the throughput champion. Superseded for image tasks by newer Qwen, still excellent for high-fps video. |
| **MiniCPM-o 4.5 / MiniCPM-V 4.6** (Feb–Mar 2026) | `openbmb/MiniCPM-o-4_5`, `openbmb/MiniCPM-V-4.6` | 9B / edge (Qwen3.5-0.8B) | streaming omni / 4–16× token compression | partial | yes | o-4.5 is omni (audio+speech) — overkill; V-4.6 is edge-class, too small for rich captions. |
| **GLM-4.6V-Flash** (Z.ai, late 2025) | `zai-org/GLM-4.6V-Flash` | 9B | yes; timestamped event detection; 128K ctx | vLLM works; **Z.ai recommends SGLang for video** | BF16 18–20 GB (tight); community AWQ | Big sibling GLM-4.6V is 106B (no). Flash is solid but vLLM-video is its weaker path. |
| **InternVL3.5** (OpenGVLab, Aug 2025) | `OpenGVLab/InternVL3_5-8B` | 8B | yes | `vllm serve OpenGVLab/InternVL3_5-8B --trust-remote-code`; recipe page exists | BF16 ~17 GB | Video-MME 66.3/68.9 (w/o / w subs) — **below Qwen3-VL-8B (71.4)**. No InternVL4 found as of 2026-06. |
| **Gemma 4** (Google, 2026-04-02; 12B on 2026-06-03) | `google/gemma-4-12b-it`, `gemma-4-31B-it`, 26B-A4B, E2B/E4B | 12B/31B | native video: **≤60 s at 1 fps**, per-frame token budgets 70–1120 | day-0 vLLM support incl. video (frame-extraction pipeline); recipe at recipes.vllm.ai | 12B needs quant (~16 GB int.) | Apache-2.0, 256K ctx. Genuinely interesting, but 12B is **6 days old** and 1 fps caps temporal resolution (misses brake/crossing dynamics). Gemma 3 27B (image-only, no native video) is obsolete for this task. |

## 3. Driving-specific considerations
- **Hallucination on safety-critical events is the documented failure mode.** ScVLM (arXiv:2410.00982, VTTI SHRP-2 data): general VLMs hallucinate on crash/near-crash clips because such events are rare in training data, and are poor at *crash vs near-crash vs baseline* discrimination — exactly our outlier-prevalence axis. Mitigations: (a) ask for **observable evidence, not classification** ("describe what is visible; output `unknown` when not visible"); (b) separate boolean `event_observed` + required `event_description` evidence; (c) keep `severity` an enum the downstream eval treats with suspicion; (d) event-windowed passes at higher fps for temporal claims (braking, crossing).
- Relevant 2025/26 benchmarks: **CrashSight** (arXiv:2604.08457; phase-aware crash captions — pre-crash / collision / aftermath structure worth mirroring in our schema), **CCTVBench** (arXiv:2604.20460; contrastive-consistency traffic VideoQA — VLMs flip answers under contrastive pairing), **DIQ-H** (hallucination persistence under temporal degradation — relevant for night/rain dashcam frames). None publish full leaderboards for our exact candidates yet; on generic temporal suites the ordering is Molmo2-8B ≥ Qwen3.5-9B > MiniCPM-V-4.5 ≈ Qwen3-VL-8B > InternVL3.5-8B.
- 2 fps sampling is the floor for "did the car brake / did a pedestrian cross"; 4 fps on the ±5 s event window is cheap (40 frames) and substantially improves temporal claims. Gemma 4's 1 fps cap is why it is not the pick despite freshness.

## 4. Practical: vLLM serving (v0.22.x)

### Topology
4 independent single-GPU servers (NO tensor parallel — no NVLink, none needed for 9B FP8):
```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i vllm serve Qwen/Qwen3.5-9B-FP8 \
    --port $((8000+i)) \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --limit-mm-per-prompt.video 1 --limit-mm-per-prompt.image 0 \
    --allowed-local-media-path /data/nexar \
    --media-io-kwargs '{"video": {"frame_recovery": true}}' \
    --mm-processor-cache-type shm \
    --reasoning-parser qwen3 &
done
```
Client round-robins over ports 8000–8003. For the fallback model swap in `Qwen/Qwen3-VL-8B-Instruct-FP8` (drop `--reasoning-parser`).

### Passing video (OpenAI-compatible)
Videos go in as `video_url` content parts: `file:///data/nexar/clip.mp4` (requires `--allowed-local-media-path`), http URL, or base64 data URI. vLLM samples frames server-side with the model's HF processor defaults (fps=2 for both Qwen3-VL and Qwen3.5); override per request:
```python
resp = client.chat.completions.create(
    model="Qwen/Qwen3.5-9B-FP8",
    messages=[{"role": "user", "content": [
        {"type": "video_url", "video_url": {"url": "file:///data/nexar/clip_0001.mp4"}},
        {"type": "text", "text": CAPTION_PROMPT},
    ]}],
    temperature=0.7, top_p=0.8, presence_penalty=1.5, max_tokens=700,
    response_format={"type": "json_schema",
                     "json_schema": {"name": "dashcam_caption", "schema": CAPTION_SCHEMA, "strict": True}},
    extra_body={
        "mm_processor_kwargs": {"fps": 1},            # clip pass; use {"fps": 4} for event windows
        "chat_template_kwargs": {"enable_thinking": False},
        "top_k": 20,
    },
)
```
- **Always pass `fps` explicitly, never bare `num_frames`** — vLLM bug #35909 (timestamp/frame-count assertion for Qwen3-VL/Qwen3.5 video when `num_frames` is set without `fps`; fix PR #36136; verify in 0.22.1).
- Frame/token budget (Qwen 32×32-px tokens): 40 s clip @ 1 fps = 40 frames; resized to ~640×360 ⇒ ~225 tokens/frame ⇒ ~9 K vision tokens/clip. Event window ±5 s @ 4 fps = 40 frames, same cost. Stay under `total_pixels = 24576*32*32`. `--max-model-len 32768` is ample.
- **Structured output works with multimodal requests** — vLLM guided decoding (xgrammar default) masks logits regardless of input modality; `response_format json_schema` / `structured_outputs` / legacy `guided_json` all apply. vLLM ≥0.11.1 specifically improved structured-output compatibility for Qwen3-VL. Keep thinking OFF when using strict JSON for the caption pass (reasoning+guided-JSON interplay needs `--reasoning-parser` and costs tokens).
- `VLLM_VIDEO_FETCH_TIMEOUT` (default 30 s) — raise to 120 for cold NFS/disk.

### Throughput (estimate — no published 4090 numbers for these exact models)
Anchors: Qwen3-VL-8B batch video captioning on 16 GB/GPU is a working OSS project (filliptm/Video-Caption-Suite); 8B-class FP8 decode on a 4090 ≈ 60–100 tok/s/stream, prefill several K tok/s; vision encoder adds ~1–3 s per 40-frame clip. Per clip: ~9 K token prefill + ~400 token JSON ⇒ **~6–15 s/clip single-stream; with 4–8 concurrent requests/GPU expect ~250–600 clips/h/GPU**. All 1,500 clips on 4 GPUs: **~45–90 min/pass**; clips + 2 event windows each (~4,500 calls): one afternoon. MiniCPM-V-4.5 would be ~5–10× cheaper on vision tokens if throughput ever becomes the bottleneck.

## 5. Known issues (GitHub)
- **Repetition loops** (Qwen3-VL family on vLLM): vllm#27157 (30B-A3B repeats phrases), QwenLM/Qwen3-VL#1611 ("infinite repetition still exists"). Mitigation: official sampling params (`presence_penalty=1.5`), `max_tokens` cap, JSON-schema guided decoding (a closed schema effectively terminates generation).
- **VRAM creep over multi-day serving** with Qwen3-VL: vllm#28230 — restart workers between passes (cheap insurance in a 2–3 day workshop).
- **Video num_frames/fps assertion** (Qwen3-VL + Qwen3.5): vllm#35909, fix PR #36136 — pass `fps` explicitly.
- **Qwen3.5 video in transformers** had a `StopIteration` in `get_rope_index` (QwenLM/Qwen3.5#58) — fixed upstream (transformers PR #44474); irrelevant if serving via vLLM but pin transformers fresh anyway.
- **High-concurrency multimodal RAM blow-up** (Qwen3.5, vllm#35191): mm-processor cache filled host RAM under heavy concurrent video requests — use `--mm-processor-cache-type shm` and cap client concurrency (~8/GPU).
- **Molmo2**: vLLM ≥0.15.0 required (vllm#31331); FP8 for its vision backbone unsupported in stock vLLM; needs `trust_remote_code` and `transformers ≥ 4.57.1`.
- Hangs after many video requests reported on multi-GPU TP setups (QwenLM/Qwen3-VL#2018) — we avoid TP entirely (DP-only).

## 6. Recommendation

- **Primary: `Qwen/Qwen3.5-9B-FP8`** (fall back to `QuantTrio/Qwen3.5-9B-AWQ` if block-FP8 kernels misbehave on Ada/SM8.9 — vLLM runs FP8 on Ada via Marlin weight-only, but smoke-test in hour 1). Thinking disabled for captioning; optionally enable thinking for a handful of hard event-window re-captions.
- **Fallback: `Qwen/Qwen3-VL-8B-Instruct-FP8`** — most battle-tested open video-VLM serving path in existence; identical API contract, so it is a 1-line swap.
- **Bench-on-the-side (optional, if time)**: `allenai/Molmo2-8B` for the event windows only — its timestamp/pointing-native temporal grounding is the best match for "did the pedestrian cross before the brake?" questions; compare 50 clips against primary.
- Pin: `vllm==0.22.1`, latest `transformers`, `qwen-vl-utils` (for any offline preprocessing), decord/torchcodec for frame extraction in the bag-of-objects branch.
- Two-tier plan stands: this 9B captions everything; the cluster-naming/judging LLM (separate research note) only needs hundreds of text-only calls.

### Caption JSON schema (guided decoding, `strict: true`)
Enums aligned to Nexar metadata (weather/lighting/scene) and NHTSA pre-crash typology groups; free text constrained to observable facts; `unknown` everywhere to absorb uncertainty instead of hallucination.
```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["scene_type","weather","lighting","road_type","road_surface","traffic_density",
               "agents","ego_maneuver","event_observed","event_type","event_description",
               "hazard_description","severity","caption"],
  "properties": {
    "scene_type":     {"enum": ["urban","suburban","rural","highway","intersection","roundabout","parking_lot","tunnel","bridge","other"]},
    "weather":        {"enum": ["clear","rain","snow","fog","overcast","unknown"]},
    "lighting":       {"enum": ["day","night_lit","night_dark","dawn_dusk","glare","unknown"]},
    "road_type":      {"enum": ["divided_highway","undivided_two_way","one_way","ramp","residential","unpaved","other"]},
    "road_surface":   {"enum": ["dry","wet","snow_ice","unknown"]},
    "traffic_density":{"enum": ["empty","light","moderate","heavy","congested"]},
    "agents": {"type": "array", "maxItems": 8, "items": {"type": "object", "additionalProperties": false,
      "required": ["kind","role","action"],
      "properties": {
        "kind":   {"enum": ["car","truck","bus","motorcycle","bicycle","pedestrian","animal","static_object_debris","train","other"]},
        "role":   {"enum": ["lead_vehicle","oncoming","crossing","merging","adjacent_lane","parked","follower","other"]},
        "action": {"type": "string", "maxLength": 120}}}},
    "ego_maneuver":   {"enum": ["going_straight","braking","hard_braking","accelerating","turn_left","turn_right",
                                 "lane_change_left","lane_change_right","swerving","stopped","reversing","unknown"]},
    "event_observed": {"type": "boolean"},
    "event_type":     {"enum": ["rear_end_striking","rear_end_struck","lane_change_sideswipe","opposite_direction",
                                 "crossing_path_junction","pedestrian_conflict","cyclist_conflict","animal_conflict",
                                 "road_departure_loss_of_control","backing_parking","object_debris_strike","other","none"]},
    "event_description":  {"type": "string", "maxLength": 300},
    "hazard_description": {"type": "string", "maxLength": 300},
    "severity":       {"enum": ["none","near_miss","minor_collision","collision","severe_collision","unknown"]},
    "caption":        {"type": "string", "maxLength": 600},
    "uncertainty_notes": {"type": "string", "maxLength": 200}
  }
}
```
`caption` is the dense free-text paragraph that feeds the text-embedding branch and Toponymy exemplar selection; `event_type` maps directly onto the NHTSA pre-crash-typology Jinja naming templates; the categorical fields feed per-cluster metadata histograms.

### Sources
- Qwen3-VL: https://github.com/QwenLM/Qwen3-VL · arXiv:2511.21631 · https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-FP8
- Qwen3.5: https://huggingface.co/Qwen/Qwen3.5-9B · https://qwen.ai/blog?id=qwen3.5 · https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html · https://unsloth.ai/docs/models/qwen3.5 · https://kaitchup.substack.com/p/qwen35-quantization-similar-accuracy
- Qwen3.6: https://github.com/QwenLM/Qwen3.6
- Molmo 2: https://allenai.org/blog/molmo2 · https://huggingface.co/allenai/Molmo2-8B · vllm#31331
- MiniCPM-V 4.5/4.6: https://huggingface.co/openbmb/MiniCPM-V-4_5 · arXiv:2509.18154 · https://github.com/openbmb/MiniCPM-V
- GLM-4.6V: https://huggingface.co/zai-org/GLM-4.6V-Flash · https://github.com/zai-org/GLM-V
- InternVL3.5: https://huggingface.co/OpenGVLab/InternVL3_5-8B · arXiv:2508.18265
- Gemma 4: https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/ · https://vllm.ai/blog/gemma4 · https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html
- vLLM video/structured: https://docs.vllm.ai/en/latest/features/multimodal_inputs/ · https://developers.redhat.com/articles/2025/06/03/structured-outputs-vllm-guiding-ai-responses · vllm#35909 · vllm#27157 · vllm#28230 · vllm#35191
- Driving/hallucination: ScVLM arXiv:2410.00982 · CrashSight arXiv:2604.08457 · CCTVBench arXiv:2604.20460 · DIQ-H arXiv:2512.03992
- Throughput anchor: https://github.com/filliptm/Video-Caption-Suite
