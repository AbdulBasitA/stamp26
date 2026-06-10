# Embedding model lineup research — STAMP 2026 dashcam/Toponymy project

Researched 2026-06-09. Scope: (A) video/visual encoders, (B) text encoders for VLM captions,
(C) video decoding pipeline, plus final lineup + precompute estimates for 4x RTX 4090 (24GB, no NVLink),
CUDA 13.1, Python 3.12, fully local.

Workload assumption used throughout: **1,500 full clips (~40 s) + ~30,000 event-centered sub-windows ≈ 31.5k video items**;
1,500 short structured captions for text embedding.

---

## A. Video / visual embeddings

### A.1 V-JEPA 2 (primary motion/temporal encoder — also the project story)

**Status (June 2026).** Two generations exist:

- **V-JEPA 2.0** (June 2025) — fully integrated in HF `transformers` (model type `vjepa2`, present in current
  transformers v5.x; doc page rendered at v5.10.2). Checkpoints in the
  [facebook V-JEPA 2 collection](https://huggingface.co/collections/facebook/v-jepa-2-6841bad8413014e185b497a6):
  - `facebook/vjepa2-vitl-fpc64-256` — ViT-L/16, 300M, 256px, hidden 1024 ← **recommended**
  - `facebook/vjepa2-vith-fpc64-256` — ViT-H, 600M, 256px
  - `facebook/vjepa2-vitg-fpc64-256` / `facebook/vjepa2-vitg-fpc64-384` — ViT-g 1B, 256/384px
  - Classification heads (attentive-probe finetuned): `facebook/vjepa2-vitl-fpc16-256-ssv2`,
    `...-fpc64-256-diving48`, `facebook/vjepa2-vitg-fpc64-384-ssv2`, etc.
- **V-JEPA 2.1** (released 2026-03-16; paper [arXiv:2603.14482](https://arxiv.org/html/2603.14482v1)) — four encoders
  at 384px: ViT-B 80M, ViT-L 300M (distilled from ViT-G), ViT-g 1B, ViT-G 2B. Improvements: corrected RoPE,
  hierarchical dense features, per-layer norms, RoPE interpolation. **NOT yet in transformers**:
  [transformers issue #45496](https://github.com/huggingface/transformers/issues/45496) (opened 2026-04-17) and PR #45497 are
  still open; weights are **torch.hub-only** from [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)
  (`torch.hub.load('facebookresearch/vjepa2', 'vjepa2_1_vit_large_384')`, ckpts at
  `dl.fbaipublicfiles.com/vjepa2/vjepa2_1_*_384.pt`); HF Hub hosting requested in
  [vjepa2 issue #137](https://github.com/facebookresearch/vjepa2/issues/137). torch.hub pulls repo code at runtime →
  **bad for an air-gapped workshop**. → Use **V-JEPA 2.0 ViT-L via transformers**; mention 2.1 in the talk.

**The Nexar/BADAS story checks out.**
[BADAS (arXiv:2510.14876)](https://arxiv.org/abs/2510.14876) and [BADAS-2.0 (arXiv:2604.05767, Apr 2026)](https://arxiv.org/html/2604.05767v2)
fine-tune a V-JEPA2 backbone for ego-centric collision prediction on Nexar data.
[`nexar-ai/BADAS-Open`](https://huggingface.co/nexar-ai/BADAS-Open) (Apache-2.0, trained on the same 1.5k public Nexar
videos we use) = **V-JEPA2 ViT-L, 16 frames @ 256×256 → 2048 patches × 1024-dim**, attentive probe (12 queries × 64-dim),
3-layer MLP head. So embedding our sub-windows with `vjepa2-vitl-fpc64-256` at 16 frames @ 256px reproduces the exact
BADAS-Open feature space — strong narrative + an optional extra feature set (BADAS probe activations / collision logits)
for the outlier eval. BADAS-2.0 scales to 178.5k labeled videos and ships distilled Flash (86M) / Flash-Lite (22M)
variants with public inference code ([getnexar/BADAS-Open](https://github.com/getnexar/BADAS-Open)).

**How to extract a clip-level embedding.** The base `VJEPA2Model` has **no CLS token and no pooler**; the attentive
pooler exists only in `VJEPA2ForVideoClassification` (task-finetuned → biased; skip it for unsupervised clustering).
Standard practice: **mean-pool `last_hidden_state` over all patch tokens**, with `skip_predictor=True` to avoid the
predictor forward. Token count N = (T/tubelet 2) × (H/16) × (W/16): 16f@256 → 2048 tokens; 64f@256 → 8192 tokens.
`frames_per_clip` in the config "does not impact inference" — you may feed 16/32/64 frames.

```python
import torch
from transformers import AutoModel, AutoVideoProcessor

mid = "facebook/vjepa2-vitl-fpc64-256"
proc = AutoVideoProcessor.from_pretrained(mid)
model = AutoModel.from_pretrained(mid, dtype=torch.bfloat16,
                                  attn_implementation="sdpa").to("cuda").eval()

# video: T x C x H x W uint8 (16 frames for sub-windows, 32-64 for full clips)
inputs = proc(video, return_tensors="pt").to("cuda")
with torch.inference_mode():
    out = model(**inputs, skip_predictor=True)
emb = out.last_hidden_state.mean(dim=1)        # (B, 1024)  == get_vision_features().mean(1)
```

**Footprint/throughput.** ViT-L bf16 ≈ 0.7GB weights; 16f@256 (2048 tokens) ≈ ~2.8 TFLOPs/clip →
~25-40 clips/s/4090; 64f (8192 tokens) roughly 5x slower (~5-8 clips/s). Sequence stays well under 24GB at batch 8-16.

### A.2 SigLIP2 (language-aligned semantic encoder — frame-level + mean-pool)

[SigLIP 2 blog](https://huggingface.co/blog/siglip2); checkpoints under the google org. Recommended:

- **`google/siglip2-so400m-patch16-384`** ← recommended (400M vision tower, 384px, 576 patch tokens)
- `google/siglip2-so400m-patch14-384` (729 tokens, equivalent quality, slightly slower)
- `google/siglip2-so400m-patch16-256` (cheaper), `...-patch16-512` (more expensive)
- `google/siglip2-so400m-patch16-naflex` — **NaFlex** native-aspect-ratio variant (max 1024 patches, processor default
  `max_num_patches=256`). Attractive for 16:9 1280×720 dashcam frames (no square-resize distortion); slightly more
  plumbing. Use fixed-res square resize first; NaFlex as ablation if time.

Usage: `AutoModel.from_pretrained(...)` + `model.get_image_features(**processor(images=...))` → 1152-dim,
L2-normalize per frame, **mean-pool across 8-16 uniformly sampled frames per window, re-normalize**. Frame-averaged
image-text embeddings are the standard strong baseline for video retrieval — Meta's Perception Encoder paper
([arXiv:2504.13181](https://arxiv.org/abs/2504.13181)) reports its zero-shot *video* retrieval numbers with exactly
this frame-averaging scheme, and SigLIP-so400m family models are top performers on MIEB
([arXiv:2504.10471](https://arxiv.org/pdf/2504.10471)) retrieval/clustering-style tasks. This is the embedding most
likely to cluster by **scenario semantics** ("left turn at night in rain") because the space is language-aligned;
it also gives free zero-shot text probes (e.g., score clusters against "rainy night intersection") for naming support.

Throughput: ~200-350 frames/s/4090 bf16 batched @384 → 31.5k items × 8 frames = 252k frames ≈ **15-25 min on one GPU**.

### A.3 DINOv3 (optional 3rd embedding — appearance contrast ablation)

[facebook/dinov3 collection](https://huggingface.co/collections/facebook/dinov3); transformers ≥ 4.56. Sizes ViT-S → ViT-7B;
relevant: **`facebook/dinov3-vitl16-pretrain-lvd1689m`** (300M, 1024-dim, use `outputs.pooler_output` (CLS);
201 tokens @224 = 1 CLS + 4 registers + 196 patches).

**Is it better for scenario semantics? No.** Self-supervised DINO-family features are sensitive to visual structure /
texture / layout and cluster "images that look similar", while language-aligned models (SigLIP2/CLIP/PE) group by
high-level semantics (object identity, scene category) — confirmed by 2026 comparisons
([SigLIP2 vs DINOv2 comparison](https://underfitted.dev/2026/03/01/siglip-2-vs-dinov2-battle-of-the-embeddings-titans/),
[Data or Language Supervision: What Makes CLIP Better than DINO?](https://arxiv.org/html/2510.11835v1)). For our goal
(cluster "left turn at night in rain", not "gray asphalt texture"), SigLIP2/V-JEPA2 + captions are the right primaries.
**Value of DINOv3 here:** a cheap third map for the workshop narrative — "what does a topic map look like when the
embedding is appearance-driven vs language-aligned?" (expect lighting/weather/road-texture clusters; conveniently our
metadata has lighting/weather histograms to show it). Include only if time permits.

**Caveat:** DINOv3 weights are **gated** on HF (must accept the dinov3-license + share contact info) — request access and
pre-download before the (possibly air-gapped) workshop.

### A.4 Other video encoders considered

| Model | Status June 2026 | Verdict |
|---|---|---|
| **Perception Encoder PE-Core** (Meta, [arXiv:2504.13181](https://arxiv.org/abs/2504.13181)) | `facebook/PE-Core-B16-224`, `PE-Core-L14-336`, `PE-Core-G14-448` (1.88B vision, 1280-dim, Apache-2.0). G14 needs the custom [perception_models](https://github.com/facebookresearch/perception_models/) repo (`pe.CLIP.from_config(...)`); B/L also exist as timm exports (`timm/vit_pe_core_*`, `facebook/pe_core_large_patch14_336_timm`). Excellent zero-shot video retrieval (frame-averaged, K400 76.9). | **Skip** (overlaps SigLIP2's role; custom-repo friction). Fallback if SigLIP2 disappoints. |
| **VideoPrism** (Google, ICML 2024; weights July 2025) | On HF: `google/videoprism-base-f16r288`, `-large-f8r288`, video-text `-lvt-*` variants. **JAX/Flax only**, custom repo [google-deepmind/videoprism](https://github.com/google-deepmind/videoprism), no PyTorch/transformers path. | **Skip** — JAX stack on an air-gapped CUDA-13 box is needless risk. |
| **InternVideo2 / 2.5** (OpenGVLab) | InternVideo2.5 = a **chat MLLM** (InternVL2.5-based), not an embedding encoder. The retrieval-capable InternVideo2-CLIP/Stage-2 (1B/6B) models require custom repo code, no clean transformers integration. | **Skip.** |
| **V-JEPA 2.1** | See A.1 — torch.hub-only, transformers PR open. | **Mention, don't use.** |

---

## B. Text embeddings for VLM captions (~1,500 short structured docs)

The 2026 open-weights default is the **Qwen3-Embedding** family (June 2025; still top open models on MTEB mid-2026,
[Qwen blog](https://qwenlm.github.io/blog/qwen3-embedding/), [GitHub](https://github.com/QwenLM/Qwen3-Embedding)).
All Apache-2.0, native sentence-transformers support, instruction-aware, MRL (truncatable dims), 32k context.

MTEB-Multilingual scores (from the [Qwen3-Embedding-4B model card](https://huggingface.co/Qwen/Qwen3-Embedding-4B)):

| Model | Params | Dim | Mean (overall) | **Clustering** | License / notes |
|---|---|---|---|---|---|
| **Qwen3-Embedding-8B** | 8B | 4096 | 70.58 | 57.65 | Apache-2.0 |
| **Qwen3-Embedding-4B** | 4B | 2560 | 69.45 | **57.15** | Apache-2.0 ← recommended |
| **Qwen3-Embedding-0.6B** | 0.6B | 1024 | 64.33 | 52.33 | Apache-2.0; fast fallback |
| gemini-embedding-exp (API) | — | — | 68.37 | 54.59 | not local |
| multilingual-e5-large-instruct | 0.6B | 1024 | 63.22 | 50.75 | |
| **BGE-M3** | 0.6B | 1024 | 59.56 | **40.88** | clustering is its weakest task — **do not use for clustering** |

Others checked: **jina-embeddings-v3/v4** — v4 is 3.8B multimodal but **CC-BY-NC-4.0** + `trust_remote_code` → skip.
**nomic-embed-text-v2-moe** (475M MoE, Apache) and **EmbeddingGemma-300m** — fine small models, but no advantage over
Qwen3-0.6B here. Nothing newer in 2026 has displaced Qwen3-Embedding as the open default
([BentoML 2026 guide](https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models),
[MTEB rankings Mar 2026](https://awesomeagents.ai/leaderboards/embedding-model-leaderboard-mteb-march-2026/)).

**Recommendation: `Qwen/Qwen3-Embedding-4B`** — at 1,500 short docs the cost difference vs 0.6B is seconds, and
clustering quality is ~5 points higher. Use `Qwen/Qwen3-Embedding-0.6B` if VRAM is contended (e.g., while vLLM holds
the big LLM); both fit easily anyway (4B ≈ 8GB bf16).

```python
# uv add sentence-transformers>=4.1
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("Qwen/Qwen3-Embedding-4B",
                            model_kwargs={"dtype": "bfloat16", "device_map": "cuda:3"},
                            tokenizer_kwargs={"padding_side": "left"})
embs = model.encode(captions, batch_size=64, normalize_embeddings=True)
# documents need NO instruction prefix; instructions are for queries only.
# Optionally truncate via MRL: model.encode(..., truncate_dim=1024) — fine for UMAP input.
```

Note: caption embeddings here are **documents** for symmetric clustering — embed all texts identically without an
instruction prefix (or use the same one for all; just be consistent).

---

## C. Video decoding pipeline

**decord is dead — do not use.** dmlc/decord's last commit is ~3 years old, with open memory-leak and large-video crash
issues; downstream projects are actively removing it
([sglang issue #5116](https://github.com/sgl-project/sglang/issues/5116)). It lingers only as a legacy dependency in
some VLM stacks.

**torchcodec is the 2026 answer** ([meta-pytorch/torchcodec](https://github.com/meta-pytorch/torchcodec),
[PyPI](https://pypi.org/project/torchcodec/)): latest **0.14.0 (released 2026-06-03)**, requires torch ≥ 2.11,
Python 3.10-3.14, FFmpeg **4-8** (uses system FFmpeg shared libs), CUDA wheels default on Linux, optional **NVDEC GPU
decoding** (`VideoDecoder(path, device="cuda")` — 4090 has NVDEC). It is PyTorch's official replacement for
torchvision video IO (removed in torchvision 0.24) and is what the HF vjepa2 docs themselves use. **PyAV** is fine and
maintained but is just raw FFmpeg bindings — more code for the same result. (`video_reader-rs` exists as another
maintained option; unnecessary here.)

```python
# uv add "torchcodec==0.14.*"   (needs system ffmpeg shared libraries, v4-8)
from torchcodec.decoders import VideoDecoder
dec = VideoDecoder(path, seek_mode="approximate")     # fast seeks; fine for sampling
frames = dec.get_frames_played_at(seconds=ts_list).data   # uint8 (N,C,H,W)
```

**Caching strategy — decode once, cache JPEGs, embed many times:**
1. One pass over the 1,500 mp4s (32-core CPU, process pool, torchcodec approximate mode; or NVDEC): sample at fixed
   **~8 fps**, resize longest side to **512px**, save **JPEG q90** as `cache/{clip_id}/{ms:08d}.jpg`.
   ≈ 1,500 × 40s × 8fps ≈ **480k frames ≈ 20-30GB** (1.7TB disk — trivial). Decode pass ≈ **10-30 min wall**.
2. Clip/sub-window datasets assemble frame lists from the cache by timestamp; each embedding model applies its own
   processor (resize/normalize) at batch time. JPEG beats tensor caching: 5-10x smaller, resolution/normalization-agnostic
   (one cache serves SigLIP2@384, V-JEPA2@256, DINOv3@224, VLM captioning, and DataMapPlot hover thumbnails).
3. Persist final embeddings as float16 `.npy`/parquet keyed by (clip_id, window_start, window_end, model) — all
   embeddings for all models < 1GB total.
4. Exception: if you want V-JEPA2 at exact 30fps-contiguous 16-frame windows (BADAS parity) rather than 8fps-sampled
   frames, do a second targeted decode for event windows only (~30k × 16 frames, still cheap), or accept 8fps sampling
   (for 2s windows @8fps you get 16 frames spanning 2s — arguably better temporal context anyway).

---

## Final recommended lineup

| Slot | Model (exact HF ID) | Input | Embedding | Why |
|---|---|---|---|---|
| Visual 1 (temporal) | `facebook/vjepa2-vitl-fpc64-256` | 16 frames @256 (sub-windows), 32-64 frames @256 (full clips) | mean of `last_hidden_state`, 1024-d, `skip_predictor=True` | motion/dynamics; exact BADAS-Open backbone (Nexar story) |
| Visual 2 (semantic) | `google/siglip2-so400m-patch16-384` | 8 frames/window, square 384 | `get_image_features` → L2-norm → mean → L2-norm, 1152-d | language-aligned scenario semantics; free zero-shot text probes |
| Visual 3 (optional ablation) | `facebook/dinov3-vitl16-pretrain-lvd1689m` (gated — pre-download) | 8 frames @224-384 | `pooler_output` mean over frames, 1024-d | appearance-vs-semantics contrast demo |
| Text | `Qwen/Qwen3-Embedding-4B` (fallback `Qwen/Qwen3-Embedding-0.6B`) | 1,500 captions | sentence-transformers, normalized, 2560-d (MRL-truncate to 1024 ok) | best open clustering quality, Apache-2.0, ST-native |
| Bonus feature set | `nexar-ai/BADAS-Open` (Apache-2.0) | 16 frames @256 | attentive-probe features / collision logits | task-aligned features for the outlier-prevalence eval |

**Version pins:** `transformers>=4.56` (DINOv3; current v5.x line fine — vjepa2/siglip2/dinov3 all in-tree),
`torchcodec==0.14.*` + `torch>=2.11`, `sentence-transformers>=4.1`, system FFmpeg 6/7 shared libs.
Pre-download all weights + accept DINOv3 gate before the workshop.

**Precompute time on our box (31.5k windows + 1.5k clips, bf16, batched, 1 GPU per model in parallel):**

| Stage | One 4090 | Notes |
|---|---|---|
| Decode + JPEG cache (CPU/NVDEC) | 10-30 min | once |
| V-JEPA2 ViT-L, 16f sub-windows | ~15-25 min (25-40 clips/s) | 64f full clips (1.5k): +5-10 min |
| SigLIP2 so400m, 252k frames | ~15-25 min | |
| DINOv3 ViT-L, 252k frames | ~10-15 min | optional |
| Qwen3-Embedding-4B, 1.5k docs | <2 min | |

**Total wall time ≈ 1-1.5 h** running the three visual models on separate GPUs (no NVLink needed — embarrassingly
parallel sharding by clip), leaving GPU3 + headroom for the vLLM captioner. Comfortably inside a 2-3 day workshop budget;
the VLM captioning pass, not embedding, will be the throughput bottleneck.

### Sources (primary)
- https://huggingface.co/docs/transformers/model_doc/vjepa2 ; https://huggingface.co/facebook/vjepa2-vitl-fpc64-256
- https://github.com/facebookresearch/vjepa2 ; https://github.com/huggingface/transformers/issues/45496 ; https://github.com/facebookresearch/vjepa2/issues/137 ; https://arxiv.org/html/2603.14482v1
- https://huggingface.co/nexar-ai/BADAS-Open ; https://arxiv.org/abs/2510.14876 ; https://arxiv.org/html/2604.05767v2
- https://huggingface.co/blog/siglip2 ; https://huggingface.co/google/siglip2-so400m-patch16-384 ; https://huggingface.co/google/siglip2-so400m-patch16-naflex
- https://huggingface.co/collections/facebook/dinov3 ; https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
- https://arxiv.org/abs/2504.13181 (Perception Encoder) ; https://huggingface.co/facebook/PE-Core-G14-448 ; https://github.com/facebookresearch/perception_models/
- https://github.com/google-deepmind/videoprism ; https://huggingface.co/google/videoprism-base-f16r288
- https://huggingface.co/Qwen/Qwen3-Embedding-4B ; https://huggingface.co/Qwen/Qwen3-Embedding-0.6B ; https://qwenlm.github.io/blog/qwen3-embedding/
- https://pypi.org/project/torchcodec/ ; https://github.com/meta-pytorch/torchcodec ; https://github.com/sgl-project/sglang/issues/5116 (decord deprecation)
- https://arxiv.org/pdf/2504.10471 (MIEB) ; https://underfitted.dev/2026/03/01/siglip-2-vs-dinov2-battle-of-the-embeddings-titans/ ; https://arxiv.org/html/2510.11835v1
