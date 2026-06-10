# STAMP 2026 — A Named Atlas of the Long Tail of Driving

**Thesis:** The Toponymy pipeline (embed → cluster → characterize → LLM-name → map) generalizes from text/cyber-telemetry to ego-centric driving video, *if* you (1) sessionize at event scale, (2) manufacture a text view of behavior via VLM captions, and (3) validate names separately for taxonomy alignment and cluster fidelity. Demonstrated on the Nexar collision dataset with collision labels as ground-truth rare events.

**Deliverables:**
1. Interactive, air-gap-safe DataMapPlot scenario atlas (hover thumbnails, NHTSA-aligned topic names, multi-layer hierarchy) for 3 embedding tracks.
2. Eval (a): low-prevalence collision-as-outlier detection — AP/P@k with CIs, clip-level vs event-window-level, per track. Headline: the detectability gap proves sessionization choice determines anomaly visibility.
3. Eval (b): name quality — taxonomy alignment %, LLM-judge per-cluster trustworthiness, rerun stability. Three separate numbers, never conflated.
4. Short deck: cyber→driving mapping, the atlas demo, the numbers, one slide per STAMP RQ touched.

**STAMP research questions addressed:** #1 (non-text characterization — captions + metadata histograms), #2 (cross-modal: visual vs caption embeddings of same clips), #3 (model selection under air-gapped constraints — entire stack local), #4 (taxonomy-aligned prompting — NHTSA instead of ATT&CK), #6 (topic validation — judge + stability), #10 (label-aware/diagnostic maps — clusters cutting across weather/scene/collision labels; BADAS failure overlay as stretch). #11 optional stretch (constrained names for ablating mundane background clusters).

**Citation (mandatory on every map/slide):** *Moura & Zvitia, Nexar Collision Dataset, Hugging Face 2025.* No face/plate close-ups in thumbnails.

---

## 1. Decisions log (adjudicated — do not relitigate during execution)

| # | Decision | Resolution | Source |
|---|---|---|---|
| D1 | Captioner | **Qwen/Qwen3.5-9B-FP8** primary; `Qwen/Qwen3-VL-8B-Instruct-FP8` fallback; `QuantTrio/Qwen3.5-9B-AWQ` escape hatch. 4× single-GPU vLLM workers, no TP. | research/vlm-captioning.md |
| D2 | Namer/judge | **Qwen/Qwen3.6-35B-A3B-FP8** TP=4, thinking disabled server-side; fallback `Qwen/Qwen3-32B-AWQ` (1 GPU). No 70B (dominated). Optional `openai/gpt-oss-20b` second-family judge. | research/naming-llm.md |
| D3 | Visual embedders | **`facebook/vjepa2-vitl-fpc64-256`** (pretrain ckpt, NOT the ssv2 classification ckpt — unbiased for unsupervised clustering) + **`google/siglip2-so400m-patch16-384`**. DINOv3 optional appearance-vs-semantics ablation (gated — request access in Phase 0). V-JEPA 2.1 is torch.hub-only → excluded; mention in talk only. | research/embeddings.md |
| D4 | Text embedders | Captions: **`Qwen/Qwen3-Embedding-4B`**. Toponymy `text_embedding_model` (keyphrases/topic names): **`Qwen/Qwen3-Embedding-0.6B`**. bge-m3 rejected (bad at clustering). | research/embeddings.md |
| D5 | Detector | **D-FINE** (`ustc-community/dfine-medium-obj2coco`, Apache, transformers-native — fits main venv, no third env). RF-DETR optional upgrade in own venv. **No Ultralytics/AGPL anywhere.** | research/vectorizers-detection.md + critic |
| D6 | Video decoding | **torchcodec 0.14** + one-pass JPEG cache. decord banned (abandoned). Verify system FFmpeg shared libs. | research/embeddings.md |
| D7 | Reweighting | **InformationWeightTransformer** (house doctrine), never TF-IDF. Hellinger straight on IWT output (metric L1-normalizes internally — no manual normalize step). | research/pipeline-examples.md |
| D8 | Outlier scorers | GLOSH via classic **`hdbscan==0.8.44`** (fast_hdbscan lacks it) on 5-D UMAP space; high-dim kNN cosine distance as distortion-free control; LOF tertiary. Never score in 2-D. | research/clustering-viz.md |
| D9 | Embedding windows | **2 s / 16 frames @ 8 fps** centered on `time_of_event` (positives) — matches BADAS, Kaggle winner, fpc16 spec. Negatives: uniform non-overlapping 2 s tiling. Per-clip event times (events NOT centered: 3.0–56.8 s). | research/nexar-badas.md + critic |
| D10 | Caption windows | **±5 s @ 4 fps** around `time_of_event` (VLM needs lead-up context; deliberately different from D9). Caption budget: 1,500 clips + 750 positive event windows ≈ 2,250 calls. **Window-level eval (a) is embedding-only — no window captions needed for it.** | critic adjudication |
| D11 | Window-level labels | Positive = event-centered windows of positive clips. Negative = all windows of negative clips. Positive-clip non-event windows: embedded and mapped, **excluded from AP** (ambiguous). | critic gap fix |
| D12 | Metadata injection | One mechanism per track: token track = metadata tokens **with/without ablation**; caption track = VLM schema fields only (no suffix append); naming prompts = per-cluster **differential** histograms (cluster-vs-corpus deltas, acme3 style) via prompt post-edit. Never stack mechanisms. | critic adjudication |
| D13 | Stability protocol | Stability **metric** = independent reruns with fresh `ToponymyClusterer`/`KeyphraseBuilder` instances (mutable-default trap). Procrustes + `init=ref-coords` only for **visual** map comparison. | critic adjudication |
| D14 | Toponymy↔vLLM | Sync `OpenAINamer(api_key='EMPTY', model='namer', base_url='http://localhost:8000/v1')`, or Async + `OPENAI_BASE_URL` env var (**AsyncOpenAINamer drops its base_url arg — verified bug**). Never in-process VLLMNamer (torch conflict). | research/toponymy.md |
| D15 | Taxonomy naming format | `"<NHTSA-group>: <specific qualifier>"` composite names; disambiguation pass neutralized via custom `extract_topic_names = lambda info, old, raw: old` (issue #101). | research/toponymy.md |
| D16 | Judge held-out set | Cluster members **minus** `exemplar_indices` (exemplars are what the namer saw — judging on them tests memorization). Sample ≤20/cluster. | critic correction |
| D17 | Serving schedule | Strictly sequential, no co-residency: decode → captioning (4×1-GPU, ports 8001-8004) → embeddings (4 GPUs) → [BADAS stretch] → namer (TP=4, port 8000). | critic adjudication |
| D18 | Venvs | **Two uv venvs**: `venvs/vllm` (vllm==0.22.1, pins own torch 2.11) + `venvs/main` (torch 2.12 cu130, transformers==4.57.6 — toponymy <5 cap, editable Tutte clones). cu128 is dead — cu130 everywhere. No flash-attn (SDPA + vLLM bundled kernels). | research/env-setup.md |
| D19 | BADAS overlay | **Stretch (Phase 8)**, on the **1,344-clip test split** (public ground truth + official scorer) — BADAS-Open trained on the 1,500 train clips, so train-split overlay = training-residual audit only. | research/nexar-badas.md |
| D20 | Upstream PR | Prepare the 1-line `AsyncOpenAINamer` base_url fix as a draft PR pre-workshop (optional, ~1 h, icebreaker with maintainers). | research/toponymy.md |
| D21 | Version pins | `toponymy==0.5.2` (breaking refactor imminent, #149), `datamapplot==0.7.3`, `umap-learn==0.5.12`, `fast-hdbscan==0.3.2`, `hdbscan==0.8.44`, `evoc==0.3.1`, `vectorizers==0.2.2`, `vllm==0.22.1`, `transformers==4.57.6` (main) / vllm-resolved (vllm env), `torchcodec==0.14.*`, `sentence-transformers==5.1.2`, `badas==1.1.3` (stretch). | all |
| D22 | Eval (a) protocol | Subsets = all 750 negatives + k positives, k ∈ {25, 50, 75} (≈3/6/9% prevalence), **30 resamples** each, stratified by alert-to-event gap; refit UMAP + scorers per subset; metrics AP, P@25, P@50 (AUROC secondary); report **lift over chance (AP/prevalence)** so clip vs window levels are comparable; sensitivity pass excluding ambiguous/label-noise clips. | earlier corrections + critic |

---

## 2. Architecture

```
                       ┌─ Track T: detections→tokens→NgramVectorizer→IWT ──(hellinger)──┐
 Nexar mp4s ─ decode ──┼─ Track V: V-JEPA2 / SigLIP2 clip+window embeddings ──(cosine)──┼─→ UMAP 5D (cluster/score)
 + metadata   (JPEG    ├─ Track C: VLM captions ─→ Qwen3-Embedding-4B ───────(cosine)──┤   UMAP 2D (display)
   .csv        cache)  └─ thumbnails (hover) ───────────────────────────────────────────┘        │
                                                                                    ToponymyClusterer (multi-layer)
                                                                                                  │
                                       captions + differential metadata histograms → Toponymy → LLM names (NHTSA templates)
                                                                                                  │
                            eval (a): GLOSH/kNN outlier ranking @ low prevalence          DataMapPlot interactive atlas
                            eval (b): alignment % + judge precision + stability           (offline_mode, thumbnails, topic tree)
```

Tracks are independently complete: if one fails, the project stands on the others. Priority order if time pressure: C (captions) > V-SigLIP2 > T (tokens) > V-JEPA2 > ablations.

---

## 3. Key specs

### 3.1 Caption JSON schema (strict guided JSON via xgrammar)
Fields: `scene_type, weather, lighting, road_type, road_surface, traffic_density, agents[], ego_maneuver, event_observed (bool), event_type (NHTSA-group enum + "none"/"unknown"), event_description, hazard_description, severity, caption (2-3 sentence free text), uncertainty_notes`. Every enum includes `"unknown"` (hallucination absorber — ScVLM documents VLM hallucination on crash clips). Sampling: `temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5` (anti-repetition, official Qwen params), `fps` passed explicitly per request (`{"mm_processor_kwargs": {"fps": 1}}` clips / `{"fps": 4}` event windows), thinking off via `chat_template_kwargs`. Treat caption `severity`/`event_type` as model output, never ground truth — evals use Nexar labels only.

### 3.2 Token schema (Track T)
Per detection @ conf ≥0.5: `{class}|{zone L/C/R by bbox center-x thirds}|{range near/mid/far by bbox-height >0.25 / 0.08–0.25 / <0.08}`, per-class-per-frame cap 10. Per frame: `traffic:{none|light|moderate|heavy}` (vehicle count 0/1-3/4-8/9+), `vru:present`. Metadata tokens (`weather:*, lighting:*, scene:*`) at ~5% bag mass — **with/without ablation mandatory** (risk: trivializes geometry to "rainy night driving"). Clips @ 1 fps; event windows @ 4 fps over ±4 s. Vocab lands <~200 columns. Thresholds are pilot-validated in Phase 3, not gospel.

### 3.3 UMAP / clustering params
- Token track: `UMAP(metric='hellinger', unique=True, init='pca')` directly on sparse IWT output.
- Dense tracks: clustering space `UMAP(n_components=5, n_neighbors=15, min_dist=0.0, n_epochs=500, metric='cosine', init='pca')`; display map separate `UMAP(n_components=2, min_dist=0.1, metric='cosine')`.
- Outlier ablation variant: 5-D space with `set_op_mix_ratio=0.25` (avoids contracting outliers into clusters).
- `ToponymyClusterer(min_clusters=4, min_samples=5, base_min_cluster_size=10–15)` — **fresh instance every fit** (D13).
- Handle UMAP disconnected-vertex NaNs (clips with unique token sets) with an `i_valid` mask, acme3-style.

### 3.4 Serving commands (flags verified in Phase 0 smoke tests — see §6)
```bash
# Captioning phase: one worker per GPU, ports 8001-8004
CUDA_VISIBLE_DEVICES=$i vllm serve Qwen/Qwen3.5-9B-FP8 --port 800$((i+1)) \
  --max-model-len 32768 --gpu-memory-utilization 0.92 \
  --limit-mm-per-prompt.video 1 --allowed-local-media-path <data dir> \
  --mm-processor-cache-type shm
# client concurrency ≤8/GPU; restart workers between passes (VRAM creep, vllm#28230)

# Naming phase (after captioning ends):
NCCL_P2P_DISABLE=1 vllm serve Qwen/Qwen3.6-35B-A3B-FP8 --port 8000 \
  --served-model-name namer --tensor-parallel-size 4 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --language-model-only --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' --enable-prefix-caching
```

### 3.5 Eval (b) operationalization
- **Alignment:** parse `"<group>: <qualifier>"` → % parse-valid; % group ∈ NHTSA 2019 9-group list; scenario-level match against the 36-scenario list where given; coverage distribution across groups. (NHTSA list: research/nexar-badas.md — **verbatim numbering needs one human browser check before the template ships**.)
- **Trustworthiness:** judge prompt per held-out member (D16): "Does this topic name accurately describe this clip's caption? yes/partial/no" at temp 0 → per-cluster precision; flag clusters <0.7 on the map (visual confidence layer). Optional gpt-oss-20b second judge for family-bias control.
- **Stability:** 3 independent reruns (D13); match clusters across runs by membership Jaccard; report exact-name match rate, group-level match rate, and caption-embedding cosine of matched names.

---

## 4. Phases

Each phase ends with explicit **Verify** gates. Don't start phase N+1 with phase N gates red.

### Phase 0 — Foundations (env, downloads, smoke tests) — pre-workshop, ~half day + downloads overnight
**Build:**
- `git init` at repo root; `.gitignore` for `data/ cache/ artifacts/ venvs/ repos/ offline/`; create remote; **push daily — /home is RAID0 at 94%, zero redundancy**.
- Directory layout per §8.
- `uv venv venvs/vllm --python 3.12 && uv pip install -p venvs/vllm vllm==0.22.1` (never override its torch).
- `uv venv venvs/main --python 3.12`; `uv pip install -p venvs/main --torch-backend=cu130 torch==2.12.0 torchvision`; then editable Tutte clones (`-e repos/toponymy -e repos/datamapplot -e repos/vectorizers -e repos/fast_hdbscan -e repos/evoc`) + `transformers==4.57.6 sentence-transformers==5.1.2 umap-learn hdbscan==0.8.44 torchcodec glasbey accelerate openai jupyterlab opencv-python-headless av pandas pyarrow datasets`.
- Request DINOv3 gated access NOW (lead time). Download manifest (§5) via `hf download` (CLI is `hf`, not `huggingface-cli`).
- `dmp_offline_cache` + `--export offline/dmp_cache.zip`.
- Optional (D20): draft the AsyncOpenAINamer base_url PR.

**Verify:**
- [ ] Hour-1 smoke checklist (§6) fully green.
- [ ] Both venvs import-clean; `uv pip freeze` snapshots committed.
- [ ] evoc + fast_hdbscan co-import test (30 s — official test suite still skips this combo).
- [ ] All manifest weights resolve and load (exact post-cutoff HF IDs are single-source claims — **verify at download, this week**).
- [ ] One end-to-end toy run: 20 fake captions + random embeddings → Toponymy (against live namer) → DataMapPlot HTML → renders in a **disconnected** browser.

### Phase 1 — Data acquisition & decode cache — pre-workshop, ~half day
**Build:**
- `hf download nexar-ai/nexar_collision_prediction --repo-type dataset` (31.4 GB).
- `manifest.parquet`: clip id, label, time_of_event/alert, weather/lighting/scene, duration, fps, decode status. Handle: 2 missing-weather clips, non-centered events, defensive decode (broken fps metadata — BADAS code has the guard pattern).
- One-pass decode @ 8 fps, longest side 512, JPEG q90 → `cache/frames/` (~25–30 GB); 1 thumbnail/clip (160 px, base64-ready) → `cache/thumbs/`.

**Verify:**
- [ ] 1,500 train clips present, 750/750 split, metadata distributions match published (Clear 919 / Cloudy 495 / Rain 83 / Snow 1; Urban 784...).
- [ ] Decode success ≥99.5%; failures logged and excluded via `i_valid`.
- [ ] Spot-check 10 event windows: frames at `time_of_event` actually show the event.

### Phase 2 — Captioning — pre-workshop, ~1 evening
**Build:**
- vLLM worker launcher + async client (concurrency ≤8/GPU); caption schema (§3.1); two passes: 1,500 clips (fps=1) + 750 positive event windows (±5 s, fps=4) → `artifacts/captions/*.parquet`.
- **20-clip pilot first**: throughput + quality + JSON-parse check before the bulk run.

**Verify:**
- [ ] JSON parse rate ≥99%; `unknown` rates sane (<30% per field); no repetition loops.
- [ ] Manual spot-check 20 captions vs video — no gross hallucination; event windows mention the event for clear positives.
- [ ] Throughput within 2× of estimate (else fallback model per D1).

### Phase 3 — Embeddings (all tracks) — pre-workshop, ~1 evening (parallel with Phase 2 tail)
**Build:**
- Window extraction per D9 (positives event-centered; negatives tiled; positive non-event windows tagged `ambiguous`).
- Track V: V-JEPA2 (16f windows / 32–64f clips, `last_hidden_state.mean(dim=1)`, `skip_predictor=True`, bf16+SDPA) + SigLIP2 (8 frames, L2-norm→mean→L2-norm). fp16 npy/parquet keyed by id.
- Track C: Qwen3-Embedding-4B over captions (`normalize_embeddings=True`).
- Track T: D-FINE detections (clips @1 fps, windows @4 fps) → token bags (§3.2) → `NgramVectorizer` → `InformationWeightTransformer()` (unsupervised — **never pass labels as y**); windows `.transform()` into clip-fitted column space.
- Optional: DINOv3 frame embeddings (ablation only).

**Verify:**
- [ ] Shapes/NaN audit across all matrices; embedding row count == manifest count per granularity.
- [ ] Sanity UMAP per track colored by scene/lighting/collision: visible structure exists (scene types separate somewhere); if a track is structureless noise, debug before workshop.
- [ ] Token vocab <~300 columns; top-IWT tokens are sensible (not stop-token garbage).

### Phase 4 — Maps & clustering — workshop day 1 AM
**Build:**
- 5-D clustering + 2-D display UMAPs per track (§3.3); ToponymyClusterer multi-layer fits; quick-look static plots (glasbey, colorblind_safe, `'Unlabelled': '#dddddd'`).
- Cluster coherence eyeball: per cluster, thumbnail contact sheet + metadata histogram.

**Verify:**
- [ ] ≥2 layers of clusters on ≥2 tracks; noise fraction <30% at base layer.
- [ ] Clusters visibly coherent (e.g., a highway-night region, an intersection region); pick 1–2 primary tracks to carry forward.

### Phase 5 — Naming — workshop day 1 PM
**Build:**
- Launch namer (§3.4). Custom Jinja `PROMPT_TEMPLATES` dict (BOTH `layer` + `disambiguate_topics` sections, keep `{"topic_name","topic_specificity"}` JSON contract): NHTSA 9-group/36-scenario vocabulary, D15 composite format, driving `object_description`/`corpus_description`.
- Differential metadata histograms injected via prompt post-edit between `make_prompts` and `name_topics` (D12).
- Disambiguation neutralized per D15. **Let Toponymy fit the clusterer** (pre-fit clusterer silently drops custom templates — trap P1).
- Name all layers on primary tracks; record `exemplar_indices` per cluster (feeds D16 + thumbnails).

**Verify:**
- [ ] 0 empty names (silent-failure mode of issue #155 — if any, thinking leaked through; fix server flag).
- [ ] ≥80% names parse as `"<group>: <qualifier>"`.
- [ ] Eyeball: names match contact sheets for 10 random clusters.

### Phase 6 — Evaluations — workshop day 2
**Build:**
- Eval (a) per D22: resampling harness; GLOSH (hdbscan on 5-D) + high-dim kNN control + LOF; clip vs window granularity × tracks; lift-over-chance table + rank-curve figures; label-noise sensitivity pass.
- Eval (b) per §3.5: alignment scoring, judge harness (temp 0, held-out per D16), 3× stability reruns (fresh instances per D13).

**Verify:**
- [ ] CIs reported from 30 resamples; conclusions stated either direction (a null clip-level result is a *finding*, per the original framing).
- [ ] Judge precision per cluster on the map as a confidence layer.
- [ ] Stability: membership-Jaccard-matched name agreement reported.

### Phase 7 — Interactive deliverable — workshop day 2 PM / day 3 AM
**Build:**
- `datamapplot.create_interactive_plot(map2d, *[layer.topic_name_vector ...], hover_text=captions, extra_point_data=df_with_thumbs_and_metadata, hover_text_html_template='<img src="{thumb}">...', enable_search=True, enable_topic_tree=True, colormaps={collision, weather, lighting, scene, outlier_score, judge_precision}, offline_mode=True, inline_data=True)`.
- **Both** `hover_text` AND `extra_point_data` must be passed (template silently ignored otherwise — issue #150). 1,500-point map: inline base64 thumbs (~10–16 MB). Window map (if shown): served thumbs via `python -m http.server`.

**Verify:**
- [ ] HTML renders fully in a disconnected browser (strip dangling maxcdn bootstrap/font-awesome links if first-paint stalls).
- [ ] Hover thumbnails, search, topic tree, all color layers work.

### Phase 8 — Stretch layers (any order, drop freely) — workshop day 3
- **BADAS-Open failure overlay** (D19): decode/embed the 1,344-clip test split; `badas==1.1.3` + `transformers>=4.53` pin + pre-downloaded weights; `model.predict()` @ window_stride=4 (~1 GPU-h); project test clips into the named map (UMAP `.transform()` + kNN topic assignment — EVoC has no predict); per-named-region FN/FP rates using `solution.csv` + official scorer.
- **EVoC comparison**: `EVoCClusterer` with `c.evoc.approx_n_clusters = None` (mandatory workaround — default silently collapses to one ~4-cluster layer); free `duplicates_` near-dup clip detection.
- **DINOv3 ablation**: appearance-clusters vs semantics-clusters figure.
- **RQ#11 mini-experiment**: conjunctive metadata-rule "names" for ablating the mundane background clusters.
- **Upstream PR** (D20) + any other rough edges found → issues/PRs with maintainers in the room.

### Phase 9 — Packaging & deck — day 3 PM
- Air-gap bundle check: `uv pip install --offline` rehearsal for BOTH venvs; `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` dry run; dmp cache zip; everything mirrored.
- Deck: mapping table, atlas demo, eval tables, RQ-per-slide, future work (temporal-mapper for scenario drift; V-JEPA 2.1 when transformers support lands).

---

## 5. Download manifest (~125 GB core, ~150 GB with optionals; disk budget OK, don't cross 97%)

| Asset | Size | Phase |
|---|---|---|
| `nexar-ai/nexar_collision_prediction` (dataset) | 31.4 GB | 1 |
| `Qwen/Qwen3.5-9B-FP8` | ~11 GB | 2 |
| `Qwen/Qwen3-VL-8B-Instruct-FP8` (fallback) | ~10 GB | 2 |
| `QuantTrio/Qwen3.5-9B-AWQ` (escape hatch) | ~7 GB | 2 |
| `Qwen/Qwen3.6-35B-A3B-FP8` | ~37 GB | 5 |
| `Qwen/Qwen3-32B-AWQ` (fallback) | ~19 GB | 5 |
| `google/siglip2-so400m-patch16-384` | ~3.5 GB | 3 |
| `facebook/vjepa2-vitl-fpc64-256` | ~2.5 GB | 3 |
| `Qwen/Qwen3-Embedding-4B` + `-0.6B` | ~9 GB | 3/5 |
| `ustc-community/dfine-medium-obj2coco` | ~0.2 GB | 3 |
| `facebook/dinov3-vitl16-pretrain-lvd1689m` (gated, optional) | ~1.5 GB | 8 |
| `nexar-ai/BADAS-Open` (stretch) | 4 GB | 8 |
| `openai/gpt-oss-20b` (optional 2nd judge) | ~13 GB | 6 |
| dmp_offline_cache zip, wheel caches (66 GB uv cache already warm) | — | 0 |

All weights verified loadable in Phase 0 (post-cutoff IDs are single-source research claims until proven).

## 6. Hour-1 smoke-test checklist (Phase 0 gate — every item is a known silent-failure or single-source claim)

1. `vllm serve --help` confirms: `--default-chat-template-kwargs`, `--language-model-only`, `--limit-mm-per-prompt.video`, `--mm-processor-cache-type`, `--reasoning-parser`. (Thinking-disable flag is highest-stakes: wrong flag → naming fails silently via Toponymy #155.)
2. Qwen3.5-9B-FP8 (block-128 FP8) loads + generates on one 4090 (Ada FP8 sits at the support cutoff; Marlin fallback = slower but correct; AWQ if broken).
3. One video request through the captioner with the **literal** caption JSON schema (xgrammar may reject string-length constraints) + explicit `fps`.
4. Qwen3.6-35B-A3B-FP8 TP=4 with `NCCL_P2P_DISABLE=1` serves; a `json_object` + `max_tokens=128` request through the **actual Toponymy OpenAINamer** returns a parseable name.
5. torchcodec imports + decodes a Nexar mp4 (system FFmpeg shared libs present).
6. V-JEPA2 + SigLIP2 + Qwen3-Embedding load in `venvs/main` under transformers 4.57.6.
7. DataMapPlot `offline_mode=True` HTML opens in a disconnected browser.
8. (Stretch) BADAS-Open loads with transformers ≥4.53; optional weight-equality check: `fpc64-256` backbone vs BADAS-Open backbone (decides whether "BADAS feature-space parity" can be claimed).

## 7. Risk register (top 6)

| Risk | Mitigation |
|---|---|
| RAID0 disk failure at 94% full | Daily git push of code/results; dataset/models re-downloadable; treat box as scratch. |
| Post-cutoff model IDs wrong / FP8-on-Ada kernel gap | Phase 0 verification + fallback chain per model (D1/D2). |
| Workshop air-gap stricter than expected | Phase 9 rehearsal; everything local by design; offline HTML tested disconnected. |
| Captions hallucinate events (ScVLM) | `unknown` enums, evidence-first schema, evals use Nexar labels only, spot-check gate in Phase 2. |
| Toponymy silent failures (templates dropped, empty names, stale shared state) | Traps encoded as D13–D15 + Phase 5 verify gates. |
| Scope creep | Tracks independently complete; Phase 8 entirely droppable; priority order in §2. |

## 8. Directory layout

```
stamp26/
  plan.md  research/        # this plan + research notes (git)
  src/                      # pipeline code (git)
  repos/                    # Tutte clones (editable installs)
  venvs/{vllm,main}/
  data/nexar/               # dataset
  cache/{frames,thumbs}/    # JPEG cache
  artifacts/{detections,captions,embeddings,maps,eval}/
  offline/                  # dmp cache zip, freeze files, manifests
```

## 9. References
Research notes: `research/{toponymy,clustering-viz,pipeline-examples,vectorizers-detection,vlm-captioning,naming-llm,embeddings,nexar-badas,env-setup}.md`. Problem sets: STAMP 2026 wiki (Google Doc, via `repos/stamp2026`). Dataset paper: arXiv 2503.03848. BADAS: arXiv 2510.14876, 2604.05767. NHTSA pre-crash typology: DOT HS 812 745 (2019, 36 scenarios / 9 groups; verbatim check pending — see Phase 5).
