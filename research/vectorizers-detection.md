# Research note: vectorizers library (bag-of-detected-objects baseline) + object detector choice

Date: 2026-06-09. Sources: local clone `/home/b3ali/projects/stamp26/repos/vectorizers` (v0.2.2, last commit 6f723e9 2026-05-10, matches latest PyPI release 0.2.2), local Tutte tutorial `/home/b3ali/projects/stamp26/repos/tutorials/1-recipes-bag-of-words.ipynb`, web verification of detector versions/licenses (June 2026).

---

# Part A — `vectorizers` library (TutteInstitute), exact API from source

Install: `uv pip install vectorizers` (PyPI 0.2.2 == local clone). Deps per `pyproject.toml`: numpy, pandas, scipy, scikit-learn, numba, pynndescent, dask. (`requirements.txt` additionally lists `pomegranate` but it is NOT in pyproject dependencies; pip install does not pull it. Numba supports Python 3.12 — fine on our box.)

Top-level exports (`vectorizers/__init__.py`): `TokenCooccurrenceVectorizer`, `TimedTokenCooccurrenceVectorizer`, `NgramCooccurrenceVectorizer`, `MultiSetCooccurrenceVectorizer`, `DistributionVectorizer`, `HistogramVectorizer`, `SkipgramVectorizer`, `NgramVectorizer`, `KDEVectorizer`, `LabelledTreeCooccurrenceVectorizer`, `WassersteinVectorizer`, `SinkhornVectorizer`, `ApproximateWassersteinVectorizer`, `EdgeListVectorizer`, `SignatureVectorizer`, plus `vectorizers.transformers`: `InformationWeightTransformer`, `CategoricalColumnTransformer`, `RowDenoisingTransformer`, `SlidingWindowTransformer`, `CountFeatureCompressionTransformer`.

## A.1 `NgramVectorizer` — the count-matrix builder we need

File: `vectorizers/ngram_vectorizer.py` (class at line 64).

With `ngram_size=1` (default) this is exactly "list of token bags -> sparse count matrix". Input: a list of token sequences (any hashable token type, must be homogeneous — use `vectorizers.cast_tokens_to_strings` if mixed). Order inside a sequence is irrelevant for `ngram_size=1`, so multisets-as-lists work directly.

Constructor (all optional):

```python
NgramVectorizer(
    ngram_size=1,                 # 1 => plain bag-of-tokens
    ngram_behaviour="exact",      # or "subgrams" (all sizes <= ngram_size)
    ngram_dictionary=None,        # fixed ngram->col mapping
    token_dictionary=None,        # fixed token->index mapping (use to freeze vocab across runs)
    max_unique_tokens=None,       # prune vocab to top-k most frequent
    min_occurrences=None, max_occurrences=None,
    min_frequency=None, max_frequency=None,
    min_document_occurrences=None, max_document_occurrences=None,
    min_document_frequency=None, max_document_frequency=None,
    excluded_tokens=None, excluded_token_regex=None,
    mask_string=None, nullify_mask=False,
    validate_data=True,
)
```

- `fit(X)` / `fit_transform(X)` -> `scipy.sparse.csr_matrix`, dtype float32, shape `(n_docs, n_vocab)`.
- Fitted attributes: `column_label_dictionary_` (token -> column index), `column_index_dictionary_` (column index -> token), `_train_matrix` (the CSR train counts), `_token_frequencies_`.
- `transform(X)` reuses the fitted vocabulary (unknown tokens silently dropped) — important for the "fit on full clips, transform event windows" trick so both live in the same column space.
- `__add__` lets you merge two fitted vectorizers (unigram only).

This is exactly what the Tutte recipes tutorial uses: `vz.NgramVectorizer().fit(data["ingredients"])` (cell 33 of `1-recipes-bag-of-words.ipynb`).

For our small vocabulary (~200 tokens, see Part B token schema) no pruning args are needed; optionally `min_document_occurrences=3` to drop freak tokens.

## A.2 `InformationWeightTransformer` — confirmed: KL-divergence-vs-row-marginal ("centroid") column reweighting

File: `vectorizers/transformers/info_weight.py`.

**Math (read directly from source).** Let `X` be the (n_docs x n_tokens) count matrix.

1. Baseline: `baseline_counts = X.sum(axis=1)` (per-document total token mass); `baseline_probabilities = baseline_counts / baseline_counts.sum()` — i.e. the distribution over *documents* proportional to document size. This is the "centroid"/marginal model: under it, every token is sprinkled over documents exactly in proportion to document length.
2. For each column (token) `j`, form the observed distribution over documents with a Bayesian pseudo-count prior: `p_obs(i) = (X[i,j] + prior_strength * baseline_probabilities[i]) / (sum_i X[i,j] + prior_strength)`.
3. Weight of column `j` = `KL(p_obs || baseline_probabilities) = sum_i p_obs(i) * log(p_obs(i)/baseline(i))` — the information gained by moving from the baseline (centroid) model to the column's observed distribution. Tokens distributed like "background mass" (stop-word-like: e.g. `car|center|far` present everywhere) get weight ~ 0; tokens concentrated in specific documents get high weight.
   - `approx_prior=True` (class default) approximates the zero-count rows' contribution in aggregate instead of looping over all rows (`column_kl_divergence_approx_prior`); exact mode loops all rows. Approx is fine and fast (numba `prange` over columns).
4. Post-processing in `fit` (unsupervised path): `w /= mean(w)`; `w = max(w, 0)`; `w = w ** weight_power` (default **2.0** — weights are squared, sharpening contrast).
5. `transform(X)` = `X @ scipy.sparse.diags(information_weights_)` — pure column scaling, stays sparse and non-negative.

Class defaults (note they differ from the bare `information_weight` function defaults): `InformationWeightTransformer(prior_strength=1e-4, approx_prior=True, weight_power=2.0, supervision_weight=0.95)`.

**Supervised mode**: if `fit(X, y)` is given labels, it additionally computes per-column KL of the token's distribution *over classes* vs class-mass baseline (`supervised_column_kl`) and combines: `w = w_unsup^((1-sw)*p) * w_sup^(sw*p)` with `sw=supervision_weight=0.95`, `p=weight_power`. **Do NOT pass `y=collision_label` for evaluation (a)** — that leaks the label into the representation. It could be a fun "supervised map" side-demo only.

## A.3 Hellinger — how the Tutte folks actually compute it

- `vectorizers/distances.py` line 8 `hellinger(x, y)` (dense) and line 378 `sparse_hellinger(ind1, data1, ind2, data2)`: both compute `sqrt(1 - BC)` where the Bhattacharyya coefficient `BC = sum_i sqrt(x_i * y_i) / sqrt(||x||_1 ||y||_1)`. **L1 normalization happens inside the metric** — you do not need to normalize rows first.
- umap-learn ships the same metric under the name `"hellinger"` for dense AND sparse inputs (verified: `umap/sparse.py` line 608, `sparse_named_distances["hellinger"] = sparse_hellinger`; current umap-learn is 0.5.12).
- Canonical Tutte usage (tutorial `1-recipes-bag-of-words.ipynb`, cells 59 and 70 — this is the institute's "steak, blé d'Inde, patates" recipe):

```python
weights_iwt = vzt.InformationWeightTransformer().fit_transform(weights)   # weights = sparse counts
weights_u2  = umap.UMAP(metric="hellinger", unique=True).fit_transform(weights_iwt)
```

  They feed the **raw info-weighted sparse counts** straight into `UMAP(metric="hellinger", unique=True)`. No manual L1 step. `unique=True` dedups identical rows (recommended — bags of small vocab will have duplicates).
- **The sqrt identity, confirmed**: classical Hellinger is `H(p,q) = (1/sqrt(2)) * ||sqrt(p) - sqrt(q)||_2` for L1-normalized p, q. Since `||sqrt(p)-sqrt(q)||_2^2 = 2 - 2*BC`, the umap/vectorizers form `sqrt(1 - BC) = (1/sqrt(2)) * euclidean(sqrt(p), sqrt(q))`. The two differ only by the constant `sqrt(2)`, so kNN graphs (hence UMAP/HDBSCAN results) are identical. The "euclidean on sqrt of L1-normalized rows" trick is therefore valid — only needed if you must use a euclidean-only ANN backend; otherwise just use `metric="hellinger"`.
- Caveat: hellinger NaNs on negative entries. IWT output is non-negative (counts x non-negative weights), so fine.
- **Flag vs project plan**: the plan's "info-weight -> l1 normalize -> Hellinger" has a redundant step. Harmless if done, but the canonical pipeline is `counts -> IWT -> UMAP(metric="hellinger")`.

## A.4 `TokenCooccurrenceVectorizer` — what it is and when we'd use it

File: `vectorizers/token_cooccurrence_vectorizer.py` (subclasses `BaseCooccurrenceVectorizer`, 777 lines).

It does **not** produce document vectors. It produces a **token x token** directed co-occurrence matrix from windows sliding over token *sequences*, then runs `n_iter` EM re-estimation passes — i.e. it learns *token representations* (rows = tokens, columns = before/after context tokens). Key init params beyond the shared vocab-pruning ones:

```python
TokenCooccurrenceVectorizer(
    window_functions="fixed", kernel_functions="flat",
    window_args=None, kernel_args=None, window_radii=5,
    mix_weights=None, window_orientations="directional",
    n_threads=1, normalize_windows=True, n_iter=0, epsilon=0,
    coo_initial_memory="0.5 GiB",
)
```

Relevance for us: optional. If we order tokens temporally within a clip (frame by frame), `TokenCooccurrenceVectorizer` gives us learned **token embeddings** ("what co-occurs with `person|center|near`?") which is exactly the `vectors=` input the WassersteinVectorizer needs (A.5). For the 2-3 day scope this is the stretch variant, not the baseline. (`MultiSetCooccurrenceVectorizer` is the multiset analogue; `TimedTokenCooccurrenceVectorizer` adds timestamps — could use real frame times.)

## A.5 `WassersteinVectorizer` / `SinkhornVectorizer` — distributions over token embeddings

File: `vectorizers/linear_optimal_transport.py` (classes at lines 1374, 2215, 2566).

Purpose: turn rows-of-a-count-matrix interpreted as *distributions over a metric space of token vectors* into dense vectors whose euclidean/cosine distance approximates Wasserstein (word-mover) distance, via Linear Optimal Transport + SVD compression.

```python
WassersteinVectorizer(
    method="LOT_exact",          # or "LOT_sinkhorn", "HeuristicLinearAlgebra" (fast, heuristic)
    input_method="spmatrix",     # or "lil", "generator"
    n_components=128, reference_size=None, reference_scale=0.01,
    metric="cosine",             # ground metric over token vectors (pynndescent named or callable)
    memory_size="2G", max_distribution_size=256,
    n_svd_iter=10, random_state=None, ...
)
wv = WassersteinVectorizer(random_state=42)
doc_vectors = wv.fit_transform(count_matrix, vectors=token_embedding_matrix)
# count_matrix: (n_docs, n_vocab) sparse; vectors: (n_vocab, dim) ndarray
```

- `fit/fit_transform(X, vectors=...)` — `vectors` is REQUIRED for spmatrix input: one embedding per vocabulary column.
- `max_distribution_size=256` truncates distributions with more than 256 *distinct* support tokens — our bags have support << 256 (vocab ~200), so default is fine.
- `SinkhornVectorizer` = same idea over entropic-regularized OT; more scalable, slight quality loss; defaults `n_components=128, reference_scale=0.1, metric="cosine", chunk_size=32, n_svd_iter=7`.
- `ApproximateWassersteinVectorizer` = the pure linear-algebra heuristic (fastest).
- For our project: token embeddings can come from (i) `TokenCooccurrenceVectorizer` on temporally-ordered tokens, or (ii) a sentence-embedding model applied to verbalized tokens ("a nearby pedestrian in the center of the view"). Output is dense -> use `UMAP(metric="cosine")`, NOT hellinger (it's no longer count data).

## A.6 Other count-matrix utilities worth knowing

- `vectorizers.transformers.CategoricalColumnTransformer(object_column_name, descriptor_column_name, include_column_name=True)` — groupby a DataFrame and emit one bag of categorical values per object. With `include_column_name=True` values come out as `"weather:rain"`-style strings; an easy way to turn the Nexar metadata table into metadata token bags to concatenate with detection bags.
- `vectorizers.transformers.CountFeatureCompressionTransformer(n_components=128, rescaling_power=0.5)` — sqrt-rescale (`rescaling_power=0.5` == the Hellinger sqrt trick) + randomized SVD -> dense vectors suitable for cosine. This is the Tutte-blessed dense alternative when you want vectors instead of a metric (e.g. to feed EVoC or to concatenate with other features).
- `vectorizers.transformers.RowDenoisingTransformer` — (EM-based row de-noising of count matrices, optional).
- `vectorizers.utils.cast_tokens_to_strings(data)` — coerce mixed token types.
- `vectorizers.utils.sparse_collapse(matrix, labels)` — collapse rows of a sparse matrix by label (e.g. sum event-window bags up to clip-level bags).
- Lower level: `vectorizers.preprocessing.preprocess_token_sequences`, `construct_token_dictionary_and_frequency`, `prune_token_dictionary` (used internally by all vectorizers).

## A.7 Full code sketch (per-clip multisets -> UMAP)

```python
import numpy as np, scipy.sparse
import vectorizers as vz
import vectorizers.transformers as vzt
import umap

# ---- 1. token bags --------------------------------------------------------
# clip_tokens: list[list[str]], one list per clip (or per event window), e.g.
# ["car|center|near", "car|left|mid", "person|right|near", "traffic:heavy",
#  "weather:rain", "weather:rain", ..., "lighting:night", ...]
# (metadata tokens repeated to give them ~5% of bag mass; see Part B schema)

ngram = vz.NgramVectorizer(min_document_occurrences=3)
X_clip = ngram.fit_transform(clip_tokens)            # CSR (n_clips, n_vocab) float32
vocab = ngram.column_index_dictionary_               # col -> token string

# event windows transformed into the SAME column space:
X_event = ngram.transform(event_window_tokens)

# ---- 2. information weighting (KL vs centroid) -----------------------------
iwt = vzt.InformationWeightTransformer()             # prior_strength=1e-4, approx_prior=True, weight_power=2.0
X_clip_w  = iwt.fit_transform(X_clip)                # fit on clips...
X_event_w = iwt.transform(X_event)                   # ...reuse weights on windows
# (do NOT pass y=collision labels: leaks eval target)

# ---- 3. UMAP with Hellinger (no manual L1 normalize needed) ----------------
um = umap.UMAP(metric="hellinger", unique=True, n_neighbors=15,
               n_components=5, random_state=42)      # 5d for clustering; 2d for the map
Z_clip = um.fit_transform(X_clip_w)
# -> fast_hdbscan / ToponymyClusterer / DataMapPlot downstream

# ---- 4. optional Wasserstein variant ---------------------------------------
# token_vecs = (n_vocab, d) embeddings of verbalized tokens, row order = ngram column order
# wv = vz.WassersteinVectorizer(random_state=42)
# V = wv.fit_transform(X_clip, vectors=token_vecs)   # dense (n_clips, 128)
# Z = umap.UMAP(metric="cosine").fit_transform(V)
```

Interpretability bonus: after clustering, per-cluster top tokens by summed info-weighted mass (`X_clip_w[cluster_rows].sum(0)` vs corpus mean) give Toponymy exactly the kind of distinguishing keyphrases it uses for text.

---

# Part B — object detector for ~60k dashcam frames (verified June 2026)

Workload: 1,500 clips x 40 s sampled at 1 fps = **60k frames** (1280x720). This is small: any modern detector finishes in minutes-to-tens-of-minutes on ONE RTX 4090; the choice is driven by **license** and **API ergonomics**, not speed.

## B.1 Candidate comparison (licenses verified via GitHub API SPDX, 2026-06-09)

| Model | License (verified) | COCO AP (50:95) | Latency (T4 TRT fp16 bs1) | Install / API | Notes |
|---|---|---|---|---|---|
| Ultralytics YOLO11 / YOLO12 / **YOLO26** (Jan 2026, latest) | **AGPL-3.0** (or paid Enterprise) | YOLO26: 40.9–57.5 across n–x | 1.7–11.8 ms | `pip install ultralytics`, nicest API | License is the problem, not quality |
| RT-DETRv2 (lyuwenyu/RT-DETR) | Apache-2.0 | ~47.9–54.3 | ~5–9 ms | also in HF `transformers` (`PekingU/rtdetr_v2_r50vd`) | solid, slightly dated |
| RT-DETRv3 (clxia12/RT-DETRv3) | Apache-2.0 | ~48–54+ | similar | research repo, clunkier | RT-DETRv4 paper (arXiv 2510.25257) newer still |
| **D-FINE** (Peterande/D-FINE; ICLR'25 spotlight) | Apache-2.0 | 48.5 (S) → 55.8 (X); **obj2coco** ckpts stronger/robuster | ~3–9 ms | **native in HF `transformers`** (`DFineForObjectDetection`, since v4.52, still in v5.x); ckpts `ustc-community/dfine-{nano,small,medium,large,xlarge}-{coco,obj2coco,obj365}` | best transformers-native option |
| **RF-DETR** (roboflow/rf-detr; ICLR 2026) | **Apache-2.0 for Nano/Small/Medium/Large + all code**; XL/2XL weights are PML 1.0 (avoid) | Nano 48.4 / Small 53.0 / **Medium 54.7** / Large 56.5 | Nano 2.3 / Small 3.5 / Medium 4.4 / Large 6.8 ms | `pip install rfdetr` (1.7.1, Python>=3.10, torch>=2.2, transformers>=5.1<6); `model.predict(list_of_images)` batched, returns `supervision.Detections` | DINOv2 backbone -> best domain robustness (RF100-VL); current accuracy/latency SOTA |
| YOLO-World (AILab-CVC) | **GPL-3.0** | open-vocab | real-time | mmyolo-based | license + setup pain |
| YOLOE (THU-MIG) | **AGPL-3.0** | open-vocab, beats YOLO-World | 100–300 FPS T4 | ultralytics-based | license pain |
| **OWLv2** (`google/owlv2-base-patch16-ensemble`) | Apache-2.0 | open-vocab (text queries) | NOT real-time: order 10–25 img/s on a 4090 fp16 batched (estimate) | native in HF `transformers` | feasible for a *subset* pass |
| Grounding-DINO / mm-GDINO | Apache-2.0 | open-vocab | a few img/s | HF transformers | slower than OWLv2; skip |

**AGPL verdict for the workshop**: Ultralytics (YOLO11/12/26, RTDETR-via-ultralytics, YOLOE) is AGPL-3.0. For a Tutte Institute (Government of Canada) workshop where code will likely be shared, AGPL contaminates the repo (entire derivative work must be AGPL; network use counts as distribution) and clashes with the BSD/Apache licensing of the whole TutteInstitute stack; many gov/enterprise policies ban AGPL outright. Since Apache-licensed RF-DETR/D-FINE now *match or beat* YOLO26 at comparable latency, there is **no reason to accept AGPL**. Avoid Ultralytics entirely.

**COCO class coverage for driving**: the relevant COCO-80 classes are `person, bicycle, car, motorcycle, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench` plus animals (`dog, cat, horse, ...`) and pedestrian-attribute objects (`backpack, umbrella, suitcase`). That is *enough for interesting tokens* (vehicles by type, VRUs, traffic control, animals) and maps onto much of the NHTSA pre-crash typology (rear-end with vehicle, pedestrian/pedalcyclist, animal). It misses: emergency vehicles (subsumed into car/truck), construction cones/zones, generic traffic signs, crosswalk markings, road debris. Plan: COCO tokens primary; optional open-vocab enrichment pass for the missing concepts (below).

## B.2 Recommendation: RF-DETR Medium primary, D-FINE obj2coco fallback, OWLv2 optional enrichment

**Primary: RF-DETR Medium** (`rfdetr==1.7.1`, Apache 2.0, 54.7 AP, 576x576 input, DINOv2 backbone good under dashcam domain shift):

```python
import torch
from rfdetr import RFDETRMedium

model = RFDETRMedium()                                   # downloads COCO checkpoint
model.optimize_for_inference(compile=True, batch_size=32, dtype=torch.float16)

dets_list = model.predict(list_of_rgb_numpy_frames,      # batched: list[np.ndarray] OK
                          threshold=0.5,
                          include_source_image=False)    # saves memory
for dets in dets_list:                                   # supervision.Detections
    names = dets.data["class_name"]                      # USE THIS, not COCO_CLASSES[class_id]
    boxes = dets.xyxy; conf = dets.confidence
```

API gotcha (from `src/rfdetr/detr.py` docstring): for pretrained COCO checkpoints `class_id` holds **raw sparse COCO category IDs (1–90)**, not 0-based indices — always read `detections.data["class_name"]`. `predict()` accepts str path / PIL / np.ndarray / torch.Tensor or a list thereof (RGB). Throughput: Medium is 4.4 ms on a T4 TRT; on a 4090 with fp16 + batch 32 expect order 150–400 img/s in plain PyTorch -> 60k frames in **<10 min on one GPU** (shard the 4 GPUs by clip if you like; embarrassingly parallel). Stay on Nano–Large only (Apache); XL/2XL are PML-licensed.

**Fallback (zero extra deps)**: D-FINE via transformers — `DFineForObjectDetection.from_pretrained("ustc-community/dfine-medium-obj2coco")` + `AutoImageProcessor`, Apache 2.0; the `obj2coco` (Objects365-pretrained, COCO-finetuned) checkpoints are the robust pick. Use if rfdetr's `transformers>=5.1,<6` pin fights the rest of the env.

**Optional open-vocab enrichment (event windows only)**: OWLv2 `google/owlv2-base-patch16-ensemble` (Apache, transformers-native) with a driving prompt vocabulary, e.g. `["emergency vehicle with flashing lights", "police car", "ambulance", "traffic cone", "construction zone", "pedestrian crossing the road", "deer on road", "fallen cargo / road debris", "school bus", "tow truck"]`. Run it ONLY on event-window frames (~1,500 clips x 32 frames = 48k, or just collision-positive clips ~24k frames): at an estimated 10–25 img/s on one 4090 that is ~0.5–1.5 h, or ~15–25 min across 4 GPUs. Emit tokens only at high confidence (>=0.3 OWLv2 score) as extra `ov:emergency_vehicle`-style tokens. This is a stretch goal — the COCO pipeline must not depend on it.

## B.3 Token schema (recommended)

**Frame sampling**: full clip at **1 fps** (~40 frames/clip, 60k frames total); event-centered window at **4 fps over ±4 s** around `time_of_event` (32 frames; for the prediction-flavored variant, center on `time_of_alert` instead). Decode with PyAV/decord, sample by timestamp not frame index (VFR-safe).

**Per-detection token** = `"{cls}|{zone}|{range}"`, conf >= 0.5:
- `cls`: detected class name; keep only driving-relevant classes `{person, bicycle, car, motorcycle, bus, train, truck, traffic light, stop sign, fire hydrant, parking meter, bench, dog, cat, horse}` (others -> drop or `other_object`).
- `zone` in `{left, center, right}` by bbox center x in thirds of frame width (ego-relevant: `center` ~ in our path).
- `range` in `{near, mid, far}` by bbox height fraction of frame height: near > 0.25, mid 0.08–0.25, far < 0.08 (monotone proxy for distance in ego view).
- Cap per (class, zone, range) count at 10 per frame (parked-car saturation guard).

**Per-frame scene tokens** (1 per frame each): `traffic:{none|light|moderate|heavy}` from vehicle count bins (0 / 1–3 / 4–8 / 9+); `vru:present` if any person/bicycle/motorcycle.

**Metadata tokens** (from Nexar annotations, `CategoricalColumnTransformer(include_column_name=True)` style): `weather:rain`, `lighting:night`, `scene:urban`, ... A single occurrence is invisible under Hellinger next to hundreds of detection counts, so inject each metadata token with multiplicity ~= `ceil(0.05 * bag_total)` (~5% of mass per field). **Run a with/without-metadata ablation** — metadata tokens can trivially dominate cluster naming ("rainy night clips") and mask the behavioral structure we actually want Toponymy to discover.

Resulting vocabulary: ~15 classes x 3 zones x 3 ranges ~ 135 + ~6 scene/congestion + ~12 metadata + optional ~10 `ov:*` tokens => **< 200 columns**. Ideal regime for `NgramVectorizer -> InformationWeightTransformer -> UMAP(metric="hellinger", unique=True)`.

The bag is a *time-marginalized* representation (no dynamics). Expected/honest failure mode for eval (a): collision vs. normal clips with identical object census look the same; that is precisely the contrast against SigLIP2/V-JEPA2 embeddings the project wants to show. Event-window bags (4 fps) partially recover dynamics via the near/center token surge just before impact.

## B.4 Version pins (June 2026)

```
vectorizers==0.2.2          # PyPI == TutteInstitute main
umap-learn==0.5.12          # sparse "hellinger" metric built-in
rfdetr==1.7.1               # Apache 2.0; needs torch>=2.2, transformers>=5.1,<6, supervision
transformers==5.10.x        # also covers D-FINE (DFineForObjectDetection) and OWLv2
# torch: any recent cu12x wheel (e.g. 2.7/2.8 + cu126/cu128) runs fine on driver 590 / 4090
```

## Key risks / contradictions with the project plan

1. **Plan wording "info-weight -> l1 normalize -> Hellinger"**: the explicit L1 step is redundant — both `vectorizers.distances.hellinger` and umap's `"hellinger"` L1-normalize inside the metric; the canonical Tutte recipe is `counts -> InformationWeightTransformer -> UMAP(metric="hellinger", unique=True)` directly on the sparse matrix.
2. **Do not use Ultralytics/YOLOE/YOLO-World** (AGPL-3.0 / GPL-3.0) for a shareable workshop artifact; Apache alternatives (RF-DETR, D-FINE) are now SOTA anyway.
3. RF-DETR **XL/2XL weights are PML 1.0, not Apache** — stick to Nano–Large.
4. RF-DETR class-ID gotcha: COCO checkpoints return sparse category IDs 1–90; read `detections.data["class_name"]`.
5. `rfdetr` pins `transformers>=5.1,<6` — coordinate with the VLM/embedding env (or isolate the detector in its own uv venv writing tokens to parquet).
6. Don't pass collision labels to `InformationWeightTransformer.fit(X, y)` — supervised weighting leaks the eval-(a) target.
7. OWLv2 throughput figures above are estimates, not benchmarks — time-box a 1k-frame pilot before committing to the open-vocab pass.

## Links

- vectorizers: https://github.com/TutteInstitute/vectorizers (PyPI 0.2.2)
- Tutte bag-of-words recipe tutorial: https://github.com/TutteInstitute/tutorials (notebook `1-recipes-bag-of-words.ipynb`)
- RF-DETR: https://github.com/roboflow/rf-detr , docs https://rfdetr.roboflow.com/ , Apache LICENSE: https://github.com/roboflow/rf-detr/blob/develop/LICENSE , release blog https://blog.roboflow.com/rf-detr/
- D-FINE: https://github.com/Peterande/D-FINE , HF docs https://huggingface.co/docs/transformers/model_doc/d_fine , ckpt https://huggingface.co/ustc-community/dfine-medium-obj2coco
- RT-DETR(v2): https://github.com/lyuwenyu/RT-DETR ; RT-DETRv3: https://github.com/clxia12/RT-DETRv3 ; RT-DETRv4 paper: https://arxiv.org/abs/2510.25257
- YOLO26 (AGPL): https://docs.ultralytics.com/models/yolo26 ; Ultralytics license: https://www.ultralytics.com/license
- YOLOE (AGPL): https://github.com/THU-MIG/yoloe ; YOLO-World (GPL): https://github.com/AILab-CVC/YOLO-World
- OWLv2: https://huggingface.co/google/owlv2-base-patch16-ensemble
- umap sparse hellinger: https://github.com/lmcinnes/umap/blob/master/umap/sparse.py (line ~608)
