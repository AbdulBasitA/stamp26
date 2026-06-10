# Tutte clustering + visualization stack deep-dive (evoc, fast_hdbscan, datamapplot, temporal-mapper, thisnotthat, scalar)

Source of truth: local clones at `/home/b3ali/projects/stamp26/repos/` (all verified against PyPI on 2026-06-09).

| package | local clone version (commit date) | latest PyPI | match? |
|---|---|---|---|
| evoc | 0.3.1 (2026-05-10) | 0.3.1 (released 2026-03-27) | yes (clone slightly ahead of wheel) |
| fast_hdbscan | 0.3.2 (2026-05-10) | fast-hdbscan 0.3.2 | yes |
| datamapplot | 0.7.3 (2026-05-31) | 0.7.3 | yes |
| toponymy | 0.5.2 | 0.5.2 | yes |
| temporal-mapper | 1.3.0 | 1.3.0 | yes |
| thisnotthat | — | 0.4.2 | — |
| hdbscan (classic, scikit-learn-contrib) | n/a | 0.8.44 | needed for GLOSH |
| umap-learn | n/a | 0.5.12 | — |

---

## 1. EVoC (Embedding Vector Oriented Clustering)

**What it is** (`evoc/evoc/clustering.py`): a single-step replacement for UMAP→HDBSCAN *specialized to high-dim embedding vectors*. Internally it:
1. builds a kNN graph of the embeddings (NN-descent; **cosine for float input, quantized-cosine for int8, bitwise-Jaccard for uint8** — metric is chosen by dtype, not by parameter; see `EVoC.fit_predict` docstring at clustering.py:574),
2. computes a **UMAP-like node embedding into ~15 dimensions** (not 2D! `node_embedding_dim = min(max(n_neighbors // 4, 4), 15)` by default), initialized by label propagation,
3. runs HDBSCAN-style density clustering (KD-tree + parallel Borůvka MST) on that node embedding,
4. extracts **multi-resolution cluster layers** by finding diverse persistence peaks over `min_cluster_size` — this is the PLSCAN algorithm (Bot, McInnes, Aerts, "Persistent Multiscale Density-based Clustering", arXiv:2512.16558, which the README says to cite).

**When the Tutte folks recommend it over UMAP→HDBSCAN**: per README — "takes all the good parts of the combination of UMAP + HDBSCAN for embedding clustering ... and removes all the time-consuming parts"; for *clustering only* of large embedding sets on CPU, fewer hyperparameters, automatic #clusters, multi-granularity, quantized-embedding support, duplicate detection. Caveats from the maintainer himself:
- **EVoC does not produce a 2D layout** — you still need UMAP for the DataMapPlot coordinates (lmcinnes in evoc issue #19: "Note that EVoC does not produce a 2D layout, so you can't extract it. On the other hand you can reduce to 2D with UMAP and then use toponymy to handle the rest").
- Because it clusters in ~15D rather than a packed 2D/5D UMAP space, it tends to produce **more noise points** than UMAP→HDBSCAN (issue #11, lmcinnes: "Since it isn't packing data together as tightly in a low dim space you can end up with more outliers"). Counter with `noise_level=0.0`.
- README: "very much an early beta ... things can and will break".

**API** (sklearn-compatible, `evoc.EVoC`):
```python
import evoc
clusterer = evoc.EVoC(
    noise_level=0.5,            # 0.0 = cluster more data, 1.0 = stricter clusters/more noise
    base_min_cluster_size=5,    # finest-layer min cluster size
    base_n_clusters=None,       # OR target #clusters for the finest layer
    approx_n_clusters=None,     # if set -> SINGLE layer only, no hierarchy!
    n_neighbors=15, min_samples=5, n_epochs=50,
    node_embedding_init="label_prop", node_embedding_dim=None,
    neighbor_scale=1.0, random_state=None,
    min_similarity_threshold=0.2,  # Jaccard threshold for layer diversity
    max_layers=10,
)
labels = clusterer.fit_predict(X)            # X float32 -> cosine; labels = best-persistence layer, NOT finest!
clusterer.cluster_layers_                     # list[np.ndarray], FINEST FIRST, coarser later
clusterer.membership_strength_layers_         # per-layer membership strengths in [0,1]
clusterer.persistence_scores_                 # per-layer persistence; argmax picks labels_
clusterer.cluster_tree_                       # dict {(layer, cluster): [(child_layer, child_cluster), ...]}
clusterer.nn_inds_, clusterer.nn_dists_       # kNN graph (reusable!)
clusterer.duplicates_                         # set of near-duplicate index pairs
```
**Gotcha** (issue #8): `fit_predict`/`labels_` returns the layer with max persistence, *not* the most granular layer. Use `cluster_layers_[0]` explicitly if you want the finest.

**Toponymy integration — YES, with a trap.** `toponymy/clustering.py:600` defines `EVoCClusterer(Clusterer)` (only exported if `import evoc` succeeds). It wraps `evoc.EVoC`, ignores `clusterable_vectors`, and clusters the **full-dim `embedding_vectors`** directly; it then wraps `evoc.cluster_layers_` into Toponymy `ClusterLayer` objects + `build_cluster_tree`, i.e. fully compatible with `Toponymy.fit`.
**TRAP (verified in source, clustering.py:643-651):** the wrapper passes `approx_n_clusters=min_clusters` (default 4) into `evoc.EVoC`. Per evoc semantics (`clustering.py:372-382`), a non-None `approx_n_clusters` short-circuits layer building and returns **exactly one layer** of ~`min_clusters` clusters. So `EVoCClusterer` as shipped in toponymy 0.5.2 silently throws away EVoC's multi-resolution capability. Workaround (params are read at fit time):
```python
from toponymy.clustering import EVoCClusterer
c = EVoCClusterer(base_min_cluster_size=10, noise_level=0.3)
c.evoc.approx_n_clusters = None      # restore multi-layer behaviour
c.fit(clusterable_vectors=umap5d, embedding_vectors=emb, layer_class=ClusterLayerText, ...)
```
Also note: `toponymy/tests/test_clustering.py:24` claims "evoc 0.3.1 is incompatible with fast_hdbscan >= 0.3.2 due to NumbaKDTree signature changes" and skips EVoC tests whenever fast_hdbscan >= 0.3. **This comment is stale**: I downloaded the evoc 0.3.1 wheel from PyPI — it vendors its own `numba_kdtree`/`boruvka`/`cluster_trees` and has **zero** `fast_hdbscan` imports (`requires_dist = numpy, scikit-learn, numba, tqdm`). evoc 0.3.1 + fast-hdbscan 0.3.2 coexist fine; still do a 30-second smoke test at env-build time.

**Suitable for 1,500–30,000 points?** Yes — these sizes are trivial for it (README example is 100k×1024 on CPU; the one open scale bug, issue #30 `IndexError: pop from empty list` in recursive label-prop init, was hit at 5.7M points and is worked around with `node_embedding_init=None`). At 1,500 points the time advantage over UMAP→HDBSCAN is irrelevant (~seconds either way); the real decision is quality/control. For the workshop: **primary pipeline = UMAP(5D)→ToponymyClusterer (deterministic-ish, gives us the 2D map anyway, and clusterable space == visual space neighborhoods); run EVoC on the raw embeddings as a comparison condition** (it is exactly the "does Tutte tooling generalize" story, and `duplicates_` is a free near-duplicate-clip detector for Nexar).

**Multi-resolution layers compatible with ToponymyClusterer?** Yes — both produce `cluster_layers_` (finest→coarsest label vectors) + `cluster_tree_` dict keyed `(layer, cluster)`; `Toponymy.fit` accepts any `Clusterer` with these attributes (toponymy.py:207-214). Note also **`fast_hdbscan.PLSCAN`** (fast_hdbscan 0.3.2, hdbscan.py:890) now exposes the same multiscale extraction on *reduced* vectors (`cluster_layers_`, `membership_strength_layers_`, `layer_persistence_scores_`, `cluster_tree_`, plus arbitrary pynndescent metrics incl. cosine/hellinger and cannot-link constraints) — a third option if we want PLSCAN layers in UMAP space without Toponymy's quantile-driven layering.

---

## 2. GLOSH outlier scores — who has them, and what to use

**Library support (verified by grep):**
- `fast_hdbscan` 0.3.2: **does NOT implement GLOSH**. No `outlier_scores_` anywhere; the only "outlier" hits are infinite-data-point bookkeeping (`hdbscan.py:119ff`). Attributes are `labels_`, `probabilities_`, `_condensed_tree`.
- `evoc`: no outlier scores either (only `membership_strengths_`, which is a *cluster membership* not an outlier score; `1 - membership_strength` is a weak proxy and is 0 for all noise points).
- **Classic `hdbscan` (scikit-learn-contrib, PyPI `hdbscan==0.8.44`): exposes GLOSH as `clusterer.outlier_scores_`** after `.fit()` (docs: https://hdbscan.readthedocs.io/en/latest/outlier_detection.html — "values close to 1 indicate strong outliers", threshold by quantile). `sklearn.cluster.HDBSCAN` does *not* expose GLOSH.
- Toponymy/DataMapPlot do not compute outlier scores.

**High-dim vs UMAP space — McInnes et al. guidance:**
- UMAP clustering doc (https://umap-learn.readthedocs.io/en/latest/clustering.html): UMAP "does not completely preserve density" and "can create false tears in clusters"; for clustering use **larger `n_neighbors` (~30), `min_dist=0.0`, and `n_components` ~ 5-10** rather than 2.
- UMAP outlier doc (https://umap-learn.readthedocs.io/en/latest/outliers.html): the danger is that the embedding "contracts outliers into clusters"; **`set_op_mix_ratio≈0.25`** (default 1.0 = pure union) preserves outliers better at some cost to global structure; with that caveat they *do* endorse running classical outlier detection (LOF) on UMAP-reduced data.
- Known pitfall: GLOSH/LOF scores computed in **2D display space measure the embedded density, which UMAP actively distorts** (it equalizes density locally). Never score outliers on the 2D map. A 5D, `min_dist=0.0` clustering embedding is the accepted compromise; `densmap=True` is an option if density fidelity matters more.

**Concrete recommendation for our eval (collision clips as low-prevalence outliers, AP / precision@k):** run **three cheap scorers** and report all (this is the interesting result either way, and at 1.5k-30k points each takes seconds):
1. **GLOSH** via classic `hdbscan` fit on the **5D UMAP clustering space** (the same space ToponymyClusterer clusters) — the "Tutte-native" score, hierarchy-aware, supports local outliers. `hdbscan.HDBSCAN(min_cluster_size=15, min_samples=10).fit(X5d).outlier_scores_`.
2. **k-NN distance in the ORIGINAL embedding space** (mean cosine distance to k=10 nearest neighbors, `sklearn.neighbors.NearestNeighbors(metric="cosine")`) — distortion-free baseline; if GLOSH-in-UMAP-space beats/loses to this we learn something about the UMAP confound.
3. **LOF** (`sklearn.neighbors.LocalOutlierFactor(n_neighbors=20, novelty=False)`) in the 5D UMAP space, per the umap-learn outlier doc's own recipe.
Fit the 5D UMAP with `set_op_mix_ratio=0.25` for the *scoring* embedding (keep default 1.0 for the display map). For the resampled 3–10%-prevalence subsets: refit UMAP+scorers per subsample (cheap at this scale) so the density estimate reflects the simulated prevalence — do NOT score on the full-data embedding and then subsample scores.

---

## 3. DataMapPlot interactive plots (datamapplot 0.7.3)

**Entry point** — `datamapplot.create_interactive_plot(data_map_coords, *label_layers, **kw)` (`create_plots.py:349`). Returns an `InteractiveFigure` with `.save(filename)` (single HTML), `.save_bundle(zipfile)` (HTML + offline data zips), `_repr_html_` for notebooks.

Key signature facts:
- `data_map_coords`: (n,2) float array (UMAP 2D). Internally rescaled to a standard extent.
- `*label_layers`: one string array per layer, **finest-grained FIRST, coarsest last** (docstring). `noise_label="Unlabelled"` marks noise — exactly what Toponymy's `cluster_layer.topic_name_vector` produces (`cluster_layer.py:803`: noise → `"Unlabelled"`). So: `create_interactive_plot(map2d, *[cl.topic_name_vector for cl in topic_model.cluster_layers_])`.
- `hover_text=` list/array of per-point strings.
- All extra rendering kwargs flow through `**render_html_kwds` to `render_html` (`interactive_rendering.py:445`): `extra_point_data`, `hover_text_html_template`, `on_click`, `enable_search`, `search_field`, `colormaps` / `colormap_rawdata`+`colormap_metadata`, `cluster_layer_colormaps`, `histogram_data`, `custom_html/css/js`, `offline_mode`, `title`, `sub_title`, `font_family`, etc.

**Per-point hover THUMBNAILS** — the officially sanctioned pattern (doc/interactive_customization_options.ipynb cell 49): put an image URL/data-URI column in `extra_point_data` (a DataFrame, one row per point) and reference it in `hover_text_html_template`:
```python
extra = pd.DataFrame({"thumb": data_uris_or_paths, "weather": ..., "time": ...})
hover_template = """
<div style="max-width:340px">
  <img src="{thumb}" width="320"/>
  <div style="font-size:11px">{hover_text}</div>
  <div style="font-size:10px;color:#888">{weather} | {lighting} | GLOSH {glosh}</div>
</div>"""
```
Template placeholders `{col}` compile to JS template literals `${hoverData.col[index]}` (`interactive_helpers.py:2195-2206`).
**Pitfall (open issue #150, verified in `prepare_hover_data`, interactive_helpers.py:2155):** `hover_text_html_template` is **silently ignored unless `extra_point_data` is not None**. Passing only `hover_text` + template → you get plain hover_text. Passing template alone → no tooltip at all. Always pass BOTH `hover_text` and `extra_point_data` (issue #137 — shape mismatch now raises an assert).
- **Inline base64 vs served files:** both work. Inline data-URIs survive `file://` double-click opening (single portable HTML). Served relative paths (`thumbnails/{clip_id}.jpg`) keep the HTML tiny but `inline_data=False` data zips are fetched via XHR, which **fails on `file://` (CORS)** — needs `python -m http.server` (fine even air-gapped). There is also `dynamic_tooltip={fetch_js,format_js,loading_js,error_js}` for lazy fetch-on-hover (needs a server).
- **Size budget @1,500 points:** hover/extra data is JSON → gzip → base64 into the HTML (`interactive_helpers.py:2551-2559`); a data-URI round-trips to ≈1.33× raw JPEG bytes. 160px-wide JPEG q≈60 thumbnails ≈ 5-8 KB each → **~10-16 MB HTML for 1,500 points: perfectly fine**. At 30k event windows that becomes 200-320 MB — do NOT inline; use relative-path thumbs + http.server, or drop thumbnails on the 30k map.
- `on_click="window.open(`clips/{clip_id}.mp4`)"` gives click-to-play of the actual dashcam clip when served by http.server — great demo value.

**Cluster label layers from Toponymy**: pass topic-name vectors as `*label_layers`; add `enable_topic_tree=True` for the hierarchy-navigation sidebar (uses `cluster_tree_` implied by the name vectors); `cluster_boundary_polygons=True` draws alpha-shape outlines. 0.7.3 added `hierarchical_collision_priority=True` (default) fixing multi-layer label flicker (issue #192; deck.gl pinned to 9.1 because 9.2/9.3 regressed CollisionFilterExtension — `interactive_helpers.py:400-405`). Issue #188 (labels invisible until first zoom due to font-atlas race) fixed in 0.7.3 — **pin datamapplot>=0.7.3**.

**Coloring by metadata**: `colormaps={"Weather": weather_arr, "Lighting": light_arr, "GLOSH": glosh_arr}` (dtype-inferred) or fine-grained `colormap_rawdata=[arr,...]` + `colormap_metadata=[{"field":..., "description":..., "cmap":"viridis"},...]`; `cluster_layer_colormaps=True` adds one categorical colormap per label layer. Adds a dropdown selector widget (pulls d3). Static per-point overrides: `marker_size_array`, `marker_color_array`, `marker_alpha_array` (e.g., bigger red markers for ground-truth collision clips). Per-point marker *shapes* are not supported (issue #85 open). `histogram_data=` gives a linked brushable histogram (e.g., time-of-event).

**Search**: `enable_search=True`, `search_field="hover_text"` (default) or any `extra_point_data` column name — substring match dims non-matching points. Entirely client-side, works offline.

**OFFLINE / air-gapped — works, with one preparation step and one wart.**
- Default output pulls from CDNs: `deck.gl@9.1` + `apache-arrow@latest` from unpkg.com always; `d3` (if histogram/colormaps), `jquery@3.7.1` (if topic tree), DataTables (some selection handlers); plus Google Fonts (`offline_mode_caching.py:19-32`). **A default HTML is dead on an air-gapped box.**
- Fix: while online run `dmp_offline_cache` (console script; `--export cache.zip` / `--import cache.zip` to carry across machines; cache lives in `platformdirs.user_data_dir("datamapplot")`), then render with `offline_mode=True` → all JS + fonts embedded base64 in the HTML (doc/offline_mode.ipynb). If the cache is missing at render time it tries to build it (needs internet), so **build/refresh the cache BEFORE the workshop** (and with >=0.7.x so the cache key matches `deck.gl@9.1`, not the old `@latest`).
- Wart (verified in `templates/head_dependencies.html.jinja2:21-28`): `<link>` tags for **bootstrap-theme 3.2.0 and font-awesome 4.6.3 CSS from maxcdn.bootstrapcdn.com are emitted unconditionally, even in offline_mode**. On an air-gapped box these fail after a timeout — the map itself renders fine, but a few widget icons may be missing and first paint can wait on the failed request. Test once; if annoying, strip those two `<link>`s from `plot._html_str` before `.save()` (one-line regex), or include font-awesome via `custom_css`.
- Issue #134 (open): extra JS deps for `custom_html` aren't auto-cached in offline mode — workarounds in the issue (dummy SelectionHandler with `dependencies=[...]`, or `loadBase64Script` from the cache). Only relevant if we add custom JS widgets.
- `inline_data=True` (default) + `offline_mode=True` → fully self-contained single file, double-clickable. `inline_data=False` + `offline_data_path=` → smaller HTML + sidecar zips, must be served over http.

**Other relevant issues scanned** (`gh issue list -R TutteInstitute/datamapplot --state all --limit 40`): #125 "how many labels is too many" (open; keep ≤ a few hundred labels/layer — non-issue for us), #114 float16 inputs broke v0.6 (closed), #131 no-labels crash (closed), #164 colored clusters w/o labels (closed), #136 tz-aware datetimes break colormap legend (open — use naive datetimes in `colormaps`), #101 setuptools>=78 build failures (open — install wheels, not sdist).

---

## 4. temporal-mapper, thisnotthat, scalar — one paragraph each

**temporal-mapper (1.3.0, `src/temporalmapper/`)**: density-based Mapper-algorithm library for *temporal topic modelling* — `temporalmapper.Mapper` (sklearn-compliant) plus `TemporalMapper` which slices a corpus along a time axis, clusters per slice, and builds a graph whose nodes are topics-at-a-time with `temporal_plot` interactive/static visualizations (UN General Debate example, integrates with Toponymy for node naming). For Nexar there is no meaningful corpus-level time axis (clips are i.i.d. events), so it's **future-work only** — the one tantalizing future idea is using *within-clip time* (frame windows ordered by t − time_of_event) to map how scene topics flow into collision topics, which is exactly a Mapper-over-time structure. Skip for the 2-3 day project.

**thisnotthat (TNT, 0.4.2)**: Panel/Bokeh-based interactive data-map *workbench* — notebook-embedded plots with linked search widgets, data tables, instance viewers, attribute-driven styling, and most notably **interactive bulk labelling** of data-map regions, deployable as Panel web apps. It's the right tool if we wanted humans to relabel/correct cluster assignments live, but it's a heavier server-backed stack (Panel+Bokeh version-matching warnings on JupyterHub) and DataMapPlot's static HTML is far better suited to an air-gapped demo artifact. **Future work** (e.g., human adjudication of LLM-as-judge disagreements).

**scalar**: "SCAlable LARge clustering" meta-estimator `ClusterScalar(clusterer, downsampler, upsampler)` — downsample (e.g., `KMeansDownsampler`), cluster the subset with any sklearn clusterer, upsample labels via `KNNUpsampler` with noise-voting. Designed for millions of points; at our 1,500–30,000 points it solves a problem we don't have. **Not worth using**; only note it exists if someone scales to full-frame-level windows (~1.8M frames).

---

## 5. GitHub issue-scan summary (pitfall shortlist)

**datamapplot** (40 most recent, all states):
- #150 OPEN — `hover_text_html_template` silently ignored without `extra_point_data` (see §3; bites our thumbnail plan if forgotten).
- #134 OPEN — extra JS deps + `offline_mode=True` need manual inclusion.
- #192/#188 CLOSED in 0.7.3 — multi-layer label collision flicker / labels invisible on first load → require 0.7.3.
- #137 CLOSED — wrong-shape `extra_point_data` now asserts.
- #136 OPEN — tz-aware datetimes break colormap legend.
- #125/#123 OPEN — label count & payload-size concerns at large scale (we're small).
- #101 OPEN — sdist build fails with setuptools>=78 (use wheel).
- #64 CLOSED — confirms offline mode is the supported answer for no-CDN environments.

**evoc** (all 18 issues):
- #30 CLOSED — `IndexError: pop from empty list` at 5.7M pts; workaround `node_embedding_init=None`.
- #19 OPEN — no 2D layout from EVoC; pair with UMAP for DataMapPlot (maintainer-confirmed pattern).
- #17 OPEN — parameter guidance thin; ~65% noise reported on 6M web pages even at `noise_level=0`.
- #11 OPEN — high noise fraction is expected behaviour; tune `noise_level` toward 0.
- #8 OPEN — `labels_` = max-persistence layer, not finest; pick `cluster_layers_[0]` deliberately.
- #10 CLOSED — `random_state` now supported (also flips on slower reproducible Borůvka).
- #6 OPEN — no `.predict()` for out-of-sample points.
- #2 OPEN(stale) — evoc IS on PyPI now (0.3.1, 2026-03-27).

---

## 6. Concrete code sketch — embeddings → UMAP → clusterer → outlier scores → DataMapPlot

```python
# pins (uv): umap-learn==0.5.12 fast-hdbscan==0.3.2 hdbscan==0.8.44 evoc==0.3.1
#            toponymy==0.5.2 datamapplot==0.7.3 pillow pyarrow
import numpy as np, pandas as pd, umap, hdbscan, datamapplot
from toponymy import Toponymy, ToponymyClusterer, ClusterLayerText

emb = np.load("siglip2_clip_embeddings.npy").astype(np.float32)   # (N, d), L2-normalized
N = emb.shape[0]                                                   # 1_500 clips or ~30_000 windows

# ---- 1. UMAP: separate CLUSTERING space (5D) and DISPLAY space (2D) -------------
# ~1,500 pts: n_neighbors=25-30 (larger fraction of data => stabler manifold)
# ~30,000 pts: n_neighbors=30-50; everything else identical. seconds-to-~2min on 32 cores.
umap_cluster = umap.UMAP(n_components=5, n_neighbors=30, min_dist=0.0,
                         metric="cosine", set_op_mix_ratio=0.25,   # preserve outliers (umap docs/outliers)
                         random_state=42).fit_transform(emb)
umap_display = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                         metric="cosine", random_state=42).fit_transform(emb)
# bag-of-objects pipeline: same calls with metric="hellinger" on the reweighted count matrix.

# ---- 2. Multi-layer clustering (feeds Toponymy directly) -----------------------
clusterer = ToponymyClusterer(min_clusters=4,
                              base_min_cluster_size=max(8, N // 150),  # ~10 @1.5k, ~200 @30k
                              next_cluster_size_quantile=0.85, verbose=True)
# Toponymy.fit(clip_captions, emb, umap_cluster) calls clusterer.fit_predict internally.
# Alternative A (comparison condition): evoc.EVoC on raw `emb`
#   import evoc; ev = evoc.EVoC(base_min_cluster_size=10, noise_level=0.3, random_state=42)
#   ev.fit(emb); layers = ev.cluster_layers_; dups = ev.duplicates_
#   (via toponymy: EVoCClusterer then `c.evoc.approx_n_clusters = None`  # un-break multi-layer)
# Alternative B: fast_hdbscan.PLSCAN(base_min_cluster_size=10).fit(umap_cluster)  # PLSCAN layers in UMAP space

# topic_model = Toponymy(llm_wrapper=..., text_embedding_model=..., clusterer=clusterer,
#                        object_description="dashcam driving clips", ...)
# topic_model.fit(captions, emb, umap_cluster)
# label_layers = [cl.topic_name_vector for cl in topic_model.cluster_layers_]  # finest first; noise="Unlabelled"

# ---- 3. Outlier scores (collision-as-outlier eval) ------------------------------
glosh = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=10).fit(umap_cluster).outlier_scores_
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
nn = NearestNeighbors(n_neighbors=11, metric="cosine").fit(emb)
knn_dist = nn.kneighbors(emb)[0][:, 1:].mean(axis=1)               # high-dim, distortion-free
lof = -LocalOutlierFactor(n_neighbors=20).fit(umap_cluster).negative_outlier_factor_
# eval: sklearn.metrics.average_precision_score(y_collision, score) on each resampled
# 3-10%-prevalence subset; REFIT umap_cluster + scorers per subset.

# ---- 4. DataMapPlot with thumbnails + topic labels + metadata colormaps ---------
import base64; from io import BytesIO; from PIL import Image
def thumb_uri(path, w=160):
    im = Image.open(path); im.thumbnail((w, w)); buf = BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=60)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

extra = pd.DataFrame({
    "thumb":   [thumb_uri(p) for p in event_frame_paths],   # ~10-16MB HTML @1.5k pts; at 30k use
    "clip_id": clip_ids,                                    #   relative paths + `python -m http.server`
    "weather": meta.weather, "lighting": meta.lighting,
    "glosh":   np.round(glosh, 3),
})
plot = datamapplot.create_interactive_plot(
    umap_display,
    *label_layers,                       # Toponymy topic-name vectors, finest first
    hover_text=captions,                 # REQUIRED alongside extra_point_data for template (issue #150)
    extra_point_data=extra,
    hover_text_html_template=(
        '<div style="max-width:340px"><img src="{thumb}" width="320"/>'
        '<div style="font-size:11px">{hover_text}</div>'
        '<div style="font-size:10px;color:#999">{weather} | {lighting} | GLOSH {glosh}</div></div>'),
    on_click="window.open(`clips/{clip_id}.mp4`)",           # works when served via http.server
    enable_search=True, search_field="hover_text",
    colormaps={"Weather": meta.weather.values, "Lighting": meta.lighting.values,
               "GLOSH": glosh, "Collision": y_collision.astype(str)},
    marker_size_array=np.where(y_collision, 8, 4),           # emphasize positives
    enable_topic_tree=True, cluster_boundary_polygons=True,
    darkmode=True, title="Nexar Dashcam Topic Map",
    inline_data=True, offline_mode=True,                     # run `dmp_offline_cache` ONLINE first!
)
plot.save("nexar_map.html")   # single self-contained file (minus the bootstrap/font-awesome CDN wart, §3)
```

## Bottom line for the project plan
- Primary: UMAP(5D, cosine, min_dist=0, n_neighbors=30) → `ToponymyClusterer` → Toponymy naming → DataMapPlot 0.7.3 with `offline_mode=True` + inline base64 thumbnails (1.5k map) / served thumbnails (30k map).
- Secondary/comparison: `evoc.EVoC` on raw embeddings (multi-layer + duplicate detection); if used through toponymy's `EVoCClusterer`, null out `approx_n_clusters`.
- Outliers: GLOSH needs classic `hdbscan` (fast_hdbscan has none); score in 5D UMAP space, never 2D, plus high-dim kNN-distance control.
- Air-gap prep checklist: `dmp_offline_cache --export` while online; pin deck.gl-9.1-era cache; test one HTML on a disconnected browser (bootstrap/font-awesome links will dangle); pre-download all wheels.
