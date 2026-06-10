# The Tutte Institute's own example pipelines — mined for STAMP 2026

Sources (local clones, all read end-to-end):

- `/home/b3ali/projects/stamp26/repos/scipy2023/` — "Commonplace detection in categorical telemetry data" (Hamelin & Healy, SciPy 2023 poster; OpTC host telemetry)
- `/home/b3ali/projects/stamp26/repos/acme3-mapping/` — "Vector space embeddings and data maps for cyber defense" (SciPy 2024 talk; ACME3 Wintap telemetry)
- `/home/b3ali/projects/stamp26/repos/tutorials/` — official Tutte tutorials: `1-recipes-bag-of-words.ipynb`, `2-topic-modelling-pokemon.ipynb`
- `/home/b3ali/projects/stamp26/repos/vectorizers_playground/` — Easydata repo, notebooks `01-vectorizers-quickstart`, `03-document-embeddings-with-vectorizers`, `04-topic-embedding` (20 newsgroups)
- Cross-checked against `/home/b3ali/projects/stamp26/repos/toponymy/` (README + `toponymy/clustering.py`, `toponymy/llm_wrappers.py`) and `/home/b3ali/projects/stamp26/repos/vectorizers/vectorizers/transformers/info_weight.py` for blessed defaults.

---

## 1. scipy2023 — the reference cyber-telemetry pipeline ("commonplace detection", OpTC)

Two notebooks: `01 Data Engineering.ipynb` (events → 2D map) and `02 Interactive visualization.ipynb` (ThisNotThat dashboard).

### 1.1 Sessionization
- Raw unit = **event** (ECAR JSON record from host-based sensor). Aggregation unit = **process instance** ("computation"), identified by `actorID` UUID. PROCESS/CREATE events are double-counted: once as an action of the parent (`actorID`) and once as the birth event of the child (`objectID`) — deliberately inducing correlations between processes that spawn the same children and between processes born of similar parents.
- Events are encoded as **bags of (kind, value) categorical tokens** via a per-object-type field schema (`categoricals_easy` dict): e.g. `"FLOW": ["object", "action", ("src_ip","ip"), ("dest_ip","ip"), ("src_port","port"), ("dest_port","port"), "l4protocol", "direction"]`. The `kind` disambiguates token namespaces (an IP-looking filename must not collide with an actual IP token).
- **Process vector = SUM of its event one-hot vectors** (explicitly acknowledged as lossy: discards timing, order, event boundaries — "we go in knowing that"). Implemented as a sparse 0/1 projection matrix: `process_matrix = (projection @ event_matrix).astype(np.float32)`.

### 1.2 Vectorization
```python
import vectorizers as vz
vzr = vz.NgramVectorizer().fit(df["tokens"])     # one-hot / counts; default unigrams
event_matrix = vzr._train_matrix                  # sparse CSR; vocab in vzr.column_index_dictionary_
```
Chunked vectorizers are merged with the `+` operator on fitted `NgramVectorizer` instances (Dask-parallel hierarchical summation).

### 1.3 Pruning conventions (quoted thresholds)
- Drop process instances with **total token weight < 10** ("does not make much sense to keep processes described by a total number of categorical features less than 10").
- Drop **orphan features** tied to <= 3 processes (`if count > 3: keep`), and would drop **spurious features** (shared at similar weight by too many objects) — none found here.
- Sanity check after culling: every remaining process keeps total weight > 5.
- **Deduplicate identical rows** before UMAP (manual MD5 row hashing + `np.unique(..., return_inverse=True)`; later repos just use `umap.UMAP(unique=True)`).

### 1.4 Reduction (exact call)
```python
from sklearn.preprocessing import Normalizer
normalized_matrix = Normalizer(norm="l1").fit_transform(unique_matrix)  # L1-normalize counts → multinomials

process_protomap = umap.UMAP(
    n_components=2,
    metric="hellinger",      # "the metric best supported by probability theory ... between multinomial distributions"
    densmap=True,            # densMAP variant: preserves density differences
    dens_lambda=4,
    n_epochs=800,
    verbose=True,
).fit_transform(normalized_matrix)
process_map = process_protomap[inverse_u, :]      # reduplicate
```
densMAP is used **specifically because the analytic goal is density-based**: commonplace behaviors = large dense clusters; preserving high-dim density differences in 2D is "critical" for that detection. (Directly relevant to our collision-as-outlier framing — see §6.)
- They explicitly warn: UMAP output is **not reproducible even with a seed** (multithreaded SGD).

### 1.5 Labeling
No clustering / no LLM in this repo. Points carry **provenance labels**: command line if available (priority 0), else `image_path` (priority 10), else `(unknown)`. Visualization restricted to the **top-15 label classes** by count.

### 1.6 Visualization (notebook 02): ThisNotThat (Panel) dashboard
- `tnt.BokehPlotPane(map_2d, labels=..., width=600, height=600, show_legend=False, tools="pan,wheel_zoom,lasso_select,tap,reset")`.
- Hierarchical floating cluster annotations built from features:
```python
from vectorizers.transformers import InformationWeightTransformer
infoweight = InformationWeightTransformer().fit_transform(features_top15).astype(np.float32)
infoweight_compressed = TruncatedSVD(n_components=1024).fit_transform(infoweight)
layer_metadata = tnt.SparseMetadataLabelLayers(
    infoweight_compressed, processes_top15, features_top15,
    {i: value for i, (_, value) in col2token_viz.items()},
    cluster_map_representation=False, random_state=42)
plot.add_cluster_labels(hier_annotations, max_text_size=24)
```
- Two bespoke selection summarizers (precursors to evaluation tooling we may reuse conceptually):
  - `SparseSupportSummarizer`: top-k features by column marginal over the selection + support fraction.
  - `SparseFeatureImportanceSummarizer`: one-vs-rest **L1 logistic regression** (`LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", tol=1e-3, max_iter=20)`) of the selection vs the rest; model accuracy is rendered as a color-coded **trustworthiness** indicator (>0.9 green / >0.8 yellow / >0.5 orange / else red). "Such experiments provide an upper bound on the performance of a classifier one may want to develop to detect the phenomenon expressed through the selected points."
- Per-point metadata rollup uses `vectorizers.transformers.CategoricalColumnTransformer(object_column_name='process_id', descriptor_column_name=[...], include_column_name=True)` → markdown info pane.

---

## 2. acme3-mapping — end-to-end ACME3 host-telemetry mapping (SciPy 2024)

Six notebooks; this is the most modern of the example repos (uses `datamapplot`, `fast_hdbscan`, `glasbey`).

### 2.1 NB0 — data engineering
DuckDB over Parquet; filter out the telemetry agents' own processes (`ptree NOT LIKE '%wintap%' AND ... '%amazon-ssm%'`); reconstruct command line as `trim('"' || process_path || '" ' || coalesce(args,''))`. Output: `process_filtered.parquet` (136,706 instances), `image_loads.parquet`.

### 2.2 NB1 — command lines as bags of tokens (the baseline map)
1. **Unique** the 136k command lines → 30,994 (`np.unique(..., return_index, return_inverse, return_counts)`); keep the de-unique index for later reduplication.
2. Tokenize with `shlex.split(cmdline, posix=False)` (retry with `'" <broken>'` suffix on ValueError).
3. Vectorize + reweight + reduce:
```python
vz_ngram = vz.NgramVectorizer().fit(cmdlines_tokenized.tolist())
cmdlines_tc = vz_ngram._train_matrix.tocsr()                          # 30994 x 20834
cmdlines_iwt = vzt.InformationWeightTransformer().fit_transform(cmdlines_tc)
bagofwords_dmap = umap.UMAP(metric="hellinger", init="pca", verbose=True).fit_transform(cmdlines_iwt)
```
   Note: **IWT output fed straight into Hellinger UMAP with no explicit L1 normalization** and otherwise default UMAP (n_neighbors=15, min_dist=0.1, n_epochs auto). `init="pca"` for stability. Disconnected vertices (strictly-unique-token rows) come out NaN and are masked out: `i_valid, = np.nonzero(~np.any(np.isnan(dmap), axis=1))`.
4. Visualization with DataMapPlot:
```python
import glasbey
palette = glasbey.create_palette(K, colorblind_safe=True)   # K = 25 top process names
label_color_map = dict(zip(names_topK, palette)); label_color_map["Unlabelled"] = "#dddddd"
dmp.create_interactive_plot(
    bagofwords_dmap[i_valid], labels_topK.iloc[i_valid],
    hover_text=hover_text.iloc[i_valid],          # "(124x) cmdline..." truncated to 100 chars
    label_color_map=label_color_map,
    title="Process instances", sub_title="as bags of command line tokens", darkmode=True)
```

### 2.3 NB2 — Wasserstein embedding of command lines (the "good" map)
```python
vz_cooc = vz.TokenCooccurrenceVectorizer(n_threads=os.cpu_count(), n_iter=3).fit(cmdlines_tokenized.tolist())
cooc_vec = vz_cooc.reduce_dimension(512)                              # token vectors, 20834 x 512
cmdlines_wass = vz.WassersteinVectorizer().fit_transform(cmdlines_iwt, vectors=cooc_vec)   # → 30994 x 128
wass_dmap = umap.UMAP(metric="cosine", init="pca", verbose=True).fit_transform(cmdlines_wass)
```
Pattern: **counts+IWT → Hellinger; dense embeddings (cooc/Wasserstein/SVD) → cosine.** This holds in every single Tutte notebook.

### 2.4 NB3 — processes as bags of DLLs + cluster labels by differential distribution
- `vzt.CategoricalColumnTransformer("pid_hash", "filename").fit_transform(image_loads)` pivots the event table into per-process token lists.
- `NgramVectorizer` → `InformationWeightTransformer` as usual.
- Cheap orderless cooccurrence instead of TokenCooccurrenceVectorizer: `images_cooc = scipy.sparse.diags(sum_rows_iwt) @ processes_iwt.T @ processes_iwt`; then `TruncatedSVD(n_components=512)` → `WassersteinVectorizer().fit_transform(processes_iwt, vectors=images_svd)`.
- `umap.UMAP(metric="cosine", unique=True, verbose=True)` (note `unique=True` — 40,982 rows dedupe to 4,635).
- **Clustering ON THE 2D MAP** with `from fast_hdbscan import HDBSCAN`:
```python
clusters = HDBSCAN(min_cluster_size=10).fit_predict(processes_dmap)   # 475 clusters
```
- **Cluster naming = differential feature distribution vs corpus null** (this is the house heuristic that Toponymy later mechanizes with an LLM):
```python
null_distrib = normalized_distribution(processes_iwt)         # column means, normalized
for cluster ...:
    distrib_cluster = normalized_distribution(processes_iwt[clusters == cluster])
    diff = np.abs(distrib_cluster - null_distrib)
    i_top = np.argsort(-diff)[:3]
    topic = "\n".join((">" if distrib_cluster[i] > null_distrib[i] else "<") + name(i) for i in i_top)
    topics[cluster] = topic + "‭" * cluster              # invisible-char hack to make labels unique
```
- `dmp.create_interactive_plot(..., enable_search=True)` with these topics as labels.

### 2.5 NB4 — hosts as bags of processes over time (hierarchical / temporal embedding)
This is the closest structural analog to our **event-window** idea.
- Resample process instances per host into **3-hour bins** (`group.resample(pd.Timedelta(hours=3), on="timestamp", ...)`) → "host-times".
- Build (host-time × command-line-index) count matrix with `vz.EdgeListVectorizer(column_label_dictionary={i: i for i in range(len(cmdlines_vec))})` (forcing full column space so columns align with the precomputed command-line vectors).
- `InformationWeightTransformer` → `WassersteinVectorizer().fit_transform(pxht_iwt, vectors=cmdlines_vec)` — i.e. **host-time = distribution over the point cloud of command-line vectors** (hierarchical lifting: tokens → cmdlines → host-times).
- `umap.UMAP(metric="cosine")` → `HDBSCAN(cluster_selection_method="leaf").fit_predict(pxht_dmap)` (leaf selection for finer clusters; 162 clusters on 2,549 host-times).
- Cluster naming: rank command lines by **summed IWT bits within the cluster**, group by process name, keep names with mean bits >= 1.0 and >= max/10, take top 3 → label like `"googleupdate.exe 9.0\nsihclient.exe 2.7\n..."`.
- Time encoded as continuous marker color: `marker_color_array=[cmap((ts - ts_min)/delta) ...]` with `color_label_text=False`.

### 2.6 NB5 — comparing two data maps of the same objects
- Re-run UMAP for map B with `init=` map A's coordinates: `umap.UMAP(metric="cosine", init=bagofwords_dmap_valid).fit_transform(wasserstein_vectors)`.
- Then `vz.utils.procrustes_align(map_a, map_b, scale_to="first")`.
- Static side-by-sides with `dmp.create_plot(..., use_medoids=True, label_font_size=9., figsize=(8,8), font_family="Roboto", label_color_map=...)`.
This is exactly the tool we need for our **rerun-stability** evaluation and for comparing our three embedding tracks on aligned axes.

---

## 3. tutorials repo — the canonical doctrine

Only two tutorials exist (`1-recipes-bag-of-words.ipynb`, `2-topic-modelling-pokemon.ipynb`). There is **no Toponymy or DataMapPlot-deep tutorial** in this repo; tutorial 2 uses datamapplot lightly and explicitly punts LLM naming to "https://github.com/tutteinstitute/topicnaming" (= Toponymy): "To turn this list of pokemon and keywords into a short name for the cluster, one could consider writing a prompt and asking an LLM ... This turns out to be a subtle problem."

### 3.1 Tutorial 1 (recipes): the mantra
> "If you can describe your data as: *Each data object is a bag of elements*, then the first thing that should be tried, **every time, without exception**, is ... **vectors, infoweight, UMAP**."
>
> "The bag of words is the minimum baseline that smart should beat."

Exact pipeline:
```python
vz_ngram = vz.NgramVectorizer().fit(data["ingredients"])                  # 39774 x 6714
weights_iwt = vzt.InformationWeightTransformer().fit_transform(vz_ngram._train_matrix)
weights_u2  = umap.UMAP(metric="hellinger", unique=True).fit_transform(weights_iwt)
```
- IWT is described as "*like TF/IDF, but properly founded in information theory*" (KL-divergence of observed vs marginal multinomial, with a Bayesian prior).
- `unique=True` is recommended as a default ("faster and more stable numerical computations, so it just makes sense").
- Visualization via `tnt.BokehPlotPane(weights_u2, labels=data["cuisine"], hover_text=..., marker_size=.025)` + `tnt.SimpleSearchWidget`.
- Closing caveat: the IWT improvement is real but subtle on this data; "further processing can be performed to improve the selection of categorical features."

### 3.2 Tutorial 2 (Pokémon): topic modelling = the mantra + 2 steps
> "vectorize, information weight, UMAP, **cluster, represent**."
- Vectorize: sklearn `CountVectorizer` over move/ability tokens; stats discretized to top-2 stat names and digram-encoded with `CountVectorizer(ngram_range=(1,2))`; matrices concatenated. (Shows the house is happy mixing token sources into one bag — a precedent for mixing detected-object tokens with discretized metadata tokens.)
- `InformationWeightTransformer()` defaults.
- `UMAP(n_components=2, metric='cosine')` (cosine here since the focus is the map).
- **HDBSCAN on the 2D map**, `min_cluster_size=4` with explicit reasoning: *set min_cluster_size to just above the smallest meaningful group* ("The largest Pokemon families have only 3 members, so I will set the minimum cluster size to 4"). ~14% noise points reported as normal.
- Represent: per-cluster sum of IWT vectors, **re-IWT the cluster×feature matrix**, take top-3 features as keywords; then hand-name clusters (LLM naming deferred to Toponymy).
- DataMapPlot finale with the custom hover template — the exact mechanism we need for thumbnails:
```python
fig = datamapplot.create_interactive_plot(
    data, string_labels, title="Pok&eacute;Map",
    hover_text=hover_data['Name'].to_list(),
    extra_point_data=hover_data,                       # DataFrame; columns usable in template
    hover_text_html_template="""
        <div><h3>{hover_text}</h3><p>Type: {Type}</p>
        <p>Top Stats: {TopStats}</p><p>Notable: {KW}</p></div>""",
    enable_search=True,
    marker_size_array=size_arr)
```
- Methodological note worth quoting for our name-trustworthiness eval: "If many different sets of choices lead to the same conclusion, that means there is a lot of evidence that it is true!" (run pipeline under perturbations; keep only invariant structure).

---

## 4. vectorizers_playground — usage patterns (20 newsgroups)

The quickstart ("I don't care why it works, just show me what to do") gives the institute's **recommended parameterizations**:

### 4.1 Word vectors (TokenCooccurrenceVectorizer)
```python
word_vectorizer = vectorizers.TokenCooccurrenceVectorizer(
    min_document_occurrences=5,     # prune rare tokens
    window_radii=20,                # large window
    window_functions='variable',
    kernel_functions='geometric',   # decaying kernel within window
    n_iter=0,                       # EM refinement off for small corpora (NB: acme3 used n_iter=3)
    normalize_windows=True,
).fit(tokenized_news)
word_vectors = word_vectorizer.reduce_dimension(dimension=160, algorithm="randomized")
```
`reduce_dimension` = normalize cols, normalize rows, Hellinger-izing transform, SVD.
`umap.UMAP(n_neighbors=25, metric='cosine', random_state=42)` for the word map.

### 4.2 Document vectors (the AWE stack)
```python
bow_vectors = vectorizers.NgramVectorizer(token_dictionary=word_vectorizer.token_label_dictionary_).fit_transform(tokenized_news)
info_doc_vectors = vectorizers.transformers.InformationWeightTransformer(
    prior_strength=1e-1, approx_prior=False).fit_transform(bow_vectors)
awe_doc_vectors = vectorizers.ApproximateWassersteinVectorizer(
    normalization_power=0.66, random_state=42).fit_transform(info_doc_vectors, vectors=word_vectors)
umap.UMAP(metric="cosine", random_state=42).fit(awe_doc_vectors)
```
NOTE: library defaults for IWT are `prior_strength=1e-4, approx_prior=True, weight_power=2.0` (verified in `/repos/vectorizers/vectorizers/transformers/info_weight.py`); the playground overrides to `1e-1, approx_prior=False` for this corpus. The exact-LOT variant is `WassersteinVectorizer(n_components=160, random_state=42)`. Raw BoW/IWT matrices get `metric="hellinger"`, all dense vectors get `metric="cosine"` — again uniformly.

### 4.3 THE canonical clustering parameterization (repeated 5x across notebooks 01 and 04)
```python
low_dim_rep = umap.UMAP(
    metric="cosine", n_components=5, min_dist=1e-4, random_state=42, n_epochs=500
).fit_transform(awe_doc_vectors)
cluster_labels = hdbscan.HDBSCAN(min_cluster_size=25).fit_predict(low_dim_rep)
```
i.e. **cluster in a 5-D UMAP with near-zero min_dist and extra epochs; visualize in a separate 2-D UMAP with default min_dist**. min_cluster_size=25 on ~18k docs.

### 4.4 Topic embedding trick
Words are pushed through the *same* document pipeline (`awe_vectorizer.transform(info_transformer.transform(scipy.sparse.eye(n_words)))`) and co-UMAPped with documents (`fit(np.vstack([doc_vectors, word_vectors]))`) so topic words land inside their document clusters.

---

## 5. Cross-check: Toponymy's blessed defaults (the productionized "cluster + name" stage)

From `/repos/toponymy/README.rst` and `toponymy/clustering.py`:
- `ToponymyClusterer(min_clusters=6, min_samples=5, base_min_cluster_size=10)` (README example uses `min_clusters=4` for 18k docs). Multi-layer: builds HDBSCAN-tree-derived layers, doubling effective granularity until fewer than `min_clusters` clusters remain.
- **Two-vector contract**: `clusterer.fit(clusterable_vectors=document_map, embedding_vectors=document_vectors)` — cluster on the low-D map (2-D in the README), characterize/exemplar-select with the full-D embedding. Same split as `Toponymy(...).fit(text, document_vectors, document_map)`.
- `Toponymy(llm_wrapper=..., text_embedding_model=..., clusterer=..., object_description="newsgroup posts", corpus_description="20-newsgroups dataset", exemplar_delimiters=["<EXAMPLE_POST>\n","\n</EXAMPLE_POST>\n\n"])`.
- LLM wrappers relevant to our air-gapped box (verified in `toponymy/llm_wrappers.py`): `VLLMNamer` / `AsyncVLLMNamer` (in-process `vllm.LLM(model=...)`), `HuggingFaceNamer`, `LlamaCppNamer`, and `OpenAINamer(api_key, model="gpt-4o-mini", base_url=None)` — **`base_url` is supported**, so `OpenAINamer(api_key="EMPTY", model="<served-model>", base_url="http://localhost:8000/v1")` against our vLLM OpenAI-compatible server is the documented path ("a hosted model supporting the openAI API").

---

## 6. Synthesis: "the canonical Tutte pipeline" and where we mirror / deviate

### 6.1 Canonical ordered steps (with blessed parameters)

1. **Engineer & filter**: drop self-instrumentation artifacts; restrict to nominal capture window; unique-ify duplicate objects keeping `return_inverse` for reduplication (`np.unique` or `umap.UMAP(unique=True)`).
2. **Tokenize into bags**: `NgramVectorizer` (lists of tokens) / `CountVectorizer` (strings) / `CategoricalColumnTransformer` (pivot a long table to per-object token lists) / `EdgeListVectorizer` ((row, col, count) triples). Discretize numeric attributes into tokens (Pokémon stats digrams).
3. **Prune**: objects with total weight < ~10; features supported by <= ~3 objects; (optionally) spurious near-universal features.
4. **Reweight**: `InformationWeightTransformer()` — the house replacement for TF-IDF (defaults `prior_strength=1e-4, approx_prior=True`; corpus-tuned `1e-1/approx_prior=False` in the playground). Optional supervised mode via labels.
5. **(Optional upgrade) Wasserstein lift**: token vectors from `TokenCooccurrenceVectorizer(...).reduce_dimension(160–512)` (or IWT-folded cooccurrence + `TruncatedSVD(512)`), then `WassersteinVectorizer().fit_transform(iwt_matrix, vectors=token_vectors)` (or `ApproximateWassersteinVectorizer(normalization_power=0.66)` for cheap). This is also the **hierarchical lifting** mechanism (container = distribution over contained-object vectors), used for hosts-over-3h-windows.
6. **Reduce**:
   - metric: **hellinger** for count/IWT matrices, **cosine** for any dense vectors. Never euclidean.
   - viz map: `UMAP(n_components=2, metric=..., init="pca", unique=True)` with otherwise defaults (n_neighbors=15, min_dist=0.1); densMAP variant (`densmap=True, dens_lambda=4, n_epochs=800`) when density itself is the signal.
   - clustering map: `UMAP(n_components=5, min_dist=1e-4, n_epochs=500, metric="cosine")`.
7. **Cluster**: HDBSCAN (`fast_hdbscan` in newer repos) on the low-D map. min_cluster_size set to "just above the smallest meaningful group": 4 (1k Pokémon), 10 (41k processes), 25 (18k docs); `cluster_selection_method="leaf"` for finer granularity; -1 noise (~14%) is expected and labeled "Unlabelled". Productionized: `ToponymyClusterer(min_clusters=4–6, min_samples=5, base_min_cluster_size=10).fit(clusterable_vectors=map, embedding_vectors=full_vectors)` for multi-layer.
8. **Represent / name clusters**: differential distribution vs corpus null on the IWT matrix (top-3 over/under-represented features), or summed-IWT-bits ranking; productionized = Toponymy (exemplars + keyphrases → LLM). Heuristic differential labels are the pre-LLM fallback and a sanity check on LLM names.
9. **Visualize**: `datamapplot.create_interactive_plot(map2d, labels, hover_text=..., extra_point_data=df, hover_text_html_template=..., enable_search=True, label_color_map=dict(zip(topK, glasbey.create_palette(K, colorblind_safe=True))) | {"Unlabelled": "#dddddd"}, marker_size_array=..., marker_color_array=...)`; static comparisons via `create_plot(..., use_medoids=True)`. Top-K-classes coloring (K=15–25) + gray Unlabelled.
10. **Compare maps / reruns**: re-UMAP with `init=<reference map>`, then `vz.utils.procrustes_align(a, b, scale_to="first")`, side-by-side render.

### 6.2 Where the driving project should MIRROR

- **Pipeline 1 (bag-of-detected-objects) IS the house recipe.** Per clip/window: tokens = detected object classes (optionally kind-namespaced like scipy2023: `("vehicle","truck")`, `("vru","pedestrian")`, plus discretized metadata tokens weather/lighting/scene exactly like the Pokémon stat digrams) → `NgramVectorizer` → `InformationWeightTransformer()` (use IWT, not sklearn TF-IDF — every Tutte example uses IWT and they describe it as TF-IDF done right) → `UMAP(metric="hellinger", init="pca", unique=True)`. This is "the minimum baseline that smart should beat" — exactly the role our plan assigns it.
- **Event windows = host-times.** Mirror acme3 NB4: clip → fixed time bins (they used 3h for hosts; we use seconds around `time_of_event`); window = bag/distribution of per-frame object tokens; if we build object-token cooccurrence vectors, lift windows via `WassersteinVectorizer(iwt, vectors=token_vecs)`. Cheap option: `ApproximateWassersteinVectorizer(normalization_power=0.66)`.
- **Two-UMAP discipline**: cluster on `UMAP(n_components=5, min_dist=1e-4, n_epochs=500, metric="cosine")` (or hand the 2-D map to ToponymyClusterer as the README does), visualize on a separate 2-D map. Don't cluster the raw 768/1152-D embeddings.
- **metric choice**: cosine for SigLIP2 / V-JEPA2 / caption-text embeddings; hellinger only for the count-based track.
- **min_cluster_size by "smallest meaningful group"**: at N=1,500 clips, the Pokémon-style reasoning gives min_cluster_size≈5–10 (a pre-crash scenario type with <5 examples isn't a nameable topic); `ToponymyClusterer` defaults (`base_min_cluster_size=10, min_samples=5, min_clusters=4`) are appropriate as-is at our scale.
- **Cluster characterization vs null**: our per-cluster metadata histograms should be reported **differentially against the corpus marginal** (acme3 NB3) rather than as raw histograms — "snow: 0.41 vs corpus 0.07" is the house style and feeds the naming prompt much better.
- **Trustworthiness via one-vs-rest sparse L1 logistic regression** (scipy2023 summarizer): train selection-vs-rest on the token matrix per cluster; the accuracy is a quantitative "is this cluster real" score — cheap, local, complements LLM-as-judge.
- **Rerun stability**: use `init=<run-1 map>` + `vz.utils.procrustes_align` for the name-stability evaluation; also remember the explicit warning that seeded UMAP is still non-deterministic — stability must be measured, not assumed.
- **densMAP for the outlier eval map**: scipy2023 chose `densmap=True, dens_lambda=4` precisely because commonplace-vs-rare density structure was the analytic target. Our collision-as-outlier-at-low-prevalence framing is the mirror image (rare = signal), so produce at least one densMAP variant and check whether collision clips sit in low-density regions / HDBSCAN noise.
- **DataMapPlot hover thumbnails**: the Pokémon tutorial's `extra_point_data` + `hover_text_html_template` is the sanctioned mechanism; put an `<img src="...">` (local thumbnail path or base64 data URI — must be self-contained if air-gapped) in the template.
- **Coloring/labels**: glasbey colorblind-safe palette over top-K categories, `"Unlabelled": "#dddddd"`, `enable_search=True`.

### 6.3 Where we should DEVIATE (and why)

- **Scale**: examples run at 18k–137k objects; we have 1.5k clips (plus a few k windows). Consequences: UMAP `n_neighbors` default 15 is ~1% of our dataset (fine, maybe try 10–25); HDBSCAN/Toponymy layers will be shallow (expect ~2 layers, 10–60 leaf clusters); cluster-level statistics will be noisy — lean harder on exemplars + judge.
- **LLM naming**: none of the four example repos actually does LLM naming (heuristic differential labels or hand-naming; tutorial 2 explicitly defers to Toponymy). So Toponymy *is* the hosts' answer; our NHTSA-typology Jinja templates are a deviation at the prompt level only — keep Toponymy's exemplar/keyphrase machinery and swap/extend templates. Serve via `OpenAINamer(..., base_url="http://localhost:8000/v1")` (supported) or `AsyncVLLMNamer` in-process.
- **Embeddings**: visual/VLM embeddings replace TokenCooccurrence/Wasserstein as the "smart" track — but keep the Wasserstein lift available for the token track; it is the hosts' demonstrated best-in-class for bag data and is nearly free at our scale.
- **Evaluation**: average precision / precision@k at simulated prevalence has no precedent in these repos (their evaluation is the "advanced eyeball test" + downstream classification in playground NB03). The L1-logreg trustworthiness score and the perturbation-invariance principle (tutorial 2 conclusion) are the native ingredients we can cite/extend.
- **ThisNotThat**: the older interactive stack (scipy2023, tutorial 1). For a 2–3 day project DataMapPlot interactive HTML is the current house choice (acme3, Pokémon, Toponymy README) — skip TNT unless we want lasso-selection analytics.

### 6.4 Contradictions / refinements to the project plan

1. Plan says "TF-IDF/info-theoretic reweighting": the hosts never use TF-IDF; use `vectorizers.transformers.InformationWeightTransformer` (their explicitly-preferred, information-theoretic TF-IDF replacement). Keep TF-IDF only as an ablation if at all.
2. Plan says "UMAP -> HDBSCAN": in every host example HDBSCAN runs on a *low-dimensional UMAP output* (2-D or the 5-D/min_dist=1e-4 recipe), never on raw embeddings, and ToponymyClusterer expects `clusterable_vectors` (low-D) *and* `embedding_vectors` (full-D) separately. Plan should specify both representations.
3. Hellinger applies to the token-count track only (optionally L1-normalize first per scipy2023; acme3 skips explicit normalization). Visual/caption embedding tracks should use cosine.
4. No example repo demonstrates outlier scoring; nearest precedent is densMAP-based commonplace/rare density separation (scipy2023) and HDBSCAN noise labeling — worth adding GLOSH/HDBSCAN `outlier_scores_` as the Tutte-native baseline for eval (a), since `fast_hdbscan`/`hdbscan` expose it.

### 6.5 Version pins observed (loose — the hosts pin almost nothing)

- scipy2023 env: `bokeh<2.5`, `panel==0.14.3`, `distributed!=2023.6.0`, `hdbscan`, `umap-learn`, `thisnotthat` (pre-datamapplot era).
- acme3 env (2024): unpinned `datamapplot`, `fast-hdbscan`, `umap-learn[plot]`, `vectorizers`, `glasbey`, `duckdb`, `hdbscan`, `thisnotthat`.
- tutorials requirements.txt: unpinned `umap`, `vectorizers`, `hdbscan`, `datamapplot`, `thisnotthat`, `bokeh`, `panel`.
- Implication: pin from the *toponymy* repo's `uv.lock`/`pyproject.toml` for our build, not from these example repos.
