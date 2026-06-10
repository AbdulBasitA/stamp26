# Toponymy deep-dive (source-of-truth: local clone @ git main == PyPI 0.5.2)

- Local clone: `/home/b3ali/projects/stamp26/repos/toponymy`, HEAD `4c82fe9` ("Bump version to 0.5.2", 2026-05-26). This commit IS the latest commit on `origin/main` (verified via `gh api repos/TutteInstitute/toponymy/commits`) and IS the latest PyPI release (`toponymy 0.5.2`). So **pip/uv install of `toponymy==0.5.2` == git main today**; no reason to install from git.
- Docs: https://toponymy.readthedocs.io ; deepwiki: https://deepwiki.com/TutteInstitute/toponymy
- All line numbers below refer to files in the local clone.

---

## 1. Exact current API

### `Toponymy` (toponymy/toponymy.py:100)

```python
Toponymy(
    llm_wrapper: LLMWrapper,                      # sync LLMWrapper OR AsyncLLMWrapper instance
    text_embedding_model: TextEmbedderProtocol,   # anything with .encode(texts, show_progress_bar=...)-> np.ndarray
    clusterer: Clusterer = ToponymyClusterer(),   # !! mutable default shared across instances — ALWAYS pass your own
    layer_class: Type[ClusterLayer] = ClusterLayerText,
    prompt_template: Dict[str, Any] = PROMPT_TEMPLATES,
    keyphrase_builder: KeyphraseBuilder = KeyphraseBuilder(),   # !! also a shared mutable default
    object_description: str = "objects",
    corpus_description: str = "collection of objects",
    lowest_detail_level: float = 0.0,
    highest_detail_level: float = 1.0,
    exemplar_delimiters: List[str] = ['    * "', '"\n'],
    verbose: Optional[bool] = None,
)
```

### `Toponymy.fit` (toponymy/toponymy.py:170) — the precomputed-embeddings entry point

```python
topic_model.fit(
    objects,               # List[str] — RAW TEXT (our 1,500 VLM captions). Python list, not np.array
    embedding_vectors,     # np.ndarray (N, D) — PRECOMPUTED full-dim embeddings (caption text embeddings
                           #   OR visual SigLIP2/V-JEPA2 vectors; see "space mixing" note below)
    clusterable_vectors,   # np.ndarray (N, 2..25) — low-dim, e.g. UMAP of embedding_vectors
    exemplar_method="central",                 # also "facility_location", "saturated_coverage", "random"
    keyphrase_method="information_weighted",   # also "central", "bm25", "saturated_coverage",
                                               #   "facility_location", "graph_cut"
    subtopic_method="central",                 # also "information_weighted", "facility_location",
                                               #   "saturated_coverage"  (fit_predict default is
                                               #   "facility_location" — inconsistent defaults)
)
```

**Inputs needed: BOTH raw documents (text captions) and precomputed embeddings.** Toponymy never embeds the documents itself — `embedding_vectors` and `clusterable_vectors` are passed in fully precomputed. The `text_embedding_model` is still **required** because it is used to (a) embed the *keyphrase vocabulary* (toponymy.py:286) if the keyphrase builder didn't, (b) embed *topic names* for the disambiguation pass (`embed_topic_names`, cluster_layer.py:188), (c) embed on-demand missing keyphrases (keyphrases.py:773).

**Space-mixing is safe**: exemplar selection compares `embedding_vectors` against layer centroids (`centroids_from_labels(labels, embedding_vectors)`) — purely in the space of `embedding_vectors` (visual OK). Keyphrase selection builds its centroid from *keyphrase vectors* only (keyphrases.py:788, weighted by counts), and subtopic selection works in *topic-name-embedding* space (subtopics.py:45). There is **no cross-space comparison**, so `embedding_vectors` may be SigLIP2/V-JEPA2 visual vectors while `text_embedding_model` is a sentence-transformer. (README confirms: the embedder "doesn't have to be the embedding model that our documents were embedded with".)

Fitted attributes: `topic_names_` (List[List[str]] per layer, layer 0 = finest), `topic_name_vectors_`, `cluster_layers_`, `cluster_tree_` (`{(layer,cluster): [(child_layer, child_cluster), ...]}`), `object_x_keyphrase_matrix_`, `keyphrase_list_`, `keyphrase_vectors_`, `topic_tree_` property. Per-document labels: `[layer.topic_name_vector for layer in topic_model.cluster_layers_]` (noise points = `"Unlabelled"`).

### `ToponymyClusterer` (toponymy/clustering.py:382)

```python
ToponymyClusterer(
    min_clusters: int = 6,
    min_samples: int = 5,
    base_min_cluster_size: Optional[int] = 10,   # finest layer min cluster size
    base_n_clusters: Optional[int] = None,       # overrides base_min_cluster_size (binary-search on tree)
    next_cluster_size_quantile: float = 0.85,    # growth rate of min_cluster_size between layers
    max_layers: Optional[int] = None,
    n_threads: int = -1,
    verbose=None,
)
clusterer.fit(clusterable_vectors, embedding_vectors, layer_class=ClusterLayerText, **layer_kwargs)
clusterer.fit_predict(...) -> (cluster_layers_, cluster_tree_)
```

Algorithm: fast_hdbscan KDTree + `parallel_boruvka` MST → linkage tree → repeatedly `condense_tree` with growing `min_cluster_size` (quantile-driven), one HDBSCAN-leaves layer per pass (`build_raw_cluster_layers`, clustering.py:94). Layer 0 is finest. For 1,500 clips, `base_min_cluster_size=10` typically yields ~40–90 leaf clusters and 2–3 layers. **If `Toponymy.fit` sees the clusterer already has `cluster_layers_`, it reuses them and skips clustering** (toponymy.py:207). Alternatives in the same module: `KMeansClusterer` (demo), `EVoCClusterer` (only if `evoc` installed). Custom clusterers subclass `Clusterer` and must produce `cluster_layers_` (list of layer-class instances built from label vectors + `centroids_from_labels`) and `cluster_tree_` (via `build_cluster_tree(list_of_label_arrays)`); label arrays use -1 for noise, 0..k-1 otherwise.

### Cluster layers (toponymy/cluster_layer.py)

- `ClusterLayerText(cluster_labels, centroid_vectors, layer_id, text_embedding_model=None, n_keyphrases=16, n_exemplars=8, n_subtopics=16, *_diversify_alpha=1.0, exemplar_delimiters=None, prompt_format="combined", prompt_template=None, verbose=None, **kwargs)` — line 390. Also accepts `object_to_text_function` (base-class kwarg, line 92) for non-text objects.
- `ClusterLayerSummaryText` — same, additionally produces `topic_summaries_` / `topic_explanations_` (needs `SUMMARY_PROMPT_TEMPLATES`; see pitfall P4).
- Per-layer pipeline (driven by `Toponymy.fit`): `make_exemplar_texts` → `make_keyphrases` → `make_subtopics` (layers>0) → `make_prompts` → `name_topics(llm, ...)` → disambiguation pass(es) → blank-name retry pass.

---

## 2. Characterization machinery

### Exemplars (toponymy/exemplar_texts.py)
- `"central"` → `diverse_exemplars`: nearest-to-centroid in `embedding_vectors` space + diversification (`diversify_max_alpha`).
- `"facility_location"` / `"saturated_coverage"` → `submodular_selection_exemplars` (apricot-select; custom `FacilityLocationSelection` re-implementation; cosine metric; clusters subsampled at 16,384).
- `"random"`.
- All return `(exemplar_texts_per_cluster, exemplar_indices_per_cluster)` — indices into the original object list (useful for our hover thumbnails / LLM-judge held-out evaluation).
- `object_to_text_function: Callable[[List[Any]], List[str]]` converts non-text objects to text at exemplar time. **Our captions are already strings, so we don't need it.** (Caution: if it IS set on layer 0, `Toponymy.fit` takes a different path and builds keyphrases ONLY from exemplar texts, toponymy.py:246–269.)

### Keyphrases (toponymy/keyphrases.py)
- `KeyphraseBuilder(object_to_text=None, ngram_range=(1,4), tokenizer=None, token_pattern="(?u)\\b\\w[-'\\w]+\\b", max_features=50_000, min_occurrences=2, stop_words=ENGLISH_STOP_WORDS, n_jobs=-1, embedder=None, verbose=None)` — line 427.
- `fit_transform(objects) -> (object_x_keyphrase_matrix [scipy.sparse, counts], keyphrase_list, keyphrase_vectors|None)`. **Runs on raw text** (CountVectorizer-style n-grams, or an HF `tokenizers` tokenizer via `create_tokenizers_ngrammer`). If `keyphrase_vectors` is None, `Toponymy.fit` embeds `keyphrase_list` with `text_embedding_model`.
- Per-cluster selection methods (`ClusterLayerText.make_keyphrases`, cluster_layer.py:640): `information_weighted` (default; `vectorizers.InformationWeightTransformer` reweighting — same info-theoretic flavor as our bag-of-objects plan), `central`, `bm25`, and submodular `saturated_coverage|facility_location|graph_cut`.

**Injecting custom "keyphrases" (metadata tokens like `night`, `rain`, `rear-end`):**
1. *Easiest*: append a metadata sentence to each caption before `fit` (e.g. `"... Conditions: night, rain, urban intersection, rear-end."`). Tokens then appear in keyphrase counts AND in exemplar texts. (Watch `min_occurrences=2` and stop-words; multiword tokens like "rear-end" survive the default token_pattern because `-` is allowed inside words.)
2. *Clean injection*: `Toponymy.fit` only calls `keyphrase_builder.fit_transform(objects)` — duck typing means **any object with that method works**. Supply a custom builder that returns your own `(csr_matrix N×K of metadata-token counts/weights, token_list, None)`; selection then runs over your tokens only (or hstack with the text-derived matrix to mix). Note issue #107: the builder is *always* refit inside `fit`, so make `fit_transform` ignore its argument / be idempotent.
3. Manual: pre-fit clusterer, call `layer.make_keyphrases(...)` yourself, or simply overwrite `layer.keyphrases` (List[List[str]] per cluster) after calling it and before `make_prompts`.

### Subtopics (toponymy/subtopics.py)
For layer i>0, "misc subtopics" are layer-0 topic *names* whose clusters fall inside the layer-i cluster, selected by `central` / `information_weighted` / submodular in topic-name-embedding space. Separately, prompt construction pulls **major** subtopics (= names of direct children one layer down in `cluster_tree_`) and **minor** (two layers down) — prompt_construction.py:348–362. `ClusterLayerSummaryText` additionally threads each child's `topic_summary`/`topic_explanation` into `<MAJOR_SUBTOPIC>` blocks.

### Injecting per-cluster metadata histograms into prompts
There is **no native hook**: `topic_name_prompt` (prompt_construction.py:269) builds a fixed `render_params` dict, so custom Jinja templates cannot receive new variables. Practical options (ranked for a 2–3 day project):
1. **Histogram-as-keyphrases**: custom keyphrase builder emitting tokens; or directly overwrite `layer.keyphrases[c]` with strings like `"night (62% vs 18% base)"` in a manual step-by-step run. They render under "Keywords for this group include:".
2. **Manual prompt post-edit**: run the stages yourself (clusterer → make_exemplar_texts/make_keyphrases/make_subtopics → `layer.make_prompts(...)`), then mutate `layer.prompts[c]` (str, or `{"system","user"}` dict) to append a `"- Metadata histogram for this group: ..."` block, then call `layer.name_topics(...)`. `name_topics` consumes `self.prompts` as-is.
3. **Subclass `ClusterLayerText`** with an extra ctor kwarg (e.g. `metadata_per_object=`) and an overridden `make_prompts` that appends the histogram; pass via `Toponymy`'s clusterer path — `clusterer.fit_predict(..., layer_class=MyLayer, metadata_per_object=df)` forwards `**layer_kwargs` to the layer constructor (clustering.py:342). This keeps the one-call `Toponymy.fit` workflow.

---

## 3. PROMPT_TEMPLATES system

Location: `toponymy/templates.py` — plain module-level dicts of `jinja2.Template` objects (NOT files). Three stock sets: `PROMPT_TEMPLATES`, `SUMMARY_PROMPT_TEMPLATES`, `MULTILINGUAL_EN_FR_PROMPT_TEMPLATES`. Pass a dict of the same shape as `Toponymy(prompt_template=...)`.

Required dict shape (cluster_layer.py accesses both sections unconditionally when `prompt_template` is not None — **you must provide BOTH**, including the extractor callables):

```python
MY_TEMPLATES = {
    "layer": {
        "system":  jinja2.Template(...),    # used when llm_wrapper.supports_system_prompts (prompt_format="system_user")
        "user":    jinja2.Template(...),
        "combined": jinja2.Template(...),   # used when prompt_format == "combined"
        "extract_topic_name": lambda json_response: str(json_response["topic_name"]),
        "get_topic_name_regex": GET_TOPIC_NAME_REGEX,   # regex that locates the JSON blob in raw LLM output
    },
    "disambiguate_topics": {
        "system": ..., "user": ..., "combined": ...,
        "extract_topic_names": default_extract_topic_names,   # fn(json_response, old_names, raw_text) -> List[str]
        "get_topic_names_regex": GET_TOPIC_CLUSTER_NAMES_REGEX,
    },
}
```

Stock regexes (templates.py:14-18):
```python
GET_TOPIC_NAME_REGEX = r'\{\s*"topic_name":\s*.*?,\s*"topic_specificity":\s*[\w.]+\s*\}'
GET_TOPIC_NAME_AND_SUMMARY_REGEX = r'\{\s*"topic_name":\s*.*?,\s*"topic_summary":\s*.*?,\s*"topic_explanation":\s*.*?,\s*"topic_specificity":\s*[\w.]+\s*\}'
GET_TOPIC_CLUSTER_NAMES_REGEX = r'\{\s*"new_topic_name_mapping":\s*.*?,\s*"topic_specificities": .*?\}'
```

**Template variables received** (fixed; rendered in `topic_name_prompt` / `distinguish_topic_names_prompt` / `topic_summary_prompt`, prompt_construction.py:387-403 / 232-250 / 564-580):

"layer" templates:
- `document_type` (= `object_description`), `corpus_description`
- `cluster_keywords` — List[str] of selected keyphrases for this cluster
- `cluster_subtopics` — dict `{"major": [...], "minor": [...], "misc": [...]}` (names of child clusters; empty at layer 0)
- `cluster_sentences` — List[str] exemplar texts (our captions)
- `exemplar_start_delimiter`, `exemplar_end_delimiter`
- `summary_kind` — one of 7 `SUMMARY_KINDS` strings chosen by detail level, e.g. "domain expert level (8 to 15 word)" ... "simple (1 or 2 word)"
- `is_very_specific_summary`, `is_general_summary`, `has_major_subtopics` — bools

"disambiguate_topics" templates additionally/instead:
- `larger_topic` — string joined from the clashing names
- `topics` — List[str] of current (similar) topic names; all of `cluster_keywords` / `cluster_subtopics[...]` / `cluster_sentences` become **lists-per-topic**, indexed in templates via `loop.index - 1`.

Expected LLM output contract: `{"topic_name": <NAME>, "topic_specificity": <SCORE 0..1>}` for naming; `{"new_topic_name_mapping": {"1. OLD": "NEW", ...}, "topic_specificities": [...]}` for disambiguation (keys `"1. OLD_NAME"` or `"1."` accepted by `default_extract_topic_names`, templates.py:21).

**Custom-domain / taxonomy-constrained naming (NHTSA pre-crash typology):** write your own "layer" `system`/`user` templates that (a) enumerate the allowed typology labels in the system prompt, (b) instruct output as e.g. `{"topic_name": "<TYPOLOGY>: <free-text qualifier>", "topic_specificity": ...}`. Keep the same JSON keys so the stock regex/extractor work; otherwise also override `get_topic_name_regex`/`extract_topic_name`. Two caveats:
- **Detail level**: with multiple layers, layer prompts differ only by `summary_kind`; if you want the taxonomy applied only at a given layer you must condition on `summary_kind` inside the template (no `layer_id` variable is exposed).
- **Disambiguation conflicts with a fixed taxonomy** (issue #101, open): duplicate names trigger LLM renaming, and a >2 duplication triggers a second pass (cluster_layer.py:559–578). For strict-taxonomy mode either (i) keep composite names "TAXONOMY: qualifier" so duplicates are rare, (ii) set `"extract_topic_names": lambda info, old, raw: old` in your `disambiguate_topics` section (LLM calls still happen but names are kept), or (iii) subclass `ClusterLayerText` and override `disambiguate_topics` to a no-op (zero extra calls; cleanest).

`prompt_format` is chosen automatically: `"system_user"` if `llm_wrapper.supports_system_prompts` else `"combined"` (toponymy.py:151).

---

## 4. LLM backends (toponymy/llm_wrappers.py, 4,733 lines)

Sync (`LLMWrapper`): `LlamaCppNamer` (no system prompts), `HuggingFaceNamer`, `VLLMNamer` (in-process `vllm.LLM`), `CohereNamer`, `AnthropicNamer`, `OpenAINamer`, `TogetherNamer`, `ReplicateNamer`, `AzureAINamer`, `OllamaNamer`, `GoogleGeminiNamer`, `LiteLLMNamer`.
Async (`AsyncLLMWrapper`): `AsyncHuggingFaceNamer`, `AsyncVLLMNamer` (both "essentially for testing"), `AsyncCohereNamer`, `CohereBatchNamer`, `AsyncAnthropicNamer`, `BatchAnthropicNamer`, `AsyncOpenAINamer`, `AsyncTogether`, `AsyncAzureAINamer`, `BatchAzureAINamer`, `AsyncOllamaNamer`, `AsyncGoogleGeminiNamer`, `AsyncLiteLLMNamer`. All gated by try/import; missing deps yield `FailedImport*` stubs that raise on construction.

### Pointing at a LOCAL vLLM OpenAI-compatible server — YES, three routes

**(a) Sync `OpenAINamer` — works as documented** (llm_wrappers.py:2665-2682; passes `base_url` to `openai.OpenAI`):
```python
from toponymy.llm_wrappers import OpenAINamer
llm = OpenAINamer(api_key="EMPTY", model="Qwen/Qwen3-32B-AWQ",   # must equal --served-model-name
                  base_url="http://localhost:8000/v1")
llm.test_llm_connectivity()
```

**(b) Async `AsyncOpenAINamer` — BUG in 0.5.2**: the constructor accepts and documents `base_url`, but line 2798 is
```python
self.client = openai.AsyncOpenAI(api_key=api_key, organization=organization)   # base_url NOT passed!
```
Workarounds (any one):
```python
# w1: env var — openai-python v1 reads OPENAI_BASE_URL
os.environ["OPENAI_BASE_URL"] = "http://localhost:8000/v1"
llm = AsyncOpenAINamer(api_key="EMPTY", model="Qwen/Qwen3-32B-AWQ", max_concurrent_requests=32)
# w2: replace the client after construction
llm = AsyncOpenAINamer(api_key="EMPTY", model="...")
llm.client = openai.AsyncOpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
```

**(c) `AsyncLiteLLMNamer` — `api_base` honored** (llm_wrappers.py:4541-4570; needs `litellm>=1.83.10` installed):
```python
from toponymy.llm_wrappers import AsyncLiteLLMNamer
llm = AsyncLiteLLMNamer(model="hosted_vllm/Qwen/Qwen3-32B-AWQ",
                        api_base="http://localhost:8000/v1",
                        api_key="EMPTY", max_concurrent_requests=32)
```
(`OllamaNamer(host=...)` is also fully local if we ever fall back to Ollama.)

### JSON mode / token budgets / retries
- Both OpenAI wrappers ALWAYS send `response_format={"type": "json_object"}`. vLLM's OpenAI server supports this (structured outputs); make sure the vLLM version on the box does (any 2025+ vLLM is fine). LiteLLM wrapper auto-detects support and can be forced with `use_json_object=True/False`.
- `max_tokens` defaults: **128** for topic-name calls, 1024 for disambiguation, 2048 for `ClusterLayerSummaryText`. **Reasoning/thinking models will eat the 128-token budget and silently return ""** (open issue #155). With vLLM serve a non-thinking instruct model, or disable thinking at the server (a 2026-05-21 commit *removed* `chat_template_kwargs` from the OpenAI calls, so it cannot be set per-request from Toponymy; `llm_specific_instructions="/no_think"` is a Qwen-style fallback).
- Async batching: per-prompt `asyncio.gather` bounded by `asyncio.Semaphore(max_concurrent_requests)` (default 10 — tuned for cloud rate limits; for local vLLM raise to 32–64). Retry: tenacity, 3 attempts, `wait_random_exponential(min=1,max=10)`; `FAIL_FAST_EXCEPTIONS` (auth/404/422...) abort immediately. Failed name generations return `""` and get one final retry sweep at the end of `name_topics`.
- Sync wrappers process prompts serially with the same 3-attempt retry. With ~50–100 clusters and a local GPU server, async is preferred but sync is fine too (hundreds of calls).
- Jupyter: `run_async` (cluster_layer.py:46) uses `nest_asyncio` if installed, else a thread fallback — no action needed.
- Debug: `callback=` (DebugCallback) on OpenAI/LiteLLM wrappers emits `llm_call_success`/`llm_call_error` events with prompt + raw response — useful for our name-trustworthiness audit; see also `toponymy/audit.py` (`create_audit_df`, `create_comparison_df`, AUDIT_USAGE.md).

---

## 5. Install

- `pyproject.toml`: `toponymy 0.5.2`, `requires-python >=3.10`, classifiers 3.10–3.13 → **Python 3.12.3 supported**.
- Base deps: `numpy>=1.21, pandas>=1.0, numba>=0.56, datasets, scikit-learn>=1.6, vectorizers, scipy, fast_hdbscan>=0.3.2, tqdm, tenacity, httpx, apricot-select, jinja2, transformers>=4.41,<5, tokenizers`.
- Extras: `dev` (openai, litellm>=1.83.10, sentence-transformers<=5.1.2, evoc, umap-learn, anthropic, ollama, ...), `llama` (llama-cpp-python), `example-notebooks` (datamapplot>=0.2.2, ...). For us only `openai` (and optionally `litellm`) is needed on top of base + `sentence-transformers`.
- Recommended: `uv add toponymy==0.5.2 openai sentence-transformers umap-learn datamapplot` (pin! — see issue #149 below). PyPI == git main today, so no git install needed; a `uv.lock` exists in the repo if we ever need their exact resolution.
- Air-gap note: only LLM/embedding calls go to our local services; but `sentence-transformers`/HF models must be pre-downloaded (`HF_HUB_OFFLINE=1` after caching). `transformers<5` pin matters if we share a venv with new VLM stacks — keep Toponymy in its own venv if our captioner needs transformers v5.

## 6. Known issues & pitfalls (gh issue scan + source verification)

GitHub issues (TutteInstitute/toponymy):
- **#155 (open, 2026-06-01)**: empty/malformed LLM responses pass silently — exactly the reasoning-model + `max_tokens=128` failure. Mitigate: instruct (non-thinking) model for naming; smoke-test with `connectivity_status()`; use the debug `callback` to log raw responses.
- **#149 (open, 2026-05-21)**: "COMING SOON: ToponymyClusterer refactor" — clustering will be outsourced to fast_hdbscan/evoc (PLSCANClusterer default) and **behavior will change**. → **Pin `toponymy==0.5.2`** for the workshop.
- **#131 (open)**: pipeline design discussion; confirms intermediate inspection is awkward and `ClusterLayer` holds runtime config (motivates our manual step-by-step variant for prompt injection).
- **#128 (closed)**: pre-fit clusterer bypassed wrapper `prompt_format`. Fixed by `_sync_layer_runtime_config` (toponymy.py:157) — but **it syncs only `prompt_format`/`exemplar_delimiters`/verbosity, NOT `prompt_template`**. ⇒ **P1: with a PRE-FIT clusterer, a custom `prompt_template` passed to `Toponymy` is silently ignored at the prompt-construction level** (layers keep `prompt_template=None` → stock templates), while extractor functions DO come from your dict (mismatched!). Fix: let `Toponymy.fit` do the clustering, or pass `prompt_template=MY_TEMPLATES` as a layer_kwarg to `clusterer.fit(...)`, or set `layer.prompt_template = MY_TEMPLATES` on each pre-fit layer.
- **#107 (open)**: `Toponymy.fit` always refits the keyphrase builder — relevant if we precompute keyphrases (make our custom builder idempotent).
- **#101 (open)**: disambiguation cannot be disabled — conflicts with taxonomy-constrained naming (workarounds in §3).
- **#57 (open)**: IndexError during disambiguation at 1k–2k docs — **our exact corpus size**. The try/except fix from the thread IS in current source (`_update_topic_names`, cluster_layer.py:277-281), so mostly mitigated, but disambiguation remains the flakiest stage; budget time for it.
- **#56 (open)**: batch/concurrent calls require the Async wrappers (sync = serial).
- #135/#134/#133/#106 (closed): dependency breakage waves — fixed by 0.5.1/0.5.2; another reason to pin.

Source-verified pitfalls:
- **P2 — shared mutable defaults** (toponymy.py:104,107): `clusterer=ToponymyClusterer()` / `keyphrase_builder=KeyphraseBuilder()` are evaluated once; a second `Toponymy` built with defaults reuses the FIRST run's fitted cluster layers (incl. cached topic names). Always construct fresh `ToponymyClusterer()` per model — critical for our rerun-stability evaluation.
- **P3 — `AsyncOpenAINamer` drops `base_url`** (llm_wrappers.py:2798) — see §4 workarounds.
- **P4 — `ClusterLayerSummaryText` auto-template switch is dead code** (toponymy.py:135-139: `isinstance(layer_class, ...)` on a *class* is always False, and `SUMMARY_PROMPT_TEMPLATES` isn't even imported). If we want per-cluster summaries+explanations (nice for LLM-as-judge), pass BOTH `layer_class=ClusterLayerSummaryText` and `prompt_template=SUMMARY_PROMPT_TEMPLATES` (import from `toponymy.templates`) explicitly.
- **P5** — objects should be a plain Python `list` of `str` (README warns against numpy arrays).
- **P6** — keyphrase defaults assume English text (`ENGLISH_STOP_WORDS`, `min_occurrences=2`); short repetitive VLM captions are fine, but consider `ngram_range=(1,3)` and a domain stoplist ("video", "dashcam", "footage").

## 7. Minimal realistic sketch for our use case

```python
# deps: toponymy==0.5.2, openai, sentence-transformers, umap-learn
# vLLM server (separate process, e.g. 1-2 of the 4090s):
#   vllm serve Qwen/Qwen3-32B-AWQ --port 8000 --served-model-name namer \
#        --max-model-len 16384   # plus --reasoning off / pick a non-thinking instruct model
import json, jinja2, numpy as np, umap
from sentence_transformers import SentenceTransformer
from toponymy import Toponymy, ToponymyClusterer
from toponymy.llm_wrappers import AsyncOpenAINamer            # or OpenAINamer (sync) — both fine at ~100 calls
from toponymy.templates import GET_TOPIC_NAME_REGEX, GET_TOPIC_CLUSTER_NAMES_REGEX, default_extract_topic_names
import openai, os

# ---------- inputs ----------
captions: list[str] = json.load(open("captions.json"))        # 1,500 VLM captions (+ appended metadata sentence:
                                                              # "Conditions: night, rain, urban, rear-end.")
embedding_vectors = np.load("clip_embeddings.npy")            # (1500, D): caption-text embeddings OR SigLIP2/V-JEPA2
clusterable_vectors = umap.UMAP(n_components=5, metric="cosine",
                                n_neighbors=15, random_state=42).fit_transform(embedding_vectors)

# ---------- local services ----------
text_embedder = SentenceTransformer("all-MiniLM-L6-v2")        # any local ST model; used for keyphrases/topic names
os.environ["OPENAI_BASE_URL"] = "http://localhost:8000/v1"     # workaround for AsyncOpenAINamer base_url bug (P3)
llm = AsyncOpenAINamer(api_key="EMPTY", model="namer", max_concurrent_requests=32)
# llm.client = openai.AsyncOpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")  # belt-and-braces

# ---------- custom NHTSA-aligned templates ----------
NHTSA = ["rear-end", "lane change / sideswipe", "left turn across path", "straight crossing path",
         "pedestrian", "pedalcyclist", "road departure", "animal", "backing", "other"]
LAYER_SYSTEM = jinja2.Template("""
You are an expert traffic-safety analyst classifying {{document_type}} from {{corpus_description}}.
Assign a {{summary_kind}} name to this group. The name MUST start with the best-matching NHTSA
pre-crash type from: """ + "; ".join(NHTSA) + """, followed by a colon and a short qualifier
covering scene, conditions and behavior (e.g. "rear-end: night highway, sudden braking ahead").
If the clips show no conflict, use "other: <description>".
Respond ONLY with JSON: {"topic_name":<NAME>, "topic_specificity":<SCORE between 0.0 and 1.0>}.
""")
LAYER_USER = jinja2.Template("""
Information about this group of {{document_type}}:
{% if cluster_keywords %}- Keywords: {{", ".join(cluster_keywords)}}{% endif %}
{% if cluster_subtopics["major"] %}- Major subtopics:{% for s in cluster_subtopics["major"] %}
  * {{s}}{% endfor %}{% endif %}
{% if cluster_sentences %}- Sample {{document_type}}:{% for s in cluster_sentences %}
{{exemplar_start_delimiter}}{{s}}{{exemplar_end_delimiter}}{% endfor %}{% endif %}
Provide the {{summary_kind}} name now. Output format: {"topic_name":<NAME>, "topic_specificity":<SCORE>}.
""")
MY_TEMPLATES = {
    "layer": {
        "system": LAYER_SYSTEM, "user": LAYER_USER, "combined": LAYER_USER,  # combined unused (vLLM supports system)
        "extract_topic_name": lambda r: str(r["topic_name"]),
        "get_topic_name_regex": GET_TOPIC_NAME_REGEX,
    },
    "disambiguate_topics": {                                   # keep stock behavior (or neutralize, see notes §3)
        "system": __import__("toponymy.templates", fromlist=["PROMPT_TEMPLATES"]).PROMPT_TEMPLATES["disambiguate_topics"]["system"],
        "user":   __import__("toponymy.templates", fromlist=["PROMPT_TEMPLATES"]).PROMPT_TEMPLATES["disambiguate_topics"]["user"],
        "combined": __import__("toponymy.templates", fromlist=["PROMPT_TEMPLATES"]).PROMPT_TEMPLATES["disambiguate_topics"]["combined"],
        "extract_topic_names": default_extract_topic_names,    # OR: lambda info, old, raw: old  -> taxonomy-strict
        "get_topic_names_regex": GET_TOPIC_CLUSTER_NAMES_REGEX,
    },
}

# ---------- fit ----------
clusterer = ToponymyClusterer(min_clusters=6, base_min_cluster_size=15, verbose=True)  # fresh instance! (P2)
topic_model = Toponymy(
    llm_wrapper=llm,
    text_embedding_model=text_embedder,
    clusterer=clusterer,                                     # NOT pre-fit -> custom templates reach the layers (P1)
    prompt_template=MY_TEMPLATES,
    object_description="dashcam driving-incident video clips (described by captions)",
    corpus_description="the Nexar collision-prediction dashcam dataset",
    exemplar_delimiters=["<CLIP_CAPTION>\n", "\n</CLIP_CAPTION>\n"],
    verbose=True,
)
topic_model.fit(list(captions), embedding_vectors, clusterable_vectors,
                exemplar_method="central", keyphrase_method="information_weighted")

print(topic_model.topic_names_[-1])                          # coarsest layer
labels_per_layer = [l.topic_name_vector for l in topic_model.cluster_layers_]  # for datamapplot layers
exemplar_idx_layer0 = topic_model.cluster_layers_[0].exemplar_indices          # for held-out LLM-judge eval
# datamapplot.create_interactive_plot(map_2d, *labels_per_layer[::-1], hover_text=captions, ...)
```

Variant for metadata-histogram prompts (manual stage control), if needed:
```python
clusterer.fit(clusterable_vectors, embedding_vectors, layer_class=ClusterLayerText,
              prompt_format="system_user", prompt_template=MY_TEMPLATES,
              exemplar_delimiters=["<CLIP_CAPTION>\n", "\n</CLIP_CAPTION>\n"])
# ... per layer: make_exemplar_texts / make_keyphrases / make_subtopics / make_prompts,
# then for c, p in enumerate(layer.prompts): layer.prompts[c]["user"] += hist_block(c)
# then layer.name_topics(llm, ...)
```

## 8. Sizing/runtime expectations for our project
- 1,500 docs → clustering is sub-second; keyphrase build seconds; total LLM calls ≈ (#clusters across layers ≈ 60–120) + disambiguation batches (≈5–20) + blank retries. At 32-way concurrency on a local vLLM 32B this is ~1–3 minutes.
- `lowest_detail_level=0, highest_detail_level=1` maps layers across the 7 `SUMMARY_KINDS`; with only 2–3 layers, layer 0 gets "domain expert level (8 to 15 word)" names — good for qualifier-rich NHTSA names; raise `lowest_detail_level` (~0.4) for shorter names.
- For rerun-stability eval: fresh `ToponymyClusterer` per run (P2); clustering itself is deterministic-ish (`reproducible=False` in boruvka — set seeds and expect minor nondeterminism; UMAP `random_state` pins the map).
