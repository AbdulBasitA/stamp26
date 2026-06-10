# Environment Setup Design — STAMP 2026 Nexar/Toponymy Project

Date: 2026-06-09. Target machine: Ubuntu (kernel 6.17.0-23), NVIDIA driver **590.48.01 / CUDA 13.1**, 4x RTX 4090 (sm_89, 24GB, no NVLink/P2P), 32 cores, system Python **3.12.3**, **uv 0.9.21**, `/home` = `/dev/md126` **RAID0** (4-disk imsm, ext4, 28T, 94% used, **1.7T free**, inodes 36%).

All facts below verified 2026-06-09 against local files, `nvidia-smi`, PyPI, and upstream repos.

---

## 1. Verified version landscape (June 2026)

| Component | Current (verified) | Notes |
|---|---|---|
| vLLM | **0.22.1** (PyPI, released 2026-06-05) | Pins `torch==2.11.0`, `torchvision==0.26.0`, `torchaudio==2.11.0`, `flashinfer-python==0.6.11.post2`, `flashinfer-cubin==0.6.11.post2` ([v0.22.1 requirements/cuda.txt](https://raw.githubusercontent.com/vllm-project/vllm/v0.22.1/requirements/cuda.txt)). Default PyPI wheel is **CUDA 13.0** since v0.20.0. transformers constraint (main): `>=4.56.0, !=5.0.*..!=5.4.*, !=5.5.0`. Python `>=3.10,<3.15`. |
| PyTorch | **2.12.0** stable | Wheel matrix: **cu130 (default on PyPI) + cu126 only — cu128 was REMOVED in 2.12**. cu130 requires driver >= 580; our 590.48 (CUDA 13.1) is fine. sm_89 (4090) fully supported by CUDA 13. ([pytorch releases](https://github.com/pytorch/pytorch/releases)) |
| transformers | **v5.x** (v5.0.0 released 2026-01-26; tags up to ~v5.10.x exist). Last 4.x line = **4.57.x** | v5 requires PyTorch-only backend, hf-hub >=1.3,<2. **Toponymy pins `transformers<5.0.0`** → analysis env stays on 4.57.6. |
| toponymy | **0.5.2** (PyPI 2026-05-26 == local clone) | `requires-python>=3.10`; py3.12 classifier. Deps: numba>=0.56, fast_hdbscan>=0.3.2, vectorizers, apricot-select, jinja2, `transformers>=4.41,<5.0`, tenacity, httpx, datasets. **No torch dep itself.** |
| datamapplot | **0.7.3** (PyPI 2026-05-31 == local clone) | New dep: `dask[complete]>=2026.1.2`, `numpy>=2.0`. Ships `dmp_offline_cache` console script. **No selenium anywhere in the repo** (grep verified). |
| evoc / fast_hdbscan / vectorizers | 0.3.1 / 0.3.2 / 0.2.2 (local clones == releases) | All py3.10–3.13 classifiers; pure numba/sklearn stacks. |
| numba / llvmlite | **0.65.1 / 0.47.0** | Proven on py3.12: toponymy's own `uv.lock` resolves numba 0.65.1 + llvmlite 0.47.0 + numpy 2.2.6 on py3.12. py3.12 supported since numba 0.59; 0.63 (Dec 2025) and 0.65 (free-threading support) current. **No py3.12 blocker.** |
| umap-learn | 0.5.12 | per toponymy uv.lock. |
| sentence-transformers | 5.1.2 | toponymy caps dev extra at `<=5.1.2`; lock resolves 5.1.2 — works with transformers 4.57.6 and torch 2.12. |
| flash-attn | 2.8.3 (Jan 2026) | Official wheels cover torch 2.4–2.9 + CUDA <=12.8 only. **No official cu13/torch-2.11+ wheels**; community wheels exist ([mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels), [Dao-AILab issue #2442](https://github.com/Dao-AILab/flash-attention/issues/2442) for cu13+torch2.11+py312). **Recommendation: do not install flash-attn at all** (see §3). |
| huggingface_hub | 0.36.2 in 4.x-transformers env; 1.x in vllm env | `hf` CLI exists in both (since 0.34). `huggingface-cli` removed in 1.x. |
| uv | 0.9.21 (installed) | `--torch-backend` verified locally: valid values include **cu130** (`uv pip install --help`). Note: `--torch-backend` works only in the `uv pip` interface, not `uv sync`/`uv lock`. |

Key local evidence file: `/home/b3ali/projects/stamp26/repos/toponymy/uv.lock` (resolved 2026-05: transformers 4.57.6, torch 2.12.0, numba 0.65.1, llvmlite 0.47.0, sentence-transformers 5.1.2, umap-learn 0.5.12, scikit-learn 1.7.2/1.8.0, fast-hdbscan 0.3.2) — this is a working py3.12 resolution from the Toponymy authors themselves.

---

## 2. Venv strategy: TWO venvs (not three), with an optional third

The proposed 3-way split (vllm / embeddings / analysis) is **partially refuted**: the embeddings env and the analysis env do **not** conflict and should be merged. The vllm env **must** stay separate.

**Why vllm must be isolated:** vLLM 0.22.1 hard-pins `torch==2.11.0` (+ exact torchvision/torchaudio/flashinfer), while the analysis stack freely resolves to torch 2.12.0. vLLM also drags ~80 pinned deps (outlines_core==0.2.14, compressed-tensors==0.17.0, llguidance, pydantic>=2.12...) and will resolve transformers to a 5.5.1+ version — **incompatible with toponymy's `transformers<5.0.0` pin**. One env for both is unresolvable.

**Why embeddings + analysis merge cleanly:** toponymy forces `transformers==4.57.x`, and both planned embedders are available well below that: SigLIP2 (transformers >= 4.49), V-JEPA2 (in transformers since ~4.53, verified present at v4.53.3 tag). torch is unconstrained by toponymy (no torch dep) → torch 2.12.0+cu130 works. numba/umap/datamapplot/numpy 2.x all co-resolve (proven by toponymy's uv.lock + datamapplot requiring numpy>=2.0). Fewer envs = less workshop friction; one Jupyter kernel sees embeddings AND clustering.

**Optional third env (`embed5`):** only if you adopt a 2026 model that requires transformers >= 5 (check the model card's "Transformers version" requirement). Recipe included below; don't build it preemptively.

| Env | Path | Purpose | Key pins |
|---|---|---|---|
| `vllm` | `/home/b3ali/projects/stamp26/venvs/vllm` | Serve VLM captioner + namer LLM + judge (OpenAI-compatible) | `vllm==0.22.1` → torch 2.11.0+cu13 auto |
| `main` | `/home/b3ali/projects/stamp26/venvs/main` | Frame extraction, SigLIP2/V-JEPA2 embedding, UMAP/HDBSCAN/EVoC, Toponymy, DataMapPlot, Jupyter, eval | torch 2.12.0 cu130, transformers 4.57.6, Tutte libs `-e` from local clones |
| (`embed5`) | `/home/b3ali/projects/stamp26/venvs/embed5` | ONLY if an embedder needs transformers>=5 | torch 2.12.0, transformers>=5.5.1 (vLLM-compatible range), NO toponymy |

The analysis code talks to vLLM **only over HTTP** (toponymy `OpenAINamer`/`AsyncOpenAINamer` accept `base_url` — verified in `/home/b3ali/projects/stamp26/repos/toponymy/toponymy/llm_wrappers.py` line ~2610/2721). Do NOT use toponymy's in-process `VLLMEmbedder`/`VLLMNamer` classes (they `import vllm` → would force vllm into the main env).

Use plain `uv venv` + `uv pip` (not a uv *project*): `--torch-backend` is uv-pip-only, the Tutte libs are editable local clones, and we want two independent envs without workspace gymnastics.

---

## 3. PyTorch wheels, CUDA, flash-attn

- **Driver 590.48 / CUDA 13.1** runs cu130 wheels natively (CUDA minor-version compatibility) and cu126 wheels via backward compat. **Choose cu130 for both envs** — it is the PyPI default for torch 2.11 and 2.12, so `uv pip install vllm` / `uv pip install torch` need no index override at all. cu128 is dead in torch 2.12 — do not pin cu128 index URLs anywhere.
- **vllm env:** never install torch manually; let `vllm==0.22.1` pull `torch==2.11.0` (cu13 build) from PyPI. Overriding torch in a vllm env breaks its precompiled kernels.
- **main env:** `uv pip install --torch-backend=cu130 torch==2.12.0 torchvision==0.27.0` (torchvision pairs 0.27 with 2.12; omit the pin and uv resolves the right one). `--torch-backend=auto` also works (detects driver 590 → picks cu130).
- **flash-attn: NOT needed. Skip it.**
  - vLLM ships its own precompiled attention (vendored vllm-flash-attn + FlashInfer via the pinned `flashinfer-cubin` package — cubins bundled as a wheel, **no runtime kernel downloads**, good for air-gap). On sm_89 vLLM auto-selects a working backend; FA3 is Hopper-only anyway.
  - For transformers-side SigLIP2 / V-JEPA2 inference, use `attn_implementation="sdpa"` — PyTorch SDPA uses fused flash kernels internally on sm_89; for ~1,500 clips of batch inference the difference vs flash-attn2 is noise.
  - If someone insists later: community wheels at `https://github.com/mjun0812/flash-attention-prebuild-wheels` (flash_attn 2.8.3, cu130, torch 2.11/2.12, py312, linux x86_64). Official Dao-AILab wheels stop at torch 2.9/cu128. Never build from source during a 2-day workshop (~1h+ build, 32 cores).

Multi-GPU note: 4090s have **no P2P**; if a tensor-parallel vLLM launch hangs in NCCL init, set `NCCL_P2P_DISABLE=1`. Suggested topology: captioner VLM on GPUs 0–1, namer/judge LLM on GPUs 2–3 (`CUDA_VISIBLE_DEVICES` per server, two `vllm serve` processes, different ports).

---

## 4. Tutte libraries: install editable from local clones

Local clones are **exactly at the latest released versions** (toponymy 4c82fe9 = v0.5.2 2026-05-26; datamapplot 5d03113 = v0.7.3 2026-05-31), so `-e` installs == PyPI versions but patchable mid-workshop (likely needed: Jinja prompt templates live in `toponymy/templates.py`, hover-thumbnail HTML in datamapplot). Install order doesn't matter if done in one `uv pip install` command (uv resolves jointly). `vectorizers`, `fast_hdbscan`, `evoc` are tiny pure-python/numba packages.

Conflict check (all verified against pyprojects in `/home/b3ali/projects/stamp26/repos/*/pyproject.toml`):
- toponymy: `transformers>=4.41,<5` ← the only binding constraint; everything else floats.
- datamapplot 0.7.3 newly requires `dask[complete]>=2026.1.2` and `numpy>=2.0` — fine.
- numba 0.65.1 supports numpy 2.x on py3.12 (lock-proven). No `umap-learn`/`pynndescent` py3.12 issues in 2026.
- No Tutte lib depends on torch → no interaction with the torch 2.12 choice.

---

## 5. Disk, caching, layout

**Critical observations:**
- `/home` is **RAID0 (striped, zero redundancy)** across 4 disks (`/proc/mdstat`: md126, external imsm). One disk failure loses everything. Treat the box as scratch: push code/notes/results to remote git daily; the Nexar dataset and HF models are re-downloadable (pre-workshop only!).
- 94% used / 1.7T free, inodes fine (36%). Budget: dataset ~50G + frames cache 50–100G + VLM ~20G + namer 20–60G + embedders ~10G + new venvs ~25G (two torch stacks + vllm) + wheel cache growth ~20G ≈ **~250–300G → ends ~95–96% full**. Workable, but ext4 performance degrades when nearly full; do not let it cross ~97%.
- **`~/.cache/huggingface` already holds 522G** including reusable assets: `models--ibnzterrell--Meta-Llama-3.3-70B-Instruct-AWQ-INT4` (namer candidate!), several Qwen2.5-VL 7B variants, `openai/clip-vit-base-patch32`, `nvidia/Alpamayo-R1-10B`, `nvidia/PhysicalAI-Autonomous-Vehicles` dataset. **Do NOT relocate HF_HOME** — keep the default `~/.cache/huggingface` to reuse these and avoid duplicating tens of GB on a 94%-full disk. If space pressure hits, reclaim from clearly unrelated cache entries (fineweb-edu, msmarco, SWE-bench ≈ likely 100G+) — confirm with the box owner first.
- `~/.cache/uv` = 66G warm cache (also reused for offline reinstalls).

**Directory layout:**
```
/home/b3ali/projects/stamp26/
├── repos/          # existing clones (editable installs point here)
├── research/       # notes (this file)
├── venvs/          # vllm/, main/, (embed5/)
├── data/nexar/     # raw clips ~50G
├── frames/         # extracted frames + event-window caches (50–100G; prunable)
├── embeddings/     # .npy/.parquet per embedding family (small)
├── captions/       # VLM caption JSONL
├── runs/           # cluster results, maps, eval outputs, exported HTML
├── offline/        # dmp_cache.zip, requirements-*.txt freezes, wheelhouse/
└── pipeline/       # project code (git repo: repos/stamp2026)
```
(HF models: default `~/.cache/huggingface` — deliberate, see above.)

---

## 6. Exact ordered setup commands

```bash
# ── 0. Layout ────────────────────────────────────────────────────────────
mkdir -p /home/b3ali/projects/stamp26/{venvs,data/nexar,frames,embeddings,captions,runs,offline/wheelhouse}
export P=/home/b3ali/projects/stamp26

# ── 1. vLLM serving env (torch 2.11.0+cu13 pulled automatically) ─────────
uv venv $P/venvs/vllm --python 3.12
uv pip install -p $P/venvs/vllm vllm==0.22.1
# smoke test (downloads nothing):
$P/venvs/vllm/bin/python -c "import vllm, torch; print(vllm.__version__, torch.__version__, torch.cuda.is_available())"

# ── 2. Main env: embeddings + analysis + viz + jupyter ───────────────────
uv venv $P/venvs/main --python 3.12
uv pip install -p $P/venvs/main --torch-backend=cu130 \
    torch==2.12.0 torchvision
uv pip install -p $P/venvs/main \
    -e $P/repos/vectorizers \
    -e $P/repos/fast_hdbscan \
    -e $P/repos/evoc \
    -e $P/repos/toponymy \
    -e $P/repos/datamapplot \
    "transformers==4.57.6" "sentence-transformers==5.1.2" accelerate \
    umap-learn pynndescent \
    openai httpx \
    jupyterlab ipykernel anywidget \
    opencv-python-headless av pillow imageio[ffmpeg] \
    pandas pyarrow datasets tqdm matplotlib
$P/venvs/main/bin/python -m ipykernel install --user --name stamp26-main
# smoke test:
$P/venvs/main/bin/python -c "
import torch, transformers, numba, toponymy, datamapplot, evoc, fast_hdbscan, umap, vectorizers
print(torch.__version__, torch.cuda.is_available(), transformers.__version__, numba.__version__)"

# ── 3. Pre-download models (uses default HF_HOME=~/.cache/huggingface) ───
HF=$P/venvs/main/bin/hf
$HF download google/siglip2-so400m-patch16-384            # visual embedder (~4.5G)
$HF download facebook/vjepa2-vitl-fpc64-256               # video embedder (~1.3G)
$HF download <CAPTIONER-VLM-ID>                            # per model-selection agent (e.g. Qwen2.5-VL-7B-Instruct — variants already partly cached)
$HF download <NAMER-LLM-ID>                                # NOTE: Meta-Llama-3.3-70B-Instruct-AWQ-INT4 ALREADY in cache
$HF download nexar-ai/nexar_collision_prediction --repo-type dataset --local-dir $P/data/nexar   # verify repo id

# ── 4. DataMapPlot offline cache (needs network ONCE) ────────────────────
$P/venvs/main/bin/dmp_offline_cache                        # caches deck.gl@9.1, arrow, d3, jquery, d3-cloud, datatables + 20 base Google fonts
$P/venvs/main/bin/dmp_offline_cache --export $P/offline/dmp_cache.zip --yes   # portable copy

# ── 5. Freeze + offline-reinstall insurance ──────────────────────────────
uv pip freeze -p $P/venvs/vllm  > $P/offline/requirements-vllm.txt
uv pip freeze -p $P/venvs/main  > $P/offline/requirements-main.txt   # strip the "-e" lines or keep paths
# Option A (primary): warm uv cache already holds every wheel; verify:
uv venv /tmp/airgap-test --python 3.12
uv pip install -p /tmp/airgap-test --offline vllm==0.22.1 && rm -rf /tmp/airgap-test
# Option B (belt-and-suspenders wheelhouse):
$P/venvs/main/bin/python -m pip download -r $P/offline/requirements-main.txt \
    -d $P/offline/wheelhouse --no-deps
# reinstall later: uv pip install -p <env> --no-index --find-links $P/offline/wheelhouse -r ...

# ── 6. Warm-run BEFORE air-gap (surfaces any lazy downloads) ─────────────
CUDA_VISIBLE_DEVICES=2,3 $P/venvs/vllm/bin/vllm serve <NAMER-LLM-ID> \
    --tensor-parallel-size 2 --port 8001 --max-model-len 16384
# + one captioning request, one SigLIP2 forward, one V-JEPA2 forward, one
#   datamapplot offline_mode=True render, one toponymy run on dummy data.

# ── 7. Air-gap switches (set during workshop) ────────────────────────────
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
```

If `vllm serve` with TP=2 hangs at NCCL init (4090s lack P2P): `export NCCL_P2P_DISABLE=1`.

Optional env if (and only if) a chosen embedder demands transformers>=5:
```bash
uv venv $P/venvs/embed5 --python 3.12
uv pip install -p $P/venvs/embed5 --torch-backend=cu130 torch==2.12.0 torchvision "transformers>=5.5.1" accelerate
```

---

## 7. DataMapPlot offline mode — verified mechanics (datamapplot 0.7.3 source)

CDN dependencies hardcoded in `/home/b3ali/projects/stamp26/repos/datamapplot/datamapplot/offline_mode_caching.py` (`DEFAULT_URLS` / `DEFAULT_CSS_URLS`):
- `https://unpkg.com/deck.gl@9.1/dist.min.js` (pinned 9.1)
- `https://unpkg.com/apache-arrow@latest/Arrow.es2015.min.js`
- `https://unpkg.com/d3@latest/dist/d3.min.js`
- `https://unpkg.com/jquery@3.7.1/dist/jquery.min.js`
- `https://unpkg.com/d3-cloud@1.2.7/build/d3.layout.cloud.js`
- `https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js` + matching CSS
- Google Fonts API for fonts (20 BASE_FONTS pre-listed; add custom via `--font_names`)

Workflow: `dmp_offline_cache` (console script, in pyproject `[project.scripts]`) fetches and base64-caches JS/CSS/fonts into `platformdirs.user_data_dir("datamapplot")` (= `~/.local/share/datamapplot/` on Linux): `datamapplot_js_encoded.json`, `datamapplot_css_encoded.json`, `datamapplot_fonts_encoded.json`. Supports `--export`/`--import` (zip or dir) to move caches between machines — export a copy to `$P/offline/dmp_cache.zip`.

Rendering offline: `create_interactive_plot(..., offline_mode=True)` — `offline_mode`, `offline_mode_js_data_file`, `offline_mode_font_data_file` are `render_html()` kwargs forwarded through `**render_html_kwds` (verified `interactive_rendering.py` lines 521–523). JS/fonts get inlined into the HTML → fully self-contained file. `inline_data=True` (default) keeps point data inline too. There is **no selenium dependency** in 0.7.3 — static PNG export concerns from older plans don't apply; static plots are pure matplotlib. If using a custom `font_family` for the map, cache it first: `dmp_offline_cache --font_names "YourFont"`.

Note for hover thumbnails: base64 data-URI images in `hover_text`/custom HTML work offline by construction; keep thumbnails small (~10–20KB JPEG) or the single HTML will balloon (1,500 × 20KB ≈ 30MB — acceptable).

---

## 8. Air-gap checklist (everything that phones home, and its fix)

| Component | Network touch | Fix |
|---|---|---|
| HF models/datasets | first `from_pretrained`/`load_dataset` | pre-download via `hf download`; then `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` |
| vLLM | model fetch only; flashinfer kernels are bundled via `flashinfer-cubin` wheel (no JIT downloads) | warm-run each served model once |
| datamapplot | unpkg/datatables/Google-Fonts at render AND at view time | `dmp_offline_cache` + `offline_mode=True` |
| uv/pip installs | PyPI | warm `~/.cache/uv` (66G already) + `--offline` verified; wheelhouse fallback in `$P/offline/wheelhouse` |
| toponymy | none at runtime (sklearn CountVectorizer keyphrases, no nltk/spacy downloads — verified `keyphrases.py` imports); LLM calls go to local vLLM via `OpenAINamer(base_url="http://localhost:8001/v1", api_key="x")` | — |
| huggingface_hub update check | startup version check in 1.x | `HF_HUB_DISABLE_UPDATE_CHECK=1` in vllm env |

---

## 9. Sources

- vLLM PyPI (0.22.1, 2026-06-05): https://pypi.org/project/vllm/
- vLLM v0.22.1 cuda reqs (torch 2.11.0 pin): https://github.com/vllm-project/vllm/blob/v0.22.1/requirements/cuda.txt
- vLLM transformers constraint (main): https://github.com/vllm-project/vllm/blob/main/requirements/common.txt
- PyTorch 2.12 wheel matrix (cu128 removed, cu130 default, cu126 legacy): https://github.com/pytorch/pytorch/releases
- uv torch integration (`--torch-backend`, uv-pip-only): https://docs.astral.sh/uv/guides/integration/pytorch/
- flash-attn official wheel coverage + cu13 community wheels: https://github.com/Dao-AILab/flash-attention/issues/2442 , https://github.com/mjun0812/flash-attention-prebuild-wheels
- transformers v5 (2026-01-26) + migration: https://huggingface.co/blog/transformers-v5 , https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md
- V-JEPA2 in transformers (present at v4.53.3): https://github.com/huggingface/transformers/blob/v4.53.3/src/transformers/models/vjepa2/modeling_vjepa2.py
- `hf` CLI / huggingface_hub 1.x migration: https://huggingface.co/blog/hf-cli , https://huggingface.co/docs/huggingface_hub/en/concepts/migration
- numba releases (0.63 Dec-2025, 0.65 free-threading): https://github.com/numba/numba/releases
- toponymy / datamapplot PyPI: https://pypi.org/project/toponymy/ , https://pypi.org/project/datamapplot/
- Local: `/home/b3ali/projects/stamp26/repos/{toponymy,datamapplot,evoc,fast_hdbscan,vectorizers}/pyproject.toml`, `repos/toponymy/uv.lock`, `repos/datamapplot/datamapplot/offline_mode_caching.py`, `interactive_rendering.py`, `repos/toponymy/toponymy/llm_wrappers.py`, `embedding_wrappers.py`, `keyphrases.py`; `nvidia-smi`, `df -h/-i /home`, `/proc/mdstat`, `du -sh ~/.cache/{uv,huggingface}`.
