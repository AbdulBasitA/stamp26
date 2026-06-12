# A Named Atlas of the Long Tail of Driving

**STAMP 2026 workshop project** — testing whether the [Toponymy](https://github.com/TutteInstitute/toponymy)
pipeline (embed → cluster → characterize → LLM-name → map) generalizes from text and cyber telemetry
to ego-centric driving video, using the Nexar collision dataset with human collision labels as
ground-truth rare events.

> Data citation (required by license): *Moura & Zvitia, Nexar Collision Dataset, Hugging Face 2025.*

## Headline results (Phases 0–7 complete)

1. **Sessionization decides whether rare events are visible.** Collision clips are essentially
   invisible as outliers at whole-clip granularity (best lift over chance: **1.7×**), but strongly
   detectable at 2-second event-window granularity (**17.6× lift, AUROC 0.93, ~30× enrichment in
   the top-25**) — measured at 3–9% simulated prevalence, 30 resamples, full UMAP+scorer refit per
   subset. This is the central claim of the STAMP telemetry-embedding problem set, demonstrated on
   a new modality.
2. **Score outliers in native space, not projected space.** Density scorers (GLOSH/LOF) in 5-D UMAP
   space sit at chance for window-level events; plain kNN distance in the original embedding space
   delivers the 17×. UMAP's density equalization eats the signal.
3. **LLM topic names are category-trustworthy, qualifier-untrustworthy.** LLM-judge precision on
   held-out cluster members ≈ 0.5 (names over-specify conditions); 38% of weather/lighting claims
   in names are unsupported by human metadata; names are 87–94% stable at category level across
   independent reruns but only 15–24% stable as exact strings.
4. **A one-word schema gap silently corrupted 26% of the corpus.** The caption schema's weather
   enum lacked "sunny"; under strict guided decoding the model's first token "s" could only
   complete as **"snow"**, the model then *rationalized* the forced value in its free-text caption,
   and the namer published "snowy" topics — all invisible to caption-based validation
   (a 12/12 flip experiment confirmed the mechanism; fixed by caption-first field ordering +
   de-collided enums: 386 → 1 snow captions). Moral for LLM-characterization pipelines: validate
   the generated view against the raw modality, and design schemas so models narrate before they
   commit to constrained tokens.

## The deliverable

Two self-contained interactive atlases (no internet required):

```bash
python3 -m http.server 8814 --directory artifacts/atlas
# open http://localhost:8814/atlas_captions.html  (or atlas_siglip2.html)
```

1,497 clips on a 2-D semantic map with: NHTSA-pre-crash-typology topic names at 3 zoom levels,
hover thumbnails + VLM captions, **click any point to play its video** (auto-seeks to ~5 s before
the annotated event), full-text search, topic tree, and switchable color layers including the kNN
outlier score and the per-cluster judge confidence.

## Pipeline

```
Nexar mp4s ─ decode(8fps JPEG cache) ─┬─ D-FINE detections → tokens → IWT  (bag-of-objects track)
                                      ├─ SigLIP2 / V-JEPA2 embeddings      (visual tracks)
                                      ├─ Qwen3.5-9B VLM structured captions → Qwen3-Embedding-4B
                                      └─ thumbnails
        → UMAP (5-D cluster space + 2-D display) → ToponymyClusterer (multi-layer)
        → Toponymy + custom NHTSA Jinja templates + per-cluster metadata-histogram injection
          (namer: Qwen3.6-35B-A3B-FP8, vLLM TP=4, thinking disabled)
        → DataMapPlot offline atlas │ evals: low-prevalence outlier AP / LLM-judge / metadata
          consistency / rerun stability
```

Everything runs locally on one 4×RTX 4090 box; two uv venvs (`venvs/vllm` for serving,
`venvs/main` for analysis — irreconcilable torch/transformers pins, they talk over HTTP).

## Repo layout

| Path | What |
|---|---|
| `plan.md` | **Source of truth**: decisions log D1–D22, 10 phases with verify gates, status |
| `research/*.md` | 9 deep-research notes (Tutte stack, model selection, dataset, env) |
| `src/phase1..7/` | Phase pipelines, each with a `run_phaseN.sh` runner + verify script |
| `src/smoke/`, `src/rerun_caption_fix.sh`, `src/run_overnight.sh` | Smoke tests + chains |
| `steps.md` | Plain-language project explainer |
| `freeze-{main,vllm}.txt` | Exact environment snapshots |
| `artifacts/`, `cache/`, `data/`, `venvs/`, `repos/` | Generated/large — gitignored, regenerable |

Key generated artifacts (on the box, not in git): `artifacts/atlas/*.html` (the atlases),
`artifacts/eval/` (outlier results, judge verdicts, stability, `outlier_lift.png`),
`artifacts/naming/` (topic names per track/layer/run), `artifacts/captions/` (v2 captions;
`enumbug/` archives the corrupted v1 for the before/after artifact).

## Reproducing

Each phase is resume-safe and gated; run in order (long jobs detach and drop `.done` markers):

```bash
bash src/phase1/run_phase1.sh    # manifest + frame cache (~20 min)
bash src/phase2/run_phase2.sh    # captioning bake-off (~2 h, 4 GPUs)
bash src/phase3/run_phase3.sh    # embeddings + detections (~1 h, 4 GPUs)
bash src/phase4/run_phase4.sh    # maps + clustering (CPU, ~15 min)
bash src/phase5/run_phase5.sh    # NHTSA naming (namer TP=4, ~30 min)
bash src/phase6/run_phase6.sh    # evaluations (~2 h, CPU+GPU concurrent)
venvs/main/bin/python src/phase7/build_atlas.py --track captions   # the atlas (~2 min)
```

Remaining: Phase 8 stretch (BADAS-Open failure overlay on the held-out test split, EVoC
comparison, DINOv3 appearance-vs-semantics ablation, upstream Toponymy `base_url` fix PR) and
Phase 9 (air-gap packaging rehearsal + slide deck). See `plan.md` §Phases.
