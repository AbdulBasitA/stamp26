# Nexar dataset + BADAS ecosystem — research notes (as of 2026-06-09)

Scope: facts verified against HF Hub APIs, GitHub, arXiv, and vendor pages on 2026-06-09 for the STAMP 2026 Toponymy-on-driving-video project. All API/file inspections below were performed live (curl against HF/GitHub), not from memory.

---

## 1. The dataset: `nexar-ai/nexar_collision_prediction`

- **HF ID (verified)**: `nexar-ai/nexar_collision_prediction` — https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction
- Last modified 2025-06-10; ~5.3k downloads; tags: video-classification, automotive, dashcam, collision, prediction; `license:other`.
- This is the **only** dataset in the `nexar-ai` HF org (verified via `GET /api/datasets?author=nexar-ai`). No newer Nexar dataset variants exist on HF as of 2026-06.

### 1.1 Repo layout (2,857 files, **31.38 GB total**; verified by summing the HF tree API)

```
LICENSE                          # custom "Nexar open" license, see 1.4
README.md
evaluate_submission.py           # official local scorer (mAP over TTA groups)
sample_submission.csv            # columns: id,score      (id = zero-padded 5-digit clip id)
solution.csv                     # TEST GROUND TRUTH: id,target,group,Usage(Public/Private)
time_to_accident_test_map.csv    # columns 0.5,1.0,1.5 -> which test ids belong to each TTA cut
train/positive/  750 mp4 + metadata.csv     \
train/negative/  750 mp4 + metadata.csv      |- train = 25.53 GB (1,500 clips, ~40 s, 1280x720@30)
test-public/{positive,negative}/  667 mp4 + 2 metadata.csv   2.88 GB
test-private/{positive,negative}/ 677 mp4 + 2 metadata.csv   2.97 GB
```

- Train: 1,500 full clips (750 positive = collision **or near-collision**, 750 negative).
- Test: 1,344 clips total (public + private), **clipped to ~10 s** and truncated at 500/1000/1500 ms *before* the event (groups in `time_to_accident_test_map.csv`). `solution.csv` contains the test ground truth, so **full local evaluation is possible offline** with `evaluate_submission.py` (sklearn `average_precision_score` per TTA group, then mean → "mAP").

### 1.2 Metadata columns (verified from `metadata.csv` headers and datasets-server `first-rows`)

`file_name (e.g. 00822.mp4), time_of_event, time_of_alert, light_conditions, weather, scene, time_to_accident`

- Loaded via `datasets`, the columns are: `video` (Video feature), `time_of_event` (f64), `time_of_alert` (f64), `light_conditions` (str), `weather` (str), `scene` (str), `time_to_accident` (f64, **test only**, null in train).
- There is no separate `id` column — the id is the 5-digit file stem (matches `sample_submission.csv` / `solution.csv` `id`).
- Verified value distributions (train, n=750+750):
  - `light_conditions`: Normal 1355, Twilight 88, Dark 47, Bright 10
  - `weather`: Clear 919, Cloudy 495, Rain 83, Snow 1, blank 2  ← two missing values
  - `scene`: Urban 784, Highway 379, Sub-urban 271, Other 34, Rural 18, Industrial 13, Nature 1
- Positive-clip timing (train, n=750): `time_of_event` min 3.03 s / median 19.80 s / **max 56.80 s** (some clips are longer than 40 s; events are NOT always centered — clip-length and event-position assumptions must be checked per clip). Alert→event gap: min 0.03 s / median 1.43 s / max 4.47 s; no missing `time_of_alert` in positives.

### 1.3 Download

```bash
# huggingface_hub >= 0.30 ships the `hf` CLI (older name: huggingface-cli)
uv pip install -U "huggingface_hub[cli]"
hf download nexar-ai/nexar_collision_prediction --repo-type dataset \
   --local-dir /data/nexar_collision_prediction
# train only (25.5 GB):
hf download nexar-ai/nexar_collision_prediction --repo-type dataset \
   --include "train/*" --local-dir /data/nexar_collision_prediction
```
Videos are plain `.mp4` files in folders (imagefolder/videofolder convention), so plain `hf download` (or even `git lfs`) works; no parquet decoding needed. No gating/auth required (public repo).

### 1.4 License (full text fetched from `LICENSE`, © 2025 Nexar Inc.)

Custom permissive license ("Nexar open data" style, MIT-like + ethical-use restrictions). Granted: **use, copy, modify, distribute**, free of charge. Conditions:
- **Attribution required**; published work must cite: *Moura, Daniel C., and Zvitia, Orly. "Nexar Collison Dataset." Hugging Face, 2025* (typo "Collison" is in the original license text).
- Redistributions must retain copyright notice + conditions; **no resale/for-profit redistribution** without written consent.
- Ethical restrictions: no malicious/unsafe-driving systems, no deepfakes/misinformation, **no re-identification of individuals/vehicles** (incl. surveillance/tracking), **no weaponization ("development, testing, or deployment of weapon systems or technology intended for combat or armed conflict")**, no exploitative practices (e.g. predatory insurance), comply with laws.
- AS-IS, no warranty.

**Workshop implications**: research demo + screenshots at a government-hosted workshop are clearly permitted (use + publication with citation). Two flags: (a) include the citation line on slides/maps that show frames; (b) the *weaponization* clause — the Tutte Institute's parent org is a security agency; a road-safety research demo is fine, but don't position the work as feeding any combat/weapon system, and avoid anything that looks like re-identification (e.g. don't zoom license plates/faces in hover thumbnails; Nexar blurs plates/faces upstream, but treat it as a constraint).

### 1.5 Known data-quality issues

- **Label ambiguity in positives** (1st-place Kaggle write-up, 3LC, https://3lc.ai/nexar-dashcam-kaggle-challenge/): models confidently predicted "collision" on a set of *negative-labeled* clips → subjectively ambiguous / arguably mislabeled near-misses exist; the winner *deleted* ambiguous samples rather than relabel. He also found post-event frames "chaotic"/harmful, and estimated only "~2–5% of the data contained the most impactful training signals".
- Positive class includes **near-collisions**, not only collisions — for our low-prevalence outlier evaluation, near-misses sit semantically between normal and crash and will blur the outlier boundary; consider stratifying positives by `time_of_alert`-to-`time_of_event` gap or by whether contact occurs.
- Minor: 2 blank `weather` values; clip durations vary (events at 3–57 s); 30 fps nominal but BADAS code defensively checks broken fps metadata (`Invalid FPS detected` guard), suggesting occasional odd containers — decode with a fallback path.
- No reports found of corrupt/unreadable files at scale (no open HF discussions on this; Kaggle forum is auth-gated, but the 3LC write-up is the de-facto data-audit reference).
- BADAS paper (arXiv 2510.14876) context: Nexar's own annotation used 10 defensive-driving-certified annotators; consensus median human alert time = **1.70 s before impact** (90% range 0.70–3.47 s) — matches the train alert-gap stats above (median 1.43 s).

---

## 2. Papers + 2025 Kaggle challenge (prior art for embeddings)

### 2.1 Dataset/challenge paper
- **arXiv 2503.03848** — *Nexar Dashcam Collision Prediction Dataset and Challenge*, Moura, Zhu, Zvitia (Nexar), v1 2025-03-05, no later revisions (verified via arXiv API). https://arxiv.org/abs/2503.03848
- Metric: mean of **average precision computed separately per time-to-event group** (clips truncated 500/1000/1500 ms before event), i.e. the `evaluate_submission.py` logic above. Emphasis: predict *early*.
- Kaggle competition: https://www.kaggle.com/competitions/nexar-collision-prediction (spring 2025).

### 2.2 Winning approaches (useful prior art)
- **1st place — Paul Endresen (3LC)**: *data-centric*, architecture deliberately boring: torchvision **`mvit_v2_s`** video transformer, 256×256 frames, **16 frames @ stride 4 ≈ 2 s of context**, trained only on frames between `time_of_alert` and `time_of_event`; removed post-event frames, ambiguous samples, and last 0.25 s before impact; used embedding-space cluster inspection to find systematic errors. Score 0.71 → **0.898** private LB. (https://3lc.ai/nexar-dashcam-kaggle-challenge/)
- **2nd place — Wang Zhiyao & Yu Yuchen (HUST/UESTC)**: **VideoMAEv2-based** classifier (Kaggle discussion 576289; details auth-gated).
- 3rd — Max Zwager (independent); honorable mention — Wuhan Univ. team (Zhao, Zou, Mao, Lin, Wu, Du). Winners showcased at CVPR 2025 workshops. (https://www.nexar-ai.com/blog/nexar-crash-prediction-challenge-winners-revealed-pioneering-ai-for-safer-roads)
- **Takeaways for us**: (a) ~2 s context windows at reduced fps are the sweet spot — matches our event-centered sub-window design; (b) generic video transformers (MViT/VideoMAE/V-JEPA2) transfer fine to this data; (c) the dataset rewards data curation over architecture → our cluster-level inspection (the Toponymy map itself) is a natural fit and demo-able story ("the 1st-place solution was won with exactly this kind of map-driven data audit").

### 2.3 BADAS papers
- **BADAS 1.0**: arXiv **2510.14876** — *BADAS: Context Aware Collision Prediction Using Real-World Dashcam Data* (Oct 2025). V-JEPA2 **ViT-L** backbone, 16 frames @ 256×256 (2×16×16 tubelets → 2048 patches, dim 1024), attentive probe (12 learned queries), 3-layer MLP head, BCE. Two variants: **BADAS-Open** (trained on the 1,500 public Nexar clips; weights+code released) and BADAS 1.0 (40k proprietary clips). Key contribution: **ego-centric re-annotation** of DAD (92% of accidents are non-ego!), DoTA (43.8% non-ego), DADA-2000 (40.5% non-ego); re-annotations CSVs ship in the BADAS-Open repo (`annotation/*.csv`). Metrics: AP, AUC, mTTA. BADAS-Open: Nexar AP 0.86 / AUC 0.88 / mTTA 4.9 s.
- **BADAS 2.0**: arXiv **2604.05767** — *Beyond the Beep: Scalable Collision Anticipation and Real-Time Explainability with BADAS-2.0* (v1 2026-04-07, v2 2026-04-12; Goldshmidt, Scott, Niccolini, Matzner). Same V-JEPA2 ViT-L (300M, dim 1024, 24 layers) + **future-prediction branch** (predicts scene representation 1 s ahead, concatenated before head); trained on 178.5k labeled videos (~2M clips) + SSL pre-training on 2.25M unlabeled dashcam videos; distilled family: **2.0 (ViT-L, 300M, 34 ms)**, **Flash (ViT-B, 86M)**, **Flash-Lite (ViT-S, 22M)**. Nexar Kaggle test mAP **0.940** (vs 1st-place human 0.898, BADAS-1.0 0.925); FPR 4.6%. Explainability: training-free attention heatmaps + **BADAS-Reason** = fine-tuned **Qwen3-VL-4B** emitting structured JSON (hazard description + driver action). Long-tail benchmark: 888 clips, 10 scenario groups (animal, pedestrian, cyclist, fog, rain, snow, intersection, infrastructure, passing/overtaking, motorcyclist).

---

## 3. BADAS 2.0 / BADAS-Open: what is actually public

| Asset | Public? | Where |
|---|---|---|
| BADAS-Open weights (1.0 arch, trained on the 1.5k public set) | **Yes** (Apache 2.0) | HF `nexar-ai/BADAS-Open`, file `weights/badas_open.pth` (**3.98 GB**, fp32) |
| BADAS-Open code | **Yes** (Apache 2.0) | GitHub `getnexar/BADAS-Open` (updated 2026-06-06); PyPI **`badas==1.1.3`** |
| Ego-centric re-annotations of DAD/DoTA/DADA-2000 | Yes | `annotation/*.csv` in the GitHub repo |
| BADAS 1.0 (40k-clip) weights | No | commercial |
| BADAS 2.0 / Flash / Flash-Lite weights | **No** | enterprise only ("on-device weights for Nano/Orin/Thor" via scoping call), https://badas.nexar.app/ has a browser drag-and-drop demo (per-frame collision probability) + API for enterprise partners |
| BADAS-2.0 long-tail benchmark (888 clips) | Not found on HF/GitHub as of 2026-06-09 (paper says "evaluation benchmarks publicly available", but only the DAD/DoTA/DADA re-annotations are actually downloadable) |
| BADAS-Reason (Qwen3-VL-4B ft) | No | not released |

- Base model (verified from HF tags + code): **`facebook/vjepa2-vitl-fpc16-256-ssv2`** (V-JEPA2 **ViT-L**, 16-frame clips, 256 px, SSv2 fine-tune head config). The "300M params" claim ≈ ViT-L backbone + probe/head; checkpoint is 3.98 GB fp32 ⇒ ~1B fp32 values incl. optimizer-free full backbone copy — fits a 4090 trivially (≈2 GB fp16, ≈4 GB fp32 weights).

### 3.1 Verified inference spec (from `badas/badas_loader.py`, `badas/utils/*.py` in the repo)

- `load_badas_model()` → `VJEPAModel(model_name="facebook/vjepa2-vitl-fpc16-256-ssv2", frame_count=16, img_size=224→overridden to 256 by model-name parsing, window_stride=1, target_fps=8.0, use_sliding_window=True)`.
- Input pipeline: decode video → resample to **8 fps** → sliding windows of **16 frames (= 2.0 s)**, stride 1 resampled frame (= 0.125 s) → resize 256×256.
- Output: `model.predict(video_path) -> List[float]` — **per-timestep collision probability at 8 Hz** (first 16 frames NaN-padded: "model sees frames [0..N-1] and predicts what happens after"); plus `model.estimate_time_to_accident(probs, fps=8.0)`. This is exactly the "risk score timeline" we need for the failure-overlay.
- Weights auto-download from `nexar-ai/badas-open` on first `BADASModel()` call — **pre-download for the air-gapped workshop** (`hf download nexar-ai/BADAS-Open`) and pass `checkpoint_path=`.

### 3.2 Running BADAS-Open locally on a 4090 (steps + cost estimate)

```bash
uv venv badas-env && source badas-env/bin/activate
uv pip install badas              # v1.1.3; or: git clone https://github.com/getnexar/BADAS-Open && uv pip install -e .
uv pip install "transformers>=4.53"   # IMPORTANT: vjepa2 model type landed in transformers 4.53 (Jun 2025);
                                       # repo's requirements.txt says >=4.40 which is too old to load the backbone.
hf download nexar-ai/BADAS-Open --local-dir /models/badas-open   # 4 GB, do this before air-gap
python - <<'PY'
from badas import BADASModel
m = BADASModel(device="cuda", checkpoint_path="/models/badas-open/weights/badas_open.pth")
probs = m.predict("/data/nexar_collision_prediction/train/positive/00822.mp4")  # 8 Hz timeline
PY
```

- Throughput: repo reports 45–52 windows/s (GPU unspecified, A100-class). A 40 s clip at stride 1 = ~320 windows → **~6–8 s/clip on one 4090** → all 1,500 train clips ≈ **2.5–3.5 GPU-hours on a single 4090** (embarrassingly parallel across our 4 GPUs → under 1 h wall). With `window_stride=4` (0.5 s resolution, plenty for the overlay) it drops ~4×. 100 clips: minutes. **Verdict: the "where does the predictor fail on the semantic map" overlay is fully realistic.**
- Per-clip overlay scalars to join onto the DataMapPlot: max prob in clip, prob at `time_of_alert`−ε, lead time = first-crossing(0.8)→`time_of_event`, AP contribution; for negatives: max prob (false-alarm propensity).

### 3.3 Major caveat for the overlay (flagged)

**BADAS-Open was trained on the very same 1,500 train clips we will map.** Overlay scores on the train split show training-set residuals, not generalization failures. Options: (a) run the overlay on the **1,344 test clips** (labels available in `solution.csv`; clips are 10 s — still embeddable/clusterable, and they match our event-window granularity anyway); (b) present the train-split overlay explicitly as "what the model still gets wrong on its own training data" (the 3LC story shows this is still informative); (c) both.

---

## 4. Anything newer / adjacent

- No BADAS-2.x open weights, no new `nexar-ai`/`getnexar` HF assets beyond the two repos (org listings verified 2026-06-09). GitHub org has only marketing repos for BADAS 2.0 (`badas-2-launch-brief`).
- `sewilliams/nexar_collision_prediction_with_captions` (HF dataset, Aug 2025) is **empty** (only `.gitattributes`) — dead end, but evidence someone attempted exactly our VLM-caption idea.
- arXiv 2503.03848 has no v2 with challenge results; the BADAS papers are the de-facto post-mortem.
- Adjacent ego-centric benchmarks if we want a transfer demo later: DoTA, DADA-2000, DAD **with the BADAS ego-involvement re-annotations** from the repo (these fix the 40–92% non-ego contamination).
- The 888-clip long-tail benchmark would be ideal for rare-scenario clusters but is not downloadable; don't plan on it.

---

## 5. Scenario typologies for the naming prompts

### 5.1 NHTSA pre-crash scenario typology — canonical sources

1. **Original "37 crashes" report**: Najm, Smith & Yanagisawa, *Pre-Crash Scenario Typology for Crash Avoidance Research*, **DOT HS 810 767**, April 2007. PDF: https://www.nhtsa.gov/sites/nhtsa.gov/files/pre-crash_scenario_typology-final_pdf_version_5-2-07.pdf (NHTSA/ROSA-P 403-block non-browser fetchers — download in a browser; companion ESV paper: https://www-nrd.nhtsa.dot.gov/pdf/ESV/Proceedings/20/07-0412-O.pdf).
2. **2019 revision (recommended)**: Swanson (Elizabeth) et al., *Statistics of Light-Vehicle Pre-Crash Scenarios Based on 2011–2015 National Crash Data*, **DOT HS 812 745**, 2019 — defines **36 scenarios in 9 groups**: *control loss, road departure, animal, pedestrian, pedalcyclist, lane change, opposite direction, rear-end, crossing paths* — and notably reworks the intersection scenarios into RTIP/RTAP/SCP/LTAP-LD/LTIP/LTAP-OD. ROSA-P: https://rosap.ntl.bts.gov/view/dot/41932. Cross-checked via two papers that reproduce entries+numbers: arXiv 2502.20789 (Table 2) and arXiv 2006.03987.
3. 2013 V2V variant: DOT HS 811 731 (https://www.nhtsa.gov/sites/nhtsa.gov/files/811731.pdf).

**2007 list (37 scenarios; transcribed from the well-known DOT HS 810 767 table — verify verbatim against the PDF before publication, NHTSA blocked automated download):**

1. Vehicle Failure · 2. Control Loss With Prior Vehicle Action · 3. Control Loss Without Prior Vehicle Action · 4. Running Red Light · 5. Running Stop Sign · 6. Road Edge Departure With Prior Vehicle Maneuver · 7. Road Edge Departure Without Prior Vehicle Maneuver · 8. Road Edge Departure While Backing Up · 9. Animal Crash With Prior Vehicle Maneuver · 10. Animal Crash Without Prior Vehicle Maneuver · 11. Pedestrian Crash With Prior Vehicle Maneuver · 12. Pedestrian Crash Without Prior Vehicle Maneuver · 13. Pedalcyclist Crash With Prior Vehicle Maneuver · 14. Pedalcyclist Crash Without Prior Vehicle Maneuver · 15. Backing Up Into Another Vehicle · 16. Vehicle(s) Turning — Same Direction · 17. Vehicle(s) Parking — Same Direction · 18. Vehicle(s) Changing Lanes — Same Direction · 19. Vehicle(s) Drifting — Same Direction · 20. Vehicle(s) Making a Maneuver — Opposite Direction · 21. Vehicle(s) Not Making a Maneuver — Opposite Direction · 22. Following Vehicle Making a Maneuver · 23. Lead Vehicle Accelerating · 24. Lead Vehicle Moving at Lower Constant Speed · 25. Lead Vehicle Decelerating · 26. Lead Vehicle Stopped · 27. Left Turn Across Path From Opposite Directions at Signalized Junctions · 28. Vehicle Turning Right at Signalized Junctions · 29. Left Turn Across Path From Opposite Directions at Non-Signalized Junctions · 30. Straight Crossing Paths at Non-Signalized Junctions · 31. Vehicle(s) Turning at Non-Signalized Junctions · 32. Evasive Action With Prior Vehicle Maneuver · 33. Evasive Action Without Prior Vehicle Maneuver · 34. Non-Collision Incident · 35. Object Crash With Prior Vehicle Maneuver · 36. Object Crash Without Prior Vehicle Maneuver · 37. Other

**2019 list (36 scenarios, 9 groups) — entries marked ✓ have name+number verified from arXiv 2502.20789 Table 2 / arXiv 2006.03987; unmarked entries are group-consistent reconstructions, verify numbering against DOT HS 812 745:**

- *Control loss*: 1 Vehicle Failure ✓ · 2 Control Loss/Vehicle Maneuver · 3 Control Loss/No Maneuver
- *Road departure*: 4 Road Departure/Maneuver · 5 Road Departure/No Maneuver · 6 Road Departure/Backing
- *Animal*: 7 Animal/Maneuver · 8 Animal/No Maneuver ✓(name)
- *Pedestrian*: 10 Pedestrian/Maneuver ✓ · 11 Pedestrian/No Maneuver ✓ (⇒ 9 is the last animal/road slot; renumber from the report)
- *Pedalcyclist*: 12 Pedalcyclist/No Maneuver ✓ · Pedalcyclist/Maneuver ✓(name)
- *Lane change / same direction*: 13 Backing Into Vehicle ✓ · 14 Turning/Same Direction ✓ · 15 Parking/Same Direction ✓ · 16 Changing Lanes/Same Direction ✓ · 17 Drifting/Same Direction ✓
- *Opposite direction*: 18 Opposite Direction/Maneuver · 19 Opposite Direction/No Maneuver ✓
- *Rear-end*: 20 Rear-end/Following Vehicle Making a Maneuver (FVM) ✓ · 21 Rear-end/Lead Vehicle Accelerating (LVA) ✓ · 22 Rear-end/Lead Vehicle Moving at Lower Constant Speed (LVM) ✓ · 23 Rear-end/Lead Vehicle Decelerating (LVD) ✓ · 24 Rear-end/Lead Vehicle Stopped (LVS) ✓
- *Crossing paths (intersections)*: 25 Right Turn Into Path (RTIP) ✓(name) · 26 Right Turn Across Path (RTAP) ✓(name) · 27 Straight Crossing Paths (SCP) ✓ · 28 Left Turn Across Path, Lateral Direction (LTAP/LD) ✓ · 29 Left Turn Into Path (LTIP) ✓ · 30 Left Turn Across Path, Opposite Direction (LTAP/OD) ✓
- *Other/non-collision*: 33 Non-collision/No Impact ✓ · 34 Object/Maneuver ✓ · 35 Object/No Maneuver ✓ · 36 Other ✓ (31–32 ≈ Evasive Action/Maneuver, Evasive Action/No Maneuver)

**Jinja recommendation**: embed the **2019 names grouped by the 9 groups** (group → scenario bullet list) as the controlled vocabulary, ask the namer to (a) pick the closest group, (b) optionally a scenario, (c) then write a free-text name; the 9 groups are robust even where our numbering is provisional, and an ego-dashcam dataset will concentrate in rear-end, crossing-paths, lane-change/same-direction, pedestrian/pedalcyclist, opposite-direction — the BADAS-2.0 long-tail group list (animal, pedestrian, cyclist, fog, rain, snow, intersection, infrastructure, passing/overtaking, motorcyclist) is a good secondary "conditions" vocabulary that matches the dataset's weather/lighting metadata.

### 5.2 ISO 34502 (alternative)

- **ISO 34502:2022** *Road vehicles — Test scenarios for ADS — Scenario based safety evaluation framework* (https://www.iso.org/standard/78951.html; **paid standard**, preview PDFs: iTeh https://cdn.standards.iteh.ai/samples/78951/aaa4e667f3d942f7a502044361312a2f/ISO-34502-2022.pdf).
- Organizes critical scenarios by three risk-factor categories aligned to perceive/plan/control: **perception disturbances** (sensor-affecting factors: glare, reflections, occluding roadside objects), **traffic disturbances** (road geometry + other-participant behavior: cut-in, cut-out, dangerous braking), **vehicle control disturbances** (wind, load, surface).
- The traffic-disturbance annex defines a **24-scenario matrix for highways**: road geometry {main road, merge zone, departure zone} × subject-vehicle behavior {lane keep, lane change} × other-vehicle behavior {cut-in, cut-out, deceleration, acceleration} (scenarios 1–8 main road, 9–16 merge, 17–24 departure). Machine-readable formalization: arXiv 2403.18764 (*Temporal Logic Formalisation of ISO 34502 Critical Scenarios*).
- Verdict: NHTSA typology is the better naming vocabulary for this dataset (crash-outcome oriented, covers urban + VRU); ISO 34502 is highway-maneuver oriented and license-encumbered — use it only as a secondary tag set for the highway clusters.

---

## 6. Direct implications for the project plan

1. **Embedding synergy**: BADAS-Open's backbone *is* `facebook/vjepa2-vitl-fpc16-256-ssv2` — the same V-JEPA2 we planned for visual embeddings. One backbone pass per 2 s window can serve both the embedding pipeline (mean-pool/probe features) and (with the BADAS probe+head) the risk timeline. Pre-download both HF repos before the air gap.
2. **Event-window design validated**: 16 frames @ ~8 fps ≈ 2 s is what the 1st-place Kaggle solution, BADAS, and V-JEPA2's `fpc16` all converge on. Use 2 s windows around `time_of_alert`/`time_of_event` for the event-centered condition.
3. **Test split is a free, untainted overlay target** (10 s clips, labels public in `solution.csv`); train-split overlay is tainted by BADAS-Open's training (see 3.3).
4. **Pin `transformers>=4.53`** (vjepa2 support); `badas==1.1.3`; checkpoint 3.98 GB; dataset 31.4 GB (25.5 GB train-only).
5. The metadata histograms for Toponymy cluster characterization should use: `light_conditions` (4 values), `weather` (4 values + missing), `scene` (7 values), label, and derived alert-lead-time bins.
