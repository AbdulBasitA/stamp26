"""Phase 1 verify gates (plan.md): counts, distributions, decode rate, frame coverage,
thumbnail coverage, and a human spot-check contact sheet of event frames."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/home/b3ali/projects/stamp26")
FRAMES, THUMBS = ROOT / "cache/frames", ROOT / "cache/thumbs"
EXPECTED_WEATHER = {"Clear": 919, "Cloudy": 495, "Rain": 83, "Snow": 1}
EXPECTED_SCENE = {"Urban": 784, "Highway": 379, "Sub-urban": 271, "Other": 34,
                  "Rural": 18, "Industrial": 13, "Nature": 1}

gates = []


def gate(name, ok, detail=""):
    gates.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")


df = pd.read_parquet(ROOT / "artifacts/manifest.parquet")
tr = df[df.split == "train"]

print("== Gate 1: counts & metadata ==")
gate("train count == 1500 (750/750)",
     len(tr) == 1500 and int(tr.label.sum()) == 750, f"got {len(tr)} ({int(tr.label.sum())} pos)")
gate("test count == 1344", len(df[df.split.str.startswith('test')]) == 1344)
w = tr.weather.value_counts().to_dict()
s = tr.scene.value_counts().to_dict()
gate("weather distribution matches published",
     all(w.get(k, 0) == v for k, v in EXPECTED_WEATHER.items()), str(w))
gate("scene distribution matches published",
     all(s.get(k, 0) == v for k, v in EXPECTED_SCENE.items()), str(s))
pos = tr[tr.label == 1]
gate("positives have time_of_event", pos.time_of_event.notna().all(),
     f"range {pos.time_of_event.min():.1f}-{pos.time_of_event.max():.1f}s")

print("== Gate 2: decode & frame coverage ==")
gate("decode success >= 99.5%", tr.decode_ok.mean() >= 0.995, f"{tr.decode_ok.mean():.3%}")
ok_clips = tr[tr.decode_ok.astype(bool)]
done = sum((FRAMES / c / ".done").exists() for c in ok_clips.clip_id)
gate("all decodable clips extracted", done == len(ok_clips), f"{done}/{len(ok_clips)}")

rng = np.random.default_rng(0)
sample = ok_clips.sample(40, random_state=0)
bad_cov = []
for _, r in sample.iterrows():
    n = len(list((FRAMES / r.clip_id).glob("*.jpg")))
    expect = r.duration_s * 8
    if not (expect - 16 <= n <= expect + 16):
        bad_cov.append((r.clip_id, n, round(expect)))
gate("frame count ~= duration*8fps (40-clip sample)", not bad_cov, str(bad_cov[:5]))

th = sum((THUMBS / f"{c}.jpg").exists() for c in ok_clips.clip_id)
gate("thumbnails present", th == len(ok_clips), f"{th}/{len(ok_clips)}")

print("== Gate 3: event-frame spot-check (human eyeball) ==")
import cv2

picks = pos[pos.decode_ok.astype(bool)].sample(12, random_state=1)
tiles = []
for _, r in picks.iterrows():
    fr_dir = FRAMES / r.clip_id
    frames = sorted(fr_dir.glob("*.jpg"))
    if not frames:
        continue
    target_ms = int(r.time_of_event * 1000)
    nearest = min(frames, key=lambda p: abs(int(p.stem) - target_ms))
    img = cv2.imread(str(nearest))
    img = cv2.resize(img, (426, 240))
    cv2.putText(img, f"{r.clip_id} @{r.time_of_event:.1f}s", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    tiles.append(img)
rows_img = [np.hstack(tiles[i:i + 4]) for i in range(0, 12, 4)]
sheet = np.vstack(rows_img)
out = ROOT / "artifacts/phase1_spotcheck.jpg"
cv2.imwrite(str(out), sheet)
print(f"  contact sheet of 12 random positive EVENT frames -> {out}")
print("  (human gate: open it and confirm the frames show incidents/imminent events)")

cache_gb = sum(f.stat().st_size for f in FRAMES.rglob("*.jpg")) / 1e9
print(f"== frame cache size: {cache_gb:.1f} GB ==")

n_fail = sum(1 for _, ok in gates if not ok)
print(f"\n{'PHASE1_VERIFY_PASS' if n_fail == 0 else f'PHASE1_VERIFY_FAIL ({n_fail} gates)'}")
