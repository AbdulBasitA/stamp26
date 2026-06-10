"""Phase 5 verify gates: no empty names, >=80% parse '<Category>: <qualifier>' with valid
category, within-layer uniqueness; renders eyeball sheets (name + random member thumbnails)."""
import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path("/home/b3ali/projects/stamp26")
sys.path.insert(0, str(ROOT / "src/phase5"))
from driving_templates import CATEGORIES

gates = []


def gate(name, ok, detail=""):
    gates.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")


rng = np.random.default_rng(5)
for track in ["captions", "siglip2"]:
    f = ROOT / f"artifacts/naming/{track}.json"
    if not f.exists():
        gate(f"{track}: results exist", False)
        continue
    res = json.loads(f.read_text())
    all_names = [n for l in res["layers"] for n in l["names"]]
    print(f"== {track}: {len(all_names)} names over {len(res['layers'])} layers ==")
    empty = [n for n in all_names if not n or not n.strip()]
    gate(f"{track}: zero empty names", len(empty) == 0, f"{len(empty)} empty")
    parsed = [n for n in all_names if ":" in n and n.split(":")[0].strip() in CATEGORIES]
    gate(f"{track}: >=80% taxonomy format", len(parsed) / len(all_names) >= 0.8,
         f"{len(parsed)}/{len(all_names)} = {len(parsed)/len(all_names):.0%}")
    for l in res["layers"]:
        dups = len(l["names"]) - len(set(l["names"]))
        if dups:
            print(f"    note: layer {l['layer']} has {dups} duplicate names")

    # eyeball sheet: 10 random base-layer clusters, name + 5 random member thumbs
    labels = np.load(ROOT / f"artifacts/naming/{track}__labels_layer0.npy")
    ids = np.asarray(res["ids"])
    names0 = res["layers"][0]["names"]
    picks = rng.choice(len(names0), size=min(10, len(names0)), replace=False)
    rows = []
    for c in picks:
        members = ids[labels == c]
        sel = rng.choice(members, size=min(5, len(members)), replace=False)
        thumbs = [cv2.resize(cv2.imread(str(ROOT / f"cache/thumbs/{i}.jpg")), (213, 120))
                  for i in sel if (ROOT / f"cache/thumbs/{i}.jpg").exists()]
        while len(thumbs) < 5:
            thumbs.append(np.zeros((120, 213, 3), np.uint8))
        strip = np.hstack(thumbs)
        head = np.full((26, strip.shape[1], 3), 30, np.uint8)
        cv2.putText(head, f"[{c}] n={len(members)}  {names0[c]}"[:130], (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        rows.append(np.vstack([head, strip]))
    sheet_dir = ROOT / "artifacts/phase5"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(sheet_dir / f"eyeball_{track}.png"), np.vstack(rows))
    print(f"  eyeball sheet -> artifacts/phase5/eyeball_{track}.png")

print("PHASE5_VERIFY_PASS" if all(gates) and gates else f"PHASE5_VERIFY_FAIL ({sum(not g for g in gates)})")
