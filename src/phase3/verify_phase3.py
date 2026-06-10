"""Phase 3 verify gates: shapes/NaN audit + sanity UMAPs colored by label/scene/lighting."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path("/home/b3ali/projects/stamp26")
ED = ROOT / "artifacts/embeddings"
gates = []


def gate(name, ok, detail=""):
    gates.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")


man = pd.read_parquet(ROOT / "artifacts/manifest.parquet")
man = man[(man.split == "train") & man.decode_ok.astype(bool)].set_index("clip_id")
win = pd.read_parquet(ROOT / "artifacts/windows.parquet")

print("== shapes & numeric health ==")
sets = sorted(ED.glob("*.npy"))
for f in sets:
    emb = np.load(f)
    ids = pd.read_parquet(f.with_name(f.stem + "_ids.parquet"))
    n_expect = {"windows": len(win), "clips": len(man)}.get(f.stem.split("_")[1], len(ids))
    finite = np.isfinite(emb.astype(np.float32)).all()
    norms = np.linalg.norm(emb.astype(np.float32), axis=1)
    gate(f.stem, len(ids) == len(emb) and finite and (norms > 1e-6).all(),
         f"{emb.shape} ids={len(ids)} expect~{n_expect} finite={finite} min_norm={norms.min():.3f}")

tok = ROOT / "artifacts/detections/tokens.parquet"
if tok.exists():
    t = pd.read_parquet(tok)
    vocab = t.token.nunique()
    gate("token track", t.unit_id.nunique() >= 2000 and vocab <= 400,
         f"{t.unit_id.nunique()} units, vocab {vocab}")

print("== sanity UMAPs (clip-level) ==")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap

sd = ROOT / "artifacts/phase3_sanity"
sd.mkdir(parents=True, exist_ok=True)
for f in sets:
    if "_clips" not in f.stem:
        continue
    emb = np.load(f).astype(np.float32)
    ids = pd.read_parquet(f.with_name(f.stem + "_ids.parquet"))["id"]
    meta = man.reindex(ids)
    m2 = umap.UMAP(n_components=2, metric="cosine", random_state=42).fit_transform(emb)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col in zip(axes, ["label", "scene", "light_conditions"]):
        vals = meta[col].astype(str).values
        for v in pd.unique(vals):
            m = vals == v
            ax.scatter(m2[m, 0], m2[m, 1], s=4, alpha=0.6, label=str(v)[:12])
        ax.set_title(f"{f.stem} by {col}")
        ax.legend(markerscale=2, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(sd / f"{f.stem}.png", dpi=110)
    plt.close(fig)
    print(f"  sanity map -> {sd / (f.stem + '.png')}")

print("\n" + ("PHASE3_VERIFY_PASS" if all(gates) else f"PHASE3_VERIFY_FAIL ({sum(not g for g in gates)})"))
