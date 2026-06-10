"""Phase 4: visual coherence artifacts per clip track —
(1) 2-D map colored by base-layer clusters (glasbey), (2) cluster cards: exemplar thumbnails
+ metadata-delta text per cluster. -> artifacts/phase4/"""
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path("/home/b3ali/projects/stamp26")
MAPS = ROOT / "artifacts/maps"
OUTD = ROOT / "artifacts/phase4"
import sys

sys.path.insert(0, str(ROOT / "src/phase4"))
from tracks import CLIP_TRACKS, manifest

import glasbey
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

THUMB_W, THUMB_H = 213, 120
N_EX = 6


def scatter_png(track, u2, labels, valid_ids):
    fig, ax = plt.subplots(figsize=(9, 7))
    pal = glasbey.create_palette(int(labels.max()) + 1, colorblind_safe=True)
    for c in range(int(labels.max()) + 1):
        m = labels == c
        ax.scatter(u2[m, 0], u2[m, 1], s=7, color=pal[c], label=f"{c} (n={m.sum()})")
    noise = labels < 0
    ax.scatter(u2[noise, 0], u2[noise, 1], s=4, color="#dddddd")
    ax.set_title(f"{track}: base-layer clusters (grey = noise {noise.mean():.0%})")
    ax.set_xticks([]); ax.set_yticks([])
    if labels.max() < 24:
        ax.legend(fontsize=6, markerscale=2, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTD / f"{track}_clusters.png", dpi=120)
    plt.close(fig)


def deltas_text(meta_c, meta_all):
    bits = []
    for col in ["scene", "light_conditions", "weather"]:
        sc = meta_c[col].value_counts(normalize=True)
        sa = meta_all[col].value_counts(normalize=True)
        d = (sc - sa.reindex(sc.index).fillna(0)).sort_values(ascending=False)
        if len(d) and d.iloc[0] > 0.08:
            bits.append(f"{d.index[0]} +{d.iloc[0]:.0%}")
    return ", ".join(bits) if bits else "no strong metadata skew"


def cards_png(track, u5, labels, ids, meta_all):
    order = [(c, (labels == c).sum()) for c in range(int(labels.max()) + 1)]
    order.sort(key=lambda t: -t[1])
    rows_img = []
    for c, size in order[:14]:
        m = labels == c
        centroid = u5[m].mean(0)
        d = np.linalg.norm(u5[m] - centroid, axis=1)
        ex_ids = np.asarray(ids)[m][np.argsort(d)[:N_EX]]
        thumbs = []
        for cid in ex_ids:
            img = cv2.imread(str(ROOT / f"cache/thumbs/{cid}.jpg"))
            img = cv2.resize(img, (THUMB_W, THUMB_H)) if img is not None else np.zeros((THUMB_H, THUMB_W, 3), np.uint8)
            thumbs.append(img)
        while len(thumbs) < N_EX:
            thumbs.append(np.zeros((THUMB_H, THUMB_W, 3), np.uint8))
        strip = np.hstack(thumbs)
        meta_c = meta_all.loc[[i for i in ex_ids if i in meta_all.index]]
        meta_full_c = meta_all.loc[[i for i in np.asarray(ids)[m] if i in meta_all.index]]
        header = np.full((26, strip.shape[1], 3), 30, np.uint8)
        txt = (f"cluster {c}  n={size}  collision={meta_full_c.label.mean():.0%}  | "
               f"{deltas_text(meta_full_c, meta_all)}")
        cv2.putText(header, txt[:150], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        rows_img.append(np.vstack([header, strip]))
    sheet = np.vstack(rows_img)
    cv2.imwrite(str(OUTD / f"{track}_cards.png"), sheet)


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    meta_all = manifest()
    for track in CLIP_TRACKS:
        info = json.loads((MAPS / f"{track}__ids.json").read_text())
        valid = np.asarray(info["valid"])
        ids = np.asarray(info["ids"])[valid]
        u5 = np.load(MAPS / f"{track}__5d.npy")[valid]
        u2 = np.load(MAPS / f"{track}__2d.npy")[valid]
        labels = np.load(MAPS / f"{track}__labels_layer0.npy")
        scatter_png(track, u2, labels, ids)
        cards_png(track, u5, labels, ids, meta_all)
        print(f"{track}: cards + scatter written")
    print("CARDS_DONE")


if __name__ == "__main__":
    main()
