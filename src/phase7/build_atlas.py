"""Phase 7: the interactive scenario atlas — DataMapPlot HTML per primary track.
Hover thumbnails (inline base64), NHTSA topic layers (finest first), search, topic tree,
color layers (collision/weather/lighting/scene/outlier score/judge confidence), offline mode.
Usage: build_atlas.py --track captions|siglip2"""
import argparse
import base64
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path("/home/b3ali/projects/stamp26")
CITATION = "STAMP 2026 · Data: Moura & Zvitia, Nexar Collision Dataset, Hugging Face 2025"

HOVER_TEMPLATE = """<div style="max-width:340px">
<img src="{thumb}" width="320" style="border-radius:4px"/>
<div style="font-size:11px;margin-top:4px">{hover_text}</div>
<div style="font-size:10px;color:#888;margin-top:3px">
{clip_id} &middot; {collision} &middot; {weather_h}/{lighting_h}/{scene_h} &middot; name-confidence {judge}</div>
</div>"""


def thumb_uri(clip_id):
    img = cv2.imread(str(ROOT / f"cache/thumbs/{clip_id}.jpg"))
    if img is None:
        return ""
    h = int(round(img.shape[0] * 200 / img.shape[1]))
    img = cv2.resize(img, (200, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=["captions", "siglip2"])
    args = ap.parse_args()
    track = args.track

    naming = json.loads((ROOT / f"artifacts/naming/{track}.json").read_text())
    ids = np.asarray(naming["ids"])
    n = len(ids)

    # 2-D display map: subset the track's map rows to the named ids
    info = json.loads((ROOT / f"artifacts/maps/{track}__ids.json").read_text())
    map_ids = np.asarray(info["ids"])
    row_of = {cid: i for i, cid in enumerate(map_ids)}
    rows = np.asarray([row_of[c] for c in ids])
    map2d = np.load(ROOT / f"artifacts/maps/{track}__2d.npy")[rows]

    # label layers, finest first
    layers = []
    for li, layer in enumerate(naming["layers"]):
        lab = np.load(ROOT / f"artifacts/naming/{track}__labels_layer{li}.npy")
        names = layer["names"]
        layers.append(np.array([names[l] if l >= 0 else "Unlabelled" for l in lab]))

    man = pd.read_parquet(ROOT / "artifacts/manifest.parquet").set_index("clip_id")
    meta = man.reindex(ids)
    recs = {}
    for line in (ROOT / "artifacts/captions/clips_a.jsonl").open():
        r = json.loads(line)
        if r.get("ok"):
            recs[r["clip_id"]] = r

    # outlier score: high-dim kNN distance on the displayed subset (the winning scorer)
    emb_file = "captions_clips_a" if track == "captions" else f"{track}_clips"
    emb = np.load(ROOT / f"artifacts/embeddings/{emb_file}.npy").astype(np.float32)
    eids = pd.read_parquet(ROOT / f"artifacts/embeddings/{emb_file}_ids.parquet")["id"].tolist()
    erow = {c: i for i, c in enumerate(eids)}
    E = emb[[erow[c] for c in ids]]
    from sklearn.neighbors import NearestNeighbors
    d, _ = NearestNeighbors(n_neighbors=11, metric="cosine").fit(E).kneighbors(E)
    outlier = d[:, 1:].mean(1)

    # judge confidence per point (its base cluster's held-out precision)
    per = pd.read_csv(ROOT / "artifacts/eval/judge_per_cluster.csv")
    per = per[per.track == track].set_index("cluster")["mean"]
    base = np.load(ROOT / f"artifacts/naming/{track}__labels_layer0.npy")
    judge = np.array([per.get(l, np.nan) if l >= 0 else np.nan for l in base])

    captions_text = np.array([recs[c]["caption"] if c in recs else "" for c in ids])
    extra = pd.DataFrame({
        "thumb": [thumb_uri(c) for c in ids],
        "clip_id": ids,
        "collision": np.where(meta.label.values == 1, "collision/near-miss", "normal"),
        "weather_h": meta.weather.fillna("?").values,
        "lighting_h": meta.light_conditions.fillna("?").values,
        "scene_h": meta.scene.fillna("?").values,
        "judge": [f"{j:.2f}" if np.isfinite(j) else "n/a" for j in judge],
    })

    import datamapplot
    plot = datamapplot.create_interactive_plot(
        map2d, *layers,
        hover_text=captions_text,
        extra_point_data=extra,
        hover_text_html_template=HOVER_TEMPLATE,
        enable_search=True,
        enable_topic_tree=True,
        colormaps={
            "collision label": extra.collision.values,
            "weather (human)": extra.weather_h.values,
            "lighting (human)": extra.lighting_h.values,
            "scene (human)": extra.scene_h.values,
            "outlier score (kNN)": outlier.astype(np.float32),
            "name confidence (judge)": np.nan_to_num(judge, nan=0.5).astype(np.float32),
        },
        title=f"A Named Atlas of the Long Tail of Driving — {track} track",
        sub_title=CITATION,
        noise_label="Unlabelled",
        darkmode=True,
        offline_mode=True,
        inline_data=True,
        cvd_safer=True,
    )
    od = ROOT / "artifacts/atlas"
    od.mkdir(parents=True, exist_ok=True)
    out = od / f"atlas_{track}.html"
    plot.save(str(out))

    # strip the unconditional maxcdn bootstrap/font-awesome links (dangle when air-gapped)
    html = out.read_text()
    cleaned, removed = [], 0
    for line in html.splitlines():
        if "maxcdn.bootstrapcdn.com" in line or "bootstrapcdn" in line:
            removed += 1
            continue
        cleaned.append(line)
    out.write_text("\n".join(cleaned))
    print(f"{out} written: {out.stat().st_size/1e6:.1f} MB ({n} points, {len(layers)} layers, "
          f"{removed} CDN links stripped)")
    print("ATLAS_DONE")


if __name__ == "__main__":
    main()
