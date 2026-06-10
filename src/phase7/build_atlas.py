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
<div style="font-size:10px;color:#6af;margin-top:2px">click point to play clip</div>
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

    # label layers, finest first; display labels truncated at a word boundary (~55 chars)
    # so the map stays readable — full names remain in the topic tree
    def display_name(nm, limit=55):
        if len(nm) <= limit:
            return nm
        cut = nm[:limit].rsplit(" ", 1)[0].rstrip(",;")
        return cut + " …"

    layers = []
    for li, layer in enumerate(naming["layers"]):
        lab = np.load(ROOT / f"artifacts/naming/{track}__labels_layer{li}.npy")
        names = [display_name(nm) for nm in layer["names"]]
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
    # click-to-play: served/local relative URL; #t= media fragment starts playback
    # ~5s before the annotated event on positive clips
    t_start = [max(0.0, float(t) - 5.0) if (l == 1 and pd.notna(t)) else 0.0
               for l, t in zip(meta.label.values, meta.time_of_event.values)]
    video_urls = [f"videos/{c}.mp4#t={ts:.0f}" for c, ts in zip(ids, t_start)]
    extra = pd.DataFrame({
        "thumb": [thumb_uri(c) for c in ids],
        "video_url": video_urls,
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
        label_wrap_width=30,
        max_fontsize=22,
        min_fontsize=14,
        on_click="window.open(`{video_url}`)",
    )
    od = ROOT / "artifacts/atlas"
    od.mkdir(parents=True, exist_ok=True)
    out = od / f"atlas_{track}.html"
    plot.save(str(out))

    # strip ALL external <link> tags (bootstrap/font CDNs dangle when air-gapped; fonts are
    # already inlined by offline_mode). Regex over the whole doc — tags can span lines.
    import re
    html = out.read_text()
    html, removed = re.subn(r'<link[^>]+href="https?://[^"]*"[^>]*/?>', "", html)
    out.write_text(html)
    print(f"{out} written: {out.stat().st_size/1e6:.1f} MB ({n} points, {len(layers)} layers, "
          f"{removed} CDN links stripped)")
    print("ATLAS_DONE")


if __name__ == "__main__":
    main()
