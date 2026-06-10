"""Phase 5: Toponymy naming on a primary track against the local vLLM namer.
Usage: name_topics.py --track captions|siglip2
Saves artifacts/naming/{track}.json + per-layer label/name vectors."""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path("/home/b3ali/projects/stamp26")
sys.path.insert(0, str(ROOT / "src/phase5"))
sys.path.insert(0, str(ROOT / "src/phase3"))
from driving_templates import DrivingClusterLayer, build_templates, set_layer_metadata
from embed_captions import render


def load_track(track):
    info = json.loads((ROOT / f"artifacts/maps/{track}__ids.json").read_text())
    ids = np.asarray(info["ids"])
    valid = np.asarray(info["valid"])
    u5 = np.load(ROOT / f"artifacts/maps/{track}__5d.npy")
    if track == "captions":
        emb = np.load(ROOT / "artifacts/embeddings/captions_clips_a.npy").astype(np.float32)
    else:
        emb = np.load(ROOT / f"artifacts/embeddings/{track}_clips.npy").astype(np.float32)

    recs = {}
    for line in (ROOT / "artifacts/captions/clips_a.jsonl").open():
        r = json.loads(line)
        if r.get("ok"):
            recs[r["clip_id"]] = r
    keep = valid & np.array([i in recs for i in ids])
    texts = [render(recs[i]) for i in ids[keep]]
    return ids[keep], texts, emb[keep], u5[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=["captions", "siglip2"])
    args = ap.parse_args()

    ids, texts, emb, u5 = load_track(args.track)
    print(f"[{args.track}] naming over {len(ids)} clips")

    man = pd.read_parquet(ROOT / "artifacts/manifest.parquet").set_index("clip_id")
    meta = man.reindex(ids)[["label", "scene", "light_conditions", "weather"]]
    set_layer_metadata(meta)

    from sentence_transformers import SentenceTransformer
    from toponymy import Toponymy, ToponymyClusterer
    from toponymy.llm_wrappers import OpenAINamer

    embedder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B",
                                   device=os.environ.get("EMBED_DEVICE", "cuda"))
    llm = OpenAINamer(api_key="EMPTY", model="namer", base_url="http://localhost:8000/v1")

    topic_model = Toponymy(
        llm_wrapper=llm,
        text_embedding_model=embedder,
        clusterer=ToponymyClusterer(min_clusters=4, min_samples=2, base_min_cluster_size=12),
        layer_class=DrivingClusterLayer,
        prompt_template=build_templates(),
        object_description="ego-centric dashcam driving scenario clips (each described by a structured caption)",
        corpus_description="a road-safety dataset of 1,500 dashcam clips, half containing collisions or near-collisions",
        show_progress_bars=True,
    )
    topic_model.fit(texts, emb, u5)

    out = {"track": args.track, "ids": ids.tolist(), "layers": []}
    od = ROOT / "artifacts/naming"
    od.mkdir(parents=True, exist_ok=True)
    for i, layer in enumerate(topic_model.cluster_layers_):
        names = topic_model.topic_names_[i]
        labels = layer.cluster_labels
        np.save(od / f"{args.track}__labels_layer{i}.npy", labels)
        exemplar_idx = getattr(layer, "exemplar_indices", None)
        out["layers"].append({
            "layer": i,
            "names": list(names),
            "sizes": [int((labels == c).sum()) for c in range(len(names))],
            "exemplar_indices": [list(map(int, e)) for e in exemplar_idx] if exemplar_idx is not None else None,
        })
        print(f"  layer {i}: {len(names)} topics")
        for c, nm in enumerate(names):
            print(f"    [{i}.{c}] (n={(labels == c).sum()}) {nm}")
    (od / f"{args.track}.json").write_text(json.dumps(out))
    np.save(od / f"{args.track}__topic_name_vectors.npy",
            np.array([v for v in topic_model.topic_name_vectors_], dtype=object), allow_pickle=True)
    print("NAMING_DONE")


if __name__ == "__main__":
    main()
