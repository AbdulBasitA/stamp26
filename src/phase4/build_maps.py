"""Phase 4: per-track 5-D clustering space + 2-D display map + multi-layer ToponymyClusterer fit.
Saves to artifacts/maps/: {track}__{5d,2d}.npy, {track}__ids.json, {track}__labels_layer{i}.npy,
{track}__tree.json, and metrics.csv. Window tracks get UMAP spaces only (no clustering)."""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path("/home/b3ali/projects/stamp26")
OUT = ROOT / "artifacts/maps"
import sys

sys.path.insert(0, str(ROOT / "src/phase4"))
from tracks import CLIP_TRACKS, WINDOW_TRACKS, manifest, save_token_artifacts

import umap
from sklearn.metrics import normalized_mutual_info_score as nmi


def fit_umaps(X, metric):
    dense = not hasattr(X, "tocsr")
    kw = dict(metric=metric, unique=True)
    if dense:
        kw["init"] = "pca"
    u5 = umap.UMAP(n_components=5, n_neighbors=15, min_dist=0.0, n_epochs=500, **kw).fit_transform(X)
    u2 = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, **kw).fit_transform(X)
    return np.asarray(u5, np.float32), np.asarray(u2, np.float32)


def cluster(track, u5, X):
    from toponymy import ToponymyClusterer
    from toponymy.cluster_layer import ClusterLayerText

    emb = np.asarray(X.todense(), np.float32) if hasattr(X, "todense") else X
    c = ToponymyClusterer(min_clusters=4, min_samples=2, base_min_cluster_size=12)  # swept 2026-06-10: noise 35%->26%/22% on siglip2/captions
    c.fit(u5, emb, ClusterLayerText)
    layers = [l.cluster_labels for l in c.cluster_layers_]
    tree = {f"{k[0]},{k[1]}": [f"{a},{b}" for a, b in v] for k, v in c.cluster_tree_.items()}
    return layers, tree


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = manifest()
    rows = []
    for track, (loader, metric) in CLIP_TRACKS.items():
        print(f"=== {track} ===", flush=True)
        ids, X = loader()
        if (OUT / f"{track}__5d.npy").exists():
            u5, u2 = np.load(OUT / f"{track}__5d.npy"), np.load(OUT / f"{track}__2d.npy")
            print("  reusing saved UMAP spaces")
        else:
            u5, u2 = fit_umaps(X, metric)
        valid = np.isfinite(u5).all(1) & np.isfinite(u2).all(1)
        if not valid.all():
            print(f"  {(~valid).sum()} disconnected points masked")
        layers, tree = cluster(track, u5[valid], X[valid] if not hasattr(X, "tocsr") else X[np.where(valid)[0]])
        np.save(OUT / f"{track}__5d.npy", u5)
        np.save(OUT / f"{track}__2d.npy", u2)
        (OUT / f"{track}__ids.json").write_text(json.dumps({"ids": list(ids), "valid": valid.tolist()}))
        for i, lab in enumerate(layers):
            np.save(OUT / f"{track}__labels_layer{i}.npy", lab)
        (OUT / f"{track}__tree.json").write_text(json.dumps(tree))

        meta = man.reindex(np.asarray(ids)[valid])
        base = layers[0]
        nz = base >= 0
        m = {
            "track": track, "n": int(valid.sum()), "layers": len(layers),
            "base_clusters": int(base.max()) + 1, "noise_frac": float((~nz).mean()),
            "clusters_per_layer": "|".join(str(l.max() + 1) for l in layers),
            "nmi_scene": round(nmi(meta.scene.astype(str)[nz], base[nz]), 3),
            "nmi_light": round(nmi(meta.light_conditions.astype(str)[nz], base[nz]), 3),
            "nmi_weather": round(nmi(meta.weather.astype(str)[nz], base[nz]), 3),
        }
        df = pd.DataFrame({"c": base[nz], "y": meta.label.values[nz]})
        szs = df.groupby("c").agg(n=("y", "size"), pos=("y", "mean"))
        big = szs[szs.n >= 20]
        m["max_cluster_posrate"] = round(float(big.pos.max()), 3) if len(big) else None
        m["min_cluster_posrate"] = round(float(big.pos.min()), 3) if len(big) else None
        rows.append(m)
        print(f"  {m}", flush=True)

    for track, (loader, metric) in WINDOW_TRACKS.items():
        print(f"=== {track} (UMAP only) ===", flush=True)
        if (OUT / f"{track}__5d.npy").exists():
            print("  reusing saved UMAP spaces")
            continue
        ids, X = loader()
        u5, u2 = fit_umaps(X, metric)
        np.save(OUT / f"{track}__5d.npy", u5)
        np.save(OUT / f"{track}__2d.npy", u2)
        (OUT / f"{track}__ids.json").write_text(json.dumps({"ids": list(ids), "valid": [True] * len(ids)}))

    save_token_artifacts()
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("BUILD_MAPS_DONE")


if __name__ == "__main__":
    main()
