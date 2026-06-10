"""Phase 6 eval (a): low-prevalence collision-as-outlier detection (D22).

Protocol per (track, k, resample): subset = ALL negatives + k stratified positives ->
full refit of 5-D UMAP + three scorers -> AP / P@25 / P@50 / AUROC / lift-over-chance.
Usage: outlier_eval.py --granularity clip|window [--resamples N] [--workers N]
"""
import argparse
import multiprocessing
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path("/home/b3ali/projects/stamp26")
ED = ROOT / "artifacts/embeddings"
K_VALUES = [25, 50, 75]


def clip_tracks():
    import sys
    sys.path.insert(0, str(ROOT / "src/phase4"))
    from tracks import build_token_matrix
    out = {}
    for name, f in [("captions_full", "captions_clips_a"), ("captions_neutral", "captions_clips_a_neutral"),
                    ("siglip2", "siglip2_clips"), ("vjepa2", "vjepa2_clips")]:
        ids = pd.read_parquet(ED / f"{f}_ids.parquet")["id"].tolist()
        out[name] = (ids, np.load(ED / f"{f}.npy").astype(np.float32), "cosine")
    ids, iwt, _ = build_token_matrix(False)
    out["tokens"] = (ids, iwt.astype(np.float32), "hellinger")
    return out


def window_tracks():
    win = pd.read_parquet(ROOT / "artifacts/windows.parquet").set_index("window_id")
    out = {}
    for name, f in [("siglip2", "siglip2_windows"), ("vjepa2", "vjepa2_windows")]:
        ids = pd.read_parquet(ED / f"{f}_ids.parquet")["id"].tolist()
        out[name] = (ids, np.load(ED / f"{f}.npy").astype(np.float32), "cosine")
    return out, win


def score_subset(args):
    """One (track, k, resample) job: returns metric rows for 3 scorers."""
    (track, metric, k, seed, X_sub, y_sub) = args
    import hdbscan
    import umap
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

    warnings.filterwarnings("ignore")
    n = len(y_sub)
    u5 = umap.UMAP(n_components=5, n_neighbors=15, min_dist=0.0, metric=metric,
                   n_epochs=300).fit_transform(X_sub)
    u5 = np.nan_to_num(np.asarray(u5, np.float64))
    mcs = 25 if n > 5000 else 12
    scores = {}
    h = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=5).fit(u5)
    scores["glosh"] = np.nan_to_num(h.outlier_scores_)
    lof = LocalOutlierFactor(n_neighbors=20).fit(u5)
    scores["lof"] = -lof.negative_outlier_factor_
    if metric == "cosine":
        nn = NearestNeighbors(n_neighbors=11, metric="cosine").fit(X_sub)
        d, _ = nn.kneighbors(X_sub)
        scores["knn_highdim"] = d[:, 1:].mean(1)
    rows = []
    prev = y_sub.mean()
    for scorer, s in scores.items():
        order = np.argsort(-s)
        ap = average_precision_score(y_sub, s)
        rows.append(dict(track=track, k=k, seed=seed, scorer=scorer, n=n, prevalence=prev,
                         ap=ap, lift=ap / prev,
                         p_at_25=y_sub[order[:25]].mean(), p_at_50=y_sub[order[:50]].mean(),
                         auroc=roc_auc_score(y_sub, s)))
    return rows


def build_jobs_clip(resamples):
    man = pd.read_parquet(ROOT / "artifacts/manifest.parquet")
    man = man[(man.split == "train") & man.decode_ok.astype(bool)].set_index("clip_id")
    pos = man[man.label == 1].copy()
    pos["gap_q"] = pd.qcut(pos.time_of_event - pos.time_of_alert, 4, labels=False, duplicates="drop")
    jobs = []
    for track, (ids, X, metric) in clip_tracks().items():
        ids = np.asarray(ids)
        lab = man.reindex(ids).label.values
        neg_idx = np.where(lab == 0)[0]
        pos_ids_avail = set(ids[lab == 1])
        pos_avail = pos[pos.index.isin(pos_ids_avail)]
        id_to_row = {cid: i for i, cid in enumerate(ids)}
        for k in K_VALUES:
            for r in range(resamples):
                rng = np.random.default_rng(1000 * k + r)
                sample = (pos_avail.groupby("gap_q", group_keys=False)
                          .apply(lambda g: g.sample(max(1, round(k * len(g) / len(pos_avail))),
                                                    random_state=rng.integers(1 << 30))))
                sel = [id_to_row[c] for c in sample.index[:k]]
                rows = np.concatenate([neg_idx, np.asarray(sel, int)])
                y = np.concatenate([np.zeros(len(neg_idx)), np.ones(len(sel))])
                jobs.append((track, metric, k, r, X[rows], y))
    return jobs


def build_jobs_window(resamples):
    tracks, win = window_tracks()
    jobs = []
    for track, (ids, X, metric) in tracks.items():
        ids = np.asarray(ids)
        kind = win.reindex(ids).kind.values
        clip_of = win.reindex(ids).clip_id.values
        neg_idx = np.where(kind == "negative")[0]
        evt_idx = {clip_of[i]: i for i in np.where(kind == "event")[0]}
        evt_clips = sorted(evt_idx)
        for k in K_VALUES:
            for r in range(resamples):
                rng = np.random.default_rng(2000 * k + r)
                sel_clips = rng.choice(evt_clips, size=k, replace=False)
                sel = [evt_idx[c] for c in sel_clips]
                rows = np.concatenate([neg_idx, np.asarray(sel, int)])
                y = np.concatenate([np.zeros(len(neg_idx)), np.ones(len(sel))])
                jobs.append((track, metric, k, r, X[rows], y))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--granularity", required=True, choices=["clip", "window"])
    ap.add_argument("--resamples", type=int, default=None)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    resamples = args.resamples or (30 if args.granularity == "clip" else 15)
    jobs = build_jobs_clip(resamples) if args.granularity == "clip" else build_jobs_window(resamples)
    print(f"{args.granularity}: {len(jobs)} subset jobs")
    rows = []
    # spawn: parent has numba/OpenMP loaded (token matrix build) — fork is unsafe
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn")) as ex:
        futs = [ex.submit(score_subset, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            rows.extend(f.result())
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)} subsets scored", flush=True)
    df = pd.DataFrame(rows)
    df["granularity"] = args.granularity
    od = ROOT / "artifacts/eval"
    od.mkdir(parents=True, exist_ok=True)
    df.to_parquet(od / f"outlier_{args.granularity}.parquet", index=False)
    print(df.groupby(["track", "scorer", "k"])[["ap", "lift", "p_at_25"]].mean().round(3).to_string())
    print(f"OUTLIER_{args.granularity.upper()}_DONE")


if __name__ == "__main__":
    main()
