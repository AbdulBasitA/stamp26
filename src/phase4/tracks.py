"""Phase 4: track registry — load each embedding track (ids, matrix, metric) at clip level,
including the token track built the hosts' way (NgramVectorizer -> InformationWeightTransformer)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/b3ali/projects/stamp26")
ED = ROOT / "artifacts/embeddings"


def manifest():
    man = pd.read_parquet(ROOT / "artifacts/manifest.parquet")
    return man[(man.split == "train") & man.decode_ok.astype(bool)].set_index("clip_id")


def _dense(name):
    emb = np.load(ED / f"{name}.npy").astype(np.float32)
    ids = pd.read_parquet(ED / f"{name}_ids.parquet")["id"].tolist()
    return ids, emb


def build_token_matrix(with_metadata: bool):
    """Clip-level bag-of-detected-objects -> IWT-weighted sparse counts (host doctrine, D7/D12)."""
    from vectorizers import NgramVectorizer
    from vectorizers.transformers import InformationWeightTransformer

    tok = pd.read_parquet(ROOT / "artifacts/detections/tokens.parquet")
    tok = tok[tok.granularity == "clip"]
    man = manifest()
    bags = {}
    for r in tok.itertuples():
        bags.setdefault(r.unit_id, []).extend([r.token] * int(r.count))
    if with_metadata:
        for cid, bag in bags.items():
            m = man.loc[cid]
            n_rep = max(1, round(0.05 * len(bag) / 3))
            for t in (f"weather:{m.weather}", f"lighting:{m.light_conditions}", f"scene:{m.scene}"):
                bag.extend([t] * n_rep)
    ids = sorted(bags)
    docs = [bags[i] for i in ids]
    ngram = NgramVectorizer(ngram_size=1)
    counts = ngram.fit_transform(docs)
    iwt = InformationWeightTransformer().fit_transform(counts)
    vocab = [t for t, _ in sorted(ngram.column_label_dictionary_.items(), key=lambda kv: kv[1])]
    return ids, iwt, vocab


CLIP_TRACKS = {
    # name: (loader, umap_metric)
    "captions": (lambda: _dense("captions_clips_a"), "cosine"),
    "siglip2": (lambda: _dense("siglip2_clips"), "cosine"),
    "vjepa2": (lambda: _dense("vjepa2_clips"), "cosine"),
    "tokens": (lambda: build_token_matrix(False)[:2], "hellinger"),
    "tokens_meta": (lambda: build_token_matrix(True)[:2], "hellinger"),
}

WINDOW_TRACKS = {
    "siglip2_windows": (lambda: _dense("siglip2_windows"), "cosine"),
    "vjepa2_windows": (lambda: _dense("vjepa2_windows"), "cosine"),
}


def save_token_artifacts():
    """Persist IWT matrices + vocab for cluster characterization and Phase 6."""
    out = ROOT / "artifacts/maps"
    out.mkdir(parents=True, exist_ok=True)
    for name, with_meta in [("tokens", False), ("tokens_meta", True)]:
        ids, iwt, vocab = build_token_matrix(with_meta)
        import scipy.sparse as sp
        sp.save_npz(out / f"{name}_iwt.npz", iwt.tocsr())
        (out / f"{name}_meta.json").write_text(json.dumps({"ids": ids, "vocab": vocab}))
