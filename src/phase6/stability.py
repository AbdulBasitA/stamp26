"""Phase 6 eval (b): naming stability across 3 independent runs (D13).
Assumes runs _run2/_run3 already produced by name_topics.py --suffix. Matches base-layer
clusters across runs by membership Jaccard, reports name agreement."""
import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path("/home/b3ali/projects/stamp26")


def load_run(track, suffix):
    res = json.loads((ROOT / f"artifacts/naming/{track}{suffix}.json").read_text())
    labels = np.load(ROOT / f"artifacts/naming/{track}{suffix}__labels_layer0.npy")
    return np.asarray(res["ids"]), labels, res["layers"][0]["names"]


def category(name):
    return name.split(":")[0].strip() if ":" in name else name


def match_runs(a, b):
    ids_a, lab_a, names_a = a
    ids_b, lab_b, names_b = b
    common = np.intersect1d(ids_a, ids_b)
    pos_a = {c: set(ids_a[lab_a == c]) & set(common) for c in range(len(names_a))}
    pos_b = {c: set(ids_b[lab_b == c]) & set(common) for c in range(len(names_b))}
    rows = []
    for ca, mem_a in pos_a.items():
        if len(mem_a) < 12:
            continue
        best, bj = None, 0.0
        for cb, mem_b in pos_b.items():
            if not mem_b:
                continue
            j = len(mem_a & mem_b) / len(mem_a | mem_b)
            if j > bj:
                best, bj = cb, j
        rows.append(dict(cluster=ca, jaccard=bj, matched=bj >= 0.3,
                         exact=bool(best is not None and names_a[ca] == names_b[best]),
                         cat=bool(best is not None and category(names_a[ca]) == category(names_b[best])),
                         name_a=names_a[ca], name_b=names_b[best] if best is not None else ""))
    return pd.DataFrame(rows)


def main():
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")
    out = []
    for track in ["captions", "siglip2"]:
        runs = [load_run(track, s) for s in ["", "_run2", "_run3"]]
        for (i, ra), (j, rb) in combinations(enumerate(runs), 2):
            df = match_runs(ra, rb)
            m = df[df.matched]
            if len(m):
                ea = st.encode(m.name_a.tolist(), normalize_embeddings=True)
                eb = st.encode(m.name_b.tolist(), normalize_embeddings=True)
                cos = float((ea * eb).sum(1).mean())
            else:
                cos = float("nan")
            out.append(dict(track=track, pair=f"{i+1}-{j+1}", clusters=len(df),
                            matched_frac=round(float(df.matched.mean()), 3),
                            mean_jaccard=round(float(df.jaccard.mean()), 3),
                            exact_name=round(float(m.exact.mean()), 3) if len(m) else None,
                            category_match=round(float(m.cat.mean()), 3) if len(m) else None,
                            name_cosine=round(cos, 3)))
    df = pd.DataFrame(out)
    df.to_parquet(ROOT / "artifacts/eval/stability.parquet", index=False)
    print(df.to_string(index=False))
    print("STABILITY_DONE")


if __name__ == "__main__":
    main()
