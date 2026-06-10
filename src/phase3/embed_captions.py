"""Phase 3: Qwen3-Embedding-4B embeddings of VLM captions (Track C), per caption tag."""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path("/home/b3ali/projects/stamp26")


def render(rec):
    agents = ", ".join(rec.get("agents") or []) or "none"
    parts = [
        rec.get("caption", ""),
        f"Scene: {rec.get('scene_type')}, {rec.get('weather')}, {rec.get('lighting')},"
        f" {rec.get('road_type')}, {rec.get('road_surface')} road, {rec.get('traffic_density')} traffic.",
        f"Agents: {agents}. Ego: {rec.get('ego_maneuver')}.",
    ]
    if rec.get("event_observed"):
        parts.append(f"Event ({rec.get('event_type')}, {rec.get('severity')}): {rec.get('event_description')}")
    if rec.get("hazard_description"):
        parts.append(f"Hazard: {rec['hazard_description']}")
    return " ".join(p for p in parts if p)


def main():
    from sentence_transformers import SentenceTransformer
    st = None
    od = ROOT / "artifacts/embeddings"
    od.mkdir(parents=True, exist_ok=True)
    for targets in ["clips", "windows"]:
        for tag in ["a", "b"]:
            f = ROOT / f"artifacts/captions/{targets}_{tag}.jsonl"
            if not f.exists():
                continue
            recs = {}
            for line in f.open():
                r = json.loads(line)
                if r.get("ok"):
                    recs[r["clip_id"]] = r  # last write wins
            if len(recs) < 50:
                continue
            if st is None:
                st = SentenceTransformer("Qwen/Qwen3-Embedding-4B", device="cuda",
                                         model_kwargs={"dtype": "bfloat16"})
            ids = sorted(recs)
            texts = [render(recs[i]) for i in ids]
            emb = st.encode(texts, batch_size=64, normalize_embeddings=True,
                            show_progress_bar=False)
            np.save(od / f"captions_{targets}_{tag}.npy", np.asarray(emb, dtype=np.float16))
            pd.DataFrame({"id": ids, "text": texts}).to_parquet(od / f"captions_{targets}_{tag}_ids.parquet")
            print(f"captions_{targets}_{tag}: {np.asarray(emb).shape}")
    print("CAPTION_EMBED_DONE")


if __name__ == "__main__":
    main()
