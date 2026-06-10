"""Phase 6: leakage-free caption embeddings for eval (a) — render WITHOUT event/hazard/severity
fields (those describe the label we're trying to detect)."""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path("/home/b3ali/projects/stamp26")


def render_neutral(rec):
    agents = ", ".join(rec.get("agents") or []) or "none"
    return (f"{rec.get('caption', '')} Scene: {rec.get('scene_type')}, {rec.get('weather')}, "
            f"{rec.get('lighting')}, {rec.get('road_type')}, {rec.get('road_surface')} road, "
            f"{rec.get('traffic_density')} traffic. Agents: {agents}. Ego: {rec.get('ego_maneuver')}.")


def main():
    recs = {}
    for line in (ROOT / "artifacts/captions/clips_a.jsonl").open():
        r = json.loads(line)
        if r.get("ok"):
            recs[r["clip_id"]] = r
    ids = sorted(recs)
    # NOTE: the free-text `caption` can still narrate the event — this is "structured-field
    # neutral". Report alongside the full render, not as a perfectly label-blind variant.
    texts = [render_neutral(recs[i]) for i in ids]
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("Qwen/Qwen3-Embedding-4B", device="cuda", model_kwargs={"dtype": "bfloat16"})
    emb = st.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    od = ROOT / "artifacts/embeddings"
    np.save(od / "captions_clips_a_neutral.npy", np.asarray(emb, dtype=np.float16))
    pd.DataFrame({"id": ids}).to_parquet(od / "captions_clips_a_neutral_ids.parquet")
    print(f"captions_clips_a_neutral: {np.asarray(emb).shape}")
    print("NEUTRAL_EMBED_DONE")


if __name__ == "__main__":
    main()
