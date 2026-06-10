"""Phase 1: build manifest.parquet — one row per clip across all splits, with video probe results."""
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/home/b3ali/projects/stamp26")
DATA = ROOT / "data/nexar"
SPLITS = [
    ("train", "positive", 1), ("train", "negative", 0),
    ("test-public", "positive", 1), ("test-public", "negative", 0),
    ("test-private", "positive", 1), ("test-private", "negative", 0),
]


def probe(path: str):
    """Probe one video with torchcodec; returns (duration_s, fps, n_frames, ok)."""
    try:
        from torchcodec.decoders import VideoDecoder
        m = VideoDecoder(path).metadata
        dur = m.duration_seconds
        fps = m.average_fps
        if not dur or not fps or fps <= 0 or fps > 120:  # broken fps metadata guard
            return (m.num_frames / 30.0 if m.num_frames else None, 30.0, m.num_frames, dur is not None)
        return (dur, fps, m.num_frames, True)
    except Exception:
        return (None, None, None, False)


def main():
    rows = []
    for split, polarity, label in SPLITS:
        meta_path = DATA / split / polarity / "metadata.csv"
        if not meta_path.exists():
            continue
        df = pd.read_csv(meta_path)
        df["clip_id"] = df["file_name"].str.replace(".mp4", "", regex=False)
        df["split"], df["label"] = split, label
        df["path"] = df["file_name"].map(lambda f: str(DATA / split / polarity / f))
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)

    with ProcessPoolExecutor(max_workers=24) as ex:
        probes = list(ex.map(probe, df["path"].tolist(), chunksize=16))
    df[["duration_s", "fps", "n_frames", "decode_ok"]] = pd.DataFrame(probes, index=df.index)

    out = ROOT / "artifacts/manifest.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    tr = df[df.split == "train"]
    print(f"manifest: {len(df)} clips total | train {len(tr)} "
          f"(pos {int(tr.label.sum())} / neg {int((1 - tr.label).sum())}) | "
          f"decode_ok {df.decode_ok.mean():.3%}")
    print(f"train weather: {tr.weather.value_counts().to_dict()}")
    print(f"train scene:   {tr.scene.value_counts().to_dict()}")
    bad = df[~df.decode_ok.astype(bool)]
    if len(bad):
        print(f"DECODE FAILURES ({len(bad)}): {bad.clip_id.tolist()[:20]}")
    print("MANIFEST_DONE")


if __name__ == "__main__":
    main()
