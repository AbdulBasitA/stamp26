"""Phase 3: define embedding windows (D9/D11) -> artifacts/windows.parquet.

Windows are 2 s / 16 frames @ 8 fps, indexed into the JPEG frame cache.
kinds: 'event' (positive, centered on time_of_event), 'negative' (tiles of negative clips),
'pos_nonevent' (tiles of positive clips not overlapping the event +/-2s — mapped but excluded from eval).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/b3ali/projects/stamp26")
WIN_S = 2.0
FPS = 8


def main():
    df = pd.read_parquet(ROOT / "artifacts/manifest.parquet")
    df = df[(df.split == "train") & df.decode_ok.astype(bool)]
    rows = []
    for _, r in df.iterrows():
        dur = r.duration_s
        if r.label == 1 and pd.notna(r.time_of_event):
            t0 = float(np.clip(r.time_of_event - WIN_S / 2, 0, dur - WIN_S - 0.05))
            rows.append((f"{r.clip_id}_evt", r.clip_id, t0, "event"))
            guard = (r.time_of_event - 2 * WIN_S, r.time_of_event + 2 * WIN_S)
        else:
            guard = None
        for ts in np.arange(0.0, dur - WIN_S - 0.05, WIN_S):
            if guard and not (ts + WIN_S < guard[0] or ts > guard[1]):
                continue  # overlaps event guard band
            kind = "negative" if r.label == 0 else "pos_nonevent"
            rows.append((f"{r.clip_id}_t{int(ts*1000):06d}", r.clip_id, float(ts), kind))
    win = pd.DataFrame(rows, columns=["window_id", "clip_id", "t_start", "kind"])
    win = win.merge(df[["clip_id", "label", "weather", "light_conditions", "scene"]], on="clip_id")
    win.to_parquet(ROOT / "artifacts/windows.parquet", index=False)
    print(f"windows: {len(win)} total | {win.kind.value_counts().to_dict()}")
    print("WINDOWS_DONE")


def window_frame_paths(clip_id: str, t_start: float, n: int = 16):
    """The n consecutive cached frames covering [t_start, t_start + n/FPS)."""
    k0 = int(round(t_start * FPS))
    return [ROOT / f"cache/frames/{clip_id}/{int(round((k0 + i) * 1000 / FPS)):07d}.jpg" for i in range(n)]


if __name__ == "__main__":
    main()
