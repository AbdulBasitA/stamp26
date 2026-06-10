"""Phase 1: one-pass frame extraction for TRAIN clips.

Per clip: frames at 8 fps, longest side 512, JPEG q90 -> cache/frames/<clip_id>/<t_ms>.jpg
plus one thumbnail (320px wide, event frame for positives / middle frame for negatives)
-> cache/thumbs/<clip_id>.jpg. Resume-safe via per-clip .done markers.
"""
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/home/b3ali/projects/stamp26")
FRAMES = ROOT / "cache/frames"
THUMBS = ROOT / "cache/thumbs"
FPS = 8.0
MAX_SIDE = 512
THUMB_W = 320
JPEG_Q = 90


def process_clip(args):
    clip_id, path, duration, time_of_event, label = args
    out_dir = FRAMES / clip_id
    done = out_dir / ".done"
    if done.exists():
        return clip_id, "skip", 0
    try:
        import cv2
        from torchcodec.decoders import VideoDecoder

        dec = VideoDecoder(path, seek_mode="approximate")
        dur = duration if duration and duration > 0 else dec.metadata.duration_seconds
        ts = np.arange(0.0, max(dur - 0.05, 0.2), 1.0 / FPS)
        out_dir.mkdir(parents=True, exist_ok=True)

        batch = dec.get_frames_played_at(seconds=ts.tolist())
        frames = batch.data.numpy()  # (N, 3, H, W) uint8 RGB
        h, w = frames.shape[2], frames.shape[3]
        scale = MAX_SIDE / max(h, w)
        new_wh = (int(round(w * scale)), int(round(h * scale)))
        for t, fr in zip(ts, frames):
            img = cv2.cvtColor(fr.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, new_wh, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / f"{int(round(t * 1000)):07d}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])

        # thumbnail: event frame for positives, middle frame for negatives
        t_thumb = float(time_of_event) if (label == 1 and pd.notna(time_of_event)) else dur / 2
        t_thumb = min(max(t_thumb, 0.0), dur - 0.1)
        fr = dec.get_frames_played_at(seconds=[t_thumb]).data[0].numpy()
        img = cv2.cvtColor(fr.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        th_h = int(round(img.shape[0] * THUMB_W / img.shape[1]))
        img = cv2.resize(img, (THUMB_W, th_h), interpolation=cv2.INTER_AREA)
        THUMBS.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(THUMBS / f"{clip_id}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])

        done.touch()
        return clip_id, "ok", len(ts)
    except Exception as e:
        return clip_id, f"FAIL: {type(e).__name__}: {e}", 0


def main():
    df = pd.read_parquet(ROOT / "artifacts/manifest.parquet")
    df = df[(df.split == "train") & df.decode_ok.astype(bool)]
    jobs = list(zip(df.clip_id, df.path, df.duration_s, df.time_of_event, df.label))
    print(f"extracting {len(jobs)} train clips @ {FPS} fps, max side {MAX_SIDE}px")

    n_ok = n_skip = n_fail = n_frames = 0
    failures = []
    with ProcessPoolExecutor(max_workers=24) as ex:
        futs = {ex.submit(process_clip, j): j[0] for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            cid, status, nf = fut.result()
            if status == "ok":
                n_ok, n_frames = n_ok + 1, n_frames + nf
            elif status == "skip":
                n_skip += 1
            else:
                n_fail += 1
                failures.append((cid, status))
            if i % 150 == 0:
                print(f"  {i}/{len(jobs)} done (ok {n_ok}, skipped {n_skip}, failed {n_fail})", flush=True)

    print(f"extraction complete: ok {n_ok}, skipped(resume) {n_skip}, failed {n_fail}, frames {n_frames}")
    if failures:
        (FRAMES / "failures.log").write_text("\n".join(f"{c}\t{s}" for c, s in failures))
        print(f"failures logged: {[c for c, _ in failures][:10]}")
    print("EXTRACT_DONE")
    sys.exit(0 if n_fail <= len(jobs) * 0.005 else 1)


if __name__ == "__main__":
    main()
