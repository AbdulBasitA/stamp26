"""Phase 2: cut +/-5s event-window mp4s (4 fps, h264) for the 750 positives -> cache/event_mp4/.
Resume-safe. Frames decoded from the ORIGINAL videos at full resolution."""
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path("/home/b3ali/projects/stamp26")
OUT = ROOT / "cache/event_mp4"
FPS = 4.0
HALF = 5.0


def cut(args):
    clip_id, path, t_event, duration = args
    out = OUT / f"{clip_id}.mp4"
    if out.exists() and out.stat().st_size > 50_000:
        return clip_id, "skip"
    try:
        import av
        from torchcodec.decoders import VideoDecoder

        dec = VideoDecoder(path, seek_mode="approximate")
        dur = duration if duration and duration > 0 else dec.metadata.duration_seconds
        t0, t1 = max(0.0, t_event - HALF), min(dur - 0.05, t_event + HALF)
        ts = np.arange(t0, t1, 1.0 / FPS)
        frames = dec.get_frames_played_at(seconds=ts.tolist()).data.numpy()  # (N,3,H,W) RGB

        OUT.mkdir(parents=True, exist_ok=True)
        with av.open(str(out), "w") as container:
            stream = container.add_stream("h264", rate=int(FPS))
            stream.width, stream.height = frames.shape[3], frames.shape[2]
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "23", "preset": "fast"}
            for fr in frames:
                vf = av.VideoFrame.from_ndarray(fr.transpose(1, 2, 0), format="rgb24")
                for pkt in stream.encode(vf):
                    container.mux(pkt)
            for pkt in stream.encode():
                container.mux(pkt)
        return clip_id, "ok"
    except Exception as e:
        return clip_id, f"FAIL: {type(e).__name__}: {e}"


def main():
    df = pd.read_parquet(ROOT / "artifacts/manifest.parquet")
    pos = df[(df.split == "train") & (df.label == 1) & df.decode_ok.astype(bool)]
    jobs = list(zip(pos.clip_id, pos.path, pos.time_of_event, pos.duration_s))
    n_ok = n_skip = 0
    fails = []
    with ProcessPoolExecutor(max_workers=16) as ex:
        for fut in as_completed({ex.submit(cut, j) for j in jobs}):
            cid, st = fut.result()
            if st == "ok":
                n_ok += 1
            elif st == "skip":
                n_skip += 1
            else:
                fails.append((cid, st))
    print(f"event windows: ok {n_ok}, skipped {n_skip}, failed {len(fails)} {fails[:5]}")
    print("CUT_WINDOWS_DONE" if len(fails) <= 3 else "CUT_WINDOWS_FAILED")


if __name__ == "__main__":
    main()
