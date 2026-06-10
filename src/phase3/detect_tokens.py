"""Phase 3: D-FINE object detection -> token bags (plan.md section 3.2).
Clips @ 1 fps (every 8th cached frame); positive event windows @ 4 fps over +/-4 s.
Output: artifacts/detections/tokens.parquet (unit_id, granularity, token, count)."""
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")
ROOT = Path("/home/b3ali/projects/stamp26")
MODEL_ID = "ustc-community/dfine-medium-obj2coco"
CONF = 0.5
PER_CLASS_CAP = 10
VEHICLES = {"car", "truck", "bus", "motorcycle"}
VRU = {"person", "bicycle"}


def frame_list():
    man = pd.read_parquet(ROOT / "artifacts/manifest.parquet")
    man = man[(man.split == "train") & man.decode_ok.astype(bool)]
    jobs = []  # (unit_id, granularity, frame_path)
    for r in man.itertuples():
        fdir = ROOT / f"cache/frames/{r.clip_id}"
        frames = sorted(fdir.glob("*.jpg"))
        for f in frames[::8]:  # 8fps cache -> 1fps
            jobs.append((r.clip_id, "clip", str(f)))
        if r.label == 1 and pd.notna(r.time_of_event):
            for dt in np.arange(-4.0, 4.0, 0.25):  # 4 fps over +/-4s
                t = r.time_of_event + dt
                if t < 0:
                    continue
                ms = int(round(round(t * 8) * 125))
                p = fdir / f"{ms:07d}.jpg"
                if p.exists():
                    jobs.append((f"{r.clip_id}_evt", "window", str(p)))
    return jobs


class Frames(Dataset):
    def __init__(self, jobs):
        self.jobs = jobs
        from transformers import AutoImageProcessor
        self.proc = AutoImageProcessor.from_pretrained(MODEL_ID)

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, i):
        import cv2
        uid, gran, path = self.jobs[i]
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        px = self.proc(images=img, return_tensors="pt")["pixel_values"][0]
        return uid, gran, px, h, w


def zone(cx, w):
    return "L" if cx < w / 3 else ("C" if cx < 2 * w / 3 else "R")


def rng(bh, h):
    f = bh / h
    return "near" if f > 0.25 else ("mid" if f >= 0.08 else "far")


def main():
    jobs = frame_list()
    print(f"detecting over {len(jobs)} frames")
    from transformers import AutoModelForObjectDetection
    model = AutoModelForObjectDetection.from_pretrained(MODEL_ID, dtype=torch.float16).cuda().eval()
    id2label = model.config.id2label
    from transformers import AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)

    dl = DataLoader(Frames(jobs), batch_size=48, num_workers=10, pin_memory=True)
    bags = {}  # (uid, gran) -> Counter
    with torch.inference_mode():
        for bi, (uids, grans, px, hs, ws) in enumerate(dl):
            out = model(pixel_values=px.cuda(non_blocking=True).half())
            sizes = torch.stack([hs, ws], dim=1)
            results = proc.post_process_object_detection(out, target_sizes=sizes, threshold=CONF)
            for uid, gran, h, w, res in zip(uids, grans, hs.tolist(), ws.tolist(), results):
                bag = bags.setdefault((uid, gran), Counter())
                per_class = Counter()
                n_veh, has_vru = 0, False
                for score, lab, box in zip(res["scores"], res["labels"], res["boxes"]):
                    name = id2label[int(lab)].replace(" ", "_")
                    if per_class[name] >= PER_CLASS_CAP:
                        continue
                    per_class[name] += 1
                    x0, y0, x1, y1 = box.tolist()
                    bag[f"{name}|{zone((x0+x1)/2, w)}|{rng(y1-y0, h)}"] += 1
                    if name in VEHICLES:
                        n_veh += 1
                    if name in VRU:
                        has_vru = True
                dens = "none" if n_veh == 0 else ("light" if n_veh <= 3 else ("moderate" if n_veh <= 8 else "heavy"))
                bag[f"traffic:{dens}"] += 1
                if has_vru:
                    bag["vru:present"] += 1
            if bi % 100 == 0:
                print(f"  frames {min((bi+1)*48, len(jobs))}/{len(jobs)}", flush=True)

    rows = [(uid, gran, tok, int(c)) for (uid, gran), bag in bags.items() for tok, c in bag.items()]
    out_df = pd.DataFrame(rows, columns=["unit_id", "granularity", "token", "count"])
    od = ROOT / "artifacts/detections"
    od.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(od / "tokens.parquet", index=False)
    print(f"token bags: {out_df.unit_id.nunique()} units, {out_df.token.nunique()} vocab, {len(out_df)} rows")
    print("DETECT_DONE")


if __name__ == "__main__":
    main()
