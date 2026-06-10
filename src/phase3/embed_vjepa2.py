"""Phase 3: V-JEPA2 ViT-L embeddings (mean-pooled patch tokens) for windows (16f) or clips (32f).
Reads frames from the JPEG cache. Usage: embed_vjepa2.py --what windows|clips"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")
ROOT = Path("/home/b3ali/projects/stamp26")
sys.path.insert(0, str(ROOT / "src/phase3"))
from build_windows import window_frame_paths

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"


def load_frames(paths):
    import cv2
    imgs = []
    last = None
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            img = last if last is not None else np.zeros((288, 512, 3), np.uint8)
        last = img
        imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return np.stack(imgs)  # (T, H, W, 3) uint8


class Units(Dataset):
    def __init__(self, units, n_frames):
        self.units, self.n = units, n_frames
        from transformers import AutoVideoProcessor
        self.proc = AutoVideoProcessor.from_pretrained(MODEL_ID)

    def __len__(self):
        return len(self.units)

    def __getitem__(self, i):
        uid, clip_id, t_start, duration = self.units[i]
        if t_start is not None:
            paths = window_frame_paths(clip_id, t_start, self.n)
        else:  # whole clip: n uniform frames over cached 8fps grid
            frames = sorted((ROOT / f"cache/frames/{clip_id}").glob("*.jpg"))
            idx = np.linspace(0, len(frames) - 1, self.n).round().astype(int)
            paths = [frames[j] for j in idx]
        vid = torch.from_numpy(load_frames(paths)).permute(0, 3, 1, 2)  # (T,3,H,W)
        px = self.proc(vid, return_tensors="pt")["pixel_values_videos"][0]
        return uid, px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", required=True, choices=["windows", "clips"])
    ap.add_argument("--batch", type=int, default=12)
    args = ap.parse_args()

    man = pd.read_parquet(ROOT / "artifacts/manifest.parquet")
    man = man[(man.split == "train") & man.decode_ok.astype(bool)]
    if args.what == "windows":
        win = pd.read_parquet(ROOT / "artifacts/windows.parquet")
        units = [(r.window_id, r.clip_id, r.t_start, None) for r in win.itertuples()]
        n_frames = 16
    else:
        units = [(r.clip_id, r.clip_id, None, r.duration_s) for r in man.itertuples()]
        n_frames = 32

    from transformers import AutoModel
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16,
                                      attn_implementation="sdpa").cuda().eval()
    dl = DataLoader(Units(units, n_frames), batch_size=args.batch, num_workers=8,
                    pin_memory=True)
    ids, out = [], []
    with torch.inference_mode():
        for i, (uids, px) in enumerate(dl):
            feats = model.get_vision_features(pixel_values_videos=px.cuda(non_blocking=True))
            out.append(feats.mean(dim=1).float().cpu().numpy())
            ids.extend(uids)
            if i % 50 == 0:
                print(f"  vjepa2/{args.what}: {len(ids)}/{len(units)}", flush=True)
    emb = np.concatenate(out).astype(np.float16)
    od = ROOT / "artifacts/embeddings"
    od.mkdir(parents=True, exist_ok=True)
    np.save(od / f"vjepa2_{args.what}.npy", emb)
    pd.DataFrame({"id": ids}).to_parquet(od / f"vjepa2_{args.what}_ids.parquet")
    print(f"vjepa2_{args.what}: {emb.shape} saved")
    print("VJEPA2_DONE")


if __name__ == "__main__":
    main()
