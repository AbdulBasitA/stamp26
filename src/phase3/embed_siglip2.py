"""Phase 3: SigLIP2 so400m frame embeddings, L2-norm -> mean -> L2-norm pooled.
8 frames per window/clip. Usage: embed_siglip2.py --what windows|clips"""
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
from embed_vjepa2 import load_frames

MODEL_ID = "google/siglip2-so400m-patch16-384"
N_FRAMES = 8


class Units(Dataset):
    def __init__(self, units):
        self.units = units
        from transformers import AutoProcessor
        self.proc = AutoProcessor.from_pretrained(MODEL_ID)

    def __len__(self):
        return len(self.units)

    def __getitem__(self, i):
        uid, clip_id, t_start = self.units[i]
        if t_start is not None:
            paths = window_frame_paths(clip_id, t_start, 16)[::2]  # 8 of the 16
        else:
            frames = sorted((ROOT / f"cache/frames/{clip_id}").glob("*.jpg"))
            idx = np.linspace(0, len(frames) - 1, N_FRAMES).round().astype(int)
            paths = [frames[j] for j in idx]
        imgs = list(load_frames(paths))
        px = self.proc(images=imgs, return_tensors="pt")["pixel_values"]  # (8,3,384,384)
        return uid, px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", required=True, choices=["windows", "clips"])
    ap.add_argument("--batch", type=int, default=24)
    args = ap.parse_args()

    man = pd.read_parquet(ROOT / "artifacts/manifest.parquet")
    man = man[(man.split == "train") & man.decode_ok.astype(bool)]
    if args.what == "windows":
        win = pd.read_parquet(ROOT / "artifacts/windows.parquet")
        units = [(r.window_id, r.clip_id, r.t_start) for r in win.itertuples()]
    else:
        units = [(r.clip_id, r.clip_id, None) for r in man.itertuples()]

    from transformers import AutoModel
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16,
                                      attn_implementation="sdpa").cuda().eval()
    dl = DataLoader(Units(units), batch_size=args.batch, num_workers=8, pin_memory=True)
    ids, out = [], []
    with torch.inference_mode():
        for i, (uids, px) in enumerate(dl):
            b, t = px.shape[0], px.shape[1]
            feats = model.get_image_features(pixel_values=px.flatten(0, 1).cuda(non_blocking=True))
            feats = torch.nn.functional.normalize(feats.float(), dim=-1).view(b, t, -1)
            pooled = torch.nn.functional.normalize(feats.mean(dim=1), dim=-1)
            out.append(pooled.cpu().numpy())
            ids.extend(uids)
            if i % 50 == 0:
                print(f"  siglip2/{args.what}: {len(ids)}/{len(units)}", flush=True)
    emb = np.concatenate(out).astype(np.float16)
    od = ROOT / "artifacts/embeddings"
    od.mkdir(parents=True, exist_ok=True)
    np.save(od / f"siglip2_{args.what}.npy", emb)
    pd.DataFrame({"id": ids}).to_parquet(od / f"siglip2_{args.what}_ids.parquet")
    print(f"siglip2_{args.what}: {emb.shape} saved")
    print("SIGLIP2_DONE")


if __name__ == "__main__":
    main()
