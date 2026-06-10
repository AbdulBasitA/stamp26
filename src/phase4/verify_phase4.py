"""Phase 4 verify gates: >=2 layers on >=2 tracks, base-layer noise <30%, metrics table."""
from pathlib import Path

import pandas as pd

ROOT = Path("/home/b3ali/projects/stamp26")
m = pd.read_csv(ROOT / "artifacts/maps/metrics.csv")
print(m.to_string(index=False))
ok_layers = (m.layers >= 2).sum()
ok_noise = (m.noise_frac < 0.30)
gates = [
    (">=2 layers on >=2 tracks", ok_layers >= 2, f"{ok_layers}/{len(m)} tracks"),
    ("noise <30% on >=2 tracks", ok_noise.sum() >= 2, f"{ok_noise.sum()}/{len(m)}"),
    ("every track produced clusters", (m.base_clusters >= 4).all(), str(m.base_clusters.tolist())),
]
fails = 0
for name, ok, detail in gates:
    print(f"[{'PASS' if ok else 'FAIL'}] {name} ({detail})")
    fails += not ok
print("PHASE4_VERIFY_PASS" if fails == 0 else f"PHASE4_VERIFY_FAIL ({fails})")
