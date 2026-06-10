"""Phase 6: summary tables + figures + completeness gates -> artifacts/eval/."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path("/home/b3ali/projects/stamp26")
EV = ROOT / "artifacts/eval"
gates = []


def gate(name, ok, detail=""):
    gates.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")


print("== eval (a): outlier detection ==")
frames = []
for g in ["clip", "window"]:
    f = EV / f"outlier_{g}.parquet"
    if f.exists():
        frames.append(pd.read_parquet(f))
df = pd.concat(frames, ignore_index=True)
gate("both granularities present", set(df.granularity) == {"clip", "window"}, str(set(df.granularity)))
gate("no NaN metrics", not df[["ap", "lift", "auroc"]].isna().any().any())
summ = (df.groupby(["granularity", "track", "scorer"])
        .agg(ap=("ap", "mean"), ap_lo=("ap", lambda s: s.quantile(0.05)),
             ap_hi=("ap", lambda s: s.quantile(0.95)), lift=("lift", "mean"),
             p25=("p_at_25", "mean"), auroc=("auroc", "mean"), prev=("prevalence", "mean"))
        .round(3).reset_index())
summ.to_csv(EV / "outlier_summary.csv", index=False)
print(summ.to_string(index=False))

best = summ.loc[summ.groupby("granularity").lift.idxmax()]
print("\nbest lift per granularity:")
print(best[["granularity", "track", "scorer", "ap", "lift", "prev"]].to_string(index=False))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, g in zip(axes, ["clip", "window"]):
    sub = df[df.granularity == g]
    combos = sorted(sub.groupby(["track", "scorer"]).groups)
    data = [sub[(sub.track == t) & (sub.scorer == s)].lift for t, s in combos]
    ax.boxplot(data, tick_labels=[f"{t}\n{s}" for t, s in combos])
    ax.axhline(1.0, color="red", ls="--", lw=1, label="chance")
    ax.set_title(f"{g}-level: lift over chance (AP/prevalence)")
    ax.tick_params(axis="x", rotation=75, labelsize=7)
    ax.set_yscale("log")
axes[0].set_ylabel("lift (log)")
fig.tight_layout()
fig.savefig(EV / "outlier_lift.png", dpi=120)
plt.close(fig)
print(f"figure -> {EV/'outlier_lift.png'}")

print("\n== eval (b): judge / consistency / stability ==")
jv = pd.read_parquet(EV / "judge_verdicts.parquet")
ok = jv[jv.verdict != "error"]
gate("judge verdicts parse >=95%", len(ok) / len(jv) >= 0.95, f"{len(ok)}/{len(jv)}")
score = ok.verdict.map({"yes": 1.0, "partial": 0.5, "no": 0.0})
per = ok.assign(s=score).groupby(["track", "cluster", "name"]).s.agg(["mean", "size"]).reset_index()
per.to_csv(EV / "judge_per_cluster.csv", index=False)
for tr, sub in per.groupby("track"):
    print(f"  {tr}: mean precision {sub['mean'].mean():.3f} | median {sub['mean'].median():.3f} | "
          f"<0.7: {(sub['mean'] < 0.7).sum()}/{len(sub)}")
print("  lowest-scoring clusters:")
print(per.nsmallest(6, "mean")[["track", "name", "mean", "size"]].to_string(index=False))

mc = pd.read_parquet(EV / "metadata_consistency.parquet")
if len(mc):
    print(f"  metadata claims: {len(mc)} checked, {(~mc.consistent).sum()} inconsistent "
          f"({(~mc.consistent).mean():.0%})")
st = pd.read_parquet(EV / "stability.parquet")
gate("stability computed for both tracks", set(st.track) == {"captions", "siglip2"})
print(st.to_string(index=False))

print("\nPHASE6_VERIFY_PASS" if all(gates) else f"PHASE6_VERIFY_FAIL ({sum(not g for g in gates)})")
