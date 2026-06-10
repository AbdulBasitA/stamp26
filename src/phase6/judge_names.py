"""Phase 6 eval (b): (1) LLM-judge per-cluster name precision on HELD-OUT members (D16);
(2) automatic name<->human-metadata consistency check. Needs the namer at localhost:8000."""
import asyncio
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path("/home/b3ali/projects/stamp26")
import sys

sys.path.insert(0, str(ROOT / "src/phase3"))
from embed_captions import render

JUDGE_SCHEMA = {"type": "object", "properties": {
    "verdict": {"type": "string", "enum": ["yes", "partial", "no"]},
    "reason": {"type": "string"}},
    "required": ["verdict", "reason"], "additionalProperties": False}

JUDGE_PROMPT = """A group of dashcam driving clips was given the scenario name: "{name}"

Here is the description of ONE clip from that group:
---
{caption}
---
Does the scenario name accurately describe this clip? Answer "yes" if the name's category and
conditions fit the clip, "partial" if it fits the broad category but misstates conditions or
specifics, "no" if it does not describe this clip. Output JSON: {{"verdict": ..., "reason": ...}}"""

WEATHER_WORDS = {"snow": ["Snow"], "rain": ["Rain"], "wet": ["Rain"], "clear": ["Clear"],
                 "sunny": ["Clear"], "cloudy": ["Cloudy"]}
LIGHT_WORDS = {"night": ["Dark", "Twilight"], "dark": ["Dark"], "twilight": ["Twilight"],
               "daytime": ["Normal", "Bright"]}


async def judge_track(track, client, model, recs, man, sem, max_members=12):
    res = json.loads((ROOT / f"artifacts/naming/{track}.json").read_text())
    ids = np.asarray(res["ids"])
    labels = np.load(ROOT / f"artifacts/naming/{track}__labels_layer0.npy")
    names = res["layers"][0]["names"]
    exemplars = res["layers"][0]["exemplar_indices"] or [[] for _ in names]
    rng = np.random.default_rng(6)

    async def one(c, cid):
        async with sem:
            try:
                r = await client.chat.completions.create(
                    model=model, max_tokens=250, temperature=0.0,
                    messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                        name=names[c], caption=render(recs[cid]))}],
                    response_format={"type": "json_schema", "json_schema": {
                        "name": "j", "strict": True, "schema": JUDGE_SCHEMA}},
                    timeout=180)
                v = json.loads(r.choices[0].message.content)["verdict"]
                return dict(track=track, cluster=c, name=names[c], clip_id=cid, verdict=v)
            except Exception as e:
                return dict(track=track, cluster=c, name=names[c], clip_id=cid,
                            verdict="error")

    tasks = []
    for c in range(len(names)):
        members = np.where(labels == c)[0]
        held_out = sorted(set(members) - set(exemplars[c] if c < len(exemplars) else []))
        pick = rng.choice(held_out, size=min(max_members, len(held_out)), replace=False)
        for j in pick:
            cid = ids[j]
            if cid in recs:
                tasks.append(one(c, cid))
    out = await asyncio.gather(*tasks)

    # metadata consistency (pure python, no LLM)
    meta_rows = []
    for c, nm in enumerate(names):
        qual = nm.lower()
        members = ids[labels == c]
        sub = man.reindex(members)
        for word, ok_vals in WEATHER_WORDS.items():
            if re.search(rf"\b{word}\b", qual):
                share = sub.weather.isin(ok_vals).mean()
                meta_rows.append(dict(track=track, cluster=c, name=nm, claim=f"weather:{word}",
                                      support=round(float(share), 3), consistent=share >= 0.3))
        for word, ok_vals in LIGHT_WORDS.items():
            if re.search(rf"\b{word}\b", qual):
                share = sub.light_conditions.isin(ok_vals).mean()
                meta_rows.append(dict(track=track, cluster=c, name=nm, claim=f"lighting:{word}",
                                      support=round(float(share), 3), consistent=share >= 0.3))
    return out, meta_rows


async def main():
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
    model = (await client.models.list()).data[0].id
    recs = {}
    for line in (ROOT / "artifacts/captions/clips_a.jsonl").open():
        r = json.loads(line)
        if r.get("ok"):
            recs[r["clip_id"]] = r
    man = pd.read_parquet(ROOT / "artifacts/manifest.parquet").set_index("clip_id")
    sem = asyncio.Semaphore(16)
    all_v, all_m = [], []
    for track in ["captions", "siglip2"]:
        v, m = await judge_track(track, client, model, recs, man, sem)
        all_v.extend(v)
        all_m.extend(m)
        df = pd.DataFrame(v)
        ok = df[df.verdict != "error"]
        score = ok.verdict.map({"yes": 1.0, "partial": 0.5, "no": 0.0})
        per = ok.assign(s=score).groupby("cluster").s.mean()
        print(f"[{track}] {len(ok)}/{len(df)} judged | mean precision {score.mean():.3f} | "
              f"clusters <0.7: {(per < 0.7).sum()}/{len(per)}")
    od = ROOT / "artifacts/eval"
    od.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_v).to_parquet(od / "judge_verdicts.parquet", index=False)
    pd.DataFrame(all_m).to_parquet(od / "metadata_consistency.parquet", index=False)
    mc = pd.DataFrame(all_m)
    if len(mc):
        print(f"metadata claims checked: {len(mc)}, inconsistent: {(~mc.consistent).sum()}")
        for _, r in mc[~mc.consistent].iterrows():
            print(f"  INCONSISTENT [{r.track}.{r.cluster}] '{r['name']}' {r.claim} support={r.support}")
    print("JUDGE_DONE")


if __name__ == "__main__":
    asyncio.run(main())
