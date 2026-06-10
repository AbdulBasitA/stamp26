"""Phase 2: async captioning client against 4 single-GPU vLLM workers (ports 8001-8004).

Usage:
  caption_clips.py --tag a|b --targets clips|windows [--pilot N]

Writes JSONL to artifacts/captions/{targets}_{tag}.jsonl, resume-safe (skips done ids).
Exits 0 and prints PILOT_PASS / RUN_PASS when parse-rate gate is met.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from schema import CAPTION_SCHEMA, CLIP_PROMPT, EXTRA_TOPK, SAMPLING, WINDOW_PROMPT

ROOT = Path("/home/b3ali/projects/stamp26")
PORTS = [8001, 8002, 8003, 8004]
PER_PORT_CONCURRENCY = 6
RETRIES = 2


async def caption_one(client, model, sem, row, targets, out_f, lock, stats):
    if targets == "clips":
        video, fps, prompt = row["path"], 1, CLIP_PROMPT
    else:
        video, fps, prompt = str(ROOT / f"cache/event_mp4/{row['clip_id']}.mp4"), 4, WINDOW_PROMPT
    async with sem:
        for attempt in range(RETRIES + 1):
            t0 = time.time()
            try:
                r = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": [
                        {"type": "video_url", "video_url": {"url": f"file://{video}"}},
                        {"type": "text", "text": prompt},
                    ]}],
                    response_format={"type": "json_schema", "json_schema": {
                        "name": "caption", "strict": True, "schema": CAPTION_SCHEMA}},
                    extra_body={"mm_processor_kwargs": {"fps": fps},
                                "chat_template_kwargs": {"enable_thinking": False},
                                **EXTRA_TOPK},
                    timeout=420,
                    **SAMPLING,
                )
                content = r.choices[0].message.content
                parsed = json.loads(content)
                rec = {"clip_id": row["clip_id"], "targets": targets, "ok": True,
                       "latency_s": round(time.time() - t0, 2),
                       "completion_tokens": r.usage.completion_tokens if r.usage else None,
                       **parsed}
                async with lock:
                    out_f.write(json.dumps(rec) + "\n")
                    out_f.flush()
                    stats["ok"] += 1
                return
            except Exception as e:
                if attempt == RETRIES:
                    rec = {"clip_id": row["clip_id"], "targets": targets, "ok": False,
                           "error": f"{type(e).__name__}: {str(e)[:300]}"}
                    async with lock:
                        out_f.write(json.dumps(rec) + "\n")
                        out_f.flush()
                        stats["fail"] += 1
                else:
                    await asyncio.sleep(3 * (attempt + 1))


async def run(args):
    from openai import AsyncOpenAI

    df = pd.read_parquet(ROOT / "artifacts/manifest.parquet")
    df = df[(df.split == "train") & df.decode_ok.astype(bool)]
    if args.targets == "windows":
        df = df[df.label == 1]
    if args.pilot:
        df = df.sample(args.pilot, random_state=7)

    out_path = ROOT / f"artifacts/captions/{args.targets}_{args.tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists() and not args.pilot:
        done = {json.loads(l)["clip_id"] for l in out_path.open() if '"ok": true' in l}
    todo = df[~df.clip_id.isin(done)]
    print(f"[{args.targets}/{args.tag}] {len(todo)} to caption ({len(done)} already done)")
    if not len(todo):
        print("RUN_PASS (nothing to do)")
        return

    clients, models, sems = [], [], []
    for p in PORTS:
        c = AsyncOpenAI(api_key="EMPTY", base_url=f"http://localhost:{p}/v1")
        clients.append(c)
        models.append((await c.models.list()).data[0].id)
        sems.append(asyncio.Semaphore(PER_PORT_CONCURRENCY))

    stats = {"ok": 0, "fail": 0}
    lock = asyncio.Lock()
    t0 = time.time()
    with out_path.open("a") as out_f:
        tasks = []
        for i, (_, row) in enumerate(todo.iterrows()):
            k = i % len(PORTS)
            tasks.append(caption_one(clients[k], models[k], sems[k], row, args.targets, out_f, lock, stats))
        # progress reporting
        async def report():
            while True:
                await asyncio.sleep(120)
                n = stats["ok"] + stats["fail"]
                rate = n / max(time.time() - t0, 1) * 3600
                print(f"  progress {n}/{len(todo)} ({stats['fail']} failed, {rate:.0f}/h)", flush=True)
        rep = asyncio.create_task(report())
        await asyncio.gather(*tasks)
        rep.cancel()

    n = stats["ok"] + stats["fail"]
    pr = stats["ok"] / max(n, 1)
    dt = time.time() - t0
    print(f"[{args.targets}/{args.tag}] done: {stats['ok']}/{n} ok ({pr:.1%}), "
          f"{dt/60:.1f} min, {n/(dt/3600):.0f} req/h")
    gate = 0.95 if args.pilot else 0.985
    label = "PILOT" if args.pilot else "RUN"
    print(f"{label}_{'PASS' if pr >= gate else 'FAIL'}")
    sys.exit(0 if pr >= gate else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["a", "b"])
    ap.add_argument("--targets", required=True, choices=["clips", "windows"])
    ap.add_argument("--pilot", type=int, default=0)
    asyncio.run(run(ap.parse_args()))
