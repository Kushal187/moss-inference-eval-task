"""Experiment 3a: Moss retrieval quality vs the saved Weaviate/BM25/rerank baselines.

Local SDK path (load_index + query). Two modes:
  filtered    metadata filter doc_id == question's page  (apples-to-apples with baseline)
  unfiltered  search all 21k dev+test chunks             (what server-side retrieval sees,
                                                           since it has no filter option)

Recall@k uses the FinQA project's own recall_at_k over gold_evidence_ids.

    python -m inference.retrieval_recall --split test --mode filtered
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time

from moss import MossClient, QueryOptions

from common.finqa import load_qa
from common.moss_http import ROOT, append_jsonl, read_jsonl
from src.eval.metrics import recall_at_k  # noqa: E402  (FinQA project, via common.finqa path)

KS = (1, 3, 5, 10)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="finqa-raw")
    ap.add_argument("--split", default="test")
    ap.add_argument("--mode", choices=["filtered", "unfiltered"], default="filtered")
    ap.add_argument("--alpha", type=float, default=None, help="None = Moss default (0.8)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    qa = [q for q in load_qa() if q["split"] == args.split]
    if args.limit:
        qa = qa[: args.limit]
    tag = f"{args.index}_{args.split}_{args.mode}_a{args.alpha if args.alpha is not None else 'default'}"
    out = ROOT / "results" / "retrieval" / f"recall_{tag}.jsonl"
    done = {r["id"] for r in read_jsonl(out)}
    qa = [q for q in qa if q["id"] not in done]

    client = MossClient(os.environ["MOSS_PROJECT_ID"], os.environ["MOSS_PROJECT_KEY"])
    t0 = time.perf_counter()
    await client.load_index(args.index)
    print(f"load_index({args.index}) took {time.perf_counter() - t0:.1f}s; {len(qa)} queries -> {out}")

    for i, q in enumerate(qa, 1):
        opts = QueryOptions(top_k=max(KS), alpha=args.alpha)
        if args.mode == "filtered":
            opts.filter = {"field": "doc_id", "condition": {"$eq": q["doc_id"]}}
        t = time.perf_counter()
        res = await client.query(args.index, q["question"], opts)
        wall = (time.perf_counter() - t) * 1000
        hits = [(d.metadata or {}) for d in res.docs]
        same_doc = [h.get("local_id") for h in hits if h.get("doc_id") == q["doc_id"]]
        ranked_local = [h.get("local_id") if h.get("doc_id") == q["doc_id"] else f"OTHER::{h.get('doc_id')}"
                        for h in hits]
        row = {
            "id": q["id"], "doc_id": q["doc_id"], "gold": q["gold_evidence_ids"],
            "ranked": ranked_local,
            "wrong_doc_in_top5": sum(1 for h in hits[:5] if h.get("doc_id") != q["doc_id"]),
            "time_taken_ms": res.time_taken_ms, "wall_ms": wall,
            **{f"r@{k}": recall_at_k(ranked_local, q["gold_evidence_ids"], k) for k in KS},
        }
        append_jsonl(out, row)
        if i % 200 == 0:
            print(f"  {i}/{len(qa)}", flush=True)

    rows = read_jsonl(out)
    summ = {f"r@{k}": round(100 * statistics.mean(r[f"r@{k}"] for r in rows), 1) for k in KS}
    summ.update(n=len(rows),
                wall_ms_p50=round(statistics.median(r["wall_ms"] for r in rows), 2),
                wall_ms_p95=round(sorted(r["wall_ms"] for r in rows)[int(0.95 * len(rows)) - 1], 2),
                wrong_doc_top5_mean=round(statistics.mean(r["wrong_doc_in_top5"] for r in rows), 2))
    print(tag, summ)
    append_jsonl(ROOT / "results" / "retrieval" / "summary.jsonl", {"run": tag, **summ})


if __name__ == "__main__":
    asyncio.run(main())
