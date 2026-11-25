"""Experiment 1: swap the generator, hold everything else fixed.

Replays the FinQA project's saved run (150 dev questions x 4 prompt strategies, rerank
k=5 evidence) through a Moss-served model. Same retrieved chunk ids, same prompt
builders, same answer parser and matcher as the llama-3.3-70b baseline, so accuracy
differences are attributable to the model. Streams every call to measure TTFT.

Resumable: rows already in --out are skipped.

    python -m inference.gen_swap --model moss/fastlane-1
    python -m inference.gen_swap --model anthropic/claude-haiku-5.5 --strategies zero_shot cot
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx

from common.finqa import (SAVED_RUNS, answers_match, build_messages, extract_final_answer,
                          get_chunk_contents, load_few_shot_examples, normalize_answer)
from common.moss_http import ROOT, RateLimiter, append_jsonl, read_jsonl, respond

STRATS = ["zero_shot", "few_shot", "cot", "structured"]


def build_body(model: str, messages: list[dict], needs_json: bool, drop: set[str]) -> dict:
    body = {"model": model, "input": messages, "max_output_tokens": 512}
    if "temperature" not in drop:
        body["temperature"] = 0.0
    if needs_json and "text" not in drop:
        body["text"] = {"format": {"type": "json_object"}}
    return body


async def run_one(client, sem, limiter, model, base, few_shot, drop, out_path):
    async with sem:
        context = get_chunk_contents(base["doc_id"], base["retrieved_ids"])
        messages, needs_json = build_messages(base["strategy"], base["question"], context, few_shot)
        refused = []
        while True:
            body = build_body(model, messages, needs_json, drop)
            rec = await respond(client, body, limiter=limiter)
            err = rec.get("error") or {}
            msg = json.dumps(err).lower()
            # Fall back once per refused parameter, and remember it for the rest of the run.
            # Check what THIS request sent (not the shared set) so concurrent calls retry too.
            if rec.get("status") == 400 and "temperature" in msg and "temperature" in body:
                drop.add("temperature"); refused.append("temperature"); continue
            if rec.get("status") == 400 and ("json" in msg or "text" in msg or "format" in msg) and "text" in body:
                drop.add("text"); refused.append("text.format=json_object"); continue
            break
        raw = rec.get("text") or ""
        answer = extract_final_answer(raw, base["strategy"]) if raw else ""
        _v, kind = normalize_answer(answer)
        row = {
            "id": base["id"], "doc_id": base["doc_id"], "question": base["question"],
            "strategy": base["strategy"], "model": model,
            "retrieved_ids": base["retrieved_ids"],
            "answer": answer, "answer_type": kind,
            "abstained": kind in ("abstention", "none"),
            "correct": answers_match(answer, base["gold_answer"], base["gold_answer_exe"]) if raw else False,
            "gold_answer": base["gold_answer"], "gold_answer_exe": base["gold_answer_exe"],
            "baseline_correct": base["correct"], "baseline_latency_ms": base["latency_ms"],
            "ttft_ms": rec.get("ttft_ms"), "total_ms": rec.get("total_ms"),
            "status": rec.get("status"), "error": rec.get("error"), "retries": rec.get("retries"),
            "usage": rec.get("usage"), "request_id": rec.get("request_id"),
            "headers": rec.get("headers"), "params_refused": refused,
            "params_dropped": sorted(drop), "raw": raw,
        }
        append_jsonl(out_path, row)
        return row


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--strategies", nargs="+", default=STRATS)
    ap.add_argument("--limit", type=int, default=None, help="first N questions")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--rps", type=float, default=1.5, help="request starts per second")
    ap.add_argument("--out", default=None)
    ap.add_argument("--drop", nargs="*", default=[], help="params to never send, e.g. text temperature")
    args = ap.parse_args()

    baseline = read_jsonl(SAVED_RUNS / "prompts_dev_rerank.jsonl")
    qids = list(dict.fromkeys(r["id"] for r in baseline))
    if args.limit:
        qids = set(qids[: args.limit])
    todo = [r for r in baseline if r["strategy"] in args.strategies and r["id"] in set(qids)]

    out = Path(args.out) if args.out else ROOT / "results" / "generation" / f"{args.model.replace('/', '__')}.jsonl"
    done = {(r["id"], r["strategy"]) for r in read_jsonl(out) if r.get("status") == 200}
    todo = [r for r in todo if (r["id"], r["strategy"]) not in done]
    print(f"{args.model}: {len(todo)} calls to make ({len(done)} already done) -> {out}")

    few_shot = load_few_shot_examples(n=3, seed=42) if "few_shot" in args.strategies else None
    sem, limiter, drop = asyncio.Semaphore(args.concurrency), RateLimiter(args.rps), set(args.drop)
    async with httpx.AsyncClient(http2=False) as client:
        n = 0
        for coro in asyncio.as_completed([run_one(client, sem, limiter, args.model, b, few_shot, drop, out) for b in todo]):
            row = await coro
            n += 1
            if n % 25 == 0 or row["status"] != 200:
                print(f"  {n}/{len(todo)} status={row['status']} ttft={row['ttft_ms'] and round(row['ttft_ms'])}ms "
                      f"err={(row['error'] or {}).get('code') if isinstance(row['error'], dict) else row['error']}", flush=True)
    if drop:
        print(f"parameters refused by {args.model} and dropped: {sorted(drop)}")


if __name__ == "__main__":
    asyncio.run(main())
