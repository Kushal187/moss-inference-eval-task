"""Experiment 3b: Moss's integrated, server-side retrieval on the same 150 dev questions.

Modes (all fastlane-1 unless --model):
  closed      no evidence, no index: what the model knows on its own (control)
  premodel    X-Moss-Indexes header: Moss searches before the model runs. No filter
              exists and retrieved chunks are not returned, so recall can't be measured;
              only accuracy, X-Moss-Retrieve-Ms, and the input-token delta.
  tool        moss_search tool: the model issues the search, and the mcp_call output item
              exposes the passages, so we CAN check whether gold evidence came back.

--ctx uses the finqa-ctx index and prefixes the question with "<TICKER> <YEAR> 10-K:"
(the only way to steer an unfiltered search toward the right filing).

    python -m inference.server_rag --mode premodel
    python -m inference.server_rag --mode tool --ctx
"""

from __future__ import annotations

import argparse
import asyncio
import re
from pathlib import Path

import httpx

from common.finqa import SAVED_RUNS, answers_match, extract_final_answer, get_chunk_contents, normalize_answer
from common.moss_http import ROOT, RateLimiter, append_jsonl, read_jsonl, respond
from inference.build_index import ctx_prefix

SYSTEM = (
    "You are a financial question-answering assistant. Answer the user's question using ONLY "
    "the evidence you are given or retrieve. Do not use outside knowledge. If the evidence does "
    "not contain the answer, respond exactly: 'I cannot determine the answer from the provided "
    "evidence.' Otherwise give the answer as a single concise number or yes/no, matching the "
    "units shown in the evidence."
)
TOOL_HINT = " Always search the index first, using the question as your query."


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


async def run_one(client, sem, limiter, args, q, out):
    async with sem:
        question = q["question"]
        if args.ctx:
            question = ctx_prefix(q["doc_id"]).replace(", page", " (page").replace(": ", "): ") + question
        index = "finqa-ctx" if args.ctx else "finqa-raw"
        body = {"model": args.model, "max_output_tokens": 512,
                "input": [{"role": "system", "content": SYSTEM + (TOOL_HINT if args.mode == "tool" else "")},
                          {"role": "user", "content": f"Question: {question}\nAnswer:"}]}
        headers = {"X-Moss-Cache": "off"}
        if args.mode == "premodel":
            headers["X-Moss-Indexes"] = index
        elif args.mode == "tool":
            body["tools"] = [{"type": "moss_search", "indexes": [index], "top_k": 5}]
        rec = await respond(client, body, headers=headers, limiter=limiter, timeout=200)

        raw = rec.get("text") or ""
        answer = extract_final_answer(raw, "zero_shot") if raw else ""
        _v, kind = normalize_answer(answer)
        calls = [o for o in (rec.get("output") or []) if o.get("type") == "mcp_call"]
        passages = "\n".join(str(c.get("output") or "") for c in calls)
        gold = get_chunk_contents(q["doc_id"], q["gold_ids"])
        gold_found = [lid for lid, txt in gold if norm(txt)[:120] in norm(passages)] if calls else None
        row = {
            "id": q["id"], "doc_id": q["doc_id"], "mode": args.mode, "ctx": args.ctx, "model": args.model,
            "question_sent": question, "answer": answer, "answer_type": kind,
            "abstained": kind in ("abstention", "none"),
            "correct": answers_match(answer, q["gold_answer"], q["gold_answer_exe"]) if raw else False,
            "status": rec.get("status"), "error": rec.get("error"),
            "ttft_ms": rec.get("ttft_ms"), "total_ms": rec.get("total_ms"),
            "retrieve_ms": (rec.get("headers") or {}).get("x-moss-retrieve-ms"),
            "input_tokens": (rec.get("usage") or {}).get("input_tokens"),
            "usage": rec.get("usage"), "request_id": rec.get("request_id"),
            "n_search_calls": len(calls),
            "search_args": [c.get("arguments") for c in calls],
            "search_errors": [c.get("error") for c in calls if c.get("error")],
            "search_ms": [c.get("duration_ms") for c in calls],
            "gold_ids": q["gold_ids"], "gold_found": gold_found,
            "gold_recall": (len(gold_found) / len(gold)) if (calls and gold) else None,
            "passages": passages[:6000], "events": rec.get("events"), "raw": raw,
        }
        append_jsonl(out, row)
        return row


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["closed", "premodel", "tool"], required=True)
    ap.add_argument("--ctx", action="store_true")
    ap.add_argument("--model", default="moss/fastlane-1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--rps", type=float, default=1.5)
    args = ap.parse_args()

    from common.finqa import load_qa
    gold = {q["id"]: q for q in load_qa()}
    base = read_jsonl(SAVED_RUNS / "prompts_dev_rerank.jsonl")
    qids = list(dict.fromkeys(r["id"] for r in base))[: args.limit]
    qs = [{**gold[i], "gold_ids": gold[i]["gold_evidence_ids"]} for i in qids]

    tag = f"{args.mode}{'_ctx' if args.ctx else ''}_{args.model.replace('/', '__')}"
    out = ROOT / "results" / "server_rag" / f"{tag}.jsonl"
    done = {r["id"] for r in read_jsonl(out) if r.get("status") == 200}
    qs = [q for q in qs if q["id"] not in done]
    print(f"{tag}: {len(qs)} calls -> {out}")
    sem, limiter = asyncio.Semaphore(args.concurrency), RateLimiter(args.rps)
    async with httpx.AsyncClient() as client:
        rows = await asyncio.gather(*[run_one(client, sem, limiter, args, q, out) for q in qs])
    bad = [r for r in rows if r["status"] != 200]
    print(f"done; {len(bad)} non-200", [(r["status"], r["error"]) for r in bad[:3]])


if __name__ == "__main__":
    asyncio.run(main())
