"""Experiment 2b: re-score every generator's answers with the FinQA project's LLM judge,
called through Moss.

Uses the project's own pieces unchanged: build_detection_set.build (labels + evidence),
llm_judge prompt builders/parsers, and the abstention short-circuit from verify().

  grounding mode  judge sees evidence + candidate answer, recomputes, says supported or not
  evidence mode   judge sees evidence + question only (never the answer): is it sufficient?
                  Depends only on (question, evidence), which is identical across generators
                  and strategies, so it is cached per question id.

Scored two ways, as in the project report:
  A  hallucination = label is not supported (arithmetic errors + fabrications)
  B  hallucination = fabricated_without_evidence only

    python -m inference.judge_rescore --judge anthropic/claude-haiku-5.5
    python -m inference.judge_rescore --judge moss/fastlane-1 --generators llama
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
from collections import defaultdict

import httpx

from common.finqa import FINQA, SAVED_RUNS, llm_judge
from common.moss_http import ROOT, RateLimiter, append_jsonl, read_jsonl, respond

spec = importlib.util.spec_from_file_location("bds", FINQA / "scripts" / "build_detection_set.py")
bds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bds)

GEN = ROOT / "results" / "generation"
GENERATORS = {
    "llama": [SAVED_RUNS / "prompts_dev_rerank.jsonl"],
    "fastlane": [GEN / "moss__fastlane-1.jsonl"],
    "haiku": [GEN / "anthropic__claude-haiku-5.5.jsonl"],
    # luna's JSON-mode structured calls all 502'd; use the no-JSON structured rerun instead
    "luna": [GEN / "openai__gpt-6-luna.jsonl", GEN / "openai__gpt-6-luna__structured_nojson.jsonl"],
}
OUT = ROOT / "results" / "judge"


def ok_rows_file(g: str):
    """Successful generations only (failed API calls aren't model answers), as one file."""
    rows = [r for p in GENERATORS[g] for r in read_jsonl(p) if r.get("status", 200) == 200]
    p = ROOT / "results" / "tmp" / f"gen_ok_{g}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def judge_body(judge: str, messages: list[dict]) -> dict:
    body = {"model": judge, "input": messages, "max_output_tokens": 600}
    if not judge.startswith("anthropic/"):          # Moss refuses both for Anthropic models
        body["temperature"] = 0.0
        body["text"] = {"format": {"type": "json_object"}}
    return body


async def call_judge(client, limiter, judge, messages, sem):
    async with sem:
        rec = await respond(client, judge_body(judge, messages), limiter=limiter, stream=False,
                            headers={"X-Moss-Cache": "off"})
        return rec


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="anthropic/claude-haiku-5.5")
    ap.add_argument("--generators", nargs="+", default=list(GENERATORS))
    ap.add_argument("--modes", nargs="+", default=["grounding", "evidence"])
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--rps", type=float, default=3.0)
    args = ap.parse_args()
    jtag = args.judge.replace("/", "__")
    sem, limiter = asyncio.Semaphore(args.concurrency), RateLimiter(args.rps)

    async with httpx.AsyncClient() as client:
        # ---- evidence mode: one call per question id, shared by all generators ----
        ev_path = OUT / f"evidence_cache__{jtag}.jsonl"
        ev_cache = {r["id"]: r for r in read_jsonl(ev_path) if r.get("status") == 200}
        all_rows = {g: bds.build([ok_rows_file(g)], 5)[0] for g in args.generators}
        if "evidence" in args.modes:
            need = {}
            for rows in all_rows.values():
                for r in rows:
                    if r["id"] not in ev_cache and not llm_judge.is_abstention(r["answer"]):
                        need.setdefault(r["id"], r)
            print(f"evidence mode: {len(need)} new question-level calls")

            async def ev_one(r):
                msgs = llm_judge.build_evidence_messages(r["question"], [tuple(c) for c in r["context"]])
                rec = await call_judge(client, limiter, args.judge, msgs, sem)
                row = {"id": r["id"], "status": rec["status"], "raw": rec.get("text"), "error": rec.get("error"),
                       "usage": rec.get("usage"), "ms": rec.get("total_ms")}
                append_jsonl(ev_path, row)
                if rec["status"] == 200:
                    ev_cache[r["id"]] = row
            await asyncio.gather(*[ev_one(r) for r in need.values()])

        # ---- grounding mode: one call per (generator, question, strategy) ----
        for g, rows in all_rows.items():
            out = OUT / f"{g}__judged_by__{jtag}.jsonl"
            done = {(r["id"], r["strategy"]) for r in read_jsonl(out) if r.get("status") == 200}
            todo = [r for r in rows if (r["id"], r["strategy"]) not in done]
            print(f"{g}: {len(rows)} labeled rows, {len(todo)} to judge")

            async def gr_one(r):
                ctx = [tuple(c) for c in r["context"]]
                res = {**{k: r[k] for k in ("id", "strategy", "question", "answer", "gold_supported",
                                             "label_reason", "gold_answer", "gold_answer_exe")}}
                if llm_judge.is_abstention(r["answer"]):
                    v = {"supported": True, "category": "abstention", "short_circuit": True}
                    res.update(status=200, grounding=v, evidence=v)
                    append_jsonl(out, res)
                    return
                if "grounding" in args.modes:
                    rec = await call_judge(client, limiter, args.judge,
                                           llm_judge.build_judge_messages(r["question"], ctx, r["answer"]), sem)
                    res.update(status=rec["status"], error=rec.get("error"), usage=rec.get("usage"),
                               ms=rec.get("total_ms"), raw=rec.get("text"),
                               grounding=llm_judge.parse_verdict(rec.get("text")) if rec["status"] == 200 else None)
                else:
                    res["status"] = 200
                ev = ev_cache.get(r["id"])
                res["evidence"] = llm_judge.parse_evidence_verdict(ev["raw"]) if ev else None
                append_jsonl(out, res)
            await asyncio.gather(*[gr_one(r) for r in todo])

    summarize(args.generators, jtag)


def prf(rows, gold_fn, mode):
    tp = fp = fn = tn = 0
    for r in rows:
        v = r.get(mode)
        if not v:
            continue
        gold_h, pred_h = gold_fn(r), not v["supported"]
        tp += gold_h and pred_h; fp += (not gold_h) and pred_h; fn += gold_h and not pred_h; tn += (not gold_h) and not pred_h
    n = tp + fp + fn + tn
    P = tp / (tp + fp) if tp + fp else 0.0
    Rc = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * Rc / (P + Rc) if P + Rc else 0.0
    base = (tp + fn) / n if n else 0.0
    return {"n": n, "positives": tp + fn, "precision": round(P, 2), "recall": round(Rc, 2), "f1": round(F, 2),
            "accuracy": round((tp + tn) / n, 2) if n else None, "flag_all_f1": round(2 * base / (1 + base), 2) if n else None}


def summarize(generators, jtag):
    res = {}
    for g in generators:
        rows = [r for r in read_jsonl(OUT / f"{g}__judged_by__{jtag}.jsonl") if r.get("status") == 200]
        res[g] = {
            "A_grounding": prf(rows, lambda r: not r["gold_supported"], "grounding"),
            "B_grounding": prf(rows, lambda r: r["label_reason"] == "fabricated_without_evidence", "grounding"),
            "B_evidence": prf(rows, lambda r: r["label_reason"] == "fabricated_without_evidence", "evidence"),
        }
        recall_by = defaultdict(lambda: [0, 0])
        for r in rows:
            if r.get("grounding") and not r["gold_supported"]:
                recall_by[r["label_reason"]][1] += 1
                recall_by[r["label_reason"]][0] += not r["grounding"]["supported"]
        res[g]["grounding_recall_by_kind"] = {k: f"{a}/{b}" for k, a in recall_by.items() for a, b in [tuple(a)]}
    (OUT / f"summary__{jtag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
