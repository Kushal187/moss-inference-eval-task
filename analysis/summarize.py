"""Build results/SUMMARY.md from the raw JSONL logs. Every number in the README and the
one-pager comes from here.

    python -m analysis.summarize
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from common.finqa import SAVED_RUNS
from common.moss_http import ROOT, read_jsonl

R = ROOT / "results"
lines: list[str] = []


def p(q, xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return round(xs[min(len(xs) - 1, int(q * len(xs)))])


def table(head, rows):
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "---|" * len(head))
    for r in rows:
        lines.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    lines.append("")


def pct(n, d):
    return f"{100 * n / d:.1f}%" if d else ""


def generation():
    lines.append("## 1. Generator swap (150 FinQA dev questions x 4 prompts, identical evidence)\n")
    lines.append("Baseline latency is non-streamed end-to-end from the original project (Groq/Bedrock); "
                 "Moss rows are streamed, TTFT measured client-side.\n")
    runs = {"llama-3.3-70b (baseline)": SAVED_RUNS / "prompts_dev_rerank.jsonl"}
    for f in sorted((R / "generation").glob("*.jsonl")):
        runs[f.stem.replace("__", "/")] = f
    rows = []
    for name, f in runs.items():
        data = read_jsonl(f)
        by = defaultdict(list)
        for r in data:
            by[r["strategy"]].append(r)
        for s in ("zero_shot", "few_shot", "cot", "structured"):
            rs = by.get(s)
            if not rs:
                continue
            ok = [r for r in rs if r.get("status", 200) == 200]
            lat = [r.get("total_ms", r.get("latency_ms")) for r in ok]
            rows.append([name, s, len(rs), len(rs) - len(ok), pct(sum(r["correct"] for r in rs), len(rs)),
                         pct(sum(not r["abstained"] for r in ok), len(ok)),
                         p(.5, [r.get("ttft_ms") for r in ok]), p(.95, [r.get("ttft_ms") for r in ok]),
                         p(.5, lat), p(.95, lat)])
    table(["model", "prompt", "n", "errors", "accuracy", "answered", "TTFT p50", "TTFT p95", "total p50", "total p95"], rows)


def hallucination():
    f = R / "hallucination_profile.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    lines.append("## 2. Hallucination profile (FinQA project's labels; same retrieved evidence for every model)\n")
    lines.append("`fab|no-evidence` = share of questions whose gold evidence was NOT retrieved where the model "
                 "still produced a (wrong) number instead of declining. `wrong|evidence` = wrong answer although "
                 "the needed figures were in front of it.\n")
    rows = []
    for m, strat in d.items():
        for s in ("zero_shot", "few_shot", "cot", "structured", "ALL"):
            if s in strat:
                v = strat[s]
                rows.append([m, s, v["n"], v["fabrication_rate_when_evidence_missing"],
                             v["wrong_rate_when_evidence_present"], v["hallucinated_total"]])
    table(["model", "prompt", "n", "fab|no-evidence", "wrong|evidence", "hallucinated"], rows)


def judge():
    files = sorted((R / "judge").glob("summary__*.json"))
    if not files:
        return
    lines.append("## 2b. LLM-judge re-scoring (FinQA project's judge prompts, called through Moss)\n")
    lines.append("A = any answer not supported by its evidence (arithmetic errors + fabrications). "
                 "B = answered although the evidence lacked the figures. `flag-all F1` is the baseline of "
                 "flagging every answer. Saved results from the original project (judging llama-3.3-70b "
                 "answers, Bedrock): A grounding: llama-judge F1 0.41, claude-haiku-4.5 F1 0.54 (P 0.61 / R 0.49); "
                 "B: grounding F1 0.35, evidence F1 0.41.\n")
    rows = []
    for f in files:
        judge_name = f.stem.replace("summary__", "").replace("__", "/")
        for gen, d in json.loads(f.read_text()).items():
            for k in ("A_grounding", "B_grounding", "B_evidence"):
                v = d[k]
                rows.append([judge_name, gen, k, v["n"], v["positives"], v["precision"], v["recall"], v["f1"],
                             v["flag_all_f1"], v["accuracy"]])
            rows.append([judge_name, gen, "grounding recall by kind", "", "", "", "",
                         json.dumps(d["grounding_recall_by_kind"]), "", ""])
    table(["judge", "answers from", "definition / mode", "n", "positives", "precision", "recall", "F1",
           "flag-all F1", "accuracy"], rows)


def retrieval():
    rows = [["Weaviate BM25 (saved)", "43.4", "", "77.1", "", "0 (in-process)", ""],
            ["Weaviate dense (saved)", "47.7", "", "79.8", "", "58 (mean)", ""],
            ["Weaviate hybrid a=0.6 (saved)", "50.7", "", "82.1", "", "53 (mean)", ""],
            ["hybrid + cross-encoder rerank (saved)", "52.6", "", "84.3", "", "112 (mean)", ""]]
    seen = {}
    for r in read_jsonl(R / "retrieval" / "summary.jsonl"):
        seen[r["run"]] = r
    for run, r in seen.items():
        if r["n"] < 1000:
            continue
        rows.append([f"Moss local ({run})", r["r@1"], r["r@3"], r["r@5"], r["r@10"],
                     f"{r['wall_ms_p50']} (p50) / {r['wall_ms_p95']} (p95)", r["wrong_doc_top5_mean"]])
    lines.append("## 3. Retrieval quality, FinQA test split (n=1147)\n")
    table(["retriever", "R@1", "R@3", "R@5", "R@10", "latency ms", "wrong-filing chunks in top 5"], rows)


def server_rag():
    lines.append("## 4. Moss server-side retrieval on the 150 dev questions (fastlane-1)\n")
    lines.append("`closed` = no evidence at all. `premodel` = X-Moss-Indexes (no filter possible, retrieved "
                 "chunks not returned). `tool` = moss_search tool (passages visible). `_ctx` = index with "
                 "ticker/year prefixes + question names the filing. Baseline for reference: the same model "
                 "given the project's rerank top-5 (zero-shot) = generator-swap table.\n")
    rows = []
    for f in sorted((R / "server_rag").glob("*.jsonl")):
        rs = [r for r in read_jsonl(f)]
        ok = [r for r in rs if r["status"] == 200]
        if not rs:
            continue
        rec = [r["gold_recall"] for r in ok if r.get("gold_recall") is not None]
        rows.append([f.stem.replace("_moss__fastlane-1", ""), len(rs), len(rs) - len(ok),
                     pct(sum(r["correct"] for r in rs), len(rs)), pct(sum(r["abstained"] for r in ok), len(ok)),
                     p(.5, [r.get("ttft_ms") for r in ok]), p(.5, [r.get("total_ms") for r in ok]),
                     p(.5, [float(r["retrieve_ms"]) for r in ok if r.get("retrieve_ms")]),
                     round(st.mean(r["input_tokens"] for r in ok if r.get("input_tokens")), 0) if ok else None,
                     round(st.mean(r["n_search_calls"] for r in ok), 2) if ok else None,
                     f"{100 * st.mean(rec):.1f}%" if rec else "n/a (not exposed)",
                     sum(1 for r in ok if r.get("search_errors"))])
    table(["mode", "n", "errors", "accuracy", "declined", "TTFT p50", "total p50", "X-Moss-Retrieve-Ms p50",
           "mean input tokens", "searches/q", "gold evidence retrieved", "search errors"], rows)


def probes():
    lines.append("## 5. API probes\n")
    errs = read_jsonl(R / "probes" / "errors.jsonl")
    if errs:
        lines.append("### Error handling\n")
        table(["case", "status", "type / code", "message"],
              [[e["case"], e["status"],
                (f"{(e['error'] or {}).get('type')} / {(e['error'] or {}).get('code')}" if isinstance(e.get("error"), dict) else ""),
                (e["error"].get("message") if isinstance(e.get("error"), dict) else str(e.get("error")))[:140]]
               for e in errs])
    fu = read_jsonl(R / "probes" / "followups.jsonl")
    if fu:
        lines.append("### Follow-ups\n")
        for r in fu:
            lines.append(f"- `{r['probe']}`: " + json.dumps({k: v for k, v in r.items() if k != "probe"})[:400])
        lines.append("")
    for name in ("overhead", "cache", "burst", "context"):
        rs = read_jsonl(R / "probes" / f"{name}.jsonl")
        if not rs:
            continue
        lines.append(f"### {name}\n")
        if name == "overhead":
            table(["model", "words sent", "input_tokens billed"],
                  [[r["model"], r["input_words"], (r.get("usage") or {}).get("input_tokens")] for r in rs])
        elif name == "cache":
            table(["probe", "cache", "#", "TTFT", "total", "input_tokens", "cached_tokens", "output_tokens"],
                  [[r["probe"], r.get("cache", ""), r["i"], r.get("ttft_ms") and round(r["ttft_ms"]), round(r["total_ms"]),
                    (r.get("usage") or {}).get("input_tokens"),
                    ((r.get("usage") or {}).get("input_tokens_details") or {}).get("cached_tokens"),
                    (r.get("usage") or {}).get("output_tokens")] for r in rs])
        elif name == "burst":
            table(["model", "n", "wall s", "statuses", "Retry-After", "error codes", "TTFT p50", "TTFT p95", "total p50", "total p95"],
                  [[r["model"], r["n"], r["wall_s"], r["status_counts"], r["retry_after"], r["err_codes"],
                    r["ttft_p50"], r["ttft_p95"], r["total_p50"], r["total_p95"]] for r in rs])
        elif name == "context":
            by = defaultdict(list)
            for r in rs:
                by[r["target_tokens"]].append(r)
            table(["~prompt tokens", "status", "input_tokens", "cached (reps)", "TTFT reps (ms)", "total reps (ms)"],
                  [[k, sorted({r["status"] for r in v}), (v[0].get("usage") or {}).get("input_tokens"),
                    [((r.get("usage") or {}).get("input_tokens_details") or {}).get("cached_tokens") for r in v],
                    [r.get("ttft_ms") and round(r["ttft_ms"]) for r in v], [round(r["total_ms"]) for r in v]]
                   for k, v in sorted(by.items())])


def tasks():
    lines.append("## 6. Tasks API\n")
    total = 0.0
    for spec in ("edgar", "traps", "injection", "lifecycle"):
        rs = read_jsonl(R / "tasks" / f"{spec}.jsonl")
        if not rs:
            continue
        lines.append(f"### {spec}\n")
        rows = []
        for r in rs:
            c = (r.get("cost") or {}).get("total_usd") or 0
            total += c
            sc = r.get("score") or {}
            verdict = sc.get("all_correct", sc.get("correct", "observe"))
            out = (r.get("result") or {}).get("output")
            out_s = json.dumps(out) if out is not None else ((r.get("result") or {}).get("output_text") or "")
            rows.append([r["id"], r.get("status"), (r.get("error") or {}).get("code") if r.get("error") else "",
                         (r.get("verification") or {}).get("passed"), verdict, r.get("wall_s"), r.get("steps"),
                         f"{c:.3f}", out_s[:160].replace("|", "/")])
        table(["task", "status", "error", "verification.passed", "correct", "wall s", "steps", "cost $", "output"], rows)
    lines.append(f"**Total task spend: ${total:.2f}**\n")


PRICES = {  # USD per 1M tokens: input, cached input, output (docs.moss.dev/docs/inference/models, 2026-11-05)
    "moss/fastlane-1": (0.25, 0.03, 1.75),
    "anthropic/claude-haiku-5.5": (0.10, 0.01, 0.50),
    "openai/gpt-6-luna": (0.10, 0.01, 0.50),
    "google/gemini-3.5-flash-lite": (0.30, 0.03, 2.50),
}


def spend():
    """Estimate inference spend from logged usage x list prices (retrievals at $0.50/1k)."""
    by = defaultdict(lambda: [0, 0.0])
    files = list((R / "generation").glob("*.jsonl")) + list((R / "server_rag").glob("*.jsonl")) + \
        list((R / "probes").glob("*.jsonl")) + list((R / "judge").glob("*.jsonl"))
    for f in files:
        for r in read_jsonl(f):
            u = r.get("usage") or {}
            model = r.get("model") or "moss/fastlane-1"
            if f.parent.name == "judge":
                model = "anthropic/claude-haiku-5.5" if "claude-haiku" in f.name else "moss/fastlane-1"
            if not u or model not in PRICES:
                continue
            pin, pc, pout = PRICES[model]
            cached = (u.get("input_tokens_details") or {}).get("cached_tokens") or 0
            cost = ((u.get("input_tokens", 0) - cached) * pin + cached * pc + u.get("output_tokens", 0) * pout) / 1e6
            if r.get("mode") == "premodel":
                cost += 0.5 / 1000
            cost += 0.5 / 1000 * (r.get("n_search_calls") or 0)
            by[f"{f.parent.name}: {model}"][0] += 1
            by[f"{f.parent.name}: {model}"][1] += cost
    lines.append("## 7. Estimated inference spend (logged usage x list price)\n")
    table(["experiment: model", "calls", "est. $"], [[k, v[0], f"{v[1]:.3f}"] for k, v in sorted(by.items())])
    lines.append(f"**Estimated inference total: ${sum(v[1] for v in by.values()):.2f}** "
                 "(excludes index ingest at $0.03/MB, ~3.2 MB x 2 indexes, and local queries)\n")


def main() -> None:
    lines.append("# Results summary (generated by analysis/summarize.py)\n")
    generation(); hallucination(); judge(); retrieval(); server_rag(); probes(); tasks(); spend()
    out = R / "SUMMARY.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
