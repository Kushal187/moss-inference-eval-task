"""Experiment 2a: hallucination profile of each generator, using the FinQA project's labels.

Labels come from the project's own rules (scripts/build_detection_set.label_row): whether
gold evidence was in the 5 retrieved chunks x what the model did. Because every model saw
the *same* retrieved chunks, the "no evidence" rows are identical across models, and the
question is purely behavioural: when the page lacks the figures, does the model decline
or invent a number?

    python -m inference.hallucination
"""

from __future__ import annotations

import importlib.util
import json
from collections import Counter, defaultdict

from common.finqa import FINQA, SAVED_RUNS, load_qa
from common.moss_http import ROOT, read_jsonl

spec = importlib.util.spec_from_file_location("bds", FINQA / "scripts" / "build_detection_set.py")
bds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bds)

RUNS = {
    "llama-3.3-70b (baseline)": SAVED_RUNS / "prompts_dev_rerank.jsonl",
    "moss/fastlane-1": ROOT / "results/generation/moss__fastlane-1.jsonl",
    "anthropic/claude-haiku-5.5": ROOT / "results/generation/anthropic__claude-haiku-5.5.jsonl",
    "openai/gpt-6-luna": ROOT / "results/generation/openai__gpt-6-luna.jsonl",
}


def main() -> None:
    gold = {q["id"]: q["gold_evidence_ids"] for q in load_qa()}
    report = {}
    for name, path in RUNS.items():
        rows = [r for r in read_jsonl(path) if r.get("status", 200) == 200]
        by_strat = defaultdict(Counter)
        for r in rows:
            _sup, reason = bds.label_row(r, gold[r["id"]], 5)
            by_strat[r["strategy"]][reason] += 1
            by_strat["ALL"][reason] += 1
        report[name] = {}
        for s, c in by_strat.items():
            no_ev = c["fabricated_without_evidence"] + c["abstention_no_evidence"]
            with_ev = c["correct"] + c["wrong_despite_evidence"] + c["abstention"]
            report[name][s] = {
                "n": sum(c.values()), **dict(c),
                "fabrication_rate_when_evidence_missing": round(c["fabricated_without_evidence"] / no_ev, 3) if no_ev else None,
                "wrong_rate_when_evidence_present": round(c["wrong_despite_evidence"] / with_ev, 3) if with_ev else None,
                "hallucinated_total": c["fabricated_without_evidence"] + c["wrong_despite_evidence"],
            }
    out = ROOT / "results" / "hallucination_profile.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"{'model':30s} {'strategy':11s} {'n':>4s} {'fab|no-ev':>10s} {'wrong|ev':>9s} {'halluc':>7s}")
    for name, d in report.items():
        for s in ["zero_shot", "few_shot", "cot", "structured", "ALL"]:
            if s in d:
                v = d[s]
                print(f"{name:30s} {s:11s} {v['n']:4d} {str(v['fabrication_rate_when_evidence_missing']):>10s} "
                      f"{str(v['wrong_rate_when_evidence_present']):>9s} {v['hallucinated_total']:7d}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
