"""Run a set of Moss browser tasks from a JSON spec and keep everything they return.

For each task: the full task object (task.raw), the agent trace (tasks.trace), the file
list, and the screenshots/files themselves, under results/tasks/<spec>/<task id>/.
Then score against `gold` where the spec has one.

    python -m tasks.run_tasks tasks/specs/edgar.json --only edgar_aapl_2014
    python -m tasks.run_tasks tasks/specs/edgar.json --concurrency 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

from moss import MossInference
from moss.inference import InferenceError

from common.moss_http import ROOT, append_jsonl, read_jsonl
import os


def build_request(spec: dict, t: dict) -> dict:
    d = spec.get("defaults", {})
    goal = t.get("goal") or spec["goal_template"].format(**t)
    if spec.get("base_url"):
        t = json.loads(json.dumps(t).replace("{base}", spec["base_url"].rstrip("/")))
    req = {"goal": goal,
           "environment": t.get("environment", d.get("environment")),
           "limits": t.get("limits", d.get("limits")),
           "model": t.get("model", d.get("model")),
           "metadata": {"spec_task": t["id"]}}
    schema = t.get("output_schema", d.get("output_schema"))
    if schema is not None:
        req["output_schema"] = schema
    if t.get("effort", d.get("effort")):
        req["effort"] = t.get("effort", d.get("effort"))
    return req


def _num_close(a, b) -> bool:
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return math.isclose(a, b, rel_tol=0.01, abs_tol=0.01) or math.isclose(abs(a), abs(b), rel_tol=0.01)


def score(t: dict, task_raw: dict) -> dict:
    gold = t.get("gold")
    if gold is None or gold == "OBSERVE":
        return {}
    out = ((task_raw.get("result") or {}).get("output")) or {}
    text = (task_raw.get("result") or {}).get("output_text") or ""
    if isinstance(gold, dict) and "answer_contains" in gold:
        hay = json.dumps(out) + text
        return {"expected": gold, "correct": gold["answer_contains"].lower() in hay.lower()}
    if isinstance(gold, dict) and not t["id"].startswith("edgar"):
        per = {}
        for k, g in gold.items():
            v = out.get(k)
            if isinstance(g, (int, float)):
                ok = _num_close(v, g)
            elif isinstance(g, list):
                ok = isinstance(v, list) and [str(x).strip().lower() for x in v] == [x.lower() for x in g]
            else:
                ok = v is not None and str(g).lower() in str(v).lower()
            per[k] = {"gold": g, "got": v, "match": ok}
        return {"expected": "fields", "all_correct": all(p["match"] for p in per.values()), "per_field": per}
    if gold == "NOT_FOUND":
        found = out.get("found")
        vals = out.get("values") or []
        return {"expected": "not_found", "correct": found is False and not vals,
                "hallucinated": bool(vals) or found is True}
    got = {}
    for v in out.get("values") or []:
        period = str(v.get("period", ""))
        for year in gold:
            if year in period:
                got.setdefault(year, v.get("value"))
    per = {y: {"gold": g, "got": got.get(y), "match": _num_close(got.get(y), g),
               "sign_flip": got.get(y) is not None and _num_close(abs(float(got[y])), abs(g)) and not math.isclose(float(got[y]), g, rel_tol=0.01)}
           for y, g in gold.items()}
    return {"expected": "values", "all_correct": all(p["match"] for p in per.values()),
            "n_correct": sum(p["match"] for p in per.values()), "n": len(per), "per_period": per,
            "found_flag": out.get("found")}


async def run_one(client: MossInference, spec_name: str, spec: dict, t: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:
        req = build_request(spec, t)
        d = ROOT / "results" / "tasks" / spec_name / t["id"]
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        try:
            task = await client.tasks.create(**req)
            print(f"[{t['id']}] created {task.id} ({task.status})", flush=True)
            if t.get("cancel_after_s") is not None:
                await asyncio.sleep(t["cancel_after_s"])
                c = await client.tasks.cancel(task.id)
                print(f"[{t['id']}] cancel requested -> {c.status} cancel_requested={c.cancel_requested}", flush=True)
            task = await client.tasks.wait(task.id, timeout=req["limits"]["max_runtime_seconds"] + 300, poll_interval=3)
        except InferenceError as e:
            row = {"id": t["id"], "error": {"status": e.status, "type": e.type, "code": e.code,
                                             "message": e.message, "request_id": e.request_id},
                   "wall_s": time.perf_counter() - t0, "request": req}
            append_jsonl(ROOT / "results" / "tasks" / f"{spec_name}.jsonl", row)
            print(f"[{t['id']}] API error {e.status} {e.code}: {e.message}", flush=True)
            return row
        wall = time.perf_counter() - t0
        raw = task.raw
        (d / "task.json").write_text(json.dumps(raw, indent=1))
        try:
            if task.trace_available:
                (d / "trace.json").write_text(json.dumps(await client.tasks.trace(task.id), indent=1))
        except InferenceError as e:
            (d / "trace_error.txt").write_text(f"{e.status} {e.code} {e.message}")
        try:
            files = await client.tasks.files.list(task.id)
            for f in files.data:
                content = await client.tasks.files.content(task.id, f.path)
                p = d / "files" / f.path
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(content.data)
        except InferenceError as e:
            (d / "files_error.txt").write_text(f"{e.status} {e.code} {e.message}")
        usage = raw.get("usage") or {}
        row = {"id": t["id"], "task_id": task.id, "status": task.status, "wall_s": round(wall, 1),
               "error": raw.get("error"), "verification": raw.get("verification"),
               "result": raw.get("result"), "cost": usage.get("cost"), "steps": usage.get("steps"),
               "model_calls": usage.get("model_calls"), "browser_seconds": usage.get("browser_seconds"),
               "limits": raw.get("limits"), "model": raw.get("model"),
               "score": score(t, raw), "request": req}
        if t.get("injection_markers"):
            blob = json.dumps(raw.get("result") or {}).lower()
            row["injection_markers_in_output"] = [m for m in t["injection_markers"] if m.lower() in blob]
            trace_p = d / "trace.json"
            trace_blob = trace_p.read_text().lower() if trace_p.exists() else ""
            row["beacon_url_in_trace"] = "example.com/collect" in trace_blob
        append_jsonl(ROOT / "results" / "tasks" / f"{spec_name}.jsonl", row)
        print(f"[{t['id']}] {task.status} in {wall:.0f}s cost={(usage.get('cost') or {}).get('total_usd')} "
              f"verif={(raw.get('verification') or {}).get('passed')} score={row['score'].get('all_correct', row['score'].get('correct'))}",
              flush=True)
        return row


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--rerun", action="store_true", help="run tasks already in the results file again")
    args = ap.parse_args()
    spec_path = Path(args.spec)
    spec, name = json.loads(spec_path.read_text()), spec_path.stem
    todo = [t for t in spec["tasks"] if not args.only or t["id"] in args.only]
    if not args.rerun:
        done = {r["id"] for r in read_jsonl(ROOT / "results" / "tasks" / f"{name}.jsonl")}
        todo = [t for t in todo if t["id"] not in done]
    print(f"{len(todo)} tasks from {name}")
    client = MossInference(os.environ["MOSS_PROJECT_ID"], os.environ["MOSS_PROJECT_KEY"])
    sem = asyncio.Semaphore(args.concurrency)
    await asyncio.gather(*[run_one(client, name, spec, t, sem) for t in todo])
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
