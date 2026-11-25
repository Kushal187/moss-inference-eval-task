"""Experiment 4: API behaviour probes for Moss Inference.

  errors     a matrix of bad requests: status, error type/code, message, request id
  cache      reply cache (X-Moss-Cache) and prefix cache (cached_tokens) behaviour
  burst      N simultaneous requests: 429s, Retry-After, tail latency
  context    TTFT / total latency as prompt length grows (fastlane, 262k context)
  overhead   hidden prompt tokens added per request, per model

    python -m inference.api_probes errors cache burst context overhead
"""

from __future__ import annotations

import asyncio
import json
import statistics as st
import sys
import time

import httpx

from common.finqa import load_chunks
from common.moss_http import BASE_URL, ROOT, append_jsonl, auth_headers, respond

OUT = ROOT / "results" / "probes"
FAST = "moss/fastlane-1"


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    return round(xs[min(len(xs) - 1, int(p * len(xs)))], 1) if xs else None


def slim(rec: dict) -> dict:
    return {k: rec.get(k) for k in ("status", "error", "ttft_ms", "total_ms", "usage", "request_id",
                                    "headers", "retries", "response_status")} | {"text": (rec.get("text") or "")[:200]}


async def errors(c: httpx.AsyncClient) -> None:
    ok = {"model": FAST, "input": "Say hi."}
    big = "lorem ipsum dolor sit amet " * 230_000  # ~1.4M tokens, ~6 MB
    long_ok_size = "lorem ipsum dolor sit amet " * 60_000  # ~360k tokens, ~1.6 MB: over 262k ctx, under 4 MB
    cases = [
        ("bad_key", ok, {"Authorization": "Bearer not:valid"}, False),
        ("missing_auth_scheme", ok, {"Authorization": "garbage"}, False),
        ("unknown_model", {**ok, "model": "moss/fastlane-99"}, {}, False),
        ("vendor_name_without_prefix_no_byok", {**ok, "model": "gpt-6-luna"}, {}, False),
        ("unknown_top_level_field", {**ok, "frobnicate": True}, {}, False),
        ("previous_response_id", {**ok, "previous_response_id": "resp_123"}, {}, False),
        ("background_true", {**ok, "background": True}, {}, False),
        ("empty_input", {**ok, "input": ""}, {}, False),
        ("max_output_tokens_0", {**ok, "max_output_tokens": 0}, {}, False),
        ("nonexistent_index", ok, {"X-Moss-Indexes": "does-not-exist"}, False),
        ("33_indexes", ok, {"X-Moss-Indexes": ",".join(f"i{i}" for i in range(33))}, False),
        ("anthropic_temperature", {**ok, "model": "anthropic/claude-haiku-5.5", "temperature": 0}, {}, False),
        ("anthropic_json_object", {**ok, "model": "anthropic/claude-haiku-5.5", "text": {"format": {"type": "json_object"}}}, {}, False),
        ("openai_seed", {**ok, "model": "openai/gpt-6-luna", "seed": 1}, {}, False),
        ("context_over_262k_fastlane", {**ok, "input": long_ok_size}, {}, False),
        ("body_over_4mb", {**ok, "input": big}, {}, False),
        ("stream_unknown_model", {**ok, "model": "moss/fastlane-99"}, {}, True),
        ("stream_bad_key", ok, {"Authorization": "Bearer not:valid"}, True),
    ]
    for name, body, hdr, stream in cases:
        hdrs = {**auth_headers(), **hdr}
        t0 = time.perf_counter()
        try:
            if stream:
                rec = await respond(c, body, headers=hdr, stream=True, max_retries=0)
            else:
                r = await c.post(f"{BASE_URL}/responses", json={**body, "stream": False}, headers=hdrs, timeout=120)
                try:
                    j = r.json()
                except ValueError:
                    j = {"_raw": r.text[:500]}
                rec = {"status": r.status_code, "error": j.get("error") if isinstance(j, dict) else j,
                       "request_id": r.headers.get("x-moss-request-id"),
                       "retry_after": r.headers.get("retry-after"),
                       "text": "" if r.status_code >= 400 else str(j.get("output"))[:200],
                       "usage": j.get("usage") if isinstance(j, dict) else None}
        except Exception as e:  # noqa: BLE001
            rec = {"status": 0, "error": repr(e)}
        rec["ms"] = round((time.perf_counter() - t0) * 1000)
        print(f"{name:38s} {rec.get('status')} {json.dumps(rec.get('error'))[:150]}")
        append_jsonl(OUT / "errors.jsonl", {"case": name, "stream": stream, **{k: rec.get(k) for k in ("status", "error", "request_id", "retry_after", "ms", "usage", "text")}})


async def cache(c: httpx.AsyncClient) -> None:
    prompt = {"model": FAST, "input": "In one sentence, what is a 10-K filing?", "max_output_tokens": 60}
    for mode in ("on", "off"):
        for i in range(5):
            rec = await respond(c, prompt, headers={"X-Moss-Cache": mode}, stream=True)
            row = {"probe": "reply_cache", "cache": mode, "i": i, **slim(rec)}
            append_jsonl(OUT / "cache.jsonl", row)
            print(f"reply cache={mode} #{i}: ttft={rec.get('ttft_ms') and round(rec['ttft_ms'])} total={round(rec['total_ms'])} "
                  f"usage={rec.get('usage')} hdr={ {k: v for k, v in (rec.get('headers') or {}).items() if 'cache' in k} }")
    # prefix cache: long shared prefix, different suffix
    prefix = "\n".join(ch["content"] for ch in load_chunks()[:400])
    for i in range(4):
        body = {"model": FAST, "max_output_tokens": 20,
                "input": [{"role": "system", "content": "Answer in one word.\n" + prefix},
                          {"role": "user", "content": f"Variant {i}: is the word 'revenue' common in finance? yes/no"}]}
        rec = await respond(c, body, headers={"X-Moss-Cache": "off"}, stream=True)
        u = rec.get("usage") or {}
        append_jsonl(OUT / "cache.jsonl", {"probe": "prefix_cache", "i": i, **slim(rec)})
        print(f"prefix #{i}: input={u.get('input_tokens')} cached={(u.get('input_tokens_details') or {}).get('cached_tokens')} "
              f"ttft={rec.get('ttft_ms') and round(rec['ttft_ms'])}")


async def burst(c: httpx.AsyncClient, model: str = FAST, n: int = 30) -> None:
    body = {"model": model, "input": "Reply with the word ok.", "max_output_tokens": 16}
    t0 = time.perf_counter()
    recs = await asyncio.gather(*[respond(c, body, headers={"X-Moss-Cache": "off"}, stream=True, max_retries=0) for _ in range(n)])
    wall = time.perf_counter() - t0
    sts = [r["status"] for r in recs]
    ok = [r for r in recs if r["status"] == 200]
    summ = {"probe": "burst", "model": model, "n": n, "wall_s": round(wall, 2),
            "status_counts": {s: sts.count(s) for s in set(sts)},
            "retry_after": sorted({(r.get("headers") or {}).get("retry-after") for r in recs if r["status"] == 429} - {None}),
            "err_codes": sorted({json.dumps((r.get("error") or {}).get("code")) for r in recs if r["status"] != 200}),
            "ttft_p50": pct([r.get("ttft_ms") for r in ok], .5), "ttft_p95": pct([r.get("ttft_ms") for r in ok], .95),
            "total_p50": pct([r.get("total_ms") for r in ok], .5), "total_p95": pct([r.get("total_ms") for r in ok], .95)}
    append_jsonl(OUT / "burst.jsonl", summ)
    print(summ)


async def context(c: httpx.AsyncClient) -> None:
    chunks = [ch["content"] for ch in load_chunks()]
    text = "\n".join(chunks)
    for target_tokens in (500, 4_000, 16_000, 64_000, 200_000):
        filler = text[: target_tokens * 4]
        for rep in range(3):
            body = {"model": FAST, "max_output_tokens": 30,
                    "input": [{"role": "system", "content": "Reference material:\n" + filler + f"\n[run {rep}-{time.time()}]"},
                              {"role": "user", "content": "Reply with the single word: done"}]}
            rec = await respond(c, body, headers={"X-Moss-Cache": "off"}, stream=True, timeout=300)
            u = rec.get("usage") or {}
            append_jsonl(OUT / "context.jsonl", {"target_tokens": target_tokens, "rep": rep, **slim(rec)})
            print(f"~{target_tokens:>7} tok rep{rep}: status={rec['status']} input={u.get('input_tokens')} "
                  f"cached={(u.get('input_tokens_details') or {}).get('cached_tokens')} "
                  f"ttft={rec.get('ttft_ms') and round(rec['ttft_ms'])} total={round(rec['total_ms'])} err={rec.get('error')}")


async def overhead(c: httpx.AsyncClient) -> None:
    for model in (FAST, "anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "google/gemini-3.5-flash-lite"):
        for inp in ("hi", "hi " * 100):
            rec = await respond(c, {"model": model, "input": inp, "max_output_tokens": 16}, stream=False,
                                headers={"X-Moss-Cache": "off"})
            u = rec.get("usage") or {}
            append_jsonl(OUT / "overhead.jsonl", {"model": model, "input_words": len(inp.split()), **slim(rec)})
            print(f"{model:32s} words={len(inp.split()):4d} input_tokens={u.get('input_tokens')} status={rec['status']}")


async def main() -> None:
    which = sys.argv[1:] or ["errors", "cache", "burst", "context", "overhead"]
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=64)) as c:
        for w in which:
            print(f"== {w}")
            if w == "burst":
                await burst(c, FAST)
                await asyncio.sleep(5)
                await burst(c, "anthropic/claude-haiku-5.5")
            else:
                await globals()[w](c)


if __name__ == "__main__":
    asyncio.run(main())
