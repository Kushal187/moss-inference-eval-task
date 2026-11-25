"""Thin async REST client for Moss Inference, built for measurement.

The Python SDK's Response object does not expose response headers (X-Moss-Retrieve-Ms,
rate-limit headers), so benchmarks call the REST API directly and time the stream
themselves: TTFT = request sent -> first output_text delta.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

BASE_URL = "https://inference.moss.dev/v1"
KEEP_HEADERS = ("x-moss-", "x-ratelimit-", "retry-after")


def auth_headers(project_id: str | None = None, project_key: str | None = None) -> dict:
    pid = project_id if project_id is not None else os.environ["MOSS_PROJECT_ID"]
    key = project_key if project_key is not None else os.environ["MOSS_PROJECT_KEY"]
    return {"Authorization": f"Bearer {pid}:{key}"}


def _kept(headers: httpx.Headers) -> dict:
    return {k: v for k, v in headers.items() if k.lower().startswith(KEEP_HEADERS)}


class RateLimiter:
    """Simple async pacer: at most `rate` request starts per second."""

    def __init__(self, rate: float):
        self.interval = 1.0 / rate if rate > 0 else 0.0
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if self._next > now:
                await asyncio.sleep(self._next - now)
            self._next = max(now, self._next) + self.interval


def _output_text(resp: dict) -> str:
    parts = []
    for item in resp.get("output") or []:
        if item.get("type") == "message":
            for c in item.get("content") or []:
                if c.get("type") == "output_text":
                    parts.append(c.get("text", ""))
    return "".join(parts)


async def respond(
    client: httpx.AsyncClient,
    body: dict,
    *,
    headers: dict | None = None,
    stream: bool = True,
    max_retries: int = 4,
    limiter: RateLimiter | None = None,
    timeout: float = 120.0,
) -> dict:
    """POST /v1/responses and return a measurement record.

    Keys: status, text, ttft_ms (stream only), total_ms, usage, headers, request_id,
    error, output (raw output items), retries, events (stream event-type counts).
    Retries only on 429/5xx before any byte of the body arrives.
    """
    hdrs = {**auth_headers(), **(headers or {})}
    payload = {**body, "stream": stream}
    retries = 0
    while True:
        if limiter:
            await limiter.wait()
        t0 = time.perf_counter()
        rec: dict = {"retries": retries}
        try:
            if not stream:
                r = await client.post(f"{BASE_URL}/responses", json=payload, headers=hdrs, timeout=timeout)
                rec["total_ms"] = (time.perf_counter() - t0) * 1000
                rec["status"] = r.status_code
                rec["headers"] = _kept(r.headers)
                rec["request_id"] = r.headers.get("x-moss-request-id")
                try:
                    j = r.json()
                except ValueError:
                    j = {"_raw": r.text[:2000]}
                if r.status_code >= 400:
                    rec["error"] = j.get("error", j)
                else:
                    rec.update(text=_output_text(j), usage=j.get("usage"), output=j.get("output"),
                               response_status=j.get("status"), error=j.get("error"))
            else:
                async with client.stream("POST", f"{BASE_URL}/responses", json=payload,
                                         headers=hdrs, timeout=timeout) as r:
                    rec["status"] = r.status_code
                    rec["headers"] = _kept(r.headers)
                    rec["request_id"] = r.headers.get("x-moss-request-id")
                    if r.status_code >= 400:
                        raw = (await r.aread()).decode(errors="replace")
                        rec["total_ms"] = (time.perf_counter() - t0) * 1000
                        try:
                            rec["error"] = json.loads(raw).get("error")
                        except ValueError:
                            rec["error"] = {"_raw": raw[:2000]}
                    else:
                        deltas, events, final = [], {}, None
                        async for line in r.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if not data or data == "[DONE]":
                                continue
                            try:
                                ev = json.loads(data)
                            except ValueError:
                                continue
                            et = ev.get("type", "?")
                            events[et] = events.get(et, 0) + 1
                            if et == "response.output_text.delta":
                                if "ttft_ms" not in rec:
                                    rec["ttft_ms"] = (time.perf_counter() - t0) * 1000
                                deltas.append(ev.get("delta", ""))
                            elif et in ("response.completed", "response.incomplete", "response.failed"):
                                final = ev.get("response") or {}
                        rec["total_ms"] = (time.perf_counter() - t0) * 1000
                        rec["events"] = events
                        final = final or {}
                        rec["text"] = "".join(deltas) or _output_text(final)
                        rec["usage"] = final.get("usage")
                        rec["output"] = final.get("output")
                        rec["response_status"] = final.get("status")
                        rec["error"] = final.get("error")
        except (httpx.TransportError, httpx.TimeoutException) as e:
            rec.update(status=0, error={"type": "transport", "message": repr(e)},
                       total_ms=(time.perf_counter() - t0) * 1000)

        st = rec.get("status", 0)
        if (st == 429 or st >= 500 or st == 0) and retries < max_retries:
            retries += 1
            ra = rec.get("headers", {}).get("retry-after")
            try:
                delay = float(ra) if ra else 2 ** retries
            except ValueError:
                delay = 2 ** retries
            await asyncio.sleep(min(delay, 30))
            continue
        return rec


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
