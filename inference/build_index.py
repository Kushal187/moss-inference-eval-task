"""Load the FinQA dev+test corpus into Moss indexes.

Two indexes (the free plan allows 3):
  finqa-raw  chunk text exactly as the Weaviate baseline embedded it
  finqa-ctx  same chunks prefixed with "<TICKER> <YEAR> 10-K, page N:" so that an
             unfiltered server-side search has a chance of finding the right filing

Every document carries doc_id / local_id metadata so the local SDK path can filter
to one filing page, matching the baseline's per-page retrieval.

    python -m inference.build_index
"""

from __future__ import annotations

import asyncio
import json
import time

from moss import DocumentInfo, MossClient

from common.finqa import load_chunks, load_qa
from common.moss_http import ROOT, append_jsonl
import os


def ctx_prefix(doc_id: str) -> str:
    ticker, year, page = doc_id.split("/")[:3]
    page = page.replace("page_", "").replace(".pdf", "")
    return f"{ticker} {year} 10-K, page {page}: "


async def main() -> None:
    docs = {q["doc_id"] for q in load_qa() if q["split"] in ("dev", "test")}
    chunks = [c for c in load_chunks() if c["doc_id"] in docs]
    print(f"{len(chunks)} chunks from {len(docs)} filing pages")

    def mk(prefix: bool) -> list[DocumentInfo]:
        return [
            DocumentInfo(
                id=c["chunk_id"],
                text=(ctx_prefix(c["doc_id"]) if prefix else "") + c["content"],
                metadata={"doc_id": c["doc_id"], "local_id": c["local_id"],
                          "chunk_type": c["chunk_type"], "split": c["split"]},
            )
            for c in chunks
        ]

    client = MossClient(os.environ["MOSS_PROJECT_ID"], os.environ["MOSS_PROJECT_KEY"])
    existing = {getattr(i, "name", None) for i in (await client.list_indexes())}
    for name, prefix in (("finqa-raw", False), ("finqa-ctx", True)):
        if name in existing:
            print(f"{name} exists, skipping")
            continue
        t0 = time.perf_counter()
        res = await client.create_index(name, mk(prefix), "moss-minilm")
        dt = time.perf_counter() - t0
        info = await client.get_index(name)
        print(f"{name}: created in {dt:.1f}s -> {res!r}\n  info: {info!r}")
        append_jsonl(ROOT / "results" / "retrieval" / "index_builds.jsonl",
                     {"index": name, "docs": len(chunks), "seconds": dt, "result": repr(res), "info": repr(info)})


if __name__ == "__main__":
    asyncio.run(main())
