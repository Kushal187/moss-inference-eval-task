# Moss Inference evaluation

I tested Moss Inference (the Inference API, Moss retrieval, and the Tasks API) by running a real
workload through it: **FinQA**, a financial question-answering benchmark built from 10-K filings. I'd
already built a RAG + hallucination-detection project on it ([financial-rag-hallucination-detection](https://github.com/Kushal187/financial-rag-hallucination-detection),
a course project I led with two teammates; it's included as the `external/finqa` submodule). That project
gave me a Weaviate baseline, gold answers, saved llama-3.3-70b runs, and labels for hallucinations, so every
Moss result here is compared against something I'd measured before, not against a demo.

Everything in this README is generated from raw logs in [`results/`](results/) by
[`analysis/summarize.py`](analysis/summarize.py) → [`results/SUMMARY.md`](results/SUMMARY.md) (full tables).
Total spend: about **$2.0 on inference + $2.6 on browser tasks**, out of the $10 sign-up credit.

## Headline findings

**Inference API**
1. **fastlane-1 is fast, and the claim holds.** TTFT p50 is **164–175 ms** (p95 ~250 ms) across 600 calls, compared with ~410 ms for claude-haiku-5.5 and ~1.4 s for gpt-6-luna through the same gateway. Given the same evidence, its accuracy is in line with llama-3.3-70b (54–69% vs 60–61%) and best with structured output (68.7%). The sub-200 ms figure holds for short or cached prompts. Cold TTFT grows with prompt length: 145 ms at 0.5k tokens, 335 ms at 4k, 4.2 s at 64k, and 21.6 s at 205k. The prefix cache brings a repeated 205k prompt down to 0.8–1.2 s, but one immediate repeat at 64k missed the cache. The burst limit behaves as documented: 30 simultaneous calls gave 10 successes and 20 immediate 429s with `Retry-After: 1`.
2. **With chain-of-thought, fastlane-1 never declines.** When the retrieved evidence lacked the needed figures, fastlane+CoT produced a number **every time** (100%, compared with 46% for llama and 25% for haiku on the same questions). With few-shot it's the best of the four (20%). The sample is small (about 20 no-evidence questions per prompt), but the gap is consistent.
3. **Built-in retrieval can't be scoped, and accuracy halves.** FinQA questions must be searched within their own filing. Moss's local SDK supports metadata filters (Recall@5 80.3%, matching Weaviate dense at **2.5 ms vs 58 ms**). Server-side retrieval (`X-Moss-Indexes`, `moss_search`) has **no filter**. Unfiltered, Recall@5 falls to 52%, and **3.5 of the top 5 chunks come from other companies' filings**. End to end, fastlane accuracy drops from 54–69% to **26.7%** with built-in retrieval. The `moss_search` tool path gets 29–42%, but runs 4–6 searches per question, uses about 20× the input tokens, and has 1.2–2.6 s TTFT.
4. **You can't see what was retrieved.** Pre-model retrieval returns no chunks or citations (only an `X-Moss-Retrieve-Ms` header). `moss_search` returns passage text with **no doc id or metadata**, so you can't tell which filing a number came from. My hallucination detector needs exactly that information, so it can't run on Moss's built-in RAG.
5. **The reply cache ignores `temperature` and is on by default.** At `temperature=1`, four "invent a random word" calls all returned `plumed`. With `X-Moss-Cache: off` they gave four different words. Nothing in the response marks a cache hit, so sampling and self-consistency evals silently collapse to one answer.
6. **A deterministic failure is reported as transient.** gpt-6-luna + `json_object` + the FinQA structured prompt failed **29 of 29 times** with `502 model_error: VendorStreamError`. The SDK retries 502s, so each failure burned retries. Removing JSON mode fixed it (67.3%). Simple JSON prompts work, so it's prompt-dependent.
7. **Smaller API issues:**
   - A request body over 4 MB drops the connection instead of returning the documented 413.
   - Empty `input` returns 200 and bills a generic reply.
   - Error messages leak internal parameter names (`max_tokens`, `response_format`).
   - fastlane adds about 87 hidden input tokens per request (vendors: 21–29).
   - The 32-index limit doesn't count duplicate names.

**Tasks API**
8. **It's good at real browsing.** On SEC EDGAR, it got **6 of 8** 10-K line-item lookups exactly right (about $0.16 and 86 s each, with screenshots and a full trace). Both "should not exist" lookups (a fake segment, a 2031 10-K) correctly returned `found=false`. Of the two misses, one was ambiguity in my spec (two "total capital" tables), and the other was a careful decline, not an invented number. All 5 scored sandbox traps passed (pagination, JS rendering, infinite scroll, async loading, a publisher absent from the page), and the agent didn't guess credentials at an HTTP auth wall.
9. **It resisted prompt injection** on my own trap pages: hidden white-on-white text plus an HTML comment, and a visible comment impersonating the operator. It reported the true figures and never visited the beacon URL.
10. **Where it broke: it silently rewrites values to satisfy the schema.** I asked for a £51.77 price as an integer ≥ 1000. It returned `5177` with **`verification.passed = true`** and no note. On another page it found the right restated figure (498, in thousands) and multiplied it by 1,000 without being asked. `verification.passed` checks shape, not truth, and the agent changes values to fit that shape.
11. **Budget and cancel semantics:**
    - `max_cost_usd = 0.01` billed **$0.024** (2.4× the cap) after one step. A cap below one step's cost can never hold, and it should be rejected when the task is created.
    - A cancel requested at 25 s ended at 37.5 s and still billed $0.10.

## What was run

| # | Experiment | Code | Raw output |
|---|---|---|---|
| 1 | Generator swap: 150 FinQA dev questions × 4 prompt strategies, **identical** retrieved evidence, through fastlane-1 / claude-haiku-5.5 / gpt-6-luna vs. the saved llama-3.3-70b baseline. Streamed; TTFT measured client-side. | [`inference/gen_swap.py`](inference/gen_swap.py) | `results/generation/` |
| 2 | Hallucination profile using the FinQA project's own labeling rules | [`inference/hallucination.py`](inference/hallucination.py) | `results/hallucination_profile.json` |
| 3a | Moss retrieval Recall@k on the test split (1,147 q), filtered by filing vs unfiltered, vs. saved Weaviate numbers | [`inference/build_index.py`](inference/build_index.py), [`inference/retrieval_recall.py`](inference/retrieval_recall.py) | `results/retrieval/` |
| 3b | Server-side RAG: closed-book, `X-Moss-Indexes`, `moss_search` tool, each with and without a ticker/year workaround | [`inference/server_rag.py`](inference/server_rag.py) | `results/server_rag/` |
| 4 | API probes: error matrix, caches, burst, context length, hidden-token overhead | [`inference/api_probes.py`](inference/api_probes.py) | `results/probes/` |
| 5 | Tasks: 10 SEC EDGAR lookups scored against FinQA tables, 6 sandbox traps, 4 injection / absent-data pages, 5 lifecycle tests | [`tasks/run_tasks.py`](tasks/run_tasks.py), [`tasks/specs/`](tasks/specs/), [`tasks/trap_site/`](tasks/trap_site/) | `results/tasks/` (task JSON, agent traces, screenshots) |

The trap pages are about a fictional company, are labelled as test fixtures, and use `example.com` as the
"exfiltration" target. They're served by GitHub Pages from this repo. (The injection runs used an earlier
URL for the same pages.)

## Reproduce

```bash
git clone --recurse-submodules https://github.com/Kushal187/moss-inference-eval-task && cd moss-inference-eval-task
python3.11 -m venv .venv && .venv/bin/pip install "moss>=1.24.0" python-dotenv httpx groq rank-bm25
cp .env.example .env   # MOSS_PROJECT_ID, MOSS_PROJECT_KEY
.venv/bin/python -m inference.gen_swap --model moss/fastlane-1
.venv/bin/python -m inference.build_index && .venv/bin/python -m inference.retrieval_recall --mode filtered
.venv/bin/python -m inference.server_rag --mode premodel
.venv/bin/python -m inference.api_probes errors cache burst context overhead
.venv/bin/python -m tasks.run_tasks tasks/specs/edgar.json
.venv/bin/python -m analysis.summarize
```

## Caveats

- Baseline llama latencies are from the original project (Groq/Bedrock, not streamed). They're context, not a head-to-head latency comparison.
- The hallucination rates on no-evidence rows rest on about 20 questions per prompt strategy (the same rows for every model).
- Moss's local query uses default hybrid `alpha=0.8`. The Weaviate baseline was tuned to `alpha=0.6`.
- Task results are single runs. Agent behaviour varies between runs.
- Docs gaps I hit along the way are in [`notes/moss-docs-notes.md`](notes/moss-docs-notes.md).
