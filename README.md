# Moss Inference evaluation

I tested Moss Inference by running an existing project through it: a RAG and hallucination-detection
system I built on FinQA (questions about 10-K filings), with two teammates. That project
([repo](https://github.com/Kushal187/financial-rag-hallucination-detection), included here as the
`external/finqa` submodule) already had a Weaviate baseline, gold answers, and saved llama-3.3-70b runs,
so every Moss number below is compared against something measured before.

Full tables are in [`results/SUMMARY.md`](results/SUMMARY.md). Raw logs, task traces and screenshots are in [`results/`](results/).

## What I ran

1. **Model swap.** The same 150 questions, 4 prompt styles and retrieved evidence as the original project, sent through `moss/fastlane-1`, `claude-haiku-5.5` and `gpt-6-luna` instead of llama.
2. **Hallucination check.** The project's labels and LLM judge applied to each model's answers, with the judge itself called through Moss.
3. **Retrieval.** The corpus loaded into Moss indexes. Recall compared with Weaviate, then end-to-end accuracy using Moss's built-in retrieval.
4. **API probes.** Error handling, caching, rate limits, and latency vs. prompt length.
5. **Tasks API.** 10 SEC EDGAR lookups checked against FinQA's numbers, 6 scraping-sandbox traps, prompt-injection pages I hosted, and budget/cancel/schema edge cases.

## Results

| Inference (150 questions) | Accuracy | Time to first token (p50) |
|---|---|---|
| llama-3.3-70b, original run | 60–61% | n/a |
| fastlane-1 | 54–69% | 165 ms |
| claude-haiku-5.5 | 52–67% | 410 ms |
| gpt-6-luna | 61–67% | 1.4 s |
| fastlane-1 + Moss built-in retrieval | 27% | 185 ms |

| Retrieval (1,147 questions) | Recall@5 | Latency |
|---|---|---|
| Weaviate hybrid (original) | 82.1% | 53 ms |
| Moss, filtered to the right filing | 80.3% | 2.5 ms |
| Moss, unfiltered (what server-side retrieval does) | 52.4% | 2.7 ms |

| Tasks | Result |
|---|---|
| EDGAR 10-K lookups | 6/8 values correct, 2/2 "doesn't exist" cases correct, ~$0.16 each |
| Sandbox traps (pagination, JS, infinite scroll, auth wall, missing data) | 5/5 scored traps passed |
| Prompt injection (hidden text, fake operator comment) | Both ignored |
| Impossible schema | Returned a wrong value and passed verification |

## What I found

**Worked well**
- fastlane-1 hits its sub-200 ms first-token claim on short prompts, and its accuracy matches llama.
- As a hallucination judge, fastlane matched my best previous judge (F1 0.54) at much lower cost.
- Moss's local retrieval, filtered to the right filing, is within 2 points of Weaviate's recall at about 20× lower latency.
- The Tasks agent handled real EDGAR filings, declined to invent missing data, didn't guess credentials, and ignored injected instructions.
- The portal's logs, task recordings (with the agent's reasoning as captions) and usage breakdown are good. The usage numbers matched my own accounting exactly.

**Problems**
- Server-side retrieval has no per-request filter. Without one, 3.5 of the top 5 chunks came from other companies' filings, and accuracy fell from about 60% to 27%. The response also doesn't say which chunks were retrieved.
- The reply cache is on by default and ignores `temperature`. Four "random word" calls at temperature 1 all returned the same word, with nothing marking the cache hit.
- With chain-of-thought, fastlane answered every question even when the evidence was missing. llama and haiku declined some of the time.
- First-token time grows with uncached prompt size: 4.2 s at 64k tokens, 21.6 s at 205k.
- gpt-6-luna with JSON mode failed 29 of 29 times with a 502, an error code that tells clients to retry.
- Tasks: asked for a £51.77 price as an integer ≥ 1000, the agent returned `5177` (pence) and `verification.passed` was true. A $0.01 budget cap billed $0.024. A cancel took 12 s and still billed $0.10.
- Pricing: index ingest is metered on the embedded index (about 12× the source text), and "local" SDK queries are billed per query. Together they were 41% of my bill, and neither is clear from the docs.
- Smaller issues: a body over 4 MB drops the connection instead of returning 413. Empty input returns 200. The onboarding sample index uses one of the free plan's 3 index slots.

## Cost

$9.18 of the $10 credit. Inference was $2.67, browser tasks $2.63, index ingest $2.65, local queries $1.08, and storage $0.15. The breakdown is in [`results/portal/`](results/portal/).

## Layout

```
inference/     model swap, hallucination profile, judge, retrieval, server-side RAG, API probes
tasks/         task runner, task specs, trap site (served via GitHub Pages)
analysis/      builds results/SUMMARY.md from the logs
results/       raw outputs, task traces and screenshots, portal screenshots
notes/         gaps I found in the docs
external/finqa original FinQA project (submodule)
```

## Running it

```bash
git clone --recurse-submodules https://github.com/Kushal187/moss-inference-eval-task
cd moss-inference-eval-task
python3.11 -m venv .venv && .venv/bin/pip install "moss>=1.24.0" python-dotenv httpx groq rank-bm25
cp .env.example .env    # add MOSS_PROJECT_ID and MOSS_PROJECT_KEY
.venv/bin/python -m inference.gen_swap --model moss/fastlane-1
.venv/bin/python -m tasks.run_tasks tasks/specs/edgar.json
.venv/bin/python -m analysis.summarize
```

## Caveats

- llama latencies come from the original project (Groq/Bedrock, not streamed), so I don't compare them directly.
- The "missing evidence" results rest on about 20 questions per prompt style.
- Each task ran once, and agent behaviour varies between runs.
- The trap pages are about a fictional company and use `example.com` as the injection target.
