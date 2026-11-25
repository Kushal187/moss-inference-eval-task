# Moss Inference — docs notes (read 2026-11-25)

Source pages: docs.moss.dev/docs/inference/{overview,quickstart,models,tasks,pricing,errors-and-limits,streaming,retrieval,openai-compatibility,tool-calling,sdks/python}.md, start/core-concepts, integrate/indexing-data, reference/python/api, api-reference/v1/inference/tasks.

## Setup
- `pip install "moss>=1.24.0"` (tasks need 1.23, BYOK tasks 1.24; docs pins disagree)
- `from moss import MossInference, MossClient, DocumentInfo`
- `MossInference(project_id, project_key, timeout=, idle_timeout=120, max_retries=2)`; async only
- Auth: `Authorization: Bearer <project_id>:<project_key>`; env `MOSS_PROJECT_ID`, `MOSS_PROJECT_KEY`
- Base URL `https://inference.moss.dev/v1` — `/responses`, `/chat/completions`, `/models`, `/tasks`, `/tasks/{id}`, `/tasks/{id}/cancel`, `/tasks/{id}/files[/{path}]`
- OpenAI-compatible (Responses primary; Chat Completions supported)

## Inference
- `await client.responses.create(model=, input=, instructions=, stream=, tools=, tool_choice=, max_output_tokens=, indexes=[...], retrieve=True, cache=True, ...)`
- Headers: `X-Moss-Indexes`, `X-Moss-Retrieve: off`, `X-Moss-Cache: off`; response headers `X-Moss-Retrieve-Ms`, `X-Moss-Request-Id`
- Stream: `async with stream: async for event in stream:` → `response.output_text.delta` / `event.delta`; `await stream.get_final_response()`
- Usage: `usage.input_tokens` (includes retrieved facts), `output_tokens`, `input_tokens_details.cached_tokens`
- **No documented citations / retrieved-docs field in the response.**
- Not supported: `previous_response_id`, `conversation`, `background`, `include`, `truncation:auto`; unknown fields → 400
- Models: `moss/fastlane-1` ($0.25 in / $0.03 cached / $1.75 out per 1M; 262k ctx), `openai/gpt-6-luna`, `openai/gpt-6.1-terra`, `anthropic/claude-haiku-5.5`, `anthropic/claude-sonnet-5.5`, `google/gemini-3.8-flash`, `google/gemini-3.5-flash-lite`, open models (`deepseek/deepseek-v4.1-flash`, `qwen/qwen3.8-flash`, `z-ai/glm-5.3-flash`, ...). Live list: `GET /v1/models`.
- Retrieval billed $0.50 / 1k retrievals; reply-cache hit is free.
- Rate limits: fastlane 120 RPM / 10 burst; vendor 300 RPM / 50 burst.

## Indexes
- `MossClient(id, key)`; `await client.create_index(name, docs, "moss-minilm")`; `add_docs(name, docs, MutationOptions(upsert=True))`; `get_job_status(job_id)`; `list_indexes()`; `delete_index(name)`
- `DocumentInfo(id=str, text=str, metadata={str: str})`
- Embedding models: `moss-minilm` (default), `moss-mediumlm`, `custom`
- Mutations are async jobs — **docs don't say whether inference must wait for the job.**

## Tasks
- `await client.tasks.run(goal=, environment={"type":"browser","start_url":...}, limits={"max_runtime_seconds":..,"max_cost_usd":..})` — polls until done; failure is returned not raised
- Also `tasks.create`, `tasks.wait`, `tasks.cancel`, `tasks.files.list`, `tasks.files.content`; **no SDK `get` documented**
- REST body: `goal` (≤20k chars), `environment`, `model` (default `moss-balanced`; `moss-fast`/`moss-advanced`/any model), `effort` (low/medium/high), `output_schema` (**SDK kwarg name not shown**), `limits.max_runtime_seconds` (60–7200, default 1800 — prose contradicts), `limits.max_cost_usd` (0.01–50, default 2), `metadata`; `Idempotency-Key` header
- Status: queued / running / completed / failed / cancelled
- Fields: `result{output_text, output}`, `verification{method, passed, problems}`, `usage.cost{model_usd, browser_usd, traffic_usd, total_usd}`, `usage.steps`, `error{code,message}`
- Schema mismatch still ends `completed` → check `verification.passed`
- `max_cost_usd` caps model calls only; in-flight calls finish and bill past the cap
- Error codes: `time_limit`, `budget_exceeded`, `no_answer`, `provider_key_rejected`
- 20 queued tasks / project; failed & cancelled tasks still bill; files kept 7 days
- Pricing: model tokens +20%, browser $0.02/hr, traffic $5/GB

## Errors
- `{"error":{"message","type","code"}}`; 400 / 401 / 403 / 413 / 424 / 429 / 502 / 503
- Python `moss.inference.InferenceError(status, type, code, message, request_id, retry_after)`

## Doc gaps already found (feedback material)
1. Install version pins inconsistent (1.22 vs 1.23 vs 1.24).
2. BYOK model naming inconsistent (`gpt-6-luna` vs `openai/gpt-6-luna`).
3. Task default runtime contradicts itself (1800 vs 1200/3600 by preset).
4. Embedding model list differs between pages (`moss-litelm`).
5. Index readiness after `create_index` undocumented.
6. No citations/retrieved chunks in response → can't audit grounding.
7. No per-model feature matrix (streaming/tools/structured output).
8. Tasks: no SDK `get`, `verification.method`/`problems` format undocumented, `output_schema` SDK param not shown.
