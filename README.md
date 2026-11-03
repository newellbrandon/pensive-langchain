# Pensive — LangGraph + MongoDB Demo

A minimal AI agent demo using **LangGraph** for workflow orchestration and **MongoDB** for both:

- **Short-term memory** — LangGraph checkpoints (`MongoDBSaver`) persist conversation state per thread
- **Long-term memory** — LangGraph Store (`MongoDBStore`) with semantic vector search across sessions

The API is **OpenAI-compatible** so you can connect [OpenWebUI](https://docs.openwebui.com/) or a coding harness (e.g. OpenCode) and chat with the agent. Workflow decisions render as native reasoning **Thought** blocks; the assistant message contains only the final answer.

## Architecture

### System overview

OpenWebUI or a coding harness sends chat requests to the FastAPI layer, which invokes a compiled LangGraph workflow. MongoDB holds short-term thread state (checkpointer) and long-term facts (store with vector search). On top of the core chat loop, Pensive runs a **server-tool layer** (subagent delegation, project browsing, Glean enterprise search), an **agent lab** (isolated, runnable/rerunnable agents), a **document knowledge base** (contextualized chunk embeddings — agents ground answers in uploaded PDFs via `documents_search` on demand), and a **management console** at `/ui` — with every agent workflow recorded as sortable, metric-bearing run history.

Pensive exposes both OpenAI-compatible protocols: **Chat Completions** (`POST /v1/chat/completions`) and the **Responses API** (`POST /v1/responses`). OpenWebUI can be pointed at either connection type.

```mermaid
flowchart TD
  client[OpenWebUI / coding harness] -->|"POST /v1/chat/completions · /v1/responses"| api[FastAPI]
  api -->|"ainvoke / astream"| lg[Compiled LangGraph]
  lg --> llm["OpenAI-compatible LLM"]
  lg -.->|"MongoDBSaver · every superstep"| checkpoints["checkpoints<br/>checkpoint_writes"]
  lg -->|"MongoDBStore · typed namespaces"| memories[("agent_memories")]
  memories -->|"vector search + iterate-on-merge"| embed["Voyage embeddings"]

  lg -.->|"spawn_subagent · digest only"| sub["Isolated subagent<br/>memory agent:sub-* · own workspace"]
  sub -->|"runs the same graph"| lg
  lg -.->|"glean_* tools · MCP + OAuth2"| glean[("Glean enterprise search")]
  lg -.->|"project_list / project_read"| projects[("~/projects · read-only")]
  lg -.->|"documents_search · on demand"| dstore[("document_chunks<br/>voyage-context-4 · 1024d")]

  subgraph console["Pensive Console — /ui"]
    ui["Memories · Agents lab · Documents · Glean · System Prompt"]
  end
  ui -->|"run_agent · same graph"| lg
  ui -->|"records executed workflows"| runs[("pensive_agent_runs<br/>pensive/test tags · metrics · 1500/30d retention")]
  ui -->|"upload → extract → embed (contextualized)"| dstore
  api --> ui
```

### LangGraph workflow

Every chat message — and every agent-lab/subagent run — runs the four nodes below. The graph also carries a **server-tool loop inside `generate_response`** (offered only in main-context chats): the model may delegate via `spawn_subagent`, browse host projects (`project_list` / `project_read`), or call the discovered Glean tools (`glean_*`); results are executed server-side and fed back until the model produces a final answer. OpenWebUI streams the **decision** from each step before the final answer — the second line in each node label matches what you see in the demo screenshots.

The checkpointer persists graph state after every superstep (not only these four nodes), which is why a single message produces multiple `checkpoints` and `checkpoint_writes` documents.

```mermaid
flowchart TD
  startNode((START)) --> analyze["analyze_intent<br/>Classify Intent"]

  analyze -->|"needs_memory"| search["search_long_term_memory<br/>Search Long Term Memory"]
  analyze -->|"skip"| generate["generate_response<br/>Generate Response"]

  search -->|"store.search · 4 types"| agentMem[("agent_memories")]
  search --> generate

  generate -.->|"OpenAI Responses API"| llmNode[["LLM"]]
  generate -.->|"server-tool loop: spawn_subagent · project_* · glean_*"| serverTools["server tools<br/>budgeted · results fed back"]
  generate --> save["save_memory<br/>Save Memory"]

  save -->|"LLM extract + classify type"| llmNode
  save -->|"store.put · typed namespace · iterate-on-merge"| agentMem
  save --> endNode((END))

  agentLab["Agents lab — run_agent<br/>memory agent:&lt;id&gt; scope"] -->|"same graph"| startNode
  agentLab -.->|"workflow history"| runs2[("pensive_agent_runs<br/>pensive/test · tokens · duration")]
  runs2 --> agentLab

  analyze -.-> cpNote["checkpoints / checkpoint_writes"]
  search -.-> cpNote
  generate -.-> cpNote
  save -.-> cpNote
```

**Demo paths** (see [Demo Results](#demo-results-openwebui)):

| Step | Session 1 — "My name is Brandon." | Session 2 — "Do you remember me?" |
| ---- | ----------------------------------- | --------------------------------- |
| Classify Intent | `memory` — personal context | `memory` — recall |
| Search Long Term Memory | `0 results` | `1 results` |
| Generate Response | `respond` — no retrieved context | `respond` — uses 1 memory |
| Save Memory | `saved` → new `agent_memories` doc | `skipped` — no new fact |

## Coding-Harness / Persistent-Memory Context

Pensive can serve as an OpenAI-compatible model for a coding harness (e.g. OpenCode)
while providing persistent memory for important context:

- **Full context is honored.** Every incoming request's messages (system prompt,
  conversation history, tool results) are passed through to the model; Pensive does
  not discard anything the client sends.
- **Tool calling works.** Tool definitions from the client are forwarded to the
  model; tool calls the model requests are relayed back on the native channel
  (`message.tool_calls` / `delta.tool_calls` on `/v1/chat/completions`,
  `function_call` output items / `response.function_call_arguments.delta` on
  `/v1/responses`), and tool results replay as `function_call_output` items — so
  a coding harness can drive read/write/execute tools through Pensive.
- **Persistent memory runs on top.** The four-type long-term memory pipeline
  retrieves and injects remembered context, and persists durable facts — scoped by
  `user_id` — so they carry across sessions, chats, and workflows.
- **Subagent delegation keeps the primary context clean.** Main-context chats are
  offered a server-side `spawn_subagent` tool: the model can delegate a
  self-contained task (research, builds, long tool chains) to an isolated
  subagent with its own memory scope (`agent:sub-…`), workspace, and tools. Only
  a short digest of the subagent's final answer returns to the conversation (at
  most 2 spawns per turn, no nesting — subagents never see the spawn tool).
  Full artifacts land in the subagent's workspace and run history.
- **Read access to your projects (incl. the OKF bundle).** The host's
  `~/projects` directory (or `PROJECTS_HOST_DIR`) is mounted read-only into the
  container at `/projects`, and main-context chats plus all agents get
  `project_list` / `project_read` tools (path-escape-safe, capped at 40k chars
  per read, disabled when `PENSIVE_PROJECTS_ROOT` is unset). The OKF knowledge
  bundle lives under `okf/`.
- **Glean enterprise search (MCP, no agentengine).** Pensive connects to the
  Glean MCP server (`GLEAN_MCP_URL`) with a native OAuth2 dynamic-client flow
  (DCR + PKCE; the `MCP` scope is required by Glean) and a bind-mounted token
  cache. Run `python scripts/glean_auth.py` once — or use the console's
  **Glean** tab (Connect button, the OAuth callback rides the existing
  `localhost:8000` port) — and the ~13 discovered Glean tools become
  server-side tools in main chats (`glean_search`, `glean_chat`,
  `glean_read_document`, …). The discovered tools are also appended to the
  tool set of **every agent run — lab agents and spawned subagents alike**
  (and the `web_search` description steers to `glean_search` for internal
  content), so "search exclusively via Glean" tasks never fall back to the
  public web tools. Calls are permission-scoped to the signed-in
  user. **Boundaries:** this is a single-user local-dev credential — never
  ship or upload the token cache (` .glean-tokens.json`).
- Connect any OpenAI chat/responses client to `http://<host>:8000/v1` with model
  `pensive` (see [OpenWebUI Configuration](#openwebui-configuration) and the OpenCode
  provider config in the OpenCode docs).

## Pensive Console (Web UI)

The management UI lives at **`http://<host>:8000/ui`** (MongoDB visual theme — dark navy
canvas, mongo-green accents) and powers five tabs:

| Tab | What you can do |
| --- | --------------- |
| **Memories** | Browse and curate the four long-term memory types per user — **Semantic** (user facts and preferences), **Episodic** (session events, decisions, outcomes), **Taxonomic** (domain vocabulary and reusable concepts like corporate policies), **Procedural** (reusable workflows and agent runbooks — skills). Correct a memory and it is **re-embedded automatically** (the write goes through the LangGraph store's vector index); prune to delete. Saves **iterate instead of duplicating**: repeated facts merge into the existing memory (vector-similarity match with lexical fallback, normalized labels), and a **Merge duplicates** button groups semantically similar memories into one complete curated memory per group — near-identical line items collapse (longer version wins) and each group gets an LLM consolidation pass so the result reads as one memory, not a concatenation. The live store went from 101 → 31 memories with the content preserved. |
| **System Prompt** | View and change the assistant persona injected at the top of every main-context request. Persisted in MongoDB; leave empty to restore the built-in default. |
| **Agents** | Deploy a **test agent** with its own system prompt and tools, then run a task in **complete isolation** — memory is scoped to `agent:<id>` so it can't pollute (or be polluted by) the main Pensive context. The tab is split into three segmented views: **Test Agents** (create form + registry; leave the name blank to **auto-generate** one like `coral-scribe-3621`, and tools empty for the built-in sandboxed set `write_file`, `append_file`, `read_file`, `run_command`, `web_search`, `web_extract`, `documents_search`), **Test Agent History** (every lab-agent run, locked to `test`, sortable), and **Agent Workflows** (every workflow incl. `pensive` subagent runs — aligned Source · Agent · When · Tokens · Tool calls · Runtime columns with clickable filter chips and sort chips). Clicking any row expands the full task, answer, step transcripts, a **Reuse as specialist agent** action, and the workspace files (assets) — a capable subagent (say, a great Glean researcher) can be promoted into a permanent, rerunnable agent with its own memory scope. **Edit any agent** (Edit button loads its name/prompt/tools back into the form), and open **Runs & artifacts** on a card to review past runs — each run shows a **step-by-step transcript** (model response, decisions, and every executed tool call with full arguments and results), a one-click **Run again** button, and a **workspace file browser** (list + content preview via `GET /agents/{id}/workspace`). |
| **Documents** | Build a **queryable knowledge base** from uploaded files (PDF / txt / md) using **contextualized chunk embeddings** (`voyage-context-4`) — extract text (pypdf), split into bounded segments, and embed each chunk **in the context of its full document** (server-side auto-chunking, 512-token chunks, char-budgeted batches to respect the API's per-request token cap). The library lists every document with its status (`ingesting → ready | failed`), pages, chunk count, tokens, and model — plus **Chunk preview** (text per chunk), **Re-embed** (retry a failed ingest), and **Delete** (drops chunks + the stored original). A **semantic search** box embeds the query (`input_type=query`) and returns the top-k chunks with scores, optionally restricted to one document. The **`documents_search` tool** then makes the corpus available to every agent on demand — it is never injected into prompts, so queries that don't need it pay nothing. |
| **Glean** | Connect Pensive to the Glean MCP server (MongoDB enterprise search) with a native **OAuth2 dynamic-client flow** (DCR + PKCE; the `MCP` scope is required by Glean): status + token expiry, **Connect** (opens the authorization screen; the callback rides the existing `localhost:8000` port), **Refresh token**, **Disconnect**, and a **smoke-test box** that calls any discovered tool (`search: …` runs `glean_search`). Once connected, the ~13 discovered Glean tools are available to main chats **and every agent run** as `glean_*` server tools, permission-scoped to the signed-in user. The token cache (` .glean-tokens.json`) is gitignored and bind-mounted — a single-user dev credential, never to be shipped. |

**Agent tool execution.** Agent tool calls are actually *executed*: file and shell tools run inside a per-agent workspace directory (`./agent-workspaces/<agent_id>/`, mounted into the container so built apps survive restarts), results are fed back to the model in a loop (up to 24 steps), and transient upstream failures or broken tool-call payloads are retried automatically. Every tool failure becomes a model-visible `[error]` result — never a crash — with actionable guidance (invalid JSON is reported with reissue instructions; large files should be built with `append_file`). Completed runs are persisted (`pensive_agent_runs`) with their answer, decisions, and executed tool calls so past agents can be reviewed and re-run — the counts **aggregate every executed call, server-tool executions included** (glean/spawn/project work inside the generate loop is folded into the step transcripts), and legacy runs derive their call/step totals from the stored transcripts. The workflow list is **retained to at most 1500 runs and none older than 30 days** — whichever limit is hit first, enforced after every recorded run.

Backing REST API (all under `/api/admin`): `memory-types`, `memories` (GET/PUT/DELETE),
`system-prompt` (GET/PUT), `agents` (CRUD + `POST /agents/{id}/run` + `GET /agents/{id}/runs`
+ `GET /agents/{id}/workspace` + `GET /agents/{id}/workspace/file?path=…`), `runs`
(global workflow feed, tagged `pensive` vs `test`, filterable via
`?executor=` and sortable by `created_at`
/ `tokens_total` / `duration_ms` / tool-call and step counts via
`?sort_by=&order=`), and `documents` (`GET` library + `POST /upload` (multipart,
background ingest) + `DELETE /{id}` + `POST /{id}/reembed` + `GET /{id}/chunks`
+ `POST /search`).

## Document Knowledge Base (contextualized chunk embeddings)

Documents uploaded on the **Documents** tab become a searchable corpus agents can
ground answers in — on demand, never ambient:

1. **Extract** — PDFs are read with `pypdf` (page count kept); `.txt`/`.md` pass
   straight through. Originals are stored under `document-uploads/` (bind-mounted —
   deliberately outside the `documents/` python package) so re-embedding works
   after restarts.
2. **Segment** — text is split into bounded segments (≤ 90K chars, preferring
   paragraph boundaries) and sent in **char-budgeted batches** — the contextualized
   embedding API caps a whole request at 120K tokens, so batches stay well inside
   the budget.
3. **Embed** — each batch goes to `POST {CONTEXT_EMBEDDING_URI}`
   (`https://ai.mongodb.com/v1/contextualizedembeddings` by default, same key as
   `LLM_EMBEDDING_KEY`) with `model=voyage-context-4`, `input_type=document` and
   **server-side auto-chunking** (512-token chunks): every chunk is encoded in the
   context of its full document, so vectors carry book-level context without any
   manual metadata augmentation. Chunk text + 1024-dim vectors land in
   `document_chunks`; status bookkeeping (`ingesting → ready | failed`, chunk
   count, tokens, pages, model) in `documents`.
4. **Query** — the same endpoint encodes the query (`input_type=query`); the top-k
   chunks are found with cosine similarity over the stored vectors (pure-Python
   scan — at lab scale no Atlas Search index is needed; swap in `$vectorSearch`
   when the corpus grows). Optionally restrict to one document.

Every agent's toolset includes **`documents_search(query, k?, document?)`** — a
**memory-first** knowledge lookup: Pensive's long-term memory is swept *before*
the corpus — the run's own namespace, the console user's (`default`), and
every other agent/subagent namespace. When a confident match exists
(similarity ≥ 0.80 — this information was already retrieved and stored, by
this agent, you, or another agent) the memory facts answer the query, the
tool says so, and the corpus is **not** queried at all. Only weak or empty
memory matches fall through to the corpus, which costs one query embedding +
scan.
The corpus is never injected into prompts automatically. When a run did
retrieve chunks, its answer ends with a deterministic one-line **Sources**
suffix — just the name(s) of the ingested resource(s), deduped and best-first
(e.g. `Sources: High Performance MongoDB — Asya Kamsky`) — parsed from the
tool digests stored in the run transcript, so the citation sits at the end and
the answer text stays clean to copy-paste.

## Long-Term Memory (Four Types)

Pensive's long-term memory is organized into the same four types used by the
Atlas Agent Engine memory server, all stored in the `agent_memories` collection
under per-type namespaces on the LangGraph MongoDB Store:

| Type        | Meaning                                              |
| ----------- | ---------------------------------------------------- |
| `semantic`  | Stable facts and preferences about the user          |
| `episodic`  | Specific past events / interactions                  |
| `taxonomic` | Categories or domain knowledge the user relies on    |
| `procedural`| How the user does things (steps, output preferences) |

The pipeline (ported from the leafy10 agent in `memory/memories.py`) works like this:

1. **Intent planning** (`analyze_intent`) — an LLM turns the latest message into a
   semantic search `query` and a `needs_memory` flag.
2. **Retrieval** (`search_long_term_memory`) — vector-searches all four types,
   drops superseded/empty facts, and reranks with Voyage (`rerank-2.5-lite`),
   falling back to vector-score order when `VOYAGE_API_KEY` is unset.
3. **Generation** — the dated, newest-first memory block is injected as a separate
   system message and personalized into the reply.
4. **Write-back** (`save_memory`) — the LLM extracts durable facts and classifies
   each into one of the four types. Saves **iterate instead of duplicating**:
   labels are normalized (casing/whitespace collapse), and before inserting, a
   fact that vector-matches an existing same-type memory (≥ 0.85 similarity,
   lexical fallback) is **merged into it** — content combined, canonical label
   and earliest `recorded_at` kept — instead of stored as a new record. The
   console's **Merge duplicates** cleanup groups semantically similar memories
   per type and curates each group into ONE memory: near-identical lines
   collapse to their longer version, then an LLM consolidation pass rewrites
   the fragments as a single flowing memory (best-effort, heuristic fallback).
   Enable rerank by setting `VOYAGE_API_KEY` in `.env`.

## Requirements

- Python 3.13+
- MongoDB with Vector Search (local 8.x or Atlas)
- OpenAI-compatible LLM endpoint. Pensive speaks the OpenAI **Responses API** (`LLM_URI`, `LLM_KEY`, `LLM_MODEL`) so it works correctly with reasoning models; the final answer is read from the response's `output` message items and any model reasoning/"thinking" is discarded.
- Voyage embeddings via MongoDB AI (`LLM_EMBEDDING_KEY`) — custom `VoyageEmbeddings` client in `utils/llm.py`
- Document ingestion uses the same key against the **contextualized embeddings** endpoint (`CONTEXT_EMBEDDING_URI`, default `https://ai.mongodb.com/v1/contextualizedembeddings`, model `CONTEXT_EMBEDDING_MODEL`, default `voyage-context-4`) — the Files tab uploads land here with server-side auto-chunking
- Optional memory rerank — set `VOYAGE_API_KEY` to enable Voyage reranking of retrieved memories; otherwise Pensive ranks by vector similarity.

## Setup

1. Copy environment template:

```bash
cp env.example .env
```

1. Edit `.env` with your MongoDB URI, LLM credentials, and embedding key.
2. Create and activate the virtual environment:

```bash
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

1. Create the vector search index (run once):

```bash
python scripts/setup_memory_index.py
```

Index definition (for manual creation in Atlas UI if the script fails):

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1024,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "user_id"
    }
  ]
}
```

1. Start the API:

```bash
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000
```

Verify health: `curl http://localhost:8000/health`

## Agent Skills

This project is a good fit for official agent skills that guide coding agents (Cursor, Claude Code, and others) when extending or debugging the workflow.

### [MongoDB Agent Skills](https://github.com/mongodb/agent-skills)

Schema design, connection setup, vector search, natural-language querying, and MCP server configuration for MongoDB.

**Cursor** — install from the marketplace or run:

```
/add-plugin mongodb
```

**Any agent** (via [skills.sh](https://skills.sh/)):

```bash
npx skills add mongodb/agent-skills
```

Relevant skills for this repo: `mongodb-connection`, `mongodb-schema-design`, `mongodb-search-and-ai`, `mongodb-natural-language-querying`.

### [LangChain Skills](https://github.com/langchain-ai/langchain-skills)

LangGraph workflows, checkpointing, persistence, and LangChain agent patterns.

```bash
npx skills add langchain-ai/langchain-skills --skill '*' --yes
```

Relevant skills for this repo: `langgraph-fundamentals`, `langgraph-persistence`, `langchain-fundamentals`, `langchain-dependencies`.

## Observability

`user_id`, `chat_id`, and `session_id` come from the OpenWebUI session — not a static env var. Enable header forwarding on your OpenWebUI instance:

```bash
ENABLE_FORWARD_USER_INFO_HEADERS=True
```

OpenWebUI then sends `X-OpenWebUI-User-Id`, `X-OpenWebUI-Chat-Id`, and `X-OpenWebUI-Message-Id` on each request to Pensive. Pensive maps `chat_id` to the LangGraph `thread_id` (checkpoint scope per conversation) and forwards all session fields to LangSmith `metadata`.

For `curl` without OpenWebUI, set optional `USER_ID` in `.env` or pass `"user": "demo-user"` in the JSON body.

LangSmith tracing is optional and controlled by environment variables in `.env`:

| Variable | Purpose |
| -------- | ------- |
| `LANGSMITH_TRACING` | Set to `true` to enable tracing |
| `LANGSMITH_API_KEY` | Your LangSmith API key |
| `LANGSMITH_PROJECT` | Project name (default: `pensive`) |
| `LANGSMITH_ENDPOINT` | API endpoint (default: `https://api.smith.langchain.com`) |
| `USER_ID` | Identity pin — when set, every client shares this memory scope (overrides forwarded OpenWebUI user headers and body `user`); leave empty for per-user memory |

When `LANGSMITH_TRACING=true` and a valid `LANGSMITH_API_KEY` is set, traces appear at [smith.langchain.com](https://smith.langchain.com) under the configured project. Each chat request produces a top-level run with tags `pensive` and `chat`, `metadata.user_id` from the OpenWebUI user, plus `chat_id` / `session_id` when forwarded, including nested LangGraph node spans, OpenAI Responses API LLM calls, and `voyage_embed` embedding spans during memory search.

If tracing is disabled or the API key is missing, the API starts normally without reporting to LangSmith.

## Docker

Run the API in a container (MongoDB stays on the host):

1. Set `MONGODB_URI` in `.env` to use `host.docker.internal` instead of `localhost`:

```bash
# Example — replace credentials with your values
MONGODB_URI=mongodb://username:password@host.docker.internal:27017/?authSource=admin&directConnection=true
```

1. Build and start:

```bash
docker compose up -d --build
```

1. Verify:

```bash
curl http://localhost:8000/health
```

Stop with `docker compose down`. Vector index setup (`scripts/setup_memory_index.py`) runs on the host against your MongoDB, not inside the container.

## OpenWebUI Configuration

In OpenWebUI → **Settings → Connections → OpenAI**:


| Field   | Value                   |
| ------- | ----------------------- |
| API URL | `http://<host>:8000/v1` |
| API Key | any non-empty string    |
| Model   | `pensive`               |

Connection type: **Chat Completions** (default) or **Responses** — pensive serves both `POST /v1/chat/completions` and `POST /v1/responses`.

On the OpenWebUI server, enable session header forwarding so Pensive receives the logged-in user and chat context:

```bash
ENABLE_FORWARD_USER_INFO_HEADERS=True
```

See [OpenWebUI environment variables](https://docs.openwebui.com/getting-started/env-configuration/) for details. Without this, Pensive cannot resolve `user_id` from the OpenWebUI session (unless you set optional `USER_ID` in `.env` for local testing).

Enable streaming in chat settings for the best experience — the final answer streams token-by-token.

The agent's workflow decisions (**Classify Intent → Search Long Term Memory → Generate Response → Save Memory**) **stream live as they are made** on the **native reasoning channel** — OpenAI-compatible clients render the growing thought as a collapsible **Thought** block (e.g. OpenCode) or a thinking pane (e.g. OpenWebUI): `/v1/responses` emits a `reasoning` output item with `response.reasoning_summary_text.delta` events; `/v1/chat/completions` sends `delta.reasoning_content`. The same trace is mirrored as structured metadata (`metadata.workflow` on `/v1/responses`, a top-level `workflow` field on `/v1/chat/completions`). The assistant message itself always contains only the final answer.

## Demo Results (OpenWebUI)

These screenshots show Pensive connected to OpenWebUI with streaming enabled. The streamed response is the final answer; the workflow trace below is shown for reference:

### Session 1 — Save a fact

**Prompt:** "My name is Brandon."

![Session 1: telling the agent your name](session_1.png)

Workflow:

- **Classify Intent** → `memory` — query references personal context
- **Search Long Term Memory** → `0 results` — no prior memories found
- **Generate Response** → `respond` — answer generated without retrieved context
- **Save Memory** → `saved` — stored "User's name is Brandon"

**Answer:** "Nice to meet you, Brandon! How can I help you today?"

#### MongoDB after this interaction

A single Session 1 message on a fresh database produces:

![MongoDB collection stats after Session 1](session_1_docs.png)


| Collection          | Documents | Role                                                   |
| ------------------- | --------- | ------------------------------------------------------ |
| `agent_memories`    | 1         | Long-term fact with embedding (LangGraph Store)        |
| `checkpoints`       | 6         | Thread-scoped graph state snapshots (~1 per superstep) |
| `checkpoint_writes` | 18        | State-channel writes across those steps                |


LangGraph persists checkpoint state at each graph superstep, not just the four workflow nodes visible in OpenWebUI — so **6 checkpoints and ~18 writes per message is expected**, not a leak.

### Session 2 — Recall across sessions

**Prompt:** "Do you remember me?" (new chat, ~1 minute after Session 1)

![Session 2: asking if the agent remembers you](session_2.png)

Workflow:

- **Classify Intent** → `memory` — query references personal context or recall
- **Search Long Term Memory** → `1 results` — found the name saved in Session 1
- **Generate Response** → `respond` — answer generated using 1 retrieved memory
- **Save Memory** → `skipped` — no new durable fact to save

**Answer:** "Yes, Brandon! Good to hear from you again. How can I help?"

Session 2 demonstrates **long-term memory**: the fact persisted in MongoDB (`agent_memories`) and was retrieved via semantic search in a separate OpenWebUI chat. Session 2 adds another checkpoint thread (~6 checkpoints, ~18 writes) but does not add a new `agent_memories` document when save is skipped.

## API Endpoints


| Endpoint                    | Description                          |
| --------------------------- | ------------------------------------ |
| `GET /health`               | Liveness check (503 if MongoDB down) |
| `GET /v1/models`            | Lists model `pensive`                |
| `POST /v1/chat/completions` | Chat (supports `stream: true`)       |
| `POST /v1/responses`        | OpenAI **Responses API** chat (supports `stream: true`) |


### Example (streaming)

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "pensive",
    "stream": true,
    "user": "demo-user",
    "messages": [{"role": "user", "content": "Remember that my favorite database is MongoDB"}]
  }'
```

## Project Structure

```
pensive/
├── Dockerfile              # container image (+ pytest for the agent lab)
├── docker-compose.yml      # mounts agent-workspaces/, ~/projects (ro), glean token cache
├── main.py                 # uvicorn entry
├── config.py               # environment configuration
├── agent/
│   ├── graph.py            # LangGraph workflow (+ subagent wiring)
│   ├── nodes.py            # analyze_intent, search, generate (server-tool loop), save
│   └── state.py            # AgentState schema (incl. token accounting)
├── admin/
│   ├── executor.py         # sandboxed agent tools (files, shell, web, projects)
│   └── service.py          # agent registry, run loop + metrics, memory curation, run history
├── memory/
│   └── memories.py         # four-type memory pipeline (iterate-on-merge + curated merge)
├── glean/
│   ├── oauth.py            # MCP dynamic-client OAuth2 (discovery, DCR, PKCE, refresh)
│   ├── client.py           # MCP client wrapper (one session per operation)
│   ├── token_cache.py      # bind-mounted token cache
│   └── transport.py
├── api/
│   ├── routes.py           # OpenAI-compatible FastAPI routes
│   ├── admin_routes.py     # /api/admin REST API (UI + agent lab + run feed)
│   ├── glean_routes.py     # /api/admin/glean + OAuth callback at /glean/callback
│   └── models.py           # request/response schemas
├── persistence/
│   └── mongo.py            # MongoDBSaver + MongoDBStore
├── static/                 # Pensive Console UI (MongoDB theme)
├── tests/                  # 176 tests: graph, memories, glean OAuth/client, agents, API
├── utils/
│   └── llm.py              # OpenAI Responses API (+ usage) + Voyage embeddings clients
├── agent-workspaces/       # per-agent sandboxes (gitignored, host-mounted)
└── scripts/
    ├── setup_memory_index.py
    └── glean_auth.py       # one-shot host-side Glean OAuth (loopback + fragment relay)
```

## Background

This project was invoked with the agent prompt in [PROMPT.md](PROMPT.md) — a Cursor session using LangGraph and MongoDB agent skills to build a minimal checkpoint + long-term memory demo with streamed workflow decisions and an OpenAI-compatible API.

Inspired by the memory architecture described in [Beyond the Chatbot: Escaping the "Groundhog Day" Loop with Agentic Memory](https://www.rosston.dev/blog/groundhog-day). 