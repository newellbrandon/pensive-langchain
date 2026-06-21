# Pensive — LangGraph + MongoDB Demo

A minimal AI agent demo using **LangGraph** for workflow orchestration and **MongoDB** for both:

- **Short-term memory** — LangGraph checkpoints (`MongoDBSaver`) persist conversation state per thread
- **Long-term memory** — LangGraph Store (`MongoDBStore`) with semantic vector search across sessions

The API is **OpenAI-compatible** so you can connect [OpenWebUI](https://docs.openwebui.com/) and chat with the agent while watching streamed workflow decisions.

## Architecture

### System overview

OpenWebUI sends chat requests to the FastAPI layer, which invokes a compiled LangGraph workflow. MongoDB holds short-term thread state (checkpointer) and long-term facts (store with vector search).

```mermaid
flowchart TD
  owui[OpenWebUI] -->|POST /v1/chat/completions| api[FastAPI]
  api -->|"ainvoke / astream"| lgWorkflow[Compiled LangGraph]
  lgWorkflow --> llm[Anthropic-compatible LLM]
  lgWorkflow -.->|"MongoDBSaver - every superstep"| mongoCP["checkpoints<br/>checkpoint_writes"]
  lgWorkflow -->|MongoDBStore| mongoStore[(agent_memories)]
  mongoStore -->|"vector search + save"| embed[Voyage embeddings]
```

### LangGraph workflow

Each chat message runs the four nodes below. OpenWebUI streams the **decision** from each step before the final answer — the second line in each node label matches what you see in the demo screenshots.

The checkpointer persists graph state after every superstep (not only these four nodes), which is why a single message produces multiple `checkpoints` and `checkpoint_writes` documents.

```mermaid
flowchart TD
  startNode((START)) --> classify["classify_intent<br/>Classify Intent"]

  classify -->|"intent: memory"| search["search_long_term_memory<br/>Search Long Term Memory"]
  classify -->|"intent: general"| generate["generate_response<br/>Generate Response"]

  search -->|store.search| agentMem[(agent_memories)]
  search --> generate

  generate -->|ChatAnthropic| llmNode{{LLM}}
  generate --> save["save_memory<br/>Save Memory"]

  save -->|LLM extract fact| llmNode
  save -->|"store.put - if durable fact"| agentMem
  save --> endNode((END))

  classify -.-> cpNote["checkpoints / checkpoint_writes"]
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

## Requirements

- Python 3.13+
- MongoDB with Vector Search (local 8.x or Atlas)
- Anthropic-compatible LLM endpoint (`LLM_URI`, `LLM_KEY`)
- Voyage embeddings via MongoDB AI (`LLM_EMBEDDING_KEY`) — custom `VoyageEmbeddings` client in `utils/llm.py`

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
| `USER_ID` | Optional fallback when not using OpenWebUI session headers |

When `LANGSMITH_TRACING=true` and a valid `LANGSMITH_API_KEY` is set, traces appear at [smith.langchain.com](https://smith.langchain.com) under the configured project. Each chat request produces a top-level run with tags `pensive` and `chat`, `metadata.user_id` from the OpenWebUI user, plus `chat_id` / `session_id` when forwarded, including nested LangGraph node spans, `ChatAnthropic` LLM calls, and `voyage_embed` embedding spans during memory search.

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

On the OpenWebUI server, enable session header forwarding so Pensive receives the logged-in user and chat context:

```bash
ENABLE_FORWARD_USER_INFO_HEADERS=True
```

See [OpenWebUI environment variables](https://docs.openwebui.com/getting-started/env-configuration/) for details. Without this, Pensive cannot resolve `user_id` from the OpenWebUI session (unless you set optional `USER_ID` in `.env` for local testing).

Enable streaming in chat settings for the best experience — workflow steps stream before the answer.

## Demo Results (OpenWebUI)

These screenshots show Pensive connected to OpenWebUI with streaming enabled. Each response includes the agent workflow steps before the answer.

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
├── Dockerfile              # container image
├── docker-compose.yml      # docker compose up -d --build
├── main.py                 # uvicorn entry
├── config.py               # environment configuration
├── agent/
│   ├── graph.py            # LangGraph workflow
│   ├── nodes.py            # classify, search, generate, save
│   └── state.py            # AgentState schema
├── api/
│   ├── routes.py           # OpenAI-compatible FastAPI routes
│   └── models.py           # request/response schemas
├── persistence/
│   └── mongo.py            # MongoDBSaver + MongoDBStore
├── utils/
│   └── llm.py              # ChatAnthropic helper
└── scripts/
    └── setup_memory_index.py
```

## Background

This project was invoked with the agent prompt in [PROMPT.md](PROMPT.md) — a Cursor session using LangGraph and MongoDB agent skills to build a minimal checkpoint + long-term memory demo with streamed workflow decisions and an OpenAI-compatible API.

Inspired by the memory architecture described in [Beyond the Chatbot: Escaping the "Groundhog Day" Loop with Agentic Memory](https://www.rosston.dev/blog/groundhog-day). 