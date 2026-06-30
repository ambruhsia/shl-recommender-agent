---
title: SHL Recommender
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# SHL Assessment Recommender

A conversational agent that takes a hiring manager from a vague intent — *"I need to hire a Java developer"* — to a grounded shortlist of SHL assessments through multi-turn dialogue.

**Live API:** `https://shl-recommender-production-e0d9.up.railway.app`
> The hosted instance may go offline. To run locally with your own API keys, see [Run locally](#run-locally).

---

## How it works

```
POST /chat  (full conversation history, stateless)
       │
       ▼
 retrieve_node   ← FAISS semantic search over 377 SHL products
       │            + framework pinning (Spring, Docker, SQL, AWS…)
       │            + seniority boosting (Senior → Advanced Level tests)
       ▼
  agent_node    ← Gemini 2.5 Flash Lite reasons over top-15 candidates
       │            10 explicit directives: clarify / recommend / refine / compare / refuse
       ▼
 format_node    ← verifies every URL/name against real catalog
                   drops hallucinations, enforces 10-item cap, schema
```

Every response is a structured JSON object. The service stores no session state — the full conversation history is sent on each call.

---

## API

### `GET /health`
```bash
curl https://shl-recommender-production-e0d9.up.railway.app/health
# {"status":"ok"}
```

### `POST /chat`

**Request**
```json
{
  "messages": [
    {"role": "user",      "content": "Hiring senior Java developers with Spring Boot and SQL"},
    {"role": "assistant", "content": "{\"reply\":\"...\",\"recommendations\":[...],\"end_of_conversation\":false}"},
    {"role": "user",      "content": "Also add a personality test"}
  ]
}
```

**Response**
```json
{
  "reply": "Added OPQ32r for behavioural fit. Here is your updated shortlist:",
  "recommendations": [
    {"name": "Core Java (Advanced Level) (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "Spring (New)",                     "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "SQL (New)",                        "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r",                           "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

`recommendations` is `[]` while the agent is clarifying or refusing. `end_of_conversation` is `true` when the user confirms the final shortlist or the 8-turn cap is reached.

---

## Agent behaviours

| Behaviour | Example |
|---|---|
| **Clarify** vague queries | *"I need an assessment"* → asks for role/level |
| **Recommend** once context is sufficient | *"Mid-level Java + SQL"* → shortlist immediately |
| **Refine** mid-conversation | *"Add a personality test"* → appends, never resets |
| **Compare** two assessments | *"OPQ vs DSI?"* → grounded answer from catalog descriptions |
| **Refuse** out-of-scope | Legal questions, prompt injection, competitor tools |

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI | Async, native OpenAPI schema, Pydantic validation |
| Agent orchestration | LangGraph | Explicit typed state machine; auditable node transitions |
| LLM | Gemini 2.5 Flash Lite | Free tier, <5s latency, strong JSON instruction-following |
| Embeddings | `all-MiniLM-L6-v2` | CPU-only, <100 MB RAM, fast synchronous inference |
| Vector index | FAISS `IndexFlatIP` | In-process, zero network overhead, no persistence needed |
| Deployment | Docker on Railway | Locked Python version, model baked into image, no cold-start downloads |

---

## Run locally

### 1. Get a Gemini API key
Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey) → **Create API key** — free, no credit card required.

### 2. Clone and configure
```bash
git clone https://github.com/ambruhsia/shl-recommender-agent.git
cd shl-recommender-agent

# Create a .env file with your key
echo "GEMINI_API_KEY=your_key_here" > .env
```

Optionally add an OpenRouter key too (only needed for running tests against other models):
```
OPENROUTER_API_KEY=your_openrouter_key_here
```

### 3a. Run with Docker (recommended — no Python setup needed)
```bash
docker build -t shl-recommender .
docker run -p 10000:10000 --env-file .env shl-recommender
# open http://localhost:10000/docs
```

### 3b. Run without Docker
```bash
pip install -r requirements.txt
python scripts/build_catalog.py   # builds FAISS index + saves ST model locally (run once)
uvicorn app.main:app --host 0.0.0.0 --port 8000
# open http://localhost:8000/docs
```

### 4. Test it
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"I need to assess senior Java developers with Spring Boot and SQL"}]}'
```

---

## Run stress tests

```bash
# Default: Gemini 2.5 Flash Lite (direct)
python tests/stress_test.py

# Via OpenRouter (any model)
SHL_TEST_PROVIDER=openrouter SHL_TEST_MODEL="meta-llama/llama-3.1-8b-instruct:free" python tests/stress_test.py
```

10 scenarios covering: schema compliance, Recall@10, shortlist persistence, seniority detection, framework pinning, off-topic refusal, pivot handling, comparison grounding, EOC flag, and turn-cap enforcement.

| Model | Score |
|---|---|
| Gemini 2.5 Flash Lite | 10 / 10 |
| Llama 3.1 8B free (OpenRouter) | 9 / 10 |

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes (default) | Google AI Studio key |
| `OPENROUTER_API_KEY` | For OpenRouter tests | OpenRouter key |
| `SHL_TEST_PROVIDER` | No | `gemini` (default) / `openrouter` / `openai` |
| `SHL_TEST_MODEL` | No | Model ID override for tests |
| `KMP_DUPLICATE_LIB_OK` | Set to `TRUE` | Suppresses Intel/LLVM OpenMP conflict warning on Windows |
