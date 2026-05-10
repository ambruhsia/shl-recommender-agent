# SHL Assessment Recommender

Conversational agent that guides hiring managers from a vague intent to a grounded SHL assessment shortlist.

## Setup

```bash
pip install -r requirements.txt
```

Add your Gemini API key to `.env`:
```
GEMINI_API_KEY=your_key_here
```

Build the FAISS vector index (run once):
```bash
python scripts/build_catalog.py
```

Start the server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Endpoints

### GET /health
```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### POST /chat
Send the full conversation history on every request (stateless).

```bash
# Turn 1 — vague query (should trigger clarification)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"I need an assessment"}]}'

# Turn 2 — with context
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role":"user","content":"I need an assessment"},
      {"role":"assistant","content":"{\"reply\":\"What role and what are you assessing for?\",\"recommendations\":[],\"end_of_conversation\":false}"},
      {"role":"user","content":"Hiring senior Java developers, want to test Spring Boot and SQL"}
    ]
  }'
```

## Response Schema

```json
{
  "reply": "string",
  "recommendations": [
    {"name": "...", "url": "...", "test_type": "..."}
  ],
  "end_of_conversation": false
}
```

## Deploy to Render

- **Build command**: `pip install -r requirements.txt && python scripts/build_catalog.py`
- **Start command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Environment variable**: `GEMINI_API_KEY` set in Render dashboard (never commit the `.env` file)
