import os

# Must be set before any FAISS import to prevent OpenMP crash on Windows/macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import ChatRequest, ChatResponse
from app.engine import CatalogEngine

load_dotenv(override=True)  # .env takes precedence over system env vars

_engine: CatalogEngine | None = None
_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load catalog and compile LangGraph at startup."""
    global _engine, _graph

    _engine = CatalogEngine()
    _engine.load()

    # Deferred import: engine must be loaded before graph.py is evaluated
    from app.graph import build_graph
    _graph = build_graph(_engine)

    print("[main] Service ready.")
    yield
    # No cleanup required for FAISS or sentence-transformers


app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for selecting SHL individual assessment solutions.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if _graph is None:
        raise HTTPException(status_code=503, detail="Service is initializing, please retry.")

    turn_count = sum(1 for m in request.messages if m.role == "user")

    initial_state = {
        "messages": [m.model_dump() for m in request.messages],
        "candidates": [],
        "shortlist": [],
        "seniority_bias": False,
        "detected_frameworks": [],
        "detected_job_level": None,
        "turn_count": turn_count,
        "response": {},
    }

    try:
        final_state = _graph.invoke(initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")

    response_dict = final_state.get("response", {})
    return ChatResponse(**response_dict)
