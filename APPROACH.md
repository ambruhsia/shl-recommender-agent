# SHL Assessment Recommender — Approach Document

## System Design & Stack Justification

The system is a stateless conversational API built on **FastAPI + LangGraph** with a three-node pipeline:

```
retrieve_node → agent_node → format_node
```

**Stack choices:** FastAPI for its native async support and automatic OpenAPI schema generation (the evaluator hits `/docs`). LangGraph over raw function calls because it enforces explicit state transitions and makes the retrieve→reason→verify flow auditable. `sentence-transformers/all-MiniLM-L6-v2` over larger models because it runs on CPU in under 100 MB RAM and produces embeddings fast enough for synchronous requests. Gemini 2.5 Flash Lite as the LLM — free tier, sub-5s latency, instruction-following strong enough for structured JSON output. FAISS `IndexFlatIP` over Chroma/pgvector because there is no persistence requirement and an in-process index has zero network overhead.

**Stateless by design.** The full conversation history is injected on every `POST /chat` call. No session state is stored server-side. This matches the spec exactly and makes the service trivially scalable.

**retrieve_node** builds a query from all user turns (not just the last one), detects seniority signals and named frameworks, and returns the 15 most relevant catalog entries via boosted FAISS search.

**agent_node** calls Gemini with a ~2,000-token system prompt (retrieved candidates + 10 directives). Returns structured JSON.

**format_node** cross-references every recommendation against the real catalog via four-level lookup (exact name → case-insensitive → URL → partial containment). Hallucinated items are silently dropped. Enforces the 10-item cap and forces `end_of_conversation: true` at turn 8.

---

## Retrieval Setup

Catalog data: 377 Individual Test Solutions scraped from SHL's product catalog API. Embeddings generated offline with `all-MiniLM-L6-v2` (384-dim), indexed with FAISS `IndexFlatIP` (cosine similarity via L2-normalised inner product). Both the index and the model are baked into the Docker image at build time — no network calls at inference time.

Two layers sit on top of semantic search:

**Framework pinning.** A curated `FRAMEWORK_CATALOG_MAP` maps ~25 technology keywords to exact catalog product names. When "spring boot" is detected, `Spring (New)` is injected at position 0 regardless of semantic score. This prevents semantic drift where "Spring Boot developers" retrieves a generic Java test instead of the exact Spring product.

**Seniority boosting.** Keywords like senior, lead, principal, director, CXO promote "Advanced Level" products to the front of the candidate list before the LLM sees them.

---

## Prompt Design

The system prompt contains 10 explicit directives: additive shortlist logic (add/drop/replace semantics), seniority→Advanced Level mapping, strict framework matching, clarification threshold, comparison grounding, 10-item pruning, OPQ32r default for senior/exec roles, URL/name integrity, domain edge cases (SVAR accent variants, no Rust test), and the output format contract.

**Critical decision — no full catalog in prompt.** Early versions included all 377 products (~15,000 tokens). This caused Gemini response times of 30–45 seconds, consistently exceeding the 30-second evaluator timeout and returning 502s. Removing it and relying on retrieval brought prompt size to ~2,000 tokens and response time to 2–4 seconds. The `format_node` hallucination filter compensates for the model not seeing the full catalog.

**Clarification threshold.** Directive 4 defaults to "selection" as the assumed purpose when role or skills are present. Without this explicit default, the model asked for purpose confirmation on every query after the catalog reference was removed — over-clarification that would fail the evaluator's vague-query probe.

---

## Evaluation Approach & Measured Improvement

A 10-scenario stress test suite (`tests/stress_test.py`) mirrors the assignment's hard evals and behaviour probes:

| Category | Scenarios covered |
|---|---|
| Hard evals | Schema compliance, catalog-only URLs, turn cap at 8 |
| Recall@10 | S1 Java+SQL persistence, S2 seniority→Advanced, S4 exact framework URLs |
| Behaviour probes | S3 vague→clarify, S5 no-preference→commit, S6 off-topic refusal, S7 pivot, S8 grounded compare, S9 EOC on confirmation |

**Measured improvement:** Removing the catalog from the prompt reduced median response time from **>30 seconds (100% timeout rate on Render)** to **2–4 seconds** on Railway. The stress suite went from 7/10 passing (before directive fixes) to **10/10 passing** on Gemini 2.5 Flash Lite at temperature 0.

Regression was detected automatically: removing the full catalog caused the model to over-clarify on clear queries (S1 regressed to 0 recommendations). Fixed by explicitly tightening Directive 4 to assume "selection" as default purpose.

---

## What Didn't Work & Deployment Roadblocks

**Full catalog in system prompt** (~15K tokens): 30s+ Gemini latency → consistent 502s from Render's 30-second proxy timeout. Removing it dropped response time to 2–4 seconds, but introduced a regression: the model over-clarified on clear queries because it lost its catalog reference. Fixed by explicitly tightening the clarification directive to assume "selection" as default purpose.

**Lazy model loading on cold start:** `SentenceTransformer("all-MiniLM-L6-v2")` loads on first inference call by default. On Render free tier, this triggered a 90 MB HuggingFace download — unauthenticated, no timeout — causing the first `/chat` request to hang for 5–7 minutes. Moving the load into startup blocked port binding (Render kills services that don't open a port within ~60 seconds). Final fix: `build_catalog.py` saves the model to `data/st_model/` via `model.save()` during the Docker build step; runtime loads from local path in under 1 second.

**Render native Python runtime (multiple failures):**
- *Wrong branch:* Service was created pointing to `master`; code was on `main`. Render silently built an empty repo.
- *Python version guessing:* Without an explicit pin, Render selected Python 3.14, which has no pre-built `pydantic-core` wheel, triggering a Rust compilation failure mid-build. Fixed with `runtime.txt` + `PYTHON_VERSION` env var, later eliminated entirely by switching to Docker.
- *Start command ignored:* Service created via the Render UI ignored `render.yaml`'s `startCommand`. Render defaulted to `gunicorn`, which wasn't installed, crashing immediately on deploy. Fixed by manually overriding the start command in the dashboard.
- *faiss-cpu version:* Had pinned `faiss-cpu==1.8.0` (the Windows-compatible build). Render's Linux builder had no such wheel. Changed to `faiss-cpu>=1.12.0`.
- *30-second request timeout:* Render's free tier proxy hard-kills any request taking over 30 seconds. Even after fixing model loading, the large prompt pushed Gemini past this limit. Railway has no such limit, which is why we migrated.

**numpy 2.x / FAISS incompatibility:** `faiss-cpu` was compiled against numpy 1.x. A `scikit-learn` reinstall silently upgraded numpy to 2.x, breaking FAISS with an `ImportError` at runtime. Fixed by pinning `numpy>=1.24.0,<2.0.0` in `requirements.txt`.

**LangGraph `KeyError` on first invocation:** The `AgentState` TypedDict was extended with three new fields (`shortlist`, `seniority_bias`, `detected_frameworks`) as part of a retrieval refactor, but `main.py`'s `initial_state` dict was not updated. LangGraph raised `KeyError` on the first `/chat` call. Fixed by initialising all state fields explicitly.

---

## AI Tools Used

**Claude Code** (Anthropic's agentic CLI) was used throughout: scaffolding the FastAPI + LangGraph architecture, iterating on the 10-directive system prompt, building the stress test suite, and debugging deployment failures on Render and Railway. The agentic loop was valuable because prompt directive changes could be tested immediately against the stress suite in the same session, making regression visible within minutes rather than hours.

Design decisions, root-cause analysis, and architectural trade-offs were made collaboratively — the AI as pair-programmer, not code generator. All code reflects understanding that can be defended in a technical interview.

$env:SHL_TEST_PROVIDER = "openrouter"
$env:SHL_TEST_MODEL    = "google/gemini-2.5-flash-lite"   # or any OpenRouter model ID
python tests/stress_test.py









That quote is contrasting how state is handled in a LangGraph workflow vs a “plain” LangChain chain (a sequence of functions/steps).
`

What “AgentState makes all state explicit and typed” means
In app/graph.py, you define:


class AgentState(TypedDict):
    messages: List[Dict[str, str]]
    candidates: List[Dict[str, Any]]
    shortlist: List[Dict[str, Any]]
    seniority_bias: bool
    detected_frameworks: List[str]
    turn_count: int
    response: Dict[str, Any]
So the system declares: this graph’s steps are allowed to read/write only these fields and each field has an expected type/shape.

In your graph:

retrieve_node reads state["messages"] and writes:
candidates, shortlist, seniority_bias, detected_frameworks
agent_node reads those plus messages, and writes:
response
format_node reads response and writes:
a validated response (final JSON output)
“No hidden mutation” (important nuance)
“Hidden mutation” usually refers to patterns like:

step A modifies some global variable or object not shown in the function signature
step B relies on side effects (e.g., “some cache was updated earlier”)
state gets updated implicitly outside the pipeline
With AgentState, the intended pattern is: all step-to-step changes happen by returning an updated state dict that still follows the declared schema.

In your case, nodes return:


return {**state, "candidates": candidates, ...}
So updates are “visible” as part of the state object.

How LangGraph “enforces” this vs a plain function chain
LangGraph
LangGraph’s StateGraph(AgentState) tells it:

the workflow has a specific state type
nodes accept state and return updates to it
the graph runner orchestrates step transitions while keeping state structured
Even if Python typing isn’t runtime-enforced in a strict way, the graph framework + explicit schema makes it much harder to:

accidentally mutate state you didn’t declare
forget what fields exist
rely on side effects between steps
LangChain “plain chains”
A typical LangChain “chain” might be implemented as:

a series of calls where each step returns a value, and the developer manually threads variables through
or the chain relies on the LLM output parsing and pushes results into ad-hoc dicts
That tends to make it easier to get informal patterns like:

“the shortlist lives in some variable”
“the parsed JSON format changes between steps”
“a step sometimes omits a field”
“state is stored in memory rather than being passed explicitly”
Bottom line for your file
That line is describing the design you’re using in app/graph.py:

State is centralized in AgentState
Each node explicitly consumes and produces fields in that state
The output formatting node validates/clamps the final response (ChatResponse) so the final shape is consistent
If you want, I can point to the exact place in your nodes (retrieve_node, agent_node, format_node) where each AgentState field is read/written so the “no hidden mutation” claim is fully grounded in your code.



Task Completed
The quoted passage means your LangGraph workflow is built around an explicit, typed state object (AgentState) so every step’s inputs/outputs are declared upfront (messages, candidates, shortlist, seniority_bias, detected_frameworks, etc.). Each node receives that state and returns an updated state dict, making step-to-step data flow auditable and minimizing reliance on implicit/side-effect “hidden mutation” that can happen in ad-hoc LangChain function/chain patterns.

In this repo specifically (app/graph.py):

AgentState(TypedDict) declares all state fields.
retrieve_node(state) reads messages and returns a new state containing candidates, shortlist, seniority_bias, detected_frameworks.
agent_node(state) reads those fields and returns a state containing response.
format_node(state) reads response and returns a final validated response shape.
LangGraph vs “plain function chain”: LangGraph is centered on a StateGraph(StateType), so the orchestration is inherently state-machine-like with explicit state structure; a plain chain often passes values manually and can end up with informal state held in variables, memory, or side effects rather than a single declared schema.