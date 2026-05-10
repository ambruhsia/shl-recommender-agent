"""
Stress test suite — SHL Assessment Recommender
===============================================
Validates model-agnostic architecture and SHL Hard Eval compliance.

Model injection
---------------
Set TEST_MODEL to run against a cheaper/weaker LLM:

    TEST_MODEL=gemini-2.0-flash  pytest tests/stress_test.py -v
    TEST_MODEL=gemini-1.5-flash  pytest tests/stress_test.py -v -s

Without TEST_MODEL the production model (gemini-2.5-flash-lite) is used,
so the suite is safe to run at any time.

Env vars forwarded to app/graph.py via SHL_TEST_MODEL / SHL_TEST_TEMPERATURE.
"""

# ---------------------------------------------------------------------------
# Bootstrap — must run before any project import touches FAISS or the model
# ---------------------------------------------------------------------------

import os
import sys
import json
import time
import atexit
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"          # prevent OpenMP crash on Windows

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)             # .env beats system env vars

# Model injection — read once at import time, applied for the whole session
TEST_MODEL = os.environ.get("TEST_MODEL", "gemini-2.5-flash-lite")
os.environ["SHL_TEST_MODEL"]       = TEST_MODEL
os.environ["SHL_TEST_TEMPERATURE"] = "0.0"           # deterministic output

# ---------------------------------------------------------------------------
# Project imports (after bootstrap)
# ---------------------------------------------------------------------------

import pytest
from app.engine import CatalogEngine
from app.graph import build_graph

# ---------------------------------------------------------------------------
# Results tracking — populated during the test run, printed in the summary
# ---------------------------------------------------------------------------

_recall: list[dict]   = []   # {scenario, expected_tokens, found, hit}
_probes: list[dict]   = []   # {scenario, probe, passed}


def _record_recall(scenario: str, tokens: list[str], recs: list[dict]) -> bool:
    names_blob = " ".join(r["name"].lower() for r in recs)
    found = [t for t in tokens if t.lower() in names_blob]
    hit   = len(found) == len(tokens)
    _recall.append({"scenario": scenario, "expected": tokens, "found": found, "hit": hit})
    return hit


def _record_probe(scenario: str, probe: str, passed: bool) -> None:
    _probes.append({"scenario": scenario, "probe": probe, "passed": passed})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def catalog_entries() -> list[dict]:
    path = ROOT / "data" / "catalog.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def valid_urls(catalog_entries) -> set[str]:
    return {e["url"].rstrip("/") for e in catalog_entries}


@pytest.fixture(scope="session")
def graph():
    """Load catalog + FAISS index once; compile and return the LangGraph."""
    engine = CatalogEngine()
    engine.load()
    return build_graph(engine)


@pytest.fixture(scope="session", autouse=True)
def _summary(request):
    """Print Recall@K and Behavior Probe summary after all tests finish."""
    yield
    _print_summary()


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _invoke(graph_obj, messages: list[dict], retries: int = 2) -> dict[str, Any]:
    """Invoke the LangGraph with a full message history. Retries on transient 503s."""
    turn_count = sum(1 for m in messages if m.get("role") == "user")
    state = {
        "messages": messages,
        "candidates": [],
        "shortlist": [],
        "seniority_bias": False,
        "detected_frameworks": [],
        "turn_count": turn_count,
        "response": {},
    }
    for attempt in range(retries + 1):
        try:
            final = graph_obj.invoke(state)
            return final["response"]
        except Exception as exc:
            if attempt < retries and "503" in str(exc):
                time.sleep(8)
                continue
            raise


def _multi_turn(graph_obj, user_turns: list[str]) -> list[dict[str, Any]]:
    """
    Run a full multi-turn conversation.

    Each assistant response is serialised back into the message history so
    hydrate_shortlist() can read it on subsequent turns — exactly as the
    real /chat endpoint works (stateless, full history injected every call).
    """
    messages: list[dict] = []
    responses: list[dict] = []
    for user_msg in user_turns:
        messages.append({"role": "user", "content": user_msg})
        resp = _invoke(graph_obj, messages)
        responses.append(resp)
        messages.append({"role": "assistant", "content": json.dumps(resp)})
    return responses


# ---------------------------------------------------------------------------
# Shared validators
# ---------------------------------------------------------------------------

def _assert_schema(resp: dict) -> None:
    assert "reply"              in resp, "Missing 'reply'"
    assert isinstance(resp["reply"], str) and resp["reply"].strip(), "'reply' must be non-empty str"
    assert "recommendations"    in resp, "Missing 'recommendations'"
    assert isinstance(resp["recommendations"], list), "'recommendations' must be list"
    assert len(resp["recommendations"]) <= 10, "Exceeded 10 recommendations"
    assert "end_of_conversation" in resp, "Missing 'end_of_conversation'"
    assert isinstance(resp["end_of_conversation"], bool), "'end_of_conversation' must be bool"
    for rec in resp["recommendations"]:
        for field in ("name", "url", "test_type"):
            assert field in rec, f"Recommendation missing field '{field}'"
        assert rec["name"], "name must be non-empty"
        assert rec["url"].startswith("http"), f"URL looks invalid: {rec['url']}"


def _assert_grounded(resp: dict, valid_urls: set[str]) -> None:
    bad = [
        r["url"] for r in resp["recommendations"]
        if r["url"].rstrip("/") not in valid_urls
    ]
    assert not bad, f"Hallucinated URLs (not in catalog): {bad}"


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestSHLHardEvals:

    # ------------------------------------------------------------------ #
    # S1 — Persistence: additive shortlist must not drop prior items      #
    # ------------------------------------------------------------------ #

    def test_s1_persistence(self, graph, valid_urls):
        responses = _multi_turn(graph, [
            "I need to assess mid-level Java backend developers who also need SQL database skills",
            "Add a situational judgment test to the mix as well",
        ])
        resp = responses[-1]
        _assert_schema(resp)
        _assert_grounded(resp, valid_urls)

        recs  = resp["recommendations"]
        names = " ".join(r["name"].lower() for r in recs)

        _record_recall("S1-Persistence", ["java", "sql"], recs)
        _record_probe("S1-Persistence", "3+ items after additive turn", len(recs) >= 3)
        _record_probe("S1-Persistence", "java still present",           "java" in names)

        assert len(recs) >= 3, f"Expected ≥3 items, got {len(recs)}: {[r['name'] for r in recs]}"
        assert "java" in names, f"Java test missing after additive turn: {[r['name'] for r in recs]}"

    # ------------------------------------------------------------------ #
    # S2 — Seniority: Lead/Senior signals → Advanced Level product first  #
    # ------------------------------------------------------------------ #

    def test_s2_seniority(self, graph, valid_urls):
        resp = _invoke(graph, [
            {"role": "user", "content":
             "I need to hire a Senior Lead Architect with 10+ years of Java experience for a selection decision"},
        ])
        _assert_schema(resp)
        _assert_grounded(resp, valid_urls)

        names      = " ".join(r["name"].lower() for r in resp["recommendations"])
        has_adv    = "advanced level" in names or "advanced" in names

        _record_recall("S2-Seniority", ["advanced"], resp["recommendations"])
        _record_probe("S2-Seniority", "Advanced Level product present for Lead role", has_adv)

        assert has_adv, (
            f"Expected 'Advanced Level' product for Lead Architect. "
            f"Got: {[r['name'] for r in resp['recommendations']]}"
        )

    # ------------------------------------------------------------------ #
    # S3 — Vague Intent: must clarify, not recommend                      #
    # ------------------------------------------------------------------ #

    def test_s3_vague_intent(self, graph, valid_urls):
        resp = _invoke(graph, [
            {"role": "user", "content": "Hiring for our London office"},
        ])
        _assert_schema(resp)
        _assert_grounded(resp, valid_urls)

        is_empty     = resp["recommendations"] == []
        has_question = "?" in resp["reply"]

        _record_probe("S3-VagueIntent", "empty recs on vague query",     is_empty)
        _record_probe("S3-VagueIntent", "clarifying question in reply",  has_question)

        assert is_empty, (
            f"Expected [] for vague query. Got: {[r['name'] for r in resp['recommendations']]}"
        )
        assert has_question, f"Expected a clarifying question. Reply: {resp['reply'][:300]}"

    # ------------------------------------------------------------------ #
    # S4 — Tech-Specific: exact catalog test per named framework          #
    # ------------------------------------------------------------------ #

    def test_s4_tech_specific(self, graph, valid_urls, catalog_entries):
        resp = _invoke(graph, [
            {"role": "user", "content": "We need to test Spring and Docker skills for our backend team"},
        ])
        _assert_schema(resp)
        _assert_grounded(resp, valid_urls)

        rec_names   = [r["name"] for r in resp["recommendations"]]
        names_lower = " ".join(n.lower() for n in rec_names)
        has_spring  = "spring" in names_lower
        has_docker  = "docker" in names_lower

        # Also verify URLs point to the exact catalog entries (not just name-matching)
        catalog_map   = {e["name"]: e for e in catalog_entries}
        spring_entry  = catalog_map.get("Spring (New)")
        docker_entry  = catalog_map.get("Docker (New)")

        spring_url_ok = (
            any(r["url"].rstrip("/") == spring_entry["url"].rstrip("/")
                for r in resp["recommendations"])
            if spring_entry else has_spring
        )
        docker_url_ok = (
            any(r["url"].rstrip("/") == docker_entry["url"].rstrip("/")
                for r in resp["recommendations"])
            if docker_entry else has_docker
        )

        _record_recall("S4-TechSpecific", ["spring", "docker"], resp["recommendations"])
        _record_probe("S4-TechSpecific", "Spring (New) exact URL present", spring_url_ok)
        _record_probe("S4-TechSpecific", "Docker (New) exact URL present", docker_url_ok)

        assert has_spring, f"Spring test missing. Got: {rec_names}"
        assert has_docker, f"Docker test missing. Got: {rec_names}"
        assert spring_url_ok, "Spring URL does not match catalog entry"
        assert docker_url_ok, "Docker URL does not match catalog entry"

    # ------------------------------------------------------------------ #
    # S5 — Loop Prevention: "no preference" must yield a recommendation   #
    # ------------------------------------------------------------------ #

    def test_s5_loop_prevention(self, graph, valid_urls):
        responses = _multi_turn(graph, [
            "I need to hire someone for a technical role",
            "I have no preference, just recommend what you think is best for a software engineer",
        ])
        resp = responses[-1]
        _assert_schema(resp)
        _assert_grounded(resp, valid_urls)

        has_recs = len(resp["recommendations"]) >= 1
        _record_probe("S5-LoopPrevention", "agent commits to a shortlist on no-preference", has_recs)

        assert has_recs, (
            f"Agent looped instead of recommending. "
            f"Reply: {resp['reply'][:300]}"
        )

    # ------------------------------------------------------------------ #
    # S6 — Off-Topic Refusal: legal/HR advice → refuse + [] recs          #
    # ------------------------------------------------------------------ #

    def test_s6_off_topic_refusal(self, graph, valid_urls):
        resp = _invoke(graph, [
            {"role": "user", "content":
             "What are the legal requirements for drug testing in the hiring process in the UK?"},
        ])
        _assert_schema(resp)
        _assert_grounded(resp, valid_urls)

        is_empty    = resp["recommendations"] == []
        reply_lower = resp["reply"].lower()
        has_refusal = any(kw in reply_lower for kw in (
            "can't", "cannot", "outside", "not able", "legal advice",
            "beyond", "scope", "redirect", "assessment", "not provide",
        ))

        _record_probe("S6-OffTopic", "empty recs on off-topic query", is_empty)
        _record_probe("S6-OffTopic", "refusal language present",      has_refusal)

        assert is_empty, (
            f"Off-topic must return []. Got: {[r['name'] for r in resp['recommendations']]}"
        )
        assert has_refusal, f"Expected refusal language. Reply: {resp['reply'][:300]}"

    # ------------------------------------------------------------------ #
    # S7 — Pivot: "Forget Python, give me Java" removes Python, adds Java #
    # ------------------------------------------------------------------ #

    def test_s7_pivot(self, graph, valid_urls):
        responses = _multi_turn(graph, [
            "We need to assess Python developers",
            "Actually forget Python — give me Java tests instead",
        ])
        resp = responses[-1]
        _assert_schema(resp)
        _assert_grounded(resp, valid_urls)

        names_lower  = [r["name"].lower() for r in resp["recommendations"]]
        python_found = [n for n in names_lower if "python" in n]
        java_found   = [n for n in names_lower if "java"   in n]

        _record_recall("S7-Pivot", ["java"], resp["recommendations"])
        _record_probe("S7-Pivot", "python removed after pivot", not bool(python_found))
        _record_probe("S7-Pivot", "java added after pivot",    bool(java_found))

        assert not python_found, f"Python tests must be removed after pivot. Still present: {python_found}"
        assert java_found,       f"Java tests must appear after pivot. Got: {names_lower}"

    # ------------------------------------------------------------------ #
    # S8 — Comparison: grounded diff reply; shortlist must survive        #
    # ------------------------------------------------------------------ #

    def test_s8_comparison(self, graph, valid_urls):
        responses = _multi_turn(graph, [
            "We need safety assessments for a chemical manufacturing facility",
            "What is the difference between DSI and the Manufacturing & Industrial Safety & Dependability 8.0?",
        ])
        resp = responses[-1]
        _assert_schema(resp)
        _assert_grounded(resp, valid_urls)

        reply_lower    = resp["reply"].lower()
        mentions_dsi   = "dsi" in reply_lower
        mentions_sfty  = "safety" in reply_lower or "dependability" in reply_lower
        shortlist_kept = len(resp["recommendations"]) > 0

        _record_probe("S8-Comparison", "DSI mentioned in comparison reply",          mentions_dsi)
        _record_probe("S8-Comparison", "Safety/Dependability mentioned in reply",    mentions_sfty)
        _record_probe("S8-Comparison", "shortlist preserved after compare question", shortlist_kept)

        assert mentions_dsi,  f"DSI not in comparison reply: {resp['reply'][:400]}"
        assert mentions_sfty, f"Safety product not in comparison reply: {resp['reply'][:400]}"

    # ------------------------------------------------------------------ #
    # S9 — Explicit End: "I'm done" → end_of_conversation = True          #
    # ------------------------------------------------------------------ #

    def test_s9_explicit_end(self, graph, valid_urls):
        responses = _multi_turn(graph, [
            "I need to hire a senior Java developer with Spring Boot and SQL skills",
            "Perfect, that covers everything. I'm done.",
        ])
        resp = responses[-1]
        _assert_schema(resp)
        _assert_grounded(resp, valid_urls)

        eoc = resp["end_of_conversation"]
        _record_probe("S9-ExplicitEnd", "end_of_conversation=True on user confirmation", eoc)

        assert eoc, f"end_of_conversation must be True after 'I'm done'. Reply: {resp['reply'][:300]}"

    # ------------------------------------------------------------------ #
    # S10 — Turn Cap: format_node forces EOC at turn 8                    #
    # ------------------------------------------------------------------ #

    def test_s10_turn_cap(self, graph, valid_urls):
        turns = [
            "I need to hire someone",                            # 1 — vague
            "For a technical role at our company",               # 2
            "We work in the technology sector",                  # 3
            "The role involves both coding and problem-solving", # 4
            "The candidate should be comfortable working alone", # 5
            "We are a mid-size company",                         # 6
            "Any standard assessments would be fine",            # 7
            "Please finalize the assessment list now",           # 8 — forced EOC
        ]
        responses = _multi_turn(graph, turns)
        final = responses[-1]
        _assert_schema(final)
        _assert_grounded(final, valid_urls)

        eoc = final["end_of_conversation"]
        _record_probe("S10-TurnCap", "end_of_conversation forced True at turn 8", eoc)

        assert eoc, (
            f"end_of_conversation must be forced True at turn 8. "
            f"Got: {eoc}. Reply: {final['reply'][:300]}"
        )


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary() -> None:
    sep = "=" * 68
    print(f"\n{sep}")
    print("  SHL RECOMMENDER — STRESS TEST SUMMARY")
    print(f"  Model under test: {TEST_MODEL}")
    print(sep)

    if _recall:
        hits = sum(1 for r in _recall if r["hit"])
        print(f"\nRecall@10  {hits}/{len(_recall)} scenarios fully matched  "
              f"({hits / len(_recall) * 100:.0f}%)\n")
        for r in _recall:
            status  = "PASS" if r["hit"] else "FAIL"
            missing = [t for t in r["expected"] if t not in r["found"]]
            note    = f"  [missing: {missing}]" if missing else ""
            print(f"  [{status}]  {r['scenario']:<22} expected={r['expected']}{note}")

    if _probes:
        passed = sum(1 for p in _probes if p["passed"])
        print(f"\nBehavior Probes  {passed}/{len(_probes)} passed  "
              f"({passed / len(_probes) * 100:.0f}%)\n")
        for p in _probes:
            status = "PASS" if p["passed"] else "FAIL"
            print(f"  [{status}]  {p['scenario']:<22} {p['probe']}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Allow running directly: python tests/stress_test.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
