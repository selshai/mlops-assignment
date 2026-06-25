"""LangGraph agent: text-to-SQL with verify+revise loop.

Graph shape:

    START -> attach_schema -> generate_sql -> execute --(exec ok)---------> END
                                                 ^        |
                                                 |   (exec failed)
                                                 |        v
                                                 |      verify --(ok)------> END
                                                 |        |
                                                 +--revise<-(not ok)

Verify runs only when execution FAILS (Phase 6 latency lever): a query that
executed cleanly is accepted without an extra LLM call. Loop capped at
MAX_ITERATIONS total generate/revise calls.

The execute node and the graph wiring are provided. `generate_sql_node` is
filled in as a worked example; you implement `verify`, `revise`, and the
conditional router following the same shape.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agent import prompts
from agent.execution import ExecutionResult, execute_sql
from agent.schema import render_schema

# Total generate + revise calls before the loop is forced to stop.
# 3-5 is a reasonable range; tune it as part of Phase 3.
MAX_ITERATIONS = 3

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
# vLLM ignores the key, but a hosted OpenAI-compatible provider needs a real one.
# Lets you point the agent at e.g. OpenAI while iterating without a running vLLM.
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")


@dataclass
class AgentState:
    """State threaded through the graph. Extend with fields you need."""

    question: str
    db_id: str
    schema: str = ""
    sql: str = ""
    execution: ExecutionResult | None = None
    verify_ok: bool = False
    verify_issue: str = ""
    iteration: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


from functools import lru_cache


@lru_cache(maxsize=1)
def llm() -> ChatOpenAI:
    """Shared chat client pointed at VLLM_BASE_URL (your local vLLM by default).

    Built ONCE and reused across all requests/nodes. Constructing a fresh
    ChatOpenAI per call (the original behaviour) spun up a new openai/httpx
    client each time, so under sustained load the agent churned sockets/FDs
    and threw resource-exhaustion 500s. A single client shares one async
    connection pool. ChatOpenAI is safe to reuse across concurrent ainvoke
    calls.
    """
    return ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0.0,
    )


# ---- Nodes ------------------------------------------------------------

def _attach_schema(state: AgentState) -> dict:
    """Provided. Render the DB schema once at the start of the run."""
    return {"schema": render_schema(state.db_id)}


def _extract_sql(text: str) -> str:
    """Pull a SQL statement out of an LLM reply, stripping markdown fences/prose.

    Intentionally simple: take the first ```sql ... ``` block if there is one,
    otherwise the whole reply. You may need to harden this for your prompts.
    """
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (fenced.group(1) if fenced else text).strip()


async def generate_sql_node(state: AgentState) -> dict:
    """Worked example - the other LLM nodes follow this same shape.

    Build messages from the prompts, call the shared llm(), extract the SQL,
    and return only the state fields you changed. `iteration` is bumped here
    (and in revise) so route_after_verify can enforce MAX_ITERATIONS.

    Async so the server's event loop can keep many requests in flight without
    being capped by FastAPI's sync threadpool (40 threads). The LLM call is the
    long pole, so awaiting it is what frees the loop.
    """
    response = await llm().ainvoke([
        ("system", prompts.GENERATE_SQL_SYSTEM),
        ("user", prompts.GENERATE_SQL_USER.format(
            schema=state.schema,
            question=state.question,
        )),
    ])
    sql = _extract_sql(response.content)
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [{"node": "generate_sql", "sql": sql}],
    }


def execute_node(state: AgentState) -> dict:
    """Provided. Runs the SQL and stores the result."""
    return {"execution": execute_sql(state.db_id, state.sql)}


def _parse_verify(text: str) -> dict:
    """Pull {"ok": bool, "issue": str} out of an LLM reply, defensively.

    The model may wrap the JSON in prose or ```json fences, or emit nothing
    parseable. On any failure we fall back to ok=False so the loop revises
    rather than terminating on a malformed verdict.
    """
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else text
    obj_match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if obj_match:
        try:
            obj = json.loads(obj_match.group(0))
            return {
                "ok": bool(obj.get("ok", False)),
                "issue": str(obj.get("issue", "") or ""),
            }
        except (json.JSONDecodeError, AttributeError):
            pass
    return {"ok": False, "issue": "could not parse verifier output"}


async def verify_node(state: AgentState) -> dict:
    """Decide whether state.execution plausibly answers state.question."""
    result_text = state.execution.render() if state.execution else "ERROR: no execution result"
    response = await llm().ainvoke([
        ("system", prompts.VERIFY_SYSTEM),
        ("user", prompts.VERIFY_USER.format(
            question=state.question,
            sql=state.sql,
            result=result_text,
        )),
    ])
    verdict = _parse_verify(response.content)
    return {
        "verify_ok": verdict["ok"],
        "verify_issue": verdict["issue"],
        "history": state.history + [{
            "node": "verify",
            "ok": verdict["ok"],
            "issue": verdict["issue"],
        }],
    }


async def revise_node(state: AgentState) -> dict:
    """Produce a revised SQL query given state.verify_issue and the prior attempt."""
    result_text = state.execution.render() if state.execution else "ERROR: no execution result"
    response = await llm().ainvoke([
        ("system", prompts.REVISE_SYSTEM),
        ("user", prompts.REVISE_USER.format(
            schema=state.schema,
            question=state.question,
            sql=state.sql,
            result=result_text,
            issue=state.verify_issue,
        )),
    ])
    sql = _extract_sql(response.content)
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [{"node": "revise", "sql": sql}],
    }


def route_after_execute(state: AgentState) -> str:
    """Skip the verify LLM call on the happy path (Phase 6 latency lever).

    If the SQL executed cleanly we accept it and END. The eval showed verify/
    revise adds 0 execution-accuracy (verify checks executability, not semantic
    correctness), so spending a full LLM call to re-check a query that already
    ran is pure latency. Only when execution FAILS do we fall into verify ->
    revise to repair it. MAX_ITERATIONS still caps the failure loop.
    """
    if (state.execution is not None and state.execution.ok) or state.iteration >= MAX_ITERATIONS:
        return "end"
    return "verify"


def route_after_verify(state: AgentState) -> str:
    """Conditional router: return "revise" to loop, "end" to terminate."""
    if state.verify_ok or state.iteration >= MAX_ITERATIONS:
        return "end"
    return "revise"


# ---- Graph wiring -----------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("attach_schema", _attach_schema)
    g.add_node("generate_sql", generate_sql_node)
    g.add_node("execute", execute_node)
    g.add_node("verify", verify_node)
    g.add_node("revise", revise_node)

    g.add_edge(START, "attach_schema")
    g.add_edge("attach_schema", "generate_sql")
    g.add_edge("generate_sql", "execute")
    g.add_conditional_edges(
        "execute",
        route_after_execute,
        {"verify": "verify", "end": END},
    )
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {"revise": "revise", "end": END},
    )
    g.add_edge("revise", "execute")
    return g.compile()


graph = build_graph()
