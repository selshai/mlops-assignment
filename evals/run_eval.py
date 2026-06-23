"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def _candidate_sqls(history: list[dict]) -> list[str]:
    """Ordered SQL attempts from the agent's history (generate_sql + revise nodes)."""
    return [h["sql"] for h in history if "sql" in h and h.get("node") in ("generate_sql", "revise")]


def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness.

    Calls the agent, then for each SQL attempt it emitted (iteration 0 =
    initial generate, 1.. = each revise) executes that SQL plus the gold SQL
    and compares canonicalized rows. This is what lets summarize() report
    whether the verify->revise loop actually improves accuracy.
    """
    db_id = question["db_id"]
    gold_sql = question["gold_sql"]

    t0 = time.monotonic()
    error: str | None = None
    payload: dict = {}
    try:
        resp = httpx.post(
            agent_url,
            json={"question": question["question"], "db": db_id},
            timeout=120.0,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
    latency = time.monotonic() - t0

    gold_ok, gold_rows, gold_err = run_sql(db_id, gold_sql)

    history = payload.get("history", []) if not error else []
    candidates = _candidate_sqls(history)
    # Fall back to the final sql if history carried no candidates.
    if not candidates and payload.get("sql"):
        candidates = [payload["sql"]]

    per_iteration: list[dict] = []
    for i, sql in enumerate(candidates):
        pred_ok, pred_rows, pred_err = run_sql(db_id, sql)
        correct = bool(gold_ok and pred_ok and matches(gold_rows, pred_rows))
        per_iteration.append({
            "iteration": i,
            "sql": sql,
            "exec_ok": pred_ok,
            "exec_error": pred_err,
            "correct": correct,
        })

    final_correct = per_iteration[-1]["correct"] if per_iteration else False

    return {
        "db_id": db_id,
        "question": question["question"],
        "gold_sql": gold_sql,
        "gold_exec_ok": gold_ok,
        "gold_error": gold_err,
        "final_sql": payload.get("sql", ""),
        "agent_ok": payload.get("ok", False),
        "agent_error": payload.get("error") or error,
        "iterations": payload.get("iterations", len(candidates)),
        "per_iteration": per_iteration,
        "final_correct": final_correct,
        "latency_seconds": latency,
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results with per-iteration carry-forward.

    If a question stopped at iteration j < k (verifier happy, or hit the cap),
    its iteration-k correctness is carried forward from iteration j - that is
    the answer that would have been served had we polled later.
    """
    n = len(results)
    if n == 0:
        return {"n_questions": 0, "overall_pass_rate": 0.0, "per_iteration_pass_rate": {}}

    max_iters = max((len(r["per_iteration"]) for r in results), default=0)

    per_iteration_pass_rate: dict[str, float] = {}
    for k in range(max_iters):
        hits = 0
        for r in results:
            steps = r["per_iteration"]
            if not steps:
                continue  # never produced any SQL -> counts as miss
            # carry-forward: clamp k to this question's last emitted iteration
            idx = min(k, len(steps) - 1)
            if steps[idx]["correct"]:
                hits += 1
        per_iteration_pass_rate[f"iter_{k}"] = round(hits / n, 4)

    overall_pass_rate = round(sum(1 for r in results if r["final_correct"]) / n, 4)
    latencies = [r["latency_seconds"] for r in results]

    return {
        "n_questions": n,
        "overall_pass_rate": overall_pass_rate,
        "per_iteration_pass_rate": per_iteration_pass_rate,
        "n_with_revision": sum(1 for r in results if len(r["per_iteration"]) > 1),
        "n_agent_errors": sum(1 for r in results if r["agent_error"]),
        "mean_latency_seconds": round(sum(latencies) / n, 3),
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
