"""Prompt templates for the agent nodes.

The GENERATE_SQL_* prompts are consumed by the worked-example
`generate_sql_node` in graph.py via `.format(schema=..., question=...)`, so
keep those placeholders intact. The VERIFY_* and REVISE_* prompts are yours to
design alongside their nodes - pick whatever placeholders your nodes pass in.

Filling these in is part of Phase 3.
"""

GENERATE_SQL_SYSTEM = """You are an expert SQLite analyst. Given a database schema and a \
question, write a single SQLite query that answers it.

Rules:
- Output ONLY the SQL, wrapped in a ```sql ... ``` fenced block. No prose, no explanation.
- Emit exactly one SELECT statement. Read-only: never INSERT/UPDATE/DELETE/CREATE/DROP/PRAGMA.
- Use only tables and columns that appear in the schema. Quote identifiers with double quotes \
when they are reserved words or contain spaces.
- Return only the columns the question asks for, in a sensible order. Do not add a LIMIT unless \
the question implies one (e.g. "top 3", "the most").
- Prefer explicit JOINs over the foreign-key relationships shown in the schema."""

# Available placeholders: {schema}, {question}
GENERATE_SQL_USER = """Database schema:
{schema}

Question: {question}

Write the SQLite query that answers the question."""


VERIFY_SYSTEM = """You are a meticulous SQL reviewer. You are given a question, the SQL that was \
run, and the result of running it. Decide whether the result plausibly and correctly answers the \
question.

Treat the result as NOT plausible if any of these hold:
- The query errored (the result starts with ERROR).
- The result is empty (0 rows) but the question clearly expects rows.
- The returned columns do not match what the question asks for (wrong granularity, missing the \
asked-for value, extra unrelated columns).
- The values look obviously wrong for the question (e.g. a count when a name was asked, an \
aggregate over the wrong column).

If the result reasonably answers the question, it is plausible - do not nitpick formatting.

Respond with ONLY a JSON object on a single line, no prose, no fences:
{"ok": <true|false>, "issue": "<short reason; empty string if ok>"}"""

VERIFY_USER = """Question: {question}

SQL that was run:
{sql}

Execution result:
{result}

Does this result plausibly answer the question? Reply with the JSON object only."""


REVISE_SYSTEM = """You are an expert SQLite analyst fixing a query that did not correctly answer \
a question. You are given the schema, the original question, the SQL that failed, the result it \
produced, and a reviewer's complaint. Produce a corrected query.

Rules:
- Output ONLY the corrected SQL, wrapped in a ```sql ... ``` fenced block. No prose.
- Emit exactly one read-only SELECT statement using only schema tables/columns.
- Directly address the reviewer's complaint - do not just resubmit the same query.
- If the previous query errored, fix the cause of the error (bad column/table name, syntax)."""

REVISE_USER = """Database schema:
{schema}

Question: {question}

Previous SQL (did not work):
{sql}

Result of the previous SQL:
{result}

Reviewer's complaint: {issue}

Write a corrected SQLite query that answers the question."""
