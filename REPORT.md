# Text-to-SQL Serving & Observability — Report

**Model:** Qwen3-30B-A3B-Instruct-2507 (MoE, ~3B active / 30B total)
**Hardware:** 1× H100 80GB (Nebius) · vLLM 0.10.2 · Prometheus + Grafana + Langfuse
**SLO target:** P95 end-to-end agent latency < 5s at 10+ RPS sustained 5 min.

---

## 1. Serving configuration & rationale (Phase 1)

`scripts/start_vllm.sh` launches vLLM with the following levers. The workload is
**large prompts (1.5–3K tokens: schema + question) → short structured output (a SQL query)**,
which drives every choice:

| Flag | Value | Why |
|------|-------|-----|
| `--tensor-parallel-size` | 1 | All experts' weights (~57 GiB bf16) fit on one 80GB H100. TP>1 would add cross-GPU comm latency for no capacity gain. |
| `--dtype` | bfloat16 | Native precision; a correctness baseline before considering quantization (FP8 was held as a Phase-6 lever). |
| `--gpu-memory-utilization` | 0.90 | Maximize KV-cache space (→ more concurrency / RPS) while leaving OOM headroom. |
| `--max-model-len` | 8192 | Covers a ~3K-token prompt + short output with margin. Smaller = more KV/seq, so it is also a tuning knob. |
| `--max-num-seqs` | 256 | Concurrency cap. High enough that it is not the bottleneck for this workload. |
| `--enable-prefix-caching` | on | Every request to a given DB shares the long schema prefix → its KV is computed once and reused → lower TTFT. |
| `--enable-chunked-prefill` | on | Interleaves long prefills with ongoing decodes so a big prompt doesn't stall other requests → steadier P95. |

Model load was ~57 GiB / ~8 min including a one-time `torch.compile` (compilation level 3 +
full CUDA-graph capture over 67 batch sizes). This compile is CPU-bound (0% GPU util, no log
output for several minutes) — important operationally because **every vLLM restart during tuning
costs ~15 min**, so Phase-6 hypotheses were batched to minimize restarts.

Serving verified: `/v1/models` → 200, a chat completion returns valid SQL
(`results/phase1_serving_proof.txt`), and Prometheus scrapes `vllm:*` from `/metrics`.

---

## 2. Observability dashboard (Phase 2)

`infra/grafana/provisioning/dashboards/serving.json` — 9 panels across the three required domains,
all driven by vLLM's `vllm:*` Prometheus metrics:

- **Latency (the SLO lives here):** end-to-end request latency p50/p95/p99 with a **5s SLO
  threshold line**, time-to-first-token (TTFT), time-per-output-token (TPOT), queue-wait p95.
- **Throughput:** generation + prompt token/s, request rate (success/s), running-vs-waiting seqs.
- **KV cache:** `gpu_cache_usage_perc` occupancy gauge + prefix-cache hit rate (concurrency headroom).

Screenshots: `screenshots/phase6_dashboard_midload.png` (all panels live under load) and
`screenshots/phase6_dashboard_after_load.png`.

---

## 3. Agent design (Phase 3)

A LangGraph agent (`agent/graph.py`, `agent/prompts.py`):

```
START → attach_schema → generate_sql → execute ─(exec ok)──────────────→ END
                                          ^         │
                                          │   (exec failed)
                                          │         ▼
                                          │       verify ─(ok)──────────→ END
                                          │         │
                                          └─ revise ←(not ok)
```

- `generate_sql` builds SQL from the rendered schema + question (1 LLM call).
- `execute` runs it read-only against the SQLite DB.
- `verify` (LLM) returns a defensive boolean `{ok, issue}`; on malformed output it falls back to
  `ok=False` so the loop revises rather than terminating on a bad verdict.
- `revise` (LLM) repairs the SQL given the failing query + result + issue.
- Loop capped at `MAX_ITERATIONS = 3`.

**Design change made in Phase 6 (see §6):** `route_after_execute` short-circuits to END when the
SQL executes cleanly, so **verify/revise runs only on an execution failure**. This keeps the loop
intact for the case it actually helps while removing a full LLM call from the happy path.

---

## 4. Agent tracing (Phase 4)

Langfuse (`agent/server.py` auto-attaches `CallbackHandler` when keys are present). Each `/answer`
produces a nested span tree with per-span latency + token counts and request tags. Verified via the
Langfuse API on a failure-path trace:

```
LangGraph (root, tag eval_set=phase4_demo)
├─ attach_schema
├─ generate_sql → ChatOpenAI  (1043 in / 14 out tokens)
├─ execute
├─ route_after_execute
├─ verify        → ChatOpenAI (267 in / 15 out tokens)
├─ route_after_verify
├─ revise        → ChatOpenAI (1065 in / 10 out tokens)
├─ execute
└─ route_after_execute
```

Tags (`eval_set`) are attached for later trace filtering. Screenshots: `screenshots/phase4_*`.

---

## 5. Evaluation (Phase 5)

`evals/run_eval.py` runs 30 curated BIRD questions and scores **execution accuracy** by comparing
canonicalized result sets (sorted rows) against gold SQL, capturing per-iteration pass rate.

| | Baseline | Tuned (skip-verify) |
|---|---|---|
| Execution accuracy | **0.30** | **0.30** |
| Per-iteration | iter_0 = iter_1 = iter_2 = 0.30 | iter_0 = 0.30 |
| Revisions triggered | 9 / 30 | 0 / 30 |
| Agent errors | 0 | 0 |
| Mean latency / query | 0.99 s | **0.45 s** |

**Does the verify/revise loop pay for itself? No (on this benchmark).** In the baseline, 9 of 30
questions triggered a revision yet per-iteration accuracy was **flat at 0.30** — revision added
**zero** execution accuracy. Root cause: `verify` validates **executability**, not **semantic
correctness**, so a query that runs but returns the wrong rows is never revised. Removing verify
from the happy path therefore cost 0 accuracy while **halving per-query latency** (0.99 → 0.45 s).

(Note: the provided `canonicalize` comparator sorts but does not de-duplicate rows, so it is
multiplicity-sensitive — e.g. a `formula_1` answer with 11 duplicate rows fails against a gold
`DISTINCT` single row. The comparator is part of the grading harness and was left intact; it caps
the achievable accuracy somewhat but affects baseline and tuned runs identically.)

Artifacts: `results/eval_baseline.json`, `results/eval_after_tuning.json`.

---

## 6. SLO tuning — metric-grounded iteration log (Phase 6)

Disciplined cycle, **one variable per iteration**, every change measured on the dashboard *and*
on end-to-end latency.

| Iteration | Change | p95 | HTTP-500s | What the metric told us |
|-----------|--------|-----|-----------|--------------------------|
| **Baseline** | sync endpoint | 14.87 s | 381 / 3000 (**12.7%**) | SLO missed; dashboard showed vLLM KV-cache near-empty → bottleneck is the **agent**, not vLLM. |
| **A** | async endpoint + shared LLM client | 10.4 s¹ | 13.3%¹ | p95 dropped, but 500-rate **unchanged** → **disproved** the "threadpool/concurrency" theory; the 500s are something else. |
| **B** | fix `schema.py` FK-None bug | 11.4 s¹ | **0.3%**¹ | 500-rate collapsed → this was the real cause. |
| **C** | skip verify on happy path | **7.27 s** | **0.4%** | p95 halved vs baseline. |

¹ Iterations A/B measured on 60s probes; baseline and C are full 300s runs.

**Diagnosing the 500s (the key finding).** A `traceback.print_exc` patch on the agent + a short
load probe revealed **33 of 34** errors were
`AttributeError: 'NoneType' object has no attribute 'replace'` in the *provided* `agent/schema.py`.
`render_schema` rendered every foreign key as `REFERENCES "tbl"("col")`, but SQLite's
`PRAGMA foreign_key_list` returns **NULL** for the parent column (`to`) when a FK implicitly targets
the parent's PRIMARY KEY. `_q(None)` then crashed `_attach_schema` — the **very first node, before
any LLM call** — which is why these 500s returned in ~1 ms. It is **deterministic per database**:
any DB containing an implicit-PK foreign key fails 100% of the time, and random sampling across the
11 BIRD DBs hits an affected DB ≈ 13% of requests — matching the 12.7% baseline error rate exactly.
Fix: render `REFERENCES "tbl"` (no column) when the parent column is NULL — valid SQL, accurate,
and crash-free. The remaining 1–2 errors are genuine **context overflow** (a few large schemas push
the prompt past `max-model-len 8192`, e.g. 14,739 tokens → vLLM 400).

**Final SLO outcome: P95 ≈ 7.27 s — SLO missed, but diagnosed.** Full 300s tuned run:
p50 1.04 s, **p95 7.27 s**, p99 10.06 s, max 20.3 s, errors 0.4%, 2984/3000 ok (vs baseline
p50 3.24 / p95 14.87 / p99 21.1 / max 111 / 12.7% errors).

The dashboard locates the residual 2.3 s gap precisely: **TTFT p95 ≈ 160 ms** and
**TPOT p95 ≈ 25 ms** are both healthy (a ~50-token SQL output costs ≈ 1.4 s of real generation),
and **queue-wait p95 ≈ 280 ms** is low — yet vLLM's own request-latency **p99 spikes to ~2 min**
during the burst. So the residual latency is **intermittent vLLM batching saturation / tail
queueing at 10 RPS**, not slow decode and (after iteration C) not the agent. Closing the last 2.3 s
would require reducing per-request work further or adding capacity (see §7), not another agent tweak.

---

## 7. Future work

- **Crush the residual P95 tail.** The diagnosis points at vLLM burst-queueing, so: (a) serve an
  **FP8** checkpoint / `--kv-cache-dtype fp8_e5m2` to widen KV-cache and batch headroom; (b) lower
  `--max-model-len` to 4096 (real prompts mostly fit) to pack more sequences and also eliminate the
  context-overflow 400s; (c) admission control / request smoothing at the agent to flatten bursts.
- **Make `verify` improve accuracy, not just executability.** Have it compare against expected
  output shape / column types / row-count sanity so semantically-wrong-but-runnable SQL is caught —
  then the revise loop would actually lift the 0.30 pass rate.
- **Schema compaction** for large DBs (only the tables/columns relevant to the question) to cut
  prompt tokens → lower latency and remove context-overflow errors.
- **Fix the eval comparator** to de-duplicate / respect `DISTINCT` semantics for a fairer accuracy
  measure (kept as-is here since it is the shared grading harness).
