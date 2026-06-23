#!/usr/bin/env bash
#
# Start vLLM with your chosen configuration.
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
#
# Workload profile (drives every choice below):
#   - Model: Qwen3-30B-A3B - Mixture-of-Experts, ~3B ACTIVE params out of 30B total.
#     Cheap to compute per token, but ALL experts' weights (~60GB in bf16) must sit in
#     VRAM. One H100 (80GB) holds them with room left for KV cache -> tensor-parallel = 1.
#   - Prompts are large (1.5-3K tokens: schema + question), outputs are short (a SQL query).
#   - Target SLO: P95 end-to-end agent latency < 5s at 10+ RPS sustained 5 min.
#
# Each flag's rationale is noted inline; the same list goes in REPORT.md (Phase 1).

set -euo pipefail

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

# --no-sync: the repo's uv.lock pins transformers 5.9.0, which is incompatible with
# vLLM 0.10.2 (vLLM's get_cached_tokenizer reads `all_special_tokens_extended`, removed
# in transformers >=4.56). We pin transformers==4.55.4 manually; --no-sync stops `uv run`
# from re-syncing the env back to the broken lock on every launch.
exec uv run --no-sync python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --max-num-seqs 256 \
    --enable-prefix-caching \
    --enable-chunked-prefill

# --- Rationale ---------------------------------------------------------------
# --tensor-parallel-size 1   Model fits on one H100; TP>1 would add cross-GPU
#                            comms latency for no capacity benefit here.
# --dtype bfloat16           Native precision; correctness baseline before trying
#                            quantization. (FP8 is a Phase 6 lever - see below.)
# --gpu-memory-utilization 0.90  Use most of the 80GB for KV cache (more concurrency
#                            = higher RPS) while leaving headroom against OOM.
# --max-model-len 8192       Covers a 3K-token prompt + short output with margin.
#                            Smaller = more KV cache per GB = more concurrent seqs,
#                            so this is a tuning knob in Phase 6.
# --max-num-seqs 256         Concurrency cap. Higher lifts throughput/RPS but can
#                            grow queue time and P95 latency - the key Phase 6 lever.
# --enable-prefix-caching    Big win for THIS workload: every request to a given DB
#                            shares the same long schema prefix, so its KV is computed
#                            once and reused -> lower TTFT.
# --enable-chunked-prefill   Interleaves long prefills with ongoing decodes so a big
#                            prompt doesn't stall other requests -> steadier P95.
#
# --- Phase 6 tuning levers (try one at a time, measure on the dashboard) -------
#   * Quantize to FP8 to free VRAM / speed compute:
#       --quantization fp8   (or serve an FP8 checkpoint and update VLLM_MODEL)
#   * Trade concurrency vs latency: adjust --max-num-seqs (e.g. 128 <-> 512).
#   * Shrink --max-model-len if real prompts stay well under 8192.
#   * --kv-cache-dtype fp8_e5m2 to roughly double KV-cache capacity.
