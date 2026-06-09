# Codex Trace Turn Dependency Scheduling Plan

## Summary

- Keep the offline mode invariant: the benchmark client still sends all unfolded codex requests up front, and the SGLang scheduler still starts simulation only after receiving `total_request`.
- Add dependency metadata to unfolded `codex-swebenchpro-traces` turns so each assistant turn knows its original conversation/session and previous included assistant turn.
- Enforce dependencies inside `C_SchedulerHook` by holding dependent requests in an internal pending pool until their predecessor finishes, then moving them into `FUTURE_QUEUE`.

## Key Changes

- In `sample_agentic_trace_requests`, populate `DatasetRow.simulation` with:
  - `trace_session_id`: source conversation index.
  - `trace_turn_index`: assistant-turn index within that conversation.
  - `trace_request_id`: deterministic id like `<session>:<turn>`.
  - `trace_prev_request_id`: previous included assistant turn id in the same conversation, or `None`.
- Keep `get_request()` behavior of adding `created_time` and `total_request`; it must preserve the trace metadata already in `simulation`.
- Bump the codex cache key version in `get_agentic_trace_cache_path` so old cached rows without dependency metadata are not reused.

## Scheduler Behavior

- In offline mode, collect all generate requests first, as today.
- After `total_request` requests are received:
  - Root requests with no `trace_prev_request_id` go into `FUTURE_QUEUE` using their nominal `created_time`.
  - Dependent requests go into `DEPENDENT_REQUESTS[trace_prev_request_id]`.
  - Then set `OFFLINE_RECV_ALL_REQUEST=True` and start simulation.
- In `wrapped_process_batch_result`, after a request fully finishes:
  - Mark its `trace_request_id` completed once.
  - Move each dependent child into `FUTURE_QUEUE` at `ready_time = max(child.created_time, predecessor_completion_time)`.
  - Update child `simulation["created_time"]` and `simulation["queue_start"]` to `ready_time` so TTFT/e2e metrics are measured from logical send time, not from blocked dependency time.
- Do not delay HTTP task creation in `bench_serving.py`; all requests must still be present in scheduler-owned state before simulation starts.

## Test Plan

- Add a tiny local codex-style JSONL fixture with one conversation containing two assistant turns.
- Unit test `sample_agentic_trace_requests`:
  - first assistant turn has no predecessor;
  - second assistant turn depends on the first;
  - metadata survives cache load by forcing a fresh cache version.
- Unit test `get_request` preserves trace metadata while adding `created_time` and `total_request`.
- Add a small pure helper or class-method test for offline dependency release:
  - all requests are initially received;
  - only root requests enter `FUTURE_QUEUE`;
  - child requests enter `FUTURE_QUEUE` only after predecessor completion.
- Run:
  - `.venv/bin/python -m py_compile src/hisim/simulation/bench_serving.py src/hisim/simulation/sglang/sglang_hook.py`
  - targeted pytest for the new sampling/dependency tests.

## Assumptions

- Dependency is per original codex conversation only; different conversations may run concurrently.
- If an earlier assistant turn is filtered out by length/context rules, the next included assistant turn starts a new dependency chain.
- This plan fixes turn ordering only. It does not fix the separate KV-hit underestimation caused by mock generation returning token id `1`.
