# Codex Trace Turn Dependency Scheduling Plan

## Summary

- Keep the offline mode invariant: the benchmark client still sends all unfolded codex requests up front, and the SGLang scheduler still starts simulation only after receiving `total_request`.
- Add dependency metadata to unfolded `codex-swebenchpro-traces` turns so each assistant turn knows its original conversation/session and previous included assistant turn.
- Enforce dependencies inside `C_SchedulerHook` by holding dependent requests in an internal pending pool until their predecessor finishes, then moving them into `FUTURE_QUEUE`.
- Important audit finding: this document describes a planned behavior. In the current checkout, the trace dependency metadata and scheduler dependency-release state are not implemented yet.

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
- Track all received generate requests with an explicit counter/state such as `RECEIVED_GENERATE_REQUESTS`.
  - Do not use `len(FUTURE_QUEUE) == total_request` after dependency partitioning.
  - Once dependent requests are stored outside `FUTURE_QUEUE`, using the current `len(FUTURE_QUEUE)` gate would deadlock: root requests would not run because `OFFLINE_RECV_ALL_REQUEST` would never become true.
- After `total_request` generate requests are received:
  - Root requests with no `trace_prev_request_id` go into `FUTURE_QUEUE` using their nominal `created_time`.
  - Dependent requests go into `DEPENDENT_REQUESTS[trace_prev_request_id]`.
  - Then set `OFFLINE_RECV_ALL_REQUEST=True` and start simulation.
- In `wrapped_process_batch_result`, after a request fully finishes:
  - Mark its `trace_request_id` completed once.
  - Move each dependent child into `FUTURE_QUEUE` at `ready_time = max(child.created_time, predecessor_completion_time)`.
  - Update child `simulation["created_time"]` and `simulation["queue_start"]` to `ready_time` so TTFT/e2e metrics are measured from logical send time, not from blocked dependency time.
- Do not delay HTTP task creation in `bench_serving.py`; all requests must still be present in scheduler-owned state before simulation starts.
- Reset all new dependency state during profile/result dump, alongside the existing `StateManager`, `REQUEST_STATS`, `ITERATION_STATS`, and `OFFLINE_RECV_ALL_REQUEST` reset.
  - This must include `DEPENDENT_REQUESTS`, completed trace ids, and received-request counters.
  - The in-process runner can execute multiple benchmarks on one engine, so stale dependency state must not survive between runs.

## Metrics Semantics

- Simulation-mode benchmark output should continue to use scheduler-produced metrics from `HISIM_OUTPUT_DIR/metrics.json`, not client-side wall-clock metrics.
- TTFT remains reasonable if, and only if, child requests are released with `simulation["created_time"]` and `simulation["queue_start"]` set to `ready_time`.
  - `C_SchedulerHook` initializes `RequestStats.created_time` and `last_event_time` from `simulation["created_time"]`.
  - The first generated-token latency is recorded as `request_response_time - last_event_time`.
  - Rewriting `created_time` to `ready_time` therefore excludes dependency-blocked time from TTFT.
- TPOT and ITL are not directly affected by dependency gating because they are computed from `gen_token_latencies[1:]`; after the first token, latencies are measured from the previous response event.
- This intentionally changes e2e/queue semantics for dependent turns:
  - Reported per-request latency means service latency after the child becomes logically sendable.
  - It does not include waiting for the predecessor turn.
  - If a future analysis needs end-user conversation latency, preserve the original nominal child timestamp in a separate metadata field instead of overwriting it.

## Lifecycle And Exit Safety

- The benchmark client creates one async task per unfolded request and awaits all tasks before triggering the final simulation profile/metric dump.
- Therefore every dependent request must eventually be released or explicitly failed/aborted; otherwise the client can hang in `asyncio.gather()` and never reach the normal script shutdown path.
- `codex-swebenchpro-traces-benchmark.sh` has traps and a kill fallback for external interruption, but normal per-run cleanup still depends on the benchmark command returning.
- The implementation should include a small invariant check when all work is complete:
  - no unreleased entries remain in `DEPENDENT_REQUESTS`;
  - the completed trace id count matches the number of trace requests that carried a `trace_request_id`;
  - any mismatch should be logged clearly before profile reset.

## Performance Constraints

- The intended overhead is acceptable if the implementation stays local and incremental:
  - one O(N) partition after all requests are received;
  - one dictionary lookup per completed trace request;
  - one heap push per released child request.
- Do not scan all pending/dependent requests in every scheduler loop.
  - `recv_requests()` runs in the scheduler hot loop.
  - `wrapped_process_batch_result()` already iterates over the current batch, so dependency release should be O(batch size + released children).
- The added metadata in `DatasetRow.simulation` is negligible compared with tokenized prompt payloads, and `get_request()` already updates this dict with `created_time` and `total_request`.
- Performance is not validated until a small before/after smoke benchmark exists for a large unfolded codex trace.

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
  - `OFFLINE_RECV_ALL_REQUEST` becomes true based on received generate request count, not `len(FUTURE_QUEUE)`;
  - child requests enter `FUTURE_QUEUE` only after predecessor completion.
- Add a lifecycle regression test:
  - a two-turn dependency chain completes all generated tasks;
  - final profile dump writes metrics/request/iteration files;
  - dependency globals are empty after reset.
- Add a performance smoke test or lightweight benchmark:
  - generate many independent roots and many dependent children;
  - assert release work does not scan the full pending set per scheduler iteration.
- Run:
  - `.venv/bin/python -m py_compile src/hisim/simulation/bench_serving.py src/hisim/simulation/sglang/sglang_hook.py`
  - targeted pytest for the new sampling/dependency tests.

## Assumptions

- Dependency is per original codex conversation only; different conversations may run concurrently.
- If an earlier assistant turn is filtered out by length/context rules, the next included assistant turn starts a new dependency chain.
- This plan fixes turn ordering only. It does not fix the separate KV-hit underestimation caused by mock generation returning token id `1`.
