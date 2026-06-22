from types import SimpleNamespace

import pytest

from hisim.simulation.sglang.sglang_hook import (
    C_SchedulerHook,
    _simulation_params_from_http,
)
from hisim.simulation.types import RequestStats, SimulationParams


NO_TRACE = object()


class FakeGenerateReq:
    is_chunked = 0

    def __init__(
        self,
        rid,
        created_time,
        total_request,
        trace_request_id=NO_TRACE,
        trace_prev_request_id=None,
        finished=False,
        simulation_as_dict=False,
    ):
        simulation = SimulationParams(
            created_time=created_time,
            total_request=total_request,
        )
        if trace_request_id is not NO_TRACE:
            simulation.trace_request_id = trace_request_id
            simulation.trace_prev_request_id = trace_prev_request_id

        if simulation_as_dict:
            simulation = simulation.to_dict()

        self.rid = rid
        self.input_ids = [1, 2]
        self.sampling_params = SimpleNamespace(
            max_new_tokens=2,
            custom_params={"simulation": simulation},
        )
        self._finished = finished

    def finished(self):
        return self._finished


@pytest.fixture(autouse=True)
def reset_scheduler_hook_state():
    C_SchedulerHook.PENDING_TRACE_REQUESTS = {}
    C_SchedulerHook._reset_trace_dependency_state()
    C_SchedulerHook.REQUEST_STATS.clear()
    C_SchedulerHook.ITERATION_STATS.clear()
    yield
    C_SchedulerHook.PENDING_TRACE_REQUESTS = {}
    C_SchedulerHook._reset_trace_dependency_state()
    C_SchedulerHook.REQUEST_STATS.clear()
    C_SchedulerHook.ITERATION_STATS.clear()


def _queued_requests():
    return [item[2] for item in C_SchedulerHook.FUTURE_QUEUE]


def test_offline_ingest_uses_received_count_not_future_queue_len():
    root = FakeGenerateReq(
        "root",
        created_time=0,
        total_request=2,
        trace_request_id="0:0",
        trace_prev_request_id=None,
    )
    child = FakeGenerateReq(
        "child",
        created_time=0,
        total_request=2,
        trace_request_id="0:1",
        trace_prev_request_id="0:0",
    )

    extra_requests = C_SchedulerHook._ingest_offline_generate_requests(
        [root, child]
    )

    assert extra_requests == []
    assert C_SchedulerHook.RECEIVED_GENERATE_REQUESTS == 2
    assert C_SchedulerHook.EXPECTED_GENERATE_REQUESTS == 2
    assert C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST is True
    assert _queued_requests() == [root]
    assert C_SchedulerHook.PENDING_TRACE_REQUESTS["0:1"] == child


def test_offline_ingest_rejects_mismatched_total_request():
    first = FakeGenerateReq("first", created_time=0, total_request=2)
    second = FakeGenerateReq("second", created_time=0, total_request=3)

    with pytest.raises(RuntimeError, match="Mismatched total_request"):
        C_SchedulerHook._ingest_offline_generate_requests([first, second])


def test_offline_ingest_rejects_dict_simulation():
    req = FakeGenerateReq(
        "root",
        created_time=0,
        total_request=1,
        trace_request_id="0:0",
        trace_prev_request_id=None,
        simulation_as_dict=True,
    )

    with pytest.raises(TypeError, match="custom_params\\['simulation'\\]"):
        C_SchedulerHook._ingest_offline_generate_requests([req])


def test_tokenizer_http_boundary_converts_dict_simulation():
    simulation = _simulation_params_from_http(
        SimulationParams(
            created_time=1.5,
            total_request=3,
            trace_request_id="0:0",
            trace_prev_request_id=None,
        ).to_dict()
    )

    assert isinstance(simulation, SimulationParams)
    assert simulation.created_time == 1.5
    assert simulation.total_request == 3
    assert simulation.trace_request_id == "0:0"


def test_child_released_after_predecessor_completion():
    root = FakeGenerateReq(
        "root",
        created_time=0,
        total_request=2,
        trace_request_id="0:0",
        trace_prev_request_id=None,
    )
    child = FakeGenerateReq(
        "child",
        created_time=0,
        total_request=2,
        trace_request_id="0:1",
        trace_prev_request_id="0:0",
    )
    C_SchedulerHook._ingest_offline_generate_requests([root, child])

    released = C_SchedulerHook._release_trace_dependents(
        trace_request_id="0:0",
        predecessor_completion_time=5.0,
    )
    assert released == [child]
    assert child in _queued_requests()
    assert C_SchedulerHook.PENDING_TRACE_REQUESTS == {}


def test_child_ready_time_rewrites_created_time_and_queue_start():
    root = FakeGenerateReq(
        "root",
        created_time=0,
        total_request=2,
        trace_request_id="0:0",
        trace_prev_request_id=None,
        finished=True,
    )
    child = FakeGenerateReq(
        "child",
        created_time=1.0,
        total_request=2,
        trace_request_id="0:1",
        trace_prev_request_id="0:0",
    )
    C_SchedulerHook._ingest_offline_generate_requests([root, child])

    C_SchedulerHook._release_trace_dependents(
        trace_request_id="0:0",
        predecessor_completion_time=5.0,
    )

    child_sim = child.sampling_params.custom_params["simulation"]
    assert child_sim.created_time == 5.0
    assert child_sim.queue_start == 5.0
    assert C_SchedulerHook.FUTURE_QUEUE[1][0] == 5.0


def test_child_ready_time_keeps_later_nominal_time():
    root = FakeGenerateReq(
        "root",
        created_time=0,
        total_request=2,
        trace_request_id="0:0",
        trace_prev_request_id=None,
        finished=True,
    )
    child = FakeGenerateReq(
        "child",
        created_time=10.0,
        total_request=2,
        trace_request_id="0:1",
        trace_prev_request_id="0:0",
    )
    C_SchedulerHook._ingest_offline_generate_requests([root, child])

    C_SchedulerHook._release_trace_dependents(
        trace_request_id="0:0",
        predecessor_completion_time=5.0,
    )

    child_sim = child.sampling_params.custom_params["simulation"]
    assert child_sim.created_time == 10.0
    assert child_sim.queue_start == 10.0


def test_two_turn_dependency_chain_drains_completion_state():
    root = FakeGenerateReq(
        "root",
        created_time=0,
        total_request=2,
        trace_request_id="0:0",
        trace_prev_request_id=None,
        finished=True,
    )
    child = FakeGenerateReq(
        "child",
        created_time=0,
        total_request=2,
        trace_request_id="0:1",
        trace_prev_request_id="0:0",
        finished=False,
    )
    C_SchedulerHook._ingest_offline_generate_requests([root, child])

    assert C_SchedulerHook._enqueue_ready_offline_requests(0) == [root]
    assert (
        C_SchedulerHook._release_trace_dependents(
            trace_request_id="0:0",
            predecessor_completion_time=3.0,
        )
        == [child]
    )
    assert C_SchedulerHook.PENDING_TRACE_REQUESTS == {}
    assert C_SchedulerHook._enqueue_ready_offline_requests(3.0) == [child]

    assert (
        C_SchedulerHook._release_trace_dependents(
            trace_request_id="0:1",
            predecessor_completion_time=3.25,
        )
        == []
    )
    assert (
        C_SchedulerHook.SEEN_TRACE_REQUEST_IDS
        == C_SchedulerHook.COMPLETED_TRACE_REQUEST_IDS
    )


def test_ttft_excludes_dependency_blocked_time():
    req_stats = RequestStats(rid="child", created_time=5.0, last_event_time=5.0)
    request_response_time = 5.25

    req_stats.gen_token_latencies.append(
        request_response_time - req_stats.last_event_time
    )

    assert req_stats.gen_token_latencies[0] == pytest.approx(0.25)
    assert req_stats.gen_token_latencies[0] != pytest.approx(4.25)


def test_reset_clears_trace_dependency_state():
    child = FakeGenerateReq(
        "child",
        created_time=0,
        total_request=1,
        trace_request_id="0:1",
        trace_prev_request_id="0:0",
    )
    C_SchedulerHook.FUTURE_QUEUE.append((0, 1, child))
    C_SchedulerHook.PENDING_TRACE_REQUESTS["0:1"] = child
    C_SchedulerHook.COMPLETED_TRACE_REQUEST_IDS.add("0:0")
    C_SchedulerHook.SEEN_TRACE_REQUEST_IDS.add("0:0")
    C_SchedulerHook.RECEIVED_GENERATE_REQUESTS = 1
    C_SchedulerHook.EXPECTED_GENERATE_REQUESTS = 1
    C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST = True

    C_SchedulerHook._reset_trace_dependency_state()

    assert C_SchedulerHook.FUTURE_QUEUE == []
    assert C_SchedulerHook.PENDING_TRACE_REQUESTS == {}
    assert C_SchedulerHook.COMPLETED_TRACE_REQUEST_IDS == set()
    assert C_SchedulerHook.SEEN_TRACE_REQUEST_IDS == set()
    assert C_SchedulerHook.RECEIVED_GENERATE_REQUESTS == 0
    assert C_SchedulerHook.EXPECTED_GENERATE_REQUESTS is None
    assert C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST is False


def test_trace_dependency_invariants_raise_for_pending_requests():
    child = FakeGenerateReq(
        "child",
        created_time=0,
        total_request=1,
        trace_request_id="0:1",
        trace_prev_request_id="0:0",
    )
    C_SchedulerHook.PENDING_TRACE_REQUESTS["0:1"] = child

    with pytest.raises(
        RuntimeError, match="Pending blocked trace requests remain"
    ):
        C_SchedulerHook._check_trace_dependency_invariants()


def test_trace_dependency_invariants_raise_for_incomplete_trace_ids():
    C_SchedulerHook.SEEN_TRACE_REQUEST_IDS.add("0:0")

    with pytest.raises(
        RuntimeError,
        match="Trace requests were seen but not completed before profile reset",
    ):
        C_SchedulerHook._check_trace_dependency_invariants()


def test_offline_ingest_rejects_inconsistent_trace_prev_id():
    root = FakeGenerateReq(
        "root",
        created_time=0,
        total_request=3,
        trace_request_id="0:0",
        trace_prev_request_id=None,
    )
    child = FakeGenerateReq(
        "child",
        created_time=0,
        total_request=3,
        trace_request_id="0:1",
        trace_prev_request_id="0:0",
    )
    inconsistent_child = FakeGenerateReq(
        "inconsistent_child",
        created_time=0,
        total_request=3,
        trace_request_id="0:2",
        trace_prev_request_id="0:0",
    )

    with pytest.raises(RuntimeError, match="Trace dependency metadata is inconsistent"):
        C_SchedulerHook._ingest_offline_generate_requests(
            [root, child, inconsistent_child]
        )


def test_non_codex_requests_keep_existing_offline_behavior():
    first = FakeGenerateReq("first", created_time=0, total_request=2)
    second = FakeGenerateReq("second", created_time=1, total_request=2)

    C_SchedulerHook._ingest_offline_generate_requests([second, first])

    assert C_SchedulerHook.PENDING_TRACE_REQUESTS == {}
    assert C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST is True
    assert C_SchedulerHook._enqueue_ready_offline_requests(0) == [first]
    assert C_SchedulerHook._enqueue_ready_offline_requests(1) == [second]


def test_mixed_trace_and_non_trace_requests_are_independent_roots():
    trace_root = FakeGenerateReq(
        "trace_root",
        created_time=0,
        total_request=2,
        trace_request_id="0:0",
        trace_prev_request_id=None,
    )
    non_trace = FakeGenerateReq("non_trace", created_time=0, total_request=2)

    C_SchedulerHook._ingest_offline_generate_requests([trace_root, non_trace])

    assert C_SchedulerHook.RECEIVED_GENERATE_REQUESTS == 2
    assert C_SchedulerHook.PENDING_TRACE_REQUESTS == {}
    assert C_SchedulerHook._enqueue_ready_offline_requests(0) == [
        trace_root,
        non_trace,
    ]


class NoScanPendingMap(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pop_calls = 0

    def pop(self, key, default=None):
        self.pop_calls += 1
        return super().pop(key, default)

    def items(self):
        raise AssertionError("release should not scan pending items")

    def values(self):
        raise AssertionError("release should not scan pending values")

    def __iter__(self):
        raise AssertionError("release should not iterate pending keys")


def test_dependency_release_does_not_scan_global_pending_set():
    root = FakeGenerateReq(
        "root",
        created_time=0,
        total_request=2,
        trace_request_id="0:0",
        trace_prev_request_id=None,
        finished=True,
    )
    child = FakeGenerateReq(
        "child",
        created_time=0,
        total_request=2,
        trace_request_id="0:1",
        trace_prev_request_id="0:0",
    )
    mapping = NoScanPendingMap(
        {f"other:{idx}": object() for idx in range(1000)}
    )
    mapping["0:1"] = child
    C_SchedulerHook.PENDING_TRACE_REQUESTS = mapping
    C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST = True

    released = C_SchedulerHook._release_trace_dependents(
        trace_request_id="0:0",
        predecessor_completion_time=1.0,
    )

    assert released == [child]
    assert mapping.pop_calls == 1
