import argparse
import asyncio
import json
import pickle
import re

import pytest

from hisim.simulation import bench_serving
from hisim.simulation.bench_serving import (
    DatasetRow,
    get_agentic_trace_cache_path,
    get_request,
    sample_agentic_trace_requests,
)
from hisim.simulation.types import SimulationParams


class FakeCodexTokenizer:
    name_or_path = "fake-codex-tokenizer"
    vocab_size = 1024
    chat_template = "fake-template"

    def encode(self, text, *args, **kwargs):
        explicit_lengths = re.findall(r"\btok(\d+)\b", str(text))
        if explicit_lengths:
            return list(range(sum(int(length) for length in explicit_lengths)))
        return list(range(len(str(text).split())))

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True
    ):
        rendered = " ".join(str(m["content"]) for m in messages)
        if tokenize:
            return self.encode(rendered)
        return rendered


@pytest.fixture
def fake_tokenizer():
    return FakeCodexTokenizer()


@pytest.fixture
def isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def codex_trace_path(tmp_path):
    dataset_path = tmp_path / "codex_trace_turn_dependencies.jsonl"
    rows = [
        {
            "conversations": [
                {"from": "human", "value": "open the file please"},
                {"from": "gpt", "value": "answer one"},
                {"from": "human", "value": "now inspect the result"},
                {"from": "gpt", "value": "answer two"},
            ]
        },
        {
            "conversations": [
                {"from": "human", "value": "run the tests please"},
                {"from": "gpt", "value": "answer three"},
            ]
        },
    ]
    dataset_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return dataset_path


def test_sample_agentic_trace_requests_generates_dependency_metadata(
    isolated_home, codex_trace_path, fake_tokenizer, capsys
):
    rows = sample_agentic_trace_requests(
        str(codex_trace_path), fake_tokenizer, return_text=True
    )
    output = capsys.readouterr().out

    assert len(rows) == 3
    assert "#Codex trace turn distribution: turn1: 2, turn2: 1" in output
    assert [
        (
            row.simulation.trace_request_id,
            row.prompt_len,
            row.output_len,
        )
        for row in rows
    ] == [
        # input: "open the file please"; output: "answer one"
        ("0:0", 4, 2),
        # input: previous human+assistant turn plus "now inspect the result"
        ("0:1", 10, 2),
        # input: "run the tests please"; output: "answer three"
        ("1:0", 4, 2),
    ]
    assert rows[0].simulation == SimulationParams(
        trace_session_id=0,
        trace_turn_index=0,
        trace_request_id="0:0",
        trace_prev_request_id=None,
    )
    assert rows[1].simulation == SimulationParams(
        trace_session_id=0,
        trace_turn_index=1,
        trace_request_id="0:1",
        trace_prev_request_id="0:0",
    )
    assert rows[2].simulation == SimulationParams(
        trace_session_id=1,
        trace_turn_index=0,
        trace_request_id="1:0",
        trace_prev_request_id=None,
    )

    cached_rows = sample_agentic_trace_requests(
        str(codex_trace_path), fake_tokenizer, return_text=True
    )
    cached_output = capsys.readouterr().out
    assert cached_rows == rows
    assert "#Codex trace turn distribution: turn1: 2, turn2: 1" in cached_output


def test_sample_agentic_trace_requests_stops_conversation_after_filtered_turn(
    isolated_home, tmp_path, fake_tokenizer
):
    dataset_path = tmp_path / "filtered_codex_trace.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "conversations": [
                    {"from": "human", "value": "tok2"},
                    {"from": "gpt", "value": "tok2"},
                    {"from": "human", "value": "tok6"},
                    {"from": "gpt", "value": "tok2"},
                    {"from": "human", "value": "tok1"},
                    {"from": "gpt", "value": "tok2"},
                ]
            }
        )
        + "\n"
    )

    rows = sample_agentic_trace_requests(
        str(dataset_path), fake_tokenizer, context_len=8, return_text=True
    )

    assert [row.simulation.trace_request_id for row in rows] == ["0:0"]
    assert rows[0].prompt_len == 2
    assert rows[0].output_len == 2
    capped_prompt = fake_tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "tok2"},
            {"role": "assistant", "content": "tok2"},
            {"role": "user", "content": "tok6"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    capped_turn_len = len(fake_tokenizer.encode(capped_prompt)) + len(
        fake_tokenizer.encode("tok2")
    )
    assert capped_turn_len == 12
    assert capped_turn_len > 8
    assert all(row.prompt_len + row.output_len <= 8 for row in rows)
    assert rows[0].simulation.trace_prev_request_id is None


def test_get_request_preserves_trace_metadata(monkeypatch):
    bench_serving.args = argparse.Namespace(bench_mode="simulation")
    simulation = SimulationParams(
        trace_session_id=0,
        trace_turn_index=0,
        trace_request_id="0:0",
        trace_prev_request_id=None,
    )
    rows = [
        DatasetRow(
            prompt="hello world",
            prompt_len=2,
            output_len=2,
            simulation=simulation,
        )
    ]
    original_id = id(rows[0].simulation)

    async def collect():
        seen = []
        async for row in get_request(rows, float("inf")):
            seen.append(row)
        return seen

    seen_rows = asyncio.run(collect())

    assert seen_rows == rows
    assert id(rows[0].simulation) == original_id
    assert rows[0].simulation.trace_request_id == "0:0"
    assert rows[0].simulation.trace_prev_request_id is None
    assert rows[0].simulation.created_time == 0
    assert rows[0].simulation.total_request == 1


def test_agentic_trace_stale_cache_is_removed_and_regenerated(
    isolated_home, codex_trace_path, fake_tokenizer
):
    cache_path = get_agentic_trace_cache_path(
        str(codex_trace_path),
        fake_tokenizer,
        num_requests=None,
        context_len=None,
        return_text=True,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(
            [DatasetRow(prompt="stale prompt", prompt_len=2, output_len=2)],
            f,
        )

    rows = sample_agentic_trace_requests(
        str(codex_trace_path), fake_tokenizer, return_text=True
    )

    assert len(rows) == 3
    assert rows[0].prompt != "stale prompt"
    assert all(row.simulation.trace_request_id is not None for row in rows)
    assert cache_path.exists()

    with open(cache_path, "rb") as f:
        cached_rows = pickle.load(f)
    assert len(cached_rows) == 3
    assert all(row.simulation.trace_request_id is not None for row in cached_rows)
