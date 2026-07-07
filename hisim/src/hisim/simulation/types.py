from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any, Optional, Union
from hisim.spec import ModelInfo, DataType, AcceleratorInfo


@dataclass
class SimulationParams:
    created_time: float = 0
    total_request: int = 0
    queue_start: Optional[float] = None
    server_created_time: Optional[float] = None
    trace_session_id: Optional[int] = None
    trace_turn_index: Optional[int] = None
    trace_request_id: Optional[str] = None
    trace_prev_request_id: Optional[str] = None
    original_created_time: Optional[float] = None
    dependency_ready_time: Optional[float] = None

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "SimulationParams":
        """Deserialize simulation metadata from an HTTP JSON payload."""
        if not isinstance(values, dict):
            raise TypeError(
                "SimulationParams.from_dict expects dict, "
                f"got {type(values).__name__}"
            )
        names = {dataclass_field.name for dataclass_field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in names})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkConfig:
    request_rate: float = float("inf")
    max_concurrency: Optional[int] = None
    with_queue_start: bool = (
        False  # For hisim hooks: include queue start time in replay schedule.
    )
    ignore_request_timestamp: bool = False


@dataclass
class SchedulerConfig:
    model: Union[ModelInfo, str]
    # For the default value, please refer to "https://docs.sglang.ai/backend/server_arguments.html".
    max_prefill_tokens: int = 16384
    chunked_prefill_size: Optional[int] = None
    data_type: Optional[DataType] = (
        None  # Data type for model weights and activations. If none is set, it will be automatically detected.
    )
    kv_cache_data_type: Optional[DataType] = None
    mem_fraction_static: Optional[float] = None
    hicache_storage_backend: Optional[str] = None
    hicache_storage_prefetch_policy: str = (
        "best_effort"  # choices: best_effort, wait_complete, timeout
    )
    schedule_policy: str = "fcfs"
    tp_size: int = 1
    ep_size: int = 1
    dp_size: int = 1
    pp_size: int = 1
    max_running_requests: Optional[int] = None
    page_size: Optional[int] = None

    # framework backend
    backend_name: str = "sglang"
    backend_version: Optional[str] = None


class MockSimulationMode(Enum):
    BLOCKING = "BLOCKING"
    OFFLINE = "OFFLINE"


@dataclass
class RequestStats:
    rid: str = ""
    last_event_time: float = 1.0
    input_length: int = 1
    output_length: int = 1
    final_reused_tokens: int = 0
    prefetch_complete_tokens: int = 0
    queue_start: float = -1
    queue_end: float = -1
    created_time: float = -1
    gen_token_latencies: list[float] = field(default_factory=list)

    def is_complete(self) -> bool:
        return True


@dataclass
class PlatformConfig:
    device: Union[AcceleratorInfo, str]
    # Storage configuration for hierarchical cache management.
    disk_capacity_gb: Optional[float] = None
    disk_read_bandwidth_gb: Optional[float] = None
    disk_write_bandwidth_gb: Optional[float] = None
    memory_capacity_gb: Optional[float] = None
    memory_read_bandwidth_gb: Optional[float] = None
    memory_write_bandwidth_gb: Optional[float] = None
    num_device_per_node: int = 8

    @property
    def disk_read_bandwidth(self):
        return (
            self.disk_read_bandwidth_gb * 1e9 if self.disk_read_bandwidth_gb else None
        )

    @property
    def disk_write_bandwidth(self):
        return (
            self.disk_write_bandwidth_gb * 1e9 if self.disk_write_bandwidth_gb else None
        )

    @property
    def memory_read_bandwidth(self):
        return (
            self.memory_read_bandwidth_gb * 1e9
            if self.memory_read_bandwidth_gb
            else None
        )

    @property
    def memory_write_bandwidth(self):
        return (
            self.memory_write_bandwidth_gb * 1e9
            if self.memory_write_bandwidth_gb
            else None
        )
