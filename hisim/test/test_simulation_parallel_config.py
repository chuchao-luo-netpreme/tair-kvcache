import argparse
import json
import os
import tempfile
from types import SimpleNamespace

import hisim.simulation.manager.config as config_mod
from hisim.simulation.manager.config import ConfigManager
from hisim.simulation.sim_args import SimulationArgs
from hisim.simulation.types import SchedulerConfig
from hisim.simulation.utils import calc_attention_tp_size, calc_kv_cache_cell_elems


def make_model(**overrides):
    values = {
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "kv_lora_rank": 0,
        "qk_rope_head_dim": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_attention_tp_size_uses_simulated_tp_size():
    assert calc_attention_tp_size(8) == 8


def test_kv_cache_cell_elems_uses_simulated_tp_size_when_dp_attention_disabled():
    model = make_model()

    no_dp_attention = calc_kv_cache_cell_elems(
        model, tp_size=8, pp_size=1
    )

    assert no_dp_attention == 1 * 128 * 32 * 2


def test_config_manager_kv_cache_bytes_uses_simulated_tp_size():
    model = make_model()
    dtype = SimpleNamespace(bytes=2)
    scheduler_config = SchedulerConfig(
        model=None,
        tp_size=8,
        dp_size=4,
        pp_size=1,
        data_type=dtype,
    )

    old_model_info = ConfigManager._model_info
    old_scheduler_config = ConfigManager._scheduler_config
    try:
        ConfigManager._model_info = model
        ConfigManager._scheduler_config = scheduler_config
        assert ConfigManager.get_kv_cache_bytes() == 1 * 128 * 32 * 2 * 2
    finally:
        ConfigManager._model_info = old_model_info
        ConfigManager._scheduler_config = old_scheduler_config


def test_sim_args_accepts_parallel_scheduler_overrides():
    parser = argparse.ArgumentParser()
    SimulationArgs.add_cli_args(parser)

    ns = parser.parse_args(
        [
            "--sim-tp-size",
            "8",
            "--sim-dp-size",
            "4",
        ]
    )

    args = SimulationArgs.from_cli_args(ns)

    assert args.scheduler.tp_size == 8
    assert args.scheduler.dp_size == 4


def test_scheduler_parallel_config_overrides_fields_independently():
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        json.dump(
            {
                "scheduler": {
                    "tp_size": 8,
                    "data_type": "FP16",
                    "kv_cache_data_type": "FP16",
                }
            },
            f,
        )
        config_path = f.name

    old_config_path = config_mod.Envs.__dict__["config_path"]
    old_get_model_info = ConfigManager.__dict__["get_model_info"]
    try:
        config_mod.Envs.config_path = classmethod(lambda cls: config_path)
        ConfigManager.get_model_info = classmethod(
            lambda cls, hf_config: make_model(torch_dtype=None)
        )

        config = ConfigManager.get_scheduler_config(
            {
                "tp_size": 2,
                "dp_size": 2,
            },
            "sglang",
            {},
        )
    finally:
        config_mod.Envs.config_path = old_config_path
        ConfigManager.get_model_info = old_get_model_info
        os.unlink(config_path)

    assert config.tp_size == 8
    assert config.dp_size == 2


def test_internal_config_does_not_carry_dp_attention():
    internal_config = ConfigManager._parse_server_args(
        {"tp_size": 2, "dp_size": 2, "enable_dp_attention": False},
        "sglang",
    )

    assert not hasattr(internal_config, "enable_dp_attention")

    try:
        ConfigManager._parse_server_args(
            {"tp_size": 2, "dp_size": 2, "enable_dp_attention": True},
            "sglang",
        )
        assert False, "expected AssertionError for internal SGLang DP attention"
    except AssertionError:
        pass


def test_scheduler_rejects_sglang_dp_attention():
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        json.dump({"scheduler": {"tp_size": 8}}, f)
        config_path = f.name

    old_config_path = config_mod.Envs.__dict__["config_path"]
    old_get_model_info = ConfigManager.__dict__["get_model_info"]
    try:
        config_mod.Envs.config_path = classmethod(lambda cls: config_path)
        ConfigManager.get_model_info = classmethod(
            lambda cls, hf_config: make_model(torch_dtype=None)
        )

        try:
            ConfigManager.get_scheduler_config(
                {"tp_size": 2, "dp_size": 2, "enable_dp_attention": True},
                "sglang",
                {},
            )
            assert False, "expected AssertionError for SGLang DP attention"
        except AssertionError:
            pass
    finally:
        config_mod.Envs.config_path = old_config_path
        ConfigManager.get_model_info = old_get_model_info
        os.unlink(config_path)


def test_scheduler_rejects_hisim_dp_attention():
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        json.dump({"scheduler": {"tp_size": 8, "enable_dp_attention": True}}, f)
        config_path = f.name

    old_config_path = config_mod.Envs.__dict__["config_path"]
    old_get_model_info = ConfigManager.__dict__["get_model_info"]
    try:
        config_mod.Envs.config_path = classmethod(lambda cls: config_path)
        ConfigManager.get_model_info = classmethod(
            lambda cls, hf_config: make_model(torch_dtype=None)
        )

        try:
            ConfigManager.get_scheduler_config(
                {"tp_size": 2, "dp_size": 1, "enable_dp_attention": False},
                "sglang",
                {},
            )
            assert False, "expected AssertionError for HiSim DP attention"
        except AssertionError:
            pass
    finally:
        config_mod.Envs.config_path = old_config_path
        ConfigManager.get_model_info = old_get_model_info
        os.unlink(config_path)


if __name__ == "__main__":
    test_attention_tp_size_uses_simulated_tp_size()
    test_kv_cache_cell_elems_uses_simulated_tp_size_when_dp_attention_disabled()
    test_config_manager_kv_cache_bytes_uses_simulated_tp_size()
    test_sim_args_accepts_parallel_scheduler_overrides()
    test_scheduler_parallel_config_overrides_fields_independently()
    test_internal_config_does_not_carry_dp_attention()
    test_scheduler_rejects_sglang_dp_attention()
    test_scheduler_rejects_hisim_dp_attention()
