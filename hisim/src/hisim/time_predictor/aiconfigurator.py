from hisim.spec.data_type import DataType
from hisim.spec.accelerator import AcceleratorInfo
from hisim.spec.model import ModelInfo
import numpy as np
import os
import re
import sys
from typing import Optional

from aiconfigurator.sdk import models
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.common import (
    CommQuantMode,
    FMHAQuantMode,
    KVCacheQuantMode,
    DatabaseMode,
)
from aiconfigurator.sdk.config import RuntimeConfig, ModelConfig
from aiconfigurator.sdk.inference_session import InferenceSession
from aiconfigurator.sdk.perf_database import get_database

from hisim.simulation.types import (
    SchedulerConfig,
)
from hisim.time_predictor import (
    InferTimePredictor,
    ScheduleBatch,
    FakeRequest,
)
from hisim.utils import get_logger


MAP_DTYPE_TO_KVCacheQuantMode = {
    DataType.FP16: KVCacheQuantMode.bfloat16,
    DataType.BF16: KVCacheQuantMode.bfloat16,
    DataType.FP8: KVCacheQuantMode.fp8,
    DataType.INT8: KVCacheQuantMode.int8,
}

MAP_DTYPE_TO_FMHAQuantMode = {
    DataType.FP16: FMHAQuantMode.bfloat16,
    DataType.BF16: FMHAQuantMode.bfloat16,
    DataType.FP8: FMHAQuantMode.fp8,
}

MAP_DTYPE_TO_CommQunatMode = {
    DataType.FP16: CommQuantMode.half,
    DataType.BF16: CommQuantMode.half,
    DataType.FP8: CommQuantMode.fp8,
    DataType.INT8: CommQuantMode.int8,
}


logger = get_logger("hisim")


# XGBoost
def _import_xgboost():
    """
    Try importing a usable XGBoost Python package.

    Note: This repository vendors the XGBoost source tree at `<repo_root>/xgboost/`, which can shadow
    the real Python package depending on cwd/sys.path. We try:
    - Normal import
    - Repository-local Python package import (adds `<repo_root>/xgboost/python-package` to sys.path)
    """

    def _is_usable(mod) -> bool:
        return hasattr(mod, "XGBRegressor") and hasattr(mod, "DMatrix")

    try:
        import xgboost as _xgb  # type: ignore

        if _is_usable(_xgb):
            return _xgb
    except Exception:
        pass

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    local_py_pkg = os.path.join(repo_root, "xgboost", "python-package")
    if os.path.isdir(local_py_pkg) and local_py_pkg not in sys.path:
        sys.path.insert(0, local_py_pkg)

    import xgboost as _xgb2  # type: ignore

    if not _is_usable(_xgb2):
        raise RuntimeError(
            "Imported 'xgboost' but it doesn't look like the python package (missing XGBRegressor/DMatrix). "
            "If you intended to use pip/conda xgboost, install it; if you intended to use the vendored source tree, "
            "make sure it is built properly."
        )
    return _xgb2


def _parse_bucket_from_model_path(path: str) -> tuple[int, int]:
    """
    Parse bucket range from model filename.
    Expected: '*bs<lo>_<hi>*' or '*bs<lo>-<hi>*' or '*bs<lo>to<hi>*'
    """
    base = os.path.basename(path)
    m = re.search(r"bs\s*(\d+)\s*[_\-]\s*(\d+)", base, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"bs\s*(\d+)\s*to\s*(\d+)", base, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse bucket range from model filename: {base}")
    lo = int(m.group(1))
    hi = int(m.group(2))
    if lo <= 0 or hi <= 0 or lo > hi:
        raise ValueError(f"Invalid bucket range in model filename: {base}")
    return lo, hi


def _load_bucket_models(model_dir: str):
    """
    Load all bucket models from a directory.
    Returns list of (lo, hi, reg) sorted by bucket width then lo.
    """
    if not model_dir or not os.path.isdir(model_dir):
        return []
    xgb = _import_xgboost()
    models = []
    for fn in os.listdir(model_dir):
        if not (fn.endswith(".json") or fn.endswith(".model")):
            continue
        p = os.path.join(model_dir, fn)
        try:
            lo, hi = _parse_bucket_from_model_path(p)
        except Exception:
            continue
        reg = xgb.XGBRegressor()
        reg.load_model(p)
        models.append((lo, hi, reg, p))
        logger.info(f"Loaded bucket XGBoost model bs{lo}_{hi}: {p}")
    models.sort(key=lambda t: ((t[1] - t[0]), t[0], t[1]))
    return models


def _pick_bucket_model(models, bs: int):
    for lo, hi, reg, p in models:
        if lo <= bs <= hi:
            return lo, hi, reg, p
    return None


def _build_xgb_feature_maxbs_2(reqs: list[FakeRequest], *, max_bs: int) -> np.ndarray:
    """
    Feature shape (max_bs, 2):
    - rows [0..max_bs-1]: if request exists -> [1, past_kv_length] else [0, 0]
    - NOTE: AIC latency is NOT included in features for ratio model.
    """
    if max_bs <= 0:
        raise ValueError("max_bs must be > 0")
    feats = np.zeros((max_bs, 2), dtype=np.float32)
    n = min(len(reqs), max_bs)
    for i in range(n):
        feats[i, 0] = 1.0
        feats[i, 1] = float(reqs[i].past_kv_length)
    return feats


def _xgb_predict_ratio(reqs: list[FakeRequest], xgb_reg, *, max_bs: int) -> float:
    if len(reqs) > max_bs:
        raise ValueError(
            f"batch_size={len(reqs)} > max_bs={max_bs} is not supported by the (max_bs,2) feature spec."
        )
    x = (
        _build_xgb_feature_maxbs_2(reqs, max_bs=max_bs)
        .reshape(-1)
        .astype(np.float32, copy=False)
    )
    pred_ratio = float(xgb_reg.predict(x.reshape(1, -1))[0])
    pred_ratio = max(pred_ratio, 1e-6)
    return pred_ratio


def _pick_generation_attn_key(latency_dict: dict) -> str:
    if not latency_dict:
        raise ValueError("Empty latency_dict")

    exact_candidates = [
        "generation_attn",
        "gen_attn",
        "gen_attention",
        "generation_attention",
        "attn",
    ]
    for k in exact_candidates:
        if k in latency_dict:
            return k

    keys = list(latency_dict.keys())
    gen_attn = [
        k for k in keys if ("attn" in str(k).lower() and "gen" in str(k).lower())
    ]
    if gen_attn:
        return max(gen_attn, key=lambda kk: float(latency_dict.get(kk, 0.0)))

    attn_keys = [k for k in keys if "attn" in str(k).lower()]
    if attn_keys:
        return max(attn_keys, key=lambda kk: float(latency_dict.get(kk, 0.0)))

    raise KeyError(f"Cannot find generation attn key from latency_dict keys={keys}")


def _split_gen_attn_other(latency_dict: dict) -> tuple[str, float, float, float]:
    gen_k = _pick_generation_attn_key(latency_dict)
    gen_ms = float(latency_dict[gen_k])
    total_ms = float(sum(float(v) for v in latency_dict.values()))
    other_ms = float(total_ms - gen_ms)
    return gen_k, gen_ms, other_ms, total_ms


def _parse_decode_bs_range_from_xgb_model_path(
    xgb_decode_model_path: str,
) -> tuple[int, int]:
    """
    Parse decode batch size range from xgboost model filename.

    Expected patterns (examples):
    - xgb_decode_bs6_32.json  -> (6, 32)
    - .../xgb_decode_bs6-32.model -> (6, 32)
    - .../xgb_decode_bs6to32.json -> (6, 32)

    If not found, returns (-1, -1).
    """
    base = os.path.basename(xgb_decode_model_path)
    m = re.search(r"bs\s*(\d+)\s*[_\-]\s*(\d+)", base, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"bs\s*(\d+)\s*to\s*(\d+)", base, flags=re.IGNORECASE)
    if not m:
        return -1, -1
    lo = int(m.group(1))
    hi = int(m.group(2))
    if lo <= 0 or hi <= 0 or lo > hi:
        return -1, -1
    return lo, hi


def _resolve_aic_model_path(model: ModelInfo) -> str:
    if model.model_path:
        model_path = model.model_path
        if os.path.isfile(model_path):
            model_path = os.path.dirname(model_path)
        return model_path

    model_name = model.name
    if model.model_type == "gpt_oss" and "/" not in model_name:
        return "openai/gpt-oss-120b" if model.n_routed_experts >= 64 else "openai/gpt-oss-20b"

    if model_name:
        return {
            "GPT_OSS_120B": "openai/gpt-oss-120b",
            "GPT_OSS_20B": "openai/gpt-oss-20b",
        }.get(model_name, model_name)

    raise ValueError("ModelInfo must provide either model_path or name for AIConfigurator.")


def _is_moe_model(model: ModelInfo) -> bool:
    return bool(
        getattr(model, "n_routed_experts", 0)
        or getattr(model, "num_experts_per_tok", 0)
        or getattr(model, "moe_intermediate_size", 0)
    )


def _resolve_aic_moe_parallelism(
    sched_config: SchedulerConfig, model: ModelInfo
) -> tuple[int, int]:
    tp_size = sched_config.tp_size or 1
    dp_size = sched_config.dp_size or 1
    ep_size = sched_config.ep_size or 1

    if not _is_moe_model(model):
        return tp_size, ep_size

    # Keep attention TP unchanged, but model MoE weights as EP-only. This satisfies
    # AIC's tp * dp == moe_tp * moe_ep constraint and avoids splitting FP8 block-
    # scaled expert weights across an unaligned MoE TP shard.
    return 1, tp_size * dp_size


def _config_value(value):
    return value.name if hasattr(value, "name") else value


def _log_aic_model_config(
    model: ModelInfo,
    model_path: str,
    sched_config: SchedulerConfig,
    model_config: ModelConfig,
) -> None:
    logger.info(
        "AIConfigurator model config: "
        "model=%s model_path=%s backend=%s backend_version=%s "
        "pp_size=%s tp_size=%s moe_tp_size=%s moe_ep_size=%s "
        "attention_dp_size=%s attention_cp_size=%s "
        "gemm_quant_mode=%s moe_quant_mode=%s kvcache_quant_mode=%s "
        "fmha_quant_mode=%s comm_quant_mode=%s workload_distribution=%s",
        getattr(model, "name", ""),
        model_path,
        sched_config.backend_name,
        sched_config.backend_version,
        model_config.pp_size,
        model_config.tp_size,
        model_config.moe_tp_size,
        model_config.moe_ep_size,
        model_config.attention_dp_size,
        getattr(model_config, "attention_cp_size", None),
        _config_value(model_config.gemm_quant_mode),
        _config_value(model_config.moe_quant_mode),
        _config_value(model_config.kvcache_quant_mode),
        _config_value(model_config.fmha_quant_mode),
        _config_value(model_config.comm_quant_mode),
        model_config.workload_distribution,
    )


def get_perf_model(sched_config: SchedulerConfig, model: ModelInfo) -> models.BaseModel:
    # Leave model weight quantization unset. AIC 0.9 infers fp8/fp8_block/fp8_static/nvfp4
    # from the raw model config; mapping HiSim DataType.FP8 here would incorrectly erase
    # the distinction between per-tensor FP8 and block-scaled FP8.
    moe_tp_size, moe_ep_size = _resolve_aic_moe_parallelism(sched_config, model)
    model_config = ModelConfig(
        pp_size=sched_config.pp_size,
        tp_size=sched_config.tp_size,
        moe_tp_size=moe_tp_size,
        moe_ep_size=moe_ep_size,
        attention_dp_size=sched_config.dp_size,  # FIXME
        kvcache_quant_mode=MAP_DTYPE_TO_KVCacheQuantMode.get(
            sched_config.kv_cache_data_type
        ),
        fmha_quant_mode=MAP_DTYPE_TO_FMHAQuantMode.get(
            sched_config.kv_cache_data_type
        ),
        comm_quant_mode=MAP_DTYPE_TO_CommQunatMode.get(
            sched_config.data_type, CommQuantMode.half
        ),
        workload_distribution="power_law_1.2",
    )

    model_path = _resolve_aic_model_path(model)
    perf_model = models.get_model(
        model_path=model_path,
        model_config=model_config,
        backend_name=sched_config.backend_name,
    )
    _log_aic_model_config(model, model_path, sched_config, model_config)
    return perf_model


class AIConfiguratorTimePredictor(InferTimePredictor):
    def __init__(
        self,
        model: ModelInfo,
        hw: AcceleratorInfo,
        config: SchedulerConfig,
        database_path: Optional[str] = None,
        database_mode: DatabaseMode | str = DatabaseMode.SILICON,
        xgb_model_path: Optional[str] = None,
        prefill_scale_factor: float = 1,
        decode_scale_factor: float = 1,
    ):
        super().__init__(model, hw, config)

        self.prefill_scale_factor = prefill_scale_factor
        self.decode_scale_factor = decode_scale_factor

        if isinstance(database_mode, str):
            database_mode = self._get_database_mode(database_mode)

        logger.info(
            "AIConfigurator predictor runtime config: "
            "model=%s hw=%s backend=%s backend_version=%s "
            "max_running_requests=%s max_prefill_tokens=%s "
            "chunked_prefill_size=%s page_size=%s "
            "database_mode=%s "
            "prefill_scale_factor=%s decode_scale_factor=%s",
            model.name,
            hw.name,
            config.backend_name,
            config.backend_version,
            config.max_running_requests,
            config.max_prefill_tokens,
            config.chunked_prefill_size,
            config.page_size,
            database_mode,
            self.prefill_scale_factor,
            self.decode_scale_factor,
        )

        database = get_database(
            system=hw.name,
            backend=config.backend_name,
            version=config.backend_version,
            systems_paths=database_path,
            database_mode=database_mode.name,
        )

        if database is None:
            raise ValueError("Failed to initialize the database.")

        database.set_default_database_mode(database_mode)

        self._session = InferenceSession(
            model=get_perf_model(config, model),
            backend=get_backend(self.config.backend_name),
            database=database,
        )

        # Load XGBoost bucket models
        self.xgb_bucket_models = []
        if xgb_model_path is not None:
            self.xgb_bucket_models = _load_bucket_models(xgb_model_path)

    def _get_database_mode(self, mode: str) -> DatabaseMode:
        return {
            "SILICON": DatabaseMode.SILICON,
            "HYBRID": DatabaseMode.HYBRID,
            "EMPIRICAL": DatabaseMode.EMPIRICAL,
            "SOL": DatabaseMode.SOL,
            "SOL_FULL": DatabaseMode.SOL_FULL,
        }.get(mode.upper(), DatabaseMode.SILICON)

    def ctx_attn_flops_ratio_with_avg(self, reqs: list[FakeRequest]) -> float:
        if len(reqs) == 1:
            return 1.0
        mean_past = np.mean([req.past_kv_length for req in reqs])
        mean_input = np.mean([req.input_length for req in reqs])
        avg_flops = (mean_past + mean_past + mean_input) * mean_input / 2 * len(reqs)

        actual_flops = 0
        for req in reqs:
            actual_flops += (
                (req.past_kv_length + req.past_kv_length + req.input_length)
                * req.input_length
                / 2
            )

        return actual_flops / avg_flops

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        infer_time = 0
        if batch.is_decode():
            # Decode: output sequence length (osl) = 2, input sequence length (isl) = mean(past_kv_length)
            isl = int(np.mean([req.past_kv_length for req in batch.reqs]))
            gen_attn_scale = -1
            # Try to use XGBoost bucket model
            if len(self.xgb_bucket_models) > 0:
                picked = _pick_bucket_model(self.xgb_bucket_models, batch.batch_size)
                if picked is not None:
                    _lo, _hi, _reg, _p = picked
                    xgb_reg = _reg
                    xgb_max_bs = int(_hi)

                    # If XGBoost is enabled, it predicts the ratio: aic_gen_attn_ms / measured_gen_attn_ms.
                    # To make AIC's generation attention closer to measured, scale it by (1 / pred_ratio).
                    pred_ratio = _xgb_predict_ratio(
                        batch.reqs, xgb_reg, max_bs=xgb_max_bs
                    )
                    gen_attn_scale = 1.0 / pred_ratio

            if gen_attn_scale < 0:
                runtime_config = RuntimeConfig(
                    batch_size=batch.batch_size, isl=isl, osl=2
                )
            else:
                runtime_config = RuntimeConfig(
                    batch_size=batch.batch_size,
                    isl=isl,
                    osl=2,
                    gen_seq_imbalance_correction_scale=float(gen_attn_scale),
                )
            summary = self._session.run_static(runtime_config, mode="static_gen")
            latency_dict = summary.get_generation_latency_dict()

        else:
            # Prefill: output sequence length (osl) = 1, input sequence length (isl) = mean(past_kv + input), prefix = mean(past_kv)
            mean_past = np.mean([req.past_kv_length for req in batch.reqs])
            mean_input = np.mean([req.input_length for req in batch.reqs])
            isl = int(mean_past + mean_input)
            prefix = int(mean_past)
            runtime_config = RuntimeConfig(
                batch_size=batch.batch_size, isl=isl, prefix=prefix, osl=1
            )

            seq_imbalance_correction_scale = self.ctx_attn_flops_ratio_with_avg(
                batch.reqs
            )
            if seq_imbalance_correction_scale >= 0.4:
                runtime_config = RuntimeConfig(
                    batch_size=batch.batch_size,
                    isl=isl,
                    prefix=prefix,
                    osl=1,
                    seq_imbalance_correction_scale=seq_imbalance_correction_scale,
                )
            else:
                runtime_config = RuntimeConfig(
                    batch_size=batch.batch_size, isl=isl, prefix=prefix, osl=1
                )

            summary = self._session.run_static(runtime_config, mode="static_ctx")
            latency_dict = summary.get_context_latency_dict()
        infer_time = sum(latency_dict.values())
        if summary.check_oom():
            logger.warning(
                f"OOM detected: batch_size={batch.batch_size}, isl={isl}, "
                f"prefix={prefix if not batch.is_decode() else 'N/A'}"
            )
            # FIXME: Negative latency doesn't make sense
            infer_time = -infer_time
        if batch.is_decode():
            infer_time *= self.decode_scale_factor
        else:
            infer_time *= self.prefill_scale_factor
        return infer_time / 1e3
