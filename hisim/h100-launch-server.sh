#!/bin/bash
set -e

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Launch hisim simulation server on a single H100.

Options:
  --model-path PATH     HuggingFace model path (default: Qwen/Qwen3-32B-FP8).
  --sim-config PATH     Sim config JSON path (default: test/assets/mock/config.qwen3_32b_fp8.h100.json).
  --port PORT           Server listening port (default: 12345).
  --hicache-size SIZE   L2 DRAM KV cache size in GB (default: 500).
                        Must be larger than the HBM KV cache pool.
  --max-running-requests N
                        Maximum number of concurrent running requests (default: unset).
  --max-total-tokens N  Maximum HBM KV cache tokens (default: unset; estimate
                        from the sim config).
  --page-size N         KV cache page size in tokens (default: 64).
  --read-bw BW          DRAM read bandwidth override in GB/s.
  --write-bw BW         DRAM write bandwidth override in GB/s.
  -h, --help            Show this help message and exit.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)   MODEL_PATH="$2";   shift 2 ;;
    --sim-config)   SIM_CONFIG="$2";   shift 2 ;;
    --hicache-size) HICACHE_SIZE="$2"; shift 2 ;;
    --max-running-requests) MAX_RUNNING_REQUESTS="$2"; shift 2 ;;
    --max-total-tokens) MAX_TOTAL_TOKENS="$2"; shift 2 ;;
    --page-size)   PAGE_SIZE="$2";   shift 2 ;;
    --port)         PORT="$2";         shift 2 ;;
    --read-bw)      READ_BW="$2";      shift 2 ;;
    --write-bw)     WRITE_BW="$2";     shift 2 ;;
    -h|--help)      usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

BW_ARGS=()
[ -n "${READ_BW}" ]  && BW_ARGS+=(--sim-memory-read-bandwidth-gb  "${READ_BW}")
[ -n "${WRITE_BW}" ] && BW_ARGS+=(--sim-memory-write-bandwidth-gb "${WRITE_BW}")
MAX_TOTAL_TOKENS_ARGS=()
[ -n "${MAX_TOTAL_TOKENS:-}" ] && MAX_TOTAL_TOKENS_ARGS+=(--max-total-tokens "${MAX_TOTAL_TOKENS}")
MAX_RUNNING_REQUESTS_ARGS=()
[ -n "${MAX_RUNNING_REQUESTS:-}" ] && MAX_RUNNING_REQUESTS_ARGS+=(--max-running-requests "${MAX_RUNNING_REQUESTS}")

# to use CPU:
#SGLANG_USE_CPU_ENGINE=1 \
#FLASHINFER_DISABLE_VERSION_CHECK=1 \
#LOG_LEVEL=DEBUG \
HISIM_RESET_HICACHE_STORAGE=1 \
HISIM_SKIP_HOST_MEMORY_CHECK=1 \
HISIM_FAKE_HOST_KV_BUFFER=1 \
python3 -m hisim.simulation.sglang.launch_server \
  --model-path "${MODEL_PATH:-Qwen/Qwen3-32B-FP8}" \
  --sim-config-path "${SIM_CONFIG:-test/assets/mock/config.qwen3_32b_fp8.h100.json}" \
  --skip-server-warmup \
  --port "${PORT:-12345}" \
  --enable-hierarchical-cache \
  --hicache-size "${HICACHE_SIZE:-500}" \
  "${MAX_RUNNING_REQUESTS_ARGS[@]}" \
  "${MAX_TOTAL_TOKENS_ARGS[@]}" \
  --page-size "${PAGE_SIZE:-64}" \
  "${BW_ARGS[@]}" \
# to enable L3: --hicache-storage-backend file \
# TODO: not supported on the current sglang version --prefill-max-requests 1 \
