#!/bin/bash
set -e

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Launch hisim simulation server for Qwen3-32B on a single H100.

Options:
  --port PORT           Server listening port (default: 12345).
  --hicache-size SIZE   L2 DRAM KV cache size in GB (default: 500).
                        Must be larger than the HBM KV cache pool.
  --read-bw BW          DRAM read bandwidth override in GB/s.
                        If not set, falls back to memory_read_bandwidth_gb
                        in config.qwen3_32b.h100.json (default: 64).
  --write-bw BW         DRAM write bandwidth override in GB/s.
                        If not set, falls back to memory_write_bandwidth_gb
                        in config.qwen3_32b.h100.json (default: 64).
  -h, --help            Show this help message and exit.

Sim config: test/assets/mock/config.qwen3_32b.h100.json
  Accelerator : H100 SXM (80 GB HBM, 3350 GB/s)
  TP size     : 1
  Data type   : FP16
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hicache-size) HICACHE_SIZE="$2"; shift 2 ;;
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

HISIM_RESET_HICACHE_STORAGE=1 \
SGLANG_USE_CPU_ENGINE=1 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
python3 -m hisim.simulation.sglang.launch_server \
  --model-path "Qwen/Qwen3-32B-FP8" \
  --sim-config-path test/assets/mock/config.qwen3_32b_fp8.h100.json \
  --skip-server-warmup \
  --port "${PORT:-12345}" \
  --enable-hierarchical-cache \
  --hicache-size "${HICACHE_SIZE:-500}" \
  --hicache-storage-backend file \
  "${BW_ARGS[@]}"
