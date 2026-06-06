#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Benchmark hisim simulation server using random requests (input=30000, output=1024, zrange=1).
Sweeps over all combinations of hicache-size × bandwidth × request-rate.
Starts a fresh server for each combination (HISIM_RESET_HICACHE_STORAGE=1).

Options:
  --port PORT                   Server port (default: 12345).
  --rates RATES                 Comma-separated request rates (default: 1).
  --hicache-size SIZES          Comma-separated L2 DRAM cache sizes in GB (default: 500).
  --hicache-rw-bandwidth BWS    Comma-separated DRAM read+write bandwidths in GB/s (default: 64).
  --output-dir DIR              Directory for results (default: ./random_bench_metrics).
  -h, --help                    Show this help message and exit.

Output:
  OUTPUT_DIR/DramSize<SIZE>gB_DramBw<BW>gB_<RATE>RPS.json
  OUTPUT_DIR/logs/DramSize<SIZE>gB_DramBw<BW>gB_<RATE>RPS_server.log
  OUTPUT_DIR/logs/DramSize<SIZE>gB_DramBw<BW>gB_<RATE>RPS_bench.log
EOF
}

PORT=12345
RATES="1"
HICACHE_SIZES="500"
HICACHE_BWS="64"
OUTPUT_DIR="${SCRIPT_DIR}/random_bench_metrics"
SERVER_PID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)                PORT="$2";          shift 2 ;;
    --rates)               RATES="$2";         shift 2 ;;
    --hicache-size)        HICACHE_SIZES="$2"; shift 2 ;;
    --hicache-rw-bandwidth) HICACHE_BWS="$2"; shift 2 ;;
    --output-dir)          OUTPUT_DIR="$2";    shift 2 ;;
    -h|--help)             usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

IFS=',' read -ra request_rates <<< "$RATES"
IFS=',' read -ra hicache_sizes <<< "$HICACHE_SIZES"
IFS=',' read -ra hicache_bws   <<< "$HICACHE_BWS"

start_server() {
  local size="$1"
  local bw="$2"
  local log_file="$3"
  setsid "${SCRIPT_DIR}/h100-launch-server.sh" \
    --port "${PORT}" \
    --hicache-size "${size}" \
    --read-bw "${bw}" \
    --write-bw "${bw}" \
    > "${log_file}" 2>&1 &
  SERVER_PID=$!
  until curl -sf "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; do
    kill -0 "${SERVER_PID}" 2>/dev/null || { echo "Server exited unexpectedly. See ${log_file}"; exit 1; }
    sleep 2
  done
}

stop_server() {
  [[ -z "${SERVER_PID}" ]] && return
  kill -INT -- -"${SERVER_PID}" 2>/dev/null || true
  ( sleep 10; kill -KILL -- -"${SERVER_PID}" 2>/dev/null || true ) &
  local watchdog=$!
  wait "${SERVER_PID}" 2>/dev/null || true
  kill "${watchdog}" 2>/dev/null || true
  wait "${watchdog}" 2>/dev/null || true
  SERVER_PID=""
}

bench() {
  local rate="$1"
  local output_file="$2"
  local log_file="$3"
  python3 -m hisim.simulation.bench_serving \
    --backend sglang \
    --port "${PORT}" \
    --dataset-name random \
    --random-input-len 30000 \
    --random-output-len 1024 \
    --random-range-ratio 1 \
    --num-prompts 10 \
    --request-rate "${rate}" \
    --bench-mode simulation \
    --warmup-requests 0 \
    --output-file "${output_file}" \
    2>&1 | tee "${log_file}"
}

TOTAL_RUNS=$(( ${#hicache_sizes[@]} * ${#hicache_bws[@]} * ${#request_rates[@]} ))
CURRENT_RUN=0

trap 'stop_server' EXIT
trap 'stop_server; exit 130' INT TERM

for size in "${hicache_sizes[@]}"; do
  for bw in "${hicache_bws[@]}"; do
    for rate in "${request_rates[@]}"; do
      CURRENT_RUN=$(( CURRENT_RUN + 1 ))
      PREFIX="[${CURRENT_RUN}/${TOTAL_RUNS}] DramSize=${size}gB DramBw=${bw}gB rate=${rate}RPS"
      LOG_PREFIX="${OUTPUT_DIR}/logs/DramSize${size}gB_DramBw${bw}gB_${rate}RPS"
      mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/logs"
      echo "${PREFIX} — starting server..."
      start_server "${size}" "${bw}" "${LOG_PREFIX}_server.log"
      echo "The server is fired up and ready to roll!"
      echo "${PREFIX} — testing..."
      bench "${rate}" "${OUTPUT_DIR}/DramSize${size}gB_DramBw${bw}gB_${rate}RPS.json" "${LOG_PREFIX}_bench.log"
      echo "${PREFIX} — shutting down..."
      stop_server
    done
  done
done
