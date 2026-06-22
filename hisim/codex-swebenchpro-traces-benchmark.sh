#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Benchmark hisim simulation server using the codex-swebenchpro-traces dataset
(multi-turn agentic traces from Inferact/codex_swebenchpro_traces on HuggingFace).
Sweeps over all combinations of hicache-size × bandwidth × request-rate.
Starts a fresh server for each combination (HISIM_RESET_HICACHE_STORAGE=1).

Options:
  --port PORT                   Server port (default: 12345).
  --rates RATES                 Comma-separated request rates (default: 1).
  --num-prompts N               Maximum unfolded assistant turns to benchmark (default: 100).
  --agentic-trace-context-len N  Drop turns where input+output exceeds N tokens (default: unset).
  --hicache-size SIZES          Comma-separated L2 DRAM cache sizes in GB (default: 500).
  --hicache-rw-bandwidth BWS    Comma-separated DRAM read+write bandwidths in GB/s (default: 64).
  --max-total-tokens N          Maximum HBM KV cache tokens (default: unset).
  --page-size N                 KV cache page size in tokens (default: 64).
  --output-dir DIR              Directory for results (default: ./codex_bench_metrics).
  -h, --help                    Show this help message and exit.

Output:
  OUTPUT_DIR/runs/DramSize<SIZE>gB_DramBw<BW>gB_<RATE>RPS/metrics.json
  OUTPUT_DIR/runs/DramSize<SIZE>gB_DramBw<BW>gB_<RATE>RPS/request.jsonl
  OUTPUT_DIR/runs/DramSize<SIZE>gB_DramBw<BW>gB_<RATE>RPS/iteration.jsonl
  OUTPUT_DIR/logs/DramSize<SIZE>gB_DramBw<BW>gB_<RATE>RPS_server.log
  OUTPUT_DIR/logs/DramSize<SIZE>gB_DramBw<BW>gB_<RATE>RPS_bench.log
EOF
}

PORT=12345
RATES="1"
NUM_PROMPTS="100"
AGENTIC_TRACE_CONTEXT_LEN=""
HICACHE_SIZES="500"
HICACHE_BWS="64"
MAX_RUNNING_REQUESTS=""
MAX_TOTAL_TOKENS=""
PAGE_SIZE="64"
OUTPUT_DIR="${SCRIPT_DIR}/codex_bench_metrics"
SERVER_PID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)                PORT="$2";         shift 2 ;;
    --rates)               RATES="$2";        shift 2 ;;
    --num-prompts)         NUM_PROMPTS="$2";  shift 2 ;;
    --agentic-trace-context-len) AGENTIC_TRACE_CONTEXT_LEN="$2"; shift 2 ;;
    --hicache-size)        HICACHE_SIZES="$2"; shift 2 ;;
    --hicache-rw-bandwidth) HICACHE_BWS="$2"; shift 2 ;;
    --max-total-tokens)    MAX_TOTAL_TOKENS="$2"; shift 2 ;;
    --page-size)           PAGE_SIZE="$2";    shift 2 ;;
    --output-dir)          OUTPUT_DIR="$2";   shift 2 ;;
    -h|--help)             usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

IFS=',' read -ra request_rates  <<< "$RATES"
IFS=',' read -ra hicache_sizes  <<< "$HICACHE_SIZES"
IFS=',' read -ra hicache_bws    <<< "$HICACHE_BWS"

start_server() {
  local size="$1"
  local bw="$2"
  local log_file="$3"
  local sim_output_dir="$4"
  local max_total_tokens_args=()
  if [[ -n "${MAX_TOTAL_TOKENS}" ]]; then
    max_total_tokens_args=(
      --max-total-tokens "${MAX_TOTAL_TOKENS}"
    )
  fi
  HISIM_OUTPUT_DIR="${sim_output_dir}" \
  setsid "${SCRIPT_DIR}/h100-launch-server.sh" \
    --model-path "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8" \
    --sim-config "test/assets/mock/config.qwen3_235b.h100.json" \
    --port "${PORT}" \
    --hicache-size "${size}" \
    --page-size "${PAGE_SIZE}" \
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
  local log_file="$2"
  local sim_output_dir="$3"
  local agentic_trace_context_len_args=()
  if [[ -n "${AGENTIC_TRACE_CONTEXT_LEN}" ]]; then
    agentic_trace_context_len_args=(
      --agentic-trace-context-len "${AGENTIC_TRACE_CONTEXT_LEN}"
    )
  fi
  HISIM_OUTPUT_DIR="${sim_output_dir}" \
  python3 -m hisim.simulation.bench_serving \
    --backend sglang \
    --port "${PORT}" \
    --dataset-name codex-swebenchpro-traces \
    --num-prompts "${NUM_PROMPTS}" \
    --request-rate "${rate}" \
    --bench-mode simulation \
    --warmup-requests 0 \
    --tokenize-prompt \
    "${agentic_trace_context_len_args[@]}" \
    --output-file /dev/null \
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
      RUN_NAME="DramSize${size}gB_DramBw${bw}gB_${rate}RPS"
      LOG_PREFIX="${OUTPUT_DIR}/logs/${RUN_NAME}"
      SIM_OUTPUT_DIR="${OUTPUT_DIR}/runs/${RUN_NAME}"
      mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/logs" "${SIM_OUTPUT_DIR}"
      echo "${PREFIX} — starting server..."
      start_server "${size}" "${bw}" "${LOG_PREFIX}_server.log" "${SIM_OUTPUT_DIR}"
      echo "${PREFIX} — testing..."
      bench "${rate}" "${LOG_PREFIX}_bench.log" "${SIM_OUTPUT_DIR}"
      echo "${PREFIX} — shutting down..."
      stop_server
    done
  done
done
