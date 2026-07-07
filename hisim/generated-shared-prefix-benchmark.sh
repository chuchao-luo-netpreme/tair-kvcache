#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Benchmark hisim simulation server using generated-shared-prefix requests.
Defaults are chosen to stress L2 offload/load: many long shared-prefix groups,
moderate repeats per group, short outputs, and a DRAM bandwidth sweep.
Sweeps over all combinations of hicache-size x bandwidth x request-rate.
Starts a fresh server for each combination (HISIM_RESET_HICACHE_STORAGE=1).

Options:
  --port PORT                   Server port (default: 12345).
  --model-path PATH             HuggingFace model path (default: Qwen/Qwen3-32B-FP8).
  --sim-config PATH             Sim config JSON path (default: test/assets/mock/config.qwen3_32b_fp8.h100.json).
  --rates RATES                 Comma-separated request rates (default: 1).
  --hicache-size SIZES          Comma-separated L2 DRAM cache sizes in GB (default: 350).
  --hicache-rw-bandwidth BWS    Comma-separated DRAM read+write bandwidths in GB/s (default: 64,128,256,512,900,1024).
  --max-running-requests N      Maximum concurrent running requests (default: unset).
  --gsp-num-groups N            Number of shared-prefix groups (default: 64).
  --gsp-prompts-per-group N     Number of prompts per group (default: 8).
  --gsp-system-prompt-len N     Shared system prompt length in tokens (default: 12000).
  --gsp-question-len N          Per-request question length in tokens (default: 256).
  --gsp-output-len N            Output length in tokens (default: 128).
  --gsp-range-ratio RATIO       Length randomization ratio (default: 1).
  --output-dir DIR              Directory for results (default: ./gsp_l2_load_bw_metrics).
  -h, --help                    Show this help message and exit.

Output:
  OUTPUT_DIR/runs/DramSize<SIZE>gB_DramBw<BW>gB_<RATE>RPS_Groups<GROUPS>_PerGroup<PER_GROUP>/metrics.json
  OUTPUT_DIR/runs/DramSize<SIZE>gB_DramBw<BW>gB_<RATE>RPS_Groups<GROUPS>_PerGroup<PER_GROUP>/request.jsonl
  OUTPUT_DIR/runs/DramSize<SIZE>gB_DramBw<BW>gB_<RATE>RPS_Groups<GROUPS>_PerGroup<PER_GROUP>/iteration.jsonl
  OUTPUT_DIR/logs/DramSize<SIZE>gB_DramBw<BW>gB_<RATE>RPS_Groups<GROUPS>_PerGroup<PER_GROUP>_server.log
  OUTPUT_DIR/logs/DramSize<SIZE>gB_DramBw<BW>gB_<RATE>RPS_Groups<GROUPS>_PerGroup<PER_GROUP>_bench.log
EOF
}

PORT=12345
MODEL_PATH="Qwen/Qwen3-32B-FP8"
SIM_CONFIG="test/assets/mock/config.qwen3_32b_fp8.h100.json"
RATES="1"
HICACHE_SIZES="350"
HICACHE_BWS="64,128,256,512,900,1024"
MAX_RUNNING_REQUESTS=""
GSP_NUM_GROUPS="64"
GSP_PROMPTS_PER_GROUP="8"
GSP_SYSTEM_PROMPT_LEN="12000"
GSP_QUESTION_LEN="256"
GSP_OUTPUT_LEN="128"
GSP_RANGE_RATIO="1"
OUTPUT_DIR="${SCRIPT_DIR}/gsp_l2_load_bw_metrics"
SERVER_PID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)                   PORT="$2";                  shift 2 ;;
    --model-path)             MODEL_PATH="$2";            shift 2 ;;
    --sim-config)             SIM_CONFIG="$2";            shift 2 ;;
    --rates)                  RATES="$2";                 shift 2 ;;
    --hicache-size)           HICACHE_SIZES="$2";         shift 2 ;;
    --hicache-rw-bandwidth)   HICACHE_BWS="$2";           shift 2 ;;
    --max-running-requests)   MAX_RUNNING_REQUESTS="$2";  shift 2 ;;
    --gsp-num-groups)         GSP_NUM_GROUPS="$2";        shift 2 ;;
    --gsp-prompts-per-group)  GSP_PROMPTS_PER_GROUP="$2"; shift 2 ;;
    --gsp-system-prompt-len)  GSP_SYSTEM_PROMPT_LEN="$2"; shift 2 ;;
    --gsp-question-len)       GSP_QUESTION_LEN="$2";      shift 2 ;;
    --gsp-output-len)         GSP_OUTPUT_LEN="$2";        shift 2 ;;
    --gsp-range-ratio)        GSP_RANGE_RATIO="$2";       shift 2 ;;
    --output-dir)             OUTPUT_DIR="$2";            shift 2 ;;
    -h|--help)                usage; exit 0 ;;
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
  local sim_output_dir="$4"
  local max_running_requests_args=()
  if [[ -n "${MAX_RUNNING_REQUESTS}" ]]; then
    max_running_requests_args=(
      --max-running-requests "${MAX_RUNNING_REQUESTS}"
    )
  fi
  HISIM_OUTPUT_DIR="${sim_output_dir}" \
    setsid "${SCRIPT_DIR}/h100-launch-server.sh" \
    --model-path "${MODEL_PATH}" \
    --sim-config "${SIM_CONFIG}" \
    --port "${PORT}" \
    --hicache-size "${size}" \
    --read-bw "${bw}" \
    --write-bw "${bw}" \
    "${max_running_requests_args[@]}" \
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
  HISIM_OUTPUT_DIR="${sim_output_dir}" \
    python3 -m hisim.simulation.bench_serving \
    --backend sglang \
    --port "${PORT}" \
    --dataset-name generated-shared-prefix \
    --gsp-num-groups "${GSP_NUM_GROUPS}" \
    --gsp-prompts-per-group "${GSP_PROMPTS_PER_GROUP}" \
    --gsp-system-prompt-len "${GSP_SYSTEM_PROMPT_LEN}" \
    --gsp-question-len "${GSP_QUESTION_LEN}" \
    --gsp-output-len "${GSP_OUTPUT_LEN}" \
    --gsp-range-ratio "${GSP_RANGE_RATIO}" \
    --request-rate "${rate}" \
    --bench-mode simulation \
    --warmup-requests 0 \
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
      PREFIX="[${CURRENT_RUN}/${TOTAL_RUNS}] DramSize=${size}gB DramBw=${bw}gB rate=${rate}RPS groups=${GSP_NUM_GROUPS} prompts_per_group=${GSP_PROMPTS_PER_GROUP}"
      RUN_NAME="DramSize${size}gB_DramBw${bw}gB_${rate}RPS_Groups${GSP_NUM_GROUPS}_PerGroup${GSP_PROMPTS_PER_GROUP}"
      LOG_PREFIX="${OUTPUT_DIR}/logs/${RUN_NAME}"
      SIM_OUTPUT_DIR="${OUTPUT_DIR}/runs/${RUN_NAME}"
      mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/logs" "${SIM_OUTPUT_DIR}"
      echo "${PREFIX} - starting server..."
      start_server "${size}" "${bw}" "${LOG_PREFIX}_server.log" "${SIM_OUTPUT_DIR}"
      echo "The server is ready."
      echo "${PREFIX} - testing..."
      bench "${rate}" "${LOG_PREFIX}_bench.log" "${SIM_OUTPUT_DIR}"
      echo "${PREFIX} - shutting down..."
      stop_server
    done
  done
done
