#!/bin/bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Benchmark hisim simulation server using the codex-swebenchpro-traces dataset
(multi-turn agentic traces from Inferact/codex_swebenchpro_traces on HuggingFace).
Runs one warmup pass to populate L2/L3 cache, then sweeps over request rates.

Options:
  --server-url URL    Base URL of the running hisim server (default: http://localhost:12345).
  --rates RATES       Comma-separated request rates to sweep (default: 1,2,4,8,16).
  --output-dir DIR    Directory to write per-rate results (default: ./data/codex-swebenchpro-traces).
  --skip-warmup       Skip the warmup pass.
  -h, --help          Show this help message and exit.

Cache warming:
  A single warmup pass fires all traces at maximum rate to populate L2 (DRAM)
  and L3 (storage). With the default write_through policy (threshold=1), each
  block is written to L3 on its first intra-conversation hit, so one pass is
  sufficient. The warmup result is discarded; only the rate sweep is saved.

  Between each rate, /flush_cache is called to reset the HBM (L1) KV cache.
  Note: /flush_cache does NOT clear L2 (DRAM) or L3 (storage) in hisim —
  MockTokenToKVPoolHost and MockHiCacheStorage are not reset by this call.
  To fully reset all cache layers, restart the server.
EOF
}

SERVER_URL="http://localhost:12345"
RATES="1,2,4,8,16"
OUTPUT_DIR="$(dirname "$0")/data/codex-swebenchpro-traces"
SKIP_WARMUP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --server-url)  SERVER_URL="$2"; shift 2 ;;
    --rates)       RATES="$2";      shift 2 ;;
    --output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
    --skip-warmup) SKIP_WARMUP=1;   shift ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

IFS=',' read -ra request_rates <<< "$RATES"

bench() {
  local rate="$1"
  local output_file="$2"
  python3 -m hisim.simulation.bench_serving \
    --backend sglang \
    --base-url "${SERVER_URL}" \
    --dataset-name codex-swebenchpro-traces \
    --request-rate "${rate}" \
    --bench-mode simulation \
    --warmup-requests 0 \
    --output-file "${output_file}"
}

if [[ "${SKIP_WARMUP}" -eq 0 ]]; then
  curl -sf "${SERVER_URL}/flush_cache" > /dev/null
  bench inf /dev/null
fi

for rate in "${request_rates[@]}"; do
  OUT_DIR="${OUTPUT_DIR}/${rate}"
  mkdir -p "${OUT_DIR}"
  curl -sf "${SERVER_URL}/flush_cache" > /dev/null
  bench "${rate}" "${OUT_DIR}/metrics.json"
done
