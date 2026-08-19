#!/usr/bin/env bash
# vram-budget-probe.sh — sample GPU VRAM + resident Ollama models.
#
# Every INTERVAL seconds (default 60), for SAMPLES iterations (default 10,
# i.e. a 10-minute window), appends one JSON line to logs/vram-budget.jsonl:
#
#   {"ts": "<iso8601>", "gpu_used_mib": <int|null>, "gpu_total_mib": <int|null>, "ollama_models": ["<name>", ...]}
#
# If nvidia-smi or ollama is absent, the corresponding field is logged as
# null / [] — the probe never crashes because a tool is missing.
#
# Each JSON line is assembled fully in memory and appended with a single
# write, so killing the probe mid-run can never leave a partial JSON line
# on disk (worst case: a missing trailing sample).
#
# Configuration (env vars, overridable by CLI flags):
#   VRAM_PROBE_INTERVAL   seconds between samples            (default 60)
#   VRAM_PROBE_SAMPLES    number of samples                  (default 10)
#   VRAM_PROBE_LOG_DIR    directory for the JSONL log        (default <repo>/logs)
#
# Usage: vram-budget-probe.sh [--interval SECONDS] [--samples COUNT]
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INTERVAL="${VRAM_PROBE_INTERVAL:-60}"
SAMPLES="${VRAM_PROBE_SAMPLES:-10}"
LOG_DIR="${VRAM_PROBE_LOG_DIR:-${REPO_ROOT}/logs}"
LOG_FILE="${LOG_DIR}/vram-budget.jsonl"

usage() {
    cat <<'EOF'
Usage: vram-budget-probe.sh [--interval SECONDS] [--samples COUNT] [-h]

Samples GPU VRAM usage and resident Ollama models, appending one JSON line
per sample to logs/vram-budget.jsonl.

  --interval SECONDS   seconds between samples (default: ${VRAM_PROBE_INTERVAL:-60})
  --samples COUNT      number of samples       (default: ${VRAM_PROBE_SAMPLES:-10})
  -h, --help           show this help

Defaults come from VRAM_PROBE_INTERVAL / VRAM_PROBE_SAMPLES / VRAM_PROBE_LOG_DIR.
EOF
}

# Parse CLI flags (override env defaults).
while [ "$#" -gt 0 ]; do
    case "$1" in
        --interval)
            [ "$#" -ge 2 ] || { echo "error: --interval requires a value" >&2; exit 2; }
            INTERVAL="$2"
            shift 2
            ;;
        --samples)
            [ "$#" -ge 2 ] || { echo "error: --samples requires a value" >&2; exit 2; }
            SAMPLES="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

# Validate numeric config.
case "$INTERVAL" in
    ''|*[!0-9]*) echo "error: --interval must be a positive integer (got: $INTERVAL)" >&2; exit 2 ;;
esac
case "$SAMPLES" in
    ''|*[!0-9]*) echo "error: --samples must be a positive integer (got: $SAMPLES)" >&2; exit 2 ;;
esac
[ "$INTERVAL" -ge 1 ] || { echo "error: --interval must be >= 1 (got: $INTERVAL)" >&2; exit 2; }
[ "$SAMPLES"   -ge 1 ] || { echo "error: --samples must be >= 1 (got: $SAMPLES)" >&2; exit 2; }

mkdir -p "$LOG_DIR"

# Detect tool availability once (nvidia-smi often absent on non-GPU hosts).
command -v nvidia-smi >/dev/null 2>&1 && HAS_NVIDIA=1 || HAS_NVIDIA=0
command -v ollama     >/dev/null 2>&1 && HAS_OLLAMA=1 || HAS_OLLAMA=0

# json_escape STR — minimal JSON string escaping (backslash, double quote).
json_escape() {
    local s="$1"
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    printf '%s' "$s"
}

sample() {
    local ts gpu_used gpu_total gpu_line gpu_clean
    local names model_json first name line

    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    # --- GPU used/total memory (MiB) ---
    gpu_used="null"
    gpu_total="null"
    if [ "$HAS_NVIDIA" -eq 1 ]; then
        # csv,noheader,nounits emits "1234, 8192"; strip spaces before parsing.
        gpu_line="$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1)"
        if [ -n "$gpu_line" ]; then
            gpu_clean="${gpu_line// /}"
            gpu_used="${gpu_clean%%,*}"
            gpu_total="${gpu_clean##*,}"
            case "$gpu_used"  in ''|*[!0-9]*) gpu_used="null"  ;; esac
            case "$gpu_total" in ''|*[!0-9]*) gpu_total="null" ;; esac
        fi
    fi

    # --- Resident Ollama models (first column of `ollama ps`, skip header) ---
    # NOTE: this ollama client lacks `ps --format json`, so parse the table.
    model_json="[]"
    if [ "$HAS_OLLAMA" -eq 1 ]; then
        names="$(ollama ps 2>/dev/null | awk 'NR > 1 && NF > 0 { print $1 }')"
        if [ -n "$names" ]; then
            model_json="["
            first=1
            while IFS= read -r name; do
                [ -n "$name" ] || continue
                if [ "$first" -eq 1 ]; then
                    first=0
                else
                    model_json="${model_json}, "
                fi
                model_json="${model_json}\"$(json_escape "$name")\""
            done <<< "$names"
            model_json="${model_json}]"
        fi
    fi

    line="{\"ts\": \"${ts}\", \"gpu_used_mib\": ${gpu_used}, \"gpu_total_mib\": ${gpu_total}, \"ollama_models\": ${model_json}}"
    printf '%s\n' "$line" >> "$LOG_FILE"
}

i=0
while [ "$i" -lt "$SAMPLES" ]; do
    sample
    i=$((i + 1))
    if [ "$i" -lt "$SAMPLES" ]; then
        sleep "$INTERVAL"
    fi
done
