#!/bin/bash

# Array to track all vLLM PIDs for cleanup
VLLM_PIDS=()

# Trap to kill all background vLLM servers on exit
cleanup() {
    if [ ${#VLLM_PIDS[@]} -gt 0 ]; then
        echo "[vLLM] $(date '+%H:%M:%S') Cleaning up vLLM servers (PIDs: ${VLLM_PIDS[*]})..."
        for pid in "${VLLM_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid"
                echo "[vLLM] $(date '+%H:%M:%S') Killed PID $pid"
            fi
        done
    fi
}
trap cleanup EXIT INT TERM

require_openrouter_key() {
    local backend="${1:-}"
    if [ "$backend" = "openrouter" ] && [ -z "${OPENROUTER_API_KEY:-}" ]; then
        echo "[ERROR] OPENROUTER_API_KEY is not set. Export it or put it in ../.env."
        exit 1
    fi
}

_check_dependencies() {
    local missing=0
    for cmd in vllm curl; do
        if ! command -v "$cmd" &>/dev/null; then
            echo "[ERROR] Required command '$cmd' not found in PATH."
            missing=1
        fi
    done
    [ "$missing" -eq 1 ] && exit 1
}

start_vllm_servers_from_config() {
    # The honesty judge always shares the player's endpoint, so only the
    # model_pool entries need a server — there is no separate honesty model
    # to launch.
    local config_file="$1"

    if ! command -v jq &>/dev/null; then
        echo "[ERROR] 'jq' is required by start_vllm_servers_from_config but not found in PATH."
        exit 1
    fi

    local n_pool i is_local model_name api_base port gpu tp_size max_len gpu_mem
    local -a extra_args

    echo "[vLLM] $(date '+%H:%M:%S') Starting pool servers from config: $config_file"

    n_pool=$(jq '.model_pool | length' "$config_file")
    for i in $(seq 0 $((n_pool - 1))); do
        is_local=$(jq -r ".model_pool[$i].is_local // true" "$config_file")
        [ "$is_local" != "true" ] && continue

        model_name=$(jq -r ".model_pool[$i].model_name"             "$config_file")
        api_base=$(jq -r   ".model_pool[$i].api_base"               "$config_file")
        gpu=$(jq -r        ".model_pool[$i].gpu // \"\""            "$config_file")
        tp_size=$(jq -r    ".model_pool[$i].tensor_parallel_size // 1" "$config_file")
        max_len=$(jq -r    ".model_pool[$i].max_model_len // \"\""  "$config_file")
        gpu_mem=$(jq -r    ".model_pool[$i].gpu_memory_utilization // \"\"" "$config_file")

        # Extract port from api_base (e.g. "http://localhost:8000/v1" → "8000")
        port="${api_base##*:}"; port="${port%%/*}"

        extra_args=()
        [ -n "$gpu_mem" ] && extra_args+=(--gpu-memory-utilization "$gpu_mem")

        start_vllm_server "$model_name" "$port" "$gpu" "$tp_size" "$max_len" "${extra_args[@]}"
    done
}

start_vllm_server() {
    local model_name="$1"
    local port="$2"
    local gpus="$3"
    local tp_size="$4"
    local max_len="${5:-}"
    local -a extra_args=("${@:6}")

    local retry_interval="${VLLM_RETRY_INTERVAL:-10}"
    local max_retries="${VLLM_MAX_RETRIES:-120}"

    local log_dir="${VLLM_LOG_DIR:-../outputs/vllm_logs}"
    mkdir -p "$log_dir"
    local log_file="${log_dir}/vllm_port${port}_${SLURM_JOB_ID:-local}.log"
    echo "[vLLM] $(date '+%H:%M:%S') Log file: $log_file"

    _check_dependencies

    local -a cmd=(
        env CUDA_VISIBLE_DEVICES="$gpus"
        vllm serve "$model_name"
        --host 127.0.0.1
        --port "$port"
        --distributed-executor-backend mp
        --tensor-parallel-size "$tp_size"
    )
    [ -n "$max_len" ] && cmd+=(--max_model_len "$max_len")
    cmd+=("${extra_args[@]}")

    echo "[vLLM] $(date '+%H:%M:%S') Launching server:"
    echo "       Model : $model_name"
    echo "       Port  : $port"
    echo "       GPUs  : $gpus  (TP size: $tp_size)"
    echo "       MaxLen: ${max_len:-<vLLM default>}"
    [ ${#extra_args[@]} -gt 0 ] && echo "       Extras: ${extra_args[*]}"

    "${cmd[@]}" > "$log_file" 2>&1 &
    local vllm_pid=$!
    VLLM_PIDS+=("$vllm_pid")
    echo "[vLLM] $(date '+%H:%M:%S') Server started (PID: $vllm_pid), waiting for readiness..."

    local retry_count=0
    while true; do
        # Check if the process is still alive
        if ! kill -0 "$vllm_pid" 2>/dev/null; then
            echo "[ERROR] $(date '+%H:%M:%S') vLLM process (PID: $vllm_pid) died unexpectedly. Exiting."
            exit 1
        fi

        # Check HTTP reachability and that the expected model is loaded
        local response
        response=$(curl -sf "http://127.0.0.1:$port/v1/models" 2>/dev/null)
        if [ $? -eq 0 ] && echo "$response" | grep -qF "$model_name"; then
            echo "[vLLM] $(date '+%H:%M:%S') Server is ready! (model: $model_name, port: $port)"
            return 0
        fi

        if [ "$retry_count" -ge "$max_retries" ]; then
            echo "[ERROR] $(date '+%H:%M:%S') vLLM server on port $port failed to become ready after $((max_retries * retry_interval))s. Exiting."
            exit 1
        fi

        echo "[vLLM] $(date '+%H:%M:%S') ... Waiting (Attempt $((retry_count + 1))/$max_retries, next in ${retry_interval}s)"
        sleep "$retry_interval"
        retry_count=$((retry_count + 1))
    done
}