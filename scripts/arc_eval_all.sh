#!/usr/bin/env bash
# Hub baselines for both packages against ONE vLLM server on a 1-GPU node:
#   dxenv-diagnostician    eval split          (T=1.0, 1024 tokens; the source's Gate B protocol)
#   long-context-navigator id_test + ood_test  (T=0.8, top_p 0.95, 256 tokens; the source's rollout config)
#
# From the prime-hub checkout, inside tmux:
#   srun --account=<alloc> --partition=a100_normal_q --qos=tc_a100_normal_int \
#        --gres=gpu:a100:1 --cpus-per-task=8 --mem=64G --time=02:00:00 \
#        bash scripts/arc_eval_all.sh
#
# Required env:
#   VLLM_ENV   absolute path of a conda env with vllm installed
#   HUB_VENV   absolute path of a venv with both packages installed:
#              $HUB_VENV/bin/pip install -e environments/dxenv-diagnostician -e environments/long-context-navigator
#   HF_HOME    HF cache that holds (or can download) the model
# Optional: MODEL, SERVED, N, CONC, PORT, RUNS (default "dx nav_id nav_ood"; add dx_holdout if time allows)
set -euo pipefail

: "${VLLM_ENV:?set VLLM_ENV}" "${HUB_VENV:?set HUB_VENV}" "${HF_HOME:?set HF_HOME}"
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
SERVED=${SERVED:-qwen2.5-7b-instruct}  # no slash: vf-eval puts it in the run dir name
N=${N:-50}
CONC=${CONC:-16}
PORT=${PORT:-8000}
RUNS=${RUNS:-dx nav_id nav_ood}
export HF_HOME PYTHONNOUSERSITE=1

echo "vllm python: $("$VLLM_ENV/bin/python" -c 'import sys, vllm; print(sys.executable, vllm.__version__)')"
echo "hub  python: $("$HUB_VENV/bin/python" -c 'import sys, dxenv_diagnostician, long_context_navigator; print(sys.executable)')"
nvidia-smi -L

LOG=$(mktemp -t vllm.XXXX.log)
# dxenv prompts are ~3k tokens + 1k completion; navigator prompts are smaller.
"$VLLM_ENV/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name "$SERVED" \
    --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.90 \
    --port "$PORT" > "$LOG" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
    curl -sf "http://localhost:$PORT/v1/models" > /dev/null && break
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM exited:"; tail -40 "$LOG"; exit 1; }
    sleep 5
done
curl -sf "http://localhost:$PORT/v1/models" > /dev/null || { echo "vLLM not ready after 15 min"; tail -40 "$LOG"; exit 1; }
echo "vLLM ready ($LOG)"

run() {  # run <env> <split> <out_dir> <sampling flags...>
    local env=$1 split=$2 out=$3; shift 3
    echo; echo "=== $env / $split ==="
    VLLM_KEY=EMPTY "$HUB_VENV/bin/vf-eval" "$env" \
        --env.taskset.split "$split" -n "$N" -r 1 -c "$CONC" -m "$SERVED" \
        --client.base_url "http://localhost:$PORT/v1" --client.api_key_var VLLM_KEY \
        "$@" --no-push --output-dir "$out"
    local latest
    latest=$(ls -td "$out/$env"--* | head -1)
    "$HUB_VENV/bin/python" "environments/$env/scripts/episode_summary.py" "$latest" | tee "$latest/episode_summary.txt"
}

DX=(--sampling.temperature 1.0 --sampling.max-tokens 1024)
NAV=(--sampling.temperature 0.8 --sampling.top-p 0.95 --sampling.max-tokens 256)
for r in $RUNS; do
    case $r in
        dx)         run dxenv-diagnostician eval    environments/dxenv-diagnostician/outputs/evals "${DX[@]}" ;;
        dx_holdout) run dxenv-diagnostician holdout environments/dxenv-diagnostician/outputs/evals "${DX[@]}" ;;
        nav_id)     run long-context-navigator id_test  environments/long-context-navigator/outputs/evals "${NAV[@]}" ;;
        nav_ood)    run long-context-navigator ood_test environments/long-context-navigator/outputs/evals "${NAV[@]}" ;;
        *) echo "unknown run $r"; exit 1 ;;
    esac
done
echo; echo "done. outputs under environments/*/outputs/evals/"
