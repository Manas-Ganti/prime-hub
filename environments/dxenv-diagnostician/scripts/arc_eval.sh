#!/usr/bin/env bash
# vf-eval of this package against a vLLM server on the same GPU node.
#
# Run from the prime-hub checkout inside a 1-GPU allocation, e.g.
#   srun --account=<alloc> --partition=a100_normal_q --qos=tc_a100_normal_int \
#        --gres=gpu:a100:1 --cpus-per-task=8 --mem=64G --time=02:00:00 \
#        bash environments/dxenv-diagnostician/scripts/arc_eval.sh
#
# Required env:
#   VLLM_ENV   absolute path of a conda env with vllm installed
#   HUB_VENV   absolute path of a venv with this package installed (pip install -e environments/dxenv-diagnostician)
#   HF_HOME    HF cache that holds (or can download) the model
# Optional: MODEL, SERVED, SPLIT, N, CONC, PORT, OUT
set -euo pipefail

: "${VLLM_ENV:?set VLLM_ENV}" "${HUB_VENV:?set HUB_VENV}" "${HF_HOME:?set HF_HOME}"
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
SERVED=${SERVED:-qwen2.5-7b-instruct}  # no slash: vf-eval puts it in the run dir name
SPLIT=${SPLIT:-eval}
N=${N:-50}
CONC=${CONC:-16}
PORT=${PORT:-8000}
OUT=${OUT:-environments/dxenv-diagnostician/outputs/evals}
export HF_HOME PYTHONNOUSERSITE=1

echo "vllm python: $("$VLLM_ENV/bin/python" -c 'import sys, vllm; print(sys.executable, vllm.__version__)')"
echo "hub  python: $("$HUB_VENV/bin/python" -c 'import sys, dxenv_diagnostician; print(sys.executable)')"
nvidia-smi -L

LOG=$(mktemp -t vllm.XXXX.log)
# ~3k-token prompts + 1k completion per turn; 8192 leaves headroom.
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

VLLM_KEY=EMPTY "$HUB_VENV/bin/vf-eval" dxenv-diagnostician \
    --env.taskset.split "$SPLIT" -n "$N" -r 1 -c "$CONC" -m "$SERVED" \
    --client.base_url "http://localhost:$PORT/v1" --client.api_key_var VLLM_KEY \
    --sampling.temperature 1.0 --sampling.max-tokens 1024 \
    --no-push --output-dir "$OUT"

latest=$(ls -td "$OUT"/dxenv-diagnostician--* | head -1)
"$HUB_VENV/bin/python" environments/dxenv-diagnostician/scripts/episode_summary.py "$latest"
