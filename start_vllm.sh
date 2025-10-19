#!/bin/bash
# Start vLLM server with the configured model

# Configuration
MODEL_NAME="${VLLM_MODEL_NAME:-unsloth/Qwen2.5-3B-Instruct}"
PORT="${VLLM_PORT:-8000}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY:-0.9}"

echo "======================================"
echo "Starting vLLM Server"
echo "======================================"
echo ""
echo "Model: $MODEL_NAME"
echo "Port: $PORT"
echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo ""

# Check if vLLM environment exists
VLLM_VENV="./venv-vllm"
if [ ! -d "$VLLM_VENV" ]; then
    echo "Error: vLLM environment not found."
    echo "Please run ./setup_vllm.sh first"
    exit 1
fi

# Activate vLLM environment
source "$VLLM_VENV/bin/activate"

# Check if vLLM is installed
if ! command -v vllm &> /dev/null; then
    echo "Error: vLLM not installed in virtual environment"
    echo "Please run ./setup_vllm.sh first"
    exit 1
fi

echo "Starting vLLM server..."
echo "The model will be downloaded on first run (~3GB)"
echo ""
echo "Server will be available at: http://localhost:$PORT/v1"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start vLLM server
vllm serve "$MODEL_NAME" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len 4096

