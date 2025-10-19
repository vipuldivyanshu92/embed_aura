#!/bin/bash
# Setup script for vLLM local model serving

set -e

echo "======================================"
echo "vLLM Local Model Setup"
echo "======================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found."
    exit 1
fi

# Create virtual environment for vLLM if it doesn't exist
VLLM_VENV="./venv-vllm"
if [ ! -d "$VLLM_VENV" ]; then
    echo "Creating virtual environment for vLLM..."
    python3 -m venv "$VLLM_VENV"
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate vLLM environment
echo "Activating vLLM environment..."
source "$VLLM_VENV/bin/activate"

# Install vLLM
echo ""
echo "Installing vLLM (this may take a few minutes)..."
pip install --upgrade pip
pip install vllm

echo ""
echo "✓ vLLM installed successfully"

# Download model information
echo ""
echo "======================================"
echo "Model Setup"
echo "======================================"
echo ""
echo "The default model is: unsloth/Qwen2.5-3B-Instruct"
echo ""
echo "To start the vLLM server, run:"
echo "  source $VLLM_VENV/bin/activate"
echo "  vllm serve unsloth/Qwen2.5-3B-Instruct"
echo ""
echo "Or use the provided start script:"
echo "  ./start_vllm.sh"
echo ""
echo "The model will be downloaded on first run (~3GB)."
echo "By default, it will be cached in ~/.cache/huggingface/"
echo ""
echo "======================================"
echo "Configuration"
echo "======================================"
echo ""
echo "To use vLLM with the embed service:"
echo "1. Start vLLM server (see above)"
echo "2. Set environment variable: SLM_IMPL=vllm"
echo "3. Optionally configure:"
echo "   - VLLM_BASE_URL (default: http://localhost:8000/v1)"
echo "   - VLLM_MODEL_NAME (default: unsloth/Qwen2.5-3B-Instruct)"
echo "   - ENABLE_TRAINING_DATA_COLLECTION=true (for model tuning)"
echo ""
echo "Example .env configuration:"
echo "  SLM_IMPL=vllm"
echo "  VLLM_BASE_URL=http://localhost:8000/v1"
echo "  ENABLE_TRAINING_DATA_COLLECTION=true"
echo ""
echo "✓ Setup complete!"

