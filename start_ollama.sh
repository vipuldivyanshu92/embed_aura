#!/bin/bash
# Start Ollama server and ensure model is available

# Configuration
MODEL_NAME="${OLLAMA_MODEL_NAME:-qwen2.5:3b}"

echo "======================================"
echo "Starting Ollama"
echo "======================================"
echo ""
echo "Model: $MODEL_NAME"
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Error: Ollama not found"
    echo "Please run ./setup_ollama.sh first"
    exit 1
fi

# Check if Ollama is running
if pgrep -x "ollama" > /dev/null || curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
    echo "✓ Ollama is already running"
else
    echo "Starting Ollama server..."
    
    # Detect OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo ""
        echo "On macOS, please ensure Ollama.app is running"
        echo "You can:"
        echo "  1. Start Ollama from Applications"
        echo "  2. Or run: ollama serve"
        echo ""
        read -p "Press Enter when Ollama is running..."
    else
        # Linux - start in background
        nohup ollama serve > /tmp/ollama.log 2>&1 &
        sleep 2
        echo "✓ Ollama server started"
    fi
fi

# Check if model is available
echo ""
echo "Checking model: $MODEL_NAME"

if ollama list | grep -q "$MODEL_NAME"; then
    echo "✓ Model $MODEL_NAME is available"
else
    echo "Model $MODEL_NAME not found. Pulling..."
    ollama pull "$MODEL_NAME"
    echo "✓ Model pulled successfully"
fi

# Test the connection
echo ""
echo "Testing Ollama..."
if curl -s http://localhost:11434/api/version > /dev/null; then
    VERSION=$(curl -s http://localhost:11434/api/version | grep -o '"version":"[^"]*"' | cut -d'"' -f4)
    echo "✓ Ollama is running (version: $VERSION)"
    echo "✓ Available at: http://localhost:11434"
    echo ""
    echo "Available models:"
    ollama list
    echo ""
    echo "======================================"
    echo "Ready!"
    echo "======================================"
    echo ""
    echo "Ollama is running with model: $MODEL_NAME"
    echo ""
    echo "Start your Embed Service with:"
    echo "  make run"
    echo ""
    echo "Or test the model directly:"
    echo "  ollama run $MODEL_NAME"
else
    echo "❌ Could not connect to Ollama"
    echo ""
    echo "Please make sure Ollama is running:"
    echo "  - macOS: Check menu bar for Ollama icon"
    echo "  - Linux: Run 'ollama serve'"
    exit 1
fi

