#!/bin/bash
# Setup script for Ollama local model serving (macOS compatible)

set -e

echo "======================================"
echo "Ollama Local Model Setup"
echo "======================================"
echo ""

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
else
    OS="Unknown"
fi

echo "Detected OS: $OS"
echo ""

# Check if Ollama is already installed
if command -v ollama &> /dev/null; then
    echo "✓ Ollama is already installed"
    ollama --version
else
    echo "Installing Ollama..."
    echo ""
    
    if [[ "$OS" == "macOS" ]]; then
        echo "For macOS, please install Ollama using one of these methods:"
        echo ""
        echo "Option 1: Download from website (Recommended)"
        echo "  1. Visit https://ollama.com/download"
        echo "  2. Download Ollama for macOS"
        echo "  3. Install the .app file"
        echo ""
        echo "Option 2: Using Homebrew"
        echo "  brew install ollama"
        echo ""
        read -p "Press Enter after installing Ollama..."
        
    elif [[ "$OS" == "Linux" ]]; then
        echo "Installing Ollama for Linux..."
        curl -fsSL https://ollama.com/install.sh | sh
        echo "✓ Ollama installed successfully"
    fi
fi

echo ""
echo "======================================"
echo "Model Setup"
echo "======================================"
echo ""

# Default model
DEFAULT_MODEL="qwen2.5:3b"
MODEL="${OLLAMA_MODEL_NAME:-$DEFAULT_MODEL}"

echo "Recommended models for the Embed Service:"
echo ""
echo "Small models (fast, ~2GB):"
echo "  - qwen2.5:3b (default, excellent quality)"
echo "  - llama3.2:3b (good balance)"
echo "  - phi3:mini (~2GB, fast)"
echo ""
echo "Medium models (better quality, ~4-5GB):"
echo "  - qwen2.5:7b (higher quality)"
echo "  - llama3.1:8b (very capable)"
echo "  - mistral:7b (good reasoning)"
echo ""
echo "Large models (best quality, ~40GB+):"
echo "  - qwen2.5:32b (production quality)"
echo "  - llama3.1:70b (state-of-the-art)"
echo ""

read -p "Do you want to pull the default model ($MODEL) now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Pulling model: $MODEL"
    echo "This may take a few minutes depending on your internet speed..."
    ollama pull "$MODEL"
    echo "✓ Model pulled successfully"
else
    echo ""
    echo "Skipping model download."
    echo "To download later, run: ollama pull $MODEL"
fi

echo ""
echo "======================================"
echo "Starting Ollama"
echo "======================================"
echo ""

# Check if Ollama is running
if pgrep -x "ollama" > /dev/null; then
    echo "✓ Ollama is already running"
else
    echo "Starting Ollama server..."
    
    if [[ "$OS" == "macOS" ]]; then
        # On macOS, Ollama app should be running
        echo "Please ensure the Ollama app is running in your menu bar"
        echo "You can start it from Applications/Ollama.app"
    else
        # On Linux, start as background service
        ollama serve > /dev/null 2>&1 &
        sleep 2
        echo "✓ Ollama server started"
    fi
fi

echo ""
echo "======================================"
echo "Configuration"
echo "======================================"
echo ""

echo "To use Ollama with the Embed Service:"
echo ""
echo "1. Ensure Ollama is running (check menu bar on Mac)"
echo "2. Set environment variable: SLM_IMPL=ollama"
echo "3. Optionally configure:"
echo "   - OLLAMA_BASE_URL (default: http://localhost:11434)"
echo "   - OLLAMA_MODEL_NAME (default: qwen2.5:3b)"
echo "   - ENABLE_TRAINING_DATA_COLLECTION=true (for model tuning)"
echo ""
echo "Example .env configuration:"
echo "  SLM_IMPL=ollama"
echo "  OLLAMA_MODEL_NAME=$MODEL"
echo "  ENABLE_TRAINING_DATA_COLLECTION=true"
echo ""

# Test Ollama connection
echo "Testing Ollama connection..."
if curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
    echo "✓ Ollama is responding at http://localhost:11434"
else
    echo "⚠ Could not connect to Ollama at http://localhost:11434"
    echo "Please make sure Ollama is running:"
    if [[ "$OS" == "macOS" ]]; then
        echo "  - Check if Ollama.app is running (menu bar icon)"
        echo "  - Or run: ollama serve"
    else
        echo "  - Run: ollama serve"
    fi
fi

echo ""
echo "======================================"
echo "Quick Test"
echo "======================================"
echo ""

if ollama list > /dev/null 2>&1; then
    echo "Available models:"
    ollama list
    echo ""
    
    if ollama list | grep -q "$MODEL"; then
        echo "✓ Model $MODEL is ready to use"
        echo ""
        echo "Test it with:"
        echo "  ollama run $MODEL \"Hello, how are you?\""
    fi
fi

echo ""
echo "======================================"
echo "Next Steps"
echo "======================================"
echo ""
echo "1. Update your .env file:"
echo "   echo \"SLM_IMPL=ollama\" >> .env"
echo "   echo \"OLLAMA_MODEL_NAME=$MODEL\" >> .env"
echo "   echo \"ENABLE_TRAINING_DATA_COLLECTION=true\" >> .env"
echo ""
echo "2. Start the Embed Service:"
echo "   make run"
echo ""
echo "3. Try the demo:"
echo "   python example_ollama_usage.py"
echo ""
echo "✓ Setup complete!"
echo ""
echo "📚 For more info:"
echo "   - Ollama models: https://ollama.com/library"
echo "   - Documentation: See OLLAMA_GUIDE.md"

