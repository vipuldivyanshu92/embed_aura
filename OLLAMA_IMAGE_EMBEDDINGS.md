# Ollama-Based Image Embeddings

This document describes how the system uses Ollama models for generating image embeddings, replacing the previous CLIP-based approach.

## Overview

The image embedding system now uses Ollama's vision and embedding models in a two-step process:

1. **Vision Model**: Analyzes the image and generates a detailed textual description
2. **Embedding Model**: Converts the description into a semantic embedding vector

This approach provides several advantages:
- **Better understanding**: Vision models like LLaVA can understand complex scenes, text in images, and contextual information
- **Consistent embeddings**: Uses the same embedding model for both text and image-derived embeddings
- **Local processing**: Everything runs locally via Ollama
- **Flexibility**: Easy to swap models based on your needs

## Architecture

```
Image (URL/Base64)
    ↓
Vision Model (llava, llama3.2-vision, minicpm-v)
    ↓
Textual Description
    ↓
Embedding Model (nomic-embed-text)
    ↓
Embedding Vector
```

## Configuration

Add these settings to your `.env` file or environment variables:

```bash
# Vision model for image understanding (default: llava)
OLLAMA_VISION_MODEL=llava

# Embedding model for text embeddings (default: nomic-embed-text)
OLLAMA_EMBED_MODEL=nomic-embed-text
```

### Available Vision Models

Common Ollama vision models you can use:

| Model | Size | Strengths | Command to Pull |
|-------|------|-----------|----------------|
| `llava` | ~4.5GB | General purpose, good balance | `ollama pull llava` |
| `llama3.2-vision` | ~7.9GB | Latest, best quality | `ollama pull llama3.2-vision` |
| `minicpm-v` | ~4.2GB | Smaller, faster | `ollama pull minicpm-v` |
| `bakllava` | ~4.7GB | BakLLaVA variant | `ollama pull bakllava` |

### Available Embedding Models

| Model | Dimensions | Strengths | Command to Pull |
|-------|-----------|-----------|----------------|
| `nomic-embed-text` | 768 | High quality, recommended | `ollama pull nomic-embed-text` |
| `mxbai-embed-large` | 1024 | Larger, more detailed | `ollama pull mxbai-embed-large` |
| `all-minilm` | 384 | Lightweight, fast | `ollama pull all-minilm` |

## Setup

1. **Install Ollama** (if not already installed):
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Pull the required models**:
   ```bash
   # Pull vision model
   ollama pull llava
   
   # Pull embedding model
   ollama pull nomic-embed-text
   ```

3. **Start Ollama server** (usually runs automatically):
   ```bash
   ollama serve
   ```

4. **Configure your application**:
   ```bash
   # In your .env file
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_VISION_MODEL=llava
   OLLAMA_EMBED_MODEL=nomic-embed-text
   ```

## Usage

The system automatically uses Ollama for image embeddings when processing images:

```python
from app.utils.embeddings import generate_image_embedding

# From URL
embedding = generate_image_embedding(
    image_url="https://example.com/image.jpg"
)

# From base64
embedding = generate_image_embedding(
    image_base64="data:image/png;base64,iVBORw0KG..."
)

# With fallback description
embedding = generate_image_embedding(
    image_url="https://example.com/image.jpg",
    description="A cat sitting on a couch"
)
```

### API Usage

When using the `/hypothesize` or `/execute` endpoints with images:

```bash
curl -X POST http://localhost:8000/hypothesize \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "media_type": "image",
    "media_url": "https://example.com/photo.jpg",
    "input_text": "What is in this image?"
  }'
```

## How It Works

### Step 1: Image Description

The vision model (e.g., LLaVA) analyzes the image with this prompt:

```
Describe this image in detail, including objects, colors, scene, context, and any text visible.
```

Example output:
```
A modern office workspace with a laptop displaying code on the screen. 
The desk is wooden with a white coffee mug on the right side. Natural 
light comes through a window in the background. The laptop appears to 
be a MacBook Pro with a dark theme IDE open, showing Python code.
```

### Step 2: Embedding Generation

The embedding model (e.g., nomic-embed-text) converts this description into a semantic vector:

```
Description → [0.123, -0.456, 0.789, ..., 0.234]  (768 dimensions)
```

This embedding can then be used for:
- Semantic search across images
- Finding similar images
- Comparing images to text queries
- Building image-text hybrid search

## Fallback Behavior

The system includes graceful fallbacks:

1. **Ollama available** → Use vision + embedding models (preferred)
2. **Ollama unavailable** → Fall back to CLIP (if installed)
3. **CLIP unavailable** → Use text description (if provided)
4. **No description** → Hash-based fallback embedding

This ensures the system continues working even if Ollama is not available.

## Performance Considerations

### Speed
- **Vision model**: ~2-5 seconds per image (depends on model size)
- **Embedding**: ~100-500ms per description
- **Total**: ~2-6 seconds per image

### Resource Usage
- Vision models use ~2-8GB RAM
- Embedding models use ~500MB-2GB RAM
- GPU/Metal acceleration recommended for faster processing

### Optimization Tips

1. **Use smaller models** for faster processing:
   ```bash
   OLLAMA_VISION_MODEL=minicpm-v
   OLLAMA_EMBED_MODEL=all-minilm
   ```

2. **Cache embeddings** to avoid re-processing:
   - Store embeddings in your database
   - Only regenerate when images change

3. **Batch processing** for multiple images:
   - Process images in parallel
   - Reuse the Ollama client instance

## Comparison: Ollama vs CLIP

| Aspect | Ollama (Vision + Embed) | CLIP |
|--------|------------------------|------|
| **Setup** | Pull 2 models (~5GB) | Download weights (~350MB) |
| **Speed** | 2-6s per image | ~100ms per image |
| **Quality** | Rich text descriptions | Direct image embedding |
| **Understanding** | Understands context, text in images | Visual features only |
| **Flexibility** | Swap models easily | Fixed model |
| **Text-Image Alignment** | High (same embed model) | Moderate |
| **Local** | ✅ Yes | ✅ Yes |

## Troubleshooting

### "this event loop is already running"
This error has been fixed in the implementation. The image embedding now runs in a separate thread with its own event loop, avoiding conflicts with FastAPI's async context.

### "Ollama client not available"
- Ensure Ollama is running: `ollama serve`
- Check the base URL: `OLLAMA_BASE_URL=http://localhost:11434`
- Verify Ollama is accessible: `curl http://localhost:11434/api/tags`

### "Model not found"
- Pull the model: `ollama pull llava`
- Check available models: `ollama list`

### Slow processing
- Use smaller vision models (minicpm-v instead of llama3.2-vision)
- Use lighter embedding models (all-minilm instead of mxbai-embed-large)
- Ensure you have enough RAM
- Consider GPU/Metal acceleration

### Out of memory
- Reduce model sizes
- Close other applications
- Process images one at a time instead of batching

## Example Output

Input image: A cat sitting on a laptop keyboard

1. **Vision model output**:
   ```
   A gray tabby cat is sitting on top of a laptop keyboard. The cat 
   appears to be looking at the screen. The laptop is silver colored, 
   possibly a MacBook. The background shows a white desk and a plant 
   in the corner. The cat's paws are positioned on the keyboard keys.
   ```

2. **Embedding** (first 10 values):
   ```
   [0.023, -0.145, 0.234, 0.089, -0.067, 0.178, -0.234, 0.123, 0.056, -0.198, ...]
   ```

3. **Semantic search** finds similar:
   - "pet on computer"
   - "cat working from home"
   - "animal using laptop"
   - "keyboard cat meme"

## Benefits

1. **Rich Understanding**: Vision models understand complex scenes
2. **Text in Images**: Can read and understand text within images
3. **Contextual**: Understands relationships between objects
4. **Consistent Embeddings**: Same embedding space for text and images
5. **Easy Customization**: Swap models via configuration
6. **Local Privacy**: All processing happens locally

## Future Enhancements

Potential improvements for the future:

- [ ] Auto-detect best vision model based on image type
- [ ] Support for multiple vision models based on use case
- [ ] Caching of image descriptions to avoid re-processing
- [ ] Direct vision embeddings (some vision models support this)
- [ ] Batch processing API for multiple images
- [ ] Prompt customization for different image types

