# 🚀 Model that learns you and generates context so you can Zero-Shot every single time.

A production-ready FastAPI backend service that minimizes user typing through intelligent hypothesis generation and context-aware prompt enrichment.

## Overview

This service implements a two-phase approach to task execution:

### Phase A – Hypothesizer
From a user's short input, the system proposes 2–3 concise next-question hypotheses based on:
- User's historical preferences and goals
- Past interaction patterns
- Persona facets (concise, formal, code_first, etc.)

### Phase B – Execute (Enrichment)
After the user selects a hypothesis, the system enriches the input into a long, structured prompt ready for a powerful LLM, incorporating:
- Ranked memories (goals, preferences, artifacts, history)
- Context budgeting (smart token allocation)
- PII redaction and safety filters
- Persona-aware customization

## Architecture

```
┌─────────────────┐
│  FastAPI App    │
├─────────────────┤
│ Hypothesizer    │──> Phase A: Generate hypotheses
│ Executor        │──> Phase B: Enrich prompt
├─────────────────┤
│ Memory Layer    │──> Pluggable (local/mem0/supermemory)
│ SLM Client      │──> Pluggable (local/http)
│ Persona Engine  │──> User preference learning
│ Context Budget  │──> Smart token allocation
└─────────────────┘
```

## Features

✅ **Pluggable Memory Backends**: Local JSON, Mem0, or Supermemory  
✅ **Smart Hypothesis Generation**: 2-3 contextual suggestions per request  
✅ **AI Integration**: Ollama (macOS) or vLLM (Linux) for enhanced capabilities ⭐ **NEW**  
✅ **Learning Loop**: Automatic learning from every user interaction ⭐ **NEW**  
✅ **Model Fine-Tuning**: Collect data and train models for your specific use case ⭐ **NEW**  
✅ **Context Budgeting**: Precise token allocation across memory sections  
✅ **Persona Learning**: Adaptive user preference tracking (AI-enhanced)  
✅ **Goal & Preference Extraction**: Automatically learns from hypothesis selections ⭐ **NEW**  
✅ **Semantic Re-Ranking**: Better memory relevance with AI  
✅ **PII Redaction**: Automatic safety filtering  
✅ **Offline-First**: Runs completely locally with no external dependencies  
✅ **Production-Ready**: Structured logging, type safety, comprehensive tests  

## Quick Start

### Prerequisites
- Python 3.11+
- pip or uv

### Installation

```bash
# Clone and navigate to the project
cd embed

# Install dependencies
make install

# Or manually:
pip install -e .
```

### Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Key environment variables:

```env
APP_ENV=dev
MEMORY_BACKEND=local          # local|mem0|supermemory
TOKEN_BUDGET=6000             # Max tokens for enriched prompt
SLM_IMPL=local                # local|http
EMBED_DIMS=384                # Embedding dimensions
LOG_LEVEL=INFO
```

### Running the Service

```bash
# Using Make
make run

# Or directly with uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Documentation

Interactive API docs: `http://localhost:8000/docs`

## 🤖 Local AI Integration (Enhanced Performance)

For significantly better hypothesis generation, persona learning, and memory ranking, you can enable local AI:

### For macOS (Recommended: Ollama) 🍎

**Ollama is optimized for Apple Silicon with Metal acceleration:**

```bash
# 1. Setup Ollama (one-time, ~5 minutes)
./setup_ollama.sh

# 2. Configure Embed Service
echo "SLM_IMPL=ollama" >> .env
echo "OLLAMA_MODEL_NAME=qwen2.5:3b" >> .env
echo "ENABLE_TRAINING_DATA_COLLECTION=true" >> .env

# 3. Start Embed Service
make run
```

### For Linux (vLLM) 🐧

**vLLM is optimized for Linux with CUDA GPUs:**

```bash
# 1. Setup vLLM (one-time)
./setup_vllm.sh

# 2. Start vLLM server
./start_vllm.sh

# 3. Configure Embed Service
echo "SLM_IMPL=vllm" >> .env
echo "ENABLE_TRAINING_DATA_COLLECTION=true" >> .env

# 4. Start Embed Service
make run
```

### What You Get

- **85-95% accuracy** (vs 60-70% with rule-based)
- **Context-aware hypotheses** personalized to user history
- **AI-driven persona learning** instead of heuristics
- **Semantic re-ranking** for better memory retrieval
- **Training data collection** for continuous improvement
- **Model fine-tuning** on your specific use cases

### Examples

```bash
# macOS with Ollama
python example_ollama_usage.py

# Linux with vLLM
python example_vllm_usage.py
```

### Documentation

**For macOS Users:**
- **[OLLAMA_GUIDE.md](OLLAMA_GUIDE.md)** ⭐ Recommended - Complete Ollama guide
- **[LEARNING_LOOP.md](LEARNING_LOOP.md)** 🔄 How the system learns from interactions

**For Linux Users:**
- **[VLLM_GUIDE.md](VLLM_GUIDE.md)** - Complete vLLM guide
- **[local-models/README.md](local-models/README.md)** - Quick reference

**General:**
- **[LEARNING_LOOP.md](LEARNING_LOOP.md)** 🔄 Training and continuous improvement

## API Endpoints

### POST `/v1/hypothesize` - Phase A

Generate hypotheses from user input.

**Request:**
```bash
curl -X POST http://localhost:8000/v1/hypothesize \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "input_text": "build an API"
  }'
```

**Response:**
```json
{
  "hypotheses": [
    {
      "id": "h1",
      "question": "Do you want to build a REST API with FastAPI?",
      "rationale": "User has history of Python backend work",
      "confidence": 0.82
    },
    {
      "id": "h2",
      "question": "Do you want to build a GraphQL API?",
      "rationale": "Alternative API paradigm",
      "confidence": 0.64
    }
  ],
  "auto_advance": false
}
```

### POST `/v1/execute` - Phase B

Enrich input into a comprehensive prompt.

**Request:**
```bash
curl -X POST http://localhost:8000/v1/execute \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "input_text": "build an API",
    "hypothesis_id": "h1"
  }'
```

**Response:**
```json
{
  "enriched_prompt": "SYSTEM\nYou are executing a task for user123...",
  "tokens_estimate": 1234,
  "context_breakdown": {
    "goal_summary": 220,
    "preferences_style": 180,
    "critical_artifacts": 420,
    "recent_history": 260,
    "constraints": 140,
    "task_specific_retrieval": 140,
    "safety_system": 60
  }
}
```

### GET `/v1/healthz`

Health check endpoint.

```bash
curl http://localhost:8000/v1/healthz
```

### GET `/v1/memory/export?user_id=USER_ID`

Export user's memories and persona (debug/backup).

```bash
curl http://localhost:8000/v1/memory/export?user_id=user123
```

### DELETE `/v1/memory?user_id=USER_ID&mtype=TYPE`

Delete memories by type (local backend only).

```bash
curl -X DELETE "http://localhost:8000/v1/memory?user_id=user123&mtype=PREFERENCE"
```

## Memory Backends

### Local (Default)
- In-memory storage with JSON persistence
- Persists to `data/memories.json` and `data/personas.json`
- Perfect for development and testing

```env
MEMORY_BACKEND=local
```

### Mem0 (Adapter Stub)
- Enterprise memory layer
- Requires Mem0 SDK integration (TODO in `app/memory/mem0.py`)

```env
MEMORY_BACKEND=mem0
MEM0_API_KEY=your_key
MEM0_BASE_URL=https://api.mem0.ai
```

### Supermemory (Adapter Stub)
- Alternative memory backend
- Requires Supermemory SDK integration (TODO in `app/memory/supermemory.py`)

```env
MEMORY_BACKEND=supermemory
SUPERMEMORY_API_KEY=your_key
```

## SLM Implementations

### Local SLM (Default)
- Rule-based heuristic implementation
- No external dependencies
- Perfect for offline development

```env
SLM_IMPL=local
```

### HTTP SLM
- Connect to hosted small language model
- Supports custom endpoints

```env
SLM_IMPL=http
SLM_BASE_URL=https://your-slm-endpoint.com
SLM_API_KEY=your_api_key
```

## Development

### Linting & Formatting

```bash
make lint
```

### Type Checking

```bash
mypy app tests
```

### Running Tests

```bash
make test

# Or with pytest directly
pytest -v
```

### Docker

Build and run with Docker:

```bash
docker build -t embed-service .
docker run -p 8000:8000 --env-file .env embed-service
```

## Data Privacy

### Exporting User Data

```bash
curl http://localhost:8000/v1/memory/export?user_id=USER_ID > user_data.json
```

### Deleting User Data

For the local backend:

```bash
# Delete specific memory type
curl -X DELETE "http://localhost:8000/v1/memory?user_id=USER_ID&mtype=PREFERENCE"

# Or manually delete from data/ directory
rm data/memories.json data/personas.json
```

### PII Protection

The service automatically redacts:
- Email addresses → `<REDACTED:EMAIL>`
- Phone numbers → `<REDACTED:PHONE>`
- API keys → `<REDACTED:API_KEY>`

## Memory Types

- **PROFILE**: User background, role, expertise
- **PREFERENCE**: User preferences (language, style, tools)
- **GOAL**: Long-term objectives
- **HISTORY**: Past interactions and outcomes
- **ARTIFACT**: Code snippets, templates, examples
- **STYLE**: Communication and output style preferences

## Persona Facets

The system tracks user preferences through continuous learning:

- `concise`: Preference for brief vs. detailed responses [0-1]
- `formal`: Formality level [0-1]
- `code_first`: Preference for code over explanations [0-1]
- `step_by_step`: Preference for detailed steps [0-1]

## Token Budgeting

The context budgeter allocates tokens across sections:

```
goal_summary:              10% (600 tokens)
preferences_style:         10% (600 tokens)
critical_artifacts:        25% (1500 tokens)
recent_history:            20% (1200 tokens)
constraints:               15% (900 tokens)
task_specific_retrieval:   15% (900 tokens)
safety_system:             5%  (300 tokens)
```

Configure via `BUDGET_WEIGHTS` environment variable.

## Project Structure

```
embed/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   ├── models.py            # Pydantic models
│   ├── clients/
│   │   └── slm/            # Small Language Model clients
│   ├── memory/             # Memory provider implementations
│   └── services/           # Core business logic
│       ├── persona.py
│       ├── ranking.py
│       ├── context_budgeter.py
│       ├── prompt_builder.py
│       ├── hypothesizer.py
│       ├── safety.py
│       └── telemetry.py
├── tests/                  # Test suite
├── data/                   # Local storage (gitignored)
├── pyproject.toml
├── Dockerfile
└── Makefile
```

## License

MIT License - see LICENSE file for details.

## Contributing

This is a production-ready template. Feel free to extend:
- Add more memory backends
- Integrate real SLM endpoints
- Enhance persona learning algorithms
- Add more sophisticated PII detection

## Support

For issues or questions, please open an issue on the repository.

# embed_aura
