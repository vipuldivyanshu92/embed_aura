# Quick Start Guide

## 🚀 Get Started in 60 Seconds

### 1. Install Dependencies

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e ".[dev]"
```

### 2. Configure Environment

```bash
# Copy environment template
cp env.example.txt .env

# Edit .env if needed (defaults work out of the box)
```

### 3. Run the Service

```bash
# Option 1: Using Make
make run

# Option 2: Direct with uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Option 3: Using Docker
docker-compose up
```

The service will be available at `http://localhost:8000`

### 4. Test the API

**Phase A - Generate Hypotheses:**

```bash
curl -X POST http://localhost:8000/v1/hypothesize \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "alice",
    "input_text": "build an API"
  }'
```

**Response:**
```json
{
  "hypotheses": [
    {
      "id": "h1",
      "question": "Do you want to build a REST API with a specific framework?",
      "rationale": "Detected intent: build an API",
      "confidence": 0.85
    },
    {
      "id": "h2",
      "question": "Do you want to design API endpoints and data models?",
      "rationale": "Detected intent: build an API",
      "confidence": 0.75
    }
  ],
  "auto_advance": false
}
```

**Phase B - Execute Enrichment:**

```bash
curl -X POST http://localhost:8000/v1/execute \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "alice",
    "input_text": "build an API",
    "hypothesis_id": "h1"
  }'
```

**Response:**
```json
{
  "enriched_prompt": "SYSTEM\nYou are executing a task for alice...",
  "tokens_estimate": 1234,
  "context_breakdown": {
    "goal_summary": 120,
    "preferences_style": 180,
    "critical_artifacts": 420,
    "recent_history": 260,
    "constraints": 140,
    "task_specific_retrieval": 114,
    "safety_system": 60
  }
}
```

### 5. View API Documentation

Open your browser to:
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### 6. Run Tests

```bash
# Run all tests
make test

# Or with pytest directly
pytest -v

# With coverage
pytest --cov=app --cov-report=term-missing
```

### 7. Check Code Quality

```bash
# Lint code
make lint

# Format code
make format

# Type check
mypy app tests
```

## 📝 Common Tasks

### Export User Data

```bash
curl http://localhost:8000/v1/memory/export?user_id=alice
```

### Delete User Memories

```bash
# Delete all memories
curl -X DELETE "http://localhost:8000/v1/memory?user_id=alice"

# Delete specific type
curl -X DELETE "http://localhost:8000/v1/memory?user_id=alice&mtype=PREFERENCE"
```

### Health Check

```bash
curl http://localhost:8000/v1/healthz
```

## 🔧 Configuration

Key environment variables (all optional, defaults provided):

```bash
# Memory backend (local works offline)
MEMORY_BACKEND=local  # or mem0, supermemory

# SLM implementation (local works offline)
SLM_IMPL=local  # or http

# Token budget
TOKEN_BUDGET=6000

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

## 🐳 Docker Usage

```bash
# Build image
docker build -t embed-service .

# Run container
docker run -p 8000:8000 \
  -e MEMORY_BACKEND=local \
  -e SLM_IMPL=local \
  embed-service

# Or use docker-compose
docker-compose up -d
```

## 🧪 Development Workflow

```bash
# 1. Make changes to code
vim app/services/ranking.py

# 2. Run tests
pytest tests/test_ranking.py -v

# 3. Check linting
ruff check app tests

# 4. Format code
black app tests

# 5. Verify all tests pass
make test

# 6. Run the service
make run
```

## 📚 Learn More

- Full documentation: See `README.md`
- Project summary: See `PROJECT_SUMMARY.md`
- API specification: See http://localhost:8000/docs (when running)

## 🆘 Troubleshooting

### Port Already in Use
```bash
# Change port in .env or use:
uvicorn app.main:app --port 8001
```

### Import Errors
```bash
# Ensure you're in the virtual environment
source venv/bin/activate

# Reinstall in editable mode
pip install -e .
```

### Tests Failing
```bash
# Clear cache and rerun
rm -rf .pytest_cache __pycache__
pytest -v
```

### Memory Not Persisting
```bash
# Check data directory exists
mkdir -p data

# Ensure MEMORY_BACKEND=local
echo "MEMORY_BACKEND=local" >> .env
```

## ✨ Next Steps

1. Try different user inputs to see how hypotheses adapt
2. Build up a user's memory and persona through interactions
3. Export memory to see what the system has learned
4. Integrate with your LLM of choice using the enriched prompts
5. Customize the hypothesis patterns in `app/clients/slm/local.py`
6. Add your own memory types and facets

Happy coding! 🎉

