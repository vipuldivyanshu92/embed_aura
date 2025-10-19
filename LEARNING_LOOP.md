# Learning Loop Documentation

## Overview

The Embed Service implements a **comprehensive learning loop** that trains from every user interaction. When a user selects a hypothesis in `/v1/execute`, the system:

1. **Logs training data** for model improvement
2. **Extracts and stores goals/preferences** from the selection
3. **Updates persona facets** (AI-enhanced with Ollama/vLLM)
4. **Stores interaction history** for future context

This creates a **continuous improvement cycle** where the system gets smarter with each interaction.

## How It Works

### Flow Diagram

```
User Input → Generate Hypotheses → User Selects → LEARNING LOOP → Better Hypotheses Next Time
                                         ↓
                            ┌────────────┴────────────┐
                            │                         │
                    Training Data            Update Memories
                    Collection               & Persona
                            │                         │
                            ↓                         ↓
                    Model Fine-tuning         Better Context
                    (periodic)                (immediate)
```

### Step-by-Step Process

When a user calls `/v1/execute` with a `hypothesis_id`:

#### 1. **Training Data Collection** (Lines 335-353)

If `ENABLE_TRAINING_DATA_COLLECTION=true`:

```python
await training_collector.log_hypothesis_selection(
    user_id=request.user_id,
    user_input=request.input_text,
    context={
        "persona_facets": persona.facets,
        "interaction_count": persona.interaction_count,
        "recent_goals": [...],
        "preferences": [...],
    },
    hypotheses=hypotheses,  # All 3 hypotheses
    selected_id=request.hypothesis_id,  # Which one user chose
)
```

**What gets logged:**
- User's input
- User's current persona state
- All generated hypotheses
- Which hypothesis was selected
- Timestamp

**Saved to:** `./data/training/hypotheses_data.jsonl`

**Used for:** Fine-tuning the model to generate better hypotheses

#### 2. **Goal Extraction** (Lines 359-377)

The system analyzes the selected hypothesis to extract **goals**:

```python
# Keywords that indicate a goal
goal_keywords = ["want to", "need to", "build", "create", "implement", "deploy"]

# Example hypothesis: "Do you want to build a REST API with FastAPI?"
# Extracted goal: "build a REST API with FastAPI"
```

**If detected, creates a GOAL memory:**
```python
MemoryItem(
    mtype=MemoryType.GOAL,
    content="build a REST API with FastAPI",
    confidence=0.88,  # From hypothesis
    tags=["learned", "hypothesis_derived"]
)
```

**Impact:** Next time user interacts, this goal will:
- Be included in hypothesis generation context
- Help personalize future suggestions
- Be ranked by relevance

#### 3. **Preference Extraction** (Lines 379-397)

The system detects **preferences** from hypothesis selection:

```python
# Keywords that indicate preferences
preference_keywords = ["prefer", "like", "using", "with", "framework"]

# Example: "...with FastAPI and Docker deployment"
# Preference learned: "Prefers: FastAPI, Docker"
```

**If detected, creates a PREFERENCE memory:**
```python
MemoryItem(
    mtype=MemoryType.PREFERENCE,
    content="Prefers: FastAPI, Docker deployment",
    confidence=0.79,  # Slightly lower than hypothesis
    tags=["learned", "hypothesis_derived"]
)
```

**Impact:** Future hypotheses will favor:
- FastAPI over other frameworks
- Docker-based solutions
- Similar technology stack

#### 4. **Persona Update** (Lines 399-425)

Updates the user's persona facets based on which hypothesis they selected:

```python
# Hypothesis position reveals preferences
if hypothesis_id == "h1":  # First/most direct
    persona_signals = {"facet_updates": {"concise": +0.02}}
    # User prefers concise responses

elif hypothesis_id in ["h2", "h3"]:  # More detailed options
    persona_signals = {"facet_updates": {"step_by_step": +0.02}}
    # User prefers detailed explanations
```

**With AI Enhancement (Ollama/vLLM):**
The PersonaService uses AI to analyze the full interaction context and compute better facet updates:

```python
# AI analyzes:
# - Which hypothesis was selected
# - User's current facets
# - Success signals
# - Interaction history

# AI determines optimal updates:
{
    "concise": +0.05,      # Strong signal
    "code_first": +0.03,   # Inferred from hypothesis
    "formal": -0.02,       # Casual language detected
}
```

**Logged for training:**
```python
await training_collector.log_persona_update(
    old_facets={"concise": 0.70, "code_first": 0.75, ...},
    new_facets={"concise": 0.75, "code_first": 0.78, ...},
    signals=persona_signals
)
```

**Saved to:** `./data/training/persona_updates.jsonl`

#### 5. **History Storage** (Lines 427-443)

Stores the complete interaction:

```python
MemoryItem(
    mtype=MemoryType.HISTORY,
    content="User requested: build api... | Selected: Do you want to build a REST API?",
    tags=["interaction", "text", "h1"],  # Includes which hypothesis
)
```

**Impact:** Provides conversation history for:
- Future context in prompt building
- Understanding user's journey
- Debugging and analytics

## Example Interaction

### First Interaction

**User:** "build api"

**System generates:**
1. h1: "Do you want to build a REST API?" (0.75)
2. h2: "Do you want to design API endpoints?" (0.68)
3. h3: "Do you want to set up authentication?" (0.60)

**User selects:** h1

**Learning loop executes:**
```
✅ Training data logged
✅ Goal extracted: "build a REST API"
✅ Preference detected: (none specific yet)
✅ Persona updated: concise +0.02
✅ History stored: interaction recorded
```

### Second Interaction (Same User)

**User:** "deploy api"

**System generates (now with learned context):**
1. h1: "Do you want to deploy your REST API to AWS?" (0.85) ⬆️
2. h2: "Do you want to containerize with Docker?" (0.78)
3. h3: "Do you want to set up CI/CD pipeline?" (0.72)

**Notice:**
- Higher confidence (0.85 vs 0.75) - learned from history
- More specific ("REST API" vs generic "API") - used GOAL memory
- AWS mentioned (if detected in context) - used PREFERENCE
- Still concise (h1 format) - respects persona facet

**User selects:** h2

**Learning loop executes:**
```
✅ Training data logged (now with more context)
✅ Goal updated: "deploy REST API with Docker"
✅ Preference learned: "Docker, containerization"
✅ Persona updated: step_by_step +0.02 (chose h2)
✅ History stored
```

### Third Interaction

**User:** "auth"

**System generates (even smarter now):**
1. h1: "Add JWT auth to your containerized REST API?" (0.90) ⬆️⬆️
2. h2: "Set up OAuth2 with Docker secrets?" (0.82)
3. h3: "Implement role-based access control?" (0.75)

**Notice:**
- Very high confidence (0.90) - strong learned context
- Connects to previous goals (REST API + Docker)
- Specific technologies inferred
- Balanced format (learned user sometimes wants detail)

## Training Data Format

### Hypothesis Selection Log

```json
{
  "timestamp": "2025-10-19T10:30:00Z",
  "user_id": "user123",
  "user_input": "build api",
  "context": {
    "persona_facets": {"concise": 0.70, "code_first": 0.80},
    "interaction_count": 5,
    "recent_goals": ["Deploy microservices"],
    "preferences": ["Python", "FastAPI", "Docker"]
  },
  "hypotheses": [
    {"id": "h1", "question": "...", "confidence": 0.85},
    {"id": "h2", "question": "...", "confidence": 0.75},
    {"id": "h3", "question": "...", "confidence": 0.65}
  ],
  "selected_id": "h1",
  "label": "positive"
}
```

### Persona Update Log

```json
{
  "timestamp": "2025-10-19T10:30:05Z",
  "user_id": "user123",
  "old_facets": {"concise": 0.70, "code_first": 0.80},
  "new_facets": {"concise": 0.72, "code_first": 0.82},
  "facet_deltas": {"concise": +0.02, "code_first": +0.02},
  "signals": {
    "selected_hypothesis_id": "h1",
    "success": true
  }
}
```

## Fine-Tuning Workflow

Once you've collected enough training data (recommended: 100+ interactions):

### 1. Check Training Data

```python
from app.services.training import TrainingDataCollector

collector = TrainingDataCollector()
stats = collector.get_training_stats()

print(f"Hypotheses: {stats['hypotheses_count']} samples")
print(f"Persona updates: {stats['persona_updates_count']}")
```

### 2. Prepare Dataset

```python
from app.services.training import ModelTrainer

trainer = ModelTrainer(collector)
datasets = await trainer.prepare_training_dataset()

# Output: ./data/training/datasets/hypotheses_chat.jsonl
```

### 3. Fine-Tune Model

For **Ollama**:
```bash
# Export for external fine-tuning
# Then import back:
ollama create my-finetuned-model -f Modelfile
```

For **vLLM** (Linux):
```bash
source venv-vllm/bin/activate
python ./data/training/train_model.py
```

### 4. Deploy Fine-Tuned Model

```env
# Update .env
OLLAMA_MODEL_NAME=my-finetuned-model
# or
VLLM_MODEL_NAME=./local-models/qwen-finetuned
```

### 5. Measure Improvement

Track metrics:
- Hypothesis selection rate (should increase)
- Auto-advance rate (should increase)
- User satisfaction (if collected)

## Configuration

Enable learning loop in `.env`:

```env
# Enable training data collection
ENABLE_TRAINING_DATA_COLLECTION=true

# Where to store training data
TRAINING_DATA_DIR=./data/training

# Minimum confidence to include in training
MIN_CONFIDENCE_FOR_TRAINING=0.7
```

## Memory Types Created

| Type | When | Example |
|------|------|---------|
| **GOAL** | Hypothesis indicates intent | "build a REST API with FastAPI" |
| **PREFERENCE** | Hypothesis mentions technology | "Prefers: Docker, containerization" |
| **HISTORY** | Every interaction | "User requested: build api..." |

## Persona Facets Updated

| Facet | Updated When | Direction |
|-------|--------------|-----------|
| **concise** | User selects h1 | Increase |
| **concise** | User selects h2/h3 | Decrease |
| **step_by_step** | User selects h2/h3 | Increase |
| **code_first** | AI detects code preference | AI determines |
| **formal** | AI detects language style | AI determines |

## Benefits

### Immediate (Per Interaction)
- ✅ Goals and preferences stored as memories
- ✅ Persona facets updated
- ✅ Better context for next interaction
- ✅ History tracked for debugging

### Short-term (Per Session)
- ✅ Increasingly accurate hypotheses
- ✅ Higher confidence scores
- ✅ Better auto-advance rate
- ✅ More personalized experience

### Long-term (Across Sessions)
- ✅ Rich training dataset collected
- ✅ Model fine-tuning possible
- ✅ Continuous improvement
- ✅ User-specific optimization

## Monitoring

Check logs for learning events:

```bash
# Goal learned
grep "goal_learned_from_hypothesis" logs/app.log

# Preference learned
grep "preference_learned" logs/app.log

# Training data logged
grep "training_data_logged" logs/app.log

# Learning loop complete
grep "learning_loop_complete" logs/app.log
```

View training stats:
```python
from app.services.training import TrainingDataCollector

collector = TrainingDataCollector()
stats = collector.get_training_stats()
```

## Best Practices

### 1. Enable from Day One
```env
ENABLE_TRAINING_DATA_COLLECTION=true
```
More data = better fine-tuning later

### 2. Monitor Quality
Review collected data periodically:
```bash
head -10 ./data/training/hypotheses_data.jsonl
```

### 3. Fine-Tune Regularly
- Weekly for high-traffic systems
- Monthly for low-traffic systems
- After major usage patterns change

### 4. A/B Test Improvements
Compare metrics before/after fine-tuning:
- Hypothesis selection rate
- Auto-advance rate
- User retention

### 5. Clean Bad Data
Remove low-quality samples:
```python
# Filter by confidence
samples = [s for s in training_data if s['selected_confidence'] > 0.7]
```

## Troubleshooting

### Training data not being logged

Check:
```env
ENABLE_TRAINING_DATA_COLLECTION=true  # Must be true
```

Verify in logs:
```bash
grep "training_data_logged" logs/app.log
```

### No goals/preferences extracted

This is normal if hypotheses don't contain relevant keywords. The system is conservative to avoid false positives.

Check logs:
```bash
grep "goal_learned\|preference_learned" logs/app.log
```

### Persona not updating

Verify AI client is enabled:
```env
SLM_IMPL=ollama  # or vllm
```

Check persona changes:
```python
persona = await persona_service.get_or_create_persona("user123")
print(persona.facets)
```

## API Response

The execute endpoint doesn't return learning details by default, but you can see the effects:

```json
{
  "enriched_prompt": "...",  // Will include learned goals/prefs
  "tokens_estimate": 1234,
  "context_breakdown": {
    "goal": 15,      // Token allocation
    "pref": 12,      // Will increase as more learned
    "history": 25    // Will grow with interactions
  }
}
```

## Summary

The learning loop transforms the Embed Service from a stateless hypothesis generator to a **continuously improving, personalized AI assistant** that:

1. **Learns from every interaction**
2. **Stores learned knowledge as memories**
3. **Updates user preferences dynamically**
4. **Collects training data automatically**
5. **Enables model fine-tuning**
6. **Gets smarter over time**

All this happens automatically with zero configuration beyond enabling training data collection!

---

**See Also:**
- [OLLAMA_GUIDE.md](OLLAMA_GUIDE.md) - AI enhancement with Ollama
- [VLLM_GUIDE.md](VLLM_GUIDE.md) - AI enhancement with vLLM
- [Training Service](app/services/training.py) - Implementation details

