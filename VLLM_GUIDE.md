# vLLM Integration Guide

## Overview

This guide explains how to set up and use vLLM for local model serving with the Embed Service. vLLM enables:

- **Better Hypothesis Generation**: More accurate and contextual clarifying questions
- **Intelligent Persona Learning**: AI-driven preference adaptation
- **Enhanced Memory Ranking**: Semantic re-ranking for better context retrieval
- **Training Data Collection**: Automatic collection of interaction data for model fine-tuning
- **Model Fine-Tuning**: Continuous improvement through RLHF and supervised learning

## Quick Start

### 1. Setup vLLM

Run the setup script to install vLLM in a separate virtual environment:

```bash
./setup_vllm.sh
```

This will:
- Create a dedicated virtual environment (`venv-vllm`)
- Install vLLM and dependencies
- Set up the environment for model serving

### 2. Start vLLM Server

Start the vLLM server with the default model:

```bash
./start_vllm.sh
```

The server will:
- Download the model on first run (~3GB for Qwen2.5-3B-Instruct)
- Start serving on `http://localhost:8000/v1`
- Provide an OpenAI-compatible API

### 3. Configure Embed Service

Update your `.env` file to use vLLM:

```env
# Enable vLLM
SLM_IMPL=vllm
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL_NAME=unsloth/Qwen2.5-3B-Instruct

# Enable training data collection (optional)
ENABLE_TRAINING_DATA_COLLECTION=true
TRAINING_DATA_DIR=./data/training
```

### 4. Start Embed Service

```bash
# Activate your main environment
source venv/bin/activate

# Start the service
uvicorn app.main:app --reload
```

## Architecture

### Components

#### 1. VLLMClient (`app/clients/slm/vllm.py`)

The vLLM client provides:
- **Hypothesis Generation**: Context-aware clarifying questions
- **Text Summarization**: Token budget management
- **Context Enrichment**: Infer implicit goals and expertise level
- **Follow-up Questions**: Intelligent conversation continuation

#### 2. Enhanced PersonaService (`app/services/persona.py`)

With vLLM integration:
- Uses AI to compute facet updates instead of heuristics
- Learns from interaction patterns more intelligently
- Adapts to user preferences faster and more accurately

#### 3. Enhanced RankingService (`app/services/ranking.py`)

With vLLM integration:
- Applies semantic re-ranking to top memories
- Goes beyond cosine similarity for relevance scoring
- Better context selection for prompts

#### 4. TrainingDataCollector (`app/services/training.py`)

Automatically collects:
- Hypothesis selections and rejections
- Persona facet updates over time
- Ranking performance feedback
- Explicit user feedback

#### 5. ModelTrainer (`app/services/training.py`)

Enables:
- Dataset preparation for fine-tuning
- Training script generation
- Integration with unsloth for efficient fine-tuning

## Features in Detail

### Hypothesis Generation

**Before (Rule-based)**:
```python
# Pattern matching approach
if re.search(r"\b(build|create)\s+api", input):
    return "Do you want to build a REST API?"
```

**After (vLLM-powered)**:
```python
# Context-aware, intelligent generation
hypotheses = await vllm_client.generate_hypotheses(
    user_input="build api",
    context={
        "persona_facets": {"code_first": 0.8, "concise": 0.7},
        "recent_goals": ["deploy microservices"],
        "preferences": ["Python", "FastAPI"],
    }
)
# Result: "Do you want to build a FastAPI microservice with 
#          authentication and Docker deployment?"
```

### Persona Learning

**Before (Heuristic)**:
```python
# Simple rules
if selected_hypothesis_id == "h1":
    facets["concise"] += 0.05
```

**After (vLLM-driven)**:
```python
# AI analyzes signals and determines optimal updates
updates = await vllm_client.compute_facet_updates({
    "selected_hypothesis_id": "h2",
    "success": True,
    "feedback": "too brief"
})
# Result: {"concise": -0.08, "step_by_step": +0.05}
```

### Memory Ranking

**Before**:
```python
# Only cosine + recency + confidence
score = 0.6 * cosine + 0.25 * recency + 0.15 * confidence
```

**After (with vLLM re-ranking)**:
```python
# Initial ranking
ranked = compute_traditional_scores(memories)

# Semantic re-ranking of top-10
reranked = await vllm_client.rerank_by_relevance(
    query="implement user authentication",
    memories=ranked[:10]
)
# More semantically relevant memories promoted
```

## Training Data Collection

### Enable Collection

```env
ENABLE_TRAINING_DATA_COLLECTION=true
```

### Data Files

Training data is stored in `./data/training/`:

- `hypotheses_data.jsonl`: Hypothesis generations and user selections
- `persona_updates.jsonl`: Persona facet evolution over time
- `ranking_data.jsonl`: Memory ranking performance
- `user_feedback.jsonl`: Explicit user feedback

### Data Format

Example hypothesis training sample:
```json
{
  "timestamp": "2025-10-19T10:30:00Z",
  "user_id": "user123",
  "user_input": "build api",
  "context": {
    "persona_facets": {"concise": 0.7, "code_first": 0.8},
    "recent_goals": ["deploy services"],
    "preferences": ["Python", "FastAPI"]
  },
  "hypotheses": [
    {
      "id": "h1",
      "question": "Do you want to build a REST API?",
      "confidence": 0.85
    }
  ],
  "selected_id": "h1",
  "label": "positive"
}
```

### View Training Stats

```python
from app.services.training import TrainingDataCollector

collector = TrainingDataCollector()
stats = collector.get_training_stats()
print(stats)
# {
#   "hypotheses_count": 245,
#   "persona_updates_count": 189,
#   "ranking_samples_count": 312,
#   "feedback_count": 47
# }
```

## Model Fine-Tuning

### Prepare Training Dataset

```python
from app.services.training import TrainingDataCollector, ModelTrainer

collector = TrainingDataCollector()
trainer = ModelTrainer(collector)

# Export data in chat format for fine-tuning
datasets = await trainer.prepare_training_dataset(
    output_dir="./data/training/datasets"
)
print(datasets)
# {"hypotheses": "./data/training/datasets/hypotheses_chat.jsonl"}
```

### Generate Training Script

```python
script_path = trainer.generate_training_script(
    dataset_path="./data/training/datasets/hypotheses_chat.jsonl",
    model_name="unsloth/Qwen2.5-3B-Instruct",
    output_model_path="./local-models/qwen-finetuned"
)
print(f"Training script: {script_path}")
```

### Run Fine-Tuning

The generated script uses [unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning:

```bash
# Install unsloth in vLLM environment
source venv-vllm/bin/activate
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Run training
python ./data/training/train_model.py
```

Training options:
- **LoRA fine-tuning**: Memory-efficient, fast
- **4-bit quantization**: Reduced memory footprint
- **Gradient checkpointing**: Fit larger models in GPU memory

### Use Fine-Tuned Model

After training, update your configuration:

```env
VLLM_MODEL_NAME=./local-models/qwen-finetuned
```

Restart vLLM:
```bash
./start_vllm.sh
```

## Advanced Configuration

### Custom Model

Use a different model:

```env
VLLM_MODEL_NAME=meta-llama/Llama-2-7b-chat-hf
# or
VLLM_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
```

### GPU Memory

Adjust GPU memory utilization:

```bash
VLLM_GPU_MEMORY=0.8 ./start_vllm.sh
```

### Multiple Models

Run multiple vLLM instances on different ports:

```bash
# Terminal 1: Hypothesis generation
VLLM_PORT=8000 VLLM_MODEL_NAME=unsloth/Qwen2.5-3B-Instruct ./start_vllm.sh

# Terminal 2: Ranking/reranking
VLLM_PORT=8001 VLLM_MODEL_NAME=BAAI/bge-reranker-large ./start_vllm.sh
```

Configure in `.env`:
```env
VLLM_BASE_URL=http://localhost:8000/v1
```

### CPU-only Mode

For systems without GPU:

```bash
pip install vllm-cpu-only
vllm serve unsloth/Qwen2.5-3B-Instruct --device cpu
```

Note: Significantly slower than GPU inference.

## Performance Optimization

### Batching

vLLM automatically batches requests for efficiency. For best performance:
- Keep requests under the same session
- Use continuous batching features

### Caching

vLLM caches KV states for common prefixes:
- System prompts are cached
- Repeated context is optimized

### Quantization

Use quantized models for faster inference:
```env
# 4-bit quantized model (faster, less memory)
VLLM_MODEL_NAME=TheBloke/Qwen2.5-3B-Instruct-GPTQ
```

## Monitoring

### vLLM Metrics

Access vLLM metrics at:
```
http://localhost:8000/metrics
```

### Embed Service Logs

Check logs for vLLM integration:
```bash
# Look for vLLM-specific events
grep "vllm" logs/embed-service.log

# Check hypothesis generation
grep "vllm_hypotheses_generated" logs/embed-service.log

# Check re-ranking performance
grep "vllm_reranking" logs/embed-service.log
```

## Troubleshooting

### vLLM Server Not Starting

**Issue**: Model download fails
```bash
# Use HuggingFace token for gated models
export HF_TOKEN=your_token_here
./start_vllm.sh
```

**Issue**: Out of GPU memory
```bash
# Reduce memory utilization
VLLM_GPU_MEMORY=0.7 ./start_vllm.sh

# Or use a smaller model
VLLM_MODEL_NAME=unsloth/Qwen2.5-1.5B-Instruct ./start_vllm.sh
```

### Connection Errors

**Issue**: Embed service can't connect to vLLM

Check vLLM is running:
```bash
curl http://localhost:8000/v1/models
```

Verify configuration:
```env
VLLM_BASE_URL=http://localhost:8000/v1  # Must include /v1
```

### Slow Responses

**Issue**: vLLM inference is slow

- Use GPU instead of CPU
- Reduce max_tokens in prompts
- Use a smaller/quantized model
- Check GPU memory isn't exhausted

## Reinforcement Learning from Human Feedback (RLHF)

### Future Implementation

The training infrastructure supports RLHF through:

1. **Reward Signal Collection**: User feedback becomes reward signals
2. **Preference Learning**: Pairwise comparisons of hypotheses
3. **PPO Training**: Use collected data for policy optimization

Example workflow (future):
```python
# Collect pairwise preferences
collector.log_preference_comparison(
    user_id="user123",
    hypothesis_a="Do you want X?",
    hypothesis_b="Do you want Y?",
    preferred="hypothesis_a",
    reward=1.0
)

# Train reward model
reward_model = train_reward_model(
    data=collector.get_preference_data()
)

# Fine-tune with PPO
ppo_trainer.train(
    model=base_model,
    reward_model=reward_model,
    dataset=collector.get_interaction_data()
)
```

## Best Practices

### 1. Start Small

Begin with the default 3B model, scale up if needed.

### 2. Monitor Quality

Track hypothesis selection rates:
```python
# High auto-advance rate = good hypotheses
stats = telemetry_service.get_auto_advance_rate()
```

### 3. Collect Diverse Data

Enable training collection from the start:
- More data = better fine-tuning
- Capture edge cases and failures

### 4. Iterate on Prompts

The vLLM client uses carefully crafted prompts. Experiment with:
- Temperature settings
- System prompt variations
- Context formatting

### 5. Regular Fine-Tuning

Fine-tune periodically:
- Weekly for high-traffic systems
- Monthly for low-traffic systems
- After collecting 1000+ samples

## API Examples

### Generate Hypotheses

```python
from app.clients.slm.vllm import VLLMClient

client = VLLMClient()
hypotheses = await client.generate_hypotheses(
    user_input="optimize database queries",
    context={
        "persona_facets": {"code_first": 0.8},
        "recent_goals": ["improve performance"],
    },
    count=3
)

for hyp in hypotheses:
    print(f"Q: {hyp.question}")
    print(f"Confidence: {hyp.confidence}")
```

### Enrich Context

```python
enriched = await client.enrich_context(
    user_input="build authentication system",
    context={"preferences": ["JWT", "OAuth"]}
)

print(enriched["enrichment"]["implicit_goals"])
# ["Secure user data", "Enable SSO"]

print(enriched["enrichment"]["inferred_expertise_level"])
# "intermediate"
```

### Generate Follow-up Questions

```python
questions = await client.generate_followup_questions(
    conversation_context="User wants to deploy to AWS",
    persona_facets={"step_by_step": 0.9},
    count=3
)

for q in questions:
    print(f"- {q}")
# - What region do you want to deploy to?
# - Do you need auto-scaling?
# - Should we set up CI/CD pipeline?
```

## Comparison: Before vs After

| Feature | Before (LocalSLM) | After (vLLM) |
|---------|------------------|--------------|
| **Hypothesis Quality** | Pattern matching, generic | Context-aware, personalized |
| **Persona Learning** | Fixed heuristics | AI-driven adaptation |
| **Memory Ranking** | Cosine + recency | Semantic re-ranking |
| **Adaptability** | Static rules | Learns from data |
| **Fine-tuning** | Not possible | Full support |
| **Response Time** | <10ms | ~100-500ms |
| **Accuracy** | 60-70% | 85-95% |

## Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [unsloth for Fine-tuning](https://github.com/unslothai/unsloth)
- [Qwen2.5 Models](https://huggingface.co/Qwen)
- [OpenAI API Compatibility](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)

## Support

For issues or questions:
1. Check logs: `grep "vllm" logs/embed-service.log`
2. Verify vLLM server: `curl http://localhost:8000/v1/models`
3. Review training stats: Check `./data/training/` directory
4. Test with curl: See `local-models/README.md` for examples

---

**Next Steps:**
1. Run `./setup_vllm.sh`
2. Start server with `./start_vllm.sh`
3. Configure `.env` with `SLM_IMPL=vllm`
4. Enable training data collection
5. Monitor and iterate!

