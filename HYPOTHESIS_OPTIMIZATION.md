# Hypothesis Generation Optimization

## Summary

The hypothesis generation system has been **significantly optimized** to use semantic search and relevant memory retrieval instead of just fetching recent memories.

## What Changed

### Before (Inefficient):
```python
# OLD: Get most RECENT memories by type
goals = await memory_provider.get_memories(user_id, MemoryType.GOAL, limit=3)
preferences = await memory_provider.get_memories(user_id, MemoryType.PREFERENCE, limit=5)
styles = await memory_provider.get_memories(user_id, MemoryType.STYLE, limit=3)

# Limited context:
# - Only 3 goals, 5 preferences, 3 styles
# - NO history or artifacts
# - NO relevance to current input
# - Just chronologically recent, not semantically relevant
```

### After (Optimized):
```python
# NEW: Generate embedding for current input
query_embedding = generate_embedding(
    text=input_text,
    media_type=media_type,
    media_url=media_url,
    media_base64=media_base64,
)

# NEW: Semantic search for RELEVANT memories
relevant_memories = await memory_provider.search_memories(
    user_id,
    query_embedding,
    limit=20,  # Get top 20 most relevant
)

# NEW: Categorize and use ALL memory types
# - Relevant goals (semantically similar to current input)
# - Relevant preferences
# - Style preferences
# - Similar past interactions (HISTORY - NEW!)
# - Relevant artifacts (code/docs - NEW!)
```

## Key Improvements

### 1. **Semantic Search Instead of Recency**
- **Before**: "Give me the 3 most recent goals"
- **After**: "Give me the 3 goals most relevant to what the user is asking about NOW"

### 2. **Expanded Memory Types**
- **Before**: Only GOAL, PREFERENCE, STYLE
- **After**: GOAL, PREFERENCE, STYLE, **HISTORY**, **ARTIFACT**

### 3. **Pattern Recognition**
- Now includes similar past interactions (`similar_history`)
- Helps identify recurring patterns and user preferences
- Model can say "You've done something similar before where you..."

### 4. **Richer Context for SLM**
The prompt now includes:
```
Relevant Context (Semantically Matched to Current Input):
Relevant Goals:
  - Build a REST API with FastAPI
  - Deploy to production with Docker

Preferences:
  - Prefers Python over JavaScript
  - Uses VS Code as primary editor

Similar Past Interactions:
  - "User requested: build an API... | Selected: FastAPI implementation"
  - "User requested: containerize app... | Selected: Docker setup"

Relevant Past Work:
  - Created FastAPI endpoint for user authentication
  - Implemented Docker multi-stage build

(Based on 15 relevant past memories)
```

### 5. **Better Confidence Scoring**
- SLM is instructed to give higher confidence to hypotheses that match past patterns
- Local SLM boosts confidence by +5% when similar history exists

## Impact on Hypothesis Quality

### Example: User asks "deploy my app"

#### Before (Recent Memories):
```json
{
  "hypotheses": [
    {
      "question": "Do you want to set up a CI/CD pipeline?",
      "confidence": 0.80,
      "rationale": "Generic deployment question"
    }
  ]
}
```

#### After (Semantic Search):
```json
{
  "hypotheses": [
    {
      "question": "Do you want to deploy your FastAPI app to AWS using Docker, similar to your previous setup?",
      "confidence": 0.92,
      "rationale": "User has deployed FastAPI apps with Docker before, and recent work shows AWS preference"
    }
  ]
}
```

## Performance Considerations

- **Embedding Generation**: Adds ~10-50ms for text, ~50-150ms for images
- **Semantic Search**: Fast (uses vector similarity)
- **Memory Limit**: Fetches 20 instead of 11 memories, but only most relevant ones
- **Net Result**: Slightly slower (50-200ms), but **much better quality**

## Configuration

No configuration changes needed! The optimization is automatic and backward compatible.

## Logs

New debug logging shows memory retrieval stats:
```python
logger.debug(
    "hypothesis_context_built",
    user_id=user_id,
    goals=len(goals),
    preferences=len(preferences),
    history=len(history),  # NEW
    artifacts=len(artifacts),  # NEW
    total_memories=len(relevant_memories),
)
```

## Testing

Test with an existing user that has history:

```bash
# First interaction (building an API)
curl -X POST http://localhost:8000/v1/hypothesize \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "alice",
    "input_text": "build an API"
  }'

# Execute and create memories...

# Later interaction (semantically similar)
curl -X POST http://localhost:8000/v1/hypothesize \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "alice",
    "input_text": "create REST endpoints"
  }'

# Should now reference past API work!
```

## Files Modified

1. **`app/services/hypothesizer.py`**
   - Updated `_build_context()` to use semantic search
   - Added categorization of memory types
   - Added debug logging

2. **`app/clients/slm/ollama.py`**
   - Updated `_build_hypothesis_prompt()` to use new context fields
   - Added instructions for SLM to leverage context
   - Updated `enrich_context()` to use semantic results

3. **`app/clients/slm/local.py`**
   - Updated documentation to reflect new context structure
   - Added confidence boost for users with relevant history

## Future Enhancements

Potential further optimizations:

1. **Cache embeddings** - Don't regenerate for identical inputs
2. **Adaptive limits** - Fetch more memories for power users
3. **Temporal weighting** - Balance recency with relevance
4. **Cross-user patterns** - Learn from similar users (privacy-aware)
5. **Training data** - Use hypothesis selections to fine-tune models

## Backward Compatibility

✅ **Fully backward compatible**
- Old clients still work (they just ignore new context fields)
- New code handles missing context fields gracefully
- No breaking changes to API

---

**Questions?** Check the implementation in `app/services/hypothesizer.py` and `app/clients/slm/ollama.py`.

