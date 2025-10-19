#!/usr/bin/env python3
"""
Example script demonstrating Ollama integration with the Embed Service.

This script shows how the Ollama-powered features improve:
1. Hypothesis generation
2. Persona learning
3. Memory ranking
4. Training data collection

Prerequisites:
- Ollama installed and running
- Model pulled (e.g., ollama pull qwen2.5:3b)
- Environment configured with SLM_IMPL=ollama
"""

import asyncio
from app.clients.slm.ollama import OllamaClient
from app.services.training import TrainingDataCollector, ModelTrainer
from app.models import Hypothesis


async def demo_hypothesis_generation():
    """Demonstrate improved hypothesis generation with Ollama."""
    print("\n" + "="*60)
    print("1. HYPOTHESIS GENERATION DEMO (Ollama)")
    print("="*60)
    
    client = OllamaClient(model_name="qwen2.5:3b")
    
    # Context: User who prefers code examples and concise responses
    context = {
        "persona_facets": {
            "code_first": 0.85,
            "concise": 0.75,
            "step_by_step": 0.3,
            "formal": 0.4,
        },
        "interaction_count": 47,
        "recent_goals": [
            "Deploy FastAPI application to AWS",
            "Implement JWT authentication",
        ],
        "preferences": [
            "Python",
            "FastAPI",
            "Docker",
            "PostgreSQL",
        ],
    }
    
    user_input = "build api"
    
    print(f"\nUser Input: '{user_input}'")
    print(f"\nUser Profile:")
    print(f"  - Interaction Count: {context['interaction_count']}")
    print(f"  - Preferences: {', '.join(context['preferences'][:3])}")
    print(f"  - Recent Goals: {context['recent_goals'][0]}")
    print(f"  - Persona: code_first ({context['persona_facets']['code_first']}), "
          f"concise ({context['persona_facets']['concise']})")
    
    print(f"\nGenerating hypotheses with Ollama...")
    hypotheses = await client.generate_hypotheses(
        user_input=user_input,
        context=context,
        count=3,
    )
    
    print(f"\nGenerated {len(hypotheses)} hypotheses:\n")
    for i, hyp in enumerate(hypotheses, 1):
        print(f"{i}. Q: {hyp.question}")
        print(f"   Confidence: {hyp.confidence:.2f}")
        print(f"   Rationale: {hyp.rationale}")
        print()
    
    await client.close()


async def demo_context_enrichment():
    """Demonstrate context enrichment capabilities."""
    print("\n" + "="*60)
    print("2. CONTEXT ENRICHMENT DEMO (Ollama)")
    print("="*60)
    
    client = OllamaClient(model_name="qwen2.5:3b")
    
    context = {
        "interaction_count": 15,
        "preferences": ["React", "TypeScript", "Tailwind CSS"],
        "recent_goals": ["Build dashboard"],
    }
    
    user_input = "create a user analytics dashboard"
    
    print(f"\nUser Input: '{user_input}'")
    print(f"Existing Context: {context['preferences']}")
    
    print(f"\nEnriching context with Ollama...")
    enriched = await client.enrich_context(
        user_input=user_input,
        context=context,
    )
    
    if "enrichment" in enriched:
        print(f"\nEnriched Context:")
        enrichment = enriched["enrichment"]
        
        print(f"\nImplicit Goals:")
        for goal in enrichment.get("implicit_goals", []):
            print(f"  - {goal}")
        
        print(f"\nSuggested Topics:")
        for topic in enrichment.get("suggested_topics", []):
            print(f"  - {topic}")
        
        print(f"\nInferred Expertise Level: {enrichment.get('inferred_expertise_level', 'N/A')}")
        
        print(f"\nLikely Next Steps:")
        for step in enrichment.get("likely_next_steps", []):
            print(f"  - {step}")
    
    await client.close()


async def demo_followup_questions():
    """Demonstrate intelligent follow-up questions."""
    print("\n" + "="*60)
    print("3. FOLLOW-UP QUESTIONS DEMO (Ollama)")
    print("="*60)
    
    client = OllamaClient(model_name="qwen2.5:3b")
    
    conversation_context = """
    User: I need to deploy my FastAPI application
    Assistant: I can help with that. Do you have a specific deployment target?
    User: Yes, AWS
    """
    
    persona_facets = {
        "step_by_step": 0.8,
        "code_first": 0.7,
        "formal": 0.4,
    }
    
    print(f"\nConversation Context:")
    print(conversation_context)
    
    print(f"\nUser Preferences: step-by-step guidance, code examples")
    print(f"\nGenerating follow-up questions with Ollama...")
    
    questions = await client.generate_followup_questions(
        conversation_context=conversation_context,
        persona_facets=persona_facets,
        count=3,
    )
    
    print(f"\nGenerated {len(questions)} follow-up questions:\n")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")
    
    await client.close()


async def demo_performance_comparison():
    """Compare Ollama performance on different model sizes."""
    print("\n" + "="*60)
    print("4. PERFORMANCE COMPARISON")
    print("="*60)
    
    import time
    
    models = ["qwen2.5:3b"]  # Add more if you have them pulled
    
    for model in models:
        try:
            client = OllamaClient(model_name=model)
            
            print(f"\nTesting model: {model}")
            
            start = time.time()
            hypotheses = await client.generate_hypotheses(
                user_input="optimize database queries",
                context={"persona_facets": {}, "interaction_count": 5},
                count=3,
            )
            elapsed = time.time() - start
            
            print(f"  ⏱️  Generation time: {elapsed:.2f}s")
            print(f"  ✅ Generated {len(hypotheses)} hypotheses")
            if hypotheses:
                print(f"  📊 Top confidence: {hypotheses[0].confidence:.2f}")
            
            await client.close()
            
        except Exception as e:
            print(f"  ❌ Error with {model}: {str(e)}")


async def demo_training_data_collection():
    """Demonstrate training data collection."""
    print("\n" + "="*60)
    print("5. TRAINING DATA COLLECTION DEMO")
    print("="*60)
    
    collector = TrainingDataCollector()
    
    # Simulate logging a hypothesis selection
    hypotheses = [
        Hypothesis(
            id="h1",
            question="Do you want to build a REST API with FastAPI?",
            rationale="Based on recent preferences",
            confidence=0.88,
        ),
        Hypothesis(
            id="h2",
            question="Do you want to design API endpoints and data models?",
            rationale="Common next step",
            confidence=0.75,
        ),
        Hypothesis(
            id="h3",
            question="Do you want to set up API authentication?",
            rationale="Security consideration",
            confidence=0.68,
        ),
    ]
    
    context = {
        "persona_facets": {"code_first": 0.8, "concise": 0.7},
        "interaction_count": 25,
        "recent_goals": ["Build microservices"],
        "preferences": ["Python", "FastAPI"],
    }
    
    print("\nLogging hypothesis selection...")
    await collector.log_hypothesis_selection(
        user_id="demo_user",
        user_input="build api",
        context=context,
        hypotheses=hypotheses,
        selected_id="h1",
        auto_advanced=False,
    )
    
    print("✓ Hypothesis selection logged")
    
    # Log persona update
    print("\nLogging persona update...")
    await collector.log_persona_update(
        user_id="demo_user",
        old_facets={"concise": 0.7, "code_first": 0.8},
        new_facets={"concise": 0.75, "code_first": 0.82},
        signals={"selected_hypothesis_id": "h1", "success": True},
    )
    
    print("✓ Persona update logged")
    
    # Get stats
    stats = collector.get_training_stats()
    
    print(f"\nTraining Data Statistics:")
    print(f"  - Hypotheses: {stats['hypotheses_count']} samples")
    print(f"  - Persona Updates: {stats['persona_updates_count']} samples")
    print(f"  - Ranking Data: {stats['ranking_samples_count']} samples")
    print(f"  - User Feedback: {stats['feedback_count']} samples")
    print(f"\nData Directory: {stats['data_dir']}")


async def main():
    """Run all demos."""
    print("\n" + "="*70)
    print(" "*15 + "OLLAMA INTEGRATION DEMONSTRATION")
    print("="*70)
    
    print("\nThis demo shows how Ollama enhances the Embed Service with:")
    print("  1. Better hypothesis generation (context-aware)")
    print("  2. Context enrichment (implicit goals, expertise)")
    print("  3. Intelligent follow-up questions")
    print("  4. Performance benchmarks")
    print("  5. Training data collection")
    
    print("\nPrerequisites:")
    print("  ✓ Ollama installed and running")
    print("  ✓ Model pulled (qwen2.5:3b recommended)")
    print("  ✓ SLM_IMPL=ollama in .env")
    
    # Check if Ollama is accessible
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/version", timeout=5.0)
            if response.status_code == 200:
                print("  ✓ Ollama is running")
            else:
                print("  ⚠️  Ollama may not be running properly")
    except Exception:
        print("\n❌ Error: Cannot connect to Ollama at http://localhost:11434")
        print("\nPlease ensure:")
        print("  1. Ollama is installed: ./setup_ollama.sh")
        print("  2. Ollama is running: Check menu bar or run 'ollama serve'")
        print("  3. Model is pulled: ollama pull qwen2.5:3b")
        return
    
    input("\nPress Enter to start the demos...")
    
    try:
        # Run demos
        await demo_hypothesis_generation()
        await demo_context_enrichment()
        await demo_followup_questions()
        await demo_performance_comparison()
        await demo_training_data_collection()
        
        print("\n" + "="*70)
        print(" "*20 + "DEMOS COMPLETED!")
        print("="*70)
        
        print("\nNext steps:")
        print("  1. Enable training data collection: ENABLE_TRAINING_DATA_COLLECTION=true")
        print("  2. Use the service to collect interaction data")
        print("  3. Review collected data for patterns")
        print("  4. Fine-tune your model with collected data (see OLLAMA_GUIDE.md)")
        print("\nSee OLLAMA_GUIDE.md for detailed documentation.\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  - Ollama is running (check menu bar or run 'ollama serve')")
        print("  - Model is pulled: ollama pull qwen2.5:3b")
        print("  - SLM_IMPL=ollama is set in .env")


if __name__ == "__main__":
    asyncio.run(main())

