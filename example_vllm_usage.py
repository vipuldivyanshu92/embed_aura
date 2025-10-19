#!/usr/bin/env python3
"""
Example script demonstrating vLLM integration with the Embed Service.

This script shows how the vLLM-powered features improve:
1. Hypothesis generation
2. Persona learning
3. Memory ranking
4. Training data collection

Prerequisites:
- vLLM server running (./start_vllm.sh)
- Environment configured with SLM_IMPL=vllm
"""

import asyncio
from app.clients.slm.vllm import VLLMClient
from app.services.training import TrainingDataCollector, ModelTrainer
from app.models import Hypothesis


async def demo_hypothesis_generation():
    """Demonstrate improved hypothesis generation."""
    print("\n" + "="*60)
    print("1. HYPOTHESIS GENERATION DEMO")
    print("="*60)
    
    client = VLLMClient()
    
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
    
    print(f"\nGenerating hypotheses with vLLM...")
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
    print("2. CONTEXT ENRICHMENT DEMO")
    print("="*60)
    
    client = VLLMClient()
    
    context = {
        "interaction_count": 15,
        "preferences": ["React", "TypeScript", "Tailwind CSS"],
        "recent_goals": ["Build dashboard"],
    }
    
    user_input = "create a user analytics dashboard"
    
    print(f"\nUser Input: '{user_input}'")
    print(f"Existing Context: {context['preferences']}")
    
    print(f"\nEnriching context with vLLM...")
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
    print("3. FOLLOW-UP QUESTIONS DEMO")
    print("="*60)
    
    client = VLLMClient()
    
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
    print(f"\nGenerating follow-up questions...")
    
    questions = await client.generate_followup_questions(
        conversation_context=conversation_context,
        persona_facets=persona_facets,
        count=3,
    )
    
    print(f"\nGenerated {len(questions)} follow-up questions:\n")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")
    
    await client.close()


async def demo_training_data_collection():
    """Demonstrate training data collection."""
    print("\n" + "="*60)
    print("4. TRAINING DATA COLLECTION DEMO")
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


async def demo_training_preparation():
    """Demonstrate training data preparation."""
    print("\n" + "="*60)
    print("5. TRAINING PREPARATION DEMO")
    print("="*60)
    
    collector = TrainingDataCollector()
    trainer = ModelTrainer(collector)
    
    stats = collector.get_training_stats()
    
    if stats['hypotheses_count'] > 0:
        print(f"\nFound {stats['hypotheses_count']} training samples")
        print("\nPreparing training datasets...")
        
        datasets = await trainer.prepare_training_dataset()
        
        print(f"\nDatasets prepared:")
        for name, path in datasets.items():
            print(f"  - {name}: {path}")
        
        print("\nGenerating training script...")
        script_path = trainer.generate_training_script(
            dataset_path=datasets["hypotheses"],
            model_name="unsloth/Qwen2.5-3B-Instruct",
            output_model_path="./local-models/qwen-finetuned",
        )
        
        print(f"✓ Training script generated: {script_path}")
        print(f"\nTo fine-tune the model, run:")
        print(f"  source venv-vllm/bin/activate")
        print(f"  pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'")
        print(f"  python {script_path}")
    else:
        print("\nNo training data available yet.")
        print("Enable data collection and use the service to generate training samples.")
        print("\nSet in .env:")
        print("  ENABLE_TRAINING_DATA_COLLECTION=true")


async def main():
    """Run all demos."""
    print("\n" + "="*70)
    print(" "*15 + "vLLM INTEGRATION DEMONSTRATION")
    print("="*70)
    
    print("\nThis demo shows how vLLM enhances the Embed Service with:")
    print("  1. Better hypothesis generation")
    print("  2. Context enrichment")
    print("  3. Intelligent follow-up questions")
    print("  4. Training data collection")
    print("  5. Model fine-tuning preparation")
    
    print("\nPrerequisites:")
    print("  ✓ vLLM server running on http://localhost:8000")
    print("  ✓ SLM_IMPL=vllm in .env")
    
    input("\nPress Enter to start the demos...")
    
    try:
        # Run demos
        await demo_hypothesis_generation()
        await demo_context_enrichment()
        await demo_followup_questions()
        await demo_training_data_collection()
        await demo_training_preparation()
        
        print("\n" + "="*70)
        print(" "*20 + "DEMOS COMPLETED!")
        print("="*70)
        
        print("\nNext steps:")
        print("  1. Enable training data collection: ENABLE_TRAINING_DATA_COLLECTION=true")
        print("  2. Use the service to collect interaction data")
        print("  3. Fine-tune your model with collected data")
        print("  4. Deploy fine-tuned model for even better accuracy")
        print("\nSee VLLM_GUIDE.md for detailed documentation.\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  - vLLM server is running (./start_vllm.sh)")
        print("  - SLM_IMPL=vllm is set in .env")
        print("  - vLLM server is accessible at http://localhost:8000/v1")


if __name__ == "__main__":
    asyncio.run(main())

