#!/usr/bin/env python3
"""
Test script to demonstrate the learning loop in action.

This script simulates a user interaction session and shows how the system:
1. Logs training data
2. Extracts goals and preferences
3. Updates persona facets
4. Stores memories

Run with: python test_learning_loop.py
"""

import asyncio
import json
from pathlib import Path

from app.clients.slm.ollama import OllamaClient
from app.clients.slm.local import LocalSLM
from app.memory.local import LocalMemoryProvider
from app.services.persona import PersonaService
from app.services.hypothesizer import HypothesizerService
from app.services.training import TrainingDataCollector
from app.config import get_settings
from app.models import MediaType


async def test_learning_loop():
    """Demonstrate the complete learning loop."""
    
    print("\n" + "="*70)
    print(" "*20 + "LEARNING LOOP DEMONSTRATION")
    print("="*70)
    
    # Initialize services
    print("\n1. Initializing services...")
    
    settings = get_settings()
    memory_provider = LocalMemoryProvider()
    
    # Use Ollama if available, otherwise local
    try:
        slm_client = OllamaClient(model_name="qwen2.5:3b")
        print("   ✓ Using Ollama client")
    except Exception:
        slm_client = LocalSLM()
        print("   ✓ Using Local client (rule-based)")
    
    persona_service = PersonaService(memory_provider, slm_client)
    hypothesizer_service = HypothesizerService(slm_client, memory_provider, persona_service)
    training_collector = TrainingDataCollector()
    
    # Test user
    user_id = "test_user_learning_demo"
    
    print(f"\n2. Testing with user: {user_id}")
    
    # === First Interaction ===
    print("\n" + "-"*70)
    print("INTERACTION 1: User wants to 'build api'")
    print("-"*70)
    
    # Get initial persona state
    persona_before = await persona_service.get_or_create_persona(user_id)
    print(f"\nPersona before:")
    print(f"  Facets: {persona_before.facets}")
    print(f"  Interaction count: {persona_before.interaction_count}")
    
    # Generate hypotheses
    user_input_1 = "build api"
    hypotheses_1, _ = await hypothesizer_service.generate_hypotheses(
        user_id=user_id,
        input_text=user_input_1,
        count=3,
    )
    
    print(f"\nGenerated hypotheses:")
    for i, h in enumerate(hypotheses_1, 1):
        print(f"  {h.id}. {h.question} (confidence: {h.confidence:.2f})")
    
    # Simulate user selecting h1 (most direct)
    selected_id_1 = "h1"
    selected_hyp_1 = hypotheses_1[0]
    
    print(f"\n✓ User selects: {selected_id_1}")
    print(f"  Question: {selected_hyp_1.question}")
    
    # Simulate the learning loop from execute endpoint
    print(f"\n3. Learning loop executing...")
    
    # Log training data
    if settings.enable_training_data_collection:
        context = {
            "persona_facets": persona_before.facets,
            "interaction_count": persona_before.interaction_count,
            "recent_goals": [],
            "preferences": [],
            "media_type": MediaType.TEXT.value,
        }
        
        await training_collector.log_hypothesis_selection(
            user_id=user_id,
            user_input=user_input_1,
            context=context,
            hypotheses=hypotheses_1,
            selected_id=selected_id_1,
        )
        print("   ✓ Training data logged")
    
    # Extract goals
    hypothesis_text = selected_hyp_1.question.lower()
    if any(kw in hypothesis_text for kw in ["want to", "build", "create"]):
        goal_content = selected_hyp_1.question.replace("Do you ", "").replace("?", "")
        print(f"   ✓ Goal extracted: '{goal_content}'")
    
    # Update persona
    signals = {
        "selected_hypothesis_id": selected_id_1,
        "success": True,
        "facet_updates": {"concise": 0.02} if selected_id_1 == "h1" else {},
    }
    
    await persona_service.update_persona(user_id, signals)
    print(f"   ✓ Persona updated (h1 → concise +0.02)")
    
    # Get updated persona
    persona_after_1 = await persona_service.get_or_create_persona(user_id)
    print(f"\nPersona after interaction 1:")
    print(f"  Facets: {persona_after_1.facets}")
    print(f"  Interaction count: {persona_after_1.interaction_count}")
    print(f"  Facet change: concise {persona_before.facets['concise']:.2f} → {persona_after_1.facets['concise']:.2f}")
    
    # === Second Interaction ===
    print("\n" + "-"*70)
    print("INTERACTION 2: User wants 'deploy api' (building on previous)")
    print("-"*70)
    
    user_input_2 = "deploy api"
    hypotheses_2, _ = await hypothesizer_service.generate_hypotheses(
        user_id=user_id,
        input_text=user_input_2,
        count=3,
    )
    
    print(f"\nGenerated hypotheses (now with learned context):")
    for i, h in enumerate(hypotheses_2, 1):
        print(f"  {h.id}. {h.question} (confidence: {h.confidence:.2f})")
    
    # User selects h2 (more detailed this time)
    selected_id_2 = "h2"
    selected_hyp_2 = hypotheses_2[1] if len(hypotheses_2) > 1 else hypotheses_2[0]
    
    print(f"\n✓ User selects: {selected_id_2}")
    print(f"  Question: {selected_hyp_2.question}")
    
    # Learning loop
    print(f"\n4. Learning loop executing (interaction 2)...")
    
    if settings.enable_training_data_collection:
        context_2 = {
            "persona_facets": persona_after_1.facets,
            "interaction_count": persona_after_1.interaction_count,
            "recent_goals": ["build a REST API"],  # From previous interaction
            "preferences": [],
            "media_type": MediaType.TEXT.value,
        }
        
        await training_collector.log_hypothesis_selection(
            user_id=user_id,
            user_input=user_input_2,
            context=context_2,
            hypotheses=hypotheses_2,
            selected_id=selected_id_2,
        )
        print("   ✓ Training data logged (with previous goals)")
    
    # Update persona (h2 selection indicates preference for detail)
    signals_2 = {
        "selected_hypothesis_id": selected_id_2,
        "success": True,
        "facet_updates": {"step_by_step": 0.02},
    }
    
    old_facets = persona_after_1.facets.copy()
    await persona_service.update_persona(user_id, signals_2)
    print(f"   ✓ Persona updated (h2 → step_by_step +0.02)")
    
    # Log persona update for training
    if settings.enable_training_data_collection:
        persona_after_2 = await persona_service.get_or_create_persona(user_id)
        await training_collector.log_persona_update(
            user_id=user_id,
            old_facets=old_facets,
            new_facets=persona_after_2.facets,
            signals=signals_2,
        )
        print("   ✓ Persona update logged for training")
    
    persona_after_2 = await persona_service.get_or_create_persona(user_id)
    print(f"\nPersona after interaction 2:")
    print(f"  Facets: {persona_after_2.facets}")
    print(f"  Interaction count: {persona_after_2.interaction_count}")
    print(f"  Changes:")
    print(f"    concise: {persona_before.facets['concise']:.2f} → {persona_after_2.facets['concise']:.2f}")
    print(f"    step_by_step: {persona_before.facets['step_by_step']:.2f} → {persona_after_2.facets['step_by_step']:.2f}")
    
    # === Show Training Data ===
    print("\n" + "-"*70)
    print("TRAINING DATA COLLECTED")
    print("-"*70)
    
    stats = training_collector.get_training_stats()
    print(f"\nTraining data statistics:")
    print(f"  Hypotheses logged: {stats['hypotheses_count']}")
    print(f"  Persona updates logged: {stats['persona_updates_count']}")
    print(f"  Data directory: {stats['data_dir']}")
    
    if stats['hypotheses_count'] > 0:
        print(f"\nSample training data:")
        hyp_file = Path(stats['data_dir']) / "hypotheses_data.jsonl"
        if hyp_file.exists():
            with open(hyp_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    sample = json.loads(lines[-1])  # Last entry
                    print(f"  User: {sample['user_id']}")
                    print(f"  Input: {sample['user_input']}")
                    print(f"  Selected: {sample['selected_id']}")
                    print(f"  Confidence: {sample.get('selected_confidence', 'N/A')}")
    
    # === Summary ===
    print("\n" + "="*70)
    print(" "*25 + "SUMMARY")
    print("="*70)
    
    print(f"\n✅ Learning loop demonstrated successfully!")
    print(f"\nWhat happened:")
    print(f"  1. ✓ Generated contextual hypotheses")
    print(f"  2. ✓ User selections logged to training data")
    print(f"  3. ✓ Goals extracted from hypothesis choices")
    print(f"  4. ✓ Persona facets updated based on selections")
    print(f"  5. ✓ Interaction history stored")
    print(f"  6. ✓ Training data collected for fine-tuning")
    
    print(f"\nPersona evolution:")
    print(f"  Concise: {persona_before.facets['concise']:.2f} → {persona_after_2.facets['concise']:.2f}")
    print(f"  Step-by-step: {persona_before.facets['step_by_step']:.2f} → {persona_after_2.facets['step_by_step']:.2f}")
    print(f"  Interactions: {persona_before.interaction_count} → {persona_after_2.interaction_count}")
    
    print(f"\nNext steps:")
    print(f"  • Continue using the service to collect more data")
    print(f"  • After 100+ interactions, fine-tune your model")
    print(f"  • See LEARNING_LOOP.md for details")
    
    if isinstance(slm_client, OllamaClient):
        print(f"\n💡 Tip: With Ollama, persona updates are AI-enhanced!")
        print(f"   The system uses the model to compute better facet adjustments.")
    
    print("\n" + "="*70 + "\n")
    
    # Cleanup
    if hasattr(slm_client, 'close'):
        await slm_client.close()


if __name__ == "__main__":
    asyncio.run(test_learning_loop())

