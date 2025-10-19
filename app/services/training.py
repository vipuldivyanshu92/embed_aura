"""Training data collection and model fine-tuning service."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from app.config import get_settings
from app.models import Hypothesis, MemoryItem

logger = structlog.get_logger()


class TrainingDataCollector:
    """
    Collects training data from user interactions for model fine-tuning.
    
    This service tracks:
    - Hypothesis selections and rejections
    - User feedback on responses
    - Successful vs unsuccessful interactions
    - Persona evolution over time
    """

    def __init__(self) -> None:
        """Initialize training data collector."""
        self.settings = get_settings()
        self.data_dir = Path(self.settings.training_data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate files for different data types
        self.hypotheses_file = self.data_dir / "hypotheses_data.jsonl"
        self.persona_file = self.data_dir / "persona_updates.jsonl"
        self.ranking_file = self.data_dir / "ranking_data.jsonl"
        self.feedback_file = self.data_dir / "user_feedback.jsonl"
    
    async def log_hypothesis_selection(
        self,
        user_id: str,
        user_input: str,
        context: dict[str, Any],
        hypotheses: list[Hypothesis],
        selected_id: str | None,
        auto_advanced: bool = False,
    ) -> None:
        """
        Log hypothesis generation and user selection.
        
        Args:
            user_id: User identifier
            user_input: Original user input
            context: Context used for generation
            hypotheses: Generated hypotheses
            selected_id: Which hypothesis was selected (or None if rejected all)
            auto_advanced: Whether it was auto-advanced
        """
        if not self.settings.enable_training_data_collection:
            return
        
        # Find selected hypothesis
        selected_hyp = None
        if selected_id:
            selected_hyp = next((h for h in hypotheses if h.id == selected_id), None)
        
        training_sample = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "user_input": user_input,
            "context": {
                "persona_facets": context.get("persona_facets", {}),
                "interaction_count": context.get("interaction_count", 0),
                "recent_goals": context.get("recent_goals", []),
                "preferences": context.get("preferences", []),
                "media_type": context.get("media_type", "text"),
            },
            "hypotheses": [
                {
                    "id": h.id,
                    "question": h.question,
                    "rationale": h.rationale,
                    "confidence": h.confidence,
                }
                for h in hypotheses
            ],
            "selected_id": selected_id,
            "selected_confidence": selected_hyp.confidence if selected_hyp else None,
            "auto_advanced": auto_advanced,
            "label": "positive" if selected_id else "negative",
        }
        
        # Only log if meets quality threshold
        if selected_hyp and selected_hyp.confidence >= self.settings.min_confidence_for_training:
            self._append_jsonl(self.hypotheses_file, training_sample)
            logger.debug(
                "training_data_logged",
                type="hypothesis",
                user_id=user_id,
                selected=bool(selected_id),
            )
    
    async def log_persona_update(
        self,
        user_id: str,
        old_facets: dict[str, float],
        new_facets: dict[str, float],
        signals: dict[str, Any],
    ) -> None:
        """
        Log persona updates for learning facet adjustment patterns.
        
        Args:
            user_id: User identifier
            old_facets: Facet values before update
            new_facets: Facet values after update
            signals: Signals that triggered the update
        """
        if not self.settings.enable_training_data_collection:
            return
        
        # Calculate facet deltas
        deltas = {
            facet: new_facets.get(facet, 0) - old_facets.get(facet, 0)
            for facet in new_facets
        }
        
        training_sample = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "old_facets": old_facets,
            "new_facets": new_facets,
            "facet_deltas": deltas,
            "signals": signals,
        }
        
        self._append_jsonl(self.persona_file, training_sample)
        logger.debug("training_data_logged", type="persona", user_id=user_id)
    
    async def log_ranking_feedback(
        self,
        user_id: str,
        query: str,
        memories: list[dict[str, Any]],
        interaction_outcome: str,
    ) -> None:
        """
        Log ranking performance for learning better scoring weights.
        
        Args:
            user_id: User identifier
            query: Query that was ranked
            memories: Ranked memories with scores
            interaction_outcome: success|partial|failure
        """
        if not self.settings.enable_training_data_collection:
            return
        
        training_sample = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "query": query,
            "memories": memories,
            "outcome": interaction_outcome,
        }
        
        self._append_jsonl(self.ranking_file, training_sample)
        logger.debug("training_data_logged", type="ranking", user_id=user_id)
    
    async def log_user_feedback(
        self,
        user_id: str,
        interaction_id: str,
        feedback_type: str,
        feedback_value: Any,
        context: dict[str, Any],
    ) -> None:
        """
        Log explicit user feedback.
        
        Args:
            user_id: User identifier
            interaction_id: ID of the interaction being rated
            feedback_type: thumbs_up|thumbs_down|rating|comment
            feedback_value: Feedback value (bool, int, or str)
            context: Additional context about the interaction
        """
        if not self.settings.enable_training_data_collection:
            return
        
        training_sample = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "interaction_id": interaction_id,
            "feedback_type": feedback_type,
            "feedback_value": feedback_value,
            "context": context,
        }
        
        self._append_jsonl(self.feedback_file, training_sample)
        logger.debug("training_data_logged", type="feedback", user_id=user_id)
    
    def _append_jsonl(self, file_path: Path, data: dict[str, Any]) -> None:
        """Append JSON line to file."""
        try:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data) + "\n")
        except Exception as e:
            logger.error("training_data_write_failed", file=str(file_path), error=str(e))
    
    def get_training_stats(self) -> dict[str, Any]:
        """
        Get statistics about collected training data.
        
        Returns:
            Dictionary with counts and metadata
        """
        stats = {
            "hypotheses_count": self._count_lines(self.hypotheses_file),
            "persona_updates_count": self._count_lines(self.persona_file),
            "ranking_samples_count": self._count_lines(self.ranking_file),
            "feedback_count": self._count_lines(self.feedback_file),
            "data_dir": str(self.data_dir),
        }
        return stats
    
    def _count_lines(self, file_path: Path) -> int:
        """Count lines in a file."""
        if not file_path.exists():
            return 0
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        except Exception:
            return 0
    
    async def export_for_fine_tuning(
        self,
        output_file: str,
        format: str = "chat",
    ) -> str:
        """
        Export collected data in format suitable for fine-tuning.
        
        Args:
            output_file: Path to output file
            format: Export format (chat|completion)
            
        Returns:
            Path to exported file
        """
        output_path = self.data_dir / output_file
        
        # Read hypothesis data
        training_samples = []
        if self.hypotheses_file.exists():
            with open(self.hypotheses_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        sample = json.loads(line)
                        # Only include positive examples (selected hypotheses)
                        if sample.get("label") == "positive" and sample.get("selected_id"):
                            training_samples.append(sample)
                    except json.JSONDecodeError:
                        continue
        
        # Convert to fine-tuning format
        if format == "chat":
            fine_tuning_data = self._convert_to_chat_format(training_samples)
        else:
            fine_tuning_data = self._convert_to_completion_format(training_samples)
        
        # Write to output file
        with open(output_path, "w", encoding="utf-8") as f:
            for item in fine_tuning_data:
                f.write(json.dumps(item) + "\n")
        
        logger.info(
            "training_data_exported",
            output_file=str(output_path),
            samples_count=len(fine_tuning_data),
            format=format,
        )
        
        return str(output_path)
    
    def _convert_to_chat_format(
        self,
        samples: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert training samples to chat format for fine-tuning."""
        fine_tuning_data = []
        
        for sample in samples:
            # Find the selected hypothesis
            selected_id = sample["selected_id"]
            selected_hyp = next(
                (h for h in sample["hypotheses"] if h["id"] == selected_id),
                None,
            )
            
            if not selected_hyp:
                continue
            
            # Build system message with context
            system_msg = "You are an AI assistant helping to understand user intent and generate clarifying questions."
            
            # Build user message with context
            context_parts = []
            ctx = sample["context"]
            if ctx.get("recent_goals"):
                context_parts.append(f"Recent Goals: {', '.join(ctx['recent_goals'][:2])}")
            if ctx.get("preferences"):
                context_parts.append(f"Preferences: {', '.join(ctx['preferences'][:3])}")
            
            context_str = "\n".join(context_parts) if context_parts else ""
            
            user_msg = f"""User Input: {sample['user_input']}

Context:
{context_str}

Generate a clarifying question about the user's intent."""
            
            # Assistant response should be the selected hypothesis
            assistant_msg = f"""Q: {selected_hyp['question']}
Confidence: {selected_hyp['confidence']}
Rationale: {selected_hyp['rationale']}"""
            
            fine_tuning_data.append({
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ]
            })
        
        return fine_tuning_data
    
    def _convert_to_completion_format(
        self,
        samples: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert training samples to completion format."""
        fine_tuning_data = []
        
        for sample in samples:
            selected_id = sample["selected_id"]
            selected_hyp = next(
                (h for h in sample["hypotheses"] if h["id"] == selected_id),
                None,
            )
            
            if not selected_hyp:
                continue
            
            # Build prompt
            prompt = f"User: {sample['user_input']}\nAssistant:"
            
            # Completion is the selected hypothesis
            completion = f" {selected_hyp['question']}"
            
            fine_tuning_data.append({
                "prompt": prompt,
                "completion": completion,
            })
        
        return fine_tuning_data


class ModelTrainer:
    """
    Service for fine-tuning models using collected training data.
    
    This can work with:
    - Local vLLM models via fine-tuning scripts
    - Reinforcement learning from human feedback (RLHF)
    - Direct supervision on collected examples
    """

    def __init__(self, data_collector: TrainingDataCollector) -> None:
        """
        Initialize model trainer.
        
        Args:
            data_collector: Training data collector instance
        """
        self.data_collector = data_collector
        self.settings = get_settings()
    
    async def prepare_training_dataset(
        self,
        output_dir: str = "./data/training/datasets",
    ) -> dict[str, str]:
        """
        Prepare training datasets from collected data.
        
        Args:
            output_dir: Directory to save prepared datasets
            
        Returns:
            Dictionary mapping dataset names to file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        datasets = {}
        
        # Export hypothesis generation dataset
        hyp_file = await self.data_collector.export_for_fine_tuning(
            output_file=str(output_path / "hypotheses_chat.jsonl"),
            format="chat",
        )
        datasets["hypotheses"] = hyp_file
        
        logger.info(
            "training_datasets_prepared",
            datasets=list(datasets.keys()),
            output_dir=output_dir,
        )
        
        return datasets
    
    def generate_training_script(
        self,
        dataset_path: str,
        model_name: str,
        output_model_path: str,
    ) -> str:
        """
        Generate a training script for fine-tuning with vLLM/unsloth.
        
        Args:
            dataset_path: Path to training dataset
            model_name: Base model name to fine-tune
            output_model_path: Where to save fine-tuned model
            
        Returns:
            Path to generated training script
        """
        script_path = Path(self.settings.training_data_dir) / "train_model.py"
        
        script_content = f"""#!/usr/bin/env python3
\"\"\"
Auto-generated training script for fine-tuning {model_name}.

This script uses unsloth for efficient fine-tuning of the model
on collected training data.
\"\"\"

import json
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

# Configuration
MODEL_NAME = "{model_name}"
DATASET_PATH = "{dataset_path}"
OUTPUT_PATH = "{output_model_path}"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

# Training hyperparameters
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4

def load_dataset(path: str) -> Dataset:
    \"\"\"Load training dataset from JSONL file.\"\"\"
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)

def format_prompt(example):
    \"\"\"Format example for training.\"\"\"
    messages = example['messages']
    text = ""
    for msg in messages:
        role = msg['role']
        content = msg['content']
        if role == 'system':
            text += f"<|system|>\\n{{content}}\\n"
        elif role == 'user':
            text += f"<|user|>\\n{{content}}\\n"
        elif role == 'assistant':
            text += f"<|assistant|>\\n{{content}}<|end|>\\n"
    return {{"text": text}}

def main():
    print(f"Loading base model: {{MODEL_NAME}}")
    
    # Load model with unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )
    
    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
    )
    
    print(f"Loading dataset: {{DATASET_PATH}}")
    dataset = load_dataset(DATASET_PATH)
    dataset = dataset.map(format_prompt)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_PATH,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=5,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        save_strategy="epoch",
    )
    
    # Train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save model
    print(f"Saving model to {{OUTPUT_PATH}}")
    model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    
    print("Training complete!")

if __name__ == "__main__":
    main()
"""
        
        # Write script
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        logger.info("training_script_generated", script_path=str(script_path))
        
        return str(script_path)

