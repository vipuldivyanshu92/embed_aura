"""Configuration management using pydantic-settings."""

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_env: str = Field(default="dev", description="Application environment")
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")

    # Memory Backend
    memory_backend: Literal["local", "mem0", "supermemory"] = Field(
        default="local", description="Memory backend to use"
    )
    mem0_api_key: str = Field(default="", description="Mem0 API key")
    mem0_base_url: str = Field(default="https://api.mem0.ai", description="Mem0 base URL")
    supermemory_api_key: str = Field(default="", description="Supermemory API key")
    supermemory_base_url: str = Field(default="", description="Supermemory base URL")

    # Token Budget
    token_budget: int = Field(default=6000, description="Maximum token budget", ge=100)
    budget_weights: str = Field(
        default="goal:0.10,pref:0.10,artifacts:0.25,history:0.20,constraints:0.15,task:0.15,safety:0.05",
        description="Budget weight allocation",
    )

    # Auto-advance
    auto_advance_confidence: float = Field(
        default=0.86, description="Confidence threshold for auto-advance", ge=0.0, le=1.0
    )

    # SLM Configuration
    slm_impl: Literal["openai", "local", "http", "vllm"] = Field(
        default="openai", description="SLM implementation"
    )
    slm_base_url: str = Field(default="", description="SLM base URL for HTTP implementation")
    slm_api_key: str = Field(default="", description="SLM API key for HTTP implementation")

    # OpenAI configuration
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(
        default="gpt-4o-mini", description="OpenAI model for text reasoning"
    )
    openai_vision_model: str = Field(
        default="gpt-4o-mini", description="OpenAI model for vision tasks"
    )
    openai_base_url: str = Field(default="", description="Optional OpenAI compatible base URL")
    openai_timeout: float = Field(default=60.0, description="OpenAI request timeout in seconds")
    
    # vLLM Configuration
    vllm_base_url: str = Field(
        default="http://localhost:8000/v1", description="vLLM server base URL"
    )
    vllm_model_name: str = Field(
        default="unsloth/Qwen2.5-3B-Instruct", description="vLLM model name"
    )
    vllm_timeout: float = Field(default=30.0, description="vLLM request timeout in seconds")
    
    # Training/Fine-tuning Configuration
    enable_training_data_collection: bool = Field(
        default=False, description="Enable collection of training data from interactions"
    )
    training_data_dir: str = Field(
        default="./data/training", description="Directory for training data storage"
    )
    min_confidence_for_training: float = Field(
        default=0.7, description="Minimum confidence to include in training data", ge=0.0, le=1.0
    )

    # Embeddings
    embed_dims: int = Field(default=384, description="Embedding dimensions", ge=1)

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Logging format")

    # Data Persistence
    data_dir: str = Field(default="./data", description="Data directory for local storage")

    # Memory Configuration
    memory_top_k: int = Field(default=10, description="Top-K memories to retrieve", ge=1)
    memory_dedup_threshold: float = Field(
        default=0.95, description="Cosine similarity threshold for dedup", ge=0.0, le=1.0
    )

    # Ranking Weights
    ranking_cosine_weight: float = Field(
        default=0.6, description="Cosine similarity weight", ge=0.0, le=1.0
    )
    ranking_recency_weight: float = Field(
        default=0.25, description="Recency weight", ge=0.0, le=1.0
    )
    ranking_confidence_weight: float = Field(
        default=0.15, description="Confidence weight", ge=0.0, le=1.0
    )

    # Recency Decay
    recency_decay_days: float = Field(
        default=30.0, description="Recency decay period in days", gt=0.0
    )

    # Persona Configuration
    persona_update_rate: float = Field(
        default=0.1, description="Persona update learning rate", ge=0.0, le=1.0
    )
    persona_facet_clamp_min: float = Field(default=0.0, description="Min facet value")
    persona_facet_clamp_max: float = Field(default=1.0, description="Max facet value")

    @field_validator("budget_weights")
    @classmethod
    def validate_budget_weights(cls, v: str) -> str:
        """Validate budget weights format and sum."""
        try:
            weights = {}
            for pair in v.split(","):
                key, val = pair.split(":")
                weights[key.strip()] = float(val.strip())

            total = sum(weights.values())
            if not (0.99 <= total <= 1.01):  # Allow small floating point errors
                raise ValueError(f"Budget weights must sum to 1.0, got {total}")

            required_keys = {
                "goal",
                "pref",
                "artifacts",
                "history",
                "constraints",
                "task",
                "safety",
            }
            missing = required_keys - set(weights.keys())
            if missing:
                raise ValueError(f"Missing budget weight keys: {missing}")

        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid budget_weights format: {e}") from e

        return v

    @field_validator("ranking_cosine_weight", "ranking_recency_weight", "ranking_confidence_weight")
    @classmethod
    def validate_ranking_weights(cls, v: float, info: dict) -> float:
        """Ensure ranking weights are valid."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{info.field_name} must be between 0.0 and 1.0")
        return v

    def get_budget_weights_dict(self) -> dict[str, float]:
        """Parse budget weights into a dictionary."""
        weights = {}
        for pair in self.budget_weights.split(","):
            key, val = pair.split(":")
            weights[key.strip()] = float(val.strip())
        return weights

    def ensure_data_dir(self) -> None:
        """Ensure data directory exists."""
        os.makedirs(self.data_dir, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    # Ensure data directory exists for local storage
    if settings.memory_backend == "local":
        settings.ensure_data_dir()
    return settings
