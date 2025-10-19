"""Pydantic models for API requests, responses, and data structures."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class MemoryType(str, Enum):
    """Enumeration of memory types."""

    PROFILE = "PROFILE"
    PREFERENCE = "PREFERENCE"
    GOAL = "GOAL"
    HISTORY = "HISTORY"
    ARTIFACT = "ARTIFACT"
    STYLE = "STYLE"


class MediaType(str, Enum):
    """Enumeration of supported media types for multi-modal input."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class MemoryItem(BaseModel):
    """A single memory item stored in the memory layer (multi-modal support)."""

    id: str = Field(..., description="Unique memory identifier")
    user_id: str = Field(..., description="User identifier")
    mtype: MemoryType = Field(..., description="Memory type")
    content: str | None = Field(None, description="Text content")
    media_type: MediaType = Field(default=MediaType.TEXT, description="Type of media content")
    media_url: str | None = Field(None, description="URL to media file (image/audio/video)")
    media_description: str | None = Field(
        None, description="Text description of media content"
    )
    embedding: list[float] = Field(..., description="Embedding vector")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    tags: list[str] = Field(default_factory=list, description="Associated tags")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )
    expires_at: datetime | None = Field(None, description="Optional expiration timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("embedding")
    @classmethod
    def validate_embedding_dims(cls, v: list[float]) -> list[float]:
        """Ensure embedding has correct dimensions."""
        if not v:
            raise ValueError("Embedding cannot be empty")
        return v


class Persona(BaseModel):
    """User persona with preferences and learned behavior patterns."""

    user_id: str = Field(..., description="User identifier")
    vector: list[float] = Field(..., description="Persona embedding vector")
    facets: dict[str, float] = Field(
        default_factory=dict,
        description="Preference facets (concise, formal, code_first, etc.)",
    )
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
    interaction_count: int = Field(default=0, description="Total interactions")

    @field_validator("facets")
    @classmethod
    def validate_facets(cls, v: dict[str, float]) -> dict[str, float]:
        """Ensure facet values are clamped [0, 1]."""
        for key, val in v.items():
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"Facet {key} must be between 0.0 and 1.0, got {val}")
        return v


class Hypothesis(BaseModel):
    """A hypothesis about user intent."""

    id: str = Field(..., description="Hypothesis identifier")
    question: str = Field(..., description="Clarifying question")
    rationale: str = Field(..., description="Why this hypothesis was generated")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class HypothesizeRequest(BaseModel):
    """Request to generate hypotheses (Phase A) - Multi-modal support."""

    user_id: str = Field(..., description="User identifier")
    input_text: str | None = Field(None, description="User's text input")
    media_type: MediaType = Field(default=MediaType.TEXT, description="Type of input media")
    media_url: str | None = Field(None, description="URL to media file (image/audio/video)")
    media_base64: str | None = Field(
        None, description="Base64 encoded media data (alternative to URL)"
    )

    @model_validator(mode="after")
    def validate_input(self) -> "HypothesizeRequest":
        """Ensure at least one input is provided."""
        if not self.input_text and not self.media_url and not self.media_base64:
            raise ValueError("At least one of input_text, media_url, or media_base64 must be provided")
        return self


class HypothesizeResponse(BaseModel):
    """Response containing hypotheses (Phase A)."""

    hypotheses: list[Hypothesis] = Field(..., description="Generated hypotheses")
    auto_advance: bool = Field(
        default=False, description="Whether top hypothesis meets auto-advance threshold"
    )


class ExecuteRequest(BaseModel):
    """Request to execute enrichment (Phase B) - Multi-modal support."""

    user_id: str = Field(..., description="User identifier")
    input_text: str | None = Field(None, description="User's text input")
    media_type: MediaType = Field(default=MediaType.TEXT, description="Type of input media")
    media_url: str | None = Field(None, description="URL to media file (image/audio/video)")
    media_base64: str | None = Field(
        None, description="Base64 encoded media data (alternative to URL)"
    )
    hypothesis_id: str = Field(..., description="Selected hypothesis ID")

    @model_validator(mode="after")
    def validate_input(self) -> "ExecuteRequest":
        """Ensure at least one input is provided."""
        if not self.input_text and not self.media_url and not self.media_base64:
            raise ValueError("At least one of input_text, media_url, or media_base64 must be provided")
        return self


class ExecuteResponse(BaseModel):
    """Response containing enriched prompt (Phase B)."""

    enriched_prompt: str = Field(..., description="Complete enriched prompt")
    tokens_estimate: int = Field(..., ge=0, description="Estimated token count")
    context_breakdown: dict[str, int] = Field(..., description="Token breakdown by section")


class RankedMemory(BaseModel):
    """Memory item with ranking score."""

    memory: MemoryItem
    score: float = Field(..., description="Ranking score")
    cosine_score: float = Field(..., description="Cosine similarity component")
    recency_score: float = Field(..., description="Recency component")
    confidence_score: float = Field(..., description="Confidence component")


class ContextSection(BaseModel):
    """A section of context with allocated tokens."""

    name: str = Field(..., description="Section name")
    content: str = Field(..., description="Section content")
    tokens: int = Field(..., ge=0, description="Token count")
    allocated: int = Field(..., ge=0, description="Allocated token budget")
    summarized: bool = Field(default=False, description="Whether content was summarized")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    memory_backend: str = Field(..., description="Active memory backend")
    slm_impl: str = Field(..., description="Active SLM implementation")


class MemoryExport(BaseModel):
    """Exported memory data for a user."""

    user_id: str = Field(..., description="User identifier")
    memories: list[MemoryItem] = Field(..., description="All user memories")
    persona: Persona | None = Field(None, description="User persona if available")
    exported_at: datetime = Field(default_factory=datetime.utcnow, description="Export timestamp")
