"""FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime

import structlog
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from app import __version__
from app.clients.slm import HttpSLM, LocalSLM, OpenAISLM, SLMClient
from app.clients.slm.vllm import VLLMClient
from app.config import get_settings
from app.memory import LocalMemoryProvider, Mem0Provider, MemoryProvider, SupermemoryProvider
from app.models import (
    ExecuteRequest,
    ExecuteResponse,
    HealthResponse,
    HypothesizeRequest,
    HypothesizeResponse,
    MemoryExport,
    MemoryItem,
    MemoryType,
    UsedMemory,
)
from app.services.context_budgeter import ContextBudgeter
from app.services.hypothesizer import HypothesizerService
from app.services.persona import PersonaService
from app.services.prompt_builder import PromptBuilder
from app.services.ranking import RankingService
from app.services.safety import SafetyService
from app.services.telemetry import TelemetryService
from app.services.training import TrainingDataCollector
from app.utils.embeddings import generate_embedding
from app.utils.tokens import estimate_tokens

# Configure structured logging
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=False,
)

logger = structlog.get_logger()

# Global services (will be initialized in lifespan or on first access)
memory_provider: MemoryProvider | None = None
slm_client: SLMClient | None = None
persona_service: PersonaService | None = None
ranking_service: RankingService | None = None
safety_service: SafetyService | None = None
context_budgeter: ContextBudgeter | None = None
prompt_builder: PromptBuilder | None = None
hypothesizer_service: HypothesizerService | None = None
telemetry_service: TelemetryService | None = None
training_collector: TrainingDataCollector | None = None


def _initialize_services() -> None:
    """Initialize all services."""
    global memory_provider, slm_client, persona_service, ranking_service
    global safety_service, context_budgeter, prompt_builder, hypothesizer_service
    global telemetry_service, training_collector

    settings = get_settings()

    logger.info(
        "application_starting",
        version=__version__,
        memory_backend=settings.memory_backend,
        slm_impl=settings.slm_impl,
    )

    # Initialize memory provider
    if settings.memory_backend == "mem0":
        if not settings.mem0_api_key:
            logger.warning(
                "mem0_config_missing",
                message="Mem0 backend selected but API key missing, falling back to local",
            )
            memory_provider = LocalMemoryProvider()
        else:
            memory_provider = Mem0Provider(settings.mem0_api_key, settings.mem0_base_url)
    elif settings.memory_backend == "supermemory":
        if not settings.supermemory_api_key:
            logger.warning(
                "supermemory_config_missing",
                message="Supermemory backend selected but API key missing, falling back to local",
            )
            memory_provider = LocalMemoryProvider()
        else:
            memory_provider = SupermemoryProvider(
                settings.supermemory_api_key, settings.supermemory_base_url
            )
    else:
        memory_provider = LocalMemoryProvider()

    # Initialize SLM client
    if settings.slm_impl == "openai":
        if not settings.openai_api_key:
            logger.warning(
                "openai_config_missing",
                message="OpenAI selected but API key missing, falling back to local",
            )
            slm_client = LocalSLM()
        else:
            slm_client = OpenAISLM(
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                vision_model=settings.openai_vision_model,
                base_url=settings.openai_base_url or None,
                timeout=settings.openai_timeout,
            )
            logger.info("openai_slm_initialized", model=settings.openai_model)

    elif settings.slm_impl == "vllm":
        slm_client = VLLMClient(
            base_url=settings.vllm_base_url,
            model_name=settings.vllm_model_name,
            timeout=settings.vllm_timeout,
        )
        logger.info("vllm_client_initialized", base_url=settings.vllm_base_url)
    elif settings.slm_impl == "http":
        if not settings.slm_base_url:
            logger.warning(
                "slm_http_config_missing",
                message="HTTP SLM selected but base URL missing, falling back to local",
            )
            slm_client = LocalSLM()
        else:
            slm_client = HttpSLM(settings.slm_base_url, settings.slm_api_key)
    else:
        slm_client = LocalSLM()

    # Initialize services
    training_collector = TrainingDataCollector()
    persona_service = PersonaService(memory_provider, slm_client if settings.slm_impl == "vllm" else None)
    ranking_service = RankingService(slm_client if settings.slm_impl == "vllm" else None)
    safety_service = SafetyService()
    context_budgeter = ContextBudgeter(slm_client)
    prompt_builder = PromptBuilder(context_budgeter, safety_service)
    hypothesizer_service = HypothesizerService(slm_client, memory_provider, persona_service)
    telemetry_service = TelemetryService()

    logger.info("application_ready", training_enabled=settings.enable_training_data_collection)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager."""
    _initialize_services()
    yield

    # Cleanup
    if slm_client and isinstance(slm_client, (HttpSLM, VLLMClient)):
        await slm_client.close()

    logger.info("application_shutdown")


# Create FastAPI app
app = FastAPI(
    title="Embed Service",
    description="Single-Shot Backend Generator - Hypothesis and Prompt Enrichment",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/v1/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    """Health check endpoint."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version=__version__,
        memory_backend=settings.memory_backend,
        slm_impl=settings.slm_impl,
    )


@app.post("/v1/hypothesize", response_model=HypothesizeResponse)
async def hypothesize(request: HypothesizeRequest) -> HypothesizeResponse:
    """
    Phase A: Generate hypotheses from user input.

    Returns 2-3 hypotheses sorted by confidence.
    """
    if telemetry_service is None or hypothesizer_service is None:
        _initialize_services()

    assert telemetry_service is not None
    assert hypothesizer_service is not None

    request_id = telemetry_service.generate_request_id()

    async with telemetry_service.track_request(
        request_id,
        "hypothesize",
        user_id=request.user_id,
        input_length=len(request.input_text or ""),
        media_type=request.media_type.value,
    ):
        # Generate hypotheses (multi-modal)
        hypotheses, auto_advance = await hypothesizer_service.generate_hypotheses(
            user_id=request.user_id,
            input_text=request.input_text,
            media_type=request.media_type,
            media_url=request.media_url,
            media_base64=request.media_base64,
            count=3,
        )

        # Track for CTR
        telemetry_service.track_hypothesis_shown(request.user_id, [h.id for h in hypotheses])

        logger.info(
            "hypotheses_returned",
            request_id=request_id,
            user_id=request.user_id,
            count=len(hypotheses),
            auto_advance=auto_advance,
        )

        return HypothesizeResponse(
            hypotheses=hypotheses,
            auto_advance=auto_advance,
        )


@app.post("/v1/execute", response_model=ExecuteResponse)
async def execute(request: ExecuteRequest) -> ExecuteResponse:
    """
    Phase B: Execute enrichment to generate comprehensive prompt.
    """
    if any(
        s is None
        for s in [
            telemetry_service,
            memory_provider,
            ranking_service,
            persona_service,
            hypothesizer_service,
            prompt_builder,
            slm_client,
        ]
    ):
        _initialize_services()

    assert all(
        s is not None
        for s in [
            telemetry_service,
            memory_provider,
            ranking_service,
            persona_service,
            hypothesizer_service,
            prompt_builder,
            slm_client,
        ]
    )

    request_id = telemetry_service.generate_request_id()

    async with telemetry_service.track_request(
        request_id,
        "execute",
        user_id=request.user_id,
        hypothesis_id=request.hypothesis_id,
    ):
        # Track hypothesis selection
        telemetry_service.track_hypothesis_selected(request.user_id, request.hypothesis_id)

        # Generate query embedding (multi-modal)
        query_embedding = generate_embedding(
            text=request.input_text,
            media_type=request.media_type,
            media_url=request.media_url,
            media_base64=request.media_base64,
        )

        # Retrieve memories
        all_memories = await memory_provider.search_memories(
            request.user_id,
            query_embedding,
            limit=50,  # Retrieve more, then rank
        )

        # Rank memories
        ranked_memories = await ranking_service.rank_memories(
            all_memories, query_embedding, request.input_text
        )

        # Deduplicate
        ranked_memories = ranking_service.deduplicate(ranked_memories)

        # Resolve conflicts
        ranked_memories = ranking_service.resolve_conflicts(ranked_memories)

        # Get persona
        persona = await persona_service.get_or_create_persona(request.user_id)

        # Find selected hypothesis (reconstruct or use stored)
        # For simplicity, we'll reconstruct it
        hypotheses, _ = await hypothesizer_service.generate_hypotheses(
            user_id=request.user_id,
            input_text=request.input_text,
            media_type=request.media_type,
            media_url=request.media_url,
            media_base64=request.media_base64,
            count=3,
        )
        selected_hypothesis = next(
            (h for h in hypotheses if h.id == request.hypothesis_id), hypotheses[0]
        )

        # Build enriched prompt
        enriched_prompt, context_breakdown = await prompt_builder.build_prompt(
            request.user_id,
            request.input_text,
            selected_hypothesis,
            ranked_memories,
            persona,
        )

        # Generate structured answer using SLM
        try:
            answer_payload = await slm_client.generate_answer(enriched_prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "answer_generation_failed",
                error=str(exc),
                request_id=request_id,
            )
            answer_payload = {
                "answer": enriched_prompt,
                "supporting_points": [],
                "confidence": 0.0,
            }

        # Estimate tokens
        tokens_estimate = estimate_tokens(enriched_prompt)

        # Log token estimate
        telemetry_service.log_token_estimate(
            "enriched_prompt",
            tokens_estimate,
            request_id=request_id,
        )

        # Update persona with learning signal
        await persona_service.update_persona(
            request.user_id,
            {
                "selected_hypothesis_id": request.hypothesis_id,
                "embedding": query_embedding,
            },
        )

        # Store interaction in history (multi-modal aware)
        content_desc = request.input_text[:100] if request.input_text else f"[{request.media_type.value} input]"
        history_memory = MemoryItem(
            id=f"history_{request_id}",
            user_id=request.user_id,
            mtype=MemoryType.HISTORY,
            content=f"User requested: {content_desc}... | Selected: {selected_hypothesis.question[:100]}",
            media_type=request.media_type,
            media_url=request.media_url,
            media_description=request.input_text,
            embedding=query_embedding,
            confidence=selected_hypothesis.confidence,
            tags=["interaction", request.media_type.value],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        await memory_provider.store_memory(history_memory)

        logger.info(
            "prompt_enriched",
            request_id=request_id,
            user_id=request.user_id,
            tokens_estimate=tokens_estimate,
            memories_used=len(ranked_memories),
        )

        used_memories = [
            UsedMemory(
                id=ranked.memory.id,
                summary=(
                    ranked.memory.content[:120]
                    if ranked.memory.content
                    else (ranked.memory.media_description or "")
                ),
            )
            for ranked in ranked_memories[:10]
        ]

        return ExecuteResponse(
            answer=answer_payload.get("answer", ""),
            supporting_points=answer_payload.get("supporting_points", []),
            confidence=float(answer_payload.get("confidence", 0.0)),
            used_memories=used_memories,
            enriched_prompt=enriched_prompt,
            tokens_estimate=tokens_estimate,
            context_breakdown=context_breakdown,
            telemetry={"request_id": request_id},
        )


@app.get("/v1/memory/export", response_model=MemoryExport)
async def export_memory(user_id: str = Query(..., description="User ID")) -> MemoryExport:
    """
    Export user's memories and persona for debugging or backup.
    """
    if memory_provider is None or persona_service is None:
        _initialize_services()

    assert memory_provider is not None
    assert persona_service is not None

    memories = await memory_provider.get_memories(user_id, limit=1000)
    persona = await persona_service.get_or_create_persona(user_id)

    logger.info(
        "memory_exported",
        user_id=user_id,
        memory_count=len(memories),
    )

    return MemoryExport(
        user_id=user_id,
        memories=memories,
        persona=persona,
    )


@app.delete("/v1/memory")
async def delete_memory(
    user_id: str = Query(..., description="User ID"),
    mtype: str | None = Query(None, description="Memory type to delete"),
) -> JSONResponse:
    """
    Delete user memories.

    If mtype is provided, deletes only that type. Otherwise, deletes all.
    Only supported for local backend.
    """
    if memory_provider is None:
        _initialize_services()

    assert memory_provider is not None

    settings = get_settings()

    if settings.memory_backend != "local":
        raise HTTPException(
            status_code=400,
            detail="Memory deletion only supported for local backend",
        )

    memory_type = None
    if mtype:
        try:
            memory_type = MemoryType(mtype)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid memory type: {mtype}") from e

    deleted_count = await memory_provider.delete_memories(user_id, memory_type)

    logger.info(
        "memories_deleted",
        user_id=user_id,
        mtype=mtype,
        count=deleted_count,
    )

    return JSONResponse(
        content={
            "user_id": user_id,
            "deleted_count": deleted_count,
            "mtype": mtype,
        }
    )


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception) -> JSONResponse:
    """Handle general exceptions."""
    logger.error("unhandled_exception", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )
