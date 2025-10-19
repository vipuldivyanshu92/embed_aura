🚀 Single‑Shot Backend Generator Prompt

ROLE: You are a principal Python backend engineer.
OBJECTIVE: Generate a production‑ready backend-only service that minimizes user typing by:

Phase A – Hypothesis: From the user’s first short input, propose 2–3 concise next‑question hypotheses that the user might intend.

Phase B – Enrichment: After a user selects one hypothesis, enrich the small input into one long, structured prompt (with memory, persona, preferences, constraints, artifacts) ready to send to a powerful LLM for final task execution.

Key design constraints (must implement):

Language: Python 3.11

Framework: FastAPI

Data modeling: Pydantic v2

Logging: structlog

HTTP client: httpx

Embeddings/math: numpy (you may use scikit-learn for cosine similarity if desired, but prefer a tiny local implementation)

Testing: pytest

Config: python‑dotenv + environment variables

Style/tooling: ruff, black, mypy (strict mode where feasible)

Packaging: pyproject.toml (PEP 621); no setup.py

Containerization: Dockerfile (+ optional docker‑compose if useful)

Memory layer (pluggable, mandatory):

Provide a MemoryProvider interface and two adapters:

Mem0Provider (adapter stub wired for real SDK later)

SupermemoryProvider (adapter stub wired for real SDK later)

Provide a fully working local fallback: LocalMemoryProvider (in‑memory, persisted to JSON on shutdown for dev).

Choose provider via env var MEMORY_BACKEND=mem0|supermemory|local (default local).

Persona / “UserMind”:

Maintain a Persona object (vector + “facets” sliders like concise, formal, code_first) per user.

Update persona with tiny signals: chosen hypothesis, success markers, simple deltas from past accepted outputs.

Store persona in the memory layer or local store (for local provider, JSON file).

Provide a small linear probe (heuristic function) mapping vector shifts → facet updates.

Core flows (must implement end‑to‑end):

Phase A – Hypothesizer

Endpoint: POST /v1/hypothesize

Input: { user_id: str, input_text: str }

Steps:

Fetch top‑k short memories (GOAL, PREFERENCE, STYLE) + Persona facets.

Use a Small Language Model client (SLMClient) to produce 2–3 hypotheses with {id, question, rationale, confidence}.

The SLMClient should be easily swappable (interface + a local “mock” rule‑based model so the service runs without external calls).

Return the hypotheses sorted by confidence.

Log a tiny event that hypotheses were shown (for learning loop).

Response shape (strict):

{
  "hypotheses": [
    { "id": "h1", "question": "…", "rationale": "…", "confidence": 0.82 },
    { "id": "h2", "question": "…", "rationale": "…", "confidence": 0.64 }
  ]
}

Phase B – Execute (Enrichment)

Endpoint: POST /v1/execute

Input: { user_id: str, input_text: str, hypothesis_id: str }

Steps:

Retrieve broader memories (PROFILE, PREFERENCE, STYLE, GOAL, HISTORY, ARTIFACT).

Rank + deduplicate memories by score = 0.6*cosine + 0.25*recency + 0.15*confidence.

Context Budgeter: allocate a configurable token budget across sections:

goal_summary 10%, preferences_style 10%, critical_artifacts 25%, recent_history 20%, constraints 15%, task_specific_retrieval 15%, safety/system 5%.

If a section exceeds its budget, compress with the SLMClient summarizer (fall back to a local heuristic summarizer if SLM unavailable).

Prompt Builder: produce a single self‑contained prompt using this template:

SYSTEM
You are executing a single task for {user_name}. Follow constraints strictly.
Primary goal:
{goal_summary}

User's immediate request (verbatim):
{user_input}

Disambiguated intent:
{selected_hypothesis.question}

Important preferences & style:
{preferences_style}

Hard constraints (do not violate):
{constraints}

Essential context & artifacts (newest first):
{ranked_artifacts}

Recent outcomes (distilled):
{recent_history}

Output contract:
{output_contract}

Safety pass: redact obvious secrets/PII patterns (emails, API keys) from memory before insertion into the final prompt; retain placeholders.

Learning loop: update Persona vector (recency‑weighted centroid) and facet sliders; write any new stable preferences as PREFERENCE memories with TTL as appropriate.

Response (strict):

{
  "enriched_prompt": "…long prompt…",
  "tokens_estimate": 1234,
  "context_breakdown": {
    "goal_summary": 220,
    "preferences_style": 180,
    "critical_artifacts": 420,
    "recent_history": 260,
    "constraints": 140,
    "task_specific_retrieval": 140,
    "safety_system": 60
  }
}

API non‑goals: This service does not call a big LLM for the final task; it returns the final prompt to the caller which can then submit it elsewhere.

Functional Requirements (implement exactly)

Endpoints

    POST /v1/hypothesize (Phase A)

    POST /v1/execute (Phase B)

    GET /v1/healthz (readiness)

    GET /v1/memory/export?user_id=… (debug/download of user’s memories/persona)

    DELETE /v1/memory?user_id=…&mtype=… (delete by type; local provider only must support)

Memory Types: PROFILE, PREFERENCE, GOAL, HISTORY, ARTIFACT, STYLE. Provide MemoryItem model with confidence, tags, expires_at.

Persona: vector: list[float], facets: dict[str, float], last_updated. Start with 384‑dim vector; use a lightweight embedder stub (e.g., hashing → pseudo‑vector) to remain runnable offline.

Ranking: cosine similarity; recency decay (exponential).

Context Budgeter: precise percentages (configurable via env), overflow summarization, dedup, conflict resolution (keep higher confidence).

SLMClient: interface + 2 impls:

LocalSLM (deterministic heuristics; summary via extractive reduction).

HttpSLM (placeholder for hosted SLM; accept SLM_BASE_URL, SLM_API_KEY).

Safety/PII: simple regex redaction for emails, phone numbers, API key‑like tokens.

Observability: structlog logging (JSON), request IDs, latency & token estimates per request; counters for hypothesis CTR (when hypothesis_id later appears in /execute).

Config: .env with defaults (see below). Config validation at startup; fail fast on invalid values.

Testing: pytest unit tests for ranking, budgeter, persona update, and both endpoints (happy & edge cases).

Docs: auto OpenAPI via FastAPI; README with quickstart & curl examples.

Token estimation: simple heuristic (e.g., 4 chars/token); put in a utility.


Output Format (very important)

Return your answer as a complete repository using one code block per file in this exact order:

README.md

pyproject.toml

.env.example

Dockerfile

Makefile

app/__init__.py

app/main.py

app/config.py

app/models.py

app/clients/slm/__init__.py

app/clients/slm/base.py

app/clients/slm/local.py

app/clients/slm/http.py

app/memory/__init__.py

app/memory/base.py

app/memory/local.py

app/memory/mem0.py # adapter stub with clear TODOs

app/memory/supermemory.py # adapter stub with clear TODOs

app/services/persona.py

app/services/ranking.py

app/services/context_budgeter.py

app/services/prompt_builder.py

app/services/hypothesizer.py

app/services/safety.py

app/services/telemetry.py

app/utils/tokens.py

tests/test_ranking.py

tests/test_budgeter.py

tests/test_persona.py

tests/test_endpoints.py

Each file must be wrapped in a code fence with the correct language tag (e.g., python, toml, ```markdown, or none for .env). Ensure imports resolve across files.

Repository Requirements & Defaults

pyproject.toml includes: fastapi, uvicorn[standard], pydantic>=2, structlog, httpx, numpy, python-dotenv, pytest, mypy, black, ruff.

.env.example keys (with sensible defaults):

APP_ENV=dev

MEMORY_BACKEND=local

TOKEN_BUDGET=6000

BUDGET_WEIGHTS=goal:0.10,pref:0.10,artifacts:0.25,history:0.20,constraints:0.15,task:0.15,safety:0.05

AUTO_ADVANCE_CONFIDENCE=0.86

SLM_IMPL=local # local|http

SLM_BASE_URL=

SLM_API_KEY=

EMBED_DIMS=384

LOG_LEVEL=INFO

Dockerfile uses python:3.11‑slim, installs deps, runs uvicorn app.main:app.

Makefile targets: install, lint, test, run.

Behavioral Details & Edge Cases

If MEMORY_BACKEND is misconfigured, fallback to LocalMemoryProvider and log a warning.

If SLM is unreachable (when SLM_IMPL=http), fallback to LocalSLM and continue.

Hypotheses count: 2–3 max; never return more than 3. If confidence of Top‑1 ≥ AUTO_ADVANCE_CONFIDENCE, include a field "auto_advance": true in /v1/hypothesize response (optional) but do not skip returning the list.

De‑dup memories by near‑duplicate threshold (cosine ≥ 0.95).

Conflict resolution: if two memories conflict (same tag/topic), keep the one with higher confidence or newer updated_at; log the decision.

Persona update: recency‑weighted moving average; facet sliders clamped [0,1].

Token estimation: use in both endpoints; include totals in logs.

PII redaction: replace with <REDACTED:EMAIL> etc.; ensure redaction happens before budget allocation to keep counts accurate.


Verification & Tests

Create tests that:

Assert ranking score ordering given crafted inputs (cosine, recency, confidence).

Assert budgeter truncates sections to configured shares and calls summarizer on overflow (use LocalSLM summarizer).

Assert persona update changes facet sliders predictably when provided with signals.

Start the app (TestClient) and:

POST /v1/hypothesize returns 2–3 items sorted by confidence.

POST /v1/execute returns a non‑empty enriched_prompt and a plausible context_breakdown whose parts sum to ~TOKEN_BUDGET (±20%).

Linting & type checks pass (ruff, mypy).

README Content (must include)

Overview of Phase A/B.

How to run locally (uvicorn app.main:app --reload), with .env setup.

Example curl for /v1/hypothesize and /v1/execute.

How to switch memory backend via MEMORY_BACKEND.

How to swap SLM implementation.

Notes on data privacy + how to export/delete local memory.

Ground Rules

Do not ask me questions. Make reasonable defaults and document them in comments.

Provide fully runnable code with no TODOs except in the external provider adapters (mem0.py, supermemory.py), where you may mark clear TODO blocks.

Ensure imports are correct and the app boots successfully with MEMORY_BACKEND=local and SLM_IMPL=local.

Keep code clean, typed, and well‑documented with concise docstrings.

Deliver now the full repository as per “Output Format”.