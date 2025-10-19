"""vLLM client implementation for local model serving."""

import json
from typing import Any

import httpx
import structlog

from app.clients.slm.base import GeneratedAnswer, SLMClient
from app.clients.slm.local import LocalSLM
from app.models import Hypothesis, MediaType
from app.utils.tokens import estimate_tokens

logger = structlog.get_logger()


class VLLMClient(SLMClient):
    """
    vLLM client for local model inference using OpenAI-compatible API.
    
    This client connects to a locally running vLLM server and uses
    the model for hypothesis generation, summarization, and enrichment.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "unsloth/Qwen2.5-3B-Instruct",
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize vLLM client.
        
        Args:
            base_url: Base URL for vLLM server (OpenAI-compatible endpoint)
            model_name: Model name to use for inference
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        
    async def generate_hypotheses(
        self,
        user_input: str | None,
        context: dict[str, Any],
        count: int = 3,
        media_type: MediaType = MediaType.TEXT,
        media_url: str | None = None,
        media_base64: str | None = None,
    ) -> list[Hypothesis]:
        """
        Generate hypotheses using vLLM model.
        
        Args:
            user_input: User's text input
            context: Context dict with 'persona_facets', 'recent_goals', etc.
            count: Number of hypotheses (max 3)
            media_type: Type of media (text, image, audio, video)
            media_url: URL to media file
            media_base64: Base64 encoded media
            
        Returns:
            List of hypotheses sorted by confidence
        """
        # Build prompt for hypothesis generation
        prompt = self._build_hypothesis_prompt(
            user_input, context, count, media_type, media_url, media_base64
        )
        
        try:
            # Call vLLM using chat completions API
            response = await self._chat_completion(
                prompt=prompt,
                temperature=0.7,
                max_tokens=800,
            )
            
            # Parse response into hypotheses
            hypotheses = self._parse_hypotheses_response(response, count)
            
            logger.info(
                "vllm_hypotheses_generated",
                count=len(hypotheses),
                top_confidence=hypotheses[0].confidence if hypotheses else 0.0,
            )
            
            return hypotheses
            
        except Exception as e:
            logger.error("vllm_hypothesis_generation_failed", error=str(e))
            # Fallback to basic hypotheses
            return self._fallback_hypotheses(user_input, count)
    
    async def summarize(self, text: str, max_tokens: int) -> str:
        """
        Summarize text using vLLM model.
        
        Args:
            text: Text to summarize
            max_tokens: Target token budget
            
        Returns:
            Summarized text
        """
        current_tokens = estimate_tokens(text)
        
        # If already under budget, return as-is
        if current_tokens <= max_tokens:
            return text
        
        # Build summarization prompt
        prompt = f"""Summarize the following text concisely in approximately {max_tokens} tokens while preserving key information:

{text}

Summary:"""
        
        try:
            summary = await self._chat_completion(
                prompt=prompt,
                temperature=0.3,
                max_tokens=max_tokens,
            )
            
            logger.debug(
                "vllm_summarization_complete",
                original_tokens=current_tokens,
                summary_tokens=estimate_tokens(summary),
            )
            
            return summary.strip()
            
        except Exception as e:
            logger.error("vllm_summarization_failed", error=str(e))
            # Fallback to simple truncation
            return text[:max_tokens * 4]

    async def describe_media(
        self,
        media_type: MediaType,
        media_url: str | None = None,
        media_base64: str | None = None,
    ) -> dict[str, Any]:
        """Use the vLLM model to produce media captions and tags when possible."""

        # vLLM does not natively support image understanding. Fallback to local heuristics.
        return await LocalSLM().describe_media(media_type, media_url, media_base64)
    
    async def enrich_context(
        self,
        user_input: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Enrich context with additional insights from vLLM.
        
        This method uses the model to generate additional context,
        infer implicit goals, and suggest relevant follow-up areas.
        
        Args:
            user_input: User's input
            context: Existing context dictionary
            
        Returns:
            Enriched context dictionary
        """
        prompt = f"""Given the user input and their profile, provide additional context insights in JSON format.

User Input: {user_input}

User Profile:
- Interaction Count: {context.get('interaction_count', 0)}
- Preferences: {', '.join(context.get('preferences', [])[:3])}
- Recent Goals: {', '.join(context.get('recent_goals', [])[:2])}

Provide enrichment in this JSON format:
{{
  "implicit_goals": ["goal1", "goal2"],
  "suggested_topics": ["topic1", "topic2"],
  "inferred_expertise_level": "beginner|intermediate|advanced",
  "likely_next_steps": ["step1", "step2"]
}}

JSON Response:"""

        try:
            response = await self._chat_completion(
                prompt=prompt,
                temperature=0.6,
                max_tokens=400,
            )
            
            # Parse JSON response
            enrichment = json.loads(response)
            
            # Add enrichment to context
            enriched_context = context.copy()
            enriched_context["enrichment"] = enrichment
            
            logger.debug("vllm_context_enriched", enrichment_keys=list(enrichment.keys()))
            
            return enriched_context
            
        except Exception as e:
            logger.error("vllm_context_enrichment_failed", error=str(e))
            return context
    
    async def generate_followup_questions(
        self,
        conversation_context: str,
        persona_facets: dict[str, float],
        count: int = 3,
    ) -> list[str]:
        """
        Generate intelligent follow-up questions.
        
        Args:
            conversation_context: Recent conversation history
            persona_facets: User's persona facet values
            count: Number of questions to generate
            
        Returns:
            List of follow-up questions
        """
        # Build persona description
        persona_desc = self._describe_persona(persona_facets)
        
        prompt = f"""Based on the conversation context and user preferences, generate {count} intelligent follow-up questions that would help clarify the user's intent or provide valuable next steps.

Conversation Context:
{conversation_context}

User Preferences:
{persona_desc}

Generate {count} concise, relevant follow-up questions. Each question should be on a new line starting with "Q:".

Questions:"""

        try:
            response = await self._chat_completion(
                prompt=prompt,
                temperature=0.8,
                max_tokens=300,
            )
            
            # Parse questions from response
            questions = []
            for line in response.split("\n"):
                line = line.strip()
                if line.startswith("Q:"):
                    questions.append(line[2:].strip())
                elif line and not line.startswith("#"):
                    # Also accept lines without Q: prefix if they look like questions
                    if line.endswith("?"):
                        questions.append(line)
            
            return questions[:count]
            
        except Exception as e:
            logger.error("vllm_followup_generation_failed", error=str(e))
            return []
    
    def _build_hypothesis_prompt(
        self,
        user_input: str | None,
        context: dict[str, Any],
        count: int,
        media_type: MediaType,
        media_url: str | None,
        media_base64: str | None,
    ) -> str:
        """Build prompt for hypothesis generation."""
        persona_desc = self._describe_persona(context.get("persona_facets", {}))
        
        # Build context summary
        context_parts = []
        if context.get("recent_goals"):
            context_parts.append(f"Recent Goals: {', '.join(context['recent_goals'][:2])}")
        if context.get("preferences"):
            context_parts.append(f"Preferences: {', '.join(context['preferences'][:3])}")
        
        context_str = "\n".join(context_parts) if context_parts else "No prior context available."
        
        media_context_lines: list[str] = []
        if media_type != MediaType.TEXT:
            media_context_lines.append(f"Media Type: {media_type.value}")
        if media_url:
            media_context_lines.append(f"Media URL: {media_url}")
        elif media_base64:
            media_context_lines.append("Media provided as base64 payload")
        media_context = "\n".join(media_context_lines)
        if media_context:
            media_context = "\n" + media_context
        
        prompt = f"""You are an AI assistant helping to understand user intent. Generate {count} hypotheses about what the user wants to accomplish.

User Input: {user_input or "(no text input)"}
{media_context}

User Profile:
{persona_desc}
Interaction Count: {context.get('interaction_count', 0)}

Context:
{context_str}

Generate {count} hypotheses as clarifying questions. Each hypothesis should:
1. Be phrased as a clear question
2. Include a confidence score (0.0-1.0)
3. Include a brief rationale

Format each hypothesis as:
Q: [question]
Confidence: [0.0-1.0]
Rationale: [brief explanation]

Hypotheses:"""

        return prompt
    
    def _describe_persona(self, facets: dict[str, float]) -> str:
        """Convert persona facets to human-readable description."""
        descriptions = []
        
        if facets.get("concise", 0.5) > 0.7:
            descriptions.append("prefers concise responses")
        elif facets.get("concise", 0.5) < 0.3:
            descriptions.append("prefers detailed explanations")
        
        if facets.get("code_first", 0.5) > 0.7:
            descriptions.append("code-first approach")
        
        if facets.get("step_by_step", 0.5) > 0.7:
            descriptions.append("likes step-by-step guidance")
        
        if facets.get("formal", 0.5) > 0.7:
            descriptions.append("formal communication style")
        elif facets.get("formal", 0.5) < 0.3:
            descriptions.append("casual communication style")
        
        return ", ".join(descriptions) if descriptions else "balanced preferences"
    
    def _parse_hypotheses_response(
        self,
        response: str,
        count: int,
    ) -> list[Hypothesis]:
        """Parse LLM response into Hypothesis objects."""
        hypotheses = []
        current_hyp = {}
        
        lines = response.strip().split("\n")
        hyp_id = 1
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("Q:"):
                # Save previous hypothesis if exists
                if current_hyp.get("question"):
                    hypotheses.append(
                        Hypothesis(
                            id=f"h{hyp_id}",
                            question=current_hyp["question"],
                            rationale=current_hyp.get("rationale", "Generated by vLLM"),
                            confidence=current_hyp.get("confidence", 0.7),
                        )
                    )
                    hyp_id += 1
                
                # Start new hypothesis
                current_hyp = {"question": line[2:].strip()}
                
            elif line.startswith("Confidence:"):
                try:
                    conf_str = line.split(":", 1)[1].strip()
                    current_hyp["confidence"] = float(conf_str)
                except (ValueError, IndexError):
                    current_hyp["confidence"] = 0.7
                    
            elif line.startswith("Rationale:"):
                current_hyp["rationale"] = line.split(":", 1)[1].strip()
        
        # Add last hypothesis
        if current_hyp.get("question"):
            hypotheses.append(
                Hypothesis(
                    id=f"h{hyp_id}",
                    question=current_hyp["question"],
                    rationale=current_hyp.get("rationale", "Generated by vLLM"),
                    confidence=current_hyp.get("confidence", 0.7),
                )
            )
        
        # Sort by confidence and limit to count
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        return hypotheses[:count]
    
    def _fallback_hypotheses(
        self,
        user_input: str | None,
        count: int,
    ) -> list[Hypothesis]:
        """Generate fallback hypotheses if vLLM fails."""
        return [
            Hypothesis(
                id=f"h{i+1}",
                question=f"Do you want help with: {user_input or 'your request'}?",
                rationale="Fallback hypothesis",
                confidence=0.6 - (i * 0.1),
            )
            for i in range(count)
        ]
    
    async def _chat_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        """
        Call vLLM chat completion API.
        
        Args:
            prompt: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    async def generate_answer(
        self,
        prompt: str,
        response_format: dict[str, Any] | None = None,
    ) -> GeneratedAnswer:
        schema_prompt = prompt + "\n\nRespond in JSON with keys: answer (string), supporting_points (array of strings), confidence (0-1)."

        try:
            response = await self._chat_completion(
                prompt=schema_prompt,
                temperature=0.3,
                max_tokens=900,
            )

            data = json.loads(response.strip())
            answer = data.get("answer", prompt)
            supporting = data.get("supporting_points", [])
            confidence = float(data.get("confidence", 0.0))

            if isinstance(supporting, str):
                supporting = [supporting]

            return GeneratedAnswer(
                answer=answer,
                supporting_points=list(supporting),
                confidence=max(0.0, min(1.0, confidence)),
            )

        except Exception as e:
            logger.error("vllm_answer_generation_failed", error=str(e))
            return await LocalSLM().generate_answer(prompt, response_format)

