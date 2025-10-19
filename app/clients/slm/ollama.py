"""Ollama client implementation for local model serving on macOS.

Ollama is optimized for macOS (especially Apple Silicon) and provides:
- Easy installation and setup
- Metal acceleration for M1/M2/M3 Macs
- OpenAI-compatible API
- Large model library
- Automatic model management
"""

import json
from typing import Any

import httpx
import structlog

from app.clients.slm.base import SLMClient
from app.models import Hypothesis, MediaType
from app.utils.tokens import estimate_tokens

logger = structlog.get_logger()


class OllamaClient(SLMClient):
    """
    Ollama client for local model inference on macOS.
    
    This client connects to a locally running Ollama server and uses
    the model for hypothesis generation, summarization, and enrichment.
    
    Optimized for Apple Silicon with Metal acceleration.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "qwen2.5:3b",
        timeout: float = 60.0,
        vision_model: str = "llava",
        embed_model: str = "nomic-embed-text",
    ) -> None:
        """
        Initialize Ollama client.
        
        Args:
            base_url: Base URL for Ollama server
            model_name: Model name to use for inference (e.g., "qwen2.5:3b", "llama3.1:8b")
            timeout: Request timeout in seconds
            vision_model: Model for vision tasks (e.g., "llava", "llama3.2-vision")
            embed_model: Model for embeddings (e.g., "nomic-embed-text")
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.vision_model = vision_model
        self.embed_model = embed_model
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
        Generate hypotheses using Ollama model.
        
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
            user_input, context, count, media_type
        )
        
        try:
            # Call Ollama API
            response = await self._generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=800,
            )
            
            # Parse response into hypotheses
            hypotheses = self._parse_hypotheses_response(response, count)
            
            # If parsing failed and returned empty, use fallback
            if not hypotheses:
                logger.warning(
                    "ollama_hypothesis_parsing_returned_empty",
                    response_preview=response[:200] if response else "None",
                )
                return self._fallback_hypotheses(user_input, count)
            
            logger.info(
                "ollama_hypotheses_generated",
                count=len(hypotheses),
                top_confidence=hypotheses[0].confidence if hypotheses else 0.0,
            )
            
            return hypotheses
            
        except Exception as e:
            logger.error("ollama_hypothesis_generation_failed", error=str(e))
            # Fallback to basic hypotheses
            return self._fallback_hypotheses(user_input, count)
    
    async def summarize(self, text: str, max_tokens: int) -> str:
        """
        Summarize text using Ollama model.
        
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
            summary = await self._generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=max_tokens,
            )
            
            logger.debug(
                "ollama_summarization_complete",
                original_tokens=current_tokens,
                summary_tokens=estimate_tokens(summary),
            )
            
            return summary.strip()
            
        except Exception as e:
            logger.error("ollama_summarization_failed", error=str(e))
            # Fallback to simple truncation
            return text[:max_tokens * 4]
    
    async def enrich_context(
        self,
        user_input: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Enrich context with additional insights from Ollama.
        
        OPTIMIZED: Uses semantic search results to provide better enrichment.
        
        Args:
            user_input: User's input
            context: Existing context dictionary (with relevant memories)
            
        Returns:
            Enriched context dictionary
        """
        # Build richer context from semantic search results
        relevant_info = []
        if context.get('relevant_goals'):
            relevant_info.append(f"Relevant Goals: {', '.join(context['relevant_goals'][:2])}")
        if context.get('relevant_preferences'):
            relevant_info.append(f"Preferences: {', '.join(context['relevant_preferences'][:3])}")
        if context.get('similar_history'):
            relevant_info.append(f"Similar Past Work: {', '.join(context['similar_history'][:2])}")
        
        context_summary = "\n- ".join(relevant_info) if relevant_info else "No prior context"
        
        prompt = f"""Given the user input and their semantically matched past context, provide additional context insights in JSON format.

User Input: {user_input}

User Profile:
- Total Interactions: {context.get('interaction_count', 0)}
- Relevant Past Context:
  - {context_summary}

Provide enrichment in this JSON format:
{{
  "implicit_goals": ["goal1", "goal2"],
  "suggested_topics": ["topic1", "topic2"],
  "inferred_expertise_level": "beginner|intermediate|advanced",
  "likely_next_steps": ["step1", "step2"]
}}

JSON Response:"""

        try:
            response = await self._generate(
                prompt=prompt,
                temperature=0.6,
                max_tokens=400,
            )
            
            # Parse JSON response
            enrichment = json.loads(response)
            
            # Add enrichment to context
            enriched_context = context.copy()
            enriched_context["enrichment"] = enrichment
            
            logger.debug("ollama_context_enriched", enrichment_keys=list(enrichment.keys()))
            
            return enriched_context
            
        except Exception as e:
            logger.error("ollama_context_enrichment_failed", error=str(e))
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
            response = await self._generate(
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
            logger.error("ollama_followup_generation_failed", error=str(e))
            return []
    
    def _build_hypothesis_prompt(
        self,
        user_input: str | None,
        context: dict[str, Any],
        count: int,
        media_type: MediaType,
    ) -> str:
        """Build prompt for hypothesis generation with enriched context."""
        persona_desc = self._describe_persona(context.get("persona_facets", {}))
        
        # Build enriched context summary using semantic search results
        context_parts = []
        
        # Relevant goals (semantically similar to current input)
        if context.get("relevant_goals"):
            goals_str = "\n  - " + "\n  - ".join(context['relevant_goals'][:3])
            context_parts.append(f"Relevant Goals:{goals_str}")
        
        # Relevant preferences  
        if context.get("relevant_preferences"):
            prefs_str = "\n  - " + "\n  - ".join(context['relevant_preferences'][:4])
            context_parts.append(f"Preferences:{prefs_str}")
        
        # Similar past interactions (NEW: helps identify patterns)
        if context.get("similar_history"):
            history_str = "\n  - " + "\n  - ".join(context['similar_history'][:2])
            context_parts.append(f"Similar Past Interactions:{history_str}")
        
        # Relevant artifacts (NEW: code snippets, docs, etc.)
        if context.get("relevant_artifacts"):
            artifacts_str = "\n  - " + "\n  - ".join(context['relevant_artifacts'][:2])
            context_parts.append(f"Relevant Past Work:{artifacts_str}")
        
        # Style preferences
        if context.get("styles"):
            styles_str = ", ".join(context['styles'])
            context_parts.append(f"Style Preferences: {styles_str}")
        
        context_str = "\n\n".join(context_parts) if context_parts else "No prior context available."
        
        # Add memory count for transparency
        mem_count = context.get('total_relevant_memories', 0)
        if mem_count > 0:
            context_str += f"\n\n(Based on {mem_count} relevant past memories)"
        
        media_context = ""
        if media_type != MediaType.TEXT:
            media_context = f"\nMedia Type: {media_type.value}"
        
        # Add content description (especially useful for images from vision model)
        content_desc = ""
        if context.get("content_description"):
            if media_type == MediaType.IMAGE:
                content_desc = f"\n\nImage Description (from vision model):\n{context['content_description']}"
            else:
                content_desc = f"\n\nContent Description:\n{context['content_description']}"
        
        prompt = f"""You are an AI assistant helping to understand user intent. Generate {count} hypotheses about what the user wants to accomplish.

IMPORTANT: Use the user's relevant past context below to generate MORE ACCURATE and PERSONALIZED hypotheses.

User Input: {user_input or "(no text input)"}
{media_context}{content_desc}

User Profile:
{persona_desc}
Total Interactions: {context.get('interaction_count', 0)}

Relevant Context (Semantically Matched to Current Input):
{context_str}

Generate {count} hypotheses as clarifying questions. Each hypothesis should:
1. Be phrased as a clear question
2. Leverage the relevant context above to be more specific and personalized
3. Include a confidence score (0.0-1.0) - higher if it matches past patterns
4. Include a brief rationale explaining WHY this hypothesis is likely

Format each hypothesis as:
Q: [question]
Confidence: [0.0-1.0]
Rationale: [brief explanation based on context]

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
                            rationale=current_hyp.get("rationale", "Generated by Ollama"),
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
                    rationale=current_hyp.get("rationale", "Generated by Ollama"),
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
        """Generate fallback hypotheses if Ollama fails."""
        # Better fallback questions
        if user_input:
            questions = [
                f"Do you want help with: {user_input}?",
                f"Would you like me to explain {user_input}?",
                f"Are you looking for information about {user_input}?",
            ]
        else:
            # When no text input (e.g., image-only)
            questions = [
                "What would you like to know about this content?",
                "Would you like me to analyze or describe this?",
                "How can I help you with this?",
            ]
        
        return [
            Hypothesis(
                id=f"h{i+1}",
                question=questions[i % len(questions)],
                rationale="Fallback hypothesis",
                confidence=0.6 - (i * 0.1),
            )
            for i in range(count)
        ]
    
    async def _generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        """
        Call Ollama generate API.
        
        Args:
            prompt: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        
        data = response.json()
        return data["response"]
    
    async def _chat_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        """
        Call Ollama chat API (for compatibility with persona/ranking services).
        
        Args:
            prompt: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        return await self._generate(prompt, temperature, max_tokens)
    
    async def generate_image_description(
        self,
        image_data: bytes,
        prompt: str = "Describe this image in detail.",
    ) -> str:
        """
        Generate a description of an image using Ollama vision model.
        
        Args:
            image_data: Raw image bytes
            prompt: Prompt for the vision model
            
        Returns:
            Text description of the image
        """
        import base64
        
        # Convert image to base64
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        
        url = f"{self.base_url}/api/generate"
        
        # Use vision-capable model if available, otherwise fallback
        # Common vision models: llava, llama3.2-vision, minicpm-v
        vision_model = self._get_vision_model_name()
        
        payload = {
            "model": vision_model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
        }
        
        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            description = data["response"].strip()
            
            logger.info(
                "ollama_image_description_generated",
                model=vision_model,
                description_length=len(description),
            )
            
            logger.info(f"Ollama image description: {description}")
            return description
            
        except Exception as e:
            logger.error("ollama_image_description_failed", error=str(e))
            raise
    
    async def generate_embedding(
        self,
        text: str,
        model: str | None = None,
    ) -> list[float]:
        """
        Generate text embedding using Ollama embedding model.
        
        Args:
            text: Text to embed
            model: Optional embedding model name (defaults to configured embed_model)
            
        Returns:
            Embedding vector
        """
        url = f"{self.base_url}/api/embeddings"
        
        # Use configured embedding model or override
        embed_model = model or self.embed_model
        
        payload = {
            "model": embed_model,
            "prompt": text,
        }
        
        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            embedding = data["embedding"]
            
            logger.debug(
                "ollama_embedding_generated",
                model=embed_model,
                dims=len(embedding),
            )
            
            return embedding
            
        except Exception as e:
            logger.error("ollama_embedding_failed", error=str(e))
            raise
    
    def _get_vision_model_name(self) -> str:
        """
        Get the vision model name to use.
        
        Returns:
            Vision model name
        """
        return self.vision_model
    
    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

