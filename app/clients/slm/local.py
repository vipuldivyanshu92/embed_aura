"""Local rule-based SLM implementation for offline operation."""

import hashlib
import re
from typing import Any

from app.clients.slm.base import SLMClient
from app.models import Hypothesis, MediaType
from app.utils.tokens import estimate_tokens


class LocalSLM(SLMClient):
    """
    Local deterministic SLM implementation using heuristics.

    This implementation allows the service to run completely offline
    without external API dependencies.
    """

    # Common task patterns and their clarifications
    PATTERNS = [
        (
            r"\b(build|create|make|develop|write)\s+(an?|the)?\s*(api|rest|endpoint)",
            "build an API",
            [
                ("Do you want to build a REST API with a specific framework?", 0.85),
                ("Do you want to design API endpoints and data models?", 0.75),
                ("Do you want to implement authentication for the API?", 0.60),
            ],
        ),
        (
            r"\b(fix|debug|solve|resolve)\s+(bug|issue|error|problem)",
            "fix a bug",
            [
                ("Do you want help debugging a specific error message?", 0.80),
                ("Do you want to trace the root cause of an issue?", 0.70),
                ("Do you want best practices for troubleshooting?", 0.55),
            ],
        ),
        (
            r"\b(test|unit test|integration test)\s",
            "write tests",
            [
                ("Do you want to write unit tests for existing code?", 0.82),
                ("Do you want to set up a testing framework?", 0.68),
                ("Do you want to improve test coverage?", 0.60),
            ],
        ),
        (
            r"\b(refactor|improve|optimize|clean)\s",
            "refactor code",
            [
                ("Do you want to refactor code for better readability?", 0.78),
                ("Do you want to optimize performance?", 0.72),
                ("Do you want to apply design patterns?", 0.58),
            ],
        ),
        (
            r"\b(deploy|deployment|ci/cd|pipeline)\s",
            "deployment",
            [
                ("Do you want to set up a CI/CD pipeline?", 0.80),
                ("Do you want deployment configuration for a specific platform?", 0.73),
                ("Do you want to automate the deployment process?", 0.65),
            ],
        ),
    ]

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
        Generate hypotheses using pattern matching and context (multi-modal support).

        OPTIMIZED: Now receives context with semantically relevant memories.

        Args:
            user_input: User's text input
            context: Context dict with 'persona_facets', 'relevant_goals', 'relevant_preferences', etc.
            count: Number of hypotheses (max 3)
            media_type: Type of media (text, image, audio, video)
            media_url: URL to media file
            media_base64: Base64 encoded media

        Returns:
            List of hypotheses sorted by confidence
        """
        hypotheses: list[Hypothesis] = []

        # Handle different media types
        if media_type == MediaType.IMAGE:
            hypotheses = self._generate_image_hypotheses(user_input, context, count)
        elif media_type == MediaType.AUDIO:
            hypotheses = self._generate_audio_hypotheses(user_input, context, count)
        elif media_type == MediaType.VIDEO:
            hypotheses = self._generate_video_hypotheses(user_input, context, count)
        else:
            # TEXT - use pattern matching
            if user_input:
                user_input_lower = user_input.lower()
                matched = False
                for pattern, topic, clarifications in self.PATTERNS:
                    if re.search(pattern, user_input_lower):
                        matched = True
                        # Generate hypotheses based on pattern
                        for idx, (question, base_conf) in enumerate(clarifications[:count]):
                            # Adjust confidence based on persona
                            confidence = self._adjust_confidence(base_conf, context, idx)
                            hyp = Hypothesis(
                                id=f"h{idx + 1}",
                                question=question,
                                rationale=f"Detected intent: {topic}",
                                confidence=confidence,
                            )
                            hypotheses.append(hyp)
                        break

                # Fallback: generic hypotheses
                if not matched:
                    hypotheses = self._generate_generic_hypotheses(user_input, context, count)
            else:
                hypotheses = self._generate_generic_hypotheses("media content", context, count)

        # Sort by confidence and limit to count
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        return hypotheses[:count]

    def _adjust_confidence(self, base_conf: float, context: dict[str, Any], idx: int) -> float:
        """Adjust confidence based on persona and relevant context."""
        facets = context.get("persona_facets", {})

        # Apply small adjustments based on facets
        adjustment = 0.0

        # If user prefers concise, boost first (most direct) hypothesis
        if idx == 0 and facets.get("concise", 0.5) > 0.7:
            adjustment += 0.03

        # If user is code-first, slightly adjust for code-related hypotheses
        if facets.get("code_first", 0.5) > 0.7:
            adjustment += 0.02

        # OPTIMIZED: Boost confidence if we have relevant past context
        if context.get("similar_history") or context.get("relevant_goals"):
            # User has done similar things before - increase confidence
            adjustment += 0.05

        # Clamp to [0, 1]
        return max(0.0, min(1.0, base_conf + adjustment))

    def _generate_generic_hypotheses(
        self, user_input: str, context: dict[str, Any], count: int
    ) -> list[Hypothesis]:
        """Generate generic hypotheses when no pattern matches."""
        # Use input hash for deterministic but varied hypotheses
        input_hash = int(hashlib.md5(user_input.encode()).hexdigest()[:8], 16)

        templates = [
            (
                "Do you want detailed step-by-step guidance for: {input}?",
                "Based on your input pattern",
                0.70,
            ),
            (
                "Do you want code examples and implementation details for: {input}?",
                "Typical user preference",
                0.65,
            ),
            (
                "Do you want high-level concepts and best practices for: {input}?",
                "Alternative approach",
                0.55,
            ),
        ]

        hypotheses = []
        for idx in range(min(count, len(templates))):
            template_idx = (input_hash + idx) % len(templates)
            question_tmpl, rationale, base_conf = templates[template_idx]

            # Truncate input if too long
            display_input = user_input[:50] + "..." if len(user_input) > 50 else user_input

            confidence = self._adjust_confidence(base_conf, context, idx)
            hyp = Hypothesis(
                id=f"h{idx + 1}",
                question=question_tmpl.format(input=display_input),
                rationale=rationale,
                confidence=confidence,
            )
            hypotheses.append(hyp)

        return hypotheses

    def _generate_image_hypotheses(
        self, caption: str | None, context: dict[str, Any], count: int
    ) -> list[Hypothesis]:
        """Generate hypotheses for image inputs."""
        templates = [
            (
                "Do you want me to analyze and describe what's in this image?",
                "Image analysis request",
                0.80,
            ),
            (
                "Do you want to extract text or specific information from this image?",
                "OCR or information extraction",
                0.72,
            ),
            (
                "Do you want me to answer questions about this image?",
                "Visual question answering",
                0.68,
            ),
        ]

        # If caption provided, use it for context-aware hypotheses
        if caption:
            templates.insert(
                0,
                (
                    f"Do you want help with: {caption[:80]}...?",
                    "Based on image context",
                    0.85,
                ),
            )

        return self._build_hypotheses_from_templates(templates, context, count)

    def _generate_audio_hypotheses(
        self, description: str | None, context: dict[str, Any], count: int
    ) -> list[Hypothesis]:
        """Generate hypotheses for audio inputs."""
        templates = [
            (
                "Do you want me to transcribe this audio content?",
                "Speech-to-text transcription",
                0.82,
            ),
            (
                "Do you want me to summarize what was said in this audio?",
                "Audio summary request",
                0.75,
            ),
            (
                "Do you want me to analyze the sentiment or tone of this audio?",
                "Audio sentiment analysis",
                0.65,
            ),
        ]

        if description:
            templates.insert(
                0,
                (
                    f"Do you want help with this audio: {description[:80]}...?",
                    "Based on audio context",
                    0.85,
                ),
            )

        return self._build_hypotheses_from_templates(templates, context, count)

    def _generate_video_hypotheses(
        self, description: str | None, context: dict[str, Any], count: int
    ) -> list[Hypothesis]:
        """Generate hypotheses for video inputs."""
        templates = [
            (
                "Do you want me to summarize the content of this video?",
                "Video summarization",
                0.80,
            ),
            (
                "Do you want me to extract key frames or moments from this video?",
                "Video keyframe extraction",
                0.72,
            ),
            (
                "Do you want me to transcribe the speech and describe the visuals?",
                "Video transcription and description",
                0.68,
            ),
        ]

        if description:
            templates.insert(
                0,
                (
                    f"Do you want help with this video: {description[:80]}...?",
                    "Based on video context",
                    0.85,
                ),
            )

        return self._build_hypotheses_from_templates(templates, context, count)

    def _build_hypotheses_from_templates(
        self,
        templates: list[tuple[str, str, float]],
        context: dict[str, Any],
        count: int,
    ) -> list[Hypothesis]:
        """Build hypotheses from templates with confidence adjustment."""
        hypotheses = []
        for idx in range(min(count, len(templates))):
            question, rationale, base_conf = templates[idx]
            confidence = self._adjust_confidence(base_conf, context, idx)
            hyp = Hypothesis(
                id=f"h{idx + 1}",
                question=question,
                rationale=rationale,
                confidence=confidence,
            )
            hypotheses.append(hyp)
        return hypotheses

    async def summarize(self, text: str, max_tokens: int) -> str:
        """
        Summarize using extractive reduction (simple heuristic).

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

        # Simple extractive summarization: take sentences up to budget
        sentences = re.split(r"(?<=[.!?])\s+", text)

        summary_parts = []
        token_count = 0

        for sentence in sentences:
            sentence_tokens = estimate_tokens(sentence)
            if token_count + sentence_tokens <= max_tokens:
                summary_parts.append(sentence)
                token_count += sentence_tokens
            else:
                # Add truncation marker if we're cutting off
                if summary_parts:
                    summary_parts.append("...")
                break

        return " ".join(summary_parts) if summary_parts else text[: max_tokens * 4]
