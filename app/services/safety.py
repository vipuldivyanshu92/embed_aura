"""Safety and PII redaction service."""

import re

import structlog

logger = structlog.get_logger()


class SafetyService:
    """
    Handles PII redaction and safety filtering.

    Redacts:
    - Email addresses
    - Phone numbers
    - API keys and tokens
    """

    # Regex patterns for PII detection
    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

    PHONE_PATTERN = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")

    # Common API key patterns (Bearer tokens, etc.)
    API_KEY_PATTERN = re.compile(r"\b(?:Bearer\s+)?[A-Za-z0-9_\-]{20,}\b")

    # Additional patterns
    SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    CREDIT_CARD_PATTERN = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")

    def redact_pii(self, text: str) -> str:
        """
        Redact PII from text.

        Args:
            text: Text potentially containing PII

        Returns:
            Text with PII redacted
        """
        if not text:
            return text

        original_length = len(text)

        # Redact emails
        text, email_count = self._redact_pattern(text, self.EMAIL_PATTERN, "<REDACTED:EMAIL>")

        # Redact phone numbers
        text, phone_count = self._redact_pattern(text, self.PHONE_PATTERN, "<REDACTED:PHONE>")

        # Redact API keys (be conservative - check length and context)
        text, key_count = self._redact_api_keys(text)

        # Redact SSN
        text, ssn_count = self._redact_pattern(text, self.SSN_PATTERN, "<REDACTED:SSN>")

        # Redact credit cards
        text, cc_count = self._redact_pattern(
            text, self.CREDIT_CARD_PATTERN, "<REDACTED:CREDIT_CARD>"
        )

        total_redactions = email_count + phone_count + key_count + ssn_count + cc_count

        if total_redactions > 0:
            logger.info(
                "pii_redacted",
                email=email_count,
                phone=phone_count,
                api_key=key_count,
                ssn=ssn_count,
                credit_card=cc_count,
                original_length=original_length,
                redacted_length=len(text),
            )

        return text

    def _redact_pattern(self, text: str, pattern: re.Pattern, replacement: str) -> tuple[str, int]:
        """
        Redact matches of a pattern.

        Args:
            text: Text to redact
            pattern: Regex pattern
            replacement: Replacement string

        Returns:
            Tuple of (redacted_text, count)
        """
        matches = list(pattern.finditer(text))
        count = len(matches)

        if count > 0:
            text = pattern.sub(replacement, text)

        return text, count

    def _redact_api_keys(self, text: str) -> tuple[str, int]:
        """
        Redact API keys with more careful heuristics.

        Args:
            text: Text to redact

        Returns:
            Tuple of (redacted_text, count)
        """
        # Look for common API key indicators
        indicators = [
            "api_key=",
            "apikey=",
            "token=",
            "secret=",
            "password=",
            "Authorization: Bearer",
        ]

        count = 0
        for indicator in indicators:
            # Case-insensitive search
            pattern = re.compile(
                rf"{re.escape(indicator)}\s*([A-Za-z0-9_\-]{{20,}})",
                re.IGNORECASE,
            )

            matches = list(pattern.finditer(text))
            count += len(matches)

            if matches:
                text = pattern.sub(
                    lambda m: f"{m.group(0).split(m.group(1))[0]}<REDACTED:API_KEY>",
                    text,
                )

        return text, count
