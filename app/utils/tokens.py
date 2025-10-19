"""Token estimation utilities."""


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text using simple heuristic.

    Rule of thumb: ~4 characters per token for English text.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    if not text:
        return 0

    # Simple heuristic: 4 chars per token
    char_count = len(text)
    token_estimate = char_count // 4

    # Add tokens for whitespace (roughly 1 token per 5 words)
    word_count = len(text.split())
    token_estimate += word_count // 5

    return max(1, token_estimate)  # At least 1 token for non-empty text
