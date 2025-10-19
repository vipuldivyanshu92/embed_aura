"""Multi-modal embedding generation with Ollama/CLIP/fallback support."""

import base64
import hashlib
import io
from typing import Any

import numpy as np
import structlog

from app.config import get_settings
from app.models import MediaType

logger = structlog.get_logger()

# Global variables for lazy loading models
_clip_model = None
_clip_processor = None
_sentence_transformer = None
_ollama_client = None


def _get_clip_model() -> tuple[Any, Any]:
    """Lazy load CLIP model for image/text embeddings."""
    global _clip_model, _clip_processor

    if _clip_model is None:
        try:
            from transformers import CLIPModel, CLIPProcessor

            logger.info("loading_clip_model", model="openai/clip-vit-base-patch32")
            _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            logger.info("clip_model_loaded")
        except ImportError:
            logger.warning(
                "clip_not_available",
                message="transformers not installed, falling back to hash-based embeddings",
            )
            _clip_model = None
            _clip_processor = None
        except Exception as e:
            logger.error("clip_load_failed", error=str(e))
            _clip_model = None
            _clip_processor = None

    return _clip_model, _clip_processor


def _get_sentence_transformer() -> Any:
    """Lazy load sentence transformer for text embeddings."""
    global _sentence_transformer

    if _sentence_transformer is None:
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("loading_sentence_transformer", model="all-MiniLM-L6-v2")
            _sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("sentence_transformer_loaded")
        except ImportError:
            logger.warning(
                "sentence_transformers_not_available",
                message="sentence-transformers not installed, using hash-based fallback",
            )
            _sentence_transformer = None
        except Exception as e:
            logger.error("sentence_transformer_load_failed", error=str(e))
            _sentence_transformer = None

    return _sentence_transformer


def _get_ollama_client() -> Any:
    """Lazy load Ollama client for image descriptions and embeddings."""
    global _ollama_client

    if _ollama_client is None:
        try:
            from app.clients.slm.ollama import OllamaClient

            settings = get_settings()
            logger.info(
                "loading_ollama_client",
                base_url=settings.ollama_base_url,
                model=settings.ollama_model_name,
            )
            _ollama_client = OllamaClient(
                base_url=settings.ollama_base_url,
                model_name=settings.ollama_model_name,
                timeout=settings.ollama_timeout,
            )
            logger.info("ollama_client_loaded")
        except Exception as e:
            logger.error("ollama_client_load_failed", error=str(e))
            _ollama_client = None

    return _ollama_client


def _hash_based_embedding(content: str, dims: int = 384) -> list[float]:
    """Generate deterministic hash-based embedding (fallback)."""
    # Normalize text
    normalized = content.lower().strip()

    # Generate deterministic seed from text
    text_hash = hashlib.sha256(normalized.encode()).digest()
    seed = int.from_bytes(text_hash[:4], byteorder="big")

    # Generate pseudo-random vector
    rng = np.random.RandomState(seed)
    vector = rng.randn(dims)

    # Normalize to unit vector
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    return vector.tolist()


def _load_image_from_url(url: str) -> Any:
    """Load image from URL."""
    try:
        import requests
        from PIL import Image

        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        return image.convert("RGB")
    except Exception as e:
        logger.error("image_load_failed", url=url, error=str(e))
        return None


def _load_image_from_base64(base64_str: str) -> Any:
    """Load image from base64 string."""
    try:
        from PIL import Image

        # Remove data URI prefix if present
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]

        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        return image.convert("RGB")
    except Exception as e:
        logger.error("image_decode_failed", error=str(e))
        return None


def generate_text_embedding(text: str) -> list[float]:
    """
    Generate embedding for text using sentence-transformers or CLIP.

    Args:
        text: Input text

    Returns:
        Embedding vector
    """
    settings = get_settings()

    # Try sentence-transformers first (faster for text)
    model = _get_sentence_transformer()
    if model is not None:
        try:
            embedding = model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.warning("sentence_transformer_failed", error=str(e))

    # Try CLIP text encoder
    clip_model, clip_processor = _get_clip_model()
    if clip_model is not None and clip_processor is not None:
        try:
            import torch

            inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
            with torch.no_grad():
                text_features = clip_model.get_text_features(**inputs)

            # Normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features[0].cpu().numpy().tolist()
        except Exception as e:
            logger.warning("clip_text_failed", error=str(e))

    # Fallback to hash-based
    logger.debug("using_hash_based_embedding", type="text")
    return _hash_based_embedding(text, settings.embed_dims)


def generate_image_embedding(
    image_url: str | None = None, image_base64: str | None = None, description: str | None = None
) -> list[float]:
    """
    Generate embedding for image using CLIP.

    Args:
        image_url: URL to image
        image_base64: Base64 encoded image
        description: Text description of image (fallback)

    Returns:
        Embedding vector
    """
    settings = get_settings()

    # Load image
    image = None
    if image_url:
        image = _load_image_from_url(image_url)
    elif image_base64:
        image = _load_image_from_base64(image_base64)

    # Try CLIP image encoder
    if image is not None:
        clip_model, clip_processor = _get_clip_model()
        if clip_model is not None and clip_processor is not None:
            try:
                import torch

                inputs = clip_processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    image_features = clip_model.get_image_features(**inputs)

                # Normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logger.info("generated_image_embedding", method="clip")
                return image_features[0].cpu().numpy().tolist()
            except Exception as e:
                logger.warning("clip_image_failed", error=str(e))

    # Fallback: use text description if available
    if description:
        logger.info("using_description_for_image_embedding")
        return generate_text_embedding(description)

    # Last resort: hash-based embedding from URL or metadata
    fallback_text = image_url or image_base64[:100] or "unknown_image"
    logger.debug("using_hash_based_embedding", type="image")
    return _hash_based_embedding(fallback_text, settings.embed_dims)


def generate_audio_embedding(
    audio_url: str | None = None, audio_base64: str | None = None, description: str | None = None
) -> list[float]:
    """
    Generate embedding for audio.

    Currently uses description fallback. Can be extended with audio models like CLAP.

    Args:
        audio_url: URL to audio file
        audio_base64: Base64 encoded audio
        description: Text description of audio

    Returns:
        Embedding vector
    """
    settings = get_settings()

    # For now, use text description
    # TODO: Integrate CLAP (Contrastive Language-Audio Pretraining) model
    if description:
        logger.info("using_description_for_audio_embedding")
        return generate_text_embedding(f"Audio: {description}")

    # Fallback to hash-based
    fallback_text = audio_url or "audio_content"
    logger.debug("using_hash_based_embedding", type="audio")
    return _hash_based_embedding(fallback_text, settings.embed_dims)


def generate_video_embedding(
    video_url: str | None = None, video_base64: str | None = None, description: str | None = None
) -> list[float]:
    """
    Generate embedding for video.

    Currently uses description fallback. Can be extended with video models like VideoCLIP.

    Args:
        video_url: URL to video file
        video_base64: Base64 encoded video
        description: Text description of video

    Returns:
        Embedding vector
    """
    settings = get_settings()

    # For now, use text description
    # TODO: Integrate video embedding model (e.g., VideoCLIP, X-CLIP)
    if description:
        logger.info("using_description_for_video_embedding")
        return generate_text_embedding(f"Video: {description}")

    # Fallback to hash-based
    fallback_text = video_url or "video_content"
    logger.debug("using_hash_based_embedding", type="video")
    return _hash_based_embedding(fallback_text, settings.embed_dims)


def generate_embedding(
    text: str | None = None,
    media_type: MediaType = MediaType.TEXT,
    media_url: str | None = None,
    media_base64: str | None = None,
    media_description: str | None = None,
) -> list[float]:
    """
    Generate multi-modal embedding based on input type.

    Args:
        text: Text content
        media_type: Type of media (text, image, audio, video)
        media_url: URL to media file
        media_base64: Base64 encoded media
        media_description: Text description of media

    Returns:
        Embedding vector
    """
    try:
        if media_type == MediaType.TEXT:
            if text:
                return generate_text_embedding(text)
            elif media_description:
                return generate_text_embedding(media_description)
            else:
                raise ValueError("Text content required for TEXT media type")

        elif media_type == MediaType.IMAGE:
            return generate_image_embedding(media_url, media_base64, media_description or text)

        elif media_type == MediaType.AUDIO:
            return generate_audio_embedding(media_url, media_base64, media_description or text)

        elif media_type == MediaType.VIDEO:
            return generate_video_embedding(media_url, media_base64, media_description or text)

        else:
            raise ValueError(f"Unsupported media type: {media_type}")

    except Exception as e:
        logger.error("embedding_generation_failed", media_type=media_type, error=str(e))
        # Emergency fallback
        settings = get_settings()
        fallback_content = text or media_description or media_url or "fallback"
        return _hash_based_embedding(fallback_content, settings.embed_dims)
