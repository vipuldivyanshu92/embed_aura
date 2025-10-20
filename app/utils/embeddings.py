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
                vision_model=settings.ollama_vision_model,
                embed_model=settings.ollama_embed_model,
            )
            _ollama_client = OllamaClient(
                base_url=settings.ollama_base_url,
                model_name=settings.ollama_model_name,
                timeout=settings.ollama_timeout,
                vision_model=settings.ollama_vision_model,
                embed_model=settings.ollama_embed_model,
            )
            logger.info("ollama_client_loaded")
        except Exception as e:
            logger.error("ollama_client_load_failed", error=str(e))
            _ollama_client = None

    return _ollama_client


def _hash_based_embedding(content: str, dims: int | None = None) -> list[float]:
    """Generate deterministic hash-based embedding (fallback)."""
    # Use Ollama dimensions if configured, otherwise use config default
    if dims is None:
        settings = get_settings()
        if settings.slm_impl == "ollama":
            # Match Ollama's embedding dimensions (nomic-embed-text = 768)
            dims = 768
        else:
            dims = settings.embed_dims
    
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
    """Load image from URL (supports both HTTP/HTTPS and file:// URLs)."""
    try:
        from PIL import Image
        
        # Handle file:// URLs
        if url.startswith("file://"):
            from urllib.parse import unquote, urlparse
            
            # Parse and decode the file path
            parsed = urlparse(url)
            file_path = unquote(parsed.path)
            
            # On Windows, remove leading slash from paths like /C:/...
            if len(file_path) > 2 and file_path[0] == '/' and file_path[2] == ':':
                file_path = file_path[1:]
            
            image = Image.open(file_path)
            return image.convert("RGB")
        
        # Handle HTTP/HTTPS URLs
        import requests
        
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
    Generate embedding for text using Ollama (if configured) or sentence-transformers.

    Args:
        text: Input text

    Returns:
        Embedding vector
    """
    settings = get_settings()

    # If using Ollama, use its embedding model for consistency
    if settings.slm_impl == "ollama":
        try:
            import asyncio
            import concurrent.futures
            
            def run_in_thread():
                from app.clients.slm.ollama import OllamaClient
                
                # Create new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    # Create fresh Ollama client
                    ollama_client = OllamaClient(
                        base_url=settings.ollama_base_url,
                        model_name=settings.ollama_model_name,
                        timeout=settings.ollama_timeout,
                        vision_model=settings.ollama_vision_model,
                        embed_model=settings.ollama_embed_model,
                    )
                    
                    # Run the async function
                    result = loop.run_until_complete(ollama_client.generate_embedding(text))
                    
                    # Close the Ollama client
                    loop.run_until_complete(ollama_client.close())
                    
                    return result
                finally:
                    loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                embedding = future.result(timeout=30)  # 30 second timeout
                logger.info("generated_text_embedding_ollama", dims=len(embedding))
                return embedding
        except Exception as e:
            logger.warning("ollama_text_embedding_failed", error=str(e))

    # Try sentence-transformers (fallback or non-Ollama)
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
    return _hash_based_embedding(text)


async def _generate_image_embedding_ollama_with_client(
    image_data: bytes,
    ollama_client: Any,
) -> tuple[list[float], str]:
    """
    Generate embedding for image using Ollama (vision + embedding).
    
    Args:
        image_data: Raw image bytes
        ollama_client: Ollama client instance
        
    Returns:
        Tuple of (embedding vector, image description from vision model)
    """
    # Step 1: Use vision model to describe the image
    description = await ollama_client.generate_image_description(
        image_data=image_data,
        prompt="Describe this image in detail, including objects, colors, scene, context, and any text visible.",
    )
    
    # Step 2: Generate embedding from the description
    embedding = await ollama_client.generate_embedding(text=description)
    
    logger.info(
        "generated_image_embedding",
        method="ollama",
        description_length=len(description),
        embedding_dims=len(embedding),
        description=description,
    )
    
    return embedding, description


def generate_image_embedding(
    image_url: str | None = None, image_base64: str | None = None, description: str | None = None
) -> tuple[list[float], str]:
    """
    Generate embedding for image using Ollama (vision model + embeddings).
    Raises error if Ollama is unavailable or fails.

    Args:
        image_url: URL to image
        image_base64: Base64 encoded image
        description: Text description of image (not used, kept for compatibility)

    Returns:
        Tuple of (embedding vector, image description)
        
    Raises:
        ValueError: If image cannot be loaded
        Exception: If Ollama client is unavailable or embedding generation fails
    """
    # Load image
    image = None
    image_bytes = None
    if image_url:
        image = _load_image_from_url(image_url)
        if image is None:
            raise ValueError(f"Failed to load image from URL: {image_url}")
        # Convert PIL Image to bytes
        import io
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        image_bytes = img_byte_arr.getvalue()
    elif image_base64:
        image = _load_image_from_base64(image_base64)
        if image is None:
            raise ValueError("Failed to load image from base64 data")
        # Convert PIL Image to bytes
        import io
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        image_bytes = img_byte_arr.getvalue()
    else:
        raise ValueError("Either image_url or image_base64 must be provided")

    import asyncio
    import concurrent.futures
    
    # Run async function in a separate thread with its own event loop
    # This avoids "event loop already running" errors when called from FastAPI
    # We create a fresh Ollama client in the thread to avoid event loop conflicts
    def run_in_thread():
        from app.clients.slm.ollama import OllamaClient
        
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Create fresh Ollama client in this thread's event loop
            settings = get_settings()
            ollama_client = OllamaClient(
                base_url=settings.ollama_base_url,
                model_name=settings.ollama_model_name,
                timeout=settings.ollama_timeout,
                vision_model=settings.ollama_vision_model,
                embed_model=settings.ollama_embed_model,
            )
            
            # Run the async function
            result = loop.run_until_complete(
                _generate_image_embedding_ollama_with_client(image_bytes, ollama_client)
            )
            
            # Close the Ollama client
            loop.run_until_complete(ollama_client.close())
            
            return result
        finally:
            loop.close()
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_in_thread)
        try:
            embedding, img_description = future.result(timeout=120)  # 2 minute timeout
            return embedding, img_description
        except Exception as e:
            logger.error("ollama_image_embedding_failed", error=str(e))
            raise Exception(f"Failed to generate image embedding with Ollama: {str(e)}") from e


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
    return _hash_based_embedding(fallback_text)


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
    return _hash_based_embedding(fallback_text)


def generate_embedding(
    text: str | None = None,
    media_type: MediaType = MediaType.TEXT,
    media_url: str | None = None,
    media_base64: str | None = None,
    media_description: str | None = None,
) -> tuple[list[float], str]:
    """
    Generate multi-modal embedding based on input type.

    Args:
        text: Text content
        media_type: Type of media (text, image, audio, video)
        media_url: URL to media file
        media_base64: Base64 encoded media
        media_description: Text description of media

    Returns:
        Tuple of (embedding vector, content description)
    """
    try:
        if media_type == MediaType.TEXT:
            if text:
                return generate_text_embedding(text), text
            elif media_description:
                return generate_text_embedding(media_description), media_description
            else:
                raise ValueError("Text content required for TEXT media type")

        elif media_type == MediaType.IMAGE:
            return generate_image_embedding(media_url, media_base64, media_description or text)

        elif media_type == MediaType.AUDIO:
            audio_desc = media_description or text or "Audio content"
            return generate_audio_embedding(media_url, media_base64, audio_desc), audio_desc

        elif media_type == MediaType.VIDEO:
            video_desc = media_description or text or "Video content"
            return generate_video_embedding(media_url, media_base64, video_desc), video_desc

        else:
            raise ValueError(f"Unsupported media type: {media_type}")

    except Exception as e:
        logger.error("embedding_generation_failed", media_type=media_type, error=str(e))
        # Emergency fallback - uses smart dimension detection
        fallback_content = text or media_description or media_url or "fallback"
        return _hash_based_embedding(fallback_content), fallback_content
