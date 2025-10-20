#!/usr/bin/env python3
"""
Utility script to re-embed memories with a new embedding model.

This is useful when:
1. Switching embedding models (e.g., from sentence-transformers to Ollama)
2. Fixing dimension mismatches in existing memories
3. Migrating to a new embedding dimension

Usage:
    python migrate_embeddings.py [--user-id USER_ID] [--force]

Options:
    --user-id USER_ID    Re-embed only for specific user (default: all users)
    --force             Re-embed all memories even if dimensions match
"""

import argparse
import asyncio
import json
from pathlib import Path

import structlog

from app.config import get_settings
from app.memory.local import LocalMemoryProvider
from app.models import MemoryItem
from app.utils.embeddings import generate_embedding

logger = structlog.get_logger()


async def migrate_embeddings(user_id: str | None = None, force: bool = False) -> None:
    """
    Re-embed memories for users with mismatched embedding dimensions.
    
    Args:
        user_id: If provided, only re-embed for this user
        force: If True, re-embed all memories regardless of dimension match
    """
    settings = get_settings()
    memory_provider = LocalMemoryProvider()
    
    # Load current memories
    memories_file = Path(settings.data_dir) / "memories.json"
    if not memories_file.exists():
        logger.error("memories_file_not_found", path=str(memories_file))
        return
    
    with open(memories_file) as f:
        data = json.load(f)
    
    # Get target embedding dimension
    test_embedding, _ = generate_embedding(text="test")
    target_dims = len(test_embedding)
    
    logger.info(
        "migration_started",
        target_dims=target_dims,
        user_filter=user_id,
        force=force,
    )
    
    total_migrated = 0
    total_skipped = 0
    
    for uid, memories in data.items():
        # Filter by user if specified
        if user_id and uid != user_id:
            continue
        
        logger.info("processing_user", user_id=uid, memory_count=len(memories))
        
        for memory_data in memories:
            memory = MemoryItem(**memory_data)
            current_dims = len(memory.embedding)
            
            # Skip if dimensions match and not forcing
            if current_dims == target_dims and not force:
                total_skipped += 1
                continue
            
            # Re-generate embedding
            try:
                new_embedding, _ = generate_embedding(
                    text=memory.content,
                    media_type=memory.media_type,
                )
                
                # Update memory
                memory.embedding = new_embedding
                await memory_provider.store_memory(memory)
                
                logger.info(
                    "memory_re_embedded",
                    memory_id=memory.id,
                    old_dims=current_dims,
                    new_dims=len(new_embedding),
                )
                total_migrated += 1
                
            except Exception as e:
                logger.error(
                    "failed_to_re_embed",
                    memory_id=memory.id,
                    error=str(e),
                )
    
    logger.info(
        "migration_completed",
        total_migrated=total_migrated,
        total_skipped=total_skipped,
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Re-embed memories with new embedding model"
    )
    parser.add_argument(
        "--user-id",
        type=str,
        help="Re-embed only for specific user",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed all memories even if dimensions match",
    )
    
    args = parser.parse_args()
    
    # Run migration
    asyncio.run(migrate_embeddings(
        user_id=args.user_id,
        force=args.force,
    ))


if __name__ == "__main__":
    main()

