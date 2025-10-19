"""Pytest configuration and fixtures."""

import os
import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set test environment variables
os.environ["APP_ENV"] = "test"
os.environ["MEMORY_BACKEND"] = "local"
os.environ["SLM_IMPL"] = "local"
os.environ["DATA_DIR"] = "/tmp/embed_test_data"


@pytest.fixture(autouse=True)
def reset_settings_cache() -> None:
    """Reset settings cache between tests."""
    from app.config import get_settings

    get_settings.cache_clear()
