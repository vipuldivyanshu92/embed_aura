"""Small Language Model client implementations."""

from app.clients.slm.base import SLMClient
from app.clients.slm.http import HttpSLM
from app.clients.slm.local import LocalSLM

__all__ = ["SLMClient", "LocalSLM", "HttpSLM"]
