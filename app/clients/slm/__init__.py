"""Small Language Model client implementations."""

from app.clients.slm.base import SLMClient
from app.clients.slm.http import HttpSLM
from app.clients.slm.local import LocalSLM
from app.clients.slm.openai import OpenAISLM
from app.clients.slm.vllm import VLLMClient

__all__ = ["SLMClient", "LocalSLM", "HttpSLM", "VLLMClient", "OpenAISLM"]
