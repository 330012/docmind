"""
LLM provider factory.

Returns a configured chat model based on settings. Centralising this
means the rest of the codebase never hardcodes a provider — switching
from Groq to OpenAI or Anthropic is a single config change.
"""

from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from backend.config import settings


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """
    Return a cached chat model instance based on the configured provider.

    Supported providers: groq, openai, anthropic.
    The model is created once and reused across the process.
    """
    provider = settings.llm_provider.lower()

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.llm_model,
            api_key=settings.groq_api_key,
            temperature=0,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            temperature=0,
        )

    raise ValueError(
        f"Unsupported LLM provider: '{provider}'. "
        "Choose from: groq, openai, anthropic."
    )