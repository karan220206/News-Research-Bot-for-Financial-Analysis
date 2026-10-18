"""Shared helpers: the chat model and friendly error messages."""

import openai
from langchain_openai import ChatOpenAI

import config


class MissingAPIKeyError(RuntimeError):
    pass


def require_api_key() -> None:
    if not config.has_api_key():
        raise MissingAPIKeyError("OPENAI_API_KEY is missing. Add it to the .env file.")


def get_chat_model(temperature: float = config.TEMPERATURE) -> ChatOpenAI:
    require_api_key()
    return ChatOpenAI(
        model=config.CHAT_MODEL,
        temperature=temperature,
        api_key=config.OPENAI_API_KEY,
        max_retries=2,
    )


def friendly_error(exc: Exception) -> str:
    """Turn an exception into a message a user can act on."""
    if isinstance(exc, MissingAPIKeyError):
        return "OpenAI API key is missing. Add OPENAI_API_KEY to the .env file and restart."
    if isinstance(exc, openai.AuthenticationError):
        return "OpenAI rejected the API key. Check OPENAI_API_KEY in the .env file."
    if isinstance(exc, openai.RateLimitError):
        if "insufficient_quota" in str(exc):
            return "Your OpenAI account is out of credits. Add credits on platform.openai.com."
        return "OpenAI rate limit hit. Wait a few seconds and try again."
    if isinstance(exc, openai.NotFoundError):
        return f"OpenAI doesn't recognise a model name in config.py ({config.CHAT_MODEL} / {config.EMBEDDING_MODEL})."
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return "Couldn't reach OpenAI. Check your internet connection and try again."
    return f"Something went wrong: {exc}"
