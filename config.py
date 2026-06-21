"""Application configuration loaded from environment variables."""
import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_LANGSMITH_PLACEHOLDER_KEYS = frozenset({"", "YOURKEY", "your-key", "changeme"})


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"true", "1", "yes"}


class Config:
    """Central configuration. No fallback URIs — fail fast if missing."""

    MONGODB_URI: str = os.getenv("MONGODB_URI", "")
    MONGODB_DB: str = os.getenv("MONGODB_DB", "agentic-memory")

    LLM_URI: str = os.getenv("LLM_URI", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "")
    LLM_KEY: str = os.getenv("LLM_KEY", "")

    LLM_EMBEDDING_URI: str = os.getenv("LLM_EMBEDDING_URI", "")
    LLM_EMBEDDING_MODEL: str = os.getenv("LLM_EMBEDDING_MODEL", "voyage-4-lite")
    LLM_EMBEDDING_KEY: str = os.getenv("LLM_EMBEDDING_KEY", "")
    EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))

    LANGSMITH_TRACING: bool = _env_truthy("LANGSMITH_TRACING")
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT: str = _strip_quotes(os.getenv("LANGSMITH_PROJECT", "pensive"))
    LANGSMITH_ENDPOINT: str = os.getenv(
        "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
    )

    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # Optional fallback for curl/local testing when no OpenWebUI session headers are present.
    USER_ID: str = os.getenv("USER_ID", "")

    OPENWEBUI_USER_ID_HEADER: str = os.getenv(
        "OPENWEBUI_USER_ID_HEADER", "X-OpenWebUI-User-Id"
    )
    OPENWEBUI_CHAT_ID_HEADER: str = os.getenv(
        "OPENWEBUI_CHAT_ID_HEADER", "X-OpenWebUI-Chat-Id"
    )
    OPENWEBUI_MESSAGE_ID_HEADER: str = os.getenv(
        "OPENWEBUI_MESSAGE_ID_HEADER", "X-OpenWebUI-Message-Id"
    )

    @classmethod
    def validate(cls) -> None:
        """Raise RuntimeError with actionable message if required vars are missing."""
        missing = []
        if not cls.MONGODB_URI:
            missing.append("MONGODB_URI")
        if not cls.LLM_URI:
            missing.append("LLM_URI")
        if not cls.LLM_MODEL:
            missing.append("LLM_MODEL")
        if not cls.LLM_KEY:
            missing.append("LLM_KEY")
        if not cls.LLM_EMBEDDING_KEY:
            missing.append("LLM_EMBEDDING_KEY")
        if missing:
            raise RuntimeError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Copy env.example to .env and configure your connection strings."
            )

    @classmethod
    def anthropic_base_url(cls) -> str:
        """Derive Anthropic API base URL from LLM_URI.

        The Anthropic SDK posts to {base_url}/v1/messages, so for Grove gateways
        whose LLM_URI ends in /v1/messages we strip that suffix entirely.
        """
        uri = cls.LLM_URI.rstrip("/")
        if uri.endswith("/v1/messages"):
            return uri[: -len("/v1/messages")]
        if uri.endswith("/messages"):
            return uri[: -len("/messages")]
        return uri


def configure_langsmith() -> bool:
    """Enable LangSmith tracing when configured. Returns whether tracing is active."""
    if not Config.LANGSMITH_TRACING:
        logger.info("LangSmith tracing disabled")
        return False

    api_key = Config.LANGSMITH_API_KEY.strip()
    if not api_key or api_key in _LANGSMITH_PLACEHOLDER_KEYS:
        logger.warning(
            "LANGSMITH_TRACING is enabled but LANGSMITH_API_KEY is missing or unset; "
            "tracing will remain off"
        )
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = Config.LANGSMITH_PROJECT
    os.environ["LANGSMITH_ENDPOINT"] = Config.LANGSMITH_ENDPOINT

    logger.info("LangSmith tracing enabled (project=%s)", Config.LANGSMITH_PROJECT)
    return True
