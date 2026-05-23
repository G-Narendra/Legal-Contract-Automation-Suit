"""
Configuration loader for Legal Contract Automation Suite.
Loads from environment variables, YAML config files, and settings.py.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Optional, Any
from functools import lru_cache


@lru_cache(maxsize=1)
def load_config() -> Dict[str, Any]:
    """Load all configuration with caching. Call once, reuse everywhere."""
    config = {
        # API Keys (from environment - never hardcoded)
        "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "pinecone_api_key": os.getenv("PINECONE_API_KEY", ""),
        "tavily_api_key": os.getenv("TAVILY_API_KEY", ""),

        # Provider selection (one line switch)
        "provider": os.getenv("LLM_PROVIDER", "google"),
        "model": os.getenv("LLM_MODEL", "gemini-2.5-flash"),
        "lite_model": os.getenv("LLM_LITE_MODEL", "gemini-2.5-flash-lite"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2"),

        # Cache
        "cache_ttl": int(os.getenv("CACHE_TTL_SECONDS", "300")),

        # Logging
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "debug": os.getenv("DEBUG", "false").lower() == "true",
    }

    # Load models.yaml
    models_path = Path("config/models.yaml")
    if models_path.exists():
        with open(models_path, "r") as f:
            config["model_config"] = yaml.safe_load(f) or {}

    # Load system prompts
    prompts_path = Path("config/prompts/system_prompts.yaml")
    if prompts_path.exists():
        with open(prompts_path, "r") as f:
            config["system_prompts"] = yaml.safe_load(f) or {}
    else:
        config["system_prompts"] = {}

    return config


def get_provider_config(provider: Optional[str] = None) -> Dict:
    """Get provider-specific config for dynamic switching."""
    cfg = load_config()
    provider = provider or cfg["provider"]

    model_cfg = cfg.get("model_config", {}).get("llm", {}).get("primary", {})
    provider_key = model_cfg.get("api_key_env", f"{provider.upper()}_API_KEY")

    return {
        "provider": provider,
        "api_key": os.getenv(provider_key, ""),
        "model": model_cfg.get("model", cfg["model"]),
        "temperature": model_cfg.get("temperature", 0.1),
        "max_tokens": model_cfg.get("max_tokens", 4096),
    }
