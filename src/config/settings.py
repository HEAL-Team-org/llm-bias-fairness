"""Configuration Management System.

This module handles loading and accessing configuration values from YAML files.
It supports a default configuration with user overrides.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager with default and user override support."""

    _instance: Optional["Config"] = None
    _config: Dict[str, Any] = {}
    _loaded: bool = False

    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize configuration (loads on first access)."""
        if not self._loaded:
            self.reload()

    def reload(self) -> None:
        """Load or reload configuration from files."""
        # Load default configuration
        default_config_path = Path(__file__).parent / "default_config.yaml"
        self._config = self._load_yaml(default_config_path)

        # Load user configuration if it exists (overrides defaults)
        user_config_path = Path.cwd() / "config" / "config.yaml"
        if user_config_path.exists():
            logger.info(f"Loading user configuration from {user_config_path}")
            user_config = self._load_yaml(user_config_path)
            self._deep_merge(self._config, user_config)
        else:
            logger.debug(f"No user configuration found at {user_config_path}, using defaults")

        # Override with environment variables where applicable
        self._apply_env_overrides()

        self._loaded = True
        logger.debug("Configuration loaded successfully")

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML file and return as dictionary.
        
        Args:
            path: Path to YAML file
            
        Returns:
            Dictionary with configuration values

        """
        try:
            with path.open(encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            logger.exception(f"Error loading configuration from {path}")
            return {}

    def _deep_merge(self, base: Dict, override: Dict) -> None:
        """Deep merge override dictionary into base dictionary.
        
        Args:
            base: Base dictionary to merge into
            override: Override dictionary with new values

        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        # OpenAI API Key
        if "OPENAI_API_KEY" in os.environ:
            self._config.setdefault("openai", {})["api_key"] = os.environ["OPENAI_API_KEY"]

        # OpenAI Base URL
        if "OPENAI_BASE_URL" in os.environ:
            self._config.setdefault("openai", {})["base_url"] = os.environ["OPENAI_BASE_URL"]

        # Log level
        if "LOG_LEVEL" in os.environ:
            self._config.setdefault("logging", {})["level"] = os.environ["LOG_LEVEL"]

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key_path: Path to configuration value using dots (e.g., 'openai.embedding_model')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Examples:
            >>> config = Config()
            >>> config.get('openai.embedding_model')
            'text-embedding-3-large'
            >>> config.get('openai.api_key')
            'sk-...'

        """
        keys = key_path.split(".")
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value using dot notation.
        
        Args:
            key_path: Path to configuration value using dots
            value: Value to set

        """
        keys = key_path.split(".")
        config = self._config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration dictionary.
        
        Returns:
            Complete configuration dictionary

        """
        return self._config.copy()

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access to configuration."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Dictionary-style setting of configuration."""
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration."""
        return self.get(key) is not None


# Global configuration instance
_config = Config()


def get_config() -> Config:
    """Get the global configuration instance.
    
    Returns:
        Global Config instance

    """
    return _config


def reload_config() -> None:
    """Reload configuration from files."""
    _config.reload()


# Convenience functions for common configuration access
def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from config or environment."""
    return _config.get("openai.api_key") or os.environ.get("OPENAI_API_KEY")


def get_openai_base_url() -> Optional[str]:
    """Get OpenAI base URL from config or environment."""
    return _config.get("openai.base_url") or os.environ.get("OPENAI_BASE_URL")


def get_embedding_model() -> str:
    """Get embedding model name."""
    return _config.get("openai.embedding_model", "text-embedding-3-large")


def get_chat_model() -> str:
    """Get chat model name."""
    return _config.get("openai.chat_model", "gpt-4")


def get_image_model() -> str:
    """Get image generation model name."""
    return _config.get("openai.image_model", "dall-e-3")


def get_cache_file(cache_type: str) -> str:
    """Get cache file path for given cache type.
    
    Args:
        cache_type: Type of cache (embeddings, stereoset, diversity, stereoset_dataset)
        
    Returns:
        Cache file path

    """
    return _config.get(f"cache.{cache_type}", f"{cache_type}.pkl")


def get_data_path(path_key: str) -> str:
    """Get data file path.
    
    Args:
        path_key: Key in paths configuration
        
    Returns:
        Data file path

    """
    return _config.get(f"paths.{path_key}", "")


def get_enhancement_param(param: str, default: Any = None) -> Any:
    """Get enhancement pipeline parameter.
    
    Args:
        param: Parameter name
        default: Default value if not found
        
    Returns:
        Parameter value

    """
    return _config.get(f"enhancement.{param}", default)


def get_image_gen_param(param: str, default: Any = None) -> Any:
    """Get image generation parameter.
    
    Args:
        param: Parameter name
        default: Default value if not found
        
    Returns:
        Parameter value

    """
    return _config.get(f"image_generation.{param}", default)
