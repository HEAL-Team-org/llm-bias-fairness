"""Prompt template loader and manager.

This module provides utilities for loading and formatting prompt templates
from YAML files.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class PromptTemplateManager:
    """Manages loading and formatting of prompt templates from YAML files."""

    def __init__(self, prompts_dir: str | Path = "prompts"):
        """Initialize prompt template manager.

        Args:
            prompts_dir: Directory containing prompt YAML files

        """
        self.prompts_dir = Path(prompts_dir)
        self._templates: dict[str, Any] = {}
        self._loaded_files: set[str] = set()

    def load_templates(self, filename: str) -> None:
        """Load templates from a YAML file.

        Args:
            filename: Name of the YAML file (e.g., "enhancement_prompts.yaml")

        """
        if filename in self._loaded_files:
            logger.debug(f"Templates from {filename} already loaded")
            return

        filepath = self.prompts_dir / filename
        if not filepath.exists():
            logger.warning(f"Prompt template file not found: {filepath}")
            return

        try:
            with filepath.open(encoding="utf-8") as f:
                templates = yaml.safe_load(f)
                if templates:
                    self._templates.update(templates)
                    self._loaded_files.add(filename)
                    logger.info(f"Loaded {len(templates)} templates from {filename}")
        except Exception:
            logger.exception(f"Error loading templates from {filepath}")

    def get_template(self, template_name: str, **kwargs: Any) -> str:
        """Get a formatted prompt template.

        Args:
            template_name: Name of the template
            **kwargs: Variables to substitute in the template

        Returns:
            Formatted prompt string

        """
        if template_name not in self._templates:
            logger.warning(f"Template '{template_name}' not found")
            return ""

        template = self._templates[template_name]

        # Handle nested dictionaries (e.g., focus_descriptions)
        if isinstance(template, dict):
            logger.warning(
                f"Template '{template_name}' is a dictionary. "
                "Use get_nested_template() instead."
            )
            return str(template)

        # Format template with provided kwargs
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing variable in template '{template_name}': {e}")
            return template

    def get_nested_template(
        self,
        template_name: str,
        key: str,
        **kwargs: Any
    ) -> str:
        """Get a formatted prompt from a nested template dictionary.

        Args:
            template_name: Name of the template dictionary
            key: Key within the nested dictionary
            **kwargs: Variables to substitute in the template

        Returns:
            Formatted prompt string

        """
        if template_name not in self._templates:
            logger.warning(f"Template '{template_name}' not found")
            return ""

        template_dict = self._templates[template_name]
        if not isinstance(template_dict, dict):
            logger.warning(f"Template '{template_name}' is not a dictionary")
            return ""

        if key not in template_dict:
            logger.warning(
                f"Key '{key}' not found in template '{template_name}'"
            )
            return ""

        template = template_dict[key]
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.warning(
                f"Missing variable in template '{template_name}.{key}': {e}"
            )
            return template

    def has_template(self, template_name: str) -> bool:
        """Check if a template exists.

        Args:
            template_name: Name of the template

        Returns:
            True if template exists

        """
        return template_name in self._templates

    def list_templates(self) -> list[str]:
        """Get list of all loaded template names.

        Returns:
            List of template names

        """
        return list(self._templates.keys())


# Global prompt manager instance
_prompt_manager = PromptTemplateManager()


def get_prompt_manager() -> PromptTemplateManager:
    """Get the global prompt template manager.

    Returns:
        Global PromptTemplateManager instance

    """
    return _prompt_manager


def load_prompts(filename: str) -> None:
    """Load prompts from a YAML file.

    Args:
        filename: Name of the YAML file

    """
    _prompt_manager.load_templates(filename)


def get_prompt(template_name: str, **kwargs: Any) -> str:
    """Get a formatted prompt template.

    Args:
        template_name: Name of the template
        **kwargs: Variables to substitute

    Returns:
        Formatted prompt string

    """
    return _prompt_manager.get_template(template_name, **kwargs)


def get_nested_prompt(template_name: str, key: str, **kwargs: Any) -> str:
    """Get a formatted prompt from a nested template.

    Args:
        template_name: Name of the template dictionary
        key: Key within the nested dictionary
        **kwargs: Variables to substitute

    Returns:
        Formatted prompt string

    """
    return _prompt_manager.get_nested_template(template_name, key, **kwargs)


# Auto-load enhancement and query prompts
def initialize_prompts() -> None:
    """Initialize and load default prompt templates."""
    load_prompts("enhancement_prompts.yaml")
    load_prompts("query_prompts.yaml")
    logger.info("Initialized default prompt templates")
