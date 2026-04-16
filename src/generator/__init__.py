from __future__ import annotations
import abc
import concurrent.futures
from typing import List
import argparse


class BaseLLM(abc.ABC):
    registry: dict = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BaseLLM.registry[cls.__name__.lower()] = cls

    model_name: str = ""  # e.g. "gpt-4o"
    short_name: str = ""  # e.g. "gpt4o"
    params: dict = {}  # temperature, max_tokens, top_p, ...

    @abc.abstractmethod
    def generate(self, text: str) -> str:
        """Return only the answer to the given text."""

    @abc.abstractmethod
    def chat(self, messages: List[dict]) -> str:
        """Accept an OpenAI-style chat template and return the assistant reply."""

    @abc.abstractmethod
    def _generate_embedding(self, text: str) -> List[float]:
        """Return the embedding vector for a single text. Implement this in subclasses."""

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts in parallel by default."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return list(executor.map(self._generate_embedding, texts))

    def run_parallel(self, fn, inputs: list) -> list:
        """Helper: run any of the above methods over a list of inputs in parallel."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return list(executor.map(fn, inputs))

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        pass

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if the backend is reachable and ready."""


from .openai_llm import OpenAILLM
from .groq_llm import GroqLLM
from .gemini_llm import GeminiLLM
from .qwen_llm import QwenLLM
