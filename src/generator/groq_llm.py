from __future__ import annotations
import argparse
import os
from typing import List
from groq import Groq
from . import BaseLLM


class GroqLLM(BaseLLM):
    model_name: str = "llama-3.3-70b-versatile"
    short_name: str = "groq"

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model_name = args.groq_model
        self.params = {
            "temperature": args.groq_temperature,
            "max_tokens": args.groq_max_tokens,
        }
        api_key = args.groq_api_key or os.environ.get("GROQ_API_KEY")
        self.client = Groq(api_key=api_key)

    def is_available(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    def generate(self, text: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": text}],
            **self.params,
        )
        return response.choices[0].message.content

    def chat(self, messages: List[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self.params,
        )
        return response.choices[0].message.content

    def _generate_embedding(self, text: str) -> List[float]:
        raise NotImplementedError(
            "Groq does not have an embedding API. Use OpenAILLM or GeminiLLM instead."
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group("Groq")
        group.add_argument(
            "--groq-api-key",
            type=str,
            default=None,
            help="Defaults to GROQ_API_KEY env var",
        )
        group.add_argument("--groq-model", type=str, default="llama-3.3-70b-versatile")
        group.add_argument("--groq-temperature", type=float, default=0.7)
        group.add_argument("--groq-max-tokens", type=int, default=1024)
        return parser
