from __future__ import annotations
import argparse
import os
from typing import List
from openai import OpenAI
from . import BaseLLM


class OpenAILLM(BaseLLM):
    model_name: str = "gpt-4o"
    short_name: str = "openai"

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model_name = args.openai_model
        self.embed_model = args.openai_embed_model
        self.params = {
            "temperature": args.openai_temperature,
            "max_tokens": args.openai_max_tokens,
        }
        api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

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
        response = self.client.embeddings.create(
            model=self.embed_model,
            input=text,
        )
        return response.data[0].embedding

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self.embed_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group("OpenAI")
        group.add_argument(
            "--openai-api-key",
            type=str,
            default=None,
            help="Defaults to OPENAI_API_KEY env var",
        )
        group.add_argument("--openai-model", type=str, default="gpt-4o")
        group.add_argument(
            "--openai-embed-model", type=str, default="text-embedding-3-small"
        )
        group.add_argument("--openai-temperature", type=float, default=0.7)
        group.add_argument("--openai-max-tokens", type=int, default=1024)
        return parser
