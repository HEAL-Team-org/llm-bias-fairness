from __future__ import annotations
import argparse
import os
from typing import List
import google.generativeai as genai
from . import BaseLLM


class GeminiLLM(BaseLLM):
    model_name: str = "gemini-1.5-flash"
    short_name: str = "gemini"

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model_name = args.gemini_model
        self.embed_model = args.gemini_embed_model
        self.params = {
            "temperature": args.gemini_temperature,
            "max_tokens": args.gemini_max_tokens,
        }
        api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def is_available(self) -> bool:
        try:
            genai.list_models()
            return True
        except Exception:
            return False

    def generate(self, text: str) -> str:
        response = self.model.generate_content(text)
        return response.text

    def chat(self, messages: List[dict]) -> str:
        history = [
            {
                "role": "model" if m["role"] == "assistant" else m["role"],
                "parts": [m["content"]],
            }
            for m in messages[:-1]
        ]
        session = self.model.start_chat(history=history)
        response = session.send_message(messages[-1]["content"])
        return response.text

    def _generate_embedding(self, text: str) -> List[float]:
        response = genai.embed_content(model=self.embed_model, content=text)
        return response["embedding"]

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group("Gemini")
        group.add_argument(
            "--gemini-api-key",
            type=str,
            default=None,
            help="Defaults to GEMINI_API_KEY env var",
        )
        group.add_argument("--gemini-model", type=str, default="gemini-1.5-flash")
        group.add_argument(
            "--gemini-embed-model", type=str, default="models/text-embedding-004"
        )
        group.add_argument("--gemini-temperature", type=float, default=0.7)
        group.add_argument("--gemini-max-tokens", type=int, default=1024)
        return parser
