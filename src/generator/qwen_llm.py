from __future__ import annotations
import argparse
import os
from typing import List
import requests
from . import BaseLLM


class QwenLLM(BaseLLM):
    model_name: str = "qwen3"
    short_name: str = "qwen"

    def __init__(self, args: argparse.Namespace):
        self.args        = args
        self.model_name  = args.qwen_model
        self.base_url    = args.qwen_url
        self.params      = {
            "temperature":      args.qwen_temperature,
            "max_new_tokens":   args.qwen_max_tokens,
            "top_p":            0.9,
            "do_sample":        True,
            "repetition_penalty": 1.05,
        }

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def generate(self, text: str) -> str:
        payload = {"prompt": text, **self.params}
        response = requests.post(f"{self.base_url}/generate", json=payload, timeout=2400)
        response.raise_for_status()
        return response.json()["output"].strip()

    def chat(self, messages: List[dict]) -> str:
        # Convert chat template to a single prompt string
        prompt = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        )
        return self.generate(prompt)

    def _generate_embedding(self, text: str) -> List[float]:
        response = requests.post(
            f"{self.base_url}/embed",
            json={"text": text},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["embedding"]

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group("Qwen")
        group.add_argument("--qwen-url",         type=str,   default="http://localhost:8001",
                                                              help="URL of the local Qwen server")
        group.add_argument("--qwen-model",       type=str,   default="qwen3")
        group.add_argument("--qwen-temperature", type=float, default=0.3)
        group.add_argument("--qwen-max-tokens",  type=int,   default=8192)
        return parser
