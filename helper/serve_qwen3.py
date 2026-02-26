#!/usr/bin/env python3
"""FastAPI server for local inference with Qwen/Qwen3-32B.

This server wraps the Qwen3-32B causal language model as a REST API.
  - POST /generate  — accepts a prompt and returns a text completion.
  - GET  /health    — returns 200 when the model is loaded and ready.

The server does NOT use any embedding model; it performs autoregressive
text generation only.  Embedding-based retrieval is handled separately
by src/graphrag.py (Qwen3-Embedding-4B via sentence-transformers).

How to run:
    # 1. Install dependencies (once)
    pip install fastapi uvicorn transformers torch accelerate

    # 2. Start the server (from the helper/ directory)
    cd helper
    uvicorn serve_qwen3:app --host 0.0.0.0 --port 8001 --workers 1

    # 3. (Optional) Verify the server is ready
    curl http://localhost:8001/health

    # 4. Send a generation request
    curl -X POST http://localhost:8001/generate \\
         -H "Content-Type: application/json" \\
         -d '{"prompt": "Describe a diverse team of engineers", "max_new_tokens": 256}'

Notes:
    - Keep --workers 1; the model is loaded into GPU memory once and shared
      via an asyncio lock.  Multiple workers would each load the model.
    - Requires a GPU with at least ~70 GB VRAM (bfloat16) for Qwen3-32B.
      For CPU-only machines the model falls back to float32 (very slow).
    - The first startup downloads the model weights from HuggingFace Hub
      (~65 GB).  Subsequent starts use the local cache.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import cast

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from logger import get_logger, log_timer

MODEL_ID = "Qwen/Qwen3-32B"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9

logger = get_logger(__name__)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(DEFAULT_MAX_NEW_TOKENS, ge=1, le=8192)
    temperature: float = Field(DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    top_p: float = Field(DEFAULT_TOP_P, ge=0.0, le=1.0)
    do_sample: bool = True
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)
    no_repeat_ngram_size: int = Field(10, ge=0, le=50)


class GenerateResponse(BaseModel):
    model: str
    prompt: str
    output: str
    finish_reason: str


_state: dict[str, object] = {
    "tokenizer": None,
    "model": None,
    "ready": False,
}
_generation_lock = asyncio.Lock()


def _build_input_ids(prompt: str) -> dict:
    """Build tokenized inputs using the chat template so Qwen3 thinking works correctly."""
    tokenizer = cast(PreTrainedTokenizerBase, _state["tokenizer"])
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Think carefully before answering."},
        {"role": "user",   "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(text, return_tensors="pt")


def _run_generation(request: GenerateRequest) -> str:
    model = cast(PreTrainedModel, _state["model"])
    tokenizer = cast(PreTrainedTokenizerBase, _state["tokenizer"])

    assert model is not None
    assert tokenizer is not None

    inputs = _build_input_ids(request.prompt)

    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            do_sample=request.do_sample,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            no_repeat_ngram_size=request.no_repeat_ngram_size,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    completion_tokens = generated[0][inputs["input_ids"].shape[1] :]
    output = tokenizer.decode(completion_tokens, skip_special_tokens=True).strip()
    return output


@asynccontextmanager
async def lifespan(_app: FastAPI):
    with log_timer(logger, f"Load model {MODEL_ID}"):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            max_memory={0: "26GiB", 1: "39GiB"},
            trust_remote_code=True,
        )

        if not torch.cuda.is_available():
            model = model.to("cpu")

        _state["tokenizer"] = tokenizer
        _state["model"] = model

    _state["ready"] = True
    logger.info("Qwen3 server is ready")
    yield
    logger.info("Qwen3 server shutting down")


app = FastAPI(title="Qwen3 Local Text Generation Server", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail="Model is loading")

    return {
        "status": "ready",
        "model": MODEL_ID,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail="Model is not ready")

    async with _generation_lock:
        loop = asyncio.get_running_loop()
        try:
            with log_timer(logger, "Qwen3 generation"):
                output = await loop.run_in_executor(None, _run_generation, request)
        except Exception as exc:
            logger.exception("Generation failed")
            raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    return GenerateResponse(
        model=MODEL_ID,
        prompt=request.prompt,
        output=output,
        finish_reason="stop",
    )
