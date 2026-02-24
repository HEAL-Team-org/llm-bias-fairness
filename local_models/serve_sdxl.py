#!/usr/bin/env python3
"""
FastAPI server for Stable Diffusion XL Base 1.0 text-to-image generation.

Architecture:
  - Model  : stabilityai/stable-diffusion-xl-base-1.0
  - Strategy: One pipeline instance split across available GPUs via
              device_map="balanced" (requires accelerate).
  - One request at a time (asyncio.Lock) to avoid VRAM contention.

Run on the SSH server:
    uvicorn local_models.serve_sdxl:app --host 0.0.0.0 --port 8002 --workers 1

    NOTE: --workers MUST be 1. Multiple workers would each load a full model
    copy and likely OOM.

Requirements:
    pip install fastapi uvicorn diffusers accelerate transformers sentencepiece
    pip install torch torchvision
"""

import asyncio
import functools
import io
import sys
from pathlib import Path

# Make the parent evaluation/ directory importable so logger.py can be found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from diffusers import StableDiffusionXLPipeline
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from logger import get_logger, log_section, log_timer

logger = get_logger(__name__)


app = FastAPI(title="SDXL Base 1.0 Generation Server")

_generation_lock = asyncio.Lock()
app.state.pipe: StableDiffusionXLPipeline | None = None
app.state.is_ready = False


class GenerateRequest(BaseModel):
    prompt: str
    seed: int
    width: int = 1024
    height: int = 1024
    negative_prompt: str = ""
    num_inference_steps: int = 30
    guidance_scale: float = 5.0


@app.on_event("startup")
async def load_model() -> None:
    log_section(logger, "Stable Diffusion XL Base 1.0  ·  server startup")
    logger.info("Loading stabilityai/stable-diffusion-xl-base-1.0 with device_map='balanced' ...")

    app.state.pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        device_map="balanced",
    )

    if hasattr(app.state.pipe, "enable_vae_tiling"):
        app.state.pipe.enable_vae_tiling()
        logger.info("VAE tiling enabled.")
    if hasattr(app.state.pipe, "enable_vae_slicing"):
        app.state.pipe.enable_vae_slicing()
        logger.info("VAE slicing enabled.")

    logger.info("Model loaded. GPU memory snapshot:")
    for i in range(torch.cuda.device_count()):
        used = torch.cuda.memory_allocated(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        logger.info("  cuda:%d  %.1f / %.1f GB used", i, used, total)

    with log_timer(logger, "SDXL warmup inference (512×512, 5 steps)"):
        _run_blocking_generate(
            prompt="a simple warmup image",
            seed=0,
            width=512,
            height=512,
            negative_prompt="",
            num_inference_steps=5,
            guidance_scale=5.0,
        )

    app.state.is_ready = True
    logger.info("Warmup complete — server is ready.")


def _run_blocking_generate(
    prompt: str,
    seed: int,
    width: int,
    height: int,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
) -> bytes:
    pipe = app.state.pipe
    if pipe is None:
        raise RuntimeError("Model pipeline is not loaded.")

    generator = torch.Generator("cpu").manual_seed(seed)
    neg = negative_prompt if negative_prompt else None

    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            negative_prompt=neg,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

    image = result.images[0]
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.read()


@app.get("/health")
async def health() -> dict:
    if not app.state.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model is still loading — poll /health until 200.",
        )
    return {
        "status": "ready",
        "model": "stabilityai/stable-diffusion-xl-base-1.0",
        "default_resolution": [1024, 1024],
    }


@app.post("/generate")
async def generate(request: GenerateRequest) -> Response:
    if not app.state.is_ready:
        raise HTTPException(status_code=503, detail="Model not ready yet.")

    async with _generation_lock:
        loop = asyncio.get_running_loop()
        log_label = (
            f"SDXL generate  seed={request.seed}  "
            f"{request.width}×{request.height}  "
            f"steps={request.num_inference_steps}  cfg={request.guidance_scale}  "
            f"prompt={request.prompt[:60]!r}"
        )
        logger.info("▶ %s", log_label)
        try:
            t0 = asyncio.get_event_loop().time()
            image_bytes: bytes = await loop.run_in_executor(
                None,
                functools.partial(
                    _run_blocking_generate,
                    request.prompt,
                    request.seed,
                    request.width,
                    request.height,
                    request.negative_prompt,
                    request.num_inference_steps,
                    request.guidance_scale,
                ),
            )
            elapsed = asyncio.get_event_loop().time() - t0
            logger.info(
                "✔ %s  →  %.2fs  %s KB",
                log_label,
                elapsed,
                f"{len(image_bytes)/1024:.1f}",
            )
        except Exception as exc:
            logger.exception("✘ Generation failed: %s", log_label)
            raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    return Response(content=image_bytes, media_type="image/png")
