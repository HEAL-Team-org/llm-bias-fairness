#!/usr/bin/env python3
"""
FastAPI server for Stable Diffusion 3.5 Large text-to-image generation.

Architecture:
  - Model    : stabilityai/stable-diffusion-3.5-large (8B param MMDiT)
  - Strategy : Pipeline/device parallelism — a single model instance is split
               across cuda:0 and cuda:1 using device_map="balanced".
  - One request at a time (asyncio.Lock).  Both GPUs are fully occupied by
    the single model; concurrent requests would OOM immediately.
  - VAE tiling + slicing enabled to prevent OOM during the decode step at
    high resolutions.

Run on the SSH server:
    uvicorn local_models.serve_sd35:app --host 0.0.0.0 --port 8002 --workers 1

    NOTE: --workers MUST be 1.  See serve_flux.py for the reason.
    NOTE: Use port 8002 to avoid conflict with other model servers.

Requirements:
    pip install fastapi uvicorn diffusers accelerate transformers sentencepiece
    pip install torch torchvision    # CUDA build matching your driver
    # HuggingFace login required (gated model):
    #   huggingface-cli login
    # or set the HF_TOKEN environment variable.
"""

import asyncio
import functools
import io
import os
import sys
from pathlib import Path

# Reduce CUDA allocator fragmentation — PyTorch's own recommendation when
# "reserved but unallocated" memory is large.  Must be set before torch loads.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# Make the parent evaluation/ directory importable so logger.py can be found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from diffusers import StableDiffusion3Pipeline
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging pipeline
# ---------------------------------------------------------------------------
from logger import get_logger, log_section, log_timer

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# SD 3.5 Large recommended resolutions.
# The model was trained primarily at 1024×1024 and common aspect ratios.
# Using wildly different sizes may degrade quality.
# ---------------------------------------------------------------------------
NATIVE_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1:1":  (1024, 1024),
    "16:9": (1344,  768),
    "9:16": ( 768, 1344),
    "4:3":  (1152,  896),
    "3:4":  ( 896, 1152),
    "3:2":  (1216,  832),
    "2:3":  ( 832, 1216),
}

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
app = FastAPI(title="Stable Diffusion 3.5 Large Generation Server")

pipe = None               # StableDiffusion3Pipeline instance; set during startup
_generation_lock = asyncio.Lock()   # serialises all generation requests
_is_ready: bool = False             # True only after warmup completes


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str
    seed: int
    width: int = 1024                # default: native 1:1 square
    height: int = 1024
    negative_prompt: str = ""        # optional; empty string = no negative prompt
    num_inference_steps: int = 28    # 28 is the model-card default
    guidance_scale: float = 3.5      # CFG scale (model-card default)
    max_sequence_length: int = 512   # T5 token budget; 256 or 512


# ---------------------------------------------------------------------------
# Model loading & warmup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def load_model() -> None:
    """
    Load SD 3.5 Large split across both GPUs, apply VAE memory
    optimisations, then run a warmup pass to pre-compile CUDA kernels.
    """
    global pipe, _is_ready

    log_section(logger, "Stable Diffusion 3.5 Large  ·  server startup")
    logger.info(
        "Loading stabilityai/stable-diffusion-3.5-large with device_map='balanced' ..."
    )
    logger.info(
        "device_map='balanced' requires the `accelerate` library and distributes "
        "the model's layers evenly across cuda:0 and cuda:1."
    )

    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        torch_dtype=torch.bfloat16,   # bfloat16 recommended; float32 OOMs both GPUs
        device_map="balanced",         # distributes transformer + VAE across GPUs
    )

    logger.info("Model loaded.  GPU memory snapshot:")
    for i in range(torch.cuda.device_count()):
        used  = torch.cuda.memory_allocated(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        logger.info(f"  cuda:{i}  {used:.1f} / {total:.1f} GB used")

    # --- VAE memory optimisations -------------------------------------------
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
        logger.info("VAE tiling enabled.")
    else:
        logger.warning(
            "enable_vae_tiling() not found on this pipeline version — "
            "high-resolution decoding may OOM."
        )

    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
        logger.info("VAE slicing enabled.")
    else:
        logger.warning("enable_vae_slicing() not found on this pipeline version.")

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing(1)
        logger.info("Attention slicing enabled (slice_size=1).")
    else:
        logger.warning("enable_attention_slicing() not found on this pipeline version.")

    # Warmup: small resolution, few steps — just compiles CUDA kernels.
    with log_timer(logger, "SD 3.5 Large warmup inference (5 steps, 512×512)"):
        _run_blocking_generate(
            prompt="a simple warmup image",
            seed=0,
            width=512,
            height=512,
            negative_prompt="",
            num_inference_steps=5,
            guidance_scale=3.5,
            max_sequence_length=256,
        )
    logger.info("Warmup complete — server is ready.")
    _is_ready = True


# ---------------------------------------------------------------------------
# Core blocking inference
# ---------------------------------------------------------------------------
def _run_blocking_generate(
    prompt: str,
    seed: int,
    width: int,
    height: int,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    max_sequence_length: int,
) -> bytes:
    """
    Synchronous inference function.  Returns raw PNG bytes.

    Dispatched via loop.run_in_executor() — never called directly from
    an async context, so it is allowed to block the calling thread.

    CPU generator rationale:
        When device_map="balanced" is active, the model's layers live on
        multiple CUDA devices.  A CUDA-device-specific generator would only
        work for layers on that device.  A CPU generator is device-agnostic
        and works correctly with any device_map configuration.
    """
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
            max_sequence_length=max_sequence_length,
            generator=generator,
        )

    image = result.images[0]
    torch.cuda.empty_cache()

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.read()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict:
    """
    Health-check endpoint.  Returns 503 while the model is loading,
    200 once the warmup pass has completed successfully.
    """
    if not _is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model is still loading — poll /health until 200.",
        )
    return {
        "status": "ready",
        "model": "stabilityai/stable-diffusion-3.5-large",
        "native_resolutions": NATIVE_RESOLUTIONS,
    }


@app.post("/generate")
async def generate(request: GenerateRequest) -> Response:
    """
    Generate one image and return it as raw PNG bytes.

    Concurrency design:
      - asyncio.Lock() → only one request runs at a time.
      - run_in_executor() → the blocking PyTorch call is offloaded to a
        thread-pool worker so the event loop stays responsive during generation.
    """
    if not _is_ready:
        raise HTTPException(status_code=503, detail="Model not ready yet.")

    native_sizes = set(NATIVE_RESOLUTIONS.values())
    if (request.width, request.height) not in native_sizes:
        logger.warning(
            f"Requested {request.width}×{request.height} is not a standard "
            f"SD 3.5 Large resolution.  Quality may be reduced. "
            f"Recommended sizes: {list(NATIVE_RESOLUTIONS.values())}"
        )

    async with _generation_lock:
        loop = asyncio.get_running_loop()
        log_label = (
            f"SD35 generate  seed={request.seed}  "
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
                    request.max_sequence_length,
                ),
            )
            elapsed = asyncio.get_event_loop().time() - t0
            logger.info(
                "✔ %s  →  %.2fs  %s KB",
                log_label, elapsed, f"{len(image_bytes)/1024:.1f}"
            )
        except Exception as exc:
            logger.exception("✘ Generation failed: %s", log_label)
            raise HTTPException(
                status_code=500,
                detail=f"Generation failed: {exc}",
            ) from exc

    return Response(content=image_bytes, media_type="image/png")
