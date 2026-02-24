#!/usr/bin/env python3
"""
FastAPI server for Tongyi-MAI/Z-Image text-to-image generation.

Architecture:
  - Model    : Tongyi-MAI/Z-Image (Single-Stream Diffusion Transformer)
  - Strategy : Pipeline/device parallelism — a single model instance is split
               across cuda:0 and cuda:1 using device_map="balanced".
  - One request at a time (asyncio.Lock).  Both GPUs are fully occupied by
    the single model; concurrent requests would OOM immediately.
  - VAE tiling + slicing enabled to prevent OOM during the decode step at
    high resolutions.

Run on the SSH server:
    pip install git+https://github.com/huggingface/diffusers   # latest diffusers required
    uvicorn local_models.serve_zimage:app --host 0.0.0.0 --port 8003 --workers 1

    NOTE: --workers MUST be 1.  See serve_flux.py for the reason.
    NOTE: Use port 8003 to avoid conflict with other model servers.

Requirements:
    pip install fastapi uvicorn diffusers accelerate transformers sentencepiece
    pip install torch torchvision    # CUDA build matching your driver
    # Latest diffusers from source is required for ZImagePipeline:
    #   pip install git+https://github.com/huggingface/diffusers
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
from diffusers import ZImagePipeline
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging pipeline
# ---------------------------------------------------------------------------
from logger import get_logger, log_section, log_timer

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Z-Image recommended resolutions.
# The model supports any aspect ratio with a total pixel area between
# 512×512 and 2048×2048.  Common portrait / landscape pairs are listed below.
# ---------------------------------------------------------------------------
NATIVE_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1:1":  (1024, 1024),
    "16:9": (1280,  720),
    "9:16": ( 720, 1280),
    "4:3":  (1152,  896),
    "3:4":  ( 896, 1152),
    "3:2":  (1216,  832),
    "2:3":  ( 832, 1216),
}

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
app = FastAPI(title="Z-Image Generation Server")

pipe = None               # ZImagePipeline instance; set during startup
_generation_lock = asyncio.Lock()   # serialises all generation requests
_is_ready: bool = False             # True only after warmup completes


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str
    seed: int
    width: int = 1024                # default: 1:1 square
    height: int = 1024
    negative_prompt: str = ""        # optional; Z-Image responds well to negatives
    num_inference_steps: int = 50    # model-card default: 28–50
    guidance_scale: float = 4.0      # model-card recommended: 3.0–5.0
    cfg_normalization: bool = False  # Z-Image-specific; False is the model default


# ---------------------------------------------------------------------------
# Model loading & warmup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def load_model() -> None:
    """
    Load Tongyi-MAI/Z-Image split across both GPUs, apply VAE memory
    optimisations, then run a warmup pass to pre-compile CUDA kernels.
    """
    global pipe, _is_ready

    log_section(logger, "Tongyi-MAI/Z-Image  ·  server startup")
    logger.info(
        "Loading Tongyi-MAI/Z-Image with device_map='balanced' ..."
    )
    logger.info(
        "device_map='balanced' requires the `accelerate` library and distributes "
        "the model's layers evenly across cuda:0 and cuda:1."
    )
    logger.info(
        "NOTE: ZImagePipeline requires the latest diffusers from source: "
        "pip install git+https://github.com/huggingface/diffusers"
    )

    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image",
        torch_dtype=torch.bfloat16,   # bfloat16 recommended
        device_map="balanced",         # distributes layers across GPUs
        low_cpu_mem_usage=True,        # required when using device_map
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
    with log_timer(logger, "Z-Image warmup inference (5 steps, 512×512)"):
        _run_blocking_generate(
            prompt="a simple warmup image",
            seed=0,
            width=512,
            height=512,
            negative_prompt="",
            num_inference_steps=5,
            guidance_scale=4.0,
            cfg_normalization=False,
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
    cfg_normalization: bool,
) -> bytes:
    """
    Synchronous inference function.  Returns raw PNG bytes.

    Dispatched via loop.run_in_executor() — never called directly from
    an async context, so it is allowed to block the calling thread.

    CPU generator rationale:
        When device_map="balanced" is active, the model's layers live on
        multiple CUDA devices.  A CPU generator is device-agnostic and
        works correctly with any device_map configuration.
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
            cfg_normalization=cfg_normalization,   # Z-Image specific parameter
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
        "model": "Tongyi-MAI/Z-Image",
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

    Resolution note:
        Z-Image supports any aspect ratio with total pixel area between
        512×512 (~262K px) and 2048×2048 (~4.2M px).  Dimensions outside
        this range may degrade quality or OOM.
    """
    if not _is_ready:
        raise HTTPException(status_code=503, detail="Model not ready yet.")

    # Warn if total pixel area is outside the supported range.
    total_pixels = request.width * request.height
    min_pixels = 512 * 512         # 262_144
    max_pixels = 2048 * 2048       # 4_194_304
    if not (min_pixels <= total_pixels <= max_pixels):
        logger.warning(
            f"Requested {request.width}×{request.height} ({total_pixels:,} px) "
            f"is outside the recommended Z-Image range "
            f"[512×512 – 2048×2048].  Quality may be reduced."
        )

    async with _generation_lock:
        loop = asyncio.get_running_loop()
        log_label = (
            f"ZImage generate  seed={request.seed}  "
            f"{request.width}×{request.height}  "
            f"steps={request.num_inference_steps}  cfg={request.guidance_scale}  "
            f"cfg_norm={request.cfg_normalization}  "
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
                    request.cfg_normalization,
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
