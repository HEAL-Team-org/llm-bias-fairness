#!/usr/bin/env python3
"""
FastAPI server for FLUX.1 [dev] text-to-image generation.

Architecture:
  - Model    : black-forest-labs/FLUX.1-dev (12B param rectified-flow transformer)
  - Strategy : Pipeline/device parallelism — a single model instance is split
               across cuda:0 and cuda:1 using device_map="balanced".
  - One request at a time (asyncio.Lock).  Both GPUs are fully occupied by
    the single model; concurrent requests would OOM immediately.
  - VAE tiling + slicing enabled to prevent OOM during the decode step at
    high resolutions.

Run on the SSH server:
    uvicorn local_models.serve_flux_dev:app --host 0.0.0.0 --port 8004 --workers 1

    NOTE: --workers MUST be 1.  See serve_flux.py for the reason.
    NOTE: Use port 8004 to avoid conflict with other model servers.

License note:
    FLUX.1 [dev] is released under the FLUX.1 [dev] Non-Commercial License.
    Commercial use requires a separate agreement with Black Forest Labs.
    See: https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md

Requirements:
    pip install fastapi uvicorn diffusers accelerate transformers sentencepiece
    pip install torch torchvision    # CUDA build matching your driver
    # HuggingFace login required (gated model):
    #   huggingface-cli login
    # or set the HF_TOKEN environment variable.

FLUX.1 [dev] specifics:
    - No negative_prompt support.  FLUX uses a single-stream architecture that
      does not use classifier-free guidance in the traditional sense; the model
      was guidance-distilled so a separate negative conditioning pass is not
      applicable.
    - max_sequence_length controls the T5 token budget (default 512; reduce to
      256 to save VRAM at the cost of shorter effective prompts).
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
from diffusers import FluxPipeline
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging pipeline
# ---------------------------------------------------------------------------
from logger import get_logger, log_section, log_timer

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# FLUX.1 [dev] common resolutions.
# The model was trained at 1024×1024 and nearby aspect ratios.
# Using very different total pixel counts may reduce quality.
# ---------------------------------------------------------------------------
NATIVE_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1:1":  (1024, 1024),
    "16:9": (1360,  768),
    "9:16": ( 768, 1360),
    "4:3":  (1152,  896),
    "3:4":  ( 896, 1152),
    "3:2":  (1216,  832),
    "2:3":  ( 832, 1216),
}

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
app = FastAPI(title="FLUX.1-dev Generation Server")

pipe = None               # FluxPipeline instance; set during startup
_generation_lock = asyncio.Lock()   # serialises all generation requests
_is_ready: bool = False             # True only after warmup completes


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str
    seed: int
    width: int = 1024                 # default: native 1:1 square
    height: int = 1024
    num_inference_steps: int = 50     # model-card default
    guidance_scale: float = 3.5       # model-card default for FLUX.1-dev
    max_sequence_length: int = 512    # T5 token budget; reduce to 256 to save VRAM


# ---------------------------------------------------------------------------
# Model loading & warmup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def load_model() -> None:
    """
    Load FLUX.1 [dev] split across both GPUs, apply VAE memory
    optimisations, then run a warmup pass to pre-compile CUDA kernels.
    """
    global pipe, _is_ready

    log_section(logger, "FLUX.1 [dev]  ·  server startup")
    logger.info(
        "Loading black-forest-labs/FLUX.1-dev with device_map='balanced' ..."
    )
    logger.info(
        "device_map='balanced' requires the `accelerate` library and distributes "
        "the model's layers evenly across cuda:0 and cuda:1."
    )

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,   # bfloat16 required; float32 OOMs both GPUs
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
    with log_timer(logger, "FLUX.1-dev warmup inference (5 steps, 512×512)"):
        _run_blocking_generate(
            prompt="a simple warmup image",
            seed=0,
            width=512,
            height=512,
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
        multiple CUDA devices.  A CPU generator is device-agnostic and
        works correctly with any device_map configuration.

    Negative prompt note:
        FLUX.1 [dev] does not support negative prompts — the model is
        guidance-distilled and lacks a separate unconditional conditioning
        branch.  The parameter is intentionally absent from this function.
    """
    generator = torch.Generator("cpu").manual_seed(seed)

    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
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
        "model": "black-forest-labs/FLUX.1-dev",
        "native_resolutions": NATIVE_RESOLUTIONS,
    }


@app.post("/generate")
async def generate(request: GenerateRequest) -> Response:
    """
    Generate one image and return it as raw PNG bytes.

    Concurrency design:
      - asyncio.Lock() → only one request runs at a time.
      - run_in_executor() → the blocking PyTorch call is offloaded to a
        thread-pool worker so the event loop stays responsive (e.g. for
        /health polls) during the ~20-60s generation window.

    Why sequential only?
        Both GPUs are fully occupied by this single 12B model.  A second
        concurrent generation call would immediately OOM.
    """
    if not _is_ready:
        raise HTTPException(status_code=503, detail="Model not ready yet.")

    native_sizes = set(NATIVE_RESOLUTIONS.values())
    if (request.width, request.height) not in native_sizes:
        logger.warning(
            f"Requested {request.width}×{request.height} is not a standard "
            f"FLUX.1-dev resolution.  Quality may be reduced. "
            f"Recommended sizes: {list(NATIVE_RESOLUTIONS.values())}"
        )

    async with _generation_lock:
        loop = asyncio.get_running_loop()
        log_label = (
            f"FLUX.1-dev generate  seed={request.seed}  "
            f"{request.width}×{request.height}  "
            f"steps={request.num_inference_steps}  cfg={request.guidance_scale}  "
            f"seq_len={request.max_sequence_length}  "
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
