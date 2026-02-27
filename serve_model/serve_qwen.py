#!/usr/bin/env python3
"""
FastAPI server for Qwen-Image-2512 text-to-image generation.

Architecture:
  - Model  : Qwen/Qwen-Image-2512 (20B params, QwenImagePipeline via diffusers)
  - Strategy: Pipeline/device parallelism — a single model instance is split
              across cuda:0 and cuda:1 using device_map="balanced".
  - One request at a time (asyncio.Lock).  Both GPUs are fully occupied by
    the single model; concurrent requests would OOM immediately.
  - VAE tiling + slicing enabled to prevent OOM during the decode step at
    the model's native high resolutions (1328×1328, 1664×928, etc.).

Run on the SSH server:
    pip install git+https://github.com/huggingface/diffusers   # latest diffusers required
    uvicorn local_models.serve_qwen:app --host 0.0.0.0 --port 8001 --workers 1

    NOTE: --workers MUST be 1.  See serve_flux.py for the reason.
    NOTE: Use port 8001 to avoid conflict with serve_flux.py (port 8000).

Requirements:
    pip install fastapi uvicorn diffusers accelerate transformers sentencepiece
    pip install torch torchvision    # CUDA build matching your driver
"""

import asyncio
import functools
import io
import logging
import os
import sys
from pathlib import Path

# Reduce CUDA allocator fragmentation — PyTorch's own recommendation when
# "reserved but unallocated" memory is large.  Must be set before torch loads.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# Make the parent evaluation/ directory importable so logger.py can be found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from diffusers import DiffusionPipeline
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging pipeline
# ---------------------------------------------------------------------------
from logger import get_logger, log_section, log_subsection, log_timer

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Native Qwen-Image-2512 aspect-ratio resolutions (from the model card).
# The model was trained at these exact sizes — using arbitrary sizes may
# degrade quality.  The client sends width/height; we leave validation
# to the caller but log a warning if the size looks non-standard.
# ---------------------------------------------------------------------------
NATIVE_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1:1":  (1328, 1328),
    "16:9": (1664,  928),
    "9:16": ( 928, 1664),
    "4:3":  (1472, 1104),
    "3:4":  (1104, 1472),
    "3:2":  (1584, 1056),
    "2:3":  (1056, 1584),
}

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
app = FastAPI(title="Qwen-Image-2512 Generation Server")

pipe = None              # DiffusionPipeline instance; set during startup
_generation_lock = asyncio.Lock()   # serialises all generation requests
_is_ready: bool = False             # True only after warmup completes


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str
    seed: int
    width: int = 1328                # default: native 1:1 square
    height: int = 1328
    negative_prompt: str = ""        # optional; empty string = no negative prompt
    num_inference_steps: int = 50    # 50 is the model-card default; reduce for speed
    true_cfg_scale: float = 4.0      # Classifier-Free Guidance scale (model default)


# ---------------------------------------------------------------------------
# Model loading & warmup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def load_model() -> None:
    """
    Load Qwen-Image-2512 split across both GPUs, apply VAE memory
    optimisations, then run a warmup pass to pre-compile CUDA kernels.
    """
    global pipe, _is_ready

    log_section(logger, "Qwen-Image-2512  ·  server startup")
    logger.info("Loading Qwen/Qwen-Image-2512 with device_map='balanced' ...")
    logger.info(
        "device_map='balanced' requires the `accelerate` library and distributes "
        "the model's layers evenly across cuda:0 and cuda:1."
    )

    # DiffusionPipeline.from_pretrained will detect and use QwenImagePipeline
    # automatically based on the repo's model_index.json.
    # device_map="balanced" requires accelerate >= 0.26.
    pipe = DiffusionPipeline.from_pretrained(
        "Qwen/Qwen-Image-2512",
        torch_dtype=torch.bfloat16,    # bfloat16 is mandatory; float32 OOMs both GPUs
        device_map="balanced",          # distributes transformer + VAE across GPUs
    )

    logger.info("Model loaded.  GPU memory snapshot:")
    for i in range(torch.cuda.device_count()):
        used  = torch.cuda.memory_allocated(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        logger.info(f"  cuda:{i}  {used:.1f} / {total:.1f} GB used")

    # --- VAE memory optimisations -------------------------------------------
    # At native resolutions (e.g. 1664×928) the VAE decoder processes very
    # large latent tensors.  Without these two flags the decode step will OOM
    # even with 80 GB combined VRAM.
    #
    # enable_vae_tiling():  decodes the latent in spatial tiles instead of all
    #                        at once.  Dramatically reduces peak VAE VRAM.
    # enable_vae_slicing(): processes the batch one image at a time inside the
    #                        VAE.  Reduces peak VRAM for the batch dimension.
    #
    # Both are no-ops if the VAE component does not implement the method, so
    # the hasattr guard below means this code is forward-compatible.
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

    # Slice attention heads one-at-a-time to cut peak transformer VRAM.
    # This is the cheapest memory saver for the OOM during the denoising loop.
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing(1)   # 1 = maximally conservative slicing
        logger.info("Attention slicing enabled (slice_size=1).")
    else:
        logger.warning("enable_attention_slicing() not found on this pipeline version.")

    # Warmup: 5-step pass at a small resolution to compile CUDA kernels.
    # Using a low step count and small size keeps the startup delay short.
    with log_timer(logger, "Qwen warmup inference (5 steps, 512×512)"):
        _run_blocking_generate(
            prompt="a simple warmup image",
            seed=0,
            width=512,
            height=512,
            negative_prompt="",
            num_inference_steps=5,
            true_cfg_scale=4.0,
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
    true_cfg_scale: float,
) -> bytes:
    """
    Synchronous inference function.  Returns raw PNG bytes.

    Dispatched via loop.run_in_executor() — never called directly from
    an async context, so it is allowed to block the calling thread.

    CPU generator rationale:
        When device_map="balanced" is active, the model's layers live on
        multiple CUDA devices.  A CUDA-device-specific generator
        (torch.Generator("cuda:0")) would only work for layers on that device.
        A CPU generator is device-agnostic and works correctly with any
        device_map configuration.
    """
    # Build the generator on CPU for device_map compatibility.
    generator = torch.Generator("cpu").manual_seed(seed)

    # Pass negative_prompt=None when the caller sends an empty string, because
    # some pipeline implementations treat "" differently from None.
    neg = negative_prompt if negative_prompt else None

    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            negative_prompt=neg,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,   # Qwen-specific CFG parameter
            generator=generator,
        )

    image = result.images[0]

    # Free cached (but unused) VRAM blocks so the next request starts with a
    # clean allocator state.  Costs ~1ms; saves the 15 MiB that was causing OOM.
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

    The client script should poll this before sending /generate requests.
    """
    if not _is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model is still loading — poll /health until 200.",
        )
    return {
        "status": "ready",
        "model": "Qwen/Qwen-Image-2512",
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
        /health polls) during the ~30-120s generation window.

    Why sequential only?
        Both GPUs are fully occupied by this single 20B model. A second
        concurrent generation call would immediately OOM.
    """
    if not _is_ready:
        raise HTTPException(status_code=503, detail="Model not ready yet.")

    # Warn about non-native resolutions (quality may degrade).
    native_sizes = set(NATIVE_RESOLUTIONS.values())
    if (request.width, request.height) not in native_sizes:
        logger.warning(
            f"Requested {request.width}×{request.height} is not a native "
            f"Qwen-Image-2512 resolution.  Quality may be reduced. "
            f"Native sizes: {list(NATIVE_RESOLUTIONS.values())}"
        )

    async with _generation_lock:
        loop = asyncio.get_running_loop()
        log_label = (
            f"Qwen generate  seed={request.seed}  "
            f"{request.width}×{request.height}  "
            f"steps={request.num_inference_steps}  cfg={request.true_cfg_scale}  "
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
                    request.true_cfg_scale,
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
