"""OmniVoice TTS server for a RunPod Serverless load-balancing endpoint.

OmniVoice is non-autoregressive: it generates a whole utterance at once and cannot stream.
LiveKit sends one request per sentence, so each request is generated in full before the
response starts. That way a failure is an HTTP error the client can see, never an empty 200.
"""

import io
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Response, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from omnivoice import OmniVoice, VoiceClonePrompt
from omnivoice.utils.lang_map import LANG_IDS, LANG_NAME_TO_ID
from pydantic import BaseModel, Field

MODEL_ID = os.getenv("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
VOICES_DIR = Path(os.getenv("VOICES_DIR", "/app/voices"))
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE") or None
NUM_STEP = int(os.getenv("OMNIVOICE_NUM_STEP", "32"))
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "1000"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("omnivoice-server")


@dataclass
class _State:
    status: Literal["loading", "ready", "failed"] = "loading"
    error: Optional[str] = None
    model: Optional[OmniVoice] = None
    voices: dict[str, VoiceClonePrompt] = field(default_factory=dict)
    sample_rate: int = 24000


state = _State()
# The model is not thread-safe. A thread lock (not an asyncio one) keeps the GPU serialised
# even when a request is cancelled while its generation thread is still running.
_gpu_lock = threading.Lock()


def _load_voices(model: OmniVoice) -> dict[str, VoiceClonePrompt]:
    """Every ``<name>.wav`` with a ``<name>.txt`` transcript in VOICES_DIR becomes a voice."""
    voices: dict[str, VoiceClonePrompt] = {}
    if not VOICES_DIR.is_dir():
        return voices
    for wav in sorted(VOICES_DIR.glob("*.wav")):
        transcript = wav.with_suffix(".txt")
        if not transcript.is_file():
            # Without a transcript OmniVoice would download and run Whisper to make one.
            logger.warning("Skipping voice '%s': %s is missing", wav.stem, transcript.name)
            continue
        with torch.inference_mode():
            voices[wav.stem] = model.create_voice_clone_prompt(
                ref_audio=str(wav), ref_text=transcript.read_text(encoding="utf-8").strip()
            )
        logger.info("Loaded voice '%s'", wav.stem)
    return voices


def _load_model() -> None:
    try:
        logger.info("Loading %s into VRAM (FP16)...", MODEL_ID)
        model = OmniVoice.from_pretrained(MODEL_ID, device_map="cuda:0", dtype=torch.float16)
        voices = _load_voices(model)
        if DEFAULT_VOICE and DEFAULT_VOICE not in voices:
            raise RuntimeError(
                f"DEFAULT_VOICE '{DEFAULT_VOICE}' is not in {VOICES_DIR} (found: {sorted(voices)})"
            )
        if not voices:
            logger.warning("No voices in %s: requests without 'instruct' get a random voice", VOICES_DIR)

        # The first generation compiles CUDA kernels; do it before taking traffic.
        started = time.perf_counter()
        model.generate(text="Hello.", num_step=NUM_STEP)
        logger.info("Warm-up generation took %.0f ms", (time.perf_counter() - started) * 1000)

        state.model, state.voices = model, voices
        state.sample_rate = model.sampling_rate or 24000
        state.status = "ready"
        logger.info("OmniVoice ready (voices: %s, num_step: %d)", sorted(voices), NUM_STEP)
    except Exception as e:
        logger.exception("Model failed to load")
        state.error = f"{type(e).__name__}: {e}"
        state.status = "failed"


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load in the background so uvicorn binds the port at once and /ping can report 204.
    threading.Thread(target=_load_model, name="model-loader", daemon=True).start()
    yield


app = FastAPI(title="OmniVoice TTS", lifespan=lifespan)


class SynthesizeRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    language: Optional[str] = None
    instruct: Optional[str] = None
    speed: Optional[float] = Field(default=None, ge=0.5, le=2.0)
    response_format: Literal["pcm", "wav"] = "pcm"


def _require_model() -> OmniVoice:
    if state.status == "loading":
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model is still loading.")
    if state.status == "failed":
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"Model failed to load: {state.error}")
    return state.model


def _resolve_language(value: Optional[str]) -> Optional[str]:
    """OmniVoice language id ("sk", "en-US" -> "en", "Slovak" -> "sk"); None lets the
    model infer the language from the text."""
    if not value:
        return None
    code = value.strip().lower().replace("_", "-")
    if code in ("auto", "none"):
        return None
    for candidate in (code, code.split("-")[0]):
        if candidate in LANG_IDS:
            return candidate
    if code in LANG_NAME_TO_ID:
        return LANG_NAME_TO_ID[code]
    logger.warning("Unsupported language '%s', letting the model infer it from the text", value)
    return None


def _generate(model: OmniVoice, **kwargs) -> tuple[np.ndarray, float]:
    """Returns the audio and the generation time, which excludes the wait for the GPU."""
    with _gpu_lock:
        started = time.perf_counter()
        audio = model.generate(num_step=NUM_STEP, **kwargs)[0]
        elapsed = time.perf_counter() - started
    return np.asarray(audio, dtype=np.float32).reshape(-1), elapsed


@app.get("/ping")
async def ping():
    """RunPod health check: 204 while loading, 200 when ready, 503 so a broken worker is replaced."""
    if state.status == "loading":
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    if state.status == "failed":
        return JSONResponse(status_code=503, content={"status": "failed", "error": state.error})
    return {"status": "healthy"}


@app.get("/v1/voices")
async def list_voices():
    _require_model()
    return {"voices": sorted(state.voices), "default_voice": DEFAULT_VOICE, "sample_rate": state.sample_rate}


@app.post("/v1/synthesize")
async def synthesize(req: SynthesizeRequest):
    """Returns 16-bit little-endian mono PCM at 24 kHz (or a WAV file with response_format=wav)."""
    model = _require_model()
    text = req.text.strip()
    if not text:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The 'text' field cannot be empty.")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"'text' is longer than {MAX_TEXT_CHARS} characters.")

    voice = req.voice or (None if req.instruct else DEFAULT_VOICE)
    prompt = None
    if voice:
        prompt = state.voices.get(voice)
        if prompt is None:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, f"Unknown voice '{voice}'. Available: {sorted(state.voices)}"
            )
    elif not req.instruct:
        logger.warning("Request has no voice or instruct: OmniVoice picks a random voice")
    language = _resolve_language(req.language)

    started = time.perf_counter()
    try:
        audio, elapsed = await run_in_threadpool(
            _generate,
            model,
            text=text,
            language=language,
            voice_clone_prompt=prompt,
            instruct=req.instruct or None,
            speed=req.speed,
        )
    except ValueError as e:
        # OmniVoice validates instruct items and raises ValueError on unknown ones.
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from e
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, f"Inference failed: {type(e).__name__}: {e}"
        ) from e
    queued = time.perf_counter() - started - elapsed

    sample_rate = state.sample_rate
    duration = len(audio) / sample_rate
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    logger.info(
        "synthesize chars=%d voice=%s language=%s queue_ms=%.0f gen_ms=%.0f audio_s=%.2f rtf=%.3f",
        len(text), voice or ("design" if req.instruct else "auto"), language or "auto",
        queued * 1000, elapsed * 1000, duration, elapsed / duration if duration else 0.0,
    )

    headers = {"X-Sample-Rate": str(sample_rate), "X-Audio-Duration": f"{duration:.3f}"}
    if req.response_format == "wav":
        buf = io.BytesIO()
        sf.write(buf, pcm, sample_rate, format="WAV", subtype="PCM_16")
        return Response(content=buf.getvalue(), media_type="audio/wav", headers=headers)
    return Response(content=pcm.tobytes(), media_type="audio/pcm", headers=headers)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
