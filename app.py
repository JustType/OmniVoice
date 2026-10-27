"""OmniVoice TTS server for a RunPod Serverless load-balancing endpoint.

OmniVoice is non-autoregressive: it generates a whole utterance at once and cannot stream.
A call keeps one WebSocket open (/v1/stream, see server/stream.py): the agent streams the
LLM's text in, the server cuts it into sentences and sends each one's audio as soon as it
is generated. /v1/synthesize generates one text per request, for scripts and smoke tests;
its audio is only sent once generation has succeeded, so a failure is an HTTP error, never
an empty 200.
"""

import io
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Literal, Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Response, WebSocket, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from server.engine import Engine, EngineError, Settings
from server.stream import StreamSession

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

engine = Engine(Settings.from_env())


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load in the background so uvicorn binds the port at once and /ping can report 204.
    threading.Thread(target=engine.load, name="model-loader", daemon=True).start()
    yield


app = FastAPI(title="OmniVoice TTS", lifespan=lifespan)


class SynthesizeRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    language: Optional[str] = None
    instruct: Optional[str] = None
    speed: Optional[float] = Field(default=None, ge=0.5, le=2.0)
    response_format: Literal["pcm", "wav"] = "pcm"


@app.get("/ping")
async def ping():
    """RunPod health check: 204 while loading, 200 when ready, 503 so a broken worker is replaced."""
    if engine.status == "loading":
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    if engine.status == "failed":
        return JSONResponse(status_code=503, content={"status": "failed", "error": engine.error})
    return {"status": "healthy"}


@app.get("/v1/voices")
async def list_voices():
    try:
        engine.require_ready()
    except EngineError as e:
        raise HTTPException(e.status, e.message) from e
    return engine.info()


@app.post("/v1/synthesize")
async def synthesize(req: SynthesizeRequest):
    """Returns 16-bit little-endian mono PCM at 24 kHz (or a WAV file with response_format=wav)."""
    try:
        voice = engine.voice(voice=req.voice, language=req.language, speed=req.speed, instruct=req.instruct)
        generation = await engine.generate(engine.text(req.text), voice, source="http")
    except EngineError as e:
        raise HTTPException(e.status, e.message) from e

    pcm = generation.pcm16()
    headers = {
        "X-Sample-Rate": str(generation.sample_rate),
        "X-Audio-Duration": f"{generation.duration:.3f}",
        "X-Gen-Ms": f"{generation.gen_ms:.0f}",
        "X-Queue-Ms": f"{generation.queue_ms:.0f}",
    }
    if req.response_format == "wav":
        buf = io.BytesIO()
        sf.write(buf, np.frombuffer(pcm, dtype="<i2"), generation.sample_rate, format="WAV", subtype="PCM_16")
        return Response(content=buf.getvalue(), media_type="audio/wav", headers=headers)
    return Response(content=pcm, media_type="audio/pcm", headers=headers)


@app.websocket("/v1/stream")
async def stream(websocket: WebSocket):
    await StreamSession(websocket, engine).run()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
