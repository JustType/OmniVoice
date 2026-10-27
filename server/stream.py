"""One WebSocket per call: streams of text in, one audio message per spoken segment out.

Client -> server (JSON):
    {"type": "start", "stream_id", "voice", "language", "speed"}   opens a stream (one utterance)
    {"type": "text", "stream_id", "text"}                          more of its text
    {"type": "flush", "stream_id"}                                 speak what is buffered now
    {"type": "end", "stream_id"}                                   no more text
    {"type": "cancel", "stream_id"}                                drop it (the guest interrupted)
    {"type": "ping"}

Server -> client (JSON):
    {"type": "ready", "sample_rate", "voices", "num_step", "accelerated", "gpu"}
    {"type": "audio", "stream_id", "seq", "text", "end", "audio", "queue_ms", "gen_ms"}
        audio is base64 16-bit little-endian mono PCM; end is the offset in the stream's
        text just past this segment, so a client that lost the connection can resume there
    {"type": "done", "stream_id"}
    {"type": "error", "stream_id", "status", "message"}
    {"type": "pong"}

A stream's first sentence is generated alone, as soon as it is complete, for the shortest
time to first audio. Whatever is complete by the time the GPU is free again is generated
together, up to MAX_SEGMENT_CHARS, which costs less than one generation per sentence.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from starlette.websockets import WebSocket, WebSocketDisconnect

from server.engine import Engine, EngineError, Voice
from server.segmenter import Segment, Segmenter

logger = logging.getLogger("omnivoice-server")

MAX_SEGMENT_CHARS = 300
# RFC 6455 "try again later": the model is not loaded yet.
CLOSE_NOT_READY = 1013


@dataclass
class _Stream:
    id: str
    voice: Voice
    segmenter: Segmenter = field(default_factory=Segmenter)
    pending: deque[Segment] = field(default_factory=deque)
    ended: bool = False
    started: bool = False
    generating: bool = False
    seq: int = 0

    @property
    def finished(self) -> bool:
        return self.ended and not self.pending and not self.generating


@dataclass(frozen=True)
class _Job:
    stream: _Stream
    text: str
    end: int


class StreamSession:
    def __init__(self, websocket: WebSocket, engine: Engine) -> None:
        self._ws = websocket
        self._engine = engine
        self._streams: dict[str, _Stream] = {}
        self._wake = asyncio.Event()
        self._send_lock = asyncio.Lock()
        self._segments = 0

    async def run(self) -> None:
        await self._ws.accept()
        if self._engine.status != "ready":
            await self._ws.close(code=CLOSE_NOT_READY, reason="model not ready")
            return
        opened = time.monotonic()
        await self._send({"type": "ready", **self._engine.info()})
        scheduler = asyncio.create_task(self._schedule(), name="omnivoice-stream-scheduler")
        try:
            while True:
                message = await self._ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("text") is None:
                    await self._error(None, 400, "Only text (JSON) messages are accepted.")
                    continue
                await self._handle(message["text"])
        except WebSocketDisconnect:
            pass
        finally:
            self._streams.clear()
            scheduler.cancel()
            await asyncio.gather(scheduler, return_exceptions=True)
            logger.info(
                "stream connection closed after %.0f s, %d segments",
                time.monotonic() - opened, self._segments,
            )

    async def _handle(self, raw: str) -> None:
        try:
            message = json.loads(raw)
            kind = message["type"]
        except (ValueError, KeyError, TypeError):
            await self._error(None, 400, "Messages must be JSON objects with a 'type'.")
            return
        if kind == "ping":
            await self._send({"type": "pong"})
            return
        stream_id = message.get("stream_id")
        if not isinstance(stream_id, str) or not stream_id:
            await self._error(None, 400, f"'{kind}' needs a 'stream_id'.")
            return
        if kind == "start":
            await self._start(stream_id, message)
            return
        stream = self._streams.get(stream_id)
        if stream is None:
            return  # cancelled, finished or failed already
        if kind == "text":
            text = message.get("text")
            if isinstance(text, str):
                self._queue(stream, stream.segmenter.push(text))
        elif kind == "flush":
            self._queue(stream, stream.segmenter.flush())
        elif kind == "end":
            stream.ended = True
            self._queue(stream, stream.segmenter.flush())
            await self._finish_if_done(stream)
        elif kind == "cancel":
            del self._streams[stream_id]
        else:
            await self._error(stream_id, 400, f"Unknown message type '{kind}'.")

    async def _start(self, stream_id: str, message: dict[str, Any]) -> None:
        if stream_id in self._streams:
            await self._error(stream_id, 400, f"Stream '{stream_id}' is already open.")
            return
        try:
            voice = self._engine.voice(
                voice=message.get("voice"),
                language=message.get("language"),
                speed=message.get("speed"),
            )
        except EngineError as e:
            await self._error(stream_id, e.status, e.message)
            return
        self._streams[stream_id] = _Stream(id=stream_id, voice=voice)

    def _queue(self, stream: _Stream, segments: list[Segment]) -> None:
        if segments:
            stream.pending.extend(segments)
            self._wake.set()

    def _next_job(self) -> Optional[_Job]:
        for stream in self._streams.values():
            if not stream.pending:
                continue
            first = stream.pending.popleft()
            parts, end = [first.text], first.end
            if stream.started:
                size = len(first.text)
                while stream.pending and size + 1 + len(stream.pending[0].text) <= MAX_SEGMENT_CHARS:
                    segment = stream.pending.popleft()
                    parts.append(segment.text)
                    size += 1 + len(segment.text)
                    end = segment.end
            stream.started = True
            return _Job(stream=stream, text=" ".join(parts), end=end)
        return None

    async def _schedule(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            while (job := self._next_job()) is not None:
                await self._generate(job)

    async def _generate(self, job: _Job) -> None:
        stream = job.stream
        stream.generating = True
        try:
            generation = await self._engine.generate(job.text, stream.voice, source="ws")
        except EngineError as e:
            if self._streams.pop(stream.id, None) is not None:
                await self._error(stream.id, e.status, e.message)
            return
        finally:
            stream.generating = False
        if self._streams.get(stream.id) is not stream:
            return  # cancelled while it was generating
        await self._send(
            {
                "type": "audio",
                "stream_id": stream.id,
                "seq": stream.seq,
                "text": job.text,
                "end": job.end,
                "audio": base64.b64encode(generation.pcm16()).decode("ascii"),
                "queue_ms": round(generation.queue_ms),
                "gen_ms": round(generation.gen_ms),
            }
        )
        stream.seq += 1
        self._segments += 1
        await self._finish_if_done(stream)

    async def _finish_if_done(self, stream: _Stream) -> None:
        if stream.finished and self._streams.pop(stream.id, None) is not None:
            await self._send({"type": "done", "stream_id": stream.id})

    async def _error(self, stream_id: Optional[str], status: int, message: str) -> None:
        await self._send({"type": "error", "stream_id": stream_id, "status": status, "message": message})

    async def _send(self, message: dict[str, Any]) -> None:
        async with self._send_lock:
            await self._ws.send_text(json.dumps(message))
