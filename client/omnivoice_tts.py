"""LiveKit Agents TTS for the OmniVoice server on RunPod (see ../app.py and ../server/).

OmniVoice generates a whole utterance at once and cannot stream audio as it goes, so the
server streams per sentence: during a call the agent keeps one WebSocket open, sends the
LLM's text as it arrives and gets each sentence's audio as soon as it is generated. The
server decides where sentences end, so "od 12. februára" or "izba č. 5" stay whole.
Self-contained so it can be copied into an agent project; it needs only ``livekit-agents``
and ``aiohttp``.

    tts = OmniVoiceTTS(voice="nina-sk", language="sk", speed=1.1)
    tts.prewarm()  # connects while the phone rings, then keeps the worker awake
    tts.update_options(language="en", voice="nina-en")  # takes effect from the next utterance
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from dataclasses import dataclass, replace
from typing import Any, Callable

import aiohttp
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

logger = logging.getLogger(__name__)

# The server returns 16-bit little-endian mono PCM at OmniVoice's native rate. LiveKit does
# not resample raw PCM, so this must match the server (its X-Sample-Rate header and "ready").
SAMPLE_RATE = 24000
NUM_CHANNELS = 1
# RunPod holds a new connection while it starts a worker, which can take about a minute.
CONNECT_TIMEOUT = 90.0
_RECONNECT_DELAY_MAX = 30.0
# How soon an old connection that was busy is checked again for replacement.
_REPLACE_RETRY = 5.0


@dataclass
class _TTSOptions:
    voice: str | None
    language: str | None
    speed: float | None
    instruct: str | None


class OmniVoiceTTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str | None = None,
        language: str | None = None,
        speed: float | None = None,
        instruct: str | None = None,
        endpoint_id: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        keepalive_interval: float | None = 10.0,
        max_connection_age: float | None = 240.0,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Args:
            voice: Preset name from the server's ``voices/`` directory (``GET /v1/voices``).
                None uses the server's DEFAULT_VOICE.
            language: OmniVoice language id such as "sk", "cs" or "en". None lets the model
                infer it from the text (slightly lower quality).
            speed: Speaking rate in the range [0.5, 2.0]; None is the model's own estimate.
            instruct: Voice-design attributes ("female, young adult"). Only ``synthesize()``
                sends it: without a voice the speaker changes on every sentence.
            endpoint_id: RunPod endpoint id, defaults to $RUNPOD_ENDPOINT_ID.
            api_key: RunPod API key, defaults to $RUNPOD_API_KEY.
            base_url: Server URL to use instead of the RunPod endpoint (local testing).
            timeout: Seconds allowed for the server to answer once it has all the text.
            keepalive_interval: Seconds between pings on the open connection, which keep the
                RunPod worker and the connection alive between utterances. None disables them.
            max_connection_age: Seconds after which an idle connection is replaced, ahead of
                any per-request limit of RunPod's load balancer. None keeps it as long as it lasts.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        if base_url is None:
            endpoint_id = endpoint_id or os.environ.get("RUNPOD_ENDPOINT_ID")
            if not endpoint_id:
                raise ValueError("Pass endpoint_id or base_url, or set RUNPOD_ENDPOINT_ID")
            base_url = f"https://{endpoint_id}.api.runpod.ai"
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        self._timeout = timeout
        self._keepalive_interval = keepalive_interval
        self._max_connection_age = max_connection_age
        self._opts = _TTSOptions(voice=voice, language=language, speed=speed, instruct=instruct)
        self._session = http_session
        self._connection: _Connection | None = None
        self._connect_lock = asyncio.Lock()
        self._keeper: asyncio.Task[None] | None = None

    @property
    def model(self) -> str:
        return "omnivoice"

    @property
    def provider(self) -> str:
        return "OmniVoice"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def _stream_url(self) -> str:
        scheme, rest = self._base_url.split("://", 1)
        return f"{'wss' if scheme == 'https' else 'ws'}://{rest}/v1/stream"

    def update_options(
        self,
        *,
        voice: NotGivenOr[str | None] = NOT_GIVEN,
        language: NotGivenOr[str | None] = NOT_GIVEN,
        speed: NotGivenOr[float | None] = NOT_GIVEN,
        instruct: NotGivenOr[str | None] = NOT_GIVEN,
    ) -> None:
        """Applies from the next utterance; one already being spoken keeps its options."""
        if is_given(voice):
            self._opts.voice = voice
        if is_given(language):
            self._opts.language = language
        if is_given(speed):
            self._opts.speed = speed
        if is_given(instruct):
            self._opts.instruct = instruct

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        return SynthesizeStream(tts=self, conn_options=conn_options)

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        """One HTTP request for the whole text, for scripts; calls use ``stream()``."""
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def prewarm(self) -> None:
        """Opens the connection in the background, waiting for a worker to start if none is
        running, then keeps it open and the worker awake until ``aclose()``."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return  # no event loop yet, e.g. built outside the job's entrypoint
        if self._keeper is None or self._keeper.done():
            self._keeper = asyncio.create_task(self._keep_connected(), name="omnivoice-tts-keeper")

    async def _connect(self) -> _Connection:
        """The open connection, or a new one."""
        async with self._connect_lock:
            if self._connection is None or self._connection.closed:
                self._connection = await self._open_connection()
            return self._connection

    async def _open_connection(self) -> _Connection:
        started = time.perf_counter()
        connection = await _Connection.open(
            self._ensure_session(),
            self._stream_url(),
            headers=self._headers(),
            keepalive_interval=self._keepalive_interval,
            timeout=CONNECT_TIMEOUT,
        )
        ready = connection.ready
        logger.info(
            "OmniVoice ready in %.0f ms (gpu: %s, accelerated: %s, num_step: %s)",
            (time.perf_counter() - started) * 1000,
            ready.get("gpu"), ready.get("accelerated"), ready.get("num_step"),
        )
        return connection

    async def _keep_connected(self) -> None:
        delay = 1.0
        while True:
            try:
                connection = await self._connect()
                delay = 1.0
                max_age = self._max_connection_age
                wait = None if max_age is None else max(max_age - connection.age, _REPLACE_RETRY)
                if await connection.wait_closed(wait):
                    logger.warning("OmniVoice connection closed (code %s), reconnecting", connection.close_code)
                    continue
                if connection.idle:
                    await self._replace(connection)
            except Exception as e:
                logger.warning("OmniVoice connection failed (%s), retrying in %.0f s", e, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, _RECONNECT_DELAY_MAX)

    async def _replace(self, old: _Connection) -> None:
        """Opens the next connection before retiring the old one, so no utterance waits."""
        replacement = await self._open_connection()
        async with self._connect_lock:
            self._connection = replacement
        old.retire()

    async def aclose(self) -> None:
        if self._keeper is not None:
            await utils.aio.cancel_and_wait(self._keeper)
            self._keeper = None
        if self._connection is not None:
            await self._connection.close()
            self._connection = None


class _Connection:
    """One WebSocket to the server; messages are routed to their stream by ``stream_id``."""

    def __init__(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        ready: dict[str, Any],
        keepalive_interval: float | None,
    ) -> None:
        self._ws = ws
        self.ready = ready
        self._opened_at = time.monotonic()
        self._streams: dict[str, asyncio.Queue[dict[str, Any] | None]] = {}
        self._closed = asyncio.Event()
        self._retiring = False
        self._tasks: set[asyncio.Task[None]] = set()
        self._reader = asyncio.create_task(self._read(), name="omnivoice-tts-reader")
        self._pinger = (
            asyncio.create_task(self._ping(keepalive_interval), name="omnivoice-tts-ping")
            if keepalive_interval
            else None
        )

    @classmethod
    async def open(
        cls,
        session: aiohttp.ClientSession,
        url: str,
        *,
        headers: dict[str, str],
        keepalive_interval: float | None,
        timeout: float,
    ) -> _Connection:
        try:
            ws = await asyncio.wait_for(session.ws_connect(url, headers=headers), timeout)
        except asyncio.TimeoutError:
            raise APITimeoutError("OmniVoice did not accept the connection in time") from None
        except aiohttp.WSServerHandshakeError as e:
            raise APIStatusError(
                f"OmniVoice refused the connection: {e.status} {e.message}", status_code=e.status
            ) from e
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"OmniVoice connection failed: {e}") from e
        try:
            msg = await ws.receive(timeout=timeout)
        except asyncio.TimeoutError:
            await ws.close()
            raise APITimeoutError("OmniVoice did not get ready in time") from None
        ready = json.loads(msg.data) if msg.type == aiohttp.WSMsgType.TEXT else {}
        if ready.get("type") != "ready":
            await ws.close()
            raise APIConnectionError(f"OmniVoice closed the connection before it was ready ({ws.close_code})")
        if ready.get("sample_rate") != SAMPLE_RATE:
            await ws.close()
            raise APIConnectionError(f"OmniVoice speaks at {ready.get('sample_rate')} Hz, not {SAMPLE_RATE}")
        return cls(ws, ready, keepalive_interval)

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    @property
    def close_code(self) -> int | None:
        return self._ws.close_code

    @property
    def idle(self) -> bool:
        return not self._streams

    @property
    def age(self) -> float:
        return time.monotonic() - self._opened_at

    def register(self, stream_id: str) -> asyncio.Queue[dict[str, Any] | None]:
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        if self.closed:
            queue.put_nowait(None)
        self._streams[stream_id] = queue
        return queue

    def unregister(self, stream_id: str) -> None:
        self._streams.pop(stream_id, None)
        if self._retiring and self.idle:
            self._spawn(self.close())

    def retire(self) -> None:
        """Closes the connection once its last stream is done; new streams use another one."""
        self._retiring = True
        if self.idle:
            self._spawn(self.close())

    def cancel(self, stream_id: str) -> None:
        """Stops a stream the server is still speaking (the guest interrupted)."""
        if not self.closed:
            self._spawn(self._send_quietly({"type": "cancel", "stream_id": stream_id}))
        self.unregister(stream_id)

    def _spawn(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def send(self, message: dict[str, Any]) -> None:
        if self.closed:
            raise APIConnectionError("OmniVoice connection is closed")
        try:
            await self._ws.send_str(json.dumps(message))
        except (aiohttp.ClientError, ConnectionError, RuntimeError) as e:
            raise APIConnectionError(f"OmniVoice connection failed: {e}") from e

    async def _send_quietly(self, message: dict[str, Any]) -> None:
        try:
            await self.send(message)
        except APIConnectionError:
            pass

    async def wait_closed(self, timeout: float | None) -> bool:
        """True once the connection closed; False if it is still open after ``timeout``."""
        try:
            await asyncio.wait_for(self._closed.wait(), timeout)
        except asyncio.TimeoutError:
            return False
        return True

    async def close(self) -> None:
        if self._pinger is not None:
            await utils.aio.cancel_and_wait(self._pinger)
        await self._ws.close()
        await self._reader

    async def _read(self) -> None:
        try:
            async for msg in self._ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                message = json.loads(msg.data)
                queue = self._streams.get(message.get("stream_id"))
                if queue is not None:
                    queue.put_nowait(message)
                elif message.get("type") == "error":
                    logger.warning("OmniVoice error: %s", message.get("message"))
        except Exception as e:
            logger.warning("OmniVoice connection failed while reading: %s", e)
        finally:
            self._closed.set()
            for queue in self._streams.values():
                queue.put_nowait(None)

    async def _ping(self, interval: float) -> None:
        while not self.closed:
            await asyncio.sleep(interval)
            await self._send_quietly({"type": "ping"})


class _Utterance:
    """One utterance on the server: its text so far and how much of it has been voiced, so
    a dropped connection resumes on a new one after the last sentence received."""

    def __init__(self, tts: OmniVoiceTTS, opts: _TTSOptions, on_text_sent: Callable[[], None]) -> None:
        self._tts = tts
        self._opts = opts
        self._on_text_sent = on_text_sent
        self._text = ""
        self._voiced = 0
        self._base = 0
        self._ended = False
        self._lock = asyncio.Lock()
        self._connection: _Connection | None = None
        self._stream_id = ""
        self._queue: asyncio.Queue[dict[str, Any] | None] | None = None
        self._finished = False
        self._last_progress = time.monotonic()

    async def open(self) -> None:
        """Starts the utterance on the server and sends the text not voiced yet."""
        async with self._lock:
            self._connection = connection = await self._tts._connect()
            self._stream_id = utils.shortuuid()
            self._queue = connection.register(self._stream_id)
            self._base = self._voiced
            self._last_progress = time.monotonic()
            await connection.send(
                {
                    "type": "start",
                    "stream_id": self._stream_id,
                    "voice": self._opts.voice,
                    "language": self._opts.language,
                    "speed": self._opts.speed,
                }
            )
            rest = self._text[self._base:]
            if rest:
                await connection.send({"type": "text", "stream_id": self._stream_id, "text": rest})
            if self._ended:
                await connection.send({"type": "end", "stream_id": self._stream_id})

    async def push(self, text: str) -> None:
        async with self._lock:
            self._text += text
            await self._send({"type": "text", "stream_id": self._stream_id, "text": text})
            self._on_text_sent()

    async def flush(self) -> None:
        async with self._lock:
            await self._send({"type": "flush", "stream_id": self._stream_id})

    async def end(self) -> None:
        async with self._lock:
            self._ended = True
            self._last_progress = time.monotonic()
            await self._send({"type": "end", "stream_id": self._stream_id})

    async def _send(self, message: dict[str, Any]) -> None:
        # Text a dropped connection did not take is sent again when the utterance resumes.
        if self._connection is not None:
            try:
                await self._connection.send(message)
            except APIConnectionError:
                pass

    async def speak(self, emitter: tts.AudioEmitter, *, max_resumes: int) -> None:
        """Pushes the audio of every sentence until the server is done with the utterance."""
        resumes = 0
        await self.open()
        while True:
            try:
                message = await self._next_message()
            except APIConnectionError:
                # Before any audio, LiveKit retries the whole utterance itself.
                if self._voiced == 0 or resumes >= max_resumes:
                    raise
                resumes += 1
                logger.warning(
                    "OmniVoice connection lost mid-utterance, resuming after character %d of %d",
                    self._voiced, len(self._text),
                )
                await self.open()
                continue
            kind = message.get("type")
            if kind == "audio":
                emitter.push(base64.b64decode(message["audio"]))
                self._voiced = self._base + int(message["end"])
            elif kind == "done":
                self._finish()
                return
            elif kind == "error":
                self._finish()
                raise APIStatusError(
                    f"OmniVoice: {message.get('message')}",
                    status_code=int(message.get("status") or 500),
                    body=message,
                )

    async def _next_message(self) -> dict[str, Any]:
        """The next message of the utterance; once all its text is sent, the server must keep
        answering within the TTS timeout."""
        assert self._queue is not None
        while True:
            try:
                message = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if self._ended and time.monotonic() - self._last_progress > self._tts._timeout:
                    raise APITimeoutError("OmniVoice did not answer in time") from None
                continue
            if message is None:
                raise APIConnectionError("OmniVoice connection closed")
            self._last_progress = time.monotonic()
            return message

    def _finish(self) -> None:
        self._finished = True
        if self._connection is not None:
            self._connection.unregister(self._stream_id)

    def cancel(self) -> None:
        """Stops the utterance on the server unless it is already done."""
        if not self._finished and self._connection is not None:
            self._finished = True
            self._connection.cancel(self._stream_id)


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: OmniVoiceTTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: OmniVoiceTTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=utils.shortuuid())
        utterance = _Utterance(self._tts, self._opts, on_text_sent=self._mark_started)
        forward = asyncio.create_task(self._forward_input(utterance))
        try:
            await utterance.speak(output_emitter, max_resumes=self._conn_options.max_retry)
        finally:
            await utils.aio.cancel_and_wait(forward)
            utterance.cancel()
        output_emitter.end_segment()

    async def _forward_input(self, utterance: _Utterance) -> None:
        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                await utterance.flush()
            else:
                await utterance.push(data)
        await utterance.end()


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: OmniVoiceTTS, input_text: str, conn_options: APIConnectOptions):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: OmniVoiceTTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        payload: dict[str, object] = {"text": self.input_text, "response_format": "pcm"}
        for name in ("voice", "language", "speed", "instruct"):
            value = getattr(self._opts, name)
            if value is not None:
                payload[name] = value

        try:
            async with self._tts._ensure_session().post(
                f"{self._tts._base_url}/v1/synthesize",
                headers=self._tts._headers(),
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    total=self._tts._timeout, sock_connect=self._conn_options.timeout
                ),
            ) as resp:
                if not resp.ok:
                    body = await resp.text()
                    raise APIStatusError(
                        f"OmniVoice returned {resp.status}: {body}",
                        status_code=resp.status,
                        body=body,
                    )
                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=SAMPLE_RATE,
                    num_channels=NUM_CHANNELS,
                    mime_type="audio/pcm",
                )
                async for data in resp.content.iter_chunked(4096):
                    output_emitter.push(data)
                output_emitter.flush()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"OmniVoice request failed: {e}") from e
