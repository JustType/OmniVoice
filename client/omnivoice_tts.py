"""LiveKit Agents TTS for the OmniVoice RunPod endpoint (see ../app.py).

OmniVoice cannot stream, so this TTS is non-streaming: an ``AgentSession`` wraps it in a
``tts.StreamAdapter`` that sends one request per sentence. Self-contained so it can be
copied into an agent project; it needs only ``livekit-agents`` and ``aiohttp``.

    tts = OmniVoiceTTS(voice="nina-sk", language="sk", speed=1.1)
    tts.update_options(language="en", voice="nina-en")  # takes effect from the next sentence
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, replace

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
# not resample raw PCM, so this must match the server (its X-Sample-Rate header).
SAMPLE_RATE = 24000
NUM_CHANNELS = 1


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
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Args:
            voice: Preset name from the server's ``voices/`` directory (``GET /v1/voices``).
                None uses the server's DEFAULT_VOICE.
            language: OmniVoice language id such as "sk", "cs" or "en". None lets the model
                infer it from the text (slightly lower quality).
            speed: Speaking rate in the range [0.5, 2.0]; None is the model's own estimate.
            instruct: Voice-design attributes ("female, young adult"). Without a voice the
                speaker changes on every sentence, so use it to design voices, not in calls.
            endpoint_id: RunPod endpoint id, defaults to $RUNPOD_ENDPOINT_ID.
            api_key: RunPod API key, defaults to $RUNPOD_API_KEY.
            base_url: Server URL to use instead of the RunPod endpoint (local testing).
            timeout: Seconds allowed for one sentence, generation included.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
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
        self._opts = _TTSOptions(voice=voice, language=language, speed=speed, instruct=instruct)
        self._session = http_session
        self._prewarm_task: asyncio.Task[None] | None = None

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

    def update_options(
        self,
        *,
        voice: NotGivenOr[str | None] = NOT_GIVEN,
        language: NotGivenOr[str | None] = NOT_GIVEN,
        speed: NotGivenOr[float | None] = NOT_GIVEN,
        instruct: NotGivenOr[str | None] = NOT_GIVEN,
    ) -> None:
        """Applies from the next sentence; a sentence already being synthesised keeps its options."""
        if is_given(voice):
            self._opts.voice = voice
        if is_given(language):
            self._opts.language = language
        if is_given(speed):
            self._opts.speed = speed
        if is_given(instruct):
            self._opts.instruct = instruct

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def prewarm(self) -> None:
        """Opens the TLS connection in the background. It does not start a GPU worker:
        keep at least one active worker on the endpoint for that."""

        async def _task() -> None:
            try:
                async with self._ensure_session().get(
                    f"{self._base_url}/ping",
                    headers=self._headers(),
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    await resp.read()
            except Exception as e:
                logger.debug("OmniVoice TTS prewarm failed", exc_info=e)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return  # no event loop yet, e.g. built outside the job's entrypoint
        if self._prewarm_task is None or self._prewarm_task.done():
            self._prewarm_task = asyncio.create_task(_task(), name="omnivoice-tts-prewarm")

    async def aclose(self) -> None:
        if self._prewarm_task is not None:
            await utils.aio.cancel_and_wait(self._prewarm_task)
            self._prewarm_task = None


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
