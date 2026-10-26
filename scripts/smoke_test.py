"""Synthesize through the LiveKit TTS class, the same path an AgentSession takes.

    export RUNPOD_ENDPOINT_ID=... RUNPOD_API_KEY=...
    python scripts/smoke_test.py --voice nina-sk --languages sk en

Writes out_<language>.wav and prints time to first audio. Needs livekit-agents and aiohttp,
e.g. run it with the agent project's virtualenv.
"""

import argparse
import asyncio
import sys
import time
import wave
from pathlib import Path

from livekit import rtc
from livekit.agents import utils

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "client"))
from omnivoice_tts import SAMPLE_RATE, OmniVoiceTTS  # noqa: E402

SENTENCES = {
    "sk": "Dobrý deň, ako vám môžem pomôcť?",
    "cs": "Dobrý den, jak vám mohu pomoci?",
    "en": "Hello, how can I help you today?",
}


async def synthesize(tts: OmniVoiceTTS, language: str, text: str) -> None:
    tts.update_options(language=language)
    started = time.perf_counter()
    first_audio = None
    frames: list[rtc.AudioFrame] = []
    async with tts.synthesize(text) as stream:
        async for event in stream:
            if first_audio is None:
                first_audio = time.perf_counter() - started
            frames.append(event.frame)
    total = time.perf_counter() - started

    if not frames:
        raise SystemExit(f"{language}: no audio received")
    frame = rtc.combine_audio_frames(frames)
    assert frame.sample_rate == SAMPLE_RATE and frame.num_channels == 1, (frame.sample_rate, frame.num_channels)
    duration = frame.samples_per_channel / frame.sample_rate
    path = Path(f"out_{language}.wav")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(frame.sample_rate)
        out.writeframes(frame.data.tobytes())
    print(
        f"{language}: first audio {first_audio * 1000:.0f} ms, total {total * 1000:.0f} ms, "
        f"audio {duration:.2f} s -> {path}"
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--voice", help="Preset name (default: the server's DEFAULT_VOICE)")
    parser.add_argument("--languages", nargs="+", default=["sk", "en"], choices=sorted(SENTENCES))
    parser.add_argument("--base-url", help="Server URL instead of the RunPod endpoint, e.g. http://localhost:8000")
    args = parser.parse_args()

    async with utils.http_context.open():
        tts = OmniVoiceTTS(voice=args.voice, base_url=args.base_url)
        try:
            for language in args.languages:
                await synthesize(tts, language, SENTENCES[language])
        finally:
            await tts.aclose()


if __name__ == "__main__":
    asyncio.run(main())
