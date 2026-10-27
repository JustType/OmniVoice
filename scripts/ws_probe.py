"""Measure the deployed endpoint the way a call uses it, without LiveKit.

    export RUNPOD_ENDPOINT_ID=... RUNPOD_API_KEY=...
    python scripts/ws_probe.py --voice 1 --language sk

1. Opens the call's WebSocket (/v1/stream) and reports how long it took to get ready
   (a cold worker shows up here) and what the worker runs on.
2. Speaks a few sentences over the WebSocket and over HTTP (/v1/synthesize) and prints the
   time to first audio next to the server's own queue and generation times.
3. Keeps the WebSocket open with pings for --hold-minutes, speaking a short sentence every
   --interval seconds, and reports if and when the connection is closed. This shows whether
   RunPod's 5.5 min request limit or its idle scale-down applies to an open WebSocket.

Needs only aiohttp.
"""

import argparse
import asyncio
import json
import os
import time

import aiohttp

SENTENCES = {
    "sk": [
        "Dobrý deň, ako Vám môžem pomôcť?",
        "Vidím dve potvrdené rezervácie na Vaše meno, od dvanásteho do pätnásteho februára.",
        "Raňajky podávame od siedmej do desiatej v reštaurácii na prízemí. Ak by ste chceli, "
        "môžem Vám rezervovať aj stôl na večeru.",
    ],
    "en": [
        "Hello, how can I help you today?",
        "I can see two confirmed reservations under your name, from the twelfth to the fifteenth of February.",
        "Breakfast is served from seven to ten in the restaurant on the ground floor. If you like, "
        "I can also book you a table for dinner.",
    ],
}
PING_SECONDS = 10


class Probe:
    def __init__(self, session: aiohttp.ClientSession, base_url: str, headers: dict, voice: str, language: str):
        self.session = session
        self.base_url = base_url
        self.headers = headers
        self.voice = voice
        self.language = language
        self.ws: aiohttp.ClientWebSocketResponse | None = None
        self.counter = 0

    @property
    def stream_url(self) -> str:
        scheme, rest = self.base_url.split("://", 1)
        return f"{'wss' if scheme == 'https' else 'ws'}://{rest}/v1/stream"

    async def connect(self) -> None:
        started = time.perf_counter()
        self.ws = await self.session.ws_connect(self.stream_url, headers=self.headers, timeout=aiohttp.ClientWSTimeout(ws_close=10))
        ready = json.loads((await self.ws.receive(timeout=150)).data)
        print(f"connected and ready in {(time.perf_counter() - started) * 1000:.0f} ms: {ready}")

    async def speak_ws(self, text: str) -> tuple[float, float, list[dict]]:
        """Time to first audio, time to done, and the audio messages without their audio."""
        self.counter += 1
        stream_id = f"probe-{self.counter}"
        started = time.perf_counter()
        await self.ws.send_json({"type": "start", "stream_id": stream_id, "voice": self.voice, "language": self.language})
        await self.ws.send_json({"type": "text", "stream_id": stream_id, "text": text})
        await self.ws.send_json({"type": "end", "stream_id": stream_id})
        first, segments = None, []
        while True:
            msg = await self.ws.receive(timeout=60)
            if msg.type != aiohttp.WSMsgType.TEXT:
                raise ConnectionError(f"connection closed (code {self.ws.close_code})")
            message = json.loads(msg.data)
            if message.get("stream_id") != stream_id:
                continue
            if message["type"] == "audio":
                first = first or time.perf_counter() - started
                message.pop("audio")
                segments.append(message)
            elif message["type"] == "done":
                return first, time.perf_counter() - started, segments
            elif message["type"] == "error":
                raise RuntimeError(message["message"])

    async def speak_http(self, text: str) -> tuple[float, str]:
        started = time.perf_counter()
        async with self.session.post(
            f"{self.base_url}/v1/synthesize",
            headers=self.headers,
            json={"text": text, "voice": self.voice, "language": self.language},
            timeout=aiohttp.ClientTimeout(total=150),
        ) as resp:
            await resp.read()
            resp.raise_for_status()
            return time.perf_counter() - started, resp.headers.get("X-Gen-Ms", "?")

    async def keep_alive(self) -> None:
        while True:
            await asyncio.sleep(PING_SECONDS)
            await self.ws.send_json({"type": "ping"})


async def compare(probe: Probe, sentences: list[str]) -> None:
    print("\nsentence chars | ws first audio | ws done | server queue/gen per segment | http total (server gen)")
    for text in sentences:
        first, done, segments = await probe.speak_ws(text)
        timings = ", ".join(f"{s['queue_ms']}/{s['gen_ms']} ms" for s in segments)
        http_total, http_gen = await probe.speak_http(text)
        print(
            f"{len(text):>14} | {first * 1000:>11.0f} ms | {done * 1000:>4.0f} ms | {timings:<28} "
            f"| {http_total * 1000:.0f} ms ({http_gen} ms)"
        )


async def hold(probe: Probe, sentence: str, minutes: float, interval: float) -> None:
    print(f"\nholding the connection for {minutes:g} min, speaking every {interval:g} s")
    opened = time.monotonic()
    pinger = asyncio.create_task(probe.keep_alive())
    try:
        while time.monotonic() - opened < minutes * 60:
            await asyncio.sleep(interval)
            elapsed = time.monotonic() - opened
            try:
                first, _, segments = await probe.speak_ws(sentence)
            except (ConnectionError, ConnectionResetError, asyncio.TimeoutError) as e:
                print(f"{elapsed / 60:5.1f} min: connection lost ({e})")
                started = time.perf_counter()
                await probe.connect()
                print(f"{elapsed / 60:5.1f} min: reconnected in {(time.perf_counter() - started) * 1000:.0f} ms")
                continue
            print(f"{elapsed / 60:5.1f} min: first audio {first * 1000:.0f} ms (server gen {segments[0]['gen_ms']} ms)")
    finally:
        pinger.cancel()
    print(f"connection still open after {(time.monotonic() - opened) / 60:.1f} min")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--voice", required=True, help="Preset name, e.g. 1")
    parser.add_argument("--language", default="sk", choices=sorted(SENTENCES))
    parser.add_argument("--hold-minutes", type=float, default=8.0, help="0 skips the hold test")
    parser.add_argument("--interval", type=float, default=60.0, help="Seconds between sentences while holding")
    parser.add_argument("--endpoint-id", default=os.environ.get("RUNPOD_ENDPOINT_ID"))
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"))
    parser.add_argument("--base-url", help="Server URL instead of the RunPod endpoint, e.g. http://localhost:8000")
    args = parser.parse_args()

    base_url = (args.base_url or f"https://{args.endpoint_id}.api.runpod.ai").rstrip("/")
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    async with aiohttp.ClientSession() as session:
        probe = Probe(session, base_url, headers, args.voice, args.language)
        await probe.connect()
        await compare(probe, SENTENCES[args.language])
        if args.hold_minutes > 0:
            await hold(probe, SENTENCES[args.language][0], args.hold_minutes, args.interval)
        await probe.ws.close()


if __name__ == "__main__":
    asyncio.run(main())
