"""The LiveKit client against the real server (with the stub model) over a real WebSocket.
Needs livekit-agents, e.g. run the tests with the agent project's virtualenv."""

import asyncio
import socket
import sys
import threading
import time
import unittest

import support

from omnivoice import SECONDS_PER_CHAR

try:
    import uvicorn
    from livekit.agents import utils
except ImportError:  # pragma: no cover
    uvicorn = None

sys.path.insert(0, str(support.ROOT / "client"))
server = support.load_app()


@unittest.skipIf(uvicorn is None, "needs uvicorn and livekit-agents")
class ClientTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            cls.port = sock.getsockname()[1]
        cls.server = uvicorn.Server(uvicorn.Config(server.app, host="127.0.0.1", port=cls.port, log_level="warning"))
        threading.Thread(target=cls.server.run, daemon=True).start()
        deadline = time.monotonic() + 10
        while not (cls.server.started and server.engine.status == "ready"):
            if time.monotonic() > deadline:
                raise RuntimeError("the server did not start")
            time.sleep(0.01)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.should_exit = True

    async def asyncSetUp(self) -> None:
        from omnivoice_tts import OmniVoiceTTS

        self._http = utils.http_context.open()
        await self._http.__aenter__()
        self.tts = OmniVoiceTTS(voice="1", language="sk", speed=1.0, base_url=f"http://127.0.0.1:{self.port}")
        self.calls = server.engine.model.calls
        self.calls_before = len(self.calls)

    async def asyncTearDown(self) -> None:
        await self.tts.aclose()
        await self._http.__aexit__(None, None, None)

    async def test_a_streamed_reply_is_spoken_sentence_by_sentence(self):
        text = "Tu je Nina z hotela Ambiente. Máte rezerváciu od 12. februára. Ako Vám môžem pomôcť?"
        duration = 0.0
        async with self.tts.stream() as stream:
            for i in range(0, len(text), 6):
                stream.push_text(text[i:i + 6])
            stream.end_input()
            async for event in stream:
                duration += event.frame.duration
        spoken = [call["text"] for call in self.calls[self.calls_before:]]
        self.assertEqual(spoken[0], "Tu je Nina z hotela Ambiente.")
        self.assertEqual(" ".join(spoken), text)
        expected = sum(SECONDS_PER_CHAR * len(part) for part in spoken)
        self.assertAlmostEqual(duration, expected, delta=0.05)

    async def test_an_interrupted_reply_stops_on_the_server(self):
        sentences = [f"Toto je dlhá veta číslo {n} pre test prerušenia." for n in "ABCDEFGH"]
        stream = self.tts.stream()
        stream.push_text(" ".join(sentences))
        stream.end_input()
        async for _ in stream:
            break
        await stream.aclose()
        await asyncio.sleep(5 * server.engine.model.generate_seconds)
        self.assertLess(len(self.calls) - self.calls_before, 4)


if __name__ == "__main__":
    unittest.main()
