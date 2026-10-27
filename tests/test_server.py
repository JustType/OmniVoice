import base64
import time
import unittest

import support
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from omnivoice import SAMPLE_RATE, SECONDS_PER_CHAR

server = support.load_app()

GREETING = "Tu je Nina z hotela Ambiente. Ako Vám môžem pomôcť s rezerváciou?"


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._client_context = TestClient(server.app)
        cls.client = cls._client_context.__enter__()
        deadline = time.monotonic() + 10
        while cls.client.get("/ping").status_code != 200:
            if time.monotonic() > deadline:
                raise RuntimeError("the server did not become ready")
            time.sleep(0.01)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._client_context.__exit__(None, None, None)

    def _open(self, ws) -> dict:
        ready = ws.receive_json()
        self.assertEqual(ready["type"], "ready")
        return ready

    def _until_done(self, ws, stream_id: str) -> list[dict]:
        audio = []
        while True:
            message = ws.receive_json()
            self.assertEqual(message["stream_id"], stream_id, message)
            if message["type"] == "done":
                return audio
            self.assertEqual(message["type"], "audio", message)
            audio.append(message)

    def test_http_synthesize_returns_pcm_with_timings(self):
        response = self.client.post("/v1/synthesize", json={"text": "Dobrý deň.", "voice": "1", "language": "sk"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-sample-rate"], str(SAMPLE_RATE))
        self.assertIn("x-gen-ms", response.headers)
        self.assertEqual(len(response.content), 2 * int(SAMPLE_RATE * SECONDS_PER_CHAR * len("Dobrý deň.")))

    def test_http_rejects_an_unknown_voice(self):
        response = self.client.post("/v1/synthesize", json={"text": "Dobrý deň.", "voice": "9"})
        self.assertEqual(response.status_code, 400)

    def test_voices_report_the_engine(self):
        info = self.client.get("/v1/voices").json()
        self.assertEqual((info["voices"], info["num_step"], info["gpu"]), (["1"], 16, "Fake GPU"))

    def test_a_stream_speaks_its_sentences_in_order(self):
        with self.client.websocket_connect("/v1/stream") as ws:
            ready = self._open(ws)
            self.assertEqual(ready["sample_rate"], SAMPLE_RATE)
            ws.send_json({"type": "start", "stream_id": "s1", "voice": "1", "language": "sk", "speed": 1.0})
            for i in range(0, len(GREETING), 7):
                ws.send_json({"type": "text", "stream_id": "s1", "text": GREETING[i:i + 7]})
            ws.send_json({"type": "end", "stream_id": "s1"})
            audio = self._until_done(ws, "s1")
        self.assertEqual([m["text"] for m in audio], ["Tu je Nina z hotela Ambiente.", "Ako Vám môžem pomôcť s rezerváciou?"])
        self.assertEqual([m["seq"] for m in audio], [0, 1])
        self.assertEqual(audio[-1]["end"], len(GREETING))
        self.assertEqual(GREETING[audio[0]["end"]:].strip(), audio[1]["text"])
        pcm = base64.b64decode(audio[0]["audio"])
        self.assertEqual(len(pcm), 2 * int(SAMPLE_RATE * SECONDS_PER_CHAR * len(audio[0]["text"])))

    def test_the_first_sentence_goes_alone_and_the_rest_together(self):
        sentences = [
            "Tu je Nina z hotela Ambiente.",
            "Vidím dve potvrdené rezervácie na Vaše meno.",
            "Prvá je od dvanásteho februára.",
            "Druhá je od devätnásteho februára.",
        ]
        with self.client.websocket_connect("/v1/stream") as ws:
            self._open(ws)
            ws.send_json({"type": "start", "stream_id": "s2", "voice": "1", "language": "sk"})
            ws.send_json({"type": "text", "stream_id": "s2", "text": " ".join(sentences)})
            ws.send_json({"type": "end", "stream_id": "s2"})
            audio = self._until_done(ws, "s2")
        self.assertEqual(audio[0]["text"], sentences[0])
        self.assertIn(" ".join(sentences[1:3]), audio[1]["text"])
        self.assertEqual(" ".join(m["text"] for m in audio), " ".join(sentences))

    def test_a_cancelled_stream_sends_no_more_audio(self):
        with self.client.websocket_connect("/v1/stream") as ws:
            self._open(ws)
            ws.send_json({"type": "start", "stream_id": "s3", "voice": "1", "language": "sk"})
            ws.send_json({"type": "text", "stream_id": "s3", "text": GREETING + " Ďalšia veta príde. "})
            ws.send_json({"type": "cancel", "stream_id": "s3"})
            ws.send_json({"type": "ping"})
            self.assertEqual(ws.receive_json(), {"type": "pong"})
            time.sleep(3 * server.engine.model.generate_seconds)
            ws.send_json({"type": "ping"})
            self.assertEqual(ws.receive_json(), {"type": "pong"})

    def test_an_unknown_voice_fails_the_stream(self):
        with self.client.websocket_connect("/v1/stream") as ws:
            self._open(ws)
            ws.send_json({"type": "start", "stream_id": "s4", "voice": "9", "language": "sk"})
            error = ws.receive_json()
        self.assertEqual((error["type"], error["stream_id"], error["status"]), ("error", "s4", 400))

    def test_an_empty_stream_is_done_at_once(self):
        with self.client.websocket_connect("/v1/stream") as ws:
            self._open(ws)
            ws.send_json({"type": "start", "stream_id": "s5", "voice": "1"})
            ws.send_json({"type": "end", "stream_id": "s5"})
            self.assertEqual(ws.receive_json(), {"type": "done", "stream_id": "s5"})

    def test_a_loading_model_closes_the_connection(self):
        server.engine.status = "loading"
        try:
            with self.client.websocket_connect("/v1/stream") as ws:
                with self.assertRaises(WebSocketDisconnect) as ctx:
                    ws.receive_json()
        finally:
            server.engine.status = "ready"
        self.assertEqual(ctx.exception.code, 1013)


if __name__ == "__main__":
    unittest.main()
