import tempfile
import unittest
from pathlib import Path

import support

import omnivoice
from omnivoice.models import omnivoice_flashinfer
from server.engine import Engine, EngineError, Settings, resolve_language


class EngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.voices = support.make_voices(Path(self._dir.name), "1", "2")
        omnivoice_flashinfer.FAIL = False
        omnivoice.OmniVoice.loads = 0

    def tearDown(self) -> None:
        omnivoice_flashinfer.FAIL = False
        self._dir.cleanup()

    def _engine(self, **settings) -> Engine:
        engine = Engine(Settings(voices_dir=self.voices, graph_buckets=(2, 4), **settings))
        engine.load()
        return engine

    def test_flashinfer_captures_every_bucket_before_ready(self):
        engine = self._engine()
        self.assertEqual((engine.status, engine.accelerated, engine.gpu), ("ready", True, "Fake GPU"))
        self.assertIsNone(engine.info()["acceleration_error"])
        self.assertEqual(engine.model.graph_buckets, [2, 4])
        self.assertEqual([call["duration"] for call in engine.model.calls], [2, 4, 2])
        self.assertTrue(all(call["num_step"] == 16 for call in engine.model.calls))

    def test_a_flashinfer_failure_reloads_the_model_and_serves_without_it(self):
        omnivoice_flashinfer.FAIL = True
        engine = self._engine()
        self.assertEqual((engine.status, engine.accelerated), ("ready", False))
        self.assertEqual(
            engine.info()["acceleration_error"],
            "RuntimeError: no kernel image is available for execution on the device",
        )
        self.assertEqual(omnivoice.OmniVoice.loads, 2)
        self.assertEqual(sorted(engine.voices), ["1", "2"])

    def test_flashinfer_can_be_turned_off(self):
        engine = self._engine(flashinfer=False)
        self.assertFalse(engine.accelerated)
        self.assertEqual([call["duration"] for call in engine.model.calls], [None])

    def test_an_unknown_default_voice_fails_the_load(self):
        engine = self._engine(default_voice="nina-sk")
        self.assertEqual(engine.status, "failed")
        self.assertIn("nina-sk", engine.error)

    def test_voice_settings_are_validated(self):
        engine = self._engine()
        with self.assertRaises(EngineError) as unknown:
            engine.voice(voice="9", language="sk", speed=None)
        with self.assertRaises(EngineError) as too_fast:
            engine.voice(voice="1", language="sk", speed=2.5)
        self.assertEqual((unknown.exception.status, too_fast.exception.status), (400, 400))
        self.assertEqual(engine.voice(voice="1", language="en-US", speed=1.1).language, "en")

    def test_a_loading_engine_is_unavailable(self):
        engine = Engine(Settings(voices_dir=self.voices))
        with self.assertRaises(EngineError) as ctx:
            engine.require_ready()
        self.assertEqual(ctx.exception.status, 503)

    def test_language_names_and_auto(self):
        self.assertEqual(resolve_language("Slovak"), "sk")
        self.assertIsNone(resolve_language("auto"))
        self.assertIsNone(resolve_language("klingon"))


if __name__ == "__main__":
    unittest.main()
