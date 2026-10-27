"""Test double for omnivoice 0.2: the same call shapes, silence instead of speech."""

import threading
import time

import numpy as np

SAMPLE_RATE = 24000
SECONDS_PER_CHAR = 0.06


class VoiceClonePrompt:
    def __init__(self, path):
        self.path = path


class OmniVoice:
    loads = 0
    generate_seconds = 0.05

    def __init__(self):
        self.sampling_rate = SAMPLE_RATE
        self.calls = []
        self._busy = threading.Lock()

    @classmethod
    def from_pretrained(cls, name, *args, **kwargs):
        cls.loads += 1
        return cls()

    def create_voice_clone_prompt(self, ref_audio, ref_text=None, preprocess_prompt=True):
        assert ref_text, "ref_text required"
        return VoiceClonePrompt(ref_audio)

    def generate(self, text, language=None, voice_clone_prompt=None, instruct=None, duration=None,
                 speed=None, **kwargs):
        if not self._busy.acquire(blocking=False):
            raise RuntimeError("concurrent generate() calls")
        try:
            if instruct and "blah" in instruct:
                raise ValueError(f"Unsupported instruct items found in {instruct}")
            self.calls.append({"text": text, "language": language, "duration": duration, "speed": speed, **kwargs})
            time.sleep(self.generate_seconds)
            seconds = duration or SECONDS_PER_CHAR * len(text) / (speed or 1.0)
            return [np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)]
        finally:
            self._busy.release()
