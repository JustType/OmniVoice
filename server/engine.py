"""The OmniVoice model on the GPU: loading, voice presets, acceleration and generation.

OmniVoice is non-autoregressive: it generates a whole utterance at once and cannot stream.
Generation is serialised on one GPU lock; FlashInfer kernels with CUDA graphs (one graph per
duration bucket) remove most of the per-step launch overhead that dominates short sentences.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from omnivoice import OmniVoice, VoiceClonePrompt
from omnivoice.utils.lang_map import LANG_IDS, LANG_NAME_TO_ID
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger("omnivoice-server")

WARM_UP_TEXT = "Dobrý deň, ako vám môžem pomôcť?"


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in ("1", "true", "yes", "on")


def _env_floats(name: str, default: str) -> tuple[float, ...]:
    raw = os.getenv(name, default)
    return tuple(sorted(float(item) for item in raw.split(",") if item.strip()))


@dataclass(frozen=True)
class Settings:
    model_id: str = "k2-fsa/OmniVoice"
    voices_dir: Path = Path("/app/voices")
    default_voice: Optional[str] = None
    num_step: int = 16
    max_text_chars: int = 1000
    flashinfer: bool = True
    # Target durations in seconds; a sentence longer than the last one decodes without graphs.
    graph_buckets: tuple[float, ...] = (2, 3, 4, 6, 8, 11, 15, 20)

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            model_id=os.getenv("OMNIVOICE_MODEL", cls.model_id),
            voices_dir=Path(os.getenv("VOICES_DIR", str(cls.voices_dir))),
            default_voice=os.getenv("DEFAULT_VOICE") or None,
            num_step=int(os.getenv("OMNIVOICE_NUM_STEP", str(cls.num_step))),
            max_text_chars=int(os.getenv("MAX_TEXT_CHARS", str(cls.max_text_chars))),
            flashinfer=_env_flag("OMNIVOICE_FLASHINFER", cls.flashinfer),
            graph_buckets=_env_floats(
                "OMNIVOICE_CUDA_GRAPH_BUCKETS", ",".join(str(b) for b in cls.graph_buckets)
            ),
        )


class EngineError(Exception):
    """A request the engine cannot serve; ``status`` is the matching HTTP status."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(frozen=True)
class Voice:
    """What a request speaks with: a preset, a voice design, or neither (a random voice)."""

    name: Optional[str]
    prompt: Optional[VoiceClonePrompt]
    instruct: Optional[str]
    language: Optional[str]
    speed: Optional[float]

    @property
    def label(self) -> str:
        return self.name or ("design" if self.instruct else "auto")


@dataclass(frozen=True)
class Generation:
    audio: np.ndarray
    sample_rate: int
    queue_ms: float
    gen_ms: float

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate

    def pcm16(self) -> bytes:
        """16-bit little-endian mono PCM."""
        return (np.clip(self.audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


@dataclass
class Engine:
    settings: Settings
    status: Literal["loading", "ready", "failed"] = "loading"
    error: Optional[str] = None
    model: Optional[OmniVoice] = None
    voices: dict[str, VoiceClonePrompt] = field(default_factory=dict)
    sample_rate: int = 24000
    accelerated: bool = False
    gpu: Optional[str] = None
    # The model is not thread-safe. A thread lock (not an asyncio one) keeps the GPU serialised
    # even when a request is cancelled while its generation thread is still running.
    _gpu_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def load(self) -> None:
        """Loads the model, the voices and the acceleration; runs in a background thread so
        the server answers /ping (204) while it loads."""
        try:
            self.gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            model, voices = self._load_model()
            if self.settings.flashinfer:
                try:
                    self._accelerate(model, voices)
                    self.accelerated = True
                except Exception:
                    # A half-patched model is unusable: start again from clean weights.
                    logger.exception("FlashInfer acceleration failed, decoding without it")
                    model, voices = self._load_model()
            if not self.accelerated:
                # The first generation compiles CUDA kernels; do it before taking traffic.
                self._timed_warm_up(model, voices, duration=None)

            self.model, self.voices = model, voices
            self.sample_rate = model.sampling_rate or 24000
            self.status = "ready"
            logger.info(
                "OmniVoice ready (gpu: %s, accelerated: %s, num_step: %d, voices: %s)",
                self.gpu, self.accelerated, self.settings.num_step, sorted(voices),
            )
        except Exception as e:
            logger.exception("Model failed to load")
            self.error = f"{type(e).__name__}: {e}"
            self.status = "failed"

    def _load_model(self) -> tuple[OmniVoice, dict[str, VoiceClonePrompt]]:
        logger.info("Loading %s into VRAM (FP16)...", self.settings.model_id)
        model = OmniVoice.from_pretrained(self.settings.model_id, device_map="cuda:0", dtype=torch.float16)
        voices = self._load_voices(model)
        default = self.settings.default_voice
        if default and default not in voices:
            raise RuntimeError(
                f"DEFAULT_VOICE '{default}' is not in {self.settings.voices_dir} (found: {sorted(voices)})"
            )
        if not voices:
            logger.warning(
                "No voices in %s: requests without 'instruct' get a random voice", self.settings.voices_dir
            )
        return model, voices

    def _load_voices(self, model: OmniVoice) -> dict[str, VoiceClonePrompt]:
        """Every ``<name>.wav`` with a ``<name>.txt`` transcript in the voices dir becomes a voice."""
        voices: dict[str, VoiceClonePrompt] = {}
        voices_dir = self.settings.voices_dir
        if not voices_dir.is_dir():
            return voices
        for wav in sorted(voices_dir.glob("*.wav")):
            transcript = wav.with_suffix(".txt")
            if not transcript.is_file():
                # Without a transcript OmniVoice would download and run Whisper to make one.
                logger.warning("Skipping voice '%s': %s is missing", wav.stem, transcript.name)
                continue
            with torch.inference_mode():
                voices[wav.stem] = model.create_voice_clone_prompt(
                    ref_audio=str(wav), ref_text=transcript.read_text(encoding="utf-8").strip()
                )
            logger.info("Loaded voice '%s'", wav.stem)
        return voices

    def _accelerate(self, model: OmniVoice, voices: dict[str, VoiceClonePrompt]) -> None:
        """FlashInfer kernels plus one CUDA graph per duration bucket, all captured now so no
        call pays for a capture."""
        from omnivoice.models.omnivoice_flashinfer import apply_flashinfer

        apply_flashinfer(model, cuda_graph_buckets=list(self.settings.graph_buckets))
        for bucket in self.settings.graph_buckets:
            self._timed_warm_up(model, voices, duration=bucket)
        # A second run of the smallest bucket shows the steady-state cost of a short sentence.
        self._timed_warm_up(model, voices, duration=self.settings.graph_buckets[0])

    def _timed_warm_up(
        self, model: OmniVoice, voices: dict[str, VoiceClonePrompt], *, duration: Optional[float]
    ) -> None:
        prompt = next(iter(voices.values()), None)
        started = time.perf_counter()
        model.generate(
            text=WARM_UP_TEXT,
            language="sk",
            voice_clone_prompt=prompt,
            duration=duration,
            num_step=self.settings.num_step,
        )
        logger.info(
            "Warm-up generation (%s) took %.0f ms",
            f"{duration:g} s bucket" if duration else "eager",
            (time.perf_counter() - started) * 1000,
        )

    def info(self) -> dict[str, object]:
        return {
            "voices": sorted(self.voices),
            "default_voice": self.settings.default_voice,
            "sample_rate": self.sample_rate,
            "num_step": self.settings.num_step,
            "accelerated": self.accelerated,
            "gpu": self.gpu,
        }

    def require_ready(self) -> OmniVoice:
        if self.status == "loading":
            raise EngineError(503, "Model is still loading.")
        if self.status == "failed":
            raise EngineError(503, f"Model failed to load: {self.error}")
        return self.model

    def voice(
        self,
        *,
        voice: Optional[str],
        language: Optional[str],
        speed: Optional[float],
        instruct: Optional[str] = None,
    ) -> Voice:
        """Validates a request's voice settings; raises EngineError(400) on bad ones."""
        self.require_ready()
        if speed is not None and not 0.5 <= speed <= 2.0:
            raise EngineError(400, "'speed' must be between 0.5 and 2.0.")
        name = voice or (None if instruct else self.settings.default_voice)
        prompt = None
        if name:
            prompt = self.voices.get(name)
            if prompt is None:
                raise EngineError(400, f"Unknown voice '{name}'. Available: {sorted(self.voices)}")
        elif not instruct:
            logger.warning("Request has no voice or instruct: OmniVoice picks a random voice")
        return Voice(
            name=name,
            prompt=prompt,
            instruct=instruct or None,
            language=resolve_language(language),
            speed=speed,
        )

    def text(self, value: str) -> str:
        text = value.strip()
        if not text:
            raise EngineError(400, "The 'text' field cannot be empty.")
        if len(text) > self.settings.max_text_chars:
            raise EngineError(400, f"'text' is longer than {self.settings.max_text_chars} characters.")
        return text

    async def generate(self, text: str, voice: Voice, *, source: str) -> Generation:
        """Generates ``text`` once the GPU is free; raises EngineError on bad input or failure."""
        model = self.require_ready()
        started = time.perf_counter()
        try:
            audio, elapsed = await run_in_threadpool(self._generate, model, text, voice)
        except ValueError as e:
            # OmniVoice validates instruct items and raises ValueError on unknown ones.
            raise EngineError(400, str(e)) from e
        except Exception as e:
            logger.exception("Inference failed")
            raise EngineError(500, f"Inference failed: {type(e).__name__}: {e}") from e
        generation = Generation(
            audio=audio,
            sample_rate=self.sample_rate,
            queue_ms=(time.perf_counter() - started - elapsed) * 1000,
            gen_ms=elapsed * 1000,
        )
        duration = generation.duration
        logger.info(
            "synthesize source=%s chars=%d voice=%s language=%s queue_ms=%.0f gen_ms=%.0f audio_s=%.2f rtf=%.3f",
            source, len(text), voice.label, voice.language or "auto", generation.queue_ms,
            generation.gen_ms, duration, generation.gen_ms / 1000 / duration if duration else 0.0,
        )
        return generation

    def _generate(self, model: OmniVoice, text: str, voice: Voice) -> tuple[np.ndarray, float]:
        """The audio and the generation time, which excludes the wait for the GPU."""
        with self._gpu_lock:
            started = time.perf_counter()
            audio = model.generate(
                text=text,
                language=voice.language,
                voice_clone_prompt=voice.prompt,
                instruct=voice.instruct,
                speed=voice.speed,
                num_step=self.settings.num_step,
            )[0]
            elapsed = time.perf_counter() - started
        return np.asarray(audio, dtype=np.float32).reshape(-1), elapsed


def resolve_language(value: Optional[str]) -> Optional[str]:
    """OmniVoice language id ("sk", "en-US" -> "en", "Slovak" -> "sk"); None lets the model
    infer the language from the text."""
    if not value:
        return None
    code = value.strip().lower().replace("_", "-")
    if code in ("auto", "none"):
        return None
    for candidate in (code, code.split("-")[0]):
        if candidate in LANG_IDS:
            return candidate
    if code in LANG_NAME_TO_ID:
        return LANG_NAME_TO_ID[code]
    logger.warning("Unsupported language '%s', letting the model infer it from the text", value)
    return None
