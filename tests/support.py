"""Puts the test doubles for torch, omnivoice and soundfile ahead of the real packages, so the
server runs without a GPU."""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "tests" / "stubs"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def make_voices(directory: Path, *names: str) -> Path:
    for name in names:
        (directory / f"{name}.wav").write_bytes(b"RIFF")
        (directory / f"{name}.txt").write_text("Dobrý deň, volám sa Nina.", encoding="utf-8")
    return directory


_voices_dir: tempfile.TemporaryDirectory | None = None


def load_app():
    """The server module, with one voice ("1") and FlashInfer off; shared by every test module
    because the engine is created when the module is imported."""
    global _voices_dir
    if _voices_dir is None:
        _voices_dir = tempfile.TemporaryDirectory()
        os.environ["VOICES_DIR"] = str(make_voices(Path(_voices_dir.name), "1"))
        os.environ["OMNIVOICE_FLASHINFER"] = "0"
    import app

    return app
