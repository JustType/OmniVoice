"""Test double for the parts of torch the server touches."""

import contextlib
from types import SimpleNamespace

float16 = "float16"
inference_mode = contextlib.nullcontext
cuda = SimpleNamespace(is_available=lambda: True, get_device_name=lambda index=0: "Fake GPU")
