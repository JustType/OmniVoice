# OmniVoice TTS on RunPod Serverless (for LiveKit Agents)

[OmniVoice](https://github.com/k2-fsa/OmniVoice) behind a FastAPI server (`app.py`, `server/`), deployed as a
RunPod Serverless **load-balancing** endpoint, plus a LiveKit Agents TTS class (`client/omnivoice_tts.py`) that
calls it.

How it works:
- OmniVoice is non-autoregressive. It generates a whole utterance at once and **cannot stream audio as it goes**:
  every decoding step refines the whole utterance, so nothing exists before the last step. The maintainers
  recommend synthesizing sentence by sentence ([#77](https://github.com/k2-fsa/OmniVoice/issues/77)).
- A call keeps **one WebSocket** open (`/v1/stream`). The agent sends the LLM's text as it arrives. The server
  cuts it into sentences, generates them back to back and sends each sentence's audio as soon as it is ready:
  - The first sentence of an utterance is generated alone, for the shortest time to first audio.
  - Whatever is complete when the GPU is free again is generated together (up to 300 characters).
  - When the guest interrupts, the agent cancels the utterance and its pending sentences are dropped.
- The WebSocket pays RunPod's load-balancer overhead (~200 ms per HTTP request) once per call, not once per
  sentence.
- Time to first audio = the LLM's first sentence + its generation + one network hop.
- Decoding uses **FlashInfer kernels with CUDA graphs** (upstream master after 0.2.1, a lossless ~2–2.4×
  speedup for one request at a time) and **16 decoding steps** by default.
- Audio is 16-bit little-endian mono PCM at 24 kHz.

## Deploy on RunPod

```bash
docker build --platform linux/amd64 -t <registry>/omnivoice-tts:<tag> .
docker push <registry>/omnivoice-tts:<tag>
```

RunPod's GitHub integration builds the same Dockerfile. The model weights are baked into the image at build
time, which needs about 4 GB of free RAM. OmniVoice is installed from a pinned upstream commit
(`OMNIVOICE_REF` in the Dockerfile), and FlashInfer from its prebuilt CUDA 12.8 wheels.

Create a Serverless endpoint from the image with these settings:

| Setting | Value | Why |
|---|---|---|
| Endpoint type | **Load Balancer** | `app.py` is an HTTP/WebSocket server, not a queue handler |
| Env `PORT` / `PORT_HEALTH` | `8000` / `8000` | RunPod defaults to port 80 |
| Expose HTTP ports | `8000` | |
| Active (min) workers | **≥ 1** | A cold start takes 35–80 s (measured), far too long for a live call |
| Idle timeout | 60–300 s | The default is 5 s, so extra workers stop almost at once |
| Max workers | concurrent calls + 1 | Each worker generates one sentence at a time |
| Request-count scaler value | 1 | |
| FlashBoot | on | |
| GPU | Ampere or newer (A100, L4, L40S, RTX 4090, H100) | FlashInfer kernels; the worker logs the GPU it got |
| Region | the region of the LiveKit agent | Every sentence pays the network hop |

Optional env vars:

| Variable | Default | |
|---|---|---|
| `DEFAULT_VOICE` | none | Preset used when a request names no voice. The worker fails to start if it is missing |
| `OMNIVOICE_NUM_STEP` | `16` | Decoding steps. `32` is upstream's default: slower, marginally better |
| `OMNIVOICE_FLASHINFER` | `1` | `0` decodes without FlashInfer. If FlashInfer fails on the GPU, the worker reloads the model and serves without it |
| `OMNIVOICE_CUDA_GRAPH_BUCKETS` | `2,3,4,6,8,11,15,20` | Target durations (s) with a CUDA graph each, all captured at startup. Longer sentences decode without a graph |
| `MAX_TEXT_CHARS` | `1000` | Longest accepted text |
| `VOICES_DIR` | `/app/voices` | Where presets are read from |

At startup the worker logs the GPU, whether FlashInfer is on, and how long each bucket's warm-up took. The second
run of the smallest bucket is the steady-state cost of a short sentence.

Health check (`GET /ping`):
- `204` while the model loads and the CUDA graphs are captured;
- `200` when ready;
- `503` with the error if loading failed, so RunPod replaces the worker.

### Check the deployment

```bash
python scripts/ws_probe.py --voice 1 --language sk   # needs only aiohttp
```

It opens the call WebSocket and reports how long the worker took to get ready. It then compares the time to first
audio over the WebSocket and over HTTP, with the server's queue and generation times. Finally it holds the
connection for 8 minutes, speaking a sentence every minute, to show whether RunPod closes an open WebSocket. The
client replaces idle connections after 4 minutes regardless.

Plain HTTP for a quick listen:

```bash
U=https://$RUNPOD_ENDPOINT_ID.api.runpod.ai
H="Authorization: Bearer $RUNPOD_API_KEY"

curl -H "$H" $U/ping                      # {"status":"healthy"}
curl -H "$H" $U/v1/voices                 # {"voices":[...],"num_step":16,"accelerated":true,"gpu":"..."}
curl -H "$H" -H 'Content-Type: application/json' -X POST $U/v1/synthesize \
     -d '{"text":"Dobrý deň, ako vám môžem pomôcť?","voice":"1","language":"sk","response_format":"wav"}' \
     -o out.wav -D - | grep -i x-
```

Each generation is logged as one line in the worker logs (`source`, `queue_ms`, `gen_ms`, `audio_s`, `rtf`).

## API

### `WS /v1/stream`: one connection per call

All messages are JSON. Audio is base64 PCM.

Client → server:

| Message | |
|---|---|
| `{"type": "start", "stream_id", "voice", "language", "speed"}` | Opens an utterance with these options |
| `{"type": "text", "stream_id", "text"}` | More of its text, in any pieces |
| `{"type": "flush", "stream_id"}` | Speak what is buffered now, even an unfinished sentence |
| `{"type": "end", "stream_id"}` | No more text |
| `{"type": "cancel", "stream_id"}` | Drop it (the guest interrupted) |
| `{"type": "ping"}` | Keepalive; answered with `pong` |

Server → client:

| Message | |
|---|---|
| `{"type": "ready", "sample_rate", "voices", "num_step", "accelerated", "gpu"}` | First message. Close code 1013 instead when the model is not loaded yet |
| `{"type": "audio", "stream_id", "seq", "text", "end", "audio", "queue_ms", "gen_ms"}` | One per generated segment. `end` is the offset in the utterance's text just past it, so a client that lost the connection resumes there |
| `{"type": "done", "stream_id"}` | All of the utterance's audio was sent |
| `{"type": "error", "stream_id", "status", "message"}` | 400: bad input such as an unknown voice. 500: inference failed. The utterance ends |

A sentence ends at `.`, `!`, `?` or `…` followed by whitespace and an uppercase letter, unless the word before is a
number or a single letter. So "od 12. februára", "izba č. 5" and "p. Novák" never split a sentence. Sentences
under 20 characters are merged into the next one.

### `POST /v1/synthesize`: one text per request

For scripts and smoke tests. Takes a JSON body:

| Field | Type | |
|---|---|---|
| `text` | string, required | One sentence or a few |
| `voice` | string | Preset name from `voices/`. Falls back to `DEFAULT_VOICE` |
| `language` | string | OmniVoice id: `sk`, `cs`, `en`, … (`en-US` and `Slovak` also work). Omit or `"auto"` to let the model infer it from the text; quality is slightly better when it is given |
| `speed` | number, 0.5–2.0 | Speaking rate; omit to use the model's own estimate |
| `instruct` | string | Voice-design attributes, e.g. `"female, young adult, moderate pitch"` |
| `response_format` | `"pcm"` \| `"wav"` | Default `pcm` |

Responses:
- **200**: the audio, with headers `X-Sample-Rate`, `X-Audio-Duration`, `X-Queue-Ms` and `X-Gen-Ms`.
- **400**: bad input, such as empty text, an unknown voice, or an unsupported `instruct` item.
- **422**: invalid field types or ranges.
- **503**: the model is still loading, or failed to load.
- **500**: inference failed.

Audio is only sent once generation has succeeded, so a failed request never comes back as an empty 200.

## Voices

OmniVoice has **no built-in named voices**. Without a reference clip it picks a different speaker on every
request, and so on every sentence. A stable voice is made once and saved as a preset:

1. Generate candidates from attributes. Each one is a different synthetic speaker:
   ```bash
   python scripts/design_voice.py --instruct "female, young adult, moderate pitch" \
       --language sk --text "Dobrý deň, volám sa Nina. Ako vám môžem pomôcť?" -n 5
   ```
2. Listen to `candidates/*.wav` and keep one. The `.txt` file is the clip's transcript; it is required.
   ```bash
   cp candidates/3.wav voices/nina-sk.wav && cp candidates/3.txt voices/nina-sk.txt
   ```
3. Rebuild and redeploy. At startup the server turns every `voices/<name>.wav` + `<name>.txt` pair into a
   reusable prompt. Requests then select it with `"voice": "nina-sk"`.

Make **one preset per language**. A voice designed in Slovak keeps a Slovak accent when it speaks English.

Attributes go comma-separated, with at most one per group:
- gender: `male` / `female`
- age: `child`, `teenager`, `young adult`, `middle-aged`, `elderly`
- pitch: `very low pitch` … `very high pitch`
- style: `whisper`
- accent (English text only): `american accent`, `british accent`, …

The full list is in OmniVoice's [voice-design docs](https://github.com/k2-fsa/OmniVoice/blob/master/docs/voice-design.md).
Any other value returns a 400.

## LiveKit

`client/omnivoice_tts.py` is self-contained; it needs only `livekit-agents` and `aiohttp`. Copy it into the
agent project.

```python
from omnivoice_tts import OmniVoiceTTS

# RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY are read from the environment
tts = OmniVoiceTTS(voice="nina-sk", language="sk", speed=1.1)
tts.prewarm()  # opens the call's WebSocket while the phone rings
session = AgentSession(tts=tts, ...)

# Switch language mid-call. An utterance already being spoken keeps its old voice.
tts.update_options(language="en", voice="nina-en")

await tts.aclose()  # when the call ends: closes the WebSocket
```

How it behaves at runtime:
- **Prewarm.** `prewarm()` opens the WebSocket in the background, waiting for a worker to start if none is running,
  and logs `OmniVoice ready in N ms (gpu, accelerated, num_step)`. Until `aclose()` it pings every 10 s, which
  keeps the connection and the worker alive between utterances. It reconnects if the connection drops, and
  replaces it, before closing the old one, once it is 4 minutes old and idle.
- **Streaming.** `stream()` sends text as `AgentSession` pushes it. The server decides where sentences end.
- **Barge-in.** Closing a stream cancels it on the server.
- **Failures.** If the connection fails before any audio, LiveKit retries the utterance (`APIConnectOptions`,
  3 retries by default). If it fails after some audio, the client resumes on a new connection after the last
  sentence it received. A 400, such as an unknown voice, is not retried.
- **Timeout.** Once all of an utterance's text is sent, the server must answer within `timeout` (default 30 s).
- `synthesize()` sends the whole text in one HTTP request, for scripts.

Language has to be set explicitly. OmniVoice takes one language per request, or none to infer it from the text.

### In VoxAgent

VoxAgent picks the TTS per agent: `agent.voice_provider = 'omnivoice'`.
- Its `voice_id`, and each language's `voice_id`, name presets here: one per language.
- The SIP channel builds the TTS and prewarms it while the phone rings, and closes it when the call ends.

## Test

```bash
# Server and client against a stub model, no GPU (uses the agent project's virtualenv for livekit-agents)
/path/to/VoxAgent/.venv/bin/python -B -m unittest discover -s tests

# The deployed endpoint: latency over WebSocket vs HTTP, and how long an open WebSocket survives
python scripts/ws_probe.py --voice 1 --language sk

# Through the LiveKit class, the same path an AgentSession takes; writes out_<lang>.wav
python scripts/smoke_test.py --voice nina-sk --languages sk en

# Against a local server instead of RunPod (needs a CUDA GPU)
docker run --gpus all -p 8000:8000 <image>
python scripts/ws_probe.py --base-url http://localhost:8000 --voice nina-sk --hold-minutes 0
```

## Limits

- **No true streaming.** The first audio waits for the whole first sentence to be generated.
- **RunPod load-balancer limits:** 5.5 min per request, 30 MB per response, 2 min wait for a free worker. There
  is no built-in queue: an overloaded endpoint drops requests. Whether the 5.5 min limit applies to WebSockets is
  what `ws_probe.py` checks; the client rotates connections every 4 minutes either way.
