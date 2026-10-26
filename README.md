# OmniVoice TTS on RunPod Serverless (for LiveKit Agents)

[OmniVoice](https://github.com/k2-fsa/OmniVoice) behind a FastAPI server (`app.py`), deployed as a
RunPod Serverless **load-balancing** endpoint, plus a LiveKit Agents TTS class
(`client/omnivoice_tts.py`) that calls it.

How it works:
- OmniVoice is non-autoregressive. It generates a whole utterance at once and **cannot stream**.
- The LiveKit class is therefore a non-streaming TTS. `AgentSession` wraps it in a `StreamAdapter`
  automatically, which sends **one request per sentence**.
- Time to first audio = LLM's first sentence + generation of that sentence + network round trip.
- The server returns 16-bit little-endian mono PCM at 24 kHz.

## Deploy on RunPod

```bash
docker build --platform linux/amd64 -t <registry>/omnivoice-tts:<tag> .
docker push <registry>/omnivoice-tts:<tag>
```

RunPod's GitHub integration builds the same Dockerfile. The model weights are baked into the image at build
time, which needs about 4 GB of free RAM.

Create a Serverless endpoint from the image with these settings:

| Setting | Value | Why |
|---|---|---|
| Endpoint type | **Load Balancer** | `app.py` is an HTTP server, not a queue handler |
| Env `PORT` / `PORT_HEALTH` | `8000` / `8000` | RunPod defaults to port 80 |
| Expose HTTP ports | `8000` | |
| Active (min) workers | **≥ 1** | A cold start (image pull + model load) takes far too long for a live call |
| Idle timeout | 20–30 min | |
| FlashBoot | on | |
| Request-count scaler value | 1–2 | Each worker generates one request at a time |
| GPU | any ≥ 8 GB CUDA GPU | ~0.6B-parameter model in FP16 plus a FP32 audio codec |
| Region | close to the LiveKit agent | Every sentence pays the network round trip |

Optional env vars:

| Variable | Default | |
|---|---|---|
| `DEFAULT_VOICE` | none | Preset used when a request names no voice. The worker fails to start if it is missing |
| `OMNIVOICE_NUM_STEP` | `32` | Decoding steps. `16` is about 2× faster, at some quality cost |
| `MAX_TEXT_CHARS` | `1000` | Longest accepted text |
| `VOICES_DIR` | `/app/voices` | Where presets are read from |

Health check (`GET /ping`):
- `204` while the model loads;
- `200` when ready;
- `503` with the error if loading failed, so RunPod replaces the worker.

### Check the deployment

```bash
U=https://$RUNPOD_ENDPOINT_ID.api.runpod.ai
H="Authorization: Bearer $RUNPOD_API_KEY"

curl -H "$H" $U/ping                      # {"status":"healthy"}
curl -H "$H" $U/v1/voices                 # {"voices":[...],"default_voice":...,"sample_rate":24000}
curl -H "$H" -H 'Content-Type: application/json' -X POST $U/v1/synthesize \
     -d '{"text":"Dobrý deň, ako vám môžem pomôcť?","voice":"nina-sk","language":"sk","response_format":"wav"}' \
     -o out.wav -w 'HTTP %{http_code}, first byte after %{time_starttransfer}s\n'
```

LiveKit uses the default `"response_format": "pcm"`. WAV is only for listening.

Each response is logged as one line in the worker logs (`queue_ms`, `gen_ms`, `audio_s`, `rtf`).
Use these to tune latency.

## API

`POST /v1/synthesize` takes a JSON body:

| Field | Type | |
|---|---|---|
| `text` | string, required | One sentence or a few |
| `voice` | string | Preset name from `voices/`. Falls back to `DEFAULT_VOICE` |
| `language` | string | OmniVoice id: `sk`, `cs`, `en`, … (`en-US` and `Slovak` also work). Omit or `"auto"` to let the model infer it from the text; quality is slightly better when it is given |
| `speed` | number, 0.5–2.0 | Speaking rate; omit to use the model's own estimate |
| `instruct` | string | Voice-design attributes, e.g. `"female, young adult, moderate pitch"` |
| `response_format` | `"pcm"` \| `"wav"` | Default `pcm` |

Responses:
- **200**: the audio, with headers `X-Sample-Rate` and `X-Audio-Duration`.
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
tts.prewarm()  # opens the TLS connection early
session = AgentSession(tts=tts, ...)

# Switch language mid-call. A sentence already playing keeps its old voice.
tts.update_options(language="en", voice="nina-en")
```

Language has to be set explicitly. OmniVoice takes one language per request, or none to infer it from
the text. It has no list of language hints like Soniox.

How it behaves at runtime:
- **Retries.** Failed sentences are retried by LiveKit (`APIConnectOptions`, 3 retries by default) for
  connection errors, timeouts and 5xx such as a worker that is still loading. A 400 is not retried.
- **Timeout.** `timeout` (default 30 s) covers one whole sentence, generation included.
- **Flushes.** `FlushSentinel` is honoured: the `StreamAdapter` sends the buffered text as its own request.

### Wiring into VoxAgent (not done; VoxAgent is unchanged)

`OmniVoiceTTS` accepts the same `update_options(language=, voice=, speed=)` call that
`src/agent/channels/sip/speech.py` already makes on `soniox.TTS`. Switching would mean:
- In `src/agent/channels/sip/session.py`, `build_tts` returns `OmniVoiceTTS(voice=voice, language=language,
  speed=speed)`, and the `soniox.TTS` type hints are widened.
- Persona `voice_id`s are changed to OmniVoice preset names, one per language (e.g. `nina-sk`, `nina-en`).
- The current Soniox speed range 0.7–1.3 fits OmniVoice's 0.5–2.0.

## Test

```bash
# Through the LiveKit class, the same path an AgentSession takes; writes out_<lang>.wav
python scripts/smoke_test.py --voice nina-sk --languages sk en

# Against a local server instead of RunPod (needs a CUDA GPU)
docker run --gpus all -p 8000:8000 <image>
python scripts/smoke_test.py --base-url http://localhost:8000 --voice nina-sk
```

## Limits

- **No true streaming.** Long sentences take longer before their first audio. `OMNIVOICE_NUM_STEP=16`
  lowers latency.
- **RunPod load-balancer limits:** 5.5 min per request, 30 MB per response, 2 min wait for a free worker.
  There is no built-in queue: an overloaded endpoint drops requests, and LiveKit retries them.
