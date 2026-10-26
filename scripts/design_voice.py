"""Generate candidate voices from attributes, to keep one as a preset in voices/.

    export RUNPOD_ENDPOINT_ID=... RUNPOD_API_KEY=...
    python scripts/design_voice.py --instruct "female, young adult, moderate pitch" \\
        --language sk --text "Dobrý deň, volám sa Nina. Ako vám môžem pomôcť?" -n 5

Every candidate is a different speaker matching the attributes. Listen to
candidates/1.wav ... 5.wav and keep the one you like:

    cp candidates/3.wav voices/nina-sk.wav && cp candidates/3.txt voices/nina-sk.txt

then rebuild and redeploy the image. The .txt is the clip's transcript; the server needs it.
Make one voice per language: a voice designed in Slovak speaks English with a Slovak accent.

Attributes (comma separated, at most one per group): male / female; child, teenager,
young adult, middle-aged, elderly; very low / low / moderate / high / very high pitch;
whisper; american / british / australian / canadian / indian / ... accent (English only).
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instruct", required=True, help='Voice attributes, e.g. "female, young adult"')
    parser.add_argument("--text", required=True, help="What the voice says; 5-10 s of speech works best")
    parser.add_argument("--language", help='Language of --text, e.g. "sk" (default: inferred)')
    parser.add_argument("-n", "--count", type=int, default=5, help="Number of candidates (default 5)")
    parser.add_argument("--out", default="candidates", help="Output directory (default candidates/)")
    parser.add_argument("--endpoint-id", default=os.environ.get("RUNPOD_ENDPOINT_ID"))
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"))
    parser.add_argument("--base-url", help="Server URL instead of the RunPod endpoint, e.g. http://localhost:8000")
    args = parser.parse_args()

    base_url = args.base_url or (args.endpoint_id and f"https://{args.endpoint_id}.api.runpod.ai")
    if not base_url:
        parser.error("pass --endpoint-id (or set RUNPOD_ENDPOINT_ID) or --base-url")

    body = json.dumps(
        {"text": args.text, "instruct": args.instruct, "language": args.language, "response_format": "wav"}
    ).encode()
    headers = {"Content-Type": "application/json", "User-Agent": "omnivoice-design-voice/1.0"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(1, args.count + 1):
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/v1/synthesize", data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                audio = response.read()
        except urllib.error.HTTPError as e:
            sys.exit(f"Request failed: HTTP {e.code}: {e.read().decode(errors='replace')}")
        (out / f"{i}.wav").write_bytes(audio)
        (out / f"{i}.txt").write_text(args.text + "\n", encoding="utf-8")
        print(out / f"{i}.wav")

    print(f"\nKeep one as a voice: cp {out}/N.wav voices/<name>.wav && cp {out}/N.txt voices/<name>.txt")


if __name__ == "__main__":
    main()
