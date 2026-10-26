FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y libsndfile1 git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir fastapi uvicorn pydantic soundfile numpy omnivoice "brotli>=1.2.0"

# Pre-bake weights so the background load finishes within a few seconds
RUN python -c "from omnivoice import OmniVoice; OmniVoice.from_pretrained('k2-fsa/OmniVoice')"

COPY app.py /app/app.py

ENV PORT=8000

CMD ["python", "app.py"]
