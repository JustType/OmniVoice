
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y libsndfile1 git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir fastapi uvicorn pydantic soundfile numpy omnivoice

# Pre-bake weights so the background load finishes within a few seconds
RUN python -c "from omnivoice import OmniVoice; OmniVoice.from_pretrained('k2-fsa/OmniVoice')"

COPY app.py /app/app.py

# Enforce standard environmental variables for the RunPod proxy system
ENV PORT=8000
ENV PORT_HEALTH=8000
ENV HEALTH_CHECK_PATH=/ping

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
