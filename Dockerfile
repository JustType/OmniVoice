
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# Install required system packages
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install web serving utilities and OmniVoice dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    soundfile \
    numpy \
    omnivoice

# Pre-download weights to ensure they are built directly into the container layer
RUN python -c "from omnivoice import OmniVoice; OmniVoice.from_pretrained('k2-fsa/OmniVoice')"

COPY app.py /app/app.py

# Inform RunPod which internal port variables to listen on
ENV PORT=8000
ENV PORT_HEALTH=8000

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
