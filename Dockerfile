# Use official PyTorch image with pre-configured CUDA runtime
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Set working system directory
WORKDIR /app

# Prevent interactive prompts during package configuration
ENV DEBIAN_FRONTEND=noninteractive

# Install required system system audio-codecs
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install RunPod SDK, OmniVoice, and sound rendering tools
RUN pip install --no-cache-dir \
    runpod \
    soundfile \
    numpy \
    omnivoice

# Pre-download weights during build to eliminate runtime cold starts
RUN python -c "from omnivoice import OmniVoice; OmniVoice.from_pretrained('k2-fsa/OmniVoice')"

# Bind deployment script
COPY rp_handler.py /app/rp_handler.py

# Execute the RunPod Serverless worker lifecycle
CMD [ "python", "-u", "/app/rp_handler.py" ]
