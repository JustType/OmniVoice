
# Use official PyTorch image with pre-configured CUDA runtime
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Set working directory inside the container
WORKDIR /app

# Prevent interactive prompts during package configuration
ENV DEBIAN_FRONTEND=noninteractive

# Install required system audio-codecs and Git
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install modern Python web serving tools along with OmniVoice core
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    soundfile \
    numpy \
    omnivoice

# Pre-download model weights during construction phase to prevent severe cold starts
RUN python -c "from omnivoice import OmniVoice; OmniVoice.from_pretrained('k2-fsa/OmniVoice')"

# Copy the entire web server architecture code into the container
COPY app.py /app/app.py

# Expose the target network routing port for the RunPod Load Balancer integration
EXPOSE 8000

# Fire up Uvicorn web daemon in unbuffered output mode
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
