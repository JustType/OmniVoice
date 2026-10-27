FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y libsndfile1 git && rm -rf /var/lib/apt/lists/*

# Upstream master after 0.2.1: it adds FlashInfer-accelerated decoding (omnivoice_flashinfer).
ARG OMNIVOICE_REF=08be0b4ccbac3e13e374e86fbfead4b4cac343e2
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic soundfile numpy "brotli>=1.2.0" \
    "omnivoice @ git+https://github.com/k2-fsa/OmniVoice@${OMNIVOICE_REF}"

# Prebuilt kernels: the runtime image has no CUDA compiler for FlashInfer's JIT.
RUN pip install --no-cache-dir flashinfer-python==0.6.15.post1 "flashinfer-jit-cache==0.6.15.post1+cu128" \
    --extra-index-url https://flashinfer.ai/whl/cu128/

# omnivoice only needs torch>=2.4: fail the build if pip replaced the image's CUDA build of torch
RUN python -c "import torch, torchaudio; assert torch.__version__.startswith('2.8'), torch.__version__"

# Pre-bake weights so the background load finishes within a few seconds
RUN python -c "from omnivoice import OmniVoice; OmniVoice.from_pretrained('k2-fsa/OmniVoice')"

# The weights are in the image, so a cold start never waits on the Hugging Face Hub.
# PORT / PORT_HEALTH must also be set in the RunPod endpoint's environment variables.
ENV HF_HUB_OFFLINE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    OMNIVOICE_NUM_STEP=16 \
    PORT=8000 \
    PORT_HEALTH=8000

COPY voices/ /app/voices/
COPY server/ /app/server/
COPY app.py /app/app.py

EXPOSE 8000

CMD ["python", "app.py"]
