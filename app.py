import os
import torch
import numpy as np
import asyncio
from fastapi import FastAPI, HTTPException, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from omnivoice import OmniVoice

app = FastAPI(title="OmniVoice Streaming Serverless Endpoint")

# Tracking states for RunPod's health monitoring
model = None
is_ready = False

def load_model_background():
    """Loads weights in a separate context to prevent locking the web port during startup."""
    global model, is_ready
    print("Loading OmniVoice model weights into VRAM (FP16)...")
    try:
        model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice", 
            device_map="cuda:0", 
            dtype=torch.float16
        )
        is_ready = True
        print("OmniVoice model successfully loaded. Ready for traffic.")
    except Exception as e:
        print(f"CRITICAL: Failed to load model weights: {str(e)}")

@app.on_event("startup")
async def startup_event():
    # Trigger weight loading asynchronously so Uvicorn can immediately bind to port 8000
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, load_model_background)

class TTSRequest(BaseModel):
    text: str
    instruct: Optional[str] = None

@app.get("/ping")
async def health_check():
    """
    RunPod Load Balancer health check handler.
    - Returns 204 while the model is loading (keeps container alive in the pool).
    - Returns 200 when ready to receive real-time text from your LiveKit Agent.
    """
    if not is_ready:
        # 204 explicitly signals 'Initializing' to RunPod's load balancer gateway
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    
    return {"status": "healthy"}

@app.post("/v1/synthesize")
async def synthesize(payload: TTSRequest):
    if not is_ready:
        raise HTTPException(status_code=503, detail="Model is still warming up.")
    if not payload.text:
        raise HTTPException(status_code=400, detail="The 'text' field cannot be empty.")

    def audio_streamer():
        try:
            if hasattr(model, "generate_stream"):
                for audio_chunk in model.generate_stream(text=payload.text, instruct=payload.instruct):
                    yield audio_chunk.tobytes()
            else:
                full_audio = model.generate(text=payload.text, instruct=payload.instruct)
                if full_audio.dtype != np.int16:
                    full_audio = (full_audio * 32767).astype(np.int16)
                
                chunk_size = 960 
                audio_bytes = full_audio.tobytes()
                for i in range(0, len(audio_bytes), chunk_size):
                    yield audio_bytes[i:i + chunk_size]
        except Exception as e:
            print(f"Inference error during generation: {str(e)}")
            return

    return StreamingResponse(audio_streamer(), media_type="audio/pcm")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
