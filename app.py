import os
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from omnivoice import OmniVoice

app = FastAPI(title="OmniVoice Streaming Serverless Endpoint")
model = None

@app.on_event("startup")
def load_model():
    """Initializes the OmniVoice model onto the GPU during container startup."""
    global model
    if model is None:
        print("Loading OmniVoice model weights into VRAM (FP16)...")
        try:
            model = OmniVoice.from_pretrained(
                "k2-fsa/OmniVoice", 
                device_map="cuda:0", 
                dtype=torch.float16
            )
            print("OmniVoice model successfully loaded and ready.")
        except Exception as e:
            print(f"Failed to load model on startup: {str(e)}")
            raise e

class TTSRequest(BaseModel):
    text: str
    instruct: Optional[str] = None

@app.get("/ping")
async def health_check():
    """
    CRITICAL: RunPod Load Balancer health check.
    Returns 200 OK once the weights are active in VRAM.
    """
    if model is None:
        # Returning 503 keeps it out of the routing pool while warming up
        raise HTTPException(status_code=503, detail="Model initializing")
    return {"status": "healthy"}

@app.post("/v1/synthesize")
async def synthesize(payload: TTSRequest):
    """Processes incoming text and streams raw PCM audio chunks back to the client."""
    if not payload.text:
        raise HTTPException(status_code=400, detail="The 'text' field cannot be empty.")

    def audio_streamer():
        try:
            if hasattr(model, "generate_stream"):
                print(f"Streaming text synthesis via generate_stream: {payload.text}")
                for audio_chunk in model.generate_stream(text=payload.text, instruct=payload.instruct):
                    yield audio_chunk.tobytes()
            else:
                print(f"Synthesizing full text array and chunking bytes: {payload.text}")
                full_audio = model.generate(text=payload.text, instruct=payload.instruct)
                
                if full_audio.dtype != np.int16:
                    full_audio = (full_audio * 32767).astype(np.int16)
                
                chunk_size = 960 
                audio_bytes = full_audio.tobytes()
                for i in range(0, len(audio_bytes), chunk_size):
                    yield audio_bytes[i:i + chunk_size]
                    
        except Exception as e:
            print(f"Inference error during generation streaming: {str(e)}")
            return

    return StreamingResponse(audio_streamer(), media_type="audio/pcm")

if __name__ == "__main__":
    import uvicorn
    # Use the port environment variable assigned by RunPod or fallback to 8000
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
