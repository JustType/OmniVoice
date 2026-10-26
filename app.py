import os
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from omnivoice import OmniVoice

# Initialize FastAPI application
app = FastAPI(title="OmniVoice Streaming Serverless Endpoint")

# Global model variable to hold OmniVoice weights in VRAM
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

@app.get("/health")
def health_check():
    """Health check endpoint for the RunPod Load Balancer."""
    if model is list or model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {"status": "healthy"}

@app.post("/v1/synthesize")
async def synthesize(payload: TTSRequest):
    """Processes incoming text and streams raw PCM audio chunks back to the client."""
    if not payload.text:
        raise HTTPException(status_code=400, detail="The 'text' field cannot be empty.")

    def audio_streamer():
        try:
            # Check if OmniVoice native generator supports streaming tokens chunk-by-chunk.
            # If your framework package generates a singular array, fall back to slicing it into chunks.
            if hasattr(model, "generate_stream"):
                print(f"Streaming text synthesis via generate_stream: {payload.text}")
                for audio_chunk in model.generate_stream(text=payload.text, instruct=payload.instruct):
                    yield audio_chunk.tobytes()
            else:
                print(f"Synthesizing full text array and chunking bytes: {payload.text}")
                full_audio = model.generate(text=payload.text, instruct=payload.instruct)
                
                # Convert the numpy float array to 16-bit PCM bytes for LiveKit compatibility
                if full_audio.dtype != np.int16:
                    full_audio = (full_audio * 32767).astype(np.int16)
                
                # Stream the audio in small 20ms chunks (960 bytes for 24kHz 16-bit mono)
                chunk_size = 960 
                audio_bytes = full_audio.tobytes()
                for i in range(0, len(audio_bytes), chunk_size):
                    yield audio_bytes[i:i + chunk_size]
                    
        except Exception as e:
            print(f"Inference error during generation streaming: {str(e)}")
            # End generation early on unexpected failures
            return

    # Return an active chunked HTTP streaming response
    return StreamingResponse(audio_streamer(), media_type="audio/pcm")

if __name__ == "__main__":
    import uvicorn
    # Bind to 0.0.0.0 so external load balancers can reach the app container internally
    uvicorn.run(app, host="0.0.0.0", port=8000)
