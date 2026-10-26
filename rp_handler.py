import os
import io
import base64
import tempfile
import torch
import numpy as np
import soundfile as sf
import runpod
from omnivoice import OmniVoice

# Global model variable to persist across warm worker activations
model = None

def load_model():
    """Initializes the OmniVoice model onto the GPU."""
    global model
    if model is None:
        print("Loading OmniVoice model weights into VRAM...")
        model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice", 
            device_map="cuda:0", 
            dtype=torch.float16
        )
        print("Model successfully loaded.")

def handler(event):
    """Processes incoming API text-to-speech payloads."""
    load_model()
    
    input_data = event.get('input', {})
    text = input_data.get('text', '')
    ref_audio_b64 = input_data.get('ref_audio', None)
    ref_text = input_data.get('ref_text', '')
    instruct = input_data.get('instruct', None)
    
    if not text:
        return {"error": "The 'text' parameter is required."}
        
    temp_ref_path = None
    try:
        # Strategy A: Zero-Shot Voice Cloning
        if ref_audio_b64:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(base64.b64decode(ref_audio_b64))
                temp_ref_path = tmp.name
            
            print(f"Generating audio via Voice Cloning for: {text}")
            audio = model.generate(text=text, ref_audio=temp_ref_path, ref_text=ref_text)
            
        # Strategy B: Attribute-Based Voice Design
        elif instruct:
            print(f"Generating audio via Voice Design ('{instruct}') for: {text}")
            audio = model.generate(text=text, instruct=instruct)
            
        # Strategy C: Standard Prompt
        else:
            print(f"Generating audio using default settings for: {text}")
            audio = model.generate(text=text)
            
        # Convert generated numpy array (sampled at 24kHz) to base64 WAV
        out_bio = io.BytesIO()
        sf.write(out_bio, audio[0], 24000, format='WAV')
        out_bio.seek(0)
        out_b64 = base64.b64encode(out_bio.read()).decode('utf-8')
        
        return {"audio": out_b64}
        
    except Exception as e:
        return {"error": str(e)}
        
    finally:
        # Clean up transient audio files from system memory
        if temp_ref_path and os.path.exists(temp_ref_path):
            os.remove(temp_ref_path)

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
