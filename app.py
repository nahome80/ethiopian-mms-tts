"""
Meta MMS Ethiopian TTS Inference API (Hugging Face Space)
Supports Amharic (amh), Afaan Oromoo (orm), Tigrinya (tir), and Somali (som).
"""

import os
import io
import time
import base64
import logging
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

import torch
import numpy as np
import scipy.io.wavfile as wav
from transformers import VitsModel, AutoTokenizer

from romanizer import romanize_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mms_space")

app = FastAPI(
    title="Ethiopian MMS TTS API",
    description="Meta MMS Speech Synthesis backend for Ethiopian Languages on Hugging Face Spaces"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MMS_MODELS = {
    "amh": "facebook/mms-tts-amh",
    "orm": "facebook/mms-tts-orm",
    "tir": "facebook/mms-tts-tir",
    "som": "facebook/mms-tts-som",
}

GEEZ_LANGS = {"amh", "tir"}

MMS_VOICE_PROFILES = {
    "mms-narrator": {
        "id": "mms-narrator",
        "name": "Classic Narrator (Standard)",
        "pitch_shift": 0.0,
        "speed_mult": 1.0,
        "noise_scale": 0.55,
    },
    "mms-deep-male": {
        "id": "mms-deep-male",
        "name": "Deep Baritone (Elder Male)",
        "pitch_shift": -2.8,
        "speed_mult": 0.95,
        "noise_scale": 0.50,
    },
    "mms-bright-female": {
        "id": "mms-bright-female",
        "name": "Bright / High Register",
        "pitch_shift": +2.8,
        "speed_mult": 1.05,
        "noise_scale": 0.55,
    },
    "mms-calm": {
        "id": "mms-calm",
        "name": "Calm Documentary Narrator",
        "pitch_shift": -0.8,
        "speed_mult": 0.92,
        "noise_scale": 0.38,
    },
    "mms-expressive": {
        "id": "mms-expressive",
        "name": "Dynamic Storyteller",
        "pitch_shift": 0.0,
        "speed_mult": 1.0,
        "noise_scale": 0.72,
    }
}

# In-memory model cache
loaded_models: Dict[str, VitsModel] = {}
loaded_tokenizers: Dict[str, AutoTokenizer] = {}

def get_model(lang_id: str):
    if lang_id not in loaded_models:
        model_name = MMS_MODELS[lang_id]
        logger.info(f"Loading MMS checkpoint: {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = VitsModel.from_pretrained(model_name)
        model.eval()
        loaded_tokenizers[lang_id] = tokenizer
        loaded_models[lang_id] = model
    return loaded_models[lang_id], loaded_tokenizers[lang_id]

class SynthesisPayload(BaseModel):
    language: str = Field(default="amh", description="'amh', 'orm', 'tir', or 'som'")
    text: str = Field(..., min_length=1, max_length=2000)
    voice_id: Optional[str] = Field(default="mms-narrator")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    noise_scale: float = Field(default=0.55, ge=0.1, le=1.0)
    seed: Optional[int] = Field(default=555)

@app.get("/")
@app.get("/health")
def health():
    return {
        "status": "online",
        "service": "Ethiopian MMS TTS Space API",
        "supported_languages": list(MMS_MODELS.keys()),
        "cached_models": list(loaded_models.keys()),
    }

@app.post("/api/synthesize")
@app.post("/synthesize")
async def synthesize(payload: SynthesisPayload, raw: bool = False):
    lang_id = payload.language.lower()
    if lang_id not in MMS_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{lang_id}'. Supported: {list(MMS_MODELS.keys())}"
        )

    start_time = time.time()
    profile = MMS_VOICE_PROFILES.get(str(payload.voice_id or ""), MMS_VOICE_PROFILES["mms-narrator"])

    effective_noise = profile["noise_scale"] if payload.voice_id in MMS_VOICE_PROFILES else payload.noise_scale
    effective_speed = profile["speed_mult"] * payload.speed
    pitch_semitones = profile.get("pitch_shift", 0.0)

    # 1. Text Romanization (uroman for Ge'ez languages)
    is_geez = lang_id in GEEZ_LANGS
    processed_text = romanize_text(payload.text, lang_id) if is_geez else payload.text.lower()

    # 2. Model inference
    model, tokenizer = get_model(lang_id)

    if payload.seed is not None:
        torch.manual_seed(payload.seed)

    if hasattr(model, "speaking_rate"):
        model.speaking_rate = float(effective_speed)
    if hasattr(model, "noise_scale"):
        model.noise_scale = float(effective_noise)

    inputs = tokenizer(processed_text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)
        waveform = output.waveform[0].cpu().numpy()

    sample_rate = model.config.sampling_rate

    # 3. Pitch shifting via linear interpolation resampling
    if abs(pitch_semitones) > 0.01:
        ratio = 2.0 ** (pitch_semitones / 12.0)
        orig_len = len(waveform)
        new_len = max(1, int(round(orig_len / ratio)))
        waveform = np.interp(
            np.linspace(0, orig_len - 1, new_len),
            np.arange(orig_len),
            waveform
        ).astype(np.float32)

    # 4. 16-bit PCM WAV export
    waveform_norm = np.clip(waveform, -1.0, 1.0)
    audio_int16 = (waveform_norm * 32767).astype(np.int16)

    wav_buf = io.BytesIO()
    wav.write(wav_buf, sample_rate, audio_int16)
    wav_bytes = wav_buf.getvalue()

    duration = round(len(waveform) / sample_rate, 2)
    elapsed = round(time.time() - start_time, 3)

    if raw:
        return Response(content=wav_bytes, media_type="audio/wav")

    audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
    return JSONResponse({
        "status": "success",
        "audio_base64": audio_b64,
        "format": "wav",
        "sample_rate": sample_rate,
        "duration": duration,
        "generation_time": elapsed,
        "processed_text": processed_text,
        "original_text": payload.text,
        "language": lang_id,
        "model": MMS_MODELS[lang_id],
        "engine": f"Meta MMS VITS — {profile['name']}"
    })

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)

