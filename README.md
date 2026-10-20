---
title: Ethiopian MMS TTS API
emoji: 🎙
colorFrom: yellow
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🇪🇹 Ethiopian MMS Speech Synthesis API

Fast, high-fidelity speech synthesis backend running Meta's MMS VITS checkpoints for Ethiopian languages:
- **Amharic (አማርኛ)**: `facebook/mms-tts-amh`
- **Afaan Oromoo**: `facebook/mms-tts-orm`
- **Tigrinya (ትግርኛ)**: `facebook/mms-tts-tir`
- **Somali**: `facebook/mms-tts-som`

## Deployment on Hugging Face Spaces (Free CPU)

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space).
2. Space name: `ethiopian-mms-tts`
3. License: `MIT`
4. Space SDK: **Docker** -> **Blank**
5. Hardware: **CPU basic · 2 vCPU · 16 GB RAM (Free)**
6. Clone the space repo or upload the files from this `hf_space` folder:
   - `app.py`
   - `romanizer.py`
   - `requirements.txt`
   - `Dockerfile`
   - `README.md`
7. Once built, your Space URL will be:
   `https://<YOUR-HF-USERNAME>-ethiopian-mms-tts.hf.space`

## API Usage

### `POST /api/synthesize`
```json
{
  "language": "amh",
  "text": "ሰላም ጤና ይስጥልኝ እንኳን ደህና መጡ",
  "voice_id": "mms-narrator",
  "speed": 1.0,
  "noise_scale": 0.55
}
```
Response:
```json
{
  "status": "success",
  "audio_base64": "...",
  "format": "wav",
  "duration": 3.45,
  "sample_rate": 16000
}
```

