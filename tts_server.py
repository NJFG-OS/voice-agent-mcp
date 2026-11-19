import os
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="AetherVoice CosyVoice TTS")


class TTSRequest(BaseModel):
    text: str
    voice_id: str = "default"
    mode: str = "zero_shot"
    max_music_seconds: int = 90
    rights_verified: bool = True


@app.get("/health")
def health():
    model_dir = os.getenv("AETHER_COSYVOICE_MODEL_DIR", "/workspace/CosyVoice_models")
    return {
        "status": "ok",
        "engine": "CosyVoice",
        "model_dir": model_dir,
        "model_dir_exists": os.path.exists(model_dir),
    }


@app.post("/synthesize")
def synthesize(req: TTSRequest):
    return {
        "status": "queued",
        "voice_id": req.voice_id,
        "mode": req.mode,
        "text_length": len(req.text),
        "rights_verified": req.rights_verified,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("AETHER_TTS_PORT", "8003")))
