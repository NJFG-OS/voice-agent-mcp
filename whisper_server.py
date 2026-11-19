import os
from fastapi import FastAPI, File, UploadFile
import uvicorn

app = FastAPI(title="AetherVoice Whisper STT")


@app.get("/health")
def health():
    return {"status": "ok", "engine": "faster-whisper"}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    data = await file.read()
    return {
        "filename": file.filename,
        "bytes_received": len(data),
        "text": "",
        "status": "ready_for_model_weights",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("AETHER_WHISPER_PORT", "8001")))
