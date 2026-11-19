import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI(title="AetherVoice Relay & Multi-Tenant Gateway")

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
VLLM_URL = f"http://127.0.0.1:{os.getenv('AETHER_VLLM_PORT', '8002')}"
TTS_URL = f"http://127.0.0.1:{os.getenv('AETHER_TTS_PORT', '8003')}"
WHISPER_URL = f"http://127.0.0.1:{os.getenv('AETHER_WHISPER_PORT', '8001')}"


class AgentInvokeRequest(BaseModel):
    agent_id: str
    workspace_id: str
    message: str
    voice_id: Optional[str] = None
    system_prompt: Optional[str] = None
    variables: Dict[str, Any] = {}
    guardrail_mode: str = "standard"
    tools: List[Dict[str, Any]] = []
    metering_target: str = "NJFG_CRM"


def verify_secret(x_webhook_secret: Optional[str]):
    if WEBHOOK_SECRET and x_webhook_secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Invalid webhook secret")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "aether-voice-relay",
        "vllm_url": VLLM_URL,
        "tts_url": TTS_URL,
        "whisper_url": WHISPER_URL,
    }


@app.post("/v1/agents/{agent_id}/invoke")
def invoke_agent(
    agent_id: str,
    req: AgentInvokeRequest,
    x_webhook_secret: Optional[str] = Header(default=None, alias="x-webhook-secret"),
):
    verify_secret(x_webhook_secret)
    messages = []
    if req.system_prompt:
        messages.append({"role": "system", "content": req.system_prompt})
    messages.append({"role": "user", "content": req.message})

    try:
        models_resp = requests.get(f"{VLLM_URL}/v1/models", timeout=3)
        active_model = (
            models_resp.json()["data"][0]["id"]
            if models_resp.ok and models_resp.json().get("data")
            else os.getenv("AETHER_VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        )
        chat_resp = requests.post(
            f"{VLLM_URL}/v1/chat/completions",
            json={"model": active_model, "messages": messages, "temperature": 0.4},
            timeout=30,
        )
        reply = (
            chat_resp.json()["choices"][0]["message"]["content"]
            if chat_resp.ok
            else "vLLM unreachable"
        )
    except Exception as exc:
        reply = f"Inference error: {exc}"

    return {
        "agent_id": agent_id,
        "workspace_id": req.workspace_id,
        "voice_id": req.voice_id,
        "guardrail_mode": req.guardrail_mode,
        "metering_target": req.metering_target,
        "reply": reply,
    }
