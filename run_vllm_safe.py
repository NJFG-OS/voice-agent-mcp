#!/usr/bin/env python3
"""AetherVoice vLLM launcher - image torch and vLLM, profile sized to the GPU."""
import json
import os
import re
import signal
import subprocess
import sys
import time
import types
import urllib.request

# MiB, never whole GB: a 24GB card reports 24564 MiB and rounding that to 23GB
# rejected an RTX 4090 with VLLM_GPU_TOO_SMALL. The floor leaves headroom for the
# MiB the driver reserves.
MIN_VRAM_MB = int(os.environ.get("AETHER_MIN_VRAM_MB") or 23000)

# Where the mode that actually started is recorded. The relay reads this file before
# it routes a completion, so a base fallback is never sent to a LoRA alias.
MODE_FILE = os.environ.get("AETHER_VLLM_MODE_FILE") or os.path.join(
    os.environ.get("AETHER_LOG_DIR", "/tmp/aether-logs"), "vllm_mode.json"
)

# What a PEFT adapter directory must carry before vLLM is asked to load it.
ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin", "adapter_model.safetensors.index.json")


STAGE_FILE = os.path.join(os.environ.get("AETHER_LOG_DIR", "/tmp/aether-logs"), "stage.json")


def stage(name, detail=""):
    """Announce a launch step, merged into the stage file start_all.sh owns."""
    payload = {"stage": name, "detail": (detail or "")[:400], "since": stamp()}
    try:
        folder = os.path.dirname(STAGE_FILE)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(STAGE_FILE + ".tmp", "w") as handle:
            json.dump(payload, handle)
        os.replace(STAGE_FILE + ".tmp", STAGE_FILE)
    except Exception:
        pass
    print("run_vllm_safe: stage=" + name + (" detail=" + detail if detail else ""), flush=True)


def model_candidates():
    """Every model the app offered, in its own order, plus the explicit fallback."""
    prefs = [row.strip() for row in (os.environ.get("AETHER_VLLM_MODEL_PREFS") or "").split(",") if row.strip()]
    explicit = (os.environ.get("AETHER_VLLM_MODEL") or "").strip()
    if explicit and explicit not in prefs:
        prefs.append(explicit)
    if not prefs:
        prefs = ["Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"]
    return prefs


def reachable(repo_id, timeout=20):
    """Can this repo's config.json be fetched right now?

    A gated repo with no token answers 401/403, and vLLM then retries the download
    until the boot window expires: that is a 20+ minute stall that reads as a hung
    engine. Probing first turns it into one log line and moves to the next candidate.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    request = urllib.request.Request("https://huggingface.co/" + repo_id + "/resolve/main/config.json", method="HEAD")
    if token:
        request.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status < 400, "http_" + str(response.status)
    except Exception as error:
        code = getattr(error, "code", None)
        return False, ("http_" + str(code)) if code else type(error).__name__


def resolve_model():
    """The first candidate Hugging Face will actually serve; the app's order decides.

    Only the FIRST preference used to be read, so a gated model without a token
    stalled the boot with no fallback even though the app had offered a second one.
    """
    candidates = model_candidates()
    for repo_id in candidates:
        ok, why = reachable(repo_id)
        print("run_vllm_safe: model candidate " + repo_id + " -> " + ("ok" if ok else why), flush=True)
        if ok:
            if repo_id != candidates[0]:
                print("run_vllm_safe: " + candidates[0] + " is not servable - using " + repo_id, flush=True)
            return repo_id
    print(
        "run_vllm_safe: no candidate is reachable - trying " + candidates[0] + " anyway (vllm.log will show the download)",
        file=sys.stderr,
        flush=True,
    )
    return candidates[0]


# MODEL is resolved below, once stamp() exists: the probe writes a stage record.
PORT = os.environ.get("AETHER_VLLM_PORT", "8002")
HF_HOME = os.environ.get("HF_HOME", "/workspace/hf-cache")
# The 24GB profile. Stated, not derived: a smaller context and four sequences leave
# the rest of the card for Whisper and the speech engine, which share it.
GPU_UTIL = os.environ.get("AETHER_GPU_MEMORY_UTILIZATION") or os.environ.get("AETHER_VLLM_GPU_UTIL") or "0.65"
MAX_MODEL_LEN = int(
    os.environ.get("AETHER_MAX_MODEL_LEN_24") or os.environ.get("AETHER_VLLM_MAX_MODEL_LEN") or 4096
)
MAX_NUM_SEQS = int(
    os.environ.get("AETHER_MAX_NUM_SEQS_24") or os.environ.get("AETHER_VLLM_MAX_NUM_SEQS") or 4
)


def _query(field):
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + field, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def gpu_vram_mb():
    """Total VRAM in MiB, exactly as nvidia-smi reports it - no GB rounding."""
    numbers = [int(n) for n in re.findall(r"\d+", _query("memory.total"))]
    return max(numbers) if numbers else None


def as_gb(mb):
    return round(mb / 1024.0, 1)


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# Resolved once, here: the probe writes a stage record, so stamp() must exist first.
stage("model_resolve", "probing the app's model preferences")
MODEL = resolve_model()


def write_mode(mode, detail, adapter=""):
    """Record the mode the engine is being started in, atomically.

    The relay reads this before it rewrites a completion's model name: with the
    pointer still on disk after a fallback, routing to the alias would 404 every
    call, so the file - not the pointer - is the routing truth.
    """
    payload = {"mode": mode, "detail": detail or "", "adapter": adapter or "", "at": stamp()}
    try:
        folder = os.path.dirname(MODE_FILE)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(MODE_FILE + ".tmp", "w") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(MODE_FILE + ".tmp", MODE_FILE)
    except Exception as error:
        print("run_vllm_safe: mode file not written (" + str(error)[:120] + ")", flush=True)
    print("run_vllm_safe: mode=" + mode + " detail=" + (detail or "none"), flush=True)


def adapter_problem(path):
    """What is wrong with this adapter directory, or "" when it looks loadable.

    Only local evidence is available before the engine starts: the config and the
    weight file. A directory that fails this is never handed to vLLM, because the
    failure would take the whole engine down instead of falling back.
    """
    if not path or not os.path.isdir(path):
        return "missing_directory"
    config_path = os.path.join(path, "adapter_config.json")
    if not os.path.isfile(config_path):
        return "missing_adapter_config"
    if not any(os.path.isfile(os.path.join(path, name)) for name in ADAPTER_WEIGHTS):
        return "missing_adapter_weights"
    try:
        with open(config_path, "r") as handle:
            config = json.load(handle)
    except Exception:
        return "unreadable_adapter_config"
    if not isinstance(config, dict) or not config:
        return "adapter_config_incomplete"
    return ""


def lora_flags():
    """This tenant's validated adapter, if one has been promoted.

    Returns (flags, adapter_path, problem). The pointer file is the ONE source of
    truth for which adapter is promoted; the mode file is the ONE source of truth for
    what actually loaded. No approved adapter means no flag and the base model serves.
    """
    tenant = os.environ.get("AETHER_TENANT_ID", "").strip()
    if not tenant:
        return [], "", "no_tenant"
    root = os.environ.get("AETHER_TENANT_ROOT", "/workspace/tenants")
    pointer = os.path.join(root, tenant, "adapters", "llm_lora", "current_adapter.json")
    try:
        with open(pointer, "r") as handle:
            data = json.load(handle)
    except Exception:
        return [], "", "no_pointer"
    path = str(data.get("adapter_path") or "")
    problem = adapter_problem(path)
    if problem:
        print("run_vllm_safe: adapter rejected before launch - " + problem + " (" + path + ")", flush=True)
        return [], path, problem
    alias = re.sub(r"[^a-z0-9_]", "_", tenant.lower())[:40]
    print("run_vllm_safe: lora alias=" + alias + " adapter=" + path, flush=True)
    return ["--enable-lora", "--max-lora-rank", "16", "--lora-modules", alias + "=" + path], path, ""


def neutralise_deep_gemm():
    """A stub, not a crash: vLLM imports deep_gemm unconditionally on some builds."""
    try:
        import deep_gemm  # noqa: F401
        return "present"
    except Exception as error:
        stub = types.ModuleType("deep_gemm")

        def _unavailable(*_args, **_kwargs):
            raise RuntimeError("deep_gemm is unavailable on this image")

        stub.__getattr__ = _unavailable
        sys.modules["deep_gemm"] = stub
        print("run_vllm_safe: deep_gemm stubbed (" + str(error)[:120] + ")", flush=True)
        return "stubbed"


def build_cmd(extra):
    return [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--host", "0.0.0.0",
        "--model", MODEL,
        "--port", str(PORT),
        "--gpu-memory-utilization", GPU_UTIL,
        "--max-model-len", str(MAX_MODEL_LEN),
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--enforce-eager",
    ] + extra


def healthy(timeout=2.0):
    """vLLM answers /health only once its API server is up."""
    try:
        with urllib.request.urlopen("http://127.0.0.1:" + str(PORT) + "/health", timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def supervise(cmd, env):
    """Run vLLM in the foreground. Returns (exit_code, ever_answered_health).

    A child, not an exec, so a failed LoRA start can be retried without it. Signals
    are forwarded, so a launcher that is stopped takes the engine with it instead of
    leaving an orphan holding the GPU.
    """
    print("run_vllm_safe: launch " + " ".join(cmd[1:]), flush=True)
    child = subprocess.Popen(cmd, env=env)

    def _forward(signum, _frame):
        try:
            child.terminate()
        except Exception:
            pass
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _forward)
        except Exception:
            pass

    became = False
    started = time.time()
    last_note = started
    while True:
        code = child.poll()
        if code is not None:
            return code, became
        if not became and healthy():
            became = True
            print("run_vllm_safe: vLLM answered /health", flush=True)
        # A one-line liveness note a minute. Without it a 15GB weight download and a
        # hung engine look identical in the log, which is how a stall went unnoticed.
        if not became and (time.time() - last_note) >= 60:
            last_note = time.time()
            print(
                "run_vllm_safe: engine still starting after %ds (no /health on %s yet)"
                % (int(time.time() - started), PORT),
                flush=True,
            )
        time.sleep(5)


def main():
    vram_mb = gpu_vram_mb()
    if vram_mb is None:
        print("run_vllm_safe: VLLM_GPU_UNREADABLE - nvidia-smi reported no memory total", file=sys.stderr, flush=True)
        return 2
    if vram_mb < MIN_VRAM_MB:
        print(
            "run_vllm_safe: VLLM_GPU_TOO_SMALL - need >=%dMiB (24GB class) for %s bf16, got %dMiB (%.1fGB)"
            % (MIN_VRAM_MB, MODEL, vram_mb, as_gb(vram_mb)),
            file=sys.stderr,
            flush=True,
        )
        return 3

    deep_gemm = neutralise_deep_gemm()

    env = dict(os.environ)
    env["HF_HOME"] = HF_HOME
    # Caches on the volume: a warm boot must not recompile the model from scratch.
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(HF_HOME, "inductor"))
    env.setdefault("TRITON_CACHE_DIR", os.path.join(HF_HOME, "triton"))
    # The mandatory torch.compile fix: no dynamo, no inductor, so the duplicate
    # template name crash cannot occur.
    env.setdefault("TORCHDYNAMO_DISABLE", "1")

    print(
        "run_vllm_safe: vram=%.1fGB (%dMiB) gpu_util=%s max_model_len=%d max_seqs=%d deep_gemm=%s python=%s"
        % (as_gb(vram_mb), vram_mb, GPU_UTIL, MAX_MODEL_LEN, MAX_NUM_SEQS, deep_gemm, sys.executable),
        flush=True,
    )
    print("run_vllm_safe: pythonpath=" + (os.environ.get("PYTHONPATH") or "none"), flush=True)

    lora, adapter, problem = lora_flags()
    if not lora:
        # Nothing loadable to attach: the base model serves and the mode file says so.
        stage("vllm_starting", "base model " + MODEL + " (" + (problem or "no_adapter") + ")")
        write_mode("base", problem or "no_adapter", adapter)
        code, _ = supervise(build_cmd([]), env)
        return code if code else 0

    stage("vllm_starting", "base model " + MODEL + " + tenant adapter")
    write_mode("lora", "starting", adapter)
    code, became = supervise(build_cmd(lora), env)
    if became:
        write_mode("lora", "ready", adapter)
        return code if code else 0

    # The LoRA start failed before the engine ever answered. One clean retry, with no
    # LoRA flag of any kind, so a broken adapter costs a restart instead of the engine.
    print(
        "run_vllm_safe: LoRA startup failed (rc=%s) - retrying on the base model without --enable-lora" % code,
        file=sys.stderr,
        flush=True,
    )
    write_mode("base", "lora_start_failed:rc=" + str(code), adapter)
    code2, _ = supervise(build_cmd([]), env)
    return code2 if code2 else 0


if __name__ == "__main__":
    sys.exit(main())
