#!/usr/bin/env python3
"""AetherVoice vLLM launcher - image torch and vLLM, profile sized to the GPU."""
import glob
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
# 8002 is a template-declared port, so the provider's proxy dials the pod's own
# interface and a loopback bind leaves <pod>-8002.proxy.runpod.net unable to connect.
# The relay keeps reaching the engine on 127.0.0.1 either way.
# AETHER_VLLM_HOST=127.0.0.1 closes the public engine and leaves the relay as the
# only way in.
HOST = os.environ.get("AETHER_VLLM_HOST", "0.0.0.0")
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


def hf_cache_dir(repo_id):
    """This model's own cache folder - the only one the launcher ever touches."""
    owner, _, name = repo_id.partition("/")
    return os.path.join(HF_HOME, "hub", "models--" + owner + "--" + (name or owner))


def snapshot_dirs(repo_id):
    return sorted(glob.glob(os.path.join(hf_cache_dir(repo_id), "snapshots", "*")))


def cache_problem(repo_id):
    """What is wrong with the cached snapshot of this model, or "" when it is usable.

    Only local evidence is checked, before vLLM starts: the config it parses first
    and the weight files it maps. A 0-byte config left by an interrupted download is
    what made the engine die in json.load with "Expecting value: line 1 column 1
    (char 0)" on every later boot, because the hub keeps serving it as a cache hit.
    """
    folders = snapshot_dirs(repo_id)
    if not folders:
        return "cold_cache"
    problems = []
    for folder in folders:
        config = os.path.join(folder, "config.json")
        if not os.path.isfile(config):
            problems.append("missing_config")
            continue
        try:
            size = os.path.getsize(config)
        except Exception:
            size = 0
        if size < 2:
            problems.append("empty_config")
            continue
        try:
            with open(config) as handle:
                data = json.load(handle)
        except Exception:
            problems.append("unreadable_config")
            continue
        if not isinstance(data, dict) or not data:
            problems.append("empty_config_object")
            continue
        try:
            names = sorted(os.listdir(folder))
        except Exception:
            problems.append("unreadable_snapshot")
            continue
        weights = [n for n in names if n.endswith(".safetensors") or n.endswith(".bin")]
        if not weights:
            problems.append("no_weights")
            continue
        for name in weights:
            try:
                if os.path.getsize(os.path.join(folder, name)) < 1024:
                    problems.append("empty_weights")
                    break
            except Exception:
                problems.append("unreadable_weights")
                break
    return "" if not problems else ",".join(sorted(set(problems)))


def repair_cache(repo_id, problem):
    """Drop ONLY the artifacts the problem names, so a warm cache is never thrown away.

    A missing or empty config is removed and re-fetched; a bad weight file is removed
    with its snapshot link so the next load downloads it again. Nothing outside this
    model's own snapshot folders is ever touched.
    """
    removed = 0
    for folder in snapshot_dirs(repo_id):
        try:
            names = sorted(os.listdir(folder))
        except Exception:
            continue
        for name in names:
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                continue
            try:
                size = os.path.getsize(path)
            except Exception:
                size = 0
            drop = False
            if name == "config.json" and "config" in problem:
                drop = True
            elif name.endswith(".json") and size == 0:
                drop = True
            elif (name.endswith(".safetensors") or name.endswith(".bin")) and (size < 1024 or "weights" in problem):
                drop = True
            if not drop:
                continue
            target = os.path.realpath(path) if os.path.islink(path) else ""
            for candidate in (path, target):
                if not candidate:
                    continue
                try:
                    os.remove(candidate)
                    removed += 1
                except Exception:
                    pass
    return removed


def fetch_config(repo_id):
    """Re-fetch config.json when the cache cannot supply one, with the pod's token."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return ""
    try:
        hf_hub_download(repo_id=repo_id, filename="config.json", token=token or None)
        return ""
    except Exception as error:
        return "config.json download failed: " + str(error)[:140]


def prepare_model_cache(repo_id):
    """Validate (and repair) this model's cached snapshot before vLLM reads it."""
    problem = cache_problem(repo_id)
    if not problem:
        print("run_vllm_safe: model cache ok for " + repo_id + " (" + hf_cache_dir(repo_id) + ")", flush=True)
        return ""
    print("run_vllm_safe: model cache problem for " + repo_id + ": " + problem, flush=True)
    if problem == "cold_cache":
        print("run_vllm_safe: no snapshot yet - vLLM downloads the weights on this boot", flush=True)
        return problem
    removed = repair_cache(repo_id, problem)
    print("run_vllm_safe: dropped " + str(removed) + " cached artifact(s); vLLM downloads them again", flush=True)
    if "config" in problem:
        note = fetch_config(repo_id)
        if note:
            print("run_vllm_safe: " + note, flush=True)
    return problem


def enable_safetensors_prefetch(env):
    """Turn on the prefetch weight loader, but only on a vLLM that reads the flag.

    Setting an env var the installed vLLM never reads is a silent no-op, so the flag
    is looked for in the engine's own sources first and left alone when absent.
    """
    try:
        import vllm
    except Exception:
        return ""
    root = os.path.dirname(vllm.__file__)
    paths = [os.path.join(root, "envs.py")]
    for base, _dirs, files in os.walk(root):
        for name in files:
            if name.endswith(".py") and "weight_utils" in name:
                paths.append(os.path.join(base, name))
    for path in paths:
        try:
            with open(path, errors="replace") as handle:
                text = handle.read()
        except Exception:
            continue
        for key in ("VLLM_SAFETENSORS_LOAD_STRATEGY", "SAFETENSORS_LOAD_STRATEGY"):
            if key in text:
                env[key] = "prefetch"
                print("run_vllm_safe: " + key + "=prefetch (this vLLM supports it)", flush=True)
                return key
    print("run_vllm_safe: this vLLM has no safetensors load strategy flag - default loader", flush=True)
    return ""


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
        "--host", HOST,
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

    # A 7-8B model in bf16 is ~15-16GB of weights BEFORE any KV cache. On a card under
    # 24.6GB (an L4 reports 23034MiB) 0.65 of its memory is less than that, so vLLM
    # loads the weights, finds no room for cache blocks and exits - and the restart
    # loop turned that into a silent 20-minute stall with nothing on 8002. Exactly
    # those cards get a raised profile and a shorter context instead.
    global GPU_UTIL, MAX_MODEL_LEN, MAX_NUM_SEQS
    if vram_mb < 24564 and not os.environ.get("AETHER_PROFILE_OVERRIDE"):
        GPU_UTIL = "0.88"
        MAX_MODEL_LEN = min(MAX_MODEL_LEN, 2048)
        MAX_NUM_SEQS = min(MAX_NUM_SEQS, 2)
        print(
            "run_vllm_safe: %.1fGB card - profile raised to util=%s max_model_len=%d max_seqs=%d "
            "(a 7-8B bf16 model does not fit in 0.65 of this card)"
            % (as_gb(vram_mb), GPU_UTIL, MAX_MODEL_LEN, MAX_NUM_SEQS),
            flush=True,
        )

    deep_gemm = neutralise_deep_gemm()

    env = dict(os.environ)
    env["HF_HOME"] = HF_HOME
    # Caches on the volume: a warm boot must not recompile the model from scratch.
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(HF_HOME, "inductor"))
    env.setdefault("TRITON_CACHE_DIR", os.path.join(HF_HOME, "triton"))
    # The mandatory torch.compile fix: no dynamo, no inductor, so the duplicate
    # template name crash cannot occur.
    env.setdefault("TORCHDYNAMO_DISABLE", "1")
    # A cold load reads ~16GB of safetensors; prefetch overlaps that with the
    # allocation instead of serialising it, where the engine supports it.
    enable_safetensors_prefetch(env)

    print(
        "run_vllm_safe: vram=%.1fGB (%dMiB) gpu_util=%s max_model_len=%d max_seqs=%d deep_gemm=%s python=%s"
        % (as_gb(vram_mb), vram_mb, GPU_UTIL, MAX_MODEL_LEN, MAX_NUM_SEQS, deep_gemm, sys.executable),
        flush=True,
    )
    print("run_vllm_safe: pythonpath=" + (os.environ.get("PYTHONPATH") or "none"), flush=True)

    # The cached snapshot is checked (and repaired) before the engine reads it: a
    # 0-byte config left by an interrupted download is a json.load crash on every
    # later boot, and the engine's own traceback never names the file.
    prepare_model_cache(MODEL)

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
