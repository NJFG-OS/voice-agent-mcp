#!/usr/bin/env python3
"""Build-time runtime pin selection for the AetherVoice pod image.

This runs during `docker build`, never on a pod. It decides which runtime the image
ships, so no pod ever installs or replaces anything at boot.

Two gates run, in this order, and the build fails if either has no passing candidate:

1. THE BRAIN — the ladder from runtime-candidates.json: the base image being built
   and the ordered vLLM versions declared for it. Each candidate is installed in
   full (`pip install vllm==<version>`) and then must pass the release gate:

       import torch
       import vllm
       import vllm._C

   `vllm._C` is the compiled CUDA extension: if it imports, the wheel was built
   against the very torch and CUDA this image ships, which is the mismatch that
   used to surface on a paid GPU instead of here.

   The gate deliberately asserts no private torch symbol. torch.library does not
   export infer_schema in torch 2.4.x and vLLM never imports it, so requiring it
   rejected every candidate and turned every healthy pod boot into
   vllm_import_failed. Only what the runtime actually needs is gated.

2. THE EAR AND THE MOUTH — `voice` in the same manifest: faster-whisper (required)
   and at least one TTS engine from the declared list. Both are imported in the
   image's own interpreter, so a pod can never boot without them and then report
   tts:false in its health contract.

A candidate that fails is uninstalled and the next one is tried. When every
candidate fails the script exits non-zero, the `RUN` step fails, the image is never
produced and nothing is published. Fail closed, by construction.
"""
import argparse
import json
import os
import subprocess
import sys

GATE = (
    "import torch\n"
    "import vllm\n"
    "import vllm._C\n"
    "print(torch.__version__, vllm.__version__)\n"
)

WHISPER_GATE = (
    "import faster_whisper\n"
    "print(getattr(faster_whisper, '__version__', 'unknown'))\n"
)

TTS_GATE = (
    "import importlib\n"
    "m = importlib.import_module('{module}')\n"
    "print(getattr(m, '__version__', 'unknown'))\n"
)

INSTALL_TIMEOUT = 3600
GATE_TIMEOUT = 900


def run(cmd, timeout=INSTALL_TIMEOUT):
    """Run one step, never raising: a timeout is just another failure."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, "", "timed out after %ss" % timeout
    except OSError as error:
        return 127, "", str(error)
    return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()


def load_manifest(manifest_path):
    with open(manifest_path) as handle:
        return json.load(handle)


def ladder_for(data, base):
    """The ordered vLLM candidates declared for exactly this base image."""
    for entry in data.get("bases") or []:
        if str(entry.get("image") or "").strip() == base:
            return [str(row).strip() for row in entry.get("vllm") or [] if str(row).strip()]
    raise SystemExit(
        "select_runtime_pins: %s is not a declared base image" % base
    )


def torch_version():
    code, out, _ = run([sys.executable, "-c", "import torch; print(torch.__version__)"], timeout=300)
    return out.splitlines()[-1].strip() if code == 0 and out else "unavailable"


def install_voice(pins, voice):
    """The Ear and the Mouth, installed and imported in the image's interpreter.

    faster-whisper is required: the relay's own health contract reports
    whisper_ready, and a pod without it can never transcribe a call. At least one
    TTS engine is required for the same reason (tts in the same contract).
    """
    whisper = str((voice or {}).get("whisper") or "").strip()
    if not whisper:
        raise SystemExit("select_runtime_pins: no faster-whisper candidate is declared")

    print("select_runtime_pins: installing %s" % whisper)
    code, out, err = run([sys.executable, "-m", "pip", "install", "--no-cache-dir", whisper])
    if code != 0:
        print("select_runtime_pins: %s failed to install: %s" % (whisper, (err or out)[-800:]), file=sys.stderr)
        return 1
    code, out, err = run([sys.executable, "-c", WHISPER_GATE], timeout=GATE_TIMEOUT)
    if code != 0:
        print("select_runtime_pins: %s failed the import gate: %s" % (whisper, (err or out)[-800:]), file=sys.stderr)
        return 1
    pins["whisper"] = whisper
    pins["whisper_version"] = out.splitlines()[-1].strip() if out else "unknown"

    rejected = []
    for candidate in (voice or {}).get("tts") or []:
        package = str(candidate.get("package") or "").strip()
        module = str(candidate.get("module") or "").strip()
        if not package or not module:
            continue
        print("select_runtime_pins: trying tts %s" % package)
        code, out, err = run([sys.executable, "-m", "pip", "install", "--no-cache-dir", package])
        if code != 0:
            rejected.append({"package": package, "stage": "install", "detail": (err or out)[-600:]})
            print("select_runtime_pins: %s failed to install" % package)
            continue
        code, out, err = run(
            [sys.executable, "-c", TTS_GATE.format(module=module)], timeout=GATE_TIMEOUT
        )
        if code != 0:
            rejected.append({"package": package, "stage": "gate", "detail": (err or out)[-600:]})
            print("select_runtime_pins: %s failed the import gate" % package)
            continue
        pins["tts"] = package
        pins["tts_module"] = module
        pins["tts_version"] = out.splitlines()[-1].strip() if out else "unknown"
        pins["tts_rejected"] = rejected
        return 0

    pins["tts_rejected"] = rejected
    print("select_runtime_pins: no TTS engine passed the import gate", file=sys.stderr)
    return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="/opt/aether/runtime-candidates.json")
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", default="/opt/aether/runtime-pins.json")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the ordered candidates and exit without installing anything",
    )
    args = parser.parse_args()

    data = load_manifest(args.manifest)
    ladder = ladder_for(data, args.base)
    if not ladder:
        raise SystemExit("select_runtime_pins: no vLLM candidate is declared for %s" % args.base)

    if args.dry_run:
        # One row per vLLM candidate and nothing else: the ladder is machine-read
        # as "image<TAB>version", so voice candidates go to stderr.
        for version in ladder:
            print("%s\t%s" % (args.base, version))
        for candidate in (data.get("voice") or {}).get("tts") or []:
            print(
                "select_runtime_pins: voice candidate %s" % candidate.get("package"),
                file=sys.stderr,
            )
        return 0

    before = torch_version()
    print("select_runtime_pins: base=%s torch_before_vllm=%s" % (args.base, before))

    rejected = []
    for version in ladder:
        print("select_runtime_pins: trying vllm==%s" % version)
        code, out, err = run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "vllm==" + version])
        if code != 0:
            rejected.append({"vllm": version, "stage": "install", "detail": (err or out)[-1200:]})
            print("select_runtime_pins: vllm==%s failed to install" % version)
            continue

        code, out, err = run([sys.executable, "-c", GATE], timeout=GATE_TIMEOUT)
        if code != 0:
            rejected.append({"vllm": version, "stage": "gate", "detail": (err or out)[-1200:]})
            print("select_runtime_pins: vllm==%s failed the import gate" % version)
            run([sys.executable, "-m", "pip", "uninstall", "-y", "vllm"], timeout=600)
            continue

        fields = out.split()
        pins = {
            "base_image": args.base,
            "torch": fields[0] if fields else "unknown",
            "torch_before_vllm": before,
            "vllm": fields[1] if len(fields) > 1 else version,
            "gate": "torch + vllm + vllm._C",
            "selected_from": ladder,
            "rejected": rejected,
        }

        # The Brain is pinned; now the Ear and the Mouth. A voice failure fails the
        # build, so an image that cannot transcribe or speak is never published.
        if install_voice(pins, data.get("voice")) != 0:
            return 1

        directory = os.path.dirname(args.out)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(pins, handle, indent=2, sort_keys=True)
        print(
            "select_runtime_pins: selected torch=%s vllm=%s whisper=%s tts=%s -> %s"
            % (pins["torch"], pins["vllm"], pins["whisper"], pins["tts"], args.out)
        )
        return 0

    print("select_runtime_pins: no candidate passed the import gate", file=sys.stderr)
    for row in rejected:
        last = row["detail"].splitlines()[-1] if row["detail"] else "no detail"
        print("  vllm==%s (%s): %s" % (row["vllm"], row["stage"], last), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())