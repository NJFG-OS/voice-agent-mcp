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
   tts:false in its health contract. Their transformers is then pinned.

3. NUMPY — the one pin that can break all of the above at once. It is installed last
   and the WHOLE stack is imported together; a pin that fails is rolled back to the
   numpy the image already had and recorded. Optional extras (pyarrow, datasets) are
   baselined first and only re-required when they already imported, so a package that
   was broken in the base image never fails a numpy change it did not cause.

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

TRANSFORMERS_GATE = (
    "import transformers\n"
    "print(transformers.__version__)\n"
)

NUMPY_GATE = (
    "import numpy\n"
    "print(numpy.__version__)\n"
)


def stack_gate_source(module, extras=()):
    """Everything that shares this numpy, imported together in one interpreter.

    A numpy pin is only safe when the WHOLE stack still imports with it: torch 2.4.x
    wheels were built against numpy 1.x, and vLLM, transformers, faster-whisper and
    the TTS engine all share the same array ABI.

    `extras` is the list of optional packages that ALREADY imported before the pin
    (pyarrow, datasets). Only those are required again afterwards: a package that was
    already broken in the base image is reported, never turned into a build failure
    for a numpy change that did not cause it.
    """
    lines = [
        "import importlib",
        "import numpy, torch, vllm, vllm._C, transformers, faster_whisper",
    ]
    if module:
        lines.append("importlib.import_module(%r)" % module)
    for name in extras or ():
        lines.append("importlib.import_module(%r)" % name)
    lines.append("print(numpy.__version__, torch.__version__, vllm.__version__, transformers.__version__)")
    return "\n".join(lines) + "\n"


# The optional training extras. They are NOT part of the inference contract, so they
# are baselined before the pin and only re-required when they already imported.
EXTRAS = ("pyarrow", "datasets")


def extras_state():
    """Which optional extras import in this image right now, before any pin."""
    probe = (
        "import importlib, importlib.util, json\n"
        "out = {}\n"
        "for name in %r:\n"
        "    if importlib.util.find_spec(name) is None:\n"
        "        out[name] = 'absent'\n"
        "        continue\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "        out[name] = 'ok'\n"
        "    except Exception as error:\n"
        "        out[name] = type(error).__name__\n"
        "print(json.dumps(out))\n"
    ) % (EXTRAS,)
    code, out, _ = run([sys.executable, "-c", probe], timeout=GATE_TIMEOUT)
    if code != 0 or not out:
        return {}
    try:
        return json.loads(out.splitlines()[-1])
    except Exception:
        return {}

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


def declared_torch(data, base):
    """The torch LABEL the manifest states for this base, or "" when unstated."""
    for entry in data.get("bases") or []:
        if str(entry.get("image") or "").strip() == base:
            return str(entry.get("torch") or "").strip()
    return ""


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
        if pin_transformers(pins, voice) != 0:
            return 1
        return pin_numpy(pins, voice)

    pins["tts_rejected"] = rejected
    print("select_runtime_pins: no TTS engine passed the import gate", file=sys.stderr)
    return 1


def pin_transformers(pins, voice):
    """The Mouth's transformers, pinned AFTER the engines are installed.

    Kokoro's warm-up failed on the transformers its own wheel pulled in, and the
    in-container fix (transformers==4.47.1) lived only as long as that container:
    every fresh pod hit the same failure again. Installing the pin last makes it
    win over whatever the TTS wheel resolved, and re-importing the chosen engine
    afterwards proves the fix holds with the engine already present. A failure
    here fails the build, so an image that cannot speak is never published.
    """
    package = str((voice or {}).get("transformers") or "").strip()
    if not package:
        return 0
    print("select_runtime_pins: installing %s (tts runtime pin)" % package)
    code, out, err = run([sys.executable, "-m", "pip", "install", "--no-cache-dir", package])
    if code != 0:
        print(
            "select_runtime_pins: %s failed to install: %s" % (package, (err or out)[-800:]),
            file=sys.stderr,
        )
        return 1
    code, out, err = run([sys.executable, "-c", TRANSFORMERS_GATE], timeout=GATE_TIMEOUT)
    if code != 0:
        print(
            "select_runtime_pins: %s failed the import gate: %s" % (package, (err or out)[-800:]),
            file=sys.stderr,
        )
        return 1
    pins["transformers"] = package
    pins["transformers_version"] = out.splitlines()[-1].strip() if out else "unknown"

    module = str(pins.get("tts_module") or "").strip()
    if module:
        code, out, err = run([sys.executable, "-c", TTS_GATE.format(module=module)], timeout=GATE_TIMEOUT)
        if code != 0:
            print(
                "select_runtime_pins: %s no longer imports with %s: %s"
                % (module, package, (err or out)[-800:]),
                file=sys.stderr,
            )
            return 1
    return 0


def installed_numpy():
    """The numpy version this image already imports, or "" when it has none."""
    code, out, _ = run([sys.executable, "-c", NUMPY_GATE], timeout=300)
    return out.splitlines()[-1].strip() if code == 0 and out else ""


def pin_numpy(pins, voice):
    """The one pin that can break torch, vLLM, transformers and training at once.

    numpy 2.x is what pyarrow and datasets resolve against, while torch 2.4.x wheels
    were built against 1.x. So the pin is installed LAST, after the engines and their
    transformers, and then the whole stack is imported together. A pin that fails
    that gate is rolled back to the numpy the image already had, which is recorded -
    the build only fails when even the rollback cannot import the stack, because that
    means the image is broken whatever the pin says.
    """
    package = str((voice or {}).get("numpy") or "").strip()
    if not package:
        return 0
    module = str(pins.get("tts_module") or "")
    before = installed_numpy()
    # What already imports TODAY. A package that is broken in the base image (datasets
    # against the hub transformers 4.47.1 pins, pyarrow against numpy 1.x) is reported
    # in the pins and never counted against the numpy pin that did not cause it.
    baseline = extras_state()
    required = [name for name, state in baseline.items() if state == "ok"]
    pins["numpy_extras"] = baseline
    if any(state not in ("ok", "absent") for state in baseline.values()):
        print("select_runtime_pins: extras already broken before the pin: %s" % baseline)
    print("select_runtime_pins: installing %s (image has %s)" % (package, before or "none"))

    code, out, err = run([sys.executable, "-m", "pip", "install", "--no-cache-dir", package])
    if code != 0:
        print(
            "select_runtime_pins: %s failed to install: %s" % (package, (err or out)[-800:]),
            file=sys.stderr,
        )
        return 1

    code, out, err = run([sys.executable, "-c", stack_gate_source(module, required)], timeout=GATE_TIMEOUT)
    if code == 0:
        fields = out.split()
        pins["numpy"] = fields[0] if fields else package
        pins["numpy_source"] = "pinned"
        pins["numpy_stack"] = out.strip()
        print("select_runtime_pins: numpy %s holds the whole stack (%s)" % (pins["numpy"], out.strip()))
        return 0

    detail = (err or out)[-1200:]
    print(
        "select_runtime_pins: %s broke the stack import - rolling back: %s" % (package, detail),
        file=sys.stderr,
    )

    restore = ("numpy==" + before) if before else "numpy<2"
    code, out, err = run([sys.executable, "-m", "pip", "install", "--no-cache-dir", restore])
    if code != 0:
        print(
            "select_runtime_pins: could not restore %s: %s" % (restore, (err or out)[-800:]),
            file=sys.stderr,
        )
        return 1

    code, out, err = run([sys.executable, "-c", stack_gate_source(module, required)], timeout=GATE_TIMEOUT)
    if code != 0:
        print(
            "select_runtime_pins: the image's own numpy no longer imports the stack: %s" % ((err or out)[-800:]),
            file=sys.stderr,
        )
        return 1
    fields = out.split()
    pins["numpy"] = fields[0] if fields else before or "unknown"
    pins["numpy_source"] = "rolled_back"
    pins["numpy_rejected"] = {"pin": package, "detail": detail}
    pins["numpy_stack"] = out.strip()
    print(
        "select_runtime_pins: kept numpy %s (%s) - the pin %s cannot import this runtime"
        % (pins["numpy"], pins["numpy_source"], package),
        file=sys.stderr,
    )
    return 0


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
    declared = declared_torch(data, args.base)
    print("select_runtime_pins: base=%s torch_before_vllm=%s declared=%s" % (args.base, before, declared or "unstated"))
    # The label is documentation, never a gate: a base image's tag and its torch can
    # disagree (the cuda12.4.1 tag ships torch 2.4.0+cu121), and the image wins.
    if declared and declared not in ("unverified", before):
        print(
            "select_runtime_pins: WARNING manifest declares torch %s but the image ships %s - recording the image's own"
            % (declared, before),
            file=sys.stderr,
        )

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
            "declared_torch": declared,
            "torch_matches_manifest": (fields[0] == declared) if declared and declared != "unverified" else None,
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