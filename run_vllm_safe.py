import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.getenv("AETHER_VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("AETHER_VLLM_PORT", "8002")),
    )
    parser.add_argument(
        "--dtype",
        default=os.getenv("AETHER_VLLM_DTYPE", "float16"),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.getenv("AETHER_GPU_MEM_UTIL", "0.85")),
    )
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--host",
        "0.0.0.0",
        "--port",
        str(args.port),
        "--dtype",
        args.dtype,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
