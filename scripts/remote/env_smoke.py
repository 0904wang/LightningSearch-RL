from __future__ import annotations

import importlib.metadata as metadata

import torch


def main() -> None:
    print("python ok")
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_runtime={torch.version.cuda}")
    print(f"device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"device_name={torch.cuda.get_device_name(0)}")
    for package in ["verl", "vllm", "transformers", "ray"]:
        print(f"{package}={metadata.version(package)}")


if __name__ == "__main__":
    main()
