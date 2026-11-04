"""Install the tested Windows Flash Attention build without changing Torch."""
from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys


BASE_URL = "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.13"
HASHES = {
    (3, 11): "ae6b7d34c79afbf84d9b63a0b48d384fae1961e04d69153d49b198a4afff9c98",
    (3, 12): "9ca9d69be86ad501e28a2123478b45b5a833a5d253cc14c69915a46e809f4d0e",
    (3, 13): "6fee83a69fa77b4bd858122545582a24de6bd9dbe96e0de39f9cbfea292d0d47",
}


def wheel_url(system, machine, implementation, python_version, torch_version, cuda):
    if system != "win32" or machine.lower() not in ("amd64", "x86_64"):
        raise ValueError("Flash Attention installer requires Windows x86_64.")
    if implementation != "cpython" or python_version not in HASHES:
        raise ValueError("Flash Attention wheels require CPython 3.11, 3.12 or 3.13.")
    if torch_version != "2.10.0+cu128" or cuda != "12.8":
        raise ValueError("Flash Attention wheel requires the pinned Torch 2.10.0+cu128 stack.")
    tag = f"cp{python_version[0]}{python_version[1]}"
    return (f"{BASE_URL}/flash_attn-2.8.3+cu128torch2.10-{tag}-{tag}-win_amd64.whl"
            f"#sha256={HASHES[python_version]}")


def install_command(url):
    uv = shutil.which("uv")
    if uv:
        return [uv, "pip", "install", "--python", sys.executable, "--no-deps", url]
    return [sys.executable, "-m", "pip", "install", "--no-deps", url]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Validate compatibility and print the command only")
    args = parser.parse_args()
    try:
        import torch
        url = wheel_url(sys.platform, platform.machine(), sys.implementation.name,
                        sys.version_info[:2], torch.__version__, torch.version.cuda)
        command = install_command(url)
        print(subprocess.list2cmdline(command), flush=True)
        if args.dry_run:
            return 0
        if sys.prefix == sys.base_prefix:
            raise ValueError("Run this installer with the trainer's venv Python.")
        subprocess.run(command, check=True)
        # Use a fresh process: replacing an already loaded DLL cannot repair its import.
        subprocess.run([sys.executable, "-c",
                        "import torch; from flash_attn import flash_attn_varlen_func; "
                        "import flash_attn_2_cuda; print('Flash Attention native extension loaded.')"],
                       check=True)
        return 0
    except (ImportError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Flash Attention installation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
