"""Activate an installed x64 MSVC environment for this process and its children.

Uses Microsoft's installer discovery; never installs software or edits the
system PATH. Kept independent of Torch/Qt so GUI jobs can check before caching.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib


INSTALL_HELP = (
    "Windows C++ build tools are unavailable. In Visual Studio Installer, install/modify "
    "Build Tools with 'Desktop development with C++', including MSVC x64/x86 tools "
    "and a Windows 10/11 SDK. The full Visual Studio IDE is not required. "
    "Installer: https://visualstudio.microsoft.com/visual-cpp-build-tools/ . "
    "Then restart the trainer; it will load the environment automatically. "
    "If discovery still fails, launch from 'x64 Native Tools Command Prompt'."
)


def validate_windows_attention(backend, pack_resolutions):
    if sys.platform == "win32" and (backend == "torch_varlen" or pack_resolutions):
        raise ValueError(
            "torch_varlen and train.pack_resolutions are temporarily disabled on Windows. "
            "Use train.attention_backend='sdpa' and train.pack_resolutions=false. "
            "Ordinary multi-resolution buckets remain supported."
        )


def _ready(env):
    # cl alone on PATH is insufficient: its standard headers/libraries must
    # also be visible. Windows environments are case-insensitive.
    values = {k.upper(): v for k, v in env.items()}
    return bool(shutil.which("cl.exe", path=values.get("PATH", ""))
                and values.get("INCLUDE") and values.get("LIB")
                and values.get("VSCMD_ARG_TGT_ARCH", "x64").lower() == "x64")


def _find_scripts(env):
    roots = [env.get("ProgramFiles(x86)"), env.get("ProgramFiles")]
    candidates = [Path(root) / "Microsoft Visual Studio/Installer/vswhere.exe"
                  for root in roots if root]
    located = shutil.which("vswhere.exe")
    if located:
        candidates.append(Path(located))
    vswhere = next((p for p in candidates if p.is_file()), None)
    if vswhere is None:
        return []
    result = subprocess.run(
        [str(vswhere), "-products", "*", "-requires",
         "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-sort",
         "-property", "installationPath", "-utf8"],
        capture_output=True, check=True, timeout=15, env=env,
    )
    # Try newest first, but allow an older complete installation to work if a
    # newer installation is broken. -products * includes standalone Build Tools.
    return [Path(line.strip()) / "Common7/Tools/VsDevCmd.bat"
            for line in result.stdout.decode("utf-8-sig").splitlines() if line.strip()]


def _capture_environment(script, env):
    child_env = dict(env)
    child_env["BLUFLOW_VSDEVCMD"] = str(script)
    cmd = env.get("COMSPEC") or str(Path(env.get("SystemRoot", r"C:\Windows")) / "System32/cmd.exe")
    if any(c in str(script) + cmd for c in ('"', '\n', '\r')):
        raise RuntimeError("Invalid compiler setup path")
    # A Windows command line, intentionally not shlex/list2cmdline: cmd.exe /s
    # consumes the outer quotes itself. The batch path is quoted and expanded
    # from an environment variable, with no CALL/delayed second expansion.
    # /d ignores shell AutoRun; /u gives lossless UTF-16 output even for Unicode paths.
    command = f'"{cmd}" /d /u /v:off /s /c ""%BLUFLOW_VSDEVCMD%" -no_logo -arch=x64 -host_arch=x64 >nul && set"'
    result = subprocess.run(command, env=child_env, capture_output=True,
                            check=True, timeout=60)
    captured = {}
    for line in result.stdout.decode("utf-16-le").splitlines():
        key, sep, value = line.partition("=")
        # cmd's hidden '=C:' drive entries are not valid os.environ keys.
        if sep and key and key.upper() != "BLUFLOW_VSDEVCMD":
            captured[key] = value
    return captured


def prepare_windows_toolchain(*, required=False):
    """Return readiness; raise an actionable error only when compilation needs it."""
    if sys.platform != "win32":
        return True
    if _ready(os.environ):
        return True
    errors = []
    try:
        scripts = _find_scripts(os.environ)
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        scripts = []
        errors.append(f"MSVC discovery failed ({type(exc).__name__})")
    for script in scripts:
        if not script.is_file():
            continue
        try:
            captured = _capture_environment(script, os.environ)
            if not _ready(captured):
                errors.append(f"Incomplete C++ environment from {script}")
                continue
            # Do not partially change our process if initialization failed.
            os.environ.update(captured)
            print(f"Loaded x64 MSVC build environment from {script}", flush=True)
            return True
        except (OSError, subprocess.SubprocessError, UnicodeError, RuntimeError) as exc:
            errors.append(f"Could not load {script} ({type(exc).__name__})")
    if required:
        detail = "\n" + "\n".join(errors) if errors else ""
        raise RuntimeError(INSTALL_HELP + detail)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    with args.config.open("rb") as stream:
        train = tomllib.load(stream).get("train", {})
    required = bool(train.get("compile") or train.get("compile_text_encoder"))
    try:
        validate_windows_attention(train.get("attention_backend", "sdpa"), train.get("pack_resolutions", False))
        prepare_windows_toolchain(required=required)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 1
    print("Windows training environment check passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
