"""Activate an installed x64 MSVC environment for this process and its children.

Uses Microsoft's installer discovery without editing the system PATH. Compile
preflight offers missing English resources through the elevated VS installer.
Kept independent of Torch/Qt so GUI jobs can check before caching.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tomllib


INSTALL_HELP = (
    "Windows C++ build tools are unavailable. In Visual Studio Installer, install/modify "
    "Build Tools with 'Desktop development with C++', including MSVC x64/x86 tools "
    "and a Windows 10/11 SDK. The full Visual Studio IDE is not required. "
    "Installer: https://visualstudio.microsoft.com/visual-cpp-build-tools/ . "
    "Then restart the trainer; it will load the environment automatically. "
    "If discovery still fails, launch from 'x64 Native Tools Command Prompt'."
)

ENGLISH_HELP = (
    "MSVC English compiler resources are missing. torch.compile requires the English "
    "language pack even when Windows/Visual Studio uses another language. "
    "Use 'Install English C++ language pack' in the trainer, or run "
    "python -m trainer.tools.windows_toolchain --install-english. "
    "Alternatively: Visual Studio Installer > Modify > Language packs > English."
)


def check_windows_cpp_language(*, install_missing=False):
    """Select English compiler diagnostics without changing the user's UI language.

    Inspect files rather than decoding localized cl /help output: that decoding is
    itself the failure in PyTorch on some Windows code pages.
    """
    if sys.platform != "win32":
        return
    compiler = shutil.which(os.environ.get("CXX", "cl.exe"))
    if not compiler:
        raise RuntimeError(INSTALL_HELP)
    if Path(compiler).name.lower() not in ("cl", "cl.exe"):
        return  # Do not impose MSVC requirements on an explicitly selected compiler.
    if not (Path(compiler).parent / "1033" / "clui.dll").is_file():
        if install_missing:
            print("MSVC English compiler resources are missing; starting Visual Studio Installer before training.", flush=True)
            install_windows_cpp_english()
            return
        raise RuntimeError(ENGLISH_HELP + f" Selected compiler: {compiler}")
    os.environ["VSLANG"] = "1033"
    print("MSVC English compiler resources selected (VSLANG=1033).", flush=True)


def install_windows_cpp_english():
    """Repair missing resources; Windows owns elevation/consent and installation UI."""
    if sys.platform != "win32":
        raise RuntimeError("The MSVC language pack installer is Windows-only.")
    prepare_windows_toolchain(required=True)
    compiler = shutil.which(os.environ.get("CXX", "cl.exe"))
    if not compiler or Path(compiler).name.lower() not in ("cl", "cl.exe"):
        raise RuntimeError("Select an installed MSVC compiler before installing its language pack.")
    installation = next((p for p in Path(compiler).parents
                         if (p / "Common7/Tools/VsDevCmd.bat").is_file()), None)
    if installation is None:
        raise RuntimeError("Cannot locate the Visual Studio installation owning this compiler. " + ENGLISH_HELP)
    if (Path(compiler).parent / "1033/clui.dll").is_file():
        check_windows_cpp_language()
        print("English compiler resources are already installed.", flush=True)
        return
    installer = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio/Installer/setup.exe"
    if not installer.is_file():
        raise RuntimeError("Visual Studio Installer was not found. " + ENGLISH_HELP)
    env = dict(os.environ)
    env["BLUFLOW_VS_INSTALLER"] = str(installer)
    env["BLUFLOW_VS_ARGUMENTS"] = subprocess.list2cmdline([
        "modify", "--installPath", str(installation), "--addProductLang", "en-US",
        "--passive", "--norestart"])
    # Paths travel as environment data, never interpolated PowerShell source.
    command = ("$ErrorActionPreference='Stop'; "
               "$p=Start-Process -FilePath $env:BLUFLOW_VS_INSTALLER "
               "-ArgumentList $env:BLUFLOW_VS_ARGUMENTS -Verb RunAs "
               "-WindowStyle Hidden -PassThru -Wait; exit $p.ExitCode")
    powershell = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    print(f"Adding English compiler resources to {installation}. Approve the Windows administrator prompt.", flush=True)
    result = subprocess.run([str(powershell), "-NoProfile", "-Command", command], env=env)
    if result.returncode == 3010:
        raise RuntimeError("English language pack installed; Windows requests a restart. Restart before training.")
    if result.returncode:
        raise RuntimeError(f"Visual Studio language pack installation failed/cancelled (exit {result.returncode}). " + ENGLISH_HELP)
    # A modification can also service the toolset. Refresh paths after installation.
    for script in _find_scripts(os.environ):
        if script == installation / "Common7/Tools/VsDevCmd.bat":
            os.environ.update(_capture_environment(script, os.environ))
            break
    check_windows_cpp_language()
    print("English language pack installed; compiler resources are ready.", flush=True)


def configure_windows_triton_compiler():
    """Keep Triton's C launcher compiler separate from Inductor's C++ compiler.

    Triton Windows prefers an activated MSVC environment over its bundled
    TinyCC. Older MSVC (including 14.38) rejects the ``= {}`` initializers in
    Triton 3.6's generated C launcher. Explicit CC choices remain authoritative.
    Call before importing Torch/SDNQ, which may initialize Triton's cached CC.
    """
    if sys.platform != "win32" or "CC" in os.environ:
        return
    compiler = Path(sysconfig.get_path("platlib")) / "triton/runtime/tcc/tcc.exe"
    if compiler.is_file():
        os.environ["CC"] = str(compiler)
        # Also handle callers that imported Triton before this helper.
        build = sys.modules.get("triton.runtime.build")
        if build is not None and hasattr(build, "get_cc"):
            build.get_cc.cache_clear()
        print(f"Triton C launcher compiler: {compiler} (MSVC retained for C++)", flush=True)


def check_windows_triton_compiler():
    """Compile/load a tiny C extension before caching, without loading a model.

    Exercises the empty initializer syntax and Python headers/import library
    that the real Triton launcher needs. This is not a GPU/kernel capability test.
    """
    if sys.platform != "win32":
        return
    configure_windows_triton_compiler()
    try:
        from triton.runtime.build import compile_module_from_src, get_cc
        compiler = get_cc()
        # Include compiler identity in the source/cache key so a previous pass
        # with TinyCC cannot conceal a newly selected, incompatible MSVC.
        resolved = Path(shutil.which(compiler) or compiler)
        identity = repr((str(resolved), resolved.stat().st_mtime_ns))
        source = r'''
#include <Python.h>
typedef struct { int id; union { int value; long long other; } data; } Attr;
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_bluflow_c_probe", NULL, -1, NULL
};
PyMODINIT_FUNC PyInit__bluflow_c_probe(void) {
    Attr first = {};
    Attr second = {};
    if (first.id || second.data.value) return NULL;
    return PyModule_Create(&module);
}
'''
        compile_module_from_src(source + "\n// " + identity + "\n", "_bluflow_c_probe")
        print(f"Triton C launcher compile/load check passed: {compiler}", flush=True)
    except Exception as exc:
        raise RuntimeError(
            "Triton could not compile/load its Windows C launcher. "
            "If CC selects an older cl.exe (e.g. MSVC 14.38), unset CC to use "
            "triton-windows' bundled TinyCC, or select an updated MSVC toolset. "
            "Keep MSVC installed for torch.compile C++ code. Restore the repository's "
            "pinned triton-windows package if its bundled compiler is missing. "
            f"Original error: {exc}"
        ) from exc


def validate_windows_attention(backend, pack_resolutions):
    if sys.platform == "win32" and (backend == "torch_varlen" or (pack_resolutions and backend != "flash_attn_2")):
        raise ValueError(
            "torch_varlen and packing without flash_attn_2 are temporarily disabled on Windows. "
            "Use train.attention_backend='flash_attn_2' with a compatible Flash Attention wheel "
            "for MOSS packed training, or 'sdpa' with train.pack_resolutions=false. "
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
    configure_windows_triton_compiler()
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
    parser.add_argument("config", type=Path, nargs="?")
    parser.add_argument("--install-english", action="store_true",
                        help="add the English MSVC language pack using Visual Studio Installer (administrator approval)")
    args = parser.parse_args()
    if args.install_english:
        try:
            install_windows_cpp_english()
        except (RuntimeError, OSError) as exc:
            print(str(exc), file=sys.stderr, flush=True)
            return 1
        return 0
    if args.config is None:
        parser.error("provide a config, or --install-english")
    with args.config.open("rb") as stream:
        train = tomllib.load(stream).get("train", {})
    required = bool(train.get("compile") or train.get("compile_text_encoder"))
    try:
        validate_windows_attention(train.get("attention_backend", "sdpa"), train.get("pack_resolutions", False))
        prepare_windows_toolchain(required=required)
        if required:
            check_windows_cpp_language(install_missing=True)
        check_windows_triton_compiler()
    except (RuntimeError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 1
    print("Windows training environment check passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
