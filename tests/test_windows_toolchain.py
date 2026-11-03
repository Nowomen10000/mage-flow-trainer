import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from trainer.tools import windows_toolchain as tools


def test_non_windows_does_no_discovery(monkeypatch):
    monkeypatch.setattr(tools, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(tools, "_find_scripts", lambda env: pytest.fail("unexpected discovery"))
    assert tools.prepare_windows_toolchain(required=True)


def test_active_environment_is_preserved(monkeypatch):
    monkeypatch.setattr(tools, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(tools, "_ready", lambda env: True)
    monkeypatch.setattr(tools, "_find_scripts", lambda env: pytest.fail("already configured"))
    before = dict(os.environ)
    assert tools.prepare_windows_toolchain(required=True)
    assert dict(os.environ) == before


def test_import_is_independent_of_torch_and_qt():
    subprocess.run([sys.executable, "-c",
                    "import sys; import trainer.tools.windows_toolchain; "
                    "assert not any(n.startswith(('torch', 'PySide6')) for n in sys.modules)"],
                   check=True, timeout=15)


def test_ready_requires_paths_and_headers(monkeypatch):
    monkeypatch.setattr(tools.shutil, "which", lambda name, path: "cl.exe" if path == "tools" else None)
    assert not tools._ready({"Path": "tools"})
    assert tools._ready({"Path": "tools", "INCLUDE": "headers", "LIB": "libs"})
    assert not tools._ready({"Path": "tools", "INCLUDE": "headers", "LIB": "libs",
                             "VSCMD_ARG_TGT_ARCH": "x86"})


def test_discovery_filters_cpp_tools_and_handles_unicode(monkeypatch, tmp_path):
    finder = tmp_path / "Microsoft Visual Studio/Installer/vswhere.exe"
    finder.parent.mkdir(parents=True)
    finder.touch()
    monkeypatch.setattr(tools.shutil, "which", lambda name: None)
    def run(args, **kwargs):
        assert str(finder) == args[0]
        assert "Microsoft.VisualStudio.Component.VC.Tools.x86.x64" in args
        assert "-products" in args and "*" in args and "-utf8" in args
        assert "-latest" not in args  # allow a complete older install
        return subprocess.CompletedProcess(args, 0, "D:\\开发 tools\nD:\\Older\n".encode("utf-8"))
    monkeypatch.setattr(tools.subprocess, "run", run)
    found = tools._find_scripts({"ProgramFiles(x86)": str(tmp_path)})
    assert len(found) == 2 and "开发 tools" in str(found[0])


def test_capture_uses_quoted_script_and_unicode_environment(monkeypatch):
    script = Path("D:/Dev tools & SDK/构建/VsDevCmd.bat")
    def run(command, **kwargs):
        assert '/d /u /v:off /s /c ""%BLUFLOW_VSDEVCMD%"' in command
        assert kwargs["env"]["BLUFLOW_VSDEVCMD"] == str(script)
        assert str(script) not in command
        assert "shell" not in kwargs
        text = "=C:=C:\\work\r\nPath=D:\\开发 tools\r\nINCLUDE=header=dir\r\nLIB=libs\r\nBLUFLOW_VSDEVCMD=temporary\r\n"
        return subprocess.CompletedProcess(command, 0, text.encode("utf-16-le"))
    monkeypatch.setattr(tools.subprocess, "run", run)
    env = tools._capture_environment(script, {"COMSPEC": r"C:\Windows\System32\cmd.exe"})
    assert env == {"Path": "D:\\开发 tools", "INCLUDE": "header=dir", "LIB": "libs"}


def test_activation_tries_older_install_and_only_applies_valid_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "sys", SimpleNamespace(platform="win32"))
    scripts = [tmp_path / "broken.bat", tmp_path / "working.bat"]
    for p in scripts:
        p.touch()
    monkeypatch.setattr(tools, "_find_scripts", lambda env: scripts)
    monkeypatch.setattr(tools, "_ready", lambda env: env.get("TEST_MSVC_READY") == "yes")
    def capture(script, env):
        assert "SHOULD_NOT_APPLY" not in env
        return ({"SHOULD_NOT_APPLY": "yes"} if script == scripts[0]
                else {"TEST_MSVC_READY": "yes"})
    monkeypatch.setattr(tools, "_capture_environment", capture)
    monkeypatch.setattr(tools.os, "environ", {})
    assert tools.prepare_windows_toolchain(required=True)
    assert tools.os.environ == {"TEST_MSVC_READY": "yes"}


def test_missing_tools_fail_clearly_only_when_required(monkeypatch):
    monkeypatch.setattr(tools, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(tools, "_ready", lambda env: False)
    monkeypatch.setattr(tools, "_find_scripts", lambda env: [])
    assert not tools.prepare_windows_toolchain()
    with pytest.raises(RuntimeError, match="Desktop development with C"):
        tools.prepare_windows_toolchain(required=True)


@pytest.mark.parametrize("settings", [dict(attention_backend="torch_varlen"),
                                     dict(pack_resolutions=True),
                                     dict(attention_backend="torch_varlen", pack_resolutions=True)])
def test_windows_cli_blocks_varlen_and_packing(monkeypatch, settings):
    from trainer.training.config import TrainConfig
    monkeypatch.setattr(tools, "sys", SimpleNamespace(platform="win32"))
    with pytest.raises(ValueError, match="temporarily disabled on Windows"):
        TrainConfig(model_family="moss_image", **settings)
    assert TrainConfig(model_family="moss_image", batch_size={"512": 4, "1024": 2}).batch_size[512] == 4


def test_preflight_reports_failure_without_torch_import(monkeypatch, tmp_path, capsys):
    config = tmp_path / "config.toml"
    config.write_text('[train]\ncompile="default"\n')
    monkeypatch.setattr(tools.sys, "argv", ["toolchain", str(config)])
    def fail(*, required):
        assert required
        raise RuntimeError(tools.INSTALL_HELP)
    monkeypatch.setattr(tools, "prepare_windows_toolchain", fail)
    assert tools.main() == 1
    assert "Desktop development with C++" in capsys.readouterr().err


@pytest.mark.skipif(os.name != "nt", reason="real Windows integration")
def test_real_windows_environment_capture():
    scripts = tools._find_scripts(os.environ)
    if not scripts:
        pytest.skip("C++ Build Tools not installed")
    captured = tools._capture_environment(scripts[0], os.environ)
    assert tools._ready(captured)
