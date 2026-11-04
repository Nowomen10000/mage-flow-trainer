import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from trainer.tools import windows_toolchain as tools


@pytest.mark.parametrize("language", ["3082", "1041", "2052"])
def test_localized_compiler_requires_english_without_decoding_output(monkeypatch, tmp_path, language):
    compiler = tmp_path / "cl.exe"
    compiler.touch()
    (tmp_path / language).mkdir()
    monkeypatch.setattr(tools, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(tools.shutil, "which", lambda name: str(compiler))
    monkeypatch.setattr(tools.subprocess, "run", lambda *a, **k: pytest.fail("must not decode cl output"))
    with pytest.raises(RuntimeError, match="--install-english"):
        tools.check_windows_cpp_language()
    (tmp_path / "1033").mkdir()
    (tmp_path / "1033/clui.dll").touch()
    monkeypatch.setattr(tools.os, "environ", {"VSLANG": language})
    tools.check_windows_cpp_language()
    assert tools.os.environ["VSLANG"] == "1033"


def test_cpp_language_does_not_touch_linux_or_custom_compiler(monkeypatch):
    monkeypatch.setattr(tools, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(tools.shutil, "which", lambda name: pytest.fail("Linux discovery"))
    tools.check_windows_cpp_language()
    monkeypatch.setattr(tools, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(tools.shutil, "which", lambda name: "D:/tools/clang-cl.exe")
    tools.check_windows_cpp_language()


def test_compile_preflight_installs_missing_english_only_once(monkeypatch, tmp_path):
    compiler = tmp_path / "cl.exe"
    compiler.touch()
    monkeypatch.setattr(tools, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(tools.shutil, "which", lambda name: str(compiler))
    monkeypatch.setattr(tools.os, "environ", {})
    installs = []
    def install():
        installs.append(True)
        (tmp_path / "1033").mkdir()
        (tmp_path / "1033/clui.dll").touch()
        tools.check_windows_cpp_language()
    monkeypatch.setattr(tools, "install_windows_cpp_english", install)
    tools.check_windows_cpp_language(install_missing=True)
    tools.check_windows_cpp_language(install_missing=True)
    assert installs == [True]
    assert tools.os.environ["VSLANG"] == "1033"


def test_failed_auto_install_stops_preflight_before_triton(monkeypatch, tmp_path):
    config = tmp_path / "run.toml"
    config.write_text('[train]\ncompile="default"\n')
    monkeypatch.setattr(tools.sys, "argv", ["toolchain", str(config)])
    monkeypatch.setattr(tools, "prepare_windows_toolchain", lambda **kw: True)
    def check(*, install_missing):
        assert install_missing
        raise RuntimeError("Installation cancelled")
    monkeypatch.setattr(tools, "check_windows_cpp_language", check)
    monkeypatch.setattr(tools, "check_windows_triton_compiler", lambda: pytest.fail("must stop"))
    assert tools.main() == 1


@pytest.mark.parametrize("exit_code", [0, 3010, 1])
def test_english_installer_targets_selected_vs_and_checks_outcome(monkeypatch, tmp_path, exit_code):
    installation = tmp_path / "VS 日本 & tools"
    compiler = installation / "VC/Tools/MSVC/version/bin/Hostx64/x64/cl.exe"
    compiler.parent.mkdir(parents=True)
    compiler.touch()
    script = installation / "Common7/Tools/VsDevCmd.bat"
    script.parent.mkdir(parents=True)
    script.touch()
    installer = tmp_path / "Microsoft Visual Studio/Installer/setup.exe"
    installer.parent.mkdir(parents=True)
    installer.touch()
    monkeypatch.setattr(tools, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(tools.os, "environ", {"ProgramFiles(x86)": str(tmp_path)})
    monkeypatch.setattr(tools.shutil, "which", lambda name: str(compiler))
    monkeypatch.setattr(tools, "prepare_windows_toolchain", lambda **kw: True)
    monkeypatch.setattr(tools, "_find_scripts", lambda env: [])
    def run(args, **kwargs):
        assert str(installation) not in args[-1]  # paths are data, not PowerShell code
        assert "-Verb RunAs" in args[-1] and "-Wait" in args[-1]
        assert kwargs["env"]["BLUFLOW_VS_INSTALLER"] == str(installer)
        assert "--addProductLang en-US" in kwargs["env"]["BLUFLOW_VS_ARGUMENTS"]
        assert str(installation) in kwargs["env"]["BLUFLOW_VS_ARGUMENTS"]
        if exit_code == 0:
            (compiler.parent / "1033").mkdir()
            (compiler.parent / "1033/clui.dll").touch()
        return subprocess.CompletedProcess(args, exit_code)
    monkeypatch.setattr(tools.subprocess, "run", run)
    if exit_code:
        with pytest.raises(RuntimeError, match="restart" if exit_code == 3010 else "failed/cancelled"):
            tools.install_windows_cpp_english()
    else:
        tools.install_windows_cpp_english()
        assert tools.os.environ["VSLANG"] == "1033"


def test_non_windows_does_no_discovery(monkeypatch):
    monkeypatch.setattr(tools, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(tools, "_find_scripts", lambda env: pytest.fail("unexpected discovery"))
    assert tools.prepare_windows_toolchain(required=True)


def test_windows_uses_bundled_c_compiler_without_changing_cpp(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "sys", SimpleNamespace(platform="win32", modules={}))
    compiler = tmp_path / "triton/runtime/tcc/tcc.exe"
    compiler.parent.mkdir(parents=True)
    compiler.touch()
    monkeypatch.setattr(tools.sysconfig, "get_path", lambda key: str(tmp_path))
    env = {"PATH": "msvc", "INCLUDE": "headers", "LIB": "libraries", "CXX": "cl.exe"}
    monkeypatch.setattr(tools.os, "environ", env)
    tools.configure_windows_triton_compiler()
    assert env == {"PATH": "msvc", "INCLUDE": "headers", "LIB": "libraries",
                   "CXX": "cl.exe", "CC": str(compiler)}


@pytest.mark.parametrize("platform,explicit", [("linux", None), ("win32", "my-compiler.exe")])
def test_c_compiler_selection_preserves_user_choice_and_linux(monkeypatch, platform, explicit):
    monkeypatch.setattr(tools, "sys", SimpleNamespace(platform=platform))
    env = {} if explicit is None else {"CC": explicit}
    before = dict(env)
    monkeypatch.setattr(tools.os, "environ", env)
    monkeypatch.setattr(tools.sysconfig, "get_path", lambda key: pytest.fail("must not select compiler"))
    tools.configure_windows_triton_compiler()
    assert env == before


def test_missing_bundled_compiler_keeps_normal_discovery(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(tools.os, "environ", {})
    monkeypatch.setattr(tools.sysconfig, "get_path", lambda key: str(tmp_path))
    tools.configure_windows_triton_compiler()
    assert "CC" not in tools.os.environ


def test_c_probe_reproduces_launcher_syntax_and_reports_selected_compiler(monkeypatch, tmp_path, capsys):
    compiler = tmp_path / "cl.exe"
    compiler.touch()
    sources = []
    build = SimpleNamespace(get_cc=lambda: str(compiler),
                            compile_module_from_src=lambda src, name: sources.append((src, name)))
    monkeypatch.setitem(sys.modules, "triton.runtime.build", build)
    monkeypatch.setattr(tools, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(tools, "configure_windows_triton_compiler", lambda: None)
    tools.check_windows_triton_compiler()
    assert "Attr first = {};" in sources[0][0]
    assert str(compiler) in capsys.readouterr().out
    def fail(*args):
        raise RuntimeError("C2059: syntax error: '}'")
    build.compile_module_from_src = fail
    with pytest.raises(RuntimeError, match="unset CC"):
        tools.check_windows_triton_compiler()


def test_active_environment_is_preserved(monkeypatch):
    monkeypatch.setattr(tools, "configure_windows_triton_compiler", lambda: None)
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
    monkeypatch.setattr(tools, "configure_windows_triton_compiler", lambda: None)
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
    monkeypatch.setattr(tools, "configure_windows_triton_compiler", lambda: None)
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


def test_windows_moss_allows_external_flash_packing_and_compile(monkeypatch):
    from trainer.training.config import TrainConfig, Config, validate_model_options
    from trainer.data.dataset import DatasetConfig
    monkeypatch.setattr(tools, "sys", SimpleNamespace(platform="win32"))
    train = TrainConfig(model_family="moss_image", attention_backend="flash_attn_2",
                        pack_resolutions=True, compile="default", batch_size=6)
    cfg = Config(train=train, dataset=DatasetConfig(path="images"))
    validate_model_options(cfg)
    assert cfg.train.pack_resolutions
    with pytest.raises(ValueError, match="Compile requires"):
        TrainConfig(model_family="flux3", attention_backend="flash_attn_2", compile="default")


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


@pytest.mark.skipif(os.name != "nt", reason="real Windows integration")
def test_real_windows_triton_c_launcher():
    tools.prepare_windows_toolchain(required=True)
    tools.check_windows_triton_compiler()
