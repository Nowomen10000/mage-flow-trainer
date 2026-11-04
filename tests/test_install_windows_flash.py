import pytest

from trainer.tools import install_windows_flash as installer


@pytest.mark.parametrize("minor", [11, 12, 13])
def test_supported_wheel(minor):
    url = installer.wheel_url("win32", "AMD64", "cpython", (3, minor), "2.10.0+cu128", "12.8")
    assert f"-cp3{minor}-cp3{minor}-win_amd64.whl#sha256=" in url
    assert "/v0.7.13/" in url
    assert len(url.split("#sha256=")[1]) == 64


@pytest.mark.parametrize("index,value", [(0, "linux"), (1, "ARM64"), (2, "pypy"),
                                        (3, (3, 14)), (4, "2.11.0+cu128"), (5, "13.0")])
def test_incompatible_environment_rejected(index, value):
    args = ["win32", "AMD64", "cpython", (3, 12), "2.10.0+cu128", "12.8"]
    args[index] = value
    with pytest.raises(ValueError):
        installer.wheel_url(*args)


@pytest.mark.parametrize("uv", [None, "C:/tools/uv.exe"])
def test_installs_only_wheel_into_active_interpreter(monkeypatch, uv):
    monkeypatch.setattr(installer.shutil, "which", lambda _: uv)
    command = installer.install_command("https://example.test/wheel.whl#sha256=abc")
    assert installer.sys.executable in command
    assert "--no-deps" in command
    assert command[0] == (uv or installer.sys.executable)
