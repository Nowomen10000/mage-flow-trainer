"""Dependency refresh uses the active venv and only records successful installs."""
import json
from pathlib import Path
import subprocess
import tomllib
from types import SimpleNamespace
from urllib.parse import urlsplit, urlunsplit

import pytest

from trainer.tools import update_dependencies as updater


def test_repository_export_matches_lock_on_both_platforms(monkeypatch):
    """The installer and post-install audit must agree, including Git provenance.

    Simulate exactly what uv.lock installs instead of testing against the
    developer's environment. Audit Windows-only pins too on Linux and vice versa.
    """
    from packaging.markers import Marker

    root = Path(__file__).resolve().parents[1]
    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    packages = {p["name"]: p for p in lock["package"]}

    def distribution(name):
        if name not in packages:
            raise updater.metadata.PackageNotFoundError(name)
        package = packages[name]
        direct = {}
        if "git" in package["source"]:
            source = urlsplit(package["source"]["git"])
            direct = {"url": urlunsplit(source._replace(query="", fragment="")),
                      "vcs_info": {"vcs": "git", "commit_id": source.fragment}}
        return SimpleNamespace(version=package["version"],
                               read_text=lambda _: json.dumps(direct))

    monkeypatch.setattr(updater.metadata, "distribution", distribution)
    monkeypatch.setattr(Marker, "evaluate", lambda *a, **kw: True)
    assert updater.dependency_issues(root) == []


def test_diffusers_source_and_installer_fallbacks_match_lock():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    package = next(p for p in lock["package"] if p["name"] == "diffusers")
    revision = urlsplit(package["source"]["git"]).fragment
    assert project["tool"]["uv"]["sources"]["diffusers"]["rev"] == revision
    for filename in ("install.sh", "install.bat"):
        script = (root / filename).read_text(encoding="utf-8")
        assert f'DIFFUSERS_REF="{revision}"' in script or f'DIFFUSERS_REF={revision}"' in script
        assert "uv sync --locked" in script


@pytest.fixture
def environment(tmp_path, monkeypatch):
    root, venv = tmp_path / "fork", tmp_path / "custom env"
    root.mkdir()
    venv.mkdir()
    (root / "requirements.txt").write_text("lycoris-lora==4.1.3\n")
    (root / "pyproject.toml").write_text('[project]\nname="fork"\n')
    monkeypatch.setattr(updater.sys, "prefix", str(venv))
    monkeypatch.setattr(updater.sys, "base_prefix", str(tmp_path / "python"))
    monkeypatch.setattr(updater.sys, "executable", str(venv / "python.exe"))
    monkeypatch.setattr(updater.shutil, "which", lambda name: None)
    monkeypatch.setattr(updater.importlib.util, "find_spec", lambda name: True)
    monkeypatch.setattr(updater.metadata, "distribution",
                        lambda name: SimpleNamespace(version="4.1.3", read_text=lambda _: None))
    return root, venv


def test_content_fingerprint_and_success_marker(environment):
    root, venv = environment
    assert not updater.dependencies_current(root)
    updater.record_current(root)
    assert updater.dependencies_current(root)
    (root / "requirements.txt").touch()
    assert updater.dependencies_current(root)
    (root / "requirements.txt").write_text("lycoris-lora==4.1.4\n")
    assert not updater.dependencies_current(root)
    (root / "requirements.txt").write_text("lycoris-lora==4.1.3\n")
    updater.record_current(root)
    (root / "uv.lock").write_text("changed lock")
    assert not updater.dependencies_current(root)
    (venv / ".mageflow-trainer-dependencies.json").write_text("broken")
    assert not updater.dependencies_current(root)


def test_pip_targets_exact_interpreter_and_pins(environment):
    root, venv = environment
    command, _ = updater.install_command(root)
    assert command[:4] == [str(venv / "python.exe"), "-m", "pip", "install"]
    assert command[command.index("-r") + 1] == str(root / "requirements.txt")
    assert "--upgrade" in command
    assert updater.TORCH_INDEX in command


def test_uv_targets_active_venv_and_preserves_optional_packages(environment, monkeypatch):
    root, venv = environment
    (root / "uv.lock").write_text("locked")
    monkeypatch.setattr(updater.shutil, "which", lambda name: "/bin/uv")
    command, env = updater.install_command(root)
    assert command[:2] == ["/bin/uv", "sync"]
    assert env["UV_PROJECT_ENVIRONMENT"] == str(venv)
    assert "--locked" in command and "--inexact" in command
    assert command[command.index("--python") + 1] == str(venv / "python.exe")


def test_failure_retries_but_success_skips_repeat_install(environment, monkeypatch):
    root, _ = environment
    calls = []

    def fail(*args):
        calls.append(args)
        raise RuntimeError("offline")

    monkeypatch.setattr(updater, "_run_installer", fail)
    with pytest.raises(RuntimeError, match="offline"):
        updater.refresh(root)
    assert not updater.dependencies_current(root)
    monkeypatch.setattr(updater, "_run_installer", lambda *args: calls.append(args))
    assert updater.refresh(root)
    assert not updater.refresh(root)
    assert len(calls) == 2


def test_mid_install_source_change_does_not_record_success(environment, monkeypatch):
    root, _ = environment
    def install(*args):
        (root / "requirements.txt").write_text("changed during install")
    monkeypatch.setattr(updater, "_run_installer", install)
    with pytest.raises(RuntimeError, match="changed during"):
        updater.refresh(root)
    assert not updater.dependencies_current(root)


def test_global_python_not_modified(environment, monkeypatch):
    root, _ = environment
    monkeypatch.setattr(updater, "in_venv", lambda: False)
    with pytest.raises(RuntimeError, match="global Python"):
        updater.refresh(root)


def test_missing_pip_bootstrapped_in_active_venv(environment, monkeypatch):
    root, venv = environment
    commands = []
    monkeypatch.setattr(updater.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(updater, "_run_installer", lambda cmd, *args: commands.append(cmd))
    assert updater.refresh(root)
    assert commands[0] == [str(venv / "python.exe"), "-m", "ensurepip", "--upgrade"]
    assert commands[1][1:4] == ["-m", "pip", "install"]


def test_failed_worker_never_launches_gui(environment, monkeypatch):
    root, _ = environment
    monkeypatch.setattr(updater, "refresh", lambda root: (_ for _ in ()).throw(RuntimeError("broken")))
    monkeypatch.setattr(updater.subprocess, "run", lambda *args, **kw: pytest.fail("GUI started after failed update"))
    assert updater.main(["--root", str(root), "--restart-gui"]) == 1


def test_worker_restarts_gui_only_after_install(environment, monkeypatch):
    root, _ = environment
    calls = []
    monkeypatch.setattr(updater, "_run_installer", lambda *args: calls.append("installed"))
    def run(argv, *, cwd):
        assert cwd == root
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 7)
    monkeypatch.setattr(updater.subprocess, "run", run)
    assert updater.main(["--root", str(root), "--restart-gui"]) == 7
    assert calls[0] == "installed"
    assert updater.dependencies_current(root)
    assert calls[-1][1:] == ["-m", "trainer.gui", "--no-update-check"]


@pytest.mark.parametrize("package,installed,pinned", [
    ("lycoris-lora", "3.0.0", "4.1.3"),
    ("sdnq", "0.3.0", "0.2.4"),
])
@pytest.mark.parametrize("use_uv", [False, True])
def test_installed_drift_repaired_in_both_directions(environment, monkeypatch, package, installed, pinned, use_uv):
    root, _ = environment
    (root / "requirements.txt").write_text(f"{package}=={pinned}\n")
    if use_uv:
        (root / "uv.lock").write_text("locked")
        monkeypatch.setattr(updater.shutil, "which", lambda _: "/bin/uv")
    versions = {package: pinned}
    monkeypatch.setattr(updater.metadata, "distribution", lambda name:
                        SimpleNamespace(version=versions[name], read_text=lambda _: None))
    updater.record_current(root)
    versions[package] = installed
    assert not updater.dependencies_current(root)  # same files, stale environment
    assert installed in updater.dependency_issues(root)[0]
    calls = []
    def install(command, *args):
        calls.append(command)
        if use_uv:
            assert command[1:3] == ["sync", "--locked"]
        else:
            assert "-r" in command
            assert f"{package}=={pinned}" in (root / "requirements.txt").read_text()
        versions[package] = pinned
    monkeypatch.setattr(updater, "_run_installer", install)
    assert updater.refresh(root)
    assert updater.dependencies_current(root)
    assert not updater.refresh(root)
    assert len(calls) == 1


def test_success_exit_with_wrong_packages_does_not_record(environment, monkeypatch):
    root, _ = environment
    monkeypatch.setattr(updater.metadata, "distribution", lambda name:
                        SimpleNamespace(version="3.0.0", read_text=lambda _: None))
    monkeypatch.setattr(updater, "_run_installer", lambda *a: None)
    with pytest.raises(RuntimeError, match="still differ"):
        updater.refresh(root)
    assert not updater.state_path().exists()


def test_git_commit_provenance_not_just_version(environment, monkeypatch):
    root, _ = environment
    revision = "7abd5709f74f9159246ff9c6cc83bc0d5118b639"
    (root / "requirements.txt").write_text(
        f"lycoris-lora @ git+https://github.com/KohakuBlueleaf/LyCORIS@{revision}\n")
    direct = {"url": "https://github.com/kohakublueleaf/lycoris.git",
              "vcs_info": {"vcs": "git", "commit_id": revision}}
    monkeypatch.setattr(updater.metadata, "distribution", lambda name:
                        SimpleNamespace(version="4.1.3", read_text=lambda _: json.dumps(direct)))
    updater.record_current(root)
    assert updater.dependencies_current(root)
    direct["vcs_info"]["commit_id"] = "0" * 40
    assert not updater.dependencies_current(root)
    direct.clear()  # PyPI build with the same version is not the pinned commit
    assert not updater.dependencies_current(root)


def test_markers_missing_packages_and_cuda_local_version(environment, monkeypatch):
    root, _ = environment
    (root / "requirements.txt").write_text(
        "torch==2.10.0\nmissing==1.0\n"
        'not-for-us==1.0 ; python_version < "2"\n')
    def distribution(name):
        if name == "torch":
            return SimpleNamespace(version="2.10.0+cu128")
        if name == "missing":
            raise updater.metadata.PackageNotFoundError(name)
        pytest.fail(f"Inactive marker was audited: {name}")
    monkeypatch.setattr(updater.metadata, "distribution", distribution)
    assert updater.dependency_issues(root) == ["missing is missing"]
