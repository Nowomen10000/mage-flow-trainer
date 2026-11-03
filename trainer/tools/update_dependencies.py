"""Refresh the launcher's venv without importing Qt, torch, or training code."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit


DEPENDENCY_FILES = ("requirements.txt", "pyproject.toml", "uv.lock")
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"


def in_venv():
    return sys.prefix != sys.base_prefix or hasattr(sys, "real_prefix")


def fingerprint(root):
    digest = hashlib.sha256()
    digest.update(sys.implementation.cache_tag.encode())
    for name in DEPENDENCY_FILES:
        path = Path(root) / name
        digest.update(name.encode() + b"\0")
        digest.update(path.read_bytes() if path.is_file() else b"<absent>")
    return digest.hexdigest()


def state_path():
    # Keep the legacy filename across the BluFlow rename.
    return Path(sys.prefix) / ".mageflow-trainer-dependencies.json"


def dependency_issues(root):
    """Audit installed metadata against exported pins without importing packages.

    Check platform markers and PEP 610 Git commit provenance, not merely a prior
    successful-install marker: users can replace a package after a successful sync.
    Optional packages absent from requirements.txt are deliberately left alone.
    """
    try:
        from packaging.requirements import Requirement, InvalidRequirement
    except ImportError:
        return ["packaging is missing; installed dependency pins cannot be checked"]
    path = Path(root) / "requirements.txt"
    if not path.is_file():
        return ["requirements.txt is missing; run the repository installer"]
    issues = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        try:
            req = Requirement(line)
        except InvalidRequirement:
            issues.append(f"Cannot check requirement: {line}")
            continue
        if req.marker and not req.marker.evaluate():
            continue
        try:
            dist = metadata.distribution(req.name)
        except metadata.PackageNotFoundError:
            issues.append(f"{req.name} is missing")
            continue
        if req.url:
            try:
                direct = json.loads(dist.read_text("direct_url.json") or "{}")
                if req.url.startswith("git+"):
                    source, revision = req.url[4:].rsplit("@", 1)
                    vcs = direct.get("vcs_info", {})
                    matches = (vcs.get("vcs") == "git"
                               and vcs.get("commit_id") == revision.split("#", 1)[0]
                               and _source_url(direct.get("url", "")) == _source_url(source))
                else:
                    matches = direct.get("url") == req.url
            except (ValueError, AttributeError, TypeError):
                matches = False
            if not matches:
                issues.append(f"{req.name} {dist.version}: installed source does not match {req.url}")
        elif req.specifier and not req.specifier.contains(dist.version, prereleases=True):
            issues.append(f"{req.name} {dist.version} installed; required {req.specifier}")
    return issues


def _source_url(url):
    parsed = urlsplit(url)
    # GitHub repository URLs are case-insensitive and may include a .git suffix.
    path = parsed.path.removesuffix("/").removesuffix(".git")
    if parsed.hostname == "github.com":
        path = path.lower()
    return parsed.hostname, path


def dependencies_current(root):
    try:
        state = json.loads(state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (isinstance(state, dict) and state.get("root") == str(Path(root).resolve())
            and state.get("fingerprint") == fingerprint(root)
            and not dependency_issues(root))


def record_current(root, expected=None):
    if not in_venv():
        raise RuntimeError("Start the trainer with its venv Python before updating packages.")
    current = fingerprint(root)
    if expected is not None and current != expected:
        raise RuntimeError("Dependency files changed during installation. Restart to retry.")
    issues = dependency_issues(root)
    if issues:
        raise RuntimeError("Installed packages still differ from the repository pins:\n"
                           + "\n".join(issues[:12]))
    destination = state_path()
    fd, temporary = tempfile.mkstemp(prefix=destination.name, dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"root": str(Path(root).resolve()), "fingerprint": current}, stream)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def install_command(root):
    root = Path(root).resolve()
    env = os.environ.copy()
    uv = shutil.which("uv")
    if uv and (root / "uv.lock").is_file():
        # Preserve optional/user packages, but synchronize the pinned base and GUI.
        # Explicitly target the interpreter that launched us, not a default .venv.
        env["UV_PROJECT_ENVIRONMENT"] = sys.prefix
        command = [uv, "sync", "--locked", "--inexact", "--extra", "gui", "--extra", "ot",
                   "--python", sys.executable, "--no-python-downloads", "--project", str(root)]
    else:
        if not (root / "requirements.txt").is_file():
            raise RuntimeError("No requirements.txt or usable uv.lock. Run the repository installer.")
        command = [sys.executable, "-m", "pip", "install", "--upgrade",
                   "--upgrade-strategy", "only-if-needed", "-r", str(root / "requirements.txt"),
                   "--extra-index-url", TORCH_INDEX]
    return command, env


def _run_installer(command, env, root, log_path):
    print(f"Refreshing packages in {sys.prefix}\nInstallation log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, errors="replace") as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            if process.wait():
                raise RuntimeError(f"Package installation failed (exit {process.returncode}). See {log_path}")


def refresh(root):
    if not in_venv():
        raise RuntimeError("Automatic package updates require a venv; global Python will not be modified.")
    if dependencies_current(root):
        return False
    expected = fingerprint(root)
    command, env = install_command(root)
    log_path = Path(sys.prefix) / ".mageflow-trainer-update.log"
    if command[1:3] == ["-m", "pip"] and importlib.util.find_spec("pip") is None:
        # Some uv-created environments have no pip; use Python's bundled bootstrap
        # if uv is no longer available on the launcher's PATH.
        _run_installer([sys.executable, "-m", "ensurepip", "--upgrade"], env, Path(root).resolve(), log_path)
    _run_installer(command, env, Path(root).resolve(), log_path)
    record_current(root, expected)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--record", action="store_true", help="Record a just-completed installer run")
    parser.add_argument("--restart-gui", action="store_true")
    parser.add_argument("gui_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        if args.record:
            record_current(args.root)
        else:
            refresh(args.root)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Dependency update failed: {exc}\nThe GUI was not launched with a partially updated environment. "
              "Run the installer or restart the GUI to retry.", file=sys.stderr, flush=True)
        return 1
    if args.restart_gui:
        extra = args.gui_args[1:] if args.gui_args[:1] == ["--"] else args.gui_args
        return subprocess.run(
            [sys.executable, "-m", "trainer.gui", "--no-update-check", *extra],
            cwd=args.root,
        ).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
