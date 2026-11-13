"""`dataset.allow_missing_captions`: caching and training images that have no `.txt` caption.

Off by default, because a missing caption is usually a mistake. On (or implied by a slider, whose
steps never read dataset captions) the cache tool, the GUI's cache step and latents-only datasets
accept uncaptioned images, which then train with an empty caption.
"""
import sys
from types import SimpleNamespace

import pytest
import toml
import torch

from trainer.data.dataset import DatasetConfig, MageFlowDataset
from trainer.training.config import load_config


def _images(root, *, captioned=("a",), uncaptioned=("b", "c")):
    from PIL import Image
    for stem in (*captioned, *uncaptioned):
        Image.new("RGB", (512, 512), (128, 64, 32)).save(root / f"{stem}.png")
    for stem in captioned:
        (root / f"{stem}.txt").write_text("a tag", encoding="utf-8")


def _latents(root, *, captioned=("a",), uncaptioned=("b",)):
    from safetensors.torch import save_file
    for stem in (*captioned, *uncaptioned):
        save_file({"latents": torch.zeros(1)}, str(root / f"{stem}_0512x0512_trainer.safetensors"),
                  metadata={"original_size": "512x512"})
    for stem in captioned:
        (root / f"{stem}.txt").write_text("a tag", encoding="utf-8")


def _config(tmp_path, dataset=None, slider=False):
    data = {
        "train": {"model_family": "mage_flow"},
        "dataset": {"path": str(tmp_path), **(dataset or {})},
        "adapter": {"kind": "lora"},
    }
    if slider:
        data["slider"] = {"enabled": True, "target_class": "person",
                          "positive_prompt": "happy person", "negative_prompt": "sad person"}
    path = tmp_path / "config.toml"
    path.write_text(toml.dumps(data), encoding="utf-8")
    return path


def _cache_cli(monkeypatch, *argv):
    from trainer.tools import cache_latents
    monkeypatch.setattr(sys, "argv", ["cache_latents", *argv])
    return cache_latents.main()


def test_config_default_is_off(tmp_path):
    assert load_config(_config(tmp_path)).dataset.allow_missing_captions is False
    assert load_config(_config(tmp_path, {"allow_missing_captions": True})).dataset.allow_missing_captions


def test_cache_stops_on_uncaptioned_images_unless_allowed(tmp_path, monkeypatch, capsys):
    _images(tmp_path)
    args = ("cache", str(tmp_path), "--resolution", "512", "--dry-run")

    assert _cache_cli(monkeypatch, *args) == 1
    out = capsys.readouterr().out
    assert "2 images have no .txt caption" in out
    assert "Allow missing captions" in out and "--dry-run: nothing written" not in out

    assert _cache_cli(monkeypatch, *args, "--allow-missing-captions") == 0
    out = capsys.readouterr().out
    assert "caching them anyway" in out and "--dry-run: nothing written" in out


def test_cache_treats_a_latents_only_folder_as_done(tmp_path, monkeypatch, capsys):
    # The GUI caches before every Start; a folder whose images were deleted after caching must
    # not stop that chain. No --dry-run: it has to return before any model is loaded.
    _latents(tmp_path)
    assert _cache_cli(monkeypatch, "cache", str(tmp_path), "--resolution", "512") == 0
    assert "only cached latents" in capsys.readouterr().out
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _cache_cli(monkeypatch, "cache", str(empty), "--resolution", "512") == 1


@pytest.mark.parametrize("dataset,slider,flag,expected", [
    ({}, False, False, False),
    ({}, False, True, True),
    ({"allow_missing_captions": True}, False, False, True),
    ({}, True, False, True),
])
def test_cache_config_passes_the_flag(tmp_path, dataset, slider, flag, expected):
    from trainer.tools.cache_latents import config_cache_commands
    args = SimpleNamespace(config=str(_config(tmp_path, dataset, slider)), resolution=None,
                           batch_size=1, devices="", overwrite=False, dry_run=False,
                           allow_missing_captions=flag)
    (command,) = config_cache_commands(args)
    assert ("--allow-missing-captions" in command) is expected
    assert command.count("--allow-missing-captions") <= 1


def test_images_dataset_trains_uncaptioned_with_empty_caption(tmp_path):
    _images(tmp_path, captioned=("a",), uncaptioned=("b",))
    ds = MageFlowDataset(DatasetConfig(path=str(tmp_path), resolution=512, source="images"),
                         require_cache=False)
    assert {e.path.stem: e.tags for e in ds.entries} == {"a": "a tag", "b": ""}


def test_latents_only_dataset_refuses_uncaptioned_unless_allowed(tmp_path):
    _latents(tmp_path)
    with pytest.raises(RuntimeError, match="allow_missing_captions"):
        MageFlowDataset(DatasetConfig(path=str(tmp_path), resolution=512, source="latents"))

    ds = MageFlowDataset(DatasetConfig(path=str(tmp_path), resolution=512, source="latents",
                                       allow_missing_captions=True))
    tags = {e.latent_path.name.split("_")[0]: e.tags for e in ds.entries}
    assert tags == {"a": "a tag", "b": ""}


def test_latents_only_dataset_with_no_captions_at_all(tmp_path):
    _latents(tmp_path, captioned=(), uncaptioned=("a", "b"))
    with pytest.raises(ValueError, match="allow_missing_captions"):
        MageFlowDataset(DatasetConfig(path=str(tmp_path), resolution=512, source="latents"))
    ds = MageFlowDataset(DatasetConfig(path=str(tmp_path), resolution=512, source="latents",
                                       allow_missing_captions=True))
    assert [e.tags for e in ds.entries] == ["", ""]


@pytest.mark.parametrize("slider,expected", [(False, False), (True, True)])
def test_trainer_dataset_allows_missing_captions_in_slider_mode(tmp_path, monkeypatch, slider, expected):
    from trainer.training import train

    class Built(Exception):
        pass

    seen = []

    def fake_dataset(config, **kwargs):
        seen.append(config)
        raise Built

    monkeypatch.setattr(train, "MageFlowDataset", fake_dataset)
    cfg = load_config(_config(tmp_path, slider=slider))
    with pytest.raises(Built):
        train.Trainer._build_data(SimpleNamespace(cfg=cfg))
    assert seen[0].allow_missing_captions is expected
    assert cfg.dataset.allow_missing_captions is False   # the loaded config itself is untouched


def test_gui_cache_launch_and_schema(tmp_path, monkeypatch):
    from trainer.gui.process import cache_launch
    kwargs = dict(min_bucket_reso=256, max_bucket_reso=4096, bucket_reso_steps=64,
                  upscale=False, multires_training=False)
    assert "--allow-missing-captions" not in cache_launch("/d", "/m", [512], **kwargs).argv
    assert "--allow-missing-captions" in cache_launch("/d", "/m", [512], allow_missing_captions=True,
                                                      **kwargs).argv

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from trainer.gui import bridge
    from trainer.gui.app import _captions_optional
    from trainer.gui.schema import SPEC, layout_keys

    key = "dataset.allow_missing_captions"
    assert key in SPEC and key in layout_keys() and bridge.defaults()[key] is False
    assert not _captions_optional({key: False, "slider.enabled": False})
    assert _captions_optional({key: True, "slider.enabled": False})
    assert _captions_optional({key: False, "slider.enabled": True})

    flat = bridge.defaults() | {"dataset.path": str(tmp_path), key: True}
    path = tmp_path / "gui.toml"
    bridge.write_toml(path, flat)
    assert bridge.flatten(bridge.read_toml(path))[key] is True
    assert load_config(path).dataset.allow_missing_captions is True
