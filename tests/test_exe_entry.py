import sys
from pathlib import Path

import pytest

from gta_felirat.__main__ import default_config_path, with_exe_defaults


def test_exe_without_source_starts_in_remote_play_mode():
    assert with_exe_defaults([]) == ["--remote-play"]


def test_exe_keeps_other_switches_when_adding_remote_play():
    assert with_exe_defaults(["--lang", "de", "--translator", "claude"]) == [
        "--remote-play", "--lang", "de", "--translator", "claude"]


@pytest.mark.parametrize("argv", [
    ["--remote-play"],
    ["--window", "Xbox"],
    ["--window=Xbox"],
    ["--device", "1"],
    ["--image", "kep.png"],
    ["--list-devices"],
    ["--help"],
])
def test_exe_respects_an_explicit_source(argv):
    assert with_exe_defaults(argv) == argv


def test_config_is_in_working_directory_when_run_from_source(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert default_config_path() == Path("config.json")


def test_config_is_next_to_the_exe_when_frozen(monkeypatch, tmp_path):
    exe = tmp_path / "GTA6Felirat" / "GTA6Felirat.exe"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    assert default_config_path() == exe.parent.resolve() / "config.json"
