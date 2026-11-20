import sys
from pathlib import Path

import pytest

from gta_felirat.__main__ import default_config_path, ensure_config_file, with_exe_defaults


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


def test_missing_config_is_created_from_the_bundled_default(tmp_path):
    template = tmp_path / "default_config.json"
    template.write_text('{"target_lang": "hu"}', encoding="utf-8")
    target = tmp_path / "exe" / "config.json"
    target.parent.mkdir()
    ensure_config_file(target, template)
    assert target.read_text(encoding="utf-8") == '{"target_lang": "hu"}'


def test_existing_config_is_never_overwritten(tmp_path):
    template = tmp_path / "default_config.json"
    template.write_text('{"target_lang": "hu"}', encoding="utf-8")
    target = tmp_path / "config.json"
    target.write_text('{"target_lang": "de"}', encoding="utf-8")
    ensure_config_file(target, template)
    assert target.read_text(encoding="utf-8") == '{"target_lang": "de"}'


def test_missing_template_leaves_config_absent(tmp_path):
    target = tmp_path / "config.json"
    ensure_config_file(target, tmp_path / "nincs.json")
    assert not target.exists()
