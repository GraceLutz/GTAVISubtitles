import sys
from pathlib import Path

import pytest

from gta_felirat.__main__ import default_config_path, ensure_config_file, with_exe_defaults


def test_exe_without_source_finds_the_source_itself():
    assert with_exe_defaults([]) == ["--auto"]


def test_exe_keeps_other_switches_when_adding_auto():
    assert with_exe_defaults(["--lang", "de", "--capture-api", "msmf"]) == [
        "--auto", "--lang", "de", "--capture-api", "msmf"]


@pytest.mark.parametrize("argv", [
    ["--auto"],
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


@pytest.mark.parametrize("argv, wants_auto", [
    (["--auto"], True),
    (["--auto", "--lang", "de"], True),
    (["--auto", "--window", "Xbox"], False),
    (["--auto", "--remote-play"], False),
    (["--auto", "--device", "1"], False),
    (["--auto", "--image", "kep.png"], False),
    ([], False),
])
def test_auto_source_is_searched_only_without_a_given_source(argv, wants_auto):
    from gta_felirat.__main__ import _wants_auto_source, parse_args

    assert _wants_auto_source(parse_args(argv)) is wants_auto


def test_abbreviated_switch_is_not_accepted():
    # Rövidítve (--win) a with_exe_defaults nem ismerné fel a forrást, és mellé tenné az --auto-t
    from gta_felirat.__main__ import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--win", "Xbox"])
