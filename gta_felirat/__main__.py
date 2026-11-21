"""Indítás: python -m gta_felirat [--device N] [--lang de] [--translator claude] [--image kép.png]

A csomagolt exe (GTA6Felirat.exe) ugyanezt a belépési pontot használja (run_exe): forrás
megadása nélkül Remote Play módban indul, a config.json pedig az exe mellett van.
"""

import argparse
import logging
import shutil
import sys
import traceback
from pathlib import Path

import cv2

from .app import OverlayApp, StaticFrameSource
from .capture import CaptureError, FrameGrabber, list_devices, open_device
from .config import SUPPORTED_TRANSLATORS, ConfigError, load_config, with_overrides
from .i18n import AUTO, set_language, t
from .ocr import SubtitleReader

CONFIG_FILE_NAME = "config.json"
BUNDLED_CONFIG_NAME = "default_config.json"
WEBCAM_WIDTH = 1280
WEBCAM_HEIGHT = 720
WEBCAM_FPS = 30
# Ha ezek egyike sincs megadva, az exe Remote Play módban indul
EXPLICIT_SOURCE_FLAGS = ("--remote-play", "--window", "--device", "--image", "--list-devices", "-h", "--help")


def is_frozen() -> bool:
    """PyInstallerrel csomagolt exe-ként fut-e."""
    return bool(getattr(sys, "frozen", False))


def default_config_path() -> Path:
    # Az exe bárhonnan indítható (parancsikon, más munkakönyvtár): a beállítás mindig mellette legyen
    if is_frozen():
        return Path(sys.executable).resolve().parent / CONFIG_FILE_NAME
    return Path(CONFIG_FILE_NAME)


def bundled_default_config() -> Path:
    """Az exe-be csomagolt alap config.json (az egyfájlos exe mellett eleinte nincs config.json)."""
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / BUNDLED_CONFIG_NAME


def ensure_config_file(target: Path, template: Path) -> None:
    """Ha még nincs config.json, a csomagolt alapértékekkel létrehozza; meglévőt sosem ír felül."""
    if target.exists() or not template.exists():
        return
    try:
        shutil.copyfile(template, target)
    except OSError as err:
        # Írásvédett mappában is induljon el, csak a beállítások nem maradnak meg
        print(t("cli.config_create_failed", path=target, err=err), file=sys.stderr)


def with_exe_defaults(argv: list[str]) -> list[str]:
    """Streamereknek nincs capture card: forrás megadása nélkül Remote Play módban indulunk."""
    has_source = any(arg.split("=", 1)[0] in EXPLICIT_SOURCE_FLAGS for arg in argv)
    return list(argv) if has_source else ["--remote-play", *argv]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="gta_felirat", description=t("cli.description"))
    parser.add_argument("--config", type=Path, default=default_config_path(), help=t("cli.help_config"))
    parser.add_argument("--device", type=int, help=t("cli.help_device"))
    parser.add_argument("--remote-play", action="store_true", help=t("cli.help_remote_play"))
    parser.add_argument("--window", help=t("cli.help_window"))
    parser.add_argument("--webcam", type=int, help=t("cli.help_webcam"))
    parser.add_argument("--translator", choices=list(SUPPORTED_TRANSLATORS), help=t("cli.help_translator"))
    parser.add_argument("--lang", help=t("cli.help_lang"))
    parser.add_argument("--no-panel", action="store_true", help=t("cli.help_no_panel"))
    parser.add_argument("--windowed", action="store_true", help=t("cli.help_windowed"))
    parser.add_argument("--list-devices", action="store_true", help=t("cli.help_list_devices"))
    parser.add_argument("--image", type=Path, help=t("cli.help_image"))
    return parser.parse_args(argv)


def _print_devices() -> int:
    devices = list_devices()
    if not devices:
        print(t("cli.no_devices"))
        return 1
    for index, width, height in devices:
        print(f"  {index}: {width}x{height}")
    return 0


def _open_source(args: argparse.Namespace, config):
    if args.image is not None:
        image = cv2.imread(str(args.image))
        if image is None:
            raise CaptureError(t("cli.image_unreadable", path=args.image))
        return StaticFrameSource(image)
    capture = open_device(config.device_index, config.capture_width,
                          config.capture_height, config.capture_fps)
    return FrameGrabber(capture).start()


def _open_webcam(config) -> FrameGrabber | None:
    if config.webcam_device_index is None:
        return None
    capture = open_device(config.webcam_device_index, WEBCAM_WIDTH, WEBCAM_HEIGHT, WEBCAM_FPS)
    return FrameGrabber(capture).start()


def _run_remote_play(args: argparse.Namespace, config, reader) -> int:
    from .remote_play import DEFAULT_WINDOW_TITLE, RemotePlayApp
    from .window_capture import WindowNotFoundError

    title = args.window or DEFAULT_WINDOW_TITLE
    try:
        app = RemotePlayApp(config, reader, window_title=title, config_path=args.config)
    except WindowNotFoundError as err:
        print(t("cli.error", err=err), file=sys.stderr)
        return 2
    print(t("cli.remote_play_mode", title=title))
    print(t("cli.remote_play_hint"))
    app.run()
    return 0


def main(argv: list[str] | None = None) -> int:
    # Ékezetes (ő, ű) naplósorok ne okozzanak hibát régi kódlapú konzolon
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    set_language(AUTO)  # a config betöltéséig (és a --help-hez) a Windows nyelve
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.list_devices:
        return _print_devices()
    is_remote = args.remote_play or args.window is not None
    if is_remote:
        from .window_capture import enable_dpi_awareness

        enable_dpi_awareness()  # minden ablak előtt, hogy a réteg pontosan fedje a Remote Playt
    try:
        config = with_overrides(load_config(args.config), device_index=args.device,
                                translator=args.translator, target_lang=args.lang,
                                webcam_device_index=args.webcam,
                                fullscreen=False if args.windowed else None)
        set_language(config.ui_language)
        reader = SubtitleReader(config.subtitle_region, config.min_text_score)
        if is_remote:
            return _run_remote_play(args, config, reader)
        source = _open_source(args, config)
        webcam = _open_webcam(config)
    except (ConfigError, CaptureError) as err:
        print(t("cli.error", err=err), file=sys.stderr)
        return 2
    print(t("cli.keys"))
    print(t("cli.panel_hint"))
    OverlayApp(config, source, reader, webcam=webcam, config_path=args.config,
               settings_window=not args.no_panel).run()
    return 0


def _wait_before_closing() -> None:
    # Dupla kattintással indítva a konzolablak hiba után azonnal bezárulna, el sem olvasható
    if sys.stdin is None or not sys.stdin.isatty():
        return
    try:
        input("\n" + t("cli.press_enter"))
    except (EOFError, KeyboardInterrupt):
        pass  # nincs olvasható bemenet, vagy Ctrl+C-vel zárták: nincs mire várni


def run_exe() -> int:
    """A csomagolt exe belépési pontja: Remote Play az alapmód, és hiba után nyitva marad az ablak."""
    try:
        ensure_config_file(default_config_path(), bundled_default_config())
        code = main(with_exe_defaults(sys.argv[1:]))
    except Exception:  # noqa: BLE001 - váratlan hibánál is látszódjon az ok, ne csak villanjon az ablak
        traceback.print_exc()
        code = 1
    if code:
        _wait_before_closing()
    return code


if __name__ == "__main__":
    sys.exit(main())
