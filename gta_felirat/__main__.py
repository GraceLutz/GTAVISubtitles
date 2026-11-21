"""Indítás: python -m gta_felirat [--device N] [--lang de] [--translator claude] [--image kép.png]

A csomagolt exe (GTA6Felirat.exe) ugyanezt a belépési pontot használja (run_exe): forrás
megadása nélkül maga keresi meg a forrást (--auto), a config.json pedig az exe mellett van.
"""

import argparse
import logging
import shutil
import sys
import traceback
from pathlib import Path

import cv2

from .app import OverlayApp, StaticFrameSource
from .capture import API_NAMES, CaptureError, FrameGrabber, list_devices, listing_apis, open_device
from .config import SUPPORTED_CAPTURE_APIS, SUPPORTED_TRANSLATORS, ConfigError, load_config, with_overrides
from .i18n import AUTO, set_language, t
from .ocr import SubtitleReader
from .source_choice import (REMOTE_PLAY_TITLE, VEYRA_TITLE_START, DeviceSource, NoSource, WindowSource,
                            choose_source, devices_for_choice)

CONFIG_FILE_NAME = "config.json"
BUNDLED_CONFIG_NAME = "default_config.json"
WEBCAM_WIDTH = 1280
WEBCAM_HEIGHT = 720
WEBCAM_FPS = 30
# Ha ezek egyike sincs megadva, az exe maga keresi meg a forrást (--auto)
EXPLICIT_SOURCE_FLAGS = ("--auto", "--remote-play", "--window", "--device", "--image", "--list-devices",
                         "-h", "--help")


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
    """Forrás megadása nélkül az exe maga keresi meg: Remote Play, Veyra vagy capture card."""
    has_source = any(arg.split("=", 1)[0] in EXPLICIT_SOURCE_FLAGS for arg in argv)
    return list(argv) if has_source else ["--auto", *argv]


def parse_args(argv: list[str]) -> argparse.Namespace:
    # allow_abbrev=False: a rövidített kapcsoló (pl. --win) ne kerülje meg a with_exe_defaults-ot
    parser = argparse.ArgumentParser(prog="gta_felirat", description=t("cli.description"), allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=default_config_path(), help=t("cli.help_config"))
    parser.add_argument("--auto", action="store_true", help=t("cli.help_auto"))
    parser.add_argument("--device", type=int, help=t("cli.help_device"))
    parser.add_argument("--capture-api", choices=list(SUPPORTED_CAPTURE_APIS), help=t("cli.help_capture_api"))
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


def _print_devices(api: str) -> int:
    devices = list_devices(api)
    if not devices:
        print(t("cli.no_devices"))
        return 1
    _print_device_lines(devices)
    return 0


def _print_device_lines(devices) -> None:
    for device in devices:
        size = f"{device.size[0]}x{device.size[1]}" if device.size else t("cli.device_no_picture")
        print(f"  --device {device.index} --capture-api {device.api}: {size} ({API_NAMES[device.api]})")


def _wants_auto_source(args: argparse.Namespace) -> bool:
    """--auto csak akkor keres, ha a felhasználó semmilyen forrást nem adott meg."""
    has_source = (args.image is not None or args.device is not None or args.window is not None
                  or args.remote_play)
    return args.auto and not has_source


def _has_window(title: str, *, by_start: bool = False) -> bool:
    if sys.platform != "win32":
        return False  # ablakot csak Windowson tudunk olvasni
    from .window_capture import has_window

    return has_window(title, by_start=by_start)


def _resolve_auto_source(args: argparse.Namespace, capture_api: str, webcam_index: int | None) -> int | None:
    """--auto: beállítja args-ban a megtalált forrást; ha nincs mit választani, a kilépési kód."""
    has_veyra = _has_window(VEYRA_TITLE_START, by_start=True)
    has_remote_play = not has_veyra and _has_window(REMOTE_PLAY_TITLE)
    # Eszközt csak akkor keresünk, ha nincs ablak (a megnyitási próba lassú, és foglalhatja az eszközt)
    devices = [] if has_remote_play or has_veyra else devices_for_choice(
        (list_devices(name) for name in listing_apis(capture_api)), webcam_index)
    choice = choose_source(has_remote_play, has_veyra, devices)
    if isinstance(choice, WindowSource):
        args.window = None if choice.title == REMOTE_PLAY_TITLE else choice.title
        args.remote_play = True
        return None
    if isinstance(choice, DeviceSource):
        args.device = choice.device.index
        if choice.device.api in SUPPORTED_CAPTURE_APIS:
            args.capture_api = choice.device.api
        print(t("cli.auto_device", index=choice.device.index, api=API_NAMES[choice.device.api]))
        return None
    assert isinstance(choice, NoSource)
    if choice.devices:
        print(t("cli.choose_device"))
        _print_device_lines(choice.devices)
    else:
        print(t("cli.no_source_found"), file=sys.stderr)
    return 2


def _open_source(args: argparse.Namespace, config):
    if args.image is not None:
        image = cv2.imread(str(args.image))
        if image is None:
            raise CaptureError(t("cli.image_unreadable", path=args.image))
        return StaticFrameSource(image)
    capture = open_device(config.device_index, config.capture_width,
                          config.capture_height, config.capture_fps, api=config.capture_api)
    return FrameGrabber(capture).start()


def _open_webcam(config) -> FrameGrabber | None:
    if config.webcam_device_index is None:
        return None
    # A webkamera a saját útján nyílik (DirectShow, ha az nem megy, Media Foundation): a capture_api a kártyáé
    capture = open_device(config.webcam_device_index, WEBCAM_WIDTH, WEBCAM_HEIGHT, WEBCAM_FPS)
    return FrameGrabber(capture).start()


def _run_remote_play(args: argparse.Namespace, config, reader) -> int:
    from .remote_play import DEFAULT_WINDOW_TITLE, RemotePlayApp
    from .window_capture import WindowNotFoundError

    title = args.window or DEFAULT_WINDOW_TITLE
    try:
        # A felhasználó által megadott címet az elejéről is felismerjük (pl. --window Veyra)
        app = RemotePlayApp(config, reader, window_title=title, match_title_start=args.window is not None,
                            config_path=args.config)
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
        return _print_devices(args.capture_api or "auto")
    try:
        loaded = load_config(args.config)
    except ConfigError as err:
        print(t("cli.error", err=err), file=sys.stderr)
        return 2
    set_language(loaded.ui_language)
    if _wants_auto_source(args):
        webcam_index = loaded.webcam_device_index if args.webcam is None else args.webcam
        code = _resolve_auto_source(args, args.capture_api or loaded.capture_api, webcam_index)
        if code is not None:
            return code
    is_remote = args.remote_play or args.window is not None
    if is_remote:
        from .window_capture import enable_dpi_awareness

        enable_dpi_awareness()  # minden ablak előtt, hogy a réteg pontosan fedje a Remote Playt
    try:
        config = with_overrides(loaded, device_index=args.device,
                                capture_api=args.capture_api,
                                translator=args.translator, target_lang=args.lang,
                                webcam_device_index=args.webcam,
                                fullscreen=False if args.windowed else None)
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
    """A csomagolt exe belépési pontja: magától keresi a forrást, és hiba után nyitva marad az ablak."""
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
