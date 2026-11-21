"""Lemezre mentett fordítások: az ismétlődő sorok („Go to hell!”) a következő indításkor is azonnal jönnek.

Csak a felirat szövegét és a fordítását tárolja, kulcsot soha. Fordítónként és
célnyelvenként külön (pl. „deepl:hu”), mert ugyanaz a mondat máshogy jön ki.
"""

import json
import logging
import os
import threading
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

from .i18n import t

FILE_VERSION = 1
MAX_ENTRIES_PER_NAMESPACE = 5000

logger = logging.getLogger(__name__)


class TranslationStore:
    def __init__(self, path: Path, max_entries: int = MAX_ENTRIES_PER_NAMESPACE) -> None:
        self._path = path
        self._max_entries = max_entries
        self._lock = threading.Lock()  # a memóriában tartott bejegyzéseké: az olvasó (get) csak erre vár
        self._write_lock = threading.Lock()  # a fájlírásé: két szál írása ne keveredjen össze
        self._entries: dict[str, OrderedDict[str, str]] = self._load()

    def _load(self) -> dict[str, OrderedDict[str, str]]:
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            namespaces = raw["entries"]
            return {
                str(name): OrderedDict((str(k), str(v)) for k, v in entries.items())
                for name, entries in namespaces.items()
            }
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as err:
            logger.warning(t("store.load_failed", err=err))
            return {}

    def get(self, namespace: str, key: str) -> str | None:
        with self._lock:
            return self._entries.get(namespace, {}).get(key)

    def put(self, namespace: str, key: str, value: str) -> None:
        self.put_many(namespace, {key: value})

    def put_many(self, namespace: str, items: Mapping[str, str]) -> None:
        """Több új fordítás egyetlen fájlírással (egy kérésben több sor is jöhet).

        A memóriában azonnal látszanak; a lassabb fájlírás alatt az olvasók (get) nem várnak.
        """
        if not items:
            return
        with self._lock:
            entries = self._entries.setdefault(namespace, OrderedDict())
            for key, value in items.items():
                entries[key] = value
                entries.move_to_end(key)
            while len(entries) > self._max_entries:
                entries.popitem(last=False)
        self._save()

    def _save(self) -> None:
        # Az írások sorban mennek, és mindegyik a pillanatnyi teljes állapotot menti:
        # így a fájlba mindig a legfrissebb kerül, akármilyen sorrendben érnek ide a szálak
        with self._write_lock:
            with self._lock:
                snapshot = {name: dict(entries) for name, entries in self._entries.items()}
            try:
                self._write(snapshot)
            except (OSError, ValueError) as err:  # ValueError: nem menthető szöveg (pl. magányos surrogate)
                logger.warning(t("store.save_failed", err=err))

    def _write(self, snapshot: dict[str, dict[str, str]]) -> None:
        temp = self._path.with_suffix(self._path.suffix + ".tmp")
        temp.write_text(json.dumps({"version": FILE_VERSION, "entries": snapshot}, ensure_ascii=False),
                        encoding="utf-8")
        os.replace(temp, self._path)
