"""Közös tesztbeállítás."""

import gc

import pytest


@pytest.fixture(autouse=True, scope="module")
def tk_garbage_dies_on_the_main_thread():
    """Minden tesztmodul után a főszálon futtatjuk a szemétgyűjtőt.

    A lebontott Tk-ablak (test_settings_window) körkörös hivatkozásban marad. Ha a szemétgyűjtő később
    egy munkaszálon (például egy fordító szálon) szabadítaná fel, a Tcl az egész folyamatot leállítja:
    „Tcl_AsyncDelete: async handler deleted by the wrong thread”. Ritka, ezért alattomos hiba volt.
    """
    yield
    gc.collect()
