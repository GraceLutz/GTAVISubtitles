"""API-kulcsok tartós tárolása a Windows Hitelesítőadat-kezelőjében.

A kulcs soha nem kerül a config.json-ba vagy más fájlba a program mappájában: a
Windows titkosítva, a bejelentkezett felhasználóhoz kötve tárolja. Kézzel is
törölhető: Vezérlőpult → Hitelesítőadat-kezelő → Windows-hitelesítő adatok →
„gta_felirat/…”.
"""

import ctypes
import ctypes.wintypes as wintypes
from typing import Protocol

DEFAULT_PREFIX = "gta_felirat/"
CRED_TYPE_GENERIC = 1
CRED_PERSIST_LOCAL_MACHINE = 2
ERROR_NOT_FOUND = 1168
KEY_ENCODING = "utf-16-le"


class KeyVaultError(RuntimeError):
    """A kulcsot nem sikerült olvasni, menteni vagy törölni (a kulcs sosem szerepel az üzenetben)."""


class KeyVault(Protocol):
    def load(self, name: str) -> str | None: ...

    def save(self, name: str, key: str) -> None: ...

    def delete(self, name: str) -> None: ...


class _Credential(ctypes.Structure):
    _fields_ = [
        ("Flags", wintypes.DWORD),
        ("Type", wintypes.DWORD),
        ("TargetName", wintypes.LPWSTR),
        ("Comment", wintypes.LPWSTR),
        ("LastWritten", wintypes.FILETIME),
        ("CredentialBlobSize", wintypes.DWORD),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
        ("Persist", wintypes.DWORD),
        ("AttributeCount", wintypes.DWORD),
        ("Attributes", ctypes.c_void_p),
        ("TargetAlias", wintypes.LPWSTR),
        ("UserName", wintypes.LPWSTR),
    ]


def _advapi32() -> ctypes.WinDLL:
    dll = ctypes.WinDLL("advapi32", use_last_error=True)
    dll.CredWriteW.argtypes = [ctypes.POINTER(_Credential), wintypes.DWORD]
    dll.CredWriteW.restype = wintypes.BOOL
    dll.CredReadW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                              ctypes.POINTER(ctypes.POINTER(_Credential))]
    dll.CredReadW.restype = wintypes.BOOL
    dll.CredDeleteW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
    dll.CredDeleteW.restype = wintypes.BOOL
    dll.CredFree.argtypes = [ctypes.c_void_p]
    dll.CredFree.restype = None
    return dll


class WindowsCredentialVault:
    def __init__(self, prefix: str = DEFAULT_PREFIX) -> None:
        self._prefix = prefix
        self._dll = _advapi32()

    def _target(self, name: str) -> str:
        return f"{self._prefix}{name}"

    def load(self, name: str) -> str | None:
        pointer = ctypes.POINTER(_Credential)()
        if not self._dll.CredReadW(self._target(name), CRED_TYPE_GENERIC, 0, ctypes.byref(pointer)):
            code = ctypes.get_last_error()
            if code == ERROR_NOT_FOUND:
                return None
            raise KeyVaultError(f"A(z) {name} kulcsot nem sikerült beolvasni (Windows-hiba {code})")
        try:
            credential = pointer.contents
            blob = ctypes.string_at(credential.CredentialBlob, credential.CredentialBlobSize)
            return blob.decode(KEY_ENCODING)
        finally:
            self._dll.CredFree(pointer)

    def save(self, name: str, key: str) -> None:
        blob = key.encode(KEY_ENCODING)
        buffer = (ctypes.c_ubyte * len(blob)).from_buffer_copy(blob)
        credential = _Credential(
            Type=CRED_TYPE_GENERIC,
            TargetName=self._target(name),
            Comment="GTA feliratfordító API-kulcs",
            CredentialBlobSize=len(blob),
            CredentialBlob=ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)),
            Persist=CRED_PERSIST_LOCAL_MACHINE,
            UserName=name,
        )
        if not self._dll.CredWriteW(ctypes.byref(credential), 0):
            code = ctypes.get_last_error()
            raise KeyVaultError(f"A(z) {name} kulcsot nem sikerült elmenteni (Windows-hiba {code})")

    def delete(self, name: str) -> None:
        if not self._dll.CredDeleteW(self._target(name), CRED_TYPE_GENERIC, 0):
            code = ctypes.get_last_error()
            if code != ERROR_NOT_FOUND:
                raise KeyVaultError(f"A(z) {name} kulcsot nem sikerült törölni (Windows-hiba {code})")
