"""Egy ablak képe a Windows.Graphics.Capture-rel (WGC), külön csomag nélkül, ctypes-szal.

A képkocka a videókártyán marad: csak a kért téglalapot másoljuk egy kis átmeneti textúrába, és azt
olvassuk ki (CopySubresourceRegion + Map). Egy feliratsáv így ~1,4 ms, az ablak méretétől függetlenül.
A képben csak az ablak saját tartalma van (a fölé rajzolt feliratrétegünk nem), és akkor is érkezik,
ha másik ablak takarja.

Szálszabály: a WindowStream-et ugyanazon az egy szálon kell létrehozni, használni és lezárni.
Minden hibája WindowStreamError (OSError): a hívó ilyenkor a PrintWindow-ra vált (band_feed.py).
"""

import ctypes
import ctypes.wintypes as wintypes
import logging
import sys
import uuid
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .band_capture import bgrx_to_bgr
from .ocr import Box

HRESULT = ctypes.c_long
POINTER_SIZE = ctypes.sizeof(ctypes.c_void_p)
BYTES_PER_PIXEL = 4
RO_INIT_MULTITHREADED = 1
D3D_DRIVER_TYPE_HARDWARE = 1
D3D11_CREATE_DEVICE_BGRA_SUPPORT = 0x20  # enélkül az eszközből nem lesz WinRT Direct3D-eszköz
D3D11_SDK_VERSION = 7
DXGI_FORMAT_B8G8R8A8_UNORM = 87
D3D11_USAGE_STAGING = 3  # a processzor felől olvasható textúra
D3D11_CPU_ACCESS_READ = 0x20000
D3D11_MAP_READ = 1
D3D11_MAP_WAITS = 0  # jelző nélkül a Map megvárja, míg a videókártya végez a másolással
FIRST_SUBRESOURCE = 0  # a textúráinknak egyetlen képszintje van
FRAME_BUFFERS = 1  # egyetlen puffer: nem gyűlnek fel régi képkockák
TEXTURE_DEPTH = 1  # a kétdimenziós textúra egyetlen „szelet” mély
TIMESPAN_TICKS_PER_SECOND = 10_000_000  # a WinRT TimeSpan 100 ns-os egységekben számol
BORDERLESS_CAPTURE_BUILD = 20348  # ettől a Windows-változattól kapcsolható ki a sárga rögzítési keret
CAPTURE_ITEM_CLASS = "Windows.Graphics.Capture.GraphicsCaptureItem"
FRAME_POOL_CLASS = "Windows.Graphics.Capture.Direct3D11CaptureFramePool"

logger = logging.getLogger(__name__)


class WindowStreamError(OSError):
    """A Windows.Graphics.Capture itt nem használható, vagy egy hívása hibát adott."""


# --- COM-hívások ctypes-szal: az ablakrögzítéstől független réteg ---------------------------------

class Guid(ctypes.Structure):
    _fields_ = [("data", ctypes.c_ubyte * 16)]


def _iid(text: str) -> Guid:
    return Guid.from_buffer_copy(uuid.UUID(text).bytes_le)


@dataclass(frozen=True, eq=False)
class _Method:
    """Egy COM-metódus: a neve (a hibaüzenetbe), a helye a virtuális táblában és a típusai."""

    name: str
    slot: int
    argtypes: tuple[type, ...] = ()
    restype: type | None = HRESULT


_OUT = ctypes.POINTER(ctypes.c_void_p)  # a hívás ide írja a visszaadott interfészmutatót
_UINT, _VOID_P = ctypes.c_uint, ctypes.c_void_p
IID_CLOSABLE = _iid("30d5a829-7fa4-4026-83bb-d75bae4ea99e")
# A helyek az interfészek deklarációs sorrendjéből jönnek: IUnknown 0–2, IInspectable 3–5, utánuk a saját metódusok
QUERY_INTERFACE = _Method("QueryInterface", 0, (ctypes.POINTER(Guid), _OUT))
RELEASE = _Method("Release", 2, (), ctypes.c_ulong)
CLOSE = _Method("IClosable.Close", 6)


def _raise_on_failure(result: int, function: Callable[..., int], _arguments: tuple) -> int:
    """ctypes errcheck: a hibás HRESULT-ból kivétel lesz, így egyik hívás hibája sem marad észrevétlen."""
    if result < 0:
        raise WindowStreamError(f"{function.__name__} failed: HRESULT 0x{result & 0xFFFFFFFF:08X}")
    return result


@lru_cache(maxsize=None)
def _function(address: int, method: _Method) -> Callable[..., int | None]:
    function = ctypes.WINFUNCTYPE(method.restype, ctypes.c_void_p, *method.argtypes)(address)
    function.__name__ = method.name
    if method.restype is HRESULT:
        function.errcheck = _raise_on_failure
    return function


def _pointer_at(address: int) -> int:
    """Mutató kiolvasása a memóriából a ctypes védett hívásával: rossz címnél OSError lesz, nem összeomlás."""
    value = ctypes.c_void_p()
    ctypes.memmove(ctypes.byref(value), address, POINTER_SIZE)
    return value.value or 0


def _call(pointer: int | None, method: _Method, *arguments: object) -> int | None:
    """A COM-objektum egy metódusának hívása; nullmutatónál és hibás HRESULT-nál WindowStreamError."""
    if not pointer:
        raise WindowStreamError(f"{method.name}: there is no object to call")
    table = _pointer_at(pointer)
    return _function(_pointer_at(table + method.slot * POINTER_SIZE), method)(pointer, *arguments)


def _out(pointer: int | None, method: _Method, *arguments: object) -> int:
    """Olyan metódus hívása, amelynek utolsó paramétere a visszaadott interfészmutató (ami nem lehet null)."""
    result = ctypes.c_void_p()
    _call(pointer, method, *arguments, ctypes.byref(result))
    if not result.value:
        raise WindowStreamError(f"{method.name} returned no object")
    return result.value


def _release(pointer: int | None) -> None:
    if pointer:
        _call(pointer, RELEASE)


@contextmanager
def _releasing(pointer: int | None) -> Iterator[int | None]:
    """A with blokk végén elengedi a mutatót, hiba esetén is."""
    try:
        yield pointer
    finally:
        _release(pointer)


def _interface(pointer: int | None, iid: Guid) -> AbstractContextManager[int | None]:
    """A kért interfész a with blokk idejére (QueryInterface, a végén Release)."""
    return _releasing(_out(pointer, QUERY_INTERFACE, ctypes.byref(iid)))


def _close_object(pointer: int | None) -> None:
    """WinRT-objektum lezárása (IClosable.Close) és elengedése; az elengedés a lezárás hibájánál sem marad el."""
    with _releasing(pointer), _interface(pointer, IID_CLOSABLE) as closable:
        _call(closable, CLOSE)


# --- Windows.Graphics.Capture és Direct3D 11 ------------------------------------------------------

class SizeInt32(ctypes.Structure):
    _fields_ = [("width", ctypes.c_int32), ("height", ctypes.c_int32)]


class Texture2dDesc(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint) for name in (
        "Width", "Height", "MipLevels", "ArraySize", "Format", "SampleCount", "SampleQuality", "Usage",
        "BindFlags", "CPUAccessFlags", "MiscFlags")]


class D3d11Box(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint) for name in ("left", "top", "front", "right", "bottom", "back")]


class MappedSubresource(ctypes.Structure):
    _fields_ = [("pData", ctypes.c_void_p), ("RowPitch", ctypes.c_uint), ("DepthPitch", ctypes.c_uint)]


IID_DXGI_DEVICE = _iid("54ec77fa-1377-44e6-8c32-88fd5f44c84c")
IID_DIRECT3D_DEVICE = _iid("a37624ab-8d5f-4650-9d3e-9eae3d9bc670")
IID_CAPTURE_ITEM_INTEROP = _iid("3628e81b-3cac-4c60-b7f4-23ce0e0c3356")
IID_CAPTURE_ITEM = _iid("79c3f95b-31f7-4ec2-a464-632ef5d30760")
IID_FRAME_POOL_STATICS = _iid("589b103f-6bbc-5df5-a991-02e28b3b66d5")
IID_SESSION_CURSOR = _iid("2c39ae40-7d2e-5044-804e-8b6799d4cf9e")  # IGraphicsCaptureSession2
IID_SESSION_BORDER = _iid("f2cdd966-22ae-5ea1-9596-3a289344c3be")  # IGraphicsCaptureSession3 (Windows 11)
IID_SESSION_INTERVAL = _iid("67c0ea62-1f85-5061-925a-239be0ac09cb")  # IGraphicsCaptureSession5 (24H2)
IID_DXGI_ACCESS = _iid("a9b3d012-3df2-4ee3-b8d1-8695f457d3c1")
IID_TEXTURE_2D = _iid("6f15aaf2-d208-4e89-9ab4-489535d34f9c")

CREATE_FOR_WINDOW = _Method("CaptureItem.CreateForWindow", 3, (wintypes.HWND, ctypes.POINTER(Guid), _OUT))
ITEM_SIZE = _Method("CaptureItem.Size", 7, (ctypes.POINTER(SizeInt32),))
CREATE_POOL = _Method("FramePool.CreateFreeThreaded", 6, (_VOID_P, ctypes.c_int, ctypes.c_int32, SizeInt32, _OUT))
RECREATE_POOL = _Method("FramePool.Recreate", 6, (_VOID_P, ctypes.c_int, ctypes.c_int32, SizeInt32))
TRY_GET_NEXT_FRAME = _Method("FramePool.TryGetNextFrame", 7, (_OUT,))
CREATE_SESSION = _Method("FramePool.CreateCaptureSession", 10, (_VOID_P, _OUT))
START_CAPTURE = _Method("CaptureSession.StartCapture", 6)
SET_CURSOR_CAPTURE = _Method("CaptureSession.put_IsCursorCaptureEnabled", 7, (ctypes.c_ubyte,))
GET_BORDER_REQUIRED = _Method("CaptureSession.get_IsBorderRequired", 6, (ctypes.POINTER(ctypes.c_ubyte),))
SET_BORDER_REQUIRED = _Method("CaptureSession.put_IsBorderRequired", 7, (ctypes.c_ubyte,))
SET_MIN_UPDATE_INTERVAL = _Method("CaptureSession.put_MinUpdateInterval", 7, (ctypes.c_int64,))
FRAME_SURFACE = _Method("CaptureFrame.Surface", 6, (_OUT,))
FRAME_CONTENT_SIZE = _Method("CaptureFrame.ContentSize", 8, (ctypes.POINTER(SizeInt32),))
GET_DXGI_INTERFACE = _Method("DxgiInterfaceAccess.GetInterface", 3, (ctypes.POINTER(Guid), _OUT))
CREATE_TEXTURE = _Method("Device.CreateTexture2D", 5, (ctypes.POINTER(Texture2dDesc), _VOID_P, _OUT))
COPY_REGION = _Method("Context.CopySubresourceRegion", 46,
                      (_VOID_P, _UINT, _UINT, _UINT, _UINT, _VOID_P, _UINT, ctypes.POINTER(D3d11Box)), None)
MAP = _Method("Context.Map", 14, (_VOID_P, _UINT, _UINT, _UINT, ctypes.POINTER(MappedSubresource)))
UNMAP = _Method("Context.Unmap", 15, (_VOID_P, _UINT), None)


@lru_cache(maxsize=None)
def _winrt() -> tuple[ctypes.WinDLL, ctypes.WinDLL]:
    """A combase és a d3d11 saját példánya pontos típusokkal (a 64 bites mutatókat a ctypes különben levágná)."""
    try:
        combase, d3d11 = ctypes.WinDLL("combase"), ctypes.WinDLL("d3d11")
        checked = [
            (combase.RoGetActivationFactory, [_VOID_P, ctypes.POINTER(Guid), _OUT]),
            (combase.WindowsCreateString, [ctypes.c_wchar_p, ctypes.c_uint32, _OUT]),
            (combase.WindowsDeleteString, [_VOID_P]),
            (d3d11.D3D11CreateDevice, [_VOID_P, _UINT, _VOID_P, _UINT, _VOID_P, _UINT, _UINT, _OUT, _VOID_P, _OUT]),
            (d3d11.CreateDirect3D11DeviceFromDXGIDevice, [_VOID_P, _OUT]),
        ]
        combase.RoInitialize.argtypes, combase.RoInitialize.restype = [ctypes.c_int], HRESULT
        combase.RoUninitialize.argtypes, combase.RoUninitialize.restype = [], None
    except (OSError, AttributeError) as err:  # régi Windows: hiányzik a DLL vagy valamelyik függvénye
        raise WindowStreamError(f"Windows.Graphics.Capture is not available: {err}") from err
    for function, argtypes in checked:
        function.argtypes, function.restype, function.errcheck = argtypes, HRESULT, _raise_on_failure
    return combase, d3d11


def _factory(class_name: str, iid: Guid) -> AbstractContextManager[int | None]:
    """Egy WinRT-osztály aktivációs gyára a with blokk idejére."""
    combase, _ = _winrt()
    name, factory = ctypes.c_void_p(), ctypes.c_void_p()
    combase.WindowsCreateString(class_name, len(class_name), ctypes.byref(name))
    try:
        combase.RoGetActivationFactory(name, ctypes.byref(iid), ctypes.byref(factory))
    finally:
        combase.WindowsDeleteString(name)
    return _releasing(factory.value)


def mapped_rows_to_bgr(address: int | None, row_pitch: int, width: int, height: int) -> np.ndarray:
    """A leképezett textúra BGRA-soraiból saját BGR-tömb; a sorok a memóriában row_pitch bájtonként követik egymást."""
    row_bytes = width * BYTES_PER_PIXEL
    if not address or width <= 0 or height <= 0 or row_pitch < row_bytes:
        raise WindowStreamError(f"unusable mapped texture: {width}x{height}, row pitch {row_pitch}")
    # Az utolsó sor után már nincs kitöltés: pontosan ennyi bájt olvasható biztosan
    buffer = (ctypes.c_uint8 * (row_pitch * (height - 1) + row_bytes)).from_address(address)
    rows = np.ndarray((height, width, BYTES_PER_PIXEL), np.uint8, buffer, strides=(row_pitch, BYTES_PER_PIXEL, 1))
    return bgrx_to_bgr(rows)


class WindowStream:
    """Egy ablak képkockái. Létrehozáskor elindul a rögzítés, egérkurzor és sárga keret nélkül.

    Ha a keret nem kapcsolható ki (Windows 10), vagy bármi más nem megy, WindowStreamError-t dob, és
    semmit sem hagy nyitva. limits_rate: a rendszer maga ritkítja-e a képkockákat (Windows 11 24H2-től);
    ha nem, a hívó dobja el a túl sűrűn érkezőket.
    """

    def __init__(self, hwnd: int, min_interval: float) -> None:
        self._hwnd = hwnd
        self._owned: list[tuple[Callable[[int | None], None], int | None]] = []  # (elengedő, mutató) sorban
        self._frame: int | None = None  # a with frame() blokkban lefoglalt képkocka
        self._frame_size = (0, 0)
        self._staging: int | None = None
        self._staging_size = (0, 0)
        if sys.getwindowsversion().platform_version[2] < BORDERLESS_CAPTURE_BUILD:  # Windows 10: kár belekezdeni
            raise WindowStreamError("this Windows version cannot switch off the capture border")
        combase, _ = _winrt()
        # Más COM-módú szálon (pl. a Tk szálán) ez hibát ad, de a WinRT ott is működik: csak nem mi zárjuk le
        self._uninitialize = combase.RoInitialize(RO_INIT_MULTITHREADED) >= 0
        try:
            self._create_device()
            self._create_session()
            self._hide_cursor_and_border()
            self.limits_rate = self._limit_rate(min_interval)
            _call(self._session, START_CAPTURE)
        except Exception:
            self.close()
            raise

    def _own(self, pointer: int | None, dispose: Callable[[int | None], None] = _release) -> int | None:
        self._owned.append((dispose, pointer))
        return pointer

    def _create_device(self) -> None:
        _, d3d11 = _winrt()
        device, context, wrapped = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
        d3d11.D3D11CreateDevice(None, D3D_DRIVER_TYPE_HARDWARE, None, D3D11_CREATE_DEVICE_BGRA_SUPPORT, None, 0,
                                D3D11_SDK_VERSION, ctypes.byref(device), None, ctypes.byref(context))
        self._device, self._context = self._own(device.value), self._own(context.value)
        with _interface(self._device, IID_DXGI_DEVICE) as dxgi_device:
            d3d11.CreateDirect3D11DeviceFromDXGIDevice(dxgi_device, ctypes.byref(wrapped))
        with _releasing(wrapped.value) as inspectable:
            self._winrt_device = self._own(_out(inspectable, QUERY_INTERFACE, ctypes.byref(IID_DIRECT3D_DEVICE)))

    def _create_session(self) -> None:
        size = SizeInt32()
        with _factory(CAPTURE_ITEM_CLASS, IID_CAPTURE_ITEM_INTEROP) as interop:
            self._item = self._own(_out(interop, CREATE_FOR_WINDOW, self._hwnd, ctypes.byref(IID_CAPTURE_ITEM)))
        _call(self._item, ITEM_SIZE, ctypes.byref(size))
        # Lekicsinyített ablaknál a méret nulla lehet: az első valódi képkocka úgyis átméretezi a készletet
        size.width, size.height = max(size.width, 1), max(size.height, 1)
        with _factory(FRAME_POOL_CLASS, IID_FRAME_POOL_STATICS) as statics:
            self._pool = self._own(_out(statics, CREATE_POOL, self._winrt_device, DXGI_FORMAT_B8G8R8A8_UNORM,
                                        FRAME_BUFFERS, size), _close_object)
        self._frame_size = (size.width, size.height)
        self._session = self._own(_out(self._pool, CREATE_SESSION, self._item), _close_object)

    def _hide_cursor_and_border(self) -> None:
        """A StartCapture előtt: se egérkurzor a képben, se sárga keret az ablak körül."""
        with _interface(self._session, IID_SESSION_CURSOR) as session:
            _call(session, SET_CURSOR_CAPTURE, False)
        required = ctypes.c_ubyte(True)
        with _interface(self._session, IID_SESSION_BORDER) as session:  # Windows 10-en nincs ilyen interfész
            _call(session, SET_BORDER_REQUIRED, False)
            _call(session, GET_BORDER_REQUIRED, ctypes.byref(required))
        if required.value:
            raise WindowStreamError("the capture border cannot be switched off on this Windows version")

    def _limit_rate(self, min_interval: float) -> bool:
        """Windows 11 24H2-től a rendszer maga ritkít; True, ha elfogadta a legkisebb képkocka-közt."""
        try:
            with _interface(self._session, IID_SESSION_INTERVAL) as session:
                _call(session, SET_MIN_UPDATE_INTERVAL, int(min_interval * TIMESPAN_TICKS_PER_SECOND))
        except WindowStreamError as err:
            logger.debug("window stream: the system cannot limit the frame rate here (%s)", err)
            return False
        return True

    @contextmanager
    def frame(self) -> Iterator[tuple[int, int] | None]:
        """A with blokk idejére lefoglalja a következő képkockát; az érték a mérete (None, ha nincs használható).

        A blokkban a read_box() ebből a képkockából olvas. Ha az ablakot átméretezték, a képkocka-készletet
        az új mérethez igazítja, és None-t ad: a következő képkocka már jó méretű lesz.
        """
        if not self._owned:
            raise WindowStreamError("the window stream is closed")
        frame = ctypes.c_void_p()
        _call(self._pool, TRY_GET_NEXT_FRAME, ctypes.byref(frame))
        if not frame.value:
            yield None
            return
        try:
            size = self._usable_size(frame.value)
            self._frame = frame.value if size else None
            yield size
        finally:
            self._frame = None
            _close_object(frame.value)

    def _usable_size(self, frame: int) -> tuple[int, int] | None:
        content = SizeInt32()
        _call(frame, FRAME_CONTENT_SIZE, ctypes.byref(content))
        size = (content.width, content.height)
        if size == self._frame_size:
            return size
        if min(size) > 0:
            _call(self._pool, RECREATE_POOL, self._winrt_device, DXGI_FORMAT_B8G8R8A8_UNORM, FRAME_BUFFERS, content)
            self._frame_size = size
        return None

    def read_box(self, box: Box) -> np.ndarray:
        """A lefoglalt képkocka téglalapja saját BGR-tömbként; csak ennyi pixel jön le a videókártyáról."""
        width, height = self._frame_size
        if self._frame is None or not (0 <= box.left < box.right <= width and 0 <= box.top < box.bottom <= height):
            raise ValueError(f"no frame is held, or {box} does not fit the {width}x{height} frame")
        with _releasing(_out(self._frame, FRAME_SURFACE)) as surface, _interface(surface, IID_DXGI_ACCESS) as access:
            texture = _out(access, GET_DXGI_INTERFACE, ctypes.byref(IID_TEXTURE_2D))
        with _releasing(texture):
            staging = self._staging_for(box.width, box.height)
            source = D3d11Box(left=box.left, top=box.top, right=box.right, bottom=box.bottom, back=TEXTURE_DEPTH)
            # A cél (0, 0, 0) pontjára: az átmeneti textúra pontosan sávméretű
            _call(self._context, COPY_REGION, staging, FIRST_SUBRESOURCE, 0, 0, 0, texture, FIRST_SUBRESOURCE,
                  ctypes.byref(source))
            mapped = MappedSubresource()
            _call(self._context, MAP, staging, FIRST_SUBRESOURCE, D3D11_MAP_READ, D3D11_MAP_WAITS,
                  ctypes.byref(mapped))
            try:
                return mapped_rows_to_bgr(mapped.pData, mapped.RowPitch, box.width, box.height)
            finally:
                _call(self._context, UNMAP, staging, FIRST_SUBRESOURCE)

    def _staging_for(self, width: int, height: int) -> int:
        """Sávméretű, a processzor felől olvasható textúra; csak a méret változásakor készül új."""
        if (width, height) != self._staging_size:
            staging, self._staging, self._staging_size = self._staging, None, (0, 0)
            _release(staging)
            description = Texture2dDesc(
                Width=width, Height=height, MipLevels=1, ArraySize=1, Format=DXGI_FORMAT_B8G8R8A8_UNORM,
                SampleCount=1, Usage=D3D11_USAGE_STAGING, CPUAccessFlags=D3D11_CPU_ACCESS_READ)
            self._staging = _out(self._device, CREATE_TEXTURE, ctypes.byref(description), None)
            self._staging_size = (width, height)
        return self._staging

    def close(self) -> None:
        """Mindent elenged a létrehozással ellentétes sorrendben; egy hiba nem állítja meg a többit. Ismételhető."""
        pending = [*self._owned, (_release, self._staging)]
        self._owned, self._frame, self._staging, self._staging_size = [], None, None, (0, 0)
        for dispose, pointer in reversed(pending):
            try:
                dispose(pointer)
            except OSError as err:
                logger.debug("window stream: releasing a capture object failed: %s", err)
        if self._uninitialize:
            self._uninitialize = False
            _winrt()[0].RoUninitialize()
