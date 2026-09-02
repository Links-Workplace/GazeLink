"""Camera adapters that keep capture hardware behind a small, testable boundary.

Frames are retained only in the returned :class:`~gazelink.domain.FramePacket`.
This module never writes a frame, image, or camera diagnostic to disk.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass
from importlib import import_module
from threading import Thread
from time import perf_counter_ns
from typing import Any, Protocol, cast

from gazelink.domain import CameraStatus, FramePacket, PixelFormat
from gazelink.errors import ErrorCode, UserFacingError, user_facing_error


class CaptureBackend(Protocol):
    """The minimal OpenCV capture surface used by ``OpenCVCameraSource``."""

    def isOpened(self) -> bool: ...  # noqa: N802 - mirrors the OpenCV API

    def read(self) -> tuple[bool, object]: ...

    def release(self) -> None: ...


# The reference target from the M1 spike is 1280x720@30FPS (docs/m1-spike-report.md).
# A camera that ignores the request keeps working: geometry is always read back
# from the actual captured frame, never assumed from what was requested.
DEFAULT_REQUESTED_WIDTH = 1280
DEFAULT_REQUESTED_HEIGHT = 720
DEFAULT_REQUESTED_FPS = 30.0

# OpenCV's VideoCapture property IDs. These are a stable part of the public C
# API and are hardcoded here so requesting a capture mode does not require
# importing cv2 -- the same reason ``_default_capture_factory`` imports it lazily.
_CAP_PROP_FRAME_WIDTH = 3
_CAP_PROP_FRAME_HEIGHT = 4
_CAP_PROP_FPS = 5

# Some Windows camera drivers block indefinitely inside ``set()`` while they
# renegotiate a capture format -- this was observed live on real hardware, not
# hypothesized. Opening the camera must never hang because of that, so all
# three property requests run together on one bounded watchdog and are
# abandoned past this shared deadline rather than awaited. 4.0s comfortably
# covers the ~0.3-1.0s per property this machine's camera was observed taking
# (all three combined, sequentially, in a single thread), without letting a
# request eat an unbounded share of startup time.
_MODE_REQUEST_TIMEOUT_S = 4.0


class CameraSource(Protocol):
    """A source of transient frame packets; implementations never persist frames."""

    @property
    def status(self) -> CameraStatus: ...

    @property
    def last_error(self) -> UserFacingError | None: ...

    def open(self) -> None: ...

    def read(self) -> FramePacket: ...

    def close(self) -> None: ...


class CameraError(RuntimeError):
    """A camera failure with a safe reason code and recovery instruction."""

    def __init__(self, code: ErrorCode, *, state: str) -> None:
        self.failure = user_facing_error(code, state=state)
        super().__init__(self.failure.message)


@dataclass(frozen=True, slots=True)
class CameraDevice:
    """A discovered device identifier, without a frame or user-specific data."""

    index: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("camera index must be non-negative")


def _monotonic_ms() -> float:
    return perf_counter_ns() / 1_000_000


def _default_capture_factory(camera_index: int) -> CaptureBackend:
    """Create an OpenCV capture lazily so imports alone never touch a camera."""

    cv2: Any = import_module("cv2")
    return cast(CaptureBackend, cv2.VideoCapture(camera_index))


def _image_geometry(image: object) -> tuple[int, int, PixelFormat]:
    shape = getattr(image, "shape", None)
    if not isinstance(shape, tuple) or len(shape) < 2:
        raise ValueError("captured frame must provide image.shape")
    height, width = shape[0], shape[1]
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("captured frame height must be a positive integer")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("captured frame width must be a positive integer")
    channels = 1 if len(shape) == 2 else shape[2]
    if channels == 1:
        pixel_format = PixelFormat.GRAY8
    elif channels == 3:
        pixel_format = PixelFormat.BGR24
    else:
        raise ValueError("captured frame must have one or three channels")
    return width, height, pixel_format


def _image_bytes(image: object) -> bytes:
    tobytes = getattr(image, "tobytes", None)
    if not callable(tobytes):
        raise ValueError("captured frame must provide image.tobytes()")
    payload = tobytes()
    if not isinstance(payload, bytes):
        raise ValueError("captured frame bytes must be immutable")
    return payload


class OpenCVCameraSource:
    """OpenCV-backed ``CameraSource`` with injected hardware and clock seams."""

    def __init__(
        self,
        camera_index: int = 0,
        *,
        capture_factory: Callable[[int], CaptureBackend] | None = None,
        clock_ms: Callable[[], float] = _monotonic_ms,
        requested_width: int = DEFAULT_REQUESTED_WIDTH,
        requested_height: int = DEFAULT_REQUESTED_HEIGHT,
        requested_fps: float = DEFAULT_REQUESTED_FPS,
    ) -> None:
        if camera_index < 0:
            raise ValueError("camera_index must be non-negative")
        if requested_width <= 0 or requested_height <= 0 or requested_fps <= 0:
            raise ValueError("requested_width, requested_height, and requested_fps must be > 0")
        self._camera_index = camera_index
        self._capture_factory = capture_factory or _default_capture_factory
        self._clock_ms = clock_ms
        self._requested_width = requested_width
        self._requested_height = requested_height
        self._requested_fps = requested_fps
        self._capture: CaptureBackend | None = None
        self._frame_id = 0
        self._status = CameraStatus.UNAVAILABLE
        self._last_error: UserFacingError | None = None

    @property
    def status(self) -> CameraStatus:
        return self._status

    @property
    def last_error(self) -> UserFacingError | None:
        return self._last_error

    @classmethod
    def enumerate_devices(
        cls,
        *,
        max_devices: int = 8,
        capture_factory: Callable[[int], CaptureBackend] | None = None,
    ) -> tuple[CameraDevice, ...]:
        """Probe a bounded index range and always release every temporary handle."""

        if max_devices < 1:
            raise ValueError("max_devices must be at least one")
        factory = capture_factory or _default_capture_factory
        devices: list[CameraDevice] = []
        for index in range(max_devices):
            capture: CaptureBackend | None = None
            try:
                capture = factory(index)
                if capture.isOpened():
                    devices.append(CameraDevice(index=index))
            except Exception:
                # Enumeration is best-effort. Opening the selected source returns
                # the structured failure that UI and callers should present.
                continue
            finally:
                if capture is not None:
                    capture.release()
        return tuple(devices)

    def open(self) -> None:
        """Acquire the selected camera, or raise a structured unavailable error."""

        if self._capture is not None:
            return
        self._status = CameraStatus.STARTING
        capture: CaptureBackend | None = None
        try:
            capture = self._capture_factory(self._camera_index)
            if not capture.isOpened():
                raise CameraError(ErrorCode.CAMERA_UNAVAILABLE, state="camera_open")
            self._request_mode(capture)
        except CameraError as error:
            self._fail(error, capture)
            raise
        except Exception as error:
            unavailable = CameraError(ErrorCode.CAMERA_UNAVAILABLE, state="camera_open")
            self._fail(unavailable, capture)
            raise unavailable from error
        self._capture = capture
        self._status = CameraStatus.ACTIVE
        self._last_error = None

    def _request_mode(self, capture: CaptureBackend) -> None:
        """Ask for the target resolution and frame rate; never assume it was granted.

        A capture backend that ignores or partially honours the request is not a
        failure: every frame's actual dimensions are read back from the frame
        itself in ``read()``, so nothing downstream can be misled by this call.

        All three properties are requested from a single daemon watchdog thread,
        not one thread per property: on real hardware, spawning a fresh OS
        thread for each call added roughly a second of overhead per property
        (observed live), almost certainly first-touch COM/driver setup cost on
        that thread. Paying that cost once instead of three times keeps a
        healthy driver's startup latency close to what a single synchronous
        call would take, while a single shared deadline still bounds the total
        wait -- opening the camera must never hang, no matter how a specific
        driver behaves.
        """

        set_property = getattr(capture, "set", None)
        if not callable(set_property):
            return

        def call_all() -> None:
            for property_id, value in (
                (_CAP_PROP_FRAME_WIDTH, float(self._requested_width)),
                (_CAP_PROP_FRAME_HEIGHT, float(self._requested_height)),
                (_CAP_PROP_FPS, float(self._requested_fps)),
            ):
                # Requesting a mode is best-effort; a backend that raises on an
                # unsupported request must not prevent opening at its default mode.
                with suppress(Exception):
                    set_property(property_id, value)

        worker = Thread(target=call_all, daemon=True)
        worker.start()
        worker.join(timeout=_MODE_REQUEST_TIMEOUT_S)

    def read(self) -> FramePacket:
        """Read one current frame and envelope it with a monotonic timestamp."""

        capture = self._capture
        if capture is None or self._status is not CameraStatus.ACTIVE:
            error = CameraError(ErrorCode.CAMERA_UNAVAILABLE, state="camera_read")
            self._last_error = error.failure
            raise error
        try:
            ok, image = capture.read()
            if not ok or image is None:
                raise CameraError(ErrorCode.CAMERA_DISCONNECTED, state="camera_read")
            width, height, pixel_format = _image_geometry(image)
            packet = FramePacket(
                frame_id=self._frame_id,
                captured_at_monotonic_ms=self._clock_ms(),
                width=width,
                height=height,
                pixel_format=pixel_format,
                image=_image_bytes(image),
            )
        except CameraError as error:
            self._fail(error, capture)
            raise
        except Exception as error:
            disconnected = CameraError(ErrorCode.CAMERA_DISCONNECTED, state="camera_read")
            self._fail(disconnected, capture)
            raise disconnected from error
        self._frame_id += 1
        return packet

    def close(self) -> None:
        """Release the device exactly once; safe to call on every exit path."""

        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()
        if self._status is not CameraStatus.DISCONNECTED:
            self._status = CameraStatus.UNAVAILABLE

    def _fail(self, error: CameraError, capture: CaptureBackend | None) -> None:
        if capture is not None:
            capture.release()
        self._capture = None
        self._last_error = error.failure
        self._status = (
            CameraStatus.DISCONNECTED
            if error.failure.code is ErrorCode.CAMERA_DISCONNECTED
            else CameraStatus.UNAVAILABLE
        )


class FixtureCameraSource:
    """Deterministic ``CameraSource`` for unit/integration tests with no hardware."""

    def __init__(self, packets: Iterable[FramePacket] = ()) -> None:
        self._packets: deque[FramePacket] = deque(packets)
        self._status = CameraStatus.UNAVAILABLE
        self._last_error: UserFacingError | None = None
        self.open_count = 0
        self.close_count = 0

    @property
    def status(self) -> CameraStatus:
        return self._status

    @property
    def last_error(self) -> UserFacingError | None:
        return self._last_error

    def open(self) -> None:
        if self._status is CameraStatus.ACTIVE:
            return
        self._status = CameraStatus.ACTIVE
        self._last_error = None
        self.open_count += 1

    def read(self) -> FramePacket:
        if self._status is not CameraStatus.ACTIVE:
            error = CameraError(ErrorCode.CAMERA_UNAVAILABLE, state="fixture_camera_read")
            self._last_error = error.failure
            raise error
        if not self._packets:
            error = CameraError(ErrorCode.CAMERA_DISCONNECTED, state="fixture_camera_read")
            self._last_error = error.failure
            self._status = CameraStatus.DISCONNECTED
            raise error
        return self._packets.popleft()

    def close(self) -> None:
        # Every call is counted, including a close that follows a disconnect.
        # Gating this on ACTIVE would make the counter unable to distinguish
        # "released after a camera error" from "never released at all", which is
        # the exact property resource-cleanup tests need to prove.
        self.close_count += 1
        self._packets.clear()
        if self._status is not CameraStatus.DISCONNECTED:
            self._status = CameraStatus.UNAVAILABLE
