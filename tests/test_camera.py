"""Camera-adapter tests using injected synthetic captures only."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import pytest

from gazelink.camera import (
    DEFAULT_REQUESTED_FPS,
    DEFAULT_REQUESTED_HEIGHT,
    DEFAULT_REQUESTED_WIDTH,
    CameraError,
    FixtureCameraSource,
    OpenCVCameraSource,
)
from gazelink.domain import CameraStatus, FramePacket, PixelFormat
from gazelink.errors import ErrorCode, RecoveryAction


@dataclass(frozen=True, slots=True)
class SyntheticImage:
    shape: tuple[int, int, int]
    payload: bytes

    def tobytes(self) -> bytes:
        return self.payload


class FakeCapture:
    def __init__(self, reads: list[tuple[bool, object]], *, opened: bool = True) -> None:
        self._reads = iter(reads)
        self._opened = opened
        self.release_count = 0

    def isOpened(self) -> bool:  # noqa: N802 - mirrors the OpenCV API
        return self._opened

    def read(self) -> tuple[bool, object]:
        return next(self._reads)

    def release(self) -> None:
        self.release_count += 1


class FakeCaptureWithMode(FakeCapture):
    """A capture backend that also exposes OpenCV's ``set(property_id, value)``."""

    def __init__(self, reads: list[tuple[bool, object]], *, raise_on_set: bool = False) -> None:
        super().__init__(reads)
        self.set_calls: list[tuple[int, float]] = []
        self._raise_on_set = raise_on_set

    def set(self, property_id: int, value: float) -> bool:  # noqa: A003 - mirrors OpenCV
        if self._raise_on_set:
            raise RuntimeError("unsupported mode")
        self.set_calls.append((property_id, value))
        return True


class FakeCaptureThatHangsOnSet(FakeCapture):
    """Mirrors a real driver observed blocking inside ``set()`` indefinitely.

    The call never returns, so this only stays safe because ``open()`` bounds
    how long it waits and abandons the call rather than joining it forever.
    """

    def set(self, property_id: int, value: float) -> bool:  # noqa: A003 - mirrors OpenCV
        threading.Event().wait()  # never set; blocks the calling thread forever
        raise AssertionError("unreachable: the wait above never returns")


def test_opencv_source_builds_frame_packet_without_loading_a_camera() -> None:
    capture = FakeCapture([(True, SyntheticImage((2, 3, 3), b"synthetic-pixels"))])
    source = OpenCVCameraSource(capture_factory=lambda _: capture, clock_ms=lambda: 123.5)

    source.open()
    packet = source.read()
    source.close()

    assert packet == FramePacket(
        frame_id=0,
        captured_at_monotonic_ms=123.5,
        width=3,
        height=2,
        pixel_format=PixelFormat.BGR24,
        image=b"synthetic-pixels",
    )
    assert source.status is CameraStatus.UNAVAILABLE
    assert capture.release_count == 1


def test_open_failure_has_safe_reason_and_releases_temporary_capture() -> None:
    capture = FakeCapture([], opened=False)
    source = OpenCVCameraSource(capture_factory=lambda _: capture)

    with pytest.raises(CameraError) as raised:
        source.open()

    assert raised.value.failure.code is ErrorCode.CAMERA_UNAVAILABLE
    assert raised.value.failure.recovery_action is RecoveryAction.CHECK_CAMERA
    assert source.status is CameraStatus.UNAVAILABLE
    assert capture.release_count == 1


def test_read_failure_disconnects_and_releases_camera() -> None:
    capture = FakeCapture([(False, None)])
    source = OpenCVCameraSource(capture_factory=lambda _: capture)
    source.open()

    with pytest.raises(CameraError) as raised:
        source.read()

    assert raised.value.failure.code is ErrorCode.CAMERA_DISCONNECTED
    assert raised.value.failure.recovery_action is RecoveryAction.CHECK_CAMERA
    assert source.status is CameraStatus.DISCONNECTED
    assert capture.release_count == 1
    source.close()
    assert capture.release_count == 1


def test_enumeration_releases_every_probe_and_returns_open_devices() -> None:
    captures = {
        0: FakeCapture([], opened=True),
        1: FakeCapture([], opened=False),
        2: FakeCapture([], opened=True),
    }

    def factory(index: int) -> FakeCapture:
        return captures[index]

    devices = OpenCVCameraSource.enumerate_devices(max_devices=3, capture_factory=factory)

    assert [device.index for device in devices] == [0, 2]
    assert [capture.release_count for capture in captures.values()] == [1, 1, 1]


def test_fixture_source_obeys_camera_interface_and_does_not_persist_packet() -> None:
    packet = FramePacket(0, 1.0, 1, 1, PixelFormat.GRAY8, image=b"x")
    source = FixtureCameraSource([packet])

    source.open()
    assert source.read() is packet
    with pytest.raises(CameraError, match="connection was lost"):
        source.read()
    source.close()

    assert source.status is CameraStatus.DISCONNECTED
    assert source.last_error is not None


def test_fixture_close_discards_unread_in_memory_packets() -> None:
    source = FixtureCameraSource([FramePacket(0, 1.0, 1, 1, PixelFormat.GRAY8, image=b"x")])
    source.open()

    source.close()

    source.open()
    with pytest.raises(CameraError):
        source.read()


def test_open_requests_the_reference_resolution_and_frame_rate() -> None:
    capture = FakeCaptureWithMode([(True, SyntheticImage((720, 1280, 3), b"x"))])
    source = OpenCVCameraSource(capture_factory=lambda _: capture)

    source.open()

    assert capture.set_calls == [
        (3, float(DEFAULT_REQUESTED_WIDTH)),
        (4, float(DEFAULT_REQUESTED_HEIGHT)),
        (5, float(DEFAULT_REQUESTED_FPS)),
    ]
    # The reported geometry always comes from the actual frame, not the request.
    packet = source.read()
    assert (packet.width, packet.height) == (1280, 720)


def test_custom_requested_mode_is_forwarded_to_the_capture_backend() -> None:
    capture = FakeCaptureWithMode([(True, SyntheticImage((480, 640, 3), b"x"))])
    source = OpenCVCameraSource(
        capture_factory=lambda _: capture,
        requested_width=640,
        requested_height=480,
        requested_fps=15.0,
    )

    source.open()

    assert capture.set_calls == [(3, 640.0), (4, 480.0), (5, 15.0)]


def test_a_backend_that_cannot_set_mode_still_opens_at_its_default() -> None:
    """A capture with no ``set`` (matching the plain ``FakeCapture`` above) must not fail."""

    capture = FakeCapture([(True, SyntheticImage((2, 3, 3), b"x"))])
    source = OpenCVCameraSource(capture_factory=lambda _: capture)

    source.open()

    assert source.status is CameraStatus.ACTIVE


def test_a_backend_that_rejects_the_requested_mode_still_opens() -> None:
    capture = FakeCaptureWithMode([(True, SyntheticImage((2, 3, 3), b"x"))], raise_on_set=True)
    source = OpenCVCameraSource(capture_factory=lambda _: capture)

    source.open()

    assert source.status is CameraStatus.ACTIVE


@pytest.mark.parametrize(
    ("width", "height", "fps"),
    [(0, 720, 30.0), (1280, 0, 30.0), (1280, 720, 0.0), (-1, 720, 30.0)],
)
def test_invalid_requested_mode_is_rejected(width: int, height: int, fps: float) -> None:
    with pytest.raises(ValueError, match="requested_"):
        OpenCVCameraSource(requested_width=width, requested_height=height, requested_fps=fps)


def test_a_driver_that_hangs_inside_set_cannot_hang_open() -> None:
    """Pins a real failure observed live: a driver's ``set()`` never returned.

    ``open()`` must come back within roughly the per-property watchdog budget
    (three properties are requested) rather than blocking forever, even though
    the fake's ``set()`` call is still parked on its ``Event`` in the
    background when this test finishes.
    """

    capture = FakeCaptureThatHangsOnSet([(True, SyntheticImage((2, 3, 3), b"x"))])
    source = OpenCVCameraSource(capture_factory=lambda _: capture)

    started = time.perf_counter()
    source.open()
    elapsed = time.perf_counter() - started

    assert source.status is CameraStatus.ACTIVE
    # Worst case is three properties times the per-property watchdog timeout;
    # this bound is deliberately generous so the test only fails on a real
    # regression back to an unbounded wait, not on timing noise.
    assert elapsed < 10.0, f"open() should be bounded by the watchdog timeout, took {elapsed:.1f}s"
