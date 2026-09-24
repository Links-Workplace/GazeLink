"""Reading the screen for the magnifier, in memory and never to disk.

Pillow's ``ImageGrab``. Measured on this rig on 2026-09-14: a 640x400 grab took
a median of 67 ms (max 81), and with ``include_layered_windows=False`` -- the
default -- our own layered overlay is NOT captured, which is the only reason
the magnifier does not photograph itself.

Privacy (CLAUDE.md 4.3). These are not biometric samples, but they are the
person's content. Nothing here writes a file, and nothing here logs pixels:
the only things that leave this module are a byte string a caller compares
against another byte string, and sizes.

Pillow is installed in the run environment (12.3.0) but is NOT declared in
``pyproject.toml`` or ``requirements.lock``. That gap is recorded in TASKS.md
rather than papered over by editing a lock file that ``pip-compile`` owns.
"""

from __future__ import annotations

from typing import Any

from gazelink_core.interaction.capture_port import DeviceRect


class CaptureUnavailable(RuntimeError):
    """The screen could not be read. Never raised into a live frame path."""


class PillowCapture:
    """One frozen region. The image stays in memory for as long as this lives."""

    def __init__(self, rect: DeviceRect, image: Any) -> None:
        self._rect = rect
        self._image = image

    @property
    def rect(self) -> DeviceRect:
        return self._rect

    def crop_bytes(self, rect: DeviceRect) -> bytes:
        if not (
            rect.x0 >= self._rect.x0
            and rect.y0 >= self._rect.y0
            and rect.x1 <= self._rect.x1
            and rect.y1 <= self._rect.y1
        ):
            raise ValueError(f"{rect} is not inside the captured {self._rect}")
        box = (
            rect.x0 - self._rect.x0,
            rect.y0 - self._rect.y0,
            rect.x1 - self._rect.x0,
            rect.y1 - self._rect.y0,
        )
        return self._image.crop(box).tobytes()


class PillowScreenCapture:
    """The real reader. Constructed at the entry point, injected everywhere else."""

    def __init__(self, *, grabber: Any = None) -> None:
        self._grabber = grabber

    def _image_grab(self) -> Any:
        if self._grabber is not None:
            return self._grabber
        try:
            from PIL import ImageGrab  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001 - reported, never a crash
            raise CaptureUnavailable(f"Pillow is not available: {exc}") from exc
        return ImageGrab.grab

    def grab(self, rect: DeviceRect) -> PillowCapture:
        grab = self._image_grab()
        try:
            # all_screens keeps the box in VIRTUAL desktop coordinates, which
            # is the space every rectangle in this path is expressed in.
            # include_layered_windows stays at its default False so our own
            # overlay is not in the picture.
            image = grab(bbox=(rect.x0, rect.y0, rect.x1, rect.y1), all_screens=True)
        except TypeError:
            # Older Pillow without all_screens. Reported rather than silently
            # returning a primary-monitor-only grab that would be offset.
            raise CaptureUnavailable(
                "this Pillow cannot grab the virtual desktop (no all_screens)"
            ) from None
        except Exception as exc:  # noqa: BLE001
            raise CaptureUnavailable(f"the screen could not be read: {exc}") from exc
        if image is None:
            raise CaptureUnavailable("the screen grab returned nothing")
        return PillowCapture(rect, image.convert("RGB"))
