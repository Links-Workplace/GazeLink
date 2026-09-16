"""The gazefollower library's lifecycle: calibration stub, warm-up, shutdown.

(ARCH-01 stage D.) The only module besides the tracking source that touches
the library's own objects. ``gazefollower`` is imported lazily: importing it
initialises native components.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from gazelink_core.calibration import profile as PROF
from gazelink_core.domain import common as C

# Library cleanup calls join() with no timeout; every step gets its own bound
# so one stalled thread cannot prevent the capture release or the CSV close.
SHUTDOWN_STEP_TIMEOUT_S = 3.0

CAMERA_WARMUP_S = 2.0  # camera open, no target: lets the tracker lock on

# --- Library adapters (imported lazily) --------------------------------------


def _library_dir() -> str:
    """Where the installed library lives; the display borrows its dot and beep."""

    import gazefollower  # noqa: PLC0415

    return str(Path(gazefollower.__file__).resolve().parent)


def make_pass_through_calibration() -> Any:
    from gazefollower.calibration import Calibration  # noqa: PLC0415

    class PassThroughCalibration(Calibration):  # type: ignore[misc]
        """Lets SAMPLING run without a fitted model. Its output is NOT a prediction."""

        def __init__(self) -> None:
            super().__init__()
            self.has_calibrated = True

        def calibrate(self, features, labels, ids=None):  # noqa: ANN001
            raise RuntimeError("recording mode never trains a model")

        def predict(self, features, estimated_coordinate):  # noqa: ANN001
            return True, (float(estimated_coordinate[0]), float(estimated_coordinate[1]))

        def save_model(self) -> bool:
            return False

        def release(self) -> None:
            return None

    return PassThroughCalibration()


def _call_with_timeout(func: Callable[[], Any], timeout_s: float) -> str:
    """Bounded call, reported as a one-line verdict for the shutdown report.

    Wraps the shared helper in :mod:`gf_common`, which exists because the
    library's ``close()`` and ``release()`` join their capture thread with no
    timeout: a stalled camera would otherwise block cleanup before the capture
    is released or the CSV closed. A timed-out step leaks a daemon thread,
    which dies with the process -- strictly better than never reaching the
    steps that follow.
    """

    finished, result = C.call_with_timeout(func, timeout_s)
    if not finished:
        return f"TIMEOUT after {timeout_s}s"
    if isinstance(result, BaseException):
        return repr(result)
    return "ok"


def shutdown_library(gf: Any, *, join_timeout_s: float = SHUTDOWN_STEP_TIMEOUT_S) -> dict[str, Any]:
    """Eval-owned cleanup, every step independent and time-bounded.

    The library's own ``close()`` only releases the capture when it is NOT
    open (an inverted condition), and ``release()`` joins a possibly-None
    thread. Neither can be relied on, and neither may prevent the steps after
    it, so each is bounded and its verdict recorded separately.
    """

    report: dict[str, Any] = {}
    report["library_release"] = _call_with_timeout(gf.release, join_timeout_s)
    camera = getattr(gf, "camera", None)
    thread = getattr(camera, "_camera_thread", None)
    try:
        if camera is not None:
            camera._camera_thread_running = False
        if thread is not None and getattr(thread, "is_alive", lambda: False)():
            thread.join(timeout=join_timeout_s)
            report["thread_joined"] = not thread.is_alive()
        else:
            report["thread_joined"] = None
    except Exception as exc:  # noqa: BLE001
        report["thread_joined"] = repr(exc)
    cap = getattr(camera, "_cap", None)
    try:
        if cap is not None and cap.isOpened():
            cap.release()
            report["capture_released"] = True
        else:
            report["capture_released"] = False
    except Exception as exc:  # noqa: BLE001
        report["capture_released"] = repr(exc)
    stream = getattr(gf, "_tmpSampleDataSteam", None)
    try:
        if stream is not None and not getattr(stream, "closed", True):
            stream.close()
            report["tmp_stream_closed"] = True
        else:
            report["tmp_stream_closed"] = False
    except Exception as exc:  # noqa: BLE001
        report["tmp_stream_closed"] = repr(exc)
    return report


def library_config(rig: Any, config: Any) -> Any:
    """The library configuration a free-running session uses, for this rig.

    One description instead of a copy per tool: pass-through calibration mode,
    camera position and physical screen from the rig, screen size in the rig's
    device pixels. (The recorder builds its own: it may keep the library's
    primary-monitor size, and verifies it against the rig.)
    """

    config.cali_mode = 9
    config.camera_position = (rig.camera_x_cm, rig.camera_y_cm)
    config.screen_physical_size = (rig.screen_w_cm, rig.screen_h_cm)
    config.screen_size = np.array([rig.device_w_px, rig.device_h_px])
    return config


def build_gaze_follower(profile: Any) -> Any:
    """Open the library for a free-running session with this profile.

    Shared by every live tool so there is exactly one description of how the
    library is configured; two copies would drift. Takes the PROFILE, not the
    rig, so no caller can open a camera without the capture pipeline its model
    was trained on: the match is checked here and a mismatch refuses to start.
    """

    # Checked before the library is even imported: a refused session must not
    # have started anything.
    pipeline = PROF.require_capture_match(profile)

    import gazefollower  # noqa: F401, PLC0415 - initialises native components
    from gazefollower import GazeFollower  # noqa: PLC0415
    from gazefollower.misc import DefaultConfig  # noqa: PLC0415

    rig = profile.rig_geometry()
    config = library_config(rig, DefaultConfig())
    kwargs: dict[str, Any] = {"config": config, "calibration": make_pass_through_calibration()}
    if pipeline == PROF.CAPTURE_HI:
        from gazelink_core.tracking import (
            capture as CAP,  # noqa: PLC0415 - only the hi path needs it
        )

        camera, alignment, estimator = CAP.make_hi_components()
        kwargs.update(camera=camera, face_alignment=alignment, gaze_estimator=estimator)
    return GazeFollower(**kwargs)
