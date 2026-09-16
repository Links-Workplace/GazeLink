"""The gazefollower library as a TrackingSource (ARCH-01 stage E).

``observe`` is the ONLY code that reads the library's ``face_info`` and
``gaze_info``. Everything downstream -- pipeline, recorder, preflight, tools --
receives a :class:`FrameObservation`.

``GazeFollowerSource`` owns the library object for a session: it opens it,
subscribes, starts sampling, drops frames once stopping, and shuts the library
down (bounded, every step independent). It makes no gaze, click or UI
decision.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from gazelink_core.domain.clock import Clock
from gazelink_core.domain.observation import FrameObservation, HeadPolicy, Reason
from gazelink_core.gaze import preflight as PRE
from gazelink_core.tracking import face_landmarks as FL
from gazelink_core.tracking import gazefollower_library as LIB

HeadBuilder = Callable[[Any], np.ndarray[Any, Any] | None]
PnP = Callable[[Any], tuple[float, float, float] | None]


def observe(
    face_info: Any,
    gaze_info: Any,
    *,
    observed_s: float,
    frame_seq: int = 0,
    head_builder: HeadBuilder = FL.head6_from_face,
    head_policy: HeadPolicy = HeadPolicy.GAZE_OR_FACE,
    pnp: PnP | None = None,
) -> FrameObservation:
    """Translate one library callback. Raises only what the head builder raises.

    Every rule below is the one the live view and the recorder applied inline
    before this existed; the tests in ``test_pipeline_equivalence`` hold them to
    identical outputs.
    """

    reasons: set[Reason] = set()
    result_present = face_info is not None or gaze_info is not None
    if not result_present:
        reasons.add(Reason.NO_RESULT)
    gaze_status = bool(getattr(gaze_info, "status", False))
    face_present = bool(getattr(face_info, "status", False))
    if not face_present:
        reasons.add(Reason.NO_FACE)
    if not gaze_status:
        reasons.add(Reason.NO_GAZE)

    features = getattr(gaze_info, "features", None) if gaze_status else None
    if gaze_status and features is None:
        reasons.add(Reason.NO_FEATURES)

    left_raw = getattr(face_info, "left_eye_openness", None)
    right_raw = getattr(face_info, "right_eye_openness", None)
    openness_available = left_raw is not None and right_raw is not None
    if not openness_available:
        reasons.add(Reason.OPENNESS_MISSING)
    openness = (float(left_raw or 0.0), float(right_raw or 0.0))

    wants_head = gaze_status or (head_policy is HeadPolicy.GAZE_OR_FACE and face_present)
    head = head_builder(face_info) if wants_head and face_info is not None else None
    if head is None:
        reasons.add(Reason.HEAD_UNAVAILABLE)
    pnp_deg = pnp(face_info) if (pnp is not None and gaze_status and head is not None) else None

    raw = getattr(gaze_info, "raw_gaze_coordinates", None) if gaze_status else None
    raw_cm: tuple[float, float] | None = None
    if raw is not None:
        try:
            raw_cm = (float(raw[0]), float(raw[1]))
        except (TypeError, ValueError, IndexError):
            reasons.add(Reason.RAW_GAZE_INVALID)

    state_obj = getattr(gaze_info, "tracking_state", None)
    stamp = int(getattr(gaze_info, "timestamp", 0) or getattr(face_info, "timestamp", 0) or 0)
    return FrameObservation(
        frame_seq=int(frame_seq),
        observed_s=float(observed_s),
        timestamp_ns=stamp,
        timestamp_available=stamp != 0,
        face_present=face_present,
        gaze_status=gaze_status,
        features=None if features is None else np.asarray(features),
        openness_image=openness,
        openness_available=openness_available,
        head6=None if head is None else np.asarray(head),
        pnp_deg=pnp_deg,
        raw_gaze_cm=raw_cm,
        tracking_state_name=getattr(state_obj, "name", None),
        tracking_state_text=str(state_obj),
        tracking_state_present=state_obj is not None,
        result_present=result_present,
        reasons=frozenset(reasons),
    )


def gaze_status_of(gaze_info: Any) -> bool:
    """Did the library produce a usable gaze for this callback?"""

    return bool(getattr(gaze_info, "status", False))


def design_row_from_library(
    model: Any, face_info: Any, gaze_info: Any, head_builder: HeadBuilder = FL.head6_from_face
) -> np.ndarray[Any, Any] | None:
    """The recorder's preflight row straight from a library callback.

    Reads the features whatever the frame's status (the caller checks status),
    and builds the head pose only if the model needs it -- exactly as the
    recorder always has.
    """

    try:
        features = getattr(gaze_info, "features", None)
    except Exception:  # noqa: BLE001 - a preflight sample must never break recording
        return None
    return PRE.design_row(model, features, lambda: head_builder(face_info))


ObservationSink = Callable[[FrameObservation], None]
# How long ``stop`` waits for a frame already being delivered. One camera frame
# is ~33 ms; a sink that takes longer than this is stuck, not busy.
STOP_WAIT_S = 1.0


class GazeFollowerSource:
    """One library session: open, observe, publish, stop, close."""

    def __init__(
        self,
        profile: Any,
        *,
        clock: Clock,
        head_builder: HeadBuilder = FL.head6_from_face,
        head_policy: HeadPolicy = HeadPolicy.GAZE_OR_FACE,
        open_library: Callable[[Any], Any] = LIB.build_gaze_follower,
        shutdown_library: Callable[[Any], Any] = LIB.shutdown_library,
    ) -> None:
        self.profile = profile
        self.clock = clock
        self.head_builder = head_builder
        self.head_policy = head_policy
        self._open_library = open_library
        self._shutdown_library = shutdown_library
        self._library: Any = None
        self._sinks: list[ObservationSink] = []
        self._stopping = threading.Event()
        self._lock = threading.Lock()
        # Held while one observation is being handed to the sinks, so ``stop``
        # can wait for a frame already in flight (re-entrant: a sink may stop
        # the source from inside its own callback).
        self._publishing = threading.RLock()
        self._frame_seq = 0
        self.closed = False
        self.shutdown_report: Any = None
        # Frames the translation itself could not turn into an observation (a
        # head builder that raised). Reported with the pipeline's own errors.
        self.errors = 0
        self.last_error: str | None = None

    def open(self) -> GazeFollowerSource:
        """Open the library and attach to it. Refusals (capture mismatch) raise here."""

        self._library = self._open_library(self.profile)
        self._library.add_subscriber(self._on_library_frame)
        return self

    def subscribe(self, sink: ObservationSink) -> None:
        with self._lock:
            self._sinks.append(sink)

    def start(self) -> None:
        # camera.start_sampling() directly, as both the recorder and the live
        # view always have: GazeFollower.start_sampling() would also attach its
        # CSV writer, which raises on status-False frames.
        self._library.camera.start_sampling()

    def stop(self, *, wait_s: float = STOP_WAIT_S) -> bool:
        """Stop publishing. Returns whether no frame is still being delivered.

        New frames are dropped from the moment the flag is set. A frame that
        was already being handed to the sinks is waited for, up to ``wait_s``:
        after a True return no sink is running and none will run again. False
        means a sink is stuck past the bound; the caller's cleanup proceeds
        regardless (a stalled camera thread must not keep input held).
        """

        self._stopping.set()
        acquired = self._publishing.acquire(timeout=wait_s)
        if acquired:
            self._publishing.release()
        return acquired

    @property
    def stopping(self) -> bool:
        return self._stopping.is_set()

    def close(self) -> Any:
        """Stop publishing and shut the library down. Idempotent."""

        self.stop()
        if self.closed:
            return self.shutdown_report
        self.closed = True
        if self._library is not None:
            self.shutdown_report = self._shutdown_library(self._library)
        return self.shutdown_report

    def _on_library_frame(self, face_info: Any, gaze_info: Any) -> None:
        if self._stopping.is_set():
            return
        with self._lock:
            seq = self._frame_seq
            self._frame_seq += 1
            sinks = list(self._sinks)
        with self._publishing:
            if self._stopping.is_set():
                return
            self._publish(face_info, gaze_info, seq, sinks)

    def _publish(
        self, face_info: Any, gaze_info: Any, seq: int, sinks: list[ObservationSink]
    ) -> None:
        try:
            obs = observe(
                face_info,
                gaze_info,
                observed_s=self.clock.now(),
                frame_seq=seq,
                head_builder=self.head_builder,
                head_policy=self.head_policy,
            )
        except Exception as exc:  # noqa: BLE001 - never raise into the library's camera thread
            with self._lock:
                self.errors += 1
                self.last_error = repr(exc)
            return
        for sink in sinks:
            if self._stopping.is_set():
                return
            sink(obs)
