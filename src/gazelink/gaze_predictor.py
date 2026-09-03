"""One prediction port, two independent engines behind it.

GAZELINK's own engine turns an already-vetted :class:`VisionObservation` into a
screen point.  An external library (see :mod:`gazelink.eyegestures_engine`)
instead wants the raw frame and owns its whole pipeline end to end.  Those two
never meet at the feature level, so this module defines the one place they do
meet: a port that receives *both* the frame and the observation, and returns
the existing :class:`~gazelink.gaze_engine.GazeEstimationResult`.

Because the return type is unchanged, nothing downstream -- gesture handling,
UI, or measurement -- can tell which engine produced a point.  That is the
whole point: an alternative engine must be swappable without any consumer
growing a branch for it.

Nothing here trains, calibrates, or corrects anything, and nothing here edits
the existing engine.  :class:`NativeGazePredictor` is a pass-through wrapper
around the untouched :class:`~gazelink.gaze_engine.GazeEstimator`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    # Annotation-only. Keeping these out of the runtime import graph is what
    # lets app.py read the engine names for its parser without dragging the
    # estimator (and numpy) into the plain ``--smoke`` path.
    from gazelink.domain import FramePacket, ScreenGeometry, VisionObservation
    from gazelink.gaze_engine import GazeEstimationResult, GazeEstimator

NATIVE_ENGINE = "native"
EYEGESTURES_ENGINE = "eyegestures"
ENGINE_CHOICES = (NATIVE_ENGINE, EYEGESTURES_ENGINE)


class GazePredictor(Protocol):
    """A source of screen points, whatever produces them underneath."""

    @property
    def engine_name(self) -> str:
        """Stable identifier shown in diagnostics, e.g. ``"native"``."""

    @property
    def screen_geometry(self) -> ScreenGeometry:
        """The geometry every returned pixel coordinate is expressed in."""

    def predict(
        self,
        *,
        frame: FramePacket,
        observation: VisionObservation,
        now_monotonic_ms: float,
    ) -> GazeEstimationResult:
        """Return a point for this frame, or a typed rejection with reasons.

        Implementations must never invent a coordinate for input they could not
        actually resolve -- a rejection is ``sample=None`` plus at least one
        reason code, exactly as the native engine already behaves.
        """

    def close(self) -> None:
        """Release engine resources; repeated calls must be safe."""


class NativeGazePredictor:
    """The existing engine, unchanged, behind the shared port.

    ``frame`` is accepted and deliberately ignored: the native pipeline already
    consumed it upstream to build ``observation``.  Taking it anyway is what
    lets both engines sit behind one signature.
    """

    def __init__(self, estimator: GazeEstimator) -> None:
        self._estimator = estimator

    @property
    def engine_name(self) -> str:
        return NATIVE_ENGINE

    @property
    def model_id(self) -> str:
        return self._estimator.model_id

    @property
    def estimator(self) -> GazeEstimator:
        """The wrapped estimator, for consumers that still need corrections."""

        return self._estimator

    @property
    def screen_geometry(self) -> ScreenGeometry:
        return self._estimator.screen_geometry

    def predict(
        self,
        *,
        frame: FramePacket,
        observation: VisionObservation,
        now_monotonic_ms: float,
    ) -> GazeEstimationResult:
        del frame  # consumed upstream; see the class docstring
        return self._estimator.estimate(observation, now_monotonic_ms=now_monotonic_ms)

    def close(self) -> None:
        """The native estimator holds no releasable resource of its own."""
