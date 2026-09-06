"""An alternative gaze engine backed by the external EyeGestures library.

This module is an *adapter*, not an integration.  EyeGestures owns its own
face mesh, its own calibration, and its own regression; it never sees this
project's features, model, or calibration data, and nothing here is imported
by ``gaze_model.py``, ``gaze_features.py``, or ``calibration.py``.  The only
thing crossing the boundary is a camera frame in and a screen point out.

LICENSING: EyeGestures is GPL-3.0 while GAZELINK is proprietary.  It is
therefore an *optional* extra (``pip install -e ".[eyegestures]"``) and is
imported lazily, so the default dependency closure stays free of GPL code and
the native engine runs whether or not the library is installed.  Distributing
GAZELINK together with it would trigger GPL-3 copyleft -- see TASKS.md.

Two safety properties this adapter exists to enforce:

1. ``Calibrator_v2.predict()`` inside EyeGestures returns ``[0.0, 0.0]``
   before it has been fitted.  That is the screen's top-left corner -- a
   perfectly legitimate-looking coordinate for a point that means "I have no
   idea".  This adapter refuses to emit any sample until its own calibration
   sequence has completed, so that value can never reach a consumer.
2. GAZELINK's confidence and tracking policy stays authoritative over what is
   EMITTED.  The frame itself is always handed to the library -- ``step()`` is
   called unconditionally, because the library runs its own calibration from
   the frames it sees and starving it hangs the calibration screen.  Feeding a
   frame is not an output; the gate sits on the ``GazeSample``.

   (An earlier version of this docstring claimed the adapter "never calls into
   the library at all" when untracked.  That was never true of the code.)

   A measurement run may opt out with ``require_tracked_observation=False``,
   so the library's own accuracy can be measured without our vision policy
   silently removing most of its output.  Opting out does not discard the
   verdict: the gate's decision is recorded per sample via
   ``gate_would_accept`` and travels in the sample's reason codes.  The
   default is ``True``, so every other caller is unaffected.
3. EyeGestures 3.2.4 *raises* on a frame with no detectable face rather than
   returning empty output (see ``predict``).  That is the normal state before
   the user sits down, so the adapter contains it and reports a lost frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from gazelink.domain import (
    ContractValidationError,
    FramePacket,
    GazePoint,
    GazeSample,
    PixelFormat,
    ReasonCode,
    ScreenGeometry,
    TrackingState,
    VisionObservation,
)
from gazelink.gaze_engine import GazeEstimationResult, normalized_to_pixel

# Matches the calibration recipe in the library's own working example
# (examples/simple_example_v2.py): a 6x6 normalized grid, shuffled, of which
# the first ``DEFAULT_CALIBRATION_POINTS`` are actually walked through.
_CALIBRATION_GRID_STEP = 0.2
DEFAULT_CALIBRATION_POINTS = 26
DEFAULT_CALIBRATION_SEED = 20260903
DEFAULT_CLASSICAL_IMPACT = 2
DEFAULT_FIXATION = 1.0
DEFAULT_CONTEXT = "gazelink"
# Qt is single-threaded and this engine adds a second face mesh per frame on top
# of ours. At the native 16ms interval the paint events never get a slot and the
# window looks hung. Lives here rather than in a window module so a measurement
# screen can use it without importing smoothing and correction it must not have.
EXTERNAL_ENGINE_TIMER_INTERVAL_MS = 66

_SUPPORTED_PIXEL_FORMATS = (PixelFormat.RGB24, PixelFormat.BGR24)


class EyeGesturesLike(Protocol):
    """The slice of the EyeGestures v2 API this adapter uses.

    Declared as a Protocol so tests can inject a double and never import the
    real GPL library or run a face mesh.  Method names intentionally mirror
    the upstream camelCase API.
    """

    def uploadCalibrationMap(  # noqa: N802 - mirrors the EyeGestures API
        self, points: Any, context: str = ...
    ) -> None: ...

    def setClassicalImpact(self, impact: int) -> None: ...  # noqa: N802

    def setFixation(self, fixation: float) -> None: ...  # noqa: N802

    def step(
        self, frame: Any, calibration: bool, width: int, height: int, context: str = ...
    ) -> tuple[Any, Any]: ...


@dataclass(frozen=True, slots=True)
class EyeGesturesCalibrationView:
    """What a window should draw for the library's own calibration pass.

    This is EyeGestures' calibration, not GAZELINK's 9-point one; the two are
    unrelated and never exchange data.
    """

    active: bool
    target_normalized: GazePoint | None
    acceptance_radius_px: float
    completed_points: int
    total_points: int
    # The library's target in ITS OWN pixels, exactly as `Cevent.point` gave
    # it. The library fits Ridge against this value, so drawing the glyph
    # anywhere else silently teaches it a wrong label. Normalizing and then
    # re-expanding would also cross the library's `n * W` convention with our
    # `n * (W - 1)` one, which is a second, smaller disagreement.
    target_pixel: tuple[float, float] | None = None

    @property
    def progress_text(self) -> str:
        return f"{self.completed_points}/{self.total_points}"


def calibration_grid_points() -> tuple[GazePoint, ...]:
    """The normalized grid EyeGestures calibrates on, as points.

    Same source as :func:`build_calibration_map`, so a held-out test set can be
    generated against the grid the library ACTUALLY trains on rather than a
    hand-copied duplicate that could drift out of step with it.
    """

    return tuple(GazePoint(float(x), float(y)) for x, y in build_calibration_map())


def gate_would_accept(observation: VisionObservation | None) -> bool:
    """Would our own confidence/tracking policy have accepted this frame?

    One definition, used by both the adapter (to stamp a sample's reason
    codes) and by a measurement run (to record the gate's verdict per sample
    without letting it filter anything).  Keeping it here rather than
    re-deriving the condition at each call site is what stops the two from
    drifting apart and quietly disagreeing about what "accepted" meant.
    """

    return observation is not None and observation.tracking_state is TrackingState.TRACKED


def build_calibration_map(
    *, seed: int = DEFAULT_CALIBRATION_SEED, step: float = _CALIBRATION_GRID_STEP
) -> npt.NDArray[np.float64]:
    """Return a deterministic shuffled normalized grid for EyeGestures.

    The library's example shuffles with the global ``numpy.random`` state,
    which would make two runs on the same machine non-comparable.  A seeded
    generator keeps the calibration order reproducible across runs and days,
    matching how ``test_points.generate_test_targets`` is already seeded.
    """

    axis = np.arange(0.0, 1.0 + step / 2.0, step)
    grid_x, grid_y = np.meshgrid(axis, axis)
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    np.random.default_rng(seed).shuffle(points)
    return points


# Every point of the grid we upload, so a measurement run walks the whole thing.
# Derived from the grid rather than written out: a count that can disagree with
# the grid is a count that eventually will. The previous default of 26 left ten
# uploaded points never visited, and nothing in the code said so.
FULL_CALIBRATION_POINTS = len(build_calibration_map())


def frame_to_rgb_array(frame: FramePacket) -> npt.NDArray[np.uint8] | None:
    """Convert a frame envelope to the contiguous RGB array EyeGestures wants.

    Returns ``None`` -- never a partly-valid array -- for a frame this adapter
    cannot faithfully represent: no image bytes, an unsupported pixel format,
    or a buffer whose length disagrees with the declared geometry.

    RGB (not BGR) is deliberate: the library's own working example passes an
    RGB array into ``step()``, and its internal channel handling is calibrated
    around that.
    """

    if not isinstance(frame, FramePacket):
        raise ContractValidationError("frame must be a FramePacket")
    if frame.image is None or frame.pixel_format not in _SUPPORTED_PIXEL_FORMATS:
        return None
    expected = frame.width * frame.height * 3
    if len(frame.image) != expected:
        return None
    array = np.frombuffer(frame.image, dtype=np.uint8).reshape(frame.height, frame.width, 3)
    if frame.pixel_format is PixelFormat.BGR24:
        array = array[:, :, ::-1]
    # ascontiguousarray both drops the negative stride from the BGR swap and
    # detaches from the read-only buffer, so downstream OpenCV calls are safe.
    return np.ascontiguousarray(array)


def _load_eyegestures_class() -> type[Any]:
    """Import the optional GPL dependency only when this engine is selected."""

    try:
        from eyeGestures import EyeGestures_v2  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ContractValidationError(
            "The 'eyegestures' engine requires the optional EyeGestures dependency. "
            'Install it with: pip install -e ".[eyegestures]". Note that EyeGestures '
            "is GPL-3.0 licensed; see TASKS.md before distributing."
        ) from error
    return EyeGestures_v2  # type: ignore[no-any-return]


# NOTE ON A MEDIAPIPE CONFLICT THIS ENGINE CAUSES
#
# EyeGestures uses MediaPipe's legacy ``mp.solutions`` API. Importing it sets a
# process-wide resource root pointing at MediaPipe's own package, so its
# bundled graphs can resolve relative paths like
# ``mediapipe/modules/face_detection/...tflite``.
#
# From that moment the Tasks API -- which ``gazelink.vision`` uses -- resolves
# *absolute* model paths against that same root, producing nonsense such as
# ``site-packages/C:\\Users\\...\\face_landmarker.task``, and every
# FaceLandmarker construction fails. GAZELINK's own tracking then dies
# silently, reported only as a bare ReasonCode.ERROR.
#
# Resetting the root here is NOT a fix: it repairs our absolute path but
# breaks their relative ones, and their failure is swallowed by their own
# try/except, so it merely moves the outage somewhere harder to see.
#
# The conflict is resolved on our side instead: ``gazelink.vision`` loads its
# model from bytes (``model_asset_buffer``) rather than a path, which is
# immune to whatever resource root is installed. Nothing needs to be undone
# here, and EyeGestures keeps the root it requires.


class EyeGesturesGazePredictor:
    """Drive EyeGestures frame by frame behind the shared prediction port."""

    def __init__(
        self,
        *,
        screen_geometry: ScreenGeometry,
        gestures: EyeGesturesLike | None = None,
        context: str = DEFAULT_CONTEXT,
        calibration_points: int = DEFAULT_CALIBRATION_POINTS,
        calibration_seed: int = DEFAULT_CALIBRATION_SEED,
        require_tracked_observation: bool = True,
    ) -> None:
        if not isinstance(screen_geometry, ScreenGeometry):
            raise ContractValidationError("screen_geometry must be a ScreenGeometry")
        if isinstance(calibration_points, bool) or not isinstance(calibration_points, int):
            raise ContractValidationError("calibration_points must be an integer")
        if calibration_points < 1:
            raise ContractValidationError("calibration_points must be at least 1")

        self._geometry = screen_geometry
        self._context = context
        self._required_points = calibration_points
        self._completed_points = 0
        self._previous_target: tuple[float, float] | None = None
        self._current_target: GazePoint | None = None
        self._current_target_pixel: tuple[float, float] | None = None
        self._acceptance_radius_px = 0.0
        self._closed = False
        # Default True keeps our confidence gate in force for every existing
        # caller.  Only a measurement run opts out, and even then the gate's
        # verdict is preserved per sample by `gate_would_accept` rather than
        # discarded -- suppression becomes a label, not a silent filter.
        self._require_tracked_observation = bool(require_tracked_observation)
        self._calibration_frozen = False

        if gestures is None:
            gestures = _load_eyegestures_class()(calibration_radius=1000)
        self._gestures = gestures
        self._gestures.uploadCalibrationMap(
            build_calibration_map(seed=calibration_seed), context=self._context
        )
        self._gestures.setClassicalImpact(DEFAULT_CLASSICAL_IMPACT)
        self._gestures.setFixation(DEFAULT_FIXATION)

    # --- port surface -----------------------------------------------------

    @property
    def engine_name(self) -> str:
        return "eyegestures"

    @property
    def screen_geometry(self) -> ScreenGeometry:
        return self._geometry

    @property
    def is_calibrated(self) -> bool:
        return self._completed_points >= self._required_points

    def calibration_view(self) -> EyeGesturesCalibrationView:
        """What to draw right now: their target while calibrating, else idle."""

        active = not self.is_calibrated
        return EyeGesturesCalibrationView(
            active=active,
            target_normalized=self._current_target if active else None,
            acceptance_radius_px=self._acceptance_radius_px,
            completed_points=min(self._completed_points, self._required_points),
            total_points=self._required_points,
            target_pixel=self._current_target_pixel if active else None,
        )

    def predict(
        self,
        *,
        frame: FramePacket,
        observation: VisionObservation | None,
        now_monotonic_ms: float,
    ) -> GazeEstimationResult:
        """Feed one frame to the engine and return a point, or a rejection.

        ``observation`` may be ``None``, or may carry a tracking state we do
        not accept. The frame is still handed to the engine in that case --
        deliberately. The engine runs its own calibration from the frames it
        sees, and gating that on OUR confidence policy starves it: measured at
        18% frame acceptance it advanced one calibration point in twelve
        seconds, i.e. roughly five minutes for a full pass, which reads as a
        hung screen. Its own detector decides what it can use.

        The safety property is unchanged and is enforced below instead: no
        GazeSample is ever returned unless our policy accepted this frame AND
        the engine finished calibrating. Feeding a frame is not an output.
        """

        if self._closed:
            raise RuntimeError("EyeGesturesGazePredictor is closed")
        if observation is not None and not isinstance(observation, VisionObservation):
            raise ContractValidationError("observation must be VisionObservation or None")

        # A frame we cannot faithfully convert is an error, not a guess.
        rgb = frame_to_rgb_array(frame)
        if rgb is None:
            return GazeEstimationResult(None, (ReasonCode.ERROR,))

        calibrating = not self.is_calibrated and not self._calibration_frozen
        try:
            gaze_event, calibration_event = self._gestures.step(
                rgb,
                calibrating,
                self._geometry.width_px,
                self._geometry.height_px,
                context=self._context,
            )
        except Exception:  # noqa: BLE001 - see below; containment is the point
            # VERIFIED UPSTREAM DEFECT, not defensive paranoia: EyeGestures
            # 3.2.4 raises TypeError from face.py's ``_landmarks`` whenever
            # MediaPipe returns a result whose ``multi_face_landmarks`` is
            # None -- i.e. on any frame with no detectable face, which is the
            # normal state before the user sits down. Their own ``process()``
            # has its try/except commented out, so nothing upstream contains
            # it. A third-party crash on a routine condition must not take
            # down the gaze window, so it is reported as a lost frame exactly
            # like any other frame we could not resolve.
            return GazeEstimationResult(None, (ReasonCode.FACE_NOT_FOUND,))

        # Calibration progress is tracked from every frame the engine saw,
        # including frames our own policy would not accept.
        self._track_calibration_progress(calibration_event)

        # No face this frame: the library returns (None, None).
        if gaze_event is None:
            return GazeEstimationResult(None, (ReasonCode.FACE_NOT_FOUND,))

        # THE [0, 0] GUARD. Until their calibration has actually completed,
        # their predictor returns the top-left corner. Never emit a sample
        # from an unfitted model, however valid the coordinate looks.
        if not self.is_calibrated:
            return GazeEstimationResult(None, (ReasonCode.CALIBRATION_INVALID,))

        # OUR confidence and tracking policy normally has the final say on
        # emitting a point, so blink and tracking-loss handling stay ours in
        # both engines.  A measurement run may opt out (see the class
        # docstring): what the library predicted is then still emitted, and
        # the gate's verdict travels with the sample instead of erasing it.
        if self._require_tracked_observation:
            if observation is None:
                return GazeEstimationResult(None, (ReasonCode.LOW_CONFIDENCE,))
            if observation.tracking_state is not TrackingState.TRACKED:
                return GazeEstimationResult(
                    None, observation.reason_codes or (ReasonCode.LOW_CONFIDENCE,)
                )

        return self._sample_from(gaze_event, frame, observation, now_monotonic_ms)

    @property
    def is_frozen(self) -> bool:
        return self._calibration_frozen

    def freeze_calibration(self) -> None:
        """Stop the library learning, permanently, for this predictor.

        EyeGestures keeps training during ordinary use: while ``step()`` is
        given ``calibration=True`` it adds a sample and refits Ridge on EVERY
        frame.  ``calibration_radius`` is not a lever -- the radius test is
        ``or``-ed with ``filled_points < 200``, and ``filled_points`` is capped
        at 20, so that second condition is always true and the radius never
        matters.  Passing ``calibration=False`` is the only real switch: it
        skips ``add()``, skips ``movePoint()``, and leaves only ``post_fit()``,
        which is a no-op in this version.

        Freezing is irreversible on purpose.  A measurement that could quietly
        resume training half way through would produce a number nobody can
        interpret afterwards.
        """

        self._calibration_frozen = True

    def calibration_fingerprint(self) -> tuple[float, ...] | None:
        """A snapshot of the library's fitted model, or ``None`` if unreadable.

        Compared before and after a measurement, this is DIRECT evidence about
        whether learning actually stopped -- unlike a fixed wait, which proves
        nothing.  ``Calibrator.add`` starts a fresh thread per sample and there
        is no public join, so a fit launched just before the freeze can still
        land afterwards and change these coefficients.

        Reaches into the library's internals deliberately, and read-only.  If a
        future version renames them this returns ``None``, which a caller must
        report as "freeze not verified" rather than as "freeze confirmed".
        """

        calibrator = self._calibrator()
        if calibrator is None:
            return None
        values: list[float] = []
        for axis in ("reg_x", "reg_y"):
            model = getattr(calibrator, axis, None)
            if model is None:
                return None
            for attribute in ("coef_", "intercept_"):
                raw = getattr(model, attribute, None)
                if raw is None:
                    return None
                try:
                    values.extend(float(item) for item in np.atleast_1d(raw).ravel())
                except (TypeError, ValueError):
                    return None
        return tuple(values)

    def pending_fit_threads(self) -> int | None:
        """How many of the library's training threads are still running.

        ``None`` means the question could not be answered, which is not the
        same as zero and must never be reported as one.
        """

        calibrator = self._calibrator()
        if calibrator is None:
            return None
        coroutines = getattr(calibrator, "fit_coroutines", None)
        if not isinstance(coroutines, list):
            return None
        try:
            return sum(1 for thread in coroutines if thread.is_alive())
        except AttributeError:
            return None

    def _calibrator(self) -> Any | None:
        """The library's per-context calibrator, if it exposes one."""

        calibrators = getattr(self._gestures, "clb", None)
        if not isinstance(calibrators, dict):
            return None
        return calibrators.get(self._context)

    def close(self) -> None:
        """EyeGestures exposes no release hook; make the adapter inert."""

        self._closed = True

    # --- internals --------------------------------------------------------

    def _track_calibration_progress(self, calibration_event: Any) -> None:
        """Count completed calibration points by watching their target move.

        The library advances its own target once it accepts a point, so a
        change of ``Cevent.point`` is the completion signal.  Reading that
        public event keeps the adapter off private attributes such as
        ``clb[context].fitted``.
        """

        point = getattr(calibration_event, "point", None)
        if point is None:
            return
        try:
            target = (float(point[0]), float(point[1]))
        except (TypeError, ValueError, IndexError):
            return

        radius = getattr(calibration_event, "acceptance_radius", None)
        if isinstance(radius, (int, float)) and not isinstance(radius, bool):
            self._acceptance_radius_px = float(radius)

        self._current_target_pixel = target
        self._current_target = GazePoint(
            target[0] / max(1, self._geometry.width_px),
            target[1] / max(1, self._geometry.height_px),
        )
        if self._previous_target is None:
            # First sighting establishes the target; it completes nothing.
            self._previous_target = target
            return
        if target != self._previous_target:
            self._completed_points += 1
            self._previous_target = target

    def _sample_from(
        self,
        gaze_event: Any,
        frame: FramePacket,
        observation: VisionObservation | None,
        now_monotonic_ms: float,
    ) -> GazeEstimationResult:
        point = getattr(gaze_event, "point", None)
        if point is None:
            return GazeEstimationResult(None, (ReasonCode.ERROR,))
        try:
            pixel_x = float(point[0])
            pixel_y = float(point[1])
        except (TypeError, ValueError, IndexError):
            return GazeEstimationResult(None, (ReasonCode.ERROR,))
        if not (np.isfinite(pixel_x) and np.isfinite(pixel_y)):
            return GazeEstimationResult(None, (ReasonCode.ERROR,))

        raw = GazePoint(
            pixel_x / max(1, self._geometry.width_px),
            pixel_y / max(1, self._geometry.height_px),
        )
        reasons: list[ReasonCode] = []
        if not 0.0 <= raw.x <= 1.0 or not 0.0 <= raw.y <= 1.0:
            # A prediction off the edge of the screen is a real prediction and
            # must stay measurable.  These codes describe `screen_position`,
            # which IS clamped for drawing; `raw_normalized` below is not.
            reasons.extend((ReasonCode.OUT_OF_RANGE, ReasonCode.CLAMPED_TO_SCREEN))
        if not gate_would_accept(observation):
            # Only reachable with the gate opted out. Record why our policy
            # would have refused this frame, so a consumer can reproduce that
            # decision instead of having the sample silently withheld.
            reasons.extend(
                observation.reason_codes
                if observation is not None
                else (ReasonCode.FACE_NOT_FOUND,)
            )

        sample = GazeSample(
            source_frame_id=frame.frame_id,
            sampled_at_monotonic_ms=now_monotonic_ms,
            # This engine performs no local correction and no temporal
            # filtering of its own; consumers apply filtering downstream
            # exactly as they do for the native engine.
            raw_normalized=raw,
            corrected_normalized=raw,
            filtered_normalized=raw,
            screen_position=normalized_to_pixel(raw, self._geometry),
            screen_id=self._geometry.screen_id,
            # Their gaze event exposes no calibrated confidence, so the honest
            # value is the one our own vision stage measured for this frame.
            # With no observation at all there is no measured confidence, and
            # 0.0 says exactly that rather than inventing one.
            confidence=0.0 if observation is None else observation.overall_confidence,
            # M2 authorizes no engine for control, and an unvalidated external
            # engine least of all.
            valid_for_control=False,
            reason_codes=tuple(dict.fromkeys(reasons)),
        )
        return GazeEstimationResult(sample, tuple(dict.fromkeys(reasons)))
