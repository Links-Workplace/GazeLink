"""Qt-free state adapter for the guided M2 9-point calibration UI.

The controller is deliberately the only code allowed to turn a live
``VisionObservation`` into a ``CalibrationSample`` and a call to
``CalibrationSession.record_sample``. It holds no frame bytes and no
OS-input adapter -- only the scalar eye/iris/head-pose values already on
``VisionObservation``.

M2-02 Stage B: ``ingest`` runs a real ingest-time quality policy, not just
the Stage-A tracking/confidence/features gates. In order, a sample can be
rejected for: not being a fresh camera frame, tracking not being ready, low
overall confidence, missing feature data, being stale (see
:func:`frame_monotonic_ms`), an eye that is not genuinely visible even
though its ``EyeFeatures`` object exists, an absolute head-pose excursion
beyond :data:`CALIBRATION_HEAD_POSE_LIMITS`. Quality-passing observations are
held provisionally until a complete, contiguous window is available. Only the
best stable subwindow is then committed as accepted; unstable candidates are
stored as ``target_outlier`` and the same target is retried automatically.
This avoids letting the first few noisy observations create a poisoned
baseline. Every threshold is an UNVALIDATED PLACEHOLDER pending a real
accuracy-driven tuning pass. There is deliberately no session-relative
head-pose-position gate: head pose is a feature the M2-03 gaze model is
expected to learn from. The window gate checks short-term head-pose *motion*,
not whether the user's settled pose differs between screen targets.

The default ``calibration_confidence`` gate below is pinned to
``ConfidenceThresholds.display`` (0.50), not ``.calibration`` (0.70). This is
not a tuning choice: ``vision.FaceLandmarkerAdapter`` currently reports a
fixed ``STRUCTURAL_CONFIDENCE`` of exactly 0.50 for every tracked observation
-- there is no real confidence model yet. Gating on ``.calibration`` (0.70)
would reject every sample unconditionally, regardless of how well the face
and eyes are actually detected, since 0.50 can never satisfy ``>= 0.70``. Do
not raise this default back to ``.calibration`` until ``vision.py`` can
produce a confidence value above 0.50, or every calibration session will
silently fail closed with a "quality too low" message that has nothing to do
with actual tracking quality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum
from time import perf_counter_ns

from gazelink.calibration import (
    NUM_CALIBRATION_TARGETS,
    CalibrationReasonCode,
    CalibrationSample,
    CalibrationSession,
    CalibrationSessionResult,
    CalibrationSessionState,
    CalibrationTarget,
)
from gazelink.confidence import Clock, HeadPoseLimits
from gazelink.config import ConfidenceThresholds
from gazelink.domain import (
    ContractValidationError,
    TrackingState,
    VisionObservation,
)
from gazelink.gaze_features import from_calibration_sample, select_stable_feature_window


def frame_monotonic_ms() -> float:
    """Default injectable clock for calibration sample-age gating.

    Returns ``perf_counter_ns() / 1_000_000`` -- the SAME monotonic time base
    ``camera.py`` stamps onto every captured frame
    (``FramePacket.captured_at_monotonic_ms``), which flows unchanged into
    ``VisionObservation.observed_at_monotonic_ms`` (see ``vision.py``).
    ``pipeline.py`` and ``runtime.py`` each keep their own private
    ``_monotonic_ms`` copy of this exact same convention, for the same
    reason: sample-age arithmetic is only meaningful when "now" and "when the
    frame was captured" come from the same clock.

    ``gazelink.confidence.monotonic_ms`` (``time.monotonic() * 1000.0``) is
    deliberately NOT the default here, even though it also produces
    "monotonic milliseconds". On Windows, ``time.monotonic()`` and
    ``time.perf_counter()`` are backed by different clocks with different,
    unrelated epochs. Subtracting a ``time.monotonic()``-based "now" from a
    ``perf_counter_ns()``-based frame timestamp does not measure an elapsed
    duration at all -- the result depends on an accident of process/clock
    start order, not on how old the frame actually is. Defaulting to that
    clock could silently mark every real, fresh sample as stale, breaking
    calibration for a real user, while every deterministic test here (which
    only ever injects a fake clock) kept passing.
    """

    return perf_counter_ns() / 1_000_000


# Deliberately more generous than confidence.HeadPoseLimits()'s live-control
# defaults (yaw 30 / pitch 25 / roll 25 degrees). Real observed calibration
# sessions showed legitimate yaw in the 36-50 degree range while the user was
# validly looking at an edge/corner target, so a 30 degree cutoff would
# reject good samples outright. Over-rejecting is the worse failure mode
# here: a user who can only control the computer with their eyes cannot
# "just try again with a mouse" if calibration itself becomes frustrating or
# impossible to complete. These three limits are UNVALIDATED PLACEHOLDERS pending a
# real accuracy-driven tuning pass, the same documented-placeholder
# treatment already given to DEFAULT_TARGET_EDGE_INSET (calibration.py) and
# to HeadPoseLimits's own defaults (confidence.py).
CALIBRATION_HEAD_POSE_LIMITS = HeadPoseLimits(
    max_abs_yaw_deg=55.0, max_abs_pitch_deg=40.0, max_abs_roll_deg=40.0
)


def _positive_finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ContractValidationError(f"{field_name} must be > 0")
    return result


def _non_negative_finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ContractValidationError(f"{field_name} must be >= 0")
    return result


def _unit_ratio(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ContractValidationError(f"{field_name} must be within [0.0, 1.0]")
    return result


@dataclass(frozen=True, slots=True)
class CalibrationQualitySettings:
    """Ingest-time sample-quality gates for M2-02 Stage B.

    Every default here is an UNVALIDATED PLACEHOLDER pending a real
    accuracy-driven tuning pass against real calibration sessions -- see
    :data:`CALIBRATION_HEAD_POSE_LIMITS` for why the head-pose limits in
    particular are deliberately more generous than live control's.

    ``max_sample_age_ms``
        Units: milliseconds. Maximum age of an observation, measured as
        ``clock_ms() - VisionObservation.observed_at_monotonic_ms`` at
        ingest time, before it is rejected as stale.
    ``head_pose_limits``
        Units: degrees, absolute value per axis (yaw/pitch/roll). A sample
        whose head pose exceeds any one of these three is rejected,
        regardless of the other two.
    ``min_eye_openness``
        Units: ratio in ``[0.0, 1.0]`` -- the same eyelid-gap-over-eye-width
        ratio ``EyeFeatures.openness`` reports. Below this, an eye is
        treated as not genuinely visible (e.g. mid-blink) even though its
        ``EyeFeatures`` object is not ``None``.
    """

    max_sample_age_ms: float = 500.0
    head_pose_limits: HeadPoseLimits = CALIBRATION_HEAD_POSE_LIMITS
    min_eye_openness: float = 0.15

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_sample_age_ms",
            _positive_finite_number(self.max_sample_age_ms, "max_sample_age_ms"),
        )
        if not isinstance(self.head_pose_limits, HeadPoseLimits):
            raise ContractValidationError("head_pose_limits must be a HeadPoseLimits")
        object.__setattr__(
            self, "min_eye_openness", _unit_ratio(self.min_eye_openness, "min_eye_openness")
        )


# Module-level singleton so `CalibrationUiController.__init__`'s default
# doesn't call `CalibrationQualitySettings()` directly in the argument-default
# list (ruff B008); the frozen instance is safe to share across every
# controller that doesn't override it.
_DEFAULT_QUALITY = CalibrationQualitySettings()


class CalibrationCollectionPhase(StrEnum):
    STABILIZING = "STABILIZING"
    COLLECTING = "COLLECTING"


@dataclass(frozen=True, slots=True)
class CalibrationTimingSettings:
    """Temporal and contiguous-window stability policy.

    Eye spread is measured in normalized iris feature space. Head-pose spread
    is the three-axis Euclidean radius in degrees. Both thresholds remain
    tuning placeholders until more real-camera sessions are measured.
    """

    stabilization_ms: float = 750.0
    capture_window_ms: float = 500.0
    min_sample_interval_ms: float = 80.0
    max_stable_eye_p95: float = 0.08
    max_stable_head_pose_p95_deg: float = 5.0

    def __post_init__(self) -> None:
        for name in ("stabilization_ms", "capture_window_ms", "min_sample_interval_ms"):
            object.__setattr__(self, name, _non_negative_finite_number(getattr(self, name), name))
        for name in ("max_stable_eye_p95", "max_stable_head_pose_p95_deg"):
            object.__setattr__(self, name, _positive_finite_number(getattr(self, name), name))


_DEFAULT_TIMING = CalibrationTimingSettings()


@dataclass(frozen=True, slots=True)
class CalibrationUiView:
    """Display-ready calibration progress, containing no biometric data."""

    target: CalibrationTarget | None
    target_number: int
    total_targets: int
    accepted_samples: int
    required_samples: int
    feedback: str
    complete: bool
    cancelled: bool
    sample_accepted: bool = False
    sample_evaluated: bool = False
    phase: CalibrationCollectionPhase = CalibrationCollectionPhase.STABILIZING
    phase_remaining_ms: float = 0.0
    """Whether this ingest call committed a complete stable window.

    It is false while an otherwise-good observation is merely provisional.
    ``sample_evaluated`` becomes true only when the full window is accepted or
    rejected, or when an immediate quality gate rejects one observation.
    """


class CalibrationUiController:
    """Safely bridge one observation at a time into a calibration session."""

    def __init__(
        self,
        session: CalibrationSession,
        *,
        calibration_confidence: float = ConfidenceThresholds().display,
        quality: CalibrationQualitySettings = _DEFAULT_QUALITY,
        timing: CalibrationTimingSettings = _DEFAULT_TIMING,
        clock_ms: Clock = frame_monotonic_ms,
    ) -> None:
        if not isinstance(session, CalibrationSession):
            raise ContractValidationError("session must be a CalibrationSession")
        if not isinstance(calibration_confidence, (int, float)) or isinstance(
            calibration_confidence, bool
        ):
            raise ContractValidationError("calibration_confidence must be a number")
        if not 0.0 <= calibration_confidence <= 1.0:
            raise ContractValidationError("calibration_confidence must be within [0.0, 1.0]")
        if not isinstance(quality, CalibrationQualitySettings):
            raise ContractValidationError("quality must be a CalibrationQualitySettings")
        if not isinstance(timing, CalibrationTimingSettings):
            raise ContractValidationError("timing must be a CalibrationTimingSettings")
        self._session = session
        self._calibration_confidence = float(calibration_confidence)
        self._quality = quality
        self._timing = timing
        self._clock_ms = clock_ms
        self._last_frame_id: int | None = None
        self._phase = CalibrationCollectionPhase.STABILIZING
        self._phase_started_ms = self._clock_ms()
        self._last_candidate_ms: float | None = None
        self._candidate_samples: list[CalibrationSample] = []
        self._feedback = "Move your gaze to the highlighted target and hold still."

    @property
    def session(self) -> CalibrationSession:
        """The owned session, available for the M2-T04 consumer once complete."""

        return self._session

    def ingest(self, observation: VisionObservation) -> CalibrationUiView:
        """Accept at most one high-quality live observation and auto-progress.

        Immediate quality failures are stored as rejected samples. Passing
        observations remain provisional until a complete contiguous window
        passes the eye/head stability limits; only that stable window is
        committed as accepted. A repeated frame is rejected before it is
        turned into a sample, preventing timer redraws from quietly turning
        one observation into several.
        """

        if not isinstance(observation, VisionObservation):
            raise ContractValidationError("observation must be a VisionObservation")
        if self._session.state is not CalibrationSessionState.COLLECTING:
            return self.view()
        if observation.frame_id == self._last_frame_id:
            self._feedback = "Waiting for a new camera frame."
            return self.view()
        self._last_frame_id = observation.frame_id
        now_ms = self._clock_ms()
        if self._phase is CalibrationCollectionPhase.STABILIZING:
            elapsed_ms = now_ms - self._phase_started_ms
            if elapsed_ms < self._timing.stabilization_ms:
                remaining = max(0.0, self._timing.stabilization_ms - elapsed_ms)
                self._feedback = f"Hold your gaze still; capture starts in {remaining / 1000:.1f}s."
                return self._view(sample_accepted=False)
            self._phase = CalibrationCollectionPhase.COLLECTING
            self._phase_started_ms = now_ms
            self._last_candidate_ms = None
        if (
            self._last_candidate_ms is not None
            and now_ms - self._last_candidate_ms < self._timing.min_sample_interval_ms
        ):
            self._feedback = "Collecting across the capture window; keep looking at the target."
            return self._view(sample_accepted=False)
        if observation.tracking_state is not TrackingState.TRACKED:
            self._discard_candidate_window()
            self._record(
                observation, accepted=False, reason=CalibrationReasonCode.TRACKING_NOT_READY
            )
            self._reset_target_timing(now_ms)
            self._feedback = "Tracking is not ready; keep your face visible."
            return self._view(sample_accepted=False, sample_evaluated=True)
        if observation.overall_confidence < self._calibration_confidence:
            self._discard_candidate_window()
            self._record(observation, accepted=False, reason=CalibrationReasonCode.LOW_CONFIDENCE)
            self._reset_target_timing(now_ms)
            self._feedback = "Tracking quality is too low; hold still and keep both eyes visible."
            return self._view(sample_accepted=False, sample_evaluated=True)

        # Sample-age gate. `age_ms` compares two stamps that must share the
        # same monotonic time base -- see `frame_monotonic_ms`'s docstring
        # for why `clock_ms` defaults there and not to
        # `confidence.monotonic_ms`. Reject only on `age_ms > max`, never
        # `>=` or on a negative age: a negative age would mean the
        # observation is stamped "in the future", which cannot happen with a
        # shared time base, so it is treated as an impossible/no-op condition
        # rather than a rejection. This is the opposite asymmetry from
        # `ConfidencePolicy.evaluate`'s `fresh = 0.0 <= age_ms <= max`: live
        # control fails CLOSED on that same impossible condition because an
        # incorrect cursor action is dangerous, while calibration fails OPEN
        # here because over-rejecting a calibration sample is the worse
        # failure mode for a user who can only drive the computer with their
        # eyes.
        age_ms = self._clock_ms() - observation.observed_at_monotonic_ms
        if age_ms > self._quality.max_sample_age_ms:
            self._discard_candidate_window()
            self._record(observation, accepted=False, reason=CalibrationReasonCode.STALE_SAMPLE)
            self._reset_target_timing(now_ms)
            self._feedback = (
                "That frame arrived too late; hold still and keep looking at the target."
            )
            return self._view(sample_accepted=False, sample_evaluated=True)

        if (
            observation.left_eye is None
            or observation.right_eye is None
            or observation.head_pose is None
        ):
            # `VisionRuntime` can report `tracking_state=TRACKED` with every
            # geometry field blanked to `None` -- its loss-debounce window
            # deliberately keeps reporting TRACKED for a short grace period to
            # avoid UI flicker, while withholding geometry that briefly wasn't
            # accepted by the confidence policy (e.g. a momentary head-pose
            # excursion). `overall_confidence` is untouched by that blanking,
            # so it can still read >= the gate above even though there is
            # nothing usable to store. Without this check, such a tick was
            # being recorded as `accepted=True` with every feature field
            # `None` -- a real, observed bug: a live session's shared log
            # showed 4 of 5 "accepted" samples for one target with no iris,
            # openness, or head-pose data at all.
            self._discard_candidate_window()
            self._record(
                observation, accepted=False, reason=CalibrationReasonCode.FEATURES_UNAVAILABLE
            )
            self._reset_target_timing(now_ms)
            self._feedback = "Tracking is not ready; keep your face visible."
            return self._view(sample_accepted=False, sample_evaluated=True)

        # Real eye-visibility gate. The check above already proved left_eye,
        # right_eye, and head_pose are non-None, so no further `is None`
        # checks are needed on those three objects here. For each eye
        # independently: reject if its iris was not actually located, or if
        # its eyelid gap is below `min_eye_openness` (e.g. mid-blink).
        #
        # `EyeFeatures.confidence` is deliberately EXCLUDED from this gate.
        # `features.py` sets `EyeFeatures.confidence = landmarks.
        # detector_confidence`, and `vision.py`'s real adapter always passes
        # `detector_confidence=STRUCTURAL_CONFIDENCE` -- the same fixed 0.50
        # constant used for `overall_confidence` above, not a real per-eye
        # signal yet. Gating on it here would repeat the exact bug
        # `test_the_real_adapters_fixed_confidence_is_sufficient_to_calibrate`
        # exists to pin: thresholding a constant that can never satisfy
        # anything above it, which would silently reject every real sample.
        # `openness` (computed from real eyelid-gap geometry in
        # `features.py`) and the presence of `iris_in_eye` are the two
        # signals that genuinely vary, so they are the only two used here.
        left_eye = observation.left_eye
        right_eye = observation.right_eye
        left_iris = left_eye.iris_in_eye
        right_iris = right_eye.iris_in_eye
        if left_iris is None or left_eye.openness < self._quality.min_eye_openness:
            self._discard_candidate_window()
            self._record(observation, accepted=False, reason=CalibrationReasonCode.EYE_NOT_VISIBLE)
            self._reset_target_timing(now_ms)
            self._feedback = "Keep both eyes open and visible."
            return self._view(sample_accepted=False, sample_evaluated=True)
        if right_iris is None or right_eye.openness < self._quality.min_eye_openness:
            self._discard_candidate_window()
            self._record(observation, accepted=False, reason=CalibrationReasonCode.EYE_NOT_VISIBLE)
            self._reset_target_timing(now_ms)
            self._feedback = "Keep both eyes open and visible."
            return self._view(sample_accepted=False, sample_evaluated=True)
        # Absolute head-pose gate, using calibration's own deliberately
        # generous limits (`CALIBRATION_HEAD_POSE_LIMITS`), not the
        # live-control defaults. There is no session-relative drift gate on
        # purpose: head pose is a feature the M2-03 gaze model is expected to
        # learn from, not noise to filter out -- see the module docstring.
        pose = observation.head_pose
        limits = self._quality.head_pose_limits
        if (
            abs(pose.yaw_deg) > limits.max_abs_yaw_deg
            or abs(pose.pitch_deg) > limits.max_abs_pitch_deg
            or abs(pose.roll_deg) > limits.max_abs_roll_deg
        ):
            self._discard_candidate_window()
            self._record(
                observation, accepted=False, reason=CalibrationReasonCode.HEAD_POSE_OUT_OF_RANGE
            )
            self._reset_target_timing(now_ms)
            self._feedback = "Face the screen more directly."
            return self._view(sample_accepted=False, sample_evaluated=True)

        candidate = self._build_sample(
            observation, accepted=True, reason=CalibrationReasonCode.ACCEPTED
        )
        self._candidate_samples.append(candidate)
        self._last_candidate_ms = now_ms
        capture_elapsed_ms = now_ms - self._phase_started_ms
        if (
            len(self._candidate_samples) < self._session.min_samples_per_target
            or capture_elapsed_ms < self._timing.capture_window_ms
        ):
            remaining = max(0.0, self._timing.capture_window_ms - capture_elapsed_ms)
            self._feedback = (
                f"Candidate samples captured; hold still for {remaining / 1000:.1f}s more."
            )
            return self._view(sample_accepted=False)

        vectors = [from_calibration_sample(sample) for sample in self._candidate_samples]
        if any(vector is None for vector in vectors):
            raise RuntimeError("validated calibration candidate unexpectedly has no feature vector")
        selection = select_stable_feature_window(
            [vector for vector in vectors if vector is not None],
            window_size=self._session.min_samples_per_target,
            max_eye_p95=self._timing.max_stable_eye_p95,
            max_head_pose_p95_deg=self._timing.max_stable_head_pose_p95_deg,
        )
        if selection is None:
            self._commit_candidate_window(selection=None)
            self._reset_target_timing(now_ms)
            self._feedback = "The gaze window was too noisy. Hold the same target and retry."
            return self._view(sample_accepted=False, sample_evaluated=True)

        start, end, _stability = selection
        self._commit_candidate_window(selection=(start, end))
        self._session.advance()
        completed = self._session.state.value == CalibrationSessionState.COMPLETE.value
        self._feedback = (
            "Calibration complete."
            if completed
            else "Target complete. Move your gaze to the next highlighted target."
        )
        if not completed:
            self._reset_target_timing(now_ms)
        return self._view(sample_accepted=True, sample_evaluated=True)

    def _reset_target_timing(self, now_ms: float | None = None) -> None:
        self._phase = CalibrationCollectionPhase.STABILIZING
        self._phase_started_ms = self._clock_ms() if now_ms is None else now_ms
        self._last_candidate_ms = None
        self._candidate_samples.clear()

    def _commit_candidate_window(self, *, selection: tuple[int, int] | None) -> None:
        """Persist a provisional window with only the selected slice accepted."""

        start, end = selection if selection is not None else (-1, -1)
        for index, sample in enumerate(self._candidate_samples):
            if start <= index < end:
                self._session.record_sample(sample)
            else:
                self._session.record_sample(
                    replace(
                        sample,
                        accepted=False,
                        reason=CalibrationReasonCode.TARGET_OUTLIER,
                    )
                )
        self._candidate_samples.clear()

    def _discard_candidate_window(self) -> None:
        """Persist an interrupted provisional window as rejected, if present."""

        if self._candidate_samples:
            self._commit_candidate_window(selection=None)

    def _record(
        self, observation: VisionObservation, *, accepted: bool, reason: CalibrationReasonCode
    ) -> None:
        """Build a :class:`CalibrationSample` from ``observation`` and store it.

        Reads only scalar/coordinate fields already present on
        ``VisionObservation`` -- never the frame image, and never anything
        this module didn't already have in hand.
        """

        self._session.record_sample(
            self._build_sample(observation, accepted=accepted, reason=reason)
        )

    def _build_sample(
        self, observation: VisionObservation, *, accepted: bool, reason: CalibrationReasonCode
    ) -> CalibrationSample:
        left_eye = observation.left_eye
        right_eye = observation.right_eye
        pose = observation.head_pose
        return CalibrationSample(
            source_frame_id=observation.frame_id,
            observed_at_monotonic_ms=observation.observed_at_monotonic_ms,
            target_index=self._session.current_target.index,
            left_iris_in_eye=None if left_eye is None else left_eye.iris_in_eye,
            right_iris_in_eye=None if right_eye is None else right_eye.iris_in_eye,
            left_openness=None if left_eye is None else left_eye.openness,
            right_openness=None if right_eye is None else right_eye.openness,
            left_iris_in_lids_y=None if left_eye is None else left_eye.iris_in_lids_y,
            right_iris_in_lids_y=None if right_eye is None else right_eye.iris_in_lids_y,
            head_yaw_deg=None if pose is None else pose.yaw_deg,
            head_pitch_deg=None if pose is None else pose.pitch_deg,
            head_roll_deg=None if pose is None else pose.roll_deg,
            confidence=observation.overall_confidence,
            accepted=accepted,
            reason=reason,
        )

    def retry(self) -> CalibrationUiView:
        """Discard only the current target's samples and show it again."""

        self._session.retry_current_target()
        self._last_frame_id = None
        self._reset_target_timing()
        self._feedback = "Current target reset. Move your gaze to it and hold still."
        return self.view()

    def cancel(self) -> CalibrationUiView:
        """Cancel without making a partial calibration result available."""

        self._candidate_samples.clear()
        self._session.cancel()
        self._feedback = "Calibration cancelled."
        return self.view()

    def restart(self) -> CalibrationUiView:
        """Discard every target and restart at the configured first target."""

        self._session.restart()
        self._last_frame_id = None
        self._reset_target_timing()
        self._feedback = "Calibration restarted. Move your gaze to the target and hold still."
        return self.view()

    def result(self) -> CalibrationSessionResult:
        """Export only a genuinely complete result; delegated to the session gate."""

        return self._session.result()

    def view(self) -> CalibrationUiView:
        """Return a privacy-safe snapshot for any display implementation.

        ``sample_accepted`` is always ``False`` here, since no observation was
        evaluated by this call -- only ``ingest`` sets it meaningfully.
        """

        return self._view(sample_accepted=False)

    def _view(self, *, sample_accepted: bool, sample_evaluated: bool = False) -> CalibrationUiView:
        state = self._session.state
        if state is CalibrationSessionState.COLLECTING:
            return CalibrationUiView(
                target=self._session.current_target,
                target_number=self._session.current_target_number,
                total_targets=NUM_CALIBRATION_TARGETS,
                accepted_samples=len(self._candidate_samples),
                required_samples=self._session.min_samples_per_target,
                feedback=self._feedback,
                complete=False,
                cancelled=False,
                sample_accepted=sample_accepted,
                sample_evaluated=sample_evaluated,
                phase=self._phase,
                phase_remaining_ms=self._phase_remaining_ms(),
            )
        if state is CalibrationSessionState.COMPLETE:
            # `result()` is read-only and safe to call on every refresh: a
            # completed session's counts never change, so this cannot drift
            # from what `record_sample`/`advance` actually collected -- unlike
            # hardcoding 0 here, which previously showed "complete" together
            # with a sample count that looked like nothing was ever captured.
            totals = self._session.result().sample_counts
            return CalibrationUiView(
                target=None,
                target_number=NUM_CALIBRATION_TARGETS,
                total_targets=NUM_CALIBRATION_TARGETS,
                accepted_samples=sum(totals),
                required_samples=NUM_CALIBRATION_TARGETS * self._session.min_samples_per_target,
                feedback=self._feedback,
                complete=True,
                cancelled=False,
                sample_accepted=sample_accepted,
                sample_evaluated=sample_evaluated,
                phase=self._phase,
                phase_remaining_ms=0.0,
            )
        return CalibrationUiView(
            target=None,
            target_number=0,
            total_targets=NUM_CALIBRATION_TARGETS,
            accepted_samples=0,
            required_samples=self._session.min_samples_per_target,
            feedback=self._feedback,
            complete=False,
            cancelled=state is CalibrationSessionState.CANCELLED,
            sample_accepted=sample_accepted,
            sample_evaluated=sample_evaluated,
            phase=self._phase,
            phase_remaining_ms=0.0,
        )

    def _phase_remaining_ms(self) -> float:
        now_ms = self._clock_ms()
        if self._phase is CalibrationCollectionPhase.STABILIZING:
            return max(0.0, self._timing.stabilization_ms - (now_ms - self._phase_started_ms))
        return max(0.0, self._timing.capture_window_ms - (now_ms - self._phase_started_ms))
