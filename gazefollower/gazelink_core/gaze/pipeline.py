"""Per-frame live pipeline: gate, predict, filter, gestures (ARCH-01 stage E).

Moved from ``gf_live.LiveRunner`` with its behaviour unchanged, now fed
:class:`FrameObservation` instead of the library's objects. It draws nothing
and emits no OS input. Runs on the source's camera thread; the loop reads
``state`` and drains events under the lock.
"""

from __future__ import annotations

import collections
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from gazelink_core.domain import common as C
from gazelink_core.domain.observation import FrameObservation
from gazelink_core.gaze import gaze_filter as GF
from gazelink_core.gaze import prediction as PRED
from gazelink_core.gaze import sample_gate as GATE
from gazelink_core.interaction import gesture as GEST


@dataclass
class LiveState:
    """What the display thread reads. Replaced as a whole, never mutated."""

    point: tuple[float, float] | None = None
    raw_model: tuple[float, float] | None = None
    unfiltered: tuple[float, float] | None = None
    tracking: bool = False
    # Whether a FACE is in the frame, which is not the same question as
    # whether there is a usable gaze point. Closing the eyes ends the point
    # and not the face, and a caller that cannot tell the two apart treats
    # every deliberate close as a tracking failure.
    face_present: bool = False
    # The raw per-eye openness the gesture layer is gated on. Exposed so a
    # screen can SHOW it: measured on round34/35/36, this signal never once
    # fell to the blink threshold across forty seconds, which cannot be true
    # of a person's eyelids and means every gesture silently did nothing.
    openness: tuple[float, float] | None = None
    # Each eye as a fraction of its own recent baseline, which is what the
    # gesture layer actually judges. Shown on screen so a gesture that the
    # signal could not see is visible as such.
    openness_ratio: tuple[float, float] | None = None
    # Both eyes open enough for the gaze estimate to be worth moving a pointer
    # with. False through the whole of a wink, including the half-closed part
    # at each end where the blink gate still passes and the prediction is
    # already made from an eye behind its own lid.
    eyes_steady: bool = False
    # Chin-up head pitch (head6 "pitch_a", nose offset over inter-ocular
    # distance; larger = chin higher). Recorded next to the ratios because
    # eye openness is polygon AREA in px^2 -- a PROJECTED area -- so lifting
    # the chin shrinks it with the eye wide open, and the gate's baseline
    # adapts at 0.02 a frame and only while the ratio is already above 0.55.
    # If a pitch change pushes both eyes under that, they read as shut, the
    # baseline stops recovering, and the asymmetry rule cannot pass. This is
    # the number that says whether that is what happens.
    head_pitch: float | None = None
    # The whole head6 vector for this frame (gf_head_features.HEAD6_NAMES),
    # or None without a face. Reported only, never used for prediction here:
    # it is what lets a live session be compared with the recordings, which
    # store the same vector from the same builder.
    head: tuple[float, ...] | None = None
    updated_s: float | None = None
    frames: int = 0
    fps: float | None = None


# Callback timings kept between drains (~2 minutes at 32 fps).
TIMING_BUFFER = 4096


class FramePipeline:
    """Predict and filter every frame, and keep nothing.

    Deliberately not a ProtocolRunner: there is no target, no gate, no
    accepted-row accounting and no RecordingBuilder here.  Reusing that class
    with a fake one-target protocol would keep a storage path alive in a mode
    whose whole promise is that it stores nothing.
    """

    policy = GATE.LIVE

    def __init__(
        self,
        model: Any,
        model_y: Any,
        rig: C.RigGeometry,
        settings: GF.FilterSettings | None,
        *,
        clock: Callable[[], float] = time.monotonic,
        head_builder: Any = None,
        gesture: GEST.GestureConfig | None = None,
        wink: GEST.WinkConfig | None = None,
        gate: GEST.OpennessGateConfig | None = None,
    ) -> None:
        self.model = model
        self.model_y = model_y
        self.rig = rig
        self.clock = clock
        self.head_builder = head_builder
        self.filter = None if settings is None else GF.GazePointFilter(settings)
        # Driven here, on the camera thread, so every frame is seen exactly
        # once. Driving it from the display loop instead would resample the
        # same frame many times over -- the display redraws far faster than
        # the camera delivers -- and a hold would appear to last longer than
        # it did.
        self.detector = GEST.EyeCloseDetector(gesture)
        self.wink_detector = GEST.RightWinkDetector(wink)
        # One gate per eye, for the GESTURE path only. `valid` below keeps the
        # absolute threshold, because it decides what counts as a gaze sample
        # and every measurement in this project was made against it.
        self.left_gate = GEST.OpennessGate(gate)
        self.right_gate = GEST.OpennessGate(gate)
        self.gesture_events: list[tuple[float, GEST.Event]] = []
        self.wink_events: list[tuple[float, tuple[float, float] | None]] = []
        # Set by the display thread, applied by the CAMERA thread. The
        # detector's state belongs to the camera thread -- it is read and
        # written there every frame -- so cancelling it from here directly is
        # a race, and one that would show up as a wink surviving a mode change
        # occasionally rather than reliably. A flag crossing under the lock is
        # deterministic, and deterministic is what a test can pin down.
        self._cancel_wink_requested = False
        # Bumped by every cancel. A frame carries the generation it STARTED
        # with, and a wink is only published if that is still current: a
        # cancel arriving mid-frame -- after the flag was read and before the
        # wink was appended -- would otherwise clear the queue and then have
        # this frame put a wink from the old mode straight back into it.
        self._wink_generation = 0
        self.winks_dropped_mid_frame = 0
        self.lock = threading.Lock()
        self.state = LiveState()
        self.frames = 0
        self._last_steady_point: tuple[float, float] | None = None
        self.errors = 0
        self.last_error: str | None = None
        self._frame_times: list[float] = []
        # (interval since the previous callback started or None, callback
        # duration), both ms. Only the callback is timed: capture and display
        # have no reliable timestamps here, so this cannot say camera->screen.
        # Bounded so a view nobody drains cannot grow without limit.
        self._timing: collections.deque[tuple[float | None, float]] = collections.deque(
            maxlen=TIMING_BUFFER
        )
        self._last_callback_start: float | None = None
        self.perf_clock = time.perf_counter

    def on_observation(self, obs: FrameObservation) -> None:
        """Called on the camera thread; must never raise into the source."""

        self._guarded(lambda: self._process(obs))

    def _guarded(self, work: Callable[[], None]) -> None:
        start = self.perf_clock()
        try:
            work()
        except Exception as exc:  # noqa: BLE001 - a view must not kill the camera thread
            with self.lock:
                self.errors += 1
                self.last_error = repr(exc)
        duration_ms = (self.perf_clock() - start) * 1000.0
        with self.lock:
            last = self._last_callback_start
            interval_ms = None if last is None else (start - last) * 1000.0
            self._last_callback_start = start
            self._timing.append((interval_ms, duration_ms))

    def drain_timing(self) -> list[tuple[float | None, float]]:
        """Every (interval_ms, callback_ms) since the last drain; clears them."""

        with self.lock:
            out = list(self._timing)
            self._timing.clear()
        return out

    def _process(self, obs: FrameObservation) -> None:
        now_s = obs.observed_s
        with self.lock:
            cancel_wink = self._cancel_wink_requested
            self._cancel_wink_requested = False
            generation = self._wink_generation
        if cancel_wink:
            # Before the detector is updated, never after: a cancel applied
            # afterwards would let this frame finish a hold that began in the
            # mode the person has just left.
            self.wink_detector.cancel()
        features = GATE.features_for(obs, self.policy)
        # Named for the PERSON from here down. The camera faces them, so the
        # library's "left" is the eye on the left of the IMAGE, which is their
        # right one. Every gesture in this project watched the wrong eye.
        left, right = GEST.eyes_as_the_person_has_them(*obs.openness_image)
        head = obs.head6 if obs.gaze_status else None

        # Same gate as the recorder's overlay: a blink is not a gaze sample,
        # and predicting through one puts the point somewhere the eye is not.
        valid = GATE.is_gaze_sample(obs, self.policy)
        # The gesture reads the FACE, not the gaze: it has to keep working
        # when the gaze is unusable, because "eyes shut" is precisely when
        # there is no gaze.
        face_present = obs.face_present
        # BOTH eyes, not either. Written as "not (left and right)" this was
        # true during a one-eyed wink as well, so a wink drove the mode menu
        # and a wink-to-click would have fired two mechanisms from one
        # gesture. The two signals are now exclusive by construction.
        # Judged against each eye's own baseline. Measured live on 9.9: over
        # 627 frames the openness never once reached the absolute threshold of
        # 10.0 -- minima of 27.0 and 64.5 against medians of 191.5 and 160.0 --
        # so every eyelid gesture was silently impossible. The same minima are
        # 0.14 and 0.40 of their own baselines, which is readable.
        left_open = self.left_gate.is_open(left)
        right_open = self.right_gate.is_open(right)
        eyes_shut = not left_open and not right_open
        event = self.detector.update(now_s, face_present=face_present, eyes_shut=eyes_shut)
        # Ratios, not the open/shut booleans: the operator's left eye narrows
        # whenever the right one closes, so a rule that needed it OPEN rejected
        # every real wink. Comparing the two depths does not.
        left_ratio = self.left_gate.ratio if self.left_gate.ratio is not None else 1.0
        right_ratio = self.right_gate.ratio if self.right_gate.ratio is not None else 1.0
        winked = self.wink_detector.update(
            now_s,
            face_present=face_present,
            left_ratio=left_ratio,
            right_ratio=right_ratio,
        )
        # An eye on its way down still passes the blink gate, so a prediction
        # is still produced -- from an eye already half behind its lid. That
        # is what makes the pointer wander at the start of a wink, well before
        # anything registers the closure.
        steady = self.left_gate.config.steady_fraction
        eyes_steady = valid and left_ratio >= steady and right_ratio >= steady
        # ``head`` above is used only when the gaze is usable, and the whole
        # question here is what the head was doing while an eye was SHUT, so
        # the pose of any frame carrying a face is reported (the source builds
        # it under HeadPolicy.GAZE_OR_FACE). Reporting only -- the prediction
        # below still takes ``head``.
        pose = head
        if pose is None and face_present:
            pose = obs.head6
        head_pitch = float(pose[2]) if pose is not None and len(pose) > 2 else None
        raw = (
            PRED.predict_overlay_point(self.model, self.model_y, features, head, self.rig)
            if valid
            else None
        )

        point = GATE.filter_point(self.filter, raw, now_s)

        raw_model = None
        if obs.raw_gaze_cm is not None:
            try:
                raw_model = self.rig.cm_to_norm(*obs.raw_gaze_cm)
            except (TypeError, ValueError):
                raw_model = None

        with self.lock:
            if event is not GEST.Event.NONE:
                self.gesture_events.append((now_s, event))
            if winked and generation != self._wink_generation:
                # A mode change happened while this frame was being computed.
                # The closure it came from belongs to the mode the person has
                # already left, so it is dropped here rather than published
                # into a queue that was cleared a moment ago.
                self.winks_dropped_mid_frame += 1
            elif winked:
                # The point from before the eye began to close, not merely the
                # last one that passed the blink gate: the half-closed frames
                # pass it too, and they are the ones that put the pointer
                # somewhere the person was never looking.
                self.wink_events.append((now_s, self._last_steady_point))
            if eyes_steady and point is not None:
                self._last_steady_point = point
            self.frames += 1
            self._frame_times.append(now_s)
            if len(self._frame_times) > 60:
                del self._frame_times[:-60]
            self.state = LiveState(
                face_present=face_present,
                openness=(left, right),
                openness_ratio=(self.left_gate.ratio or 0.0, self.right_gate.ratio or 0.0),
                eyes_steady=eyes_steady,
                head_pitch=head_pitch,
                head=None if pose is None else tuple(float(v) for v in pose),
                point=point,
                raw_model=raw_model,
                unfiltered=raw,
                tracking=valid,
                updated_s=now_s,
                frames=self.frames,
                fps=self._fps(),
            )

    def drain_gesture_events(self) -> list[tuple[float, GEST.Event]]:
        with self.lock:
            events = self.gesture_events
            self.gesture_events = []
        return events

    def drain_wink_events(self) -> list[tuple[float, tuple[float, float] | None]]:
        """Deliberate right winks, each with the gaze point it was aimed at."""

        with self.lock:
            events = self.wink_events
            self.wink_events = []
        return events

    def cancel_wink(self) -> int:
        """Every wink in flight is dropped: fired, and part-way through.

        Returns how many had already fired. Called on every change of mode.

        Dropping the queue alone was the old rule at the scroll edge, and it
        only covers half the case. A wink can be MID-DETECTION at the moment
        the mode changes -- the eye is already shut and the hold accumulating,
        with nothing yet in any queue -- and it would then fire a moment later
        in the new mode, from a closure made in the old one. So the detector
        is cancelled as well, and it will not start accumulating again until
        the eye has been seen open (``RightWinkDetector.cancel``).
        """

        with self.lock:
            dropped = len(self.wink_events)
            self.wink_events = []
            self._cancel_wink_requested = True
            # Anything the camera thread is part-way through publishing is
            # invalidated by this, not merely anything already published.
            self._wink_generation += 1
        return dropped

    def _fps(self, window: int = 30) -> float | None:
        times = self._frame_times[-window:]
        if len(times) < 2:
            return None
        span = times[-1] - times[0]
        return None if span <= 0 else (len(times) - 1) / span
