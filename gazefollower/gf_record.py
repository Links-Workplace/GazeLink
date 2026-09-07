"""Record GazeFollower's per-frame features on scripted protocols (M2-05).

The one live component of the experiment. It runs the REAL library pipeline
(camera -> MediaPipe -> MGazeNet -> pass-through calibration -> filter) in
SAMPLING state, subscribes to the per-frame ``(face_info, gaze_info)`` pair
the library dispatches, and writes one row per frame: the 258-d model output,
the head6 vector from the same frame's landmarks, the target on screen, the
protocol phase, and the diagnostics the fitter needs. Nothing is trained
here; ``gf_fit.py`` does that offline so every arm sees identical rows.

Protocols (plan §4):
  A     GazeFollower's own 9-point calibration, mirrored frame for frame from
        CalibrationController (warm-up centre unstored, 1.5 s prepare, 45
        ACCEPTED frames, blink gate, 0.5 s wait), white background, dot.png.
  TUNE   10 held-out targets from targets_tune.json, 1.5 s settle + 1.5 s
         collect. Used ONLY to pick a fitter configuration.
  GRID16 the targets document's 16-point grid. Eight of its points sit closer
         to a GazeFollower calibration point than the project's held-out
         threshold, so each target carries its distance and the near ones are
         reported apart from the far ones.
  T1     10 held-out targets from targets.json, same timing. Scored once, and
         the only set a generalisation claim may rest on.
  T2    3 targets, gaze fixed, guided slow nod, 20 s each. Head-robustness.

Faithfulness to the library's calibration UI: same point order and
positions, same timing constants, same acceptance gate, same background
colour, same dot image and beep, fullscreen at the library's device
resolution. The self-check at start-up compares our px2cm copy with the
library's on all nine grid points and refuses to run on a mismatch.

Safety and privacy: no OS input of any kind; no frames, images or landmarks
are written -- only derived scalars and the model's feature vector, under
``recordings/`` (gitignored). ``Esc`` aborts; partial recordings are saved
with ``aborted=true`` so nothing is silently lost or silently reused.

``gazefollower`` is imported only inside the functions that need it (its
import initialises native components), so the protocol logic here is unit
tested with fakes and no camera.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_common as C  # noqa: E402
import gf_head_features as H  # noqa: E402
import gf_display as GD  # noqa: E402
import gf_gaze_filter as GF  # noqa: E402
import gf_setup as GS  # noqa: E402
import gf_targets as GT  # noqa: E402
import gf_schema as S  # noqa: E402

PACKAGE_DIR = Path(__file__).resolve().parent
RECORDINGS_DIR = PACKAGE_DIR / "recordings"

WATCHDOG_S = 5.0  # no frame for this long -> abort the run
# A subscriber that keeps throwing produces frames that advance the watchdog
# while no row is ever stored, so counting errors is not enough: after this
# many consecutive failures the run stops loudly instead of quietly recording
# nothing.
MAX_CONSECUTIVE_ERRORS = 10
# Frames can arrive perfectly while the detector never finds a face: a covered
# lens, a dark room, the operator out of shot, or another application holding
# the camera. Nothing above catches that -- the watchdog sees frames and the
# subscriber raises nothing -- so round12 recorded 903 frames of FACE_MISSING
# and still reported finished=true with no failure. A run with no usable gaze
# is not a run, so it stops loudly instead. Blinks and brief look-aways are
# far shorter than this; at ~32 fps it is about ten seconds.
MAX_CONSECUTIVE_INVALID_FRAMES = 320
# Library cleanup calls join() with no timeout; every step gets its own bound
# so one stalled thread cannot prevent the capture release or the CSV close.
SHUTDOWN_STEP_TIMEOUT_S = 3.0
# After the last target, give any callback already inside the subscriber time
# to finish before the row list is frozen.
DRAIN_S = 0.25
CAMERA_WARMUP_S = 2.0  # camera open, no target: lets the tracker lock on
# Protocols wait for a key press rather than a countdown: a three-second
# countdown did not leave time to read the instruction.
EXPECTED_FEATURE_DIM = 258
OVERLAY_STALE_S = 0.25

T2_TARGETS: tuple[tuple[float, float], ...] = ((0.5, 0.10), (0.5, 0.50), (0.5, 0.90))
T2_SECONDS = 20.0
T2_SETTLE_S = 1.5

INSTRUCTIONS = {
    "A": "Calibration. Keep your head STILL and look at each dot.",
    "FULL": "Full-screen targets: centre, sides, top, bottom and corners. Head still.",
    "MOVE": "Same targets. Sit naturally and let your head move a little as you normally would.",
    "TUNE": "Tuning targets. Keep your head still and look at each dot.",
    "GRID16": "16-point grid. Keep your head still and look at each dot.",
    "T1": "Held-out targets. Keep your head still and look at each dot.",
    "T2": "Keep your EYES on the dot and NOD your head SLOWLY up and down.",
}


# --- Gates: per-target state machines, driven by frame arrival --------------


@dataclass
class CollectionGate:
    """Mirror of CalibrationController.add_cali_feature for ONE target.

    ``observe`` is called once per processed frame with that frame's clock
    time and gaze/openness values; it returns the phase label for the row and
    whether the row is ACCEPTED (counts toward the 45 AND is stored). For the
    warm-up target ``stored=False``: frames count but are never stored, as in
    the library (``if self._current_index != 0``).
    """

    onset_s: float
    stored: bool = True
    prepare_s: float = C.PREPARE_S
    wait_s: float = C.WAIT_S
    n_frames: int = C.N_FRAMES_PER_POINT
    blink_threshold: float = C.BLINK_THRESHOLD
    n_accepted: int = 0
    full_time_s: float | None = None
    done: bool = False

    def observe(self, now_s: float, gaze_status: bool, left_openness: float, right_openness: float) -> tuple[str, bool]:
        if self.done:
            return (S.PHASE_WAIT if self.stored else S.PHASE_WARMUP), False
        if now_s - self.onset_s < self.prepare_s:
            return (S.PHASE_PREPARE if self.stored else S.PHASE_WARMUP), False
        accepted = False
        if self.n_accepted < self.n_frames:
            ok = bool(gaze_status) and left_openness > self.blink_threshold and right_openness > self.blink_threshold
            if ok:
                self.n_accepted += 1
                accepted = self.stored
                if self.n_accepted == self.n_frames:
                    self.full_time_s = now_s
            return (S.PHASE_COLLECT if self.stored else S.PHASE_WARMUP), accepted
        # n_accepted == n_frames: wait, then release the target.
        assert self.full_time_s is not None
        if now_s - self.full_time_s >= self.wait_s:
            self.done = True
        return (S.PHASE_WAIT if self.stored else S.PHASE_WARMUP), False


@dataclass
class TimedGate:
    """A target shown for settle + collect seconds; phase from elapsed time."""

    onset_s: float
    settle_s: float
    collect_s: float
    done: bool = False

    def observe(self, now_s: float) -> str:
        elapsed = now_s - self.onset_s
        if elapsed >= self.settle_s + self.collect_s:
            self.done = True
            return S.PHASE_COLLECTING
        return S.PHASE_STABILIZING if elapsed < self.settle_s else S.PHASE_COLLECTING


@dataclass
class Target:
    index: int  # -1 for the warm-up
    name: str
    x: float
    y: float
    block: int = 0


@dataclass
class ProtocolSpec:
    name: str
    targets: list[Target]
    kind: str  # "calibration" | "timed"
    settle_s: float = C.DEFAULT_SETTLE_MS / 1000.0
    collect_s: float = C.DEFAULT_COLLECT_MS / 1000.0
    instruction: str = ""

    def exported_targets(self) -> list[dict[str, Any]]:
        return [
            {"index": t.index, "name": t.name, "screen_position": {"x": t.x, "y": t.y}}
            for t in self.targets
            if t.index >= 0
        ]


def protocol_a(grid_x: tuple[float, float, float] | None = None) -> ProtocolSpec:
    """The library's nine-point calibration, or the same protocol on a narrower band."""

    sequence = C.NINE_POINT_SEQUENCE if grid_x is None else C.nine_point_sequence(grid_x)
    targets = [Target(-1, "WARMUP", *sequence[0])]
    targets += [Target(i, f"CAL_{i}", x, y) for i, (x, y) in enumerate(sequence[1:])]
    return ProtocolSpec("A", targets, "calibration", instruction=INSTRUCTIONS["A"])


def protocol_timed(name: str, targets_file: Path, settle_s: float, collect_s: float) -> ProtocolSpec:
    raw, _ = C.load_targets(targets_file)
    targets = [Target(int(t["index"]), str(t["name"]), float(t["screen_position"]["x"]), float(t["screen_position"]["y"])) for t in raw]
    return ProtocolSpec(name, targets, "timed", settle_s=settle_s, collect_s=collect_s, instruction=INSTRUCTIONS[name])


def protocol_t2() -> ProtocolSpec:
    targets = [Target(i, f"T2_{i}", x, y) for i, (x, y) in enumerate(T2_TARGETS)]
    return ProtocolSpec("T2", targets, "timed", settle_s=T2_SETTLE_S, collect_s=T2_SECONDS - T2_SETTLE_S, instruction=INSTRUCTIONS["T2"])


# --- The runner: all protocol logic on the camera thread, like the library --


@dataclass
class RunnerState:
    """What the display loop reads. Replaced as a whole, never mutated.

    ``raw_norm`` and ``overlay_norm`` exist so the operator can SEE, every
    frame, whether the system is tracking something at all -- not only read a
    number after the session ends. The user asked for this explicitly after
    several numeric reports turned out to contain mistakes a live dot would
    have made obvious immediately: no exception for calibration points, this
    is shown for every frame of every protocol.

    ``raw_norm``     the model's raw 2-D output for THIS frame, converted to
                      screen fractions with no calibration applied. Always
                      available once a face is tracked; it moves, but not
                      accurately -- it exists to prove the pipeline is alive.
    ``overlay_raw_norm`` is the fitted model's unfiltered prediction for this
                      frame. It is retained for a deliberate diagnostic view.
    ``overlay_norm``  is the time-filtered calibrated prediction drawn blue;
                      ``None`` when tracking is invalid or no model is loaded.
    """

    protocol: str
    target: Target | None
    phase: str
    progress: int  # 0..100 for calibration targets
    target_pos: int
    target_count: int
    frames: int
    fps: float | None
    head_valid: bool
    finished: bool
    pitch_a: float | None
    raw_norm: tuple[float, float] | None = None
    frame_updated_s: float | None = None
    overlay_raw_norm: tuple[float, float] | None = None
    overlay_norm: tuple[float, float] | None = None
    overlay_updated_s: float | None = None


class ProtocolRunner:
    """Drives one ProtocolSpec from frames. Thread-safe via one lock.

    ``on_frame`` is what the camera thread calls (wrapped by the subscriber);
    it appends exactly one row per frame and advances targets. Nothing here
    imports the library: ``face_info``/``gaze_info`` are duck-typed.
    """

    def __init__(
        self,
        spec: ProtocolSpec,
        builder: S.RecordingBuilder,
        rig: C.RigGeometry,
        *,
        clock: Callable[[], float] = time.monotonic,
        speed: float = 1.0,
        head_builder: Callable[[Any], np.ndarray | None] = H.build,
        pnp: Callable[[Any], tuple[float, float, float] | None] = H.pnp_degrees,
        overlay_model: Any = None,
        overlay_filter_settings: GF.FilterSettings | None = None,
    ) -> None:
        self.spec = spec
        self.builder = builder
        self.rig = rig
        self.clock = clock
        self.speed = float(speed)
        self.head_builder = head_builder
        self.pnp = pnp
        # A model fitted in an EARLIER round, loaded so its live predictions
        # can be drawn as a moving dot this session. It is a visual sanity
        # check, not a claim that this session is calibrated by it.
        self.overlay_model = overlay_model
        self.overlay_filter = (
            None if overlay_model is None or overlay_filter_settings is None else GF.GazePointFilter(overlay_filter_settings)
        )
        self.lock = threading.Lock()
        self._index = -1  # -1: idle, before first target
        self._gate: CollectionGate | TimedGate | None = None
        self._onset_s: float | None = None
        self.frames = 0
        self.errors = 0
        self.consecutive_errors = 0
        self.failed = False
        self.last_error: str | None = None
        self.valid_gaze_frames = 0
        self.consecutive_invalid = 0
        self.no_face_stall = False
        self._frame_times: list[float] = []
        self.finished = False
        self.state = RunnerState(
            protocol=spec.name,
            target=None,
            phase=S.PHASE_IDLE,
            progress=0,
            target_pos=0,
            target_count=len(spec.targets),
            frames=0,
            fps=None,
            head_valid=False,
            finished=False,
            pitch_a=None,
        )
        self.feature_dim: int | None = builder.feature_dim

    # -- control from the main thread
    def start(self) -> None:
        with self.lock:
            self._advance(self.clock())

    def _advance(self, now_s: float) -> None:
        self._index += 1
        if self._index >= len(self.spec.targets):
            self._gate = None
            self.finished = True
            return
        target = self.spec.targets[self._index]
        self._onset_s = now_s
        if self.spec.kind == "calibration":
            self._gate = CollectionGate(
                onset_s=now_s,
                stored=target.index >= 0,
                prepare_s=C.PREPARE_S / self.speed,
                wait_s=C.WAIT_S / self.speed,
            )
        else:
            self._gate = TimedGate(now_s, self.spec.settle_s / self.speed, self.spec.collect_s / self.speed)

    def current_target(self) -> Target | None:
        if 0 <= self._index < len(self.spec.targets):
            return self.spec.targets[self._index]
        return None

    # -- called per frame from the camera thread
    def on_frame(self, face_info: Any, gaze_info: Any) -> None:
        try:
            self._on_frame(face_info, gaze_info)
        except Exception as exc:  # noqa: BLE001 - must never propagate into the library's thread
            with self.lock:
                self.errors += 1
                self.consecutive_errors += 1
                self.last_error = repr(exc)
                if self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    self.failed = True
        else:
            with self.lock:
                self.consecutive_errors = 0

    def _on_frame(self, face_info: Any, gaze_info: Any) -> None:
        now_s = self.clock()
        with self.lock:
            self.frames += 1
            self._frame_times.append(now_s)
            target = self.current_target()
            gate = self._gate
            gaze_status = bool(getattr(gaze_info, "status", False))
            features = getattr(gaze_info, "features", None) if gaze_status else None
            if features is not None:
                features = np.asarray(features, dtype=np.float32).reshape(-1)
                if self.feature_dim is None:
                    self.feature_dim = int(features.shape[0])
            left = float(getattr(face_info, "left_eye_openness", 0.0) or 0.0)
            right = float(getattr(face_info, "right_eye_openness", 0.0) or 0.0)
            head = self.head_builder(face_info) if gaze_status else None
            pnp = self.pnp(face_info) if head is not None else None
            raw = getattr(gaze_info, "raw_gaze_coordinates", None) if gaze_status else None
            raw_cm = None if raw is None else (float(raw[0]), float(raw[1]))
            state_obj = getattr(gaze_info, "tracking_state", None)
            tracking = getattr(state_obj, "name", None) or (str(state_obj) if state_obj is not None else "UNKNOWN")

            # A frame carrying no gaze is normal in ones and twos -- a blink,
            # a glance away. An unbroken run of them means the camera is not
            # seeing a face at all, and the rest of the guards cannot tell.
            if gaze_status:
                self.valid_gaze_frames += 1
                self.consecutive_invalid = 0
            else:
                self.consecutive_invalid += 1
                if self.consecutive_invalid >= MAX_CONSECUTIVE_INVALID_FRAMES:
                    self.no_face_stall = True

            phase = S.PHASE_IDLE
            accepted = False
            elapsed_ms: float | None = None
            target_xy = None
            label_cm = None
            progress = 0
            if target is not None and gate is not None and self._onset_s is not None:
                elapsed_ms = (now_s - self._onset_s) * 1000.0 * self.speed
                target_xy = (target.x, target.y)
                label_cm = self.rig.norm_to_label_cm(target.x, target.y)
                if isinstance(gate, CollectionGate):
                    phase, accepted = gate.observe(now_s, gaze_status, left, right)
                    progress = int(round(gate.n_accepted * 100 / gate.n_frames))
                else:
                    phase = gate.observe(now_s)
            # An idle frame -- no target on screen -- is kept for frame
            # accounting and gap detection, but its embedding is dropped: the
            # fitter never reads idle rows, so storing a face template for
            # them would be exposure with no use.
            idle = target is None or phase == S.PHASE_IDLE
            raw_norm = None
            if raw_cm is not None:
                try:
                    raw_norm = self.rig.cm_to_norm(*raw_cm)
                except (TypeError, ValueError):
                    raw_norm = None
            overlay_raw_norm = None
            overlay_norm = None
            overlay_updated_s = None
            overlay_valid = (
                not idle
                and gaze_status
                and left > C.BLINK_THRESHOLD
                and right > C.BLINK_THRESHOLD
            )
            if overlay_valid:
                overlay_raw_norm = self._predict_overlay(features, head)
            if overlay_raw_norm is not None:
                try:
                    overlay_norm = (
                        overlay_raw_norm
                        if self.overlay_filter is None
                        else self.overlay_filter.update(overlay_raw_norm, now_s)
                    )
                    overlay_updated_s = now_s
                except ValueError:
                    overlay_norm = None
                    if self.overlay_filter is not None:
                        self.overlay_filter.reset()
            elif self.overlay_filter is not None:
                # A blink, occlusion, tracking loss or bad model result must
                # not pull a future valid point toward stale history.
                self.overlay_filter.reset()

            self.builder.append(
                frame_seq=self.frames - 1,
                timestamp_ns=int(getattr(gaze_info, "timestamp", 0) or getattr(face_info, "timestamp", 0) or 0),
                elapsed_ms=elapsed_ms,
                target_id=-1 if target is None else target.index,
                block=0 if target is None else target.block,
                phase=phase,
                target_xy=target_xy,
                label_cm=label_cm,
                features=None if idle else features,
                head=None if idle else head,
                pnp_deg=None if idle else pnp,
                raw_cm=None if idle else raw_cm,
                openness=(left, right),
                tracking_state=tracking,
                gaze_status=gaze_status,
                accepted=accepted,
            )
            if gate is not None and gate.done:
                self._advance(now_s)
            self.state = RunnerState(
                protocol=self.spec.name,
                target=self.current_target(),
                phase=phase,
                progress=progress,
                target_pos=self._index + 1,
                target_count=len(self.spec.targets),
                frames=self.frames,
                fps=self.fps_recent(),
                head_valid=head is not None,
                finished=self.finished,
                pitch_a=None if head is None else float(head[H.HEAD6_NAMES.index("pitch_a")]),
                raw_norm=raw_norm,
                frame_updated_s=now_s,
                overlay_raw_norm=overlay_raw_norm,
                overlay_norm=overlay_norm,
                overlay_updated_s=overlay_updated_s,
            )

    def _predict_overlay(self, features: np.ndarray | None, head: np.ndarray | None) -> tuple[float, float] | None:
        """Run this frame through the loaded overlay model, if any.

        Never raises into the camera thread: a bad frame for the overlay is a
        missing dot, not a crashed recording.
        """

        if self.overlay_model is None or features is None:
            return None
        try:
            head_names = self.overlay_model.schema.head_names
            if head_names:
                if head is None:
                    return None
                design = S.assemble(features.reshape(1, -1), head.reshape(1, -1), head_names, H.HEAD6_NAMES)
            else:
                design = features.reshape(1, -1)
            point = self.overlay_model.predict_norm(design, self.rig)[0]
            if not np.all(np.isfinite(point)):
                return None
            return float(point[0]), float(point[1])
        except Exception:  # noqa: BLE001 - a visual aid must never break recording
            return None

    def fps_recent(self, window: int = 30) -> float | None:
        times = self._frame_times[-window:]
        if len(times) < 2 or times[-1] <= times[0]:
            return None
        return (len(times) - 1) / (times[-1] - times[0])

    def fps_median(self) -> float | None:
        if len(self._frame_times) < 3:
            return None
        gaps = [b - a for a, b in zip(self._frame_times, self._frame_times[1:]) if b > a]
        return None if not gaps else 1.0 / statistics.median(gaps)

    def integrity(self, *, watchdog_tripped: bool, aborted: bool) -> dict[str, Any]:
        duration = (self._frame_times[-1] - self._frame_times[0]) if len(self._frame_times) > 1 else 0.0
        return {
            "frames": self.frames,
            "duration_s": duration,
            "fps_median": self.fps_median(),
            "subscriber_errors": self.errors,
            "last_error": self.last_error,
            "watchdog_tripped": watchdog_tripped,
            "aborted": aborted,
            "finished": self.finished,
            "feature_dim": self.feature_dim,
            "valid_gaze_frames": self.valid_gaze_frames,
            "valid_gaze_fraction": (self.valid_gaze_frames / self.frames) if self.frames else 0.0,
            "no_face_stall": self.no_face_stall,
        }


# --- Library adapters (imported lazily) --------------------------------------


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


MIN_USABLE_FPS = 25.0


@dataclass(frozen=True)
class CameraVerdict:
    """What the camera probe means for whether recording may start.

    Three outcomes, and the distinction matters: a camera that is merely SLOW
    is the operator's call, while a camera that is STUCK or absent is not a
    missing measurement to shrug at. An earlier version only guarded on
    "measured a rate below the threshold", so a probe that timed out -- which
    reports no rate at all -- fell through that test and started recording
    anyway, with the leaked probe thread still holding the camera.
    """

    message: str
    fatal: bool = False
    reason: str = ""
    needs_confirmation: bool = False


def camera_verdict(camera: Mapping[str, Any], min_fps: float = MIN_USABLE_FPS) -> CameraVerdict:
    if camera.get("timed_out"):
        return CameraVerdict(
            message="camera: did not answer within the probe timeout",
            fatal=True,
            reason=(
                "the camera is stuck. Another program is almost certainly holding it "
                "(video call, browser tab, or an earlier run that never exited). The "
                "probe thread is still blocked on it, so recording would fail too. "
                "Close whatever holds the camera and run again."
            ),
        )
    if camera.get("opened") is False:
        detail = camera.get("error")
        return CameraVerdict(
            message=f"camera: failed to open{' -- ' + str(detail) if detail else ''}",
            fatal=True,
            reason="the camera could not be opened at all.",
        )
    fps = camera.get("measured_fps")
    if fps is None:
        return CameraVerdict(
            message="camera: opened but produced no usable frames",
            fatal=True,
            reason="no frame rate could be measured, so there is nothing to record with.",
        )
    message = (
        f"camera: {fps:.1f} fps measured, {camera.get('frame_width')}x{camera.get('frame_height')} "
        f"via {camera.get('backend')}"
    )
    if fps < min_fps:
        return CameraVerdict(
            message=message
            + (
                f"\n\nThat is below the {min_fps:.0f} fps this experiment needs.\n"
                "  Usual causes, in order: another program is holding the webcam\n"
                "  (video call, browser tab, a previous run that did not exit);\n"
                "  or the room is dark enough that the camera stretched its\n"
                "  exposure. Close other users of the camera, add light, and\n"
                "  run again. Recording now would produce data that fails the\n"
                "  validity check anyway."
            ),
            needs_confirmation=True,
        )
    return CameraVerdict(message=message)


def confirm_low_frame_rate() -> bool:
    """Ask before recording at a rate that will fail validity anyway.

    With no interactive terminal there is nobody to ask, so the answer is no:
    a script that cannot be questioned must not silently record unusable data.
    """

    if not sys.stdin or not sys.stdin.isatty():
        print("no interactive terminal to confirm on; refusing to record at this frame rate")
        return False
    try:
        return input("continue anyway? [y/N] ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def watchdog_tripped(last_frame_s: float | None, now_s: float, limit_s: float = WATCHDOG_S) -> bool:
    """True when the last frame is older than the limit. Never true without one."""

    return last_frame_s is not None and (now_s - last_frame_s) > limit_s


@dataclass
class FrameWatchdog:
    """Trips when frames stop arriving DURING one protocol.

    Armed per protocol, and deliberately so. A single watchdog shared across
    protocols carries the previous one's last frame time across the wait for
    a key press; since that wait is unbounded and records nothing, any pause
    longer than the limit would trip the watchdog on the next protocol's very
    first loop -- aborting it before it recorded a frame, because the operator
    took time to read.

    Before the first frame arrives, the clock runs from ``started_s``: a
    protocol that never receives anything must still trip.
    """

    limit_s: float
    started_s: float
    last_frame_s: float | None = None

    def note(self, frame_times: Sequence[float]) -> None:
        if frame_times:
            self.last_frame_s = frame_times[-1]

    def tripped(self, now_s: float) -> bool:
        reference = self.last_frame_s if self.last_frame_s is not None else self.started_s
        return (now_s - reference) > self.limit_s

    def silent_for(self, now_s: float) -> float:
        reference = self.last_frame_s if self.last_frame_s is not None else self.started_s
        return now_s - reference


def check_px2cm_against_library(rig: C.RigGeometry) -> None:
    from gazefollower.misc import px2cm  # noqa: PLC0415

    for nx, ny in C.NINE_POINT_STORED:
        ours = rig.norm_to_label_cm(nx, ny)
        theirs = px2cm(
            (nx * rig.device_w_px, ny * rig.device_h_px),
            (rig.camera_x_cm, rig.camera_y_cm),
            (rig.screen_w_cm, rig.screen_h_cm),
            (rig.device_w_px, rig.device_h_px),
        )
        if not (math.isclose(ours[0], theirs[0], abs_tol=1e-9) and math.isclose(ours[1], theirs[1], abs_tol=1e-9)):
            raise RuntimeError(f"px2cm mismatch at ({nx}, {ny}): ours {ours} vs library {theirs}")


# --- Dry-run fakes: real GazeFollower.process_frame, no camera, no models ----


def make_dry_run_components(fps: float = 30.0) -> tuple[Any, Any, Any]:
    from gazefollower.camera import Camera  # noqa: PLC0415
    from gazefollower.misc import CameraRunningState, FaceInfo, GazeInfo, TrackingState  # noqa: PLC0415

    class FakeCamera(Camera):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self._thread: threading.Thread | None = None
            self._running = False
            self._camera_thread = None
            self._camera_thread_running = None
            self._cap = None

        def open(self) -> None:
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._camera_thread = self._thread
            self._camera_thread_running = True
            self._thread.start()

        def _loop(self) -> None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            while self._running and self._camera_thread_running:
                with self.callback_and_param_lock:
                    cb = self.callback_func
                if cb is not None and self.camera_running_state != CameraRunningState.CLOSING:
                    cb(self.camera_running_state, time.time_ns(), frame, *self.callback_args, **self.callback_kwargs)
                time.sleep(1.0 / fps)

        def close(self) -> None:
            self._running = False
            if self._thread is not None:
                self._thread.join(timeout=2.0)

        def release(self) -> None:
            self.close()

    class FakeFaceAlignment:
        def __init__(self) -> None:
            self._rng = np.random.default_rng(1)
            self._t = 0.0

        def detect(self, timestamp: int, image: np.ndarray) -> Any:
            self._t += 0.05
            info = FaceInfo()
            info.timestamp = timestamp
            info.status = True
            info.can_gaze_estimation = True
            info.img_w, info.img_h = image.shape[1], image.shape[0]
            lm = np.zeros((478, 3), dtype=np.float64)
            lm[:, 0] = 320 + self._rng.normal(scale=40, size=478)
            lm[:, 1] = 240 + self._rng.normal(scale=40, size=478)
            nod = 12.0 * math.sin(self._t)
            lm[H.EYE_OUTER_IMAGE_LEFT, :2] = (250, 200)
            lm[H.EYE_OUTER_IMAGE_RIGHT, :2] = (390, 200)
            lm[H.NOSE_TIP, :2] = (320, 260 + nod)
            lm[H.CHIN, :2] = (320, 340 + nod)
            lm[H.MOUTH_IMAGE_LEFT, :2] = (290, 305 + nod)
            lm[H.MOUTH_IMAGE_RIGHT, :2] = (350, 305 + nod)
            info.face_landmarks = np.round(lm).astype(np.int16)
            info.face_rect = np.array([230, 150, 180, 220])
            info.left_rect = np.array([240, 185, 60, 30])
            info.right_rect = np.array([340, 185, 60, 30])
            info.left_eye_openness = 120.0
            info.right_eye_openness = 110.0
            return info

        def release(self) -> None:
            return None

    class FakeEstimator:
        def __init__(self) -> None:
            self._rng = np.random.default_rng(2)

        def detect(self, image: np.ndarray, face_info: Any) -> Any:
            info = GazeInfo()
            info.timestamp = face_info.timestamp
            if not face_info.status:
                info.tracking_state = TrackingState.FACE_MISSING
                return info
            info.features = self._rng.normal(size=EXPECTED_FEATURE_DIM).astype(np.float32)
            info.raw_gaze_coordinates = info.features[:2]
            info.status = True
            info.left_openness = face_info.left_eye_openness
            info.right_openness = face_info.right_eye_openness
            info.tracking_state = TrackingState.SUCCESS
            return info

        def release(self) -> None:
            return None

    return FakeCamera(), FakeFaceAlignment(), FakeEstimator()


# --- Display -----------------------------------------------------------------


class Display:
    """pygame fullscreen on the CHOSEN monitor; white like the library's UI.

    ``origin`` is the monitor's position in the virtual desktop. On a second
    screen it is not (0, 0), and a window opened without it lands on the
    primary display -- where the targets would be drawn on one monitor while
    the geometry describes another.
    """

    def __init__(
        self,
        width: int,
        height: int,
        *,
        headless: bool,
        origin: tuple[int, int] = (0, 0),
        overlay_available: bool = False,
        show_unfiltered_overlay: bool = False,
        overlay_stale_s: float = OVERLAY_STALE_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.headless = headless
        self.width, self.height = width, height
        self.origin = origin
        self._overlay_available = overlay_available
        self._show_unfiltered_overlay = show_unfiltered_overlay
        self._overlay_stale_s = overlay_stale_s
        self._clock = clock
        if headless:
            return
        import os  # noqa: PLC0415

        # SDL reads this at video-subsystem init, so it must be set first.
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{origin[0]},{origin[1]}"
        import pygame  # noqa: PLC0415

        pygame.init()
        try:
            pygame.mixer.init()
        except Exception:  # noqa: BLE001 - sound is a courtesy
            pass
        self.pg = pygame
        flags = pygame.NOFRAME if origin != (0, 0) else pygame.FULLSCREEN
        self.screen = pygame.display.set_mode((width, height), flags)
        pygame.display.set_caption("GAZELINK - GazeFollower recording")
        self.font = pygame.font.Font(None, 44)
        self.big = pygame.font.Font(None, 64)
        res = Path(_library_dir()) / "res"
        self.dot = None
        self.beep = None
        try:
            self.dot = pygame.transform.smoothscale(pygame.image.load(str(res / "image" / "dot.png")), (70, 70))
        except Exception:  # noqa: BLE001
            self.dot = None
        try:
            self.beep = pygame.mixer.Sound(str(res / "audio" / "beep.wav"))
        except Exception:  # noqa: BLE001
            self.beep = None

    def poll_escape(self) -> bool:
        if self.headless:
            return False
        for event in self.pg.event.get():
            if event.type == self.pg.KEYDOWN and event.key == self.pg.K_ESCAPE:
                return True
        return False

    def wait_for_key(self, lines: Sequence[str], *, timeout_s: float | None = None) -> str:
        """Hold the instruction on screen until the operator is ready.

        Returns "go" on Space or Enter, "abort" on Esc, "timeout" if a
        timeout was given and expired. A countdown used to do this job, which
        meant the instruction vanished before it could be read; a protocol
        that starts before the person knows what it asks for produces data
        about their confusion rather than about their gaze.

        Headless runs have nobody to press a key, so they proceed at once.
        """

        if self.headless:
            return "go"
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        prompt = list(lines) + ["", "Press SPACE or ENTER to start   (Esc to abort)"]
        # Drain anything queued before the prompt appeared, so a stray key
        # press during the previous protocol cannot skip this one.
        self.pg.event.clear()
        while True:
            # Draw BEFORE polling: otherwise a key already in flight ends the
            # wait on the first pass and the instruction is never shown at all,
            # which is the failure this screen exists to prevent.
            self.draw_message(prompt)
            for event in self.pg.event.get():
                if event.type == self.pg.KEYDOWN:
                    if event.key == self.pg.K_ESCAPE:
                        return "abort"
                    if event.key in (self.pg.K_SPACE, self.pg.K_RETURN, self.pg.K_KP_ENTER):
                        return "go"
                elif event.type == self.pg.QUIT:
                    return "abort"
            if deadline is not None and time.monotonic() > deadline:
                return "timeout"
            time.sleep(0.02)

    def play_beep(self) -> None:
        if not self.headless and self.beep is not None:
            self.beep.play()

    def draw_message(self, lines: Sequence[str]) -> None:
        if self.headless:
            return
        self.screen.fill((255, 255, 255))
        y = self.height // 2 - 40 * len(lines)
        for line in lines:
            surf = self.big.render(line, True, (20, 20, 20))
            self.screen.blit(surf, (self.width // 2 - surf.get_width() // 2, y))
            y += 80
        self.pg.display.flip()

    def draw_target(self, state: RunnerState, hud: Sequence[str]) -> None:
        """Draw the target, plus what the system currently sees.

        Shown for EVERY frame of EVERY protocol, calibration points included:
        a live dot following (or failing to follow) the target is a sanity
        check no post-hoc number can substitute for, and hiding it during
        calibration would exempt exactly the phase most worth watching.
        """

        if self.headless:
            return
        self.screen.fill((255, 255, 255))
        target = state.target
        if target is not None:
            cx = int(round(target.x * self.width))
            cy = int(round(target.y * self.height))
            if self.dot is not None:
                self.screen.blit(self.dot, (cx - 35, cy - 35))
            else:
                self.pg.draw.circle(self.screen, (30, 30, 30), (cx, cy), 24)
                self.pg.draw.circle(self.screen, (255, 255, 255), (cx, cy), 6)
            if state.protocol in S.CALIBRATION_PROTOCOLS and state.phase in (S.PHASE_COLLECT, S.PHASE_WAIT):
                label = self.font.render(str(state.progress), True, (255, 255, 255))
                self.screen.blit(label, (cx - label.get_width() // 2, cy - label.get_height() // 2))
        raw = visible_point(state.raw_norm, state.frame_updated_s, self._clock(), self._overlay_stale_s)
        self._draw_live_point(raw, (255, 160, 0), radius=6)  # raw model output: orange
        overlay = visible_overlay_point(state, self._clock(), self._overlay_stale_s)
        if self._show_unfiltered_overlay and overlay is not None:
            self._draw_live_point(state.overlay_raw_norm, (150, 70, 180), radius=7, cross=True)
        self._draw_live_point(overlay, (30, 110, 255), radius=10, cross=True)  # filtered calibrated: blue
        y = 20
        for line in hud:
            surf = self.font.render(line, True, (60, 60, 60))
            self.screen.blit(surf, (20, y))
            y += 40
        legend_y = self.height - 60
        self.pg.draw.circle(self.screen, (255, 160, 0), (30, legend_y), 6)
        self.screen.blit(self.font.render("raw model output (no calibration)", True, (90, 90, 90)), (46, legend_y - 12))
        self.pg.draw.circle(self.screen, (30, 110, 255), (30, legend_y + 28), 6)
        self.screen.blit(
            self.font.render(
                "filtered calibrated" if self._overlay_available else "calibrated: no --overlay-model loaded",
                True,
                (90, 90, 90),
            ),
            (46, legend_y + 16),
        )
        self.pg.display.flip()

    def _draw_live_point(self, point_norm: tuple[float, float] | None, colour: tuple[int, int, int], *, radius: int, cross: bool = False) -> None:
        """One moving dot: what the system currently reports, on or off screen.

        A point outside [0, 1] is drawn clamped to the edge with a ring, so
        "predicting off the display" is visibly different from "not tracking
        at all" (nothing drawn) rather than silently invisible.
        """

        if point_norm is None:
            return
        x, y = point_norm
        off_screen = not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)
        cx = int(round(min(1.0, max(0.0, x)) * self.width))
        cy = int(round(min(1.0, max(0.0, y)) * self.height))
        if cross:
            self.pg.draw.line(self.screen, colour, (cx - radius, cy), (cx + radius, cy), 3)
            self.pg.draw.line(self.screen, colour, (cx, cy - radius), (cx, cy + radius), 3)
        else:
            self.pg.draw.circle(self.screen, colour, (cx, cy), radius, 0 if not off_screen else 2)
        if off_screen:
            self.pg.draw.circle(self.screen, (200, 40, 40), (cx, cy), radius + 6, 2)

    def close(self) -> None:
        if self.headless:
            return
        try:
            self.pg.quit()
        except Exception:  # noqa: BLE001
            pass


def visible_overlay_point(
    state: RunnerState, now_s: float, stale_after_s: float = OVERLAY_STALE_S
) -> tuple[float, float] | None:
    """Hide a prediction when drawing continues but camera frames stop."""

    return visible_point(state.overlay_norm, state.overlay_updated_s, now_s, stale_after_s)


def visible_point(
    point: tuple[float, float] | None,
    updated_s: float | None,
    now_s: float,
    stale_after_s: float = OVERLAY_STALE_S,
) -> tuple[float, float] | None:
    """Return a display point only while its source frame is current."""

    if point is None or updated_s is None:
        return None
    age_s = float(now_s) - updated_s
    if not math.isfinite(age_s) or age_s < 0.0 or age_s > stale_after_s:
        return None
    return point


def _library_dir() -> str:
    import gazefollower  # noqa: PLC0415

    return str(Path(gazefollower.__file__).resolve().parent)


# --- Session -----------------------------------------------------------------


@dataclass
class SessionResult:
    recordings: dict[str, Path] = field(default_factory=dict)
    aborted: bool = False
    watchdog_tripped: bool = False
    failure: str | None = None
    shutdown: dict[str, Any] = field(default_factory=dict)


def run_session(
    *,
    protocols: Sequence[str],
    round_id: int,
    rig: C.RigGeometry,
    targets_t1: Path,
    targets_tune: Path,
    targets_grid16: Path,
    target_geometry: dict[str, Any],
    out_root: Path,
    dry_run: bool,
    headless: bool,
    speed: float,
    manifest: Any = None,
    x_range: tuple[float, float] | None = None,
    monitor: Any = None,
    device_size: tuple[int, int] | None = None,
    overlay_model_dir: Path | None = None,
    overlay_filter_settings: GF.FilterSettings | None = None,
    show_unfiltered_overlay: bool = False,
    allow_overwrite: bool = False,
) -> SessionResult:
    import gazefollower  # noqa: PLC0415  (initialises native components; unavoidable for the live path)
    from gazefollower import GazeFollower  # noqa: PLC0415
    from gazefollower.misc import DefaultConfig  # noqa: PLC0415

    result = SessionResult()
    session_stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = resolve_output_root(out_root)
    _ensure_gitignore(out_root)
    round_dir = out_root / f"round{round_id}"
    guard_existing_round(round_dir, allow_overwrite=allow_overwrite)

    if not dry_run:
        check_px2cm_against_library(rig)

    config = DefaultConfig()
    config.cali_mode = 9
    config.camera_position = (rig.camera_x_cm, rig.camera_y_cm)
    config.screen_physical_size = (rig.screen_w_cm, rig.screen_h_cm)
    # The library reads the PRIMARY monitor. When the experiment runs on
    # another display we override its idea of the screen, or every label would
    # be computed in the primary's pixel space.
    if device_size is not None:
        config.screen_size = np.array([device_size[0], device_size[1]])
    lib_w, lib_h = int(config.screen_size[0]), int(config.screen_size[1])
    if (lib_w, lib_h) != (rig.device_w_px, rig.device_h_px):
        raise RuntimeError(
            f"display {lib_w}x{lib_h} != --device-w/h {rig.device_w_px}x{rig.device_h_px}; "
            "labels would be in a different space than the targets"
        )

    # A horizontal band narrows the calibration grid too. The model's raw X
    # output goes flat beyond roughly a third of the screen from the centre,
    # so calibrating at the library's edge points trains on a signal that is
    # not there. The run is marked as not faithful to the library's protocol.
    grid_x = None if x_range is None else (x_range[0], 0.5, x_range[1])
    calibration_grid = GT.calibration_points(grid_x)

    specs: list[ProtocolSpec] = []
    for name in protocols:
        if name == "A":
            specs.append(protocol_a(grid_x))
        elif name == "TUNE":
            specs.append(protocol_timed("TUNE", targets_tune, C.DEFAULT_SETTLE_MS / 1000.0, C.DEFAULT_COLLECT_MS / 1000.0))
        elif name == "GRID16":
            specs.append(protocol_timed("GRID16", targets_grid16, C.DEFAULT_SETTLE_MS / 1000.0, C.DEFAULT_COLLECT_MS / 1000.0))
        elif name == "T1":
            specs.append(protocol_timed("T1", targets_t1, C.DEFAULT_SETTLE_MS / 1000.0, C.DEFAULT_COLLECT_MS / 1000.0))
        elif name in ("FULL", "MOVE"):
            # Generated for the display actually in use, so a change of monitor
            # cannot leave the targets describing the previous one.
            coverage = GD.coverage_targets(
                int(target_geometry["width_px"]),
                int(target_geometry["height_px"]),
                calibration=calibration_grid,
            )
            targets = [
                Target(int(t["index"]), str(t["name"]), float(t["screen_position"]["x"]), float(t["screen_position"]["y"]))
                for t in coverage
            ]
            specs.append(
                ProtocolSpec(
                    name,
                    targets,
                    "timed",
                    settle_s=C.DEFAULT_SETTLE_MS / 1000.0,
                    collect_s=C.DEFAULT_COLLECT_MS / 1000.0,
                    instruction=INSTRUCTIONS[name],
                )
            )
        elif name == "T2":
            specs.append(protocol_t2())
        else:
            raise ValueError(
                f"protocol {name!r} is not known "
                "(A, TUNE, GRID16, T1, T2 for phase 0; FULL, MOVE for the full-screen sessions)"
            )

    kwargs: dict[str, Any] = {"config": config, "calibration": make_pass_through_calibration()}
    if dry_run:
        camera, face_alignment, estimator = make_dry_run_components()
        kwargs.update(camera=camera, face_alignment=face_alignment, gaze_estimator=estimator)
    gf = GazeFollower(**kwargs)
    library_tmp = [str(getattr(gf, "_tmpSampleDataPath", ""))]

    base_meta = {
        "session": session_stamp,
        "rig": rig.to_dict(),
        "setup": None if manifest is None else manifest.to_dict(),
        "monitor": None if monitor is None else monitor.to_dict(),
        "target_geometry": dict(target_geometry),
        "library": {"name": "gazefollower", "version": getattr(gazefollower, "__version__", "1.0.2"), "screen_size": [lib_w, lib_h]},
        "versions": S.environment_versions(),
        "builder_version": H.BUILDER_VERSION,
        "head_names": list(H.HEAD6_NAMES),
        "dry_run": dry_run,
        "speed": speed,
        "background": "white",
        "region": (
            {"x_range": None, "faithful_to_library_calibration": True}
            if x_range is None
            else {
                "x_range": [x_range[0], x_range[1]],
                "faithful_to_library_calibration": False,
                "calibration_grid_x": list(grid_x),
                "why": "central-band experiment: the model's horizontal output saturates toward the screen edges",
            }
        ),
        "library_tmp_files": library_tmp,
        "expected_feature_dim": EXPECTED_FEATURE_DIM,
        "overlay_filter": None if overlay_filter_settings is None else asdict(overlay_filter_settings),
    }

    overlay_model = None
    if overlay_model_dir is not None:
        import gf_fit as FIT  # noqa: PLC0415 - only needed when an overlay is requested

        overlay_model = FIT.FittedModel.load(overlay_model_dir)
        print(f"overlay model loaded from {overlay_model_dir} (columns: {len(overlay_model.schema.columns)})")

    origin = (0, 0) if monitor is None else monitor.origin
    display = Display(
        lib_w,
        lib_h,
        headless=headless,
        origin=origin,
        overlay_available=overlay_model is not None,
        show_unfiltered_overlay=show_unfiltered_overlay,
    )
    runner_ref: dict[str, ProtocolRunner | None] = {"runner": None}

    def subscriber(face_info: Any, gaze_info: Any) -> None:
        runner = runner_ref["runner"]
        if runner is not None:
            runner.on_frame(face_info, gaze_info)

    gf.add_subscriber(subscriber)

    try:
        # camera.start_sampling() directly: GazeFollower.start_sampling() would
        # also subscribe its CSV writer, which raises on status-False frames.
        gf.camera.start_sampling()
        display.draw_message(["Camera warming up..."])
        _sleep_with_escape(display, CAMERA_WARMUP_S / speed)
        # No runner is installed yet, so nothing is recorded during this wait.
        if display.wait_for_key(
            [
                "Ready.",
                "",
                f"{len(specs)} protocols: {', '.join(s.name for s in specs)}",
                "",
                "Each one waits for you before it starts, so take the time to",
                "read what it asks for. Esc aborts at any point.",
            ]
        ) == "abort":
            result.aborted = True
            specs = []

        for spec in specs:
            # Proximity to the nearest calibration point rides with every
            # target, so a reader never has to recompute which of them were
            # measured beside a training point.
            annotated = GT.annotate_targets(
                spec.exported_targets(),
                int(target_geometry["width_px"]),
                int(target_geometry["height_px"]),
                calibration=calibration_grid,
            )
            for entry in annotated:
                entry.setdefault(
                    "region",
                    GD.region_of(entry["screen_position"]["x"], entry["screen_position"]["y"]),
                )
            meta = dict(
                base_meta,
                targets=annotated,
                target_proximity=GT.proximity_summary(annotated) if annotated else None,
                region_counts=GD.region_summary(annotated) if annotated else None,
                protocol_kind=spec.kind,
                instruction=spec.instruction,
            )
            builder = S.RecordingBuilder(spec.name, round_id, meta)
            runner = ProtocolRunner(
                spec,
                builder,
                rig,
                speed=speed,
                overlay_model=overlay_model,
                overlay_filter_settings=overlay_filter_settings,
            )
            remaining = [s.name for s in specs[specs.index(spec) :]]
            # The wait is unbounded by design, so the recorder must NOT be
            # listening during it. The camera keeps running -- the tracker
            # needs to stay locked on -- but with no runner installed the
            # subscriber discards those frames instead of storing a face
            # embedding for every one of them for as long as the person is
            # away from the desk.
            if display.wait_for_key(
                [
                    f"{spec.name}   ({specs.index(spec) + 1} of {len(specs)})",
                    "",
                    spec.instruction,
                    "",
                    f"{len(spec.exported_targets())} targets, about {_estimate_seconds(spec):.0f} seconds",
                    f"still to come: {', '.join(remaining[1:]) or 'nothing, this is the last one'}",
                ]
            ) == "abort":
                result.aborted = True
                break
            runner_ref["runner"] = runner
            runner.start()
            # Armed here, not before the wait: the clock for "frames have
            # stopped" starts when this protocol does.
            dog = FrameWatchdog(limit_s=WATCHDOG_S, started_s=time.monotonic())
            display.play_beep()
            last_target: Target | None = runner.current_target()
            watchdog = False
            while not runner.finished:
                if display.poll_escape():
                    result.aborted = True
                    break
                if runner.failed:
                    result.aborted = True
                    result.failure = f"{runner.consecutive_errors} consecutive subscriber errors: {runner.last_error}"
                    print(f"ABORTED: {result.failure}")
                    break
                if runner.no_face_stall:
                    result.aborted = True
                    result.failure = (
                        f"no face detected for {runner.consecutive_invalid} consecutive frames "
                        "-- check the camera is uncovered, the room is lit, you are in shot, "
                        "and no other application is holding the camera"
                    )
                    print(f"ABORTED: {result.failure}")
                    break
                state = runner.state
                now = time.monotonic()
                dog.note(runner._frame_times)
                if dog.tripped(now):
                    watchdog = True
                    result.watchdog_tripped = True
                    result.failure = (
                        f"no camera frame for {dog.silent_for(now):.1f}s during {spec.name}"
                    )
                    print(f"ABORTED: {result.failure}")
                    break
                if state.target is not last_target:
                    last_target = state.target
                    display.play_beep()
                hud = [
                    f"{spec.name}  target {state.target_pos}/{state.target_count}  {state.phase}",
                    f"fps {state.fps:.1f}" if state.fps else "fps --",
                    f"head {'ok' if state.head_valid else 'INVALID'}" + (f"  pitch_a {state.pitch_a:+.3f}" if state.pitch_a is not None else ""),
                    f"errors {runner.errors}" + (f"  LAST: {runner.last_error[:60]}" if runner.last_error else ""),
                    (
                        f"blue filter {overlay_filter_settings.kind.value}"
                        if overlay_model is not None and overlay_filter_settings is not None
                        else "blue filter --"
                    ),
                    "Esc to abort",
                ]
                display.draw_target(state, hud)
                time.sleep(0.01)
            runner_ref["runner"] = None
            # A callback that already read runner_ref is still inside on_frame;
            # let it finish, then freeze under the runner's own lock so the row
            # list cannot grow while it is being turned into arrays.
            time.sleep(DRAIN_S)
            with runner.lock:
                rec = runner.builder.freeze()
            rec.meta["integrity"] = runner.integrity(watchdog_tripped=watchdog, aborted=result.aborted)
            rec.meta["integrity"]["failure"] = result.failure
            rec.meta["integrity"]["fps_median"] = runner.fps_median()
            npz_path, _ = rec.save(round_dir)
            result.recordings[spec.name] = npz_path
            dim = runner.feature_dim
            if dim is not None and dim != EXPECTED_FEATURE_DIM:
                print(f"WARNING: feature dim {dim} != expected {EXPECTED_FEATURE_DIM}")
            print(_summary_line(spec.name, rec))
            if result.aborted or watchdog:
                break
        if not result.aborted and not result.watchdog_tripped:
            runner_ref["runner"] = None  # nothing recorded while this sits on screen
            display.wait_for_key(
                [
                    "Done. Thank you.",
                    "",
                    f"Recorded: {', '.join(result.recordings)}",
                ],
                timeout_s=60.0,
            )
    finally:
        # stop_sampling() reaches the library's unbounded join; bound it too.
        result.shutdown = {"stop_sampling": _call_with_timeout(gf.camera.stop_sampling, SHUTDOWN_STEP_TIMEOUT_S)}
        result.shutdown.update(shutdown_library(gf))
        display.close()
    return result


def _summary_line(name: str, rec: S.Recording) -> str:
    integ = rec.meta.get("integrity", {})
    n_acc = int(np.sum(rec.rows_accepted()))
    n_col = int(np.sum(rec.rows_collecting()))
    gaze_rows = np.asarray(rec.gaze_status, dtype=bool)
    head_invalid = float(np.mean(~rec.rows_head_valid()[gaze_rows])) if np.any(gaze_rows) else float("nan")
    return (
        f"{name}: rows {rec.n_rows}, accepted {n_acc}, collecting {n_col}, fps {integ.get('fps_median')}, "
        f"errors {integ.get('subscriber_errors')}, head-invalid {100 * head_invalid:.1f}%, "
        f"aborted {integ.get('aborted')}, watchdog {integ.get('watchdog_tripped')}"
    )


def _estimate_seconds(spec: ProtocolSpec) -> float:
    """Roughly how long a protocol takes, for the "before you start" screen.

    Calibration points run until 45 frames are ACCEPTED, so their duration
    depends on tracking; the estimate assumes a healthy 30 fps and is labelled
    "about" on screen for that reason.
    """

    stored = len(spec.exported_targets())
    if spec.kind == "calibration":
        per_point = C.PREPARE_S + C.N_FRAMES_PER_POINT / 30.0 + C.WAIT_S
        return (stored + 1) * per_point  # + the unstored warm-up point
    return stored * (spec.settle_s + spec.collect_s)


def _sleep_with_escape(display: Display, seconds: float) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if display.poll_escape():
            return True
        time.sleep(0.02)
    return False


def resolve_output_root(candidate: Path) -> Path:
    """Refuse to write embeddings anywhere but the approved directory.

    Recording rows contain the 258-d model embedding of the face and eye
    crops. The privacy rule for this experiment is that they live under
    ``gazefollower_eval/recordings/`` and are deleted by ``gf_purge.py``; a
    ``--out`` pointing elsewhere -- the project repo, say -- would put them
    outside the purge scope, so the invariant is enforced here rather than
    trusted to the caller.
    """

    root = Path(candidate).resolve()
    allowed = RECORDINGS_DIR.resolve()
    if root != allowed and allowed not in root.parents:
        raise ValueError(
            f"refusing to write recordings to {root}: embeddings must stay under {allowed}"
        )
    return root


def guard_existing_round(round_dir: Path, *, allow_overwrite: bool = False) -> None:
    """Refuse to record over a round that already holds data.

    A recording is minutes of the operator's time and cannot be reproduced --
    the conditions have moved on by the time anyone notices. Re-running with a
    round number that was already used silently replaced round9's good 903
    frames with an aborted 109-frame capture, and the original was gone. The
    number is easy to repeat by editing one digit of a previous command, so
    the recorder refuses rather than trusting the operator to remember which
    numbers are taken.
    """

    if allow_overwrite or not round_dir.exists():
        return
    existing = sorted(p.name for p in round_dir.glob("*.npz") if _holds_usable_rows(p))
    if not existing:
        # An aborted attempt that captured no usable gaze leaves a stub behind.
        # Protecting that stub would lock the operator out of the round number
        # they just tried, which is exactly when they want to retry it.
        return
    raise SystemExit(
        f"refusing to overwrite {round_dir}: it already holds {', '.join(existing)}.\n"
        "Recordings cannot be reproduced -- the session conditions are gone.\n"
        "Use a round number that is not taken, or pass --overwrite-round if you "
        "genuinely mean to discard the existing data."
    )


def _holds_usable_rows(npz_path: Path) -> bool:
    """Whether a saved protocol carries gaze worth protecting.

    Reads the sibling metadata rather than the archive: the question is only
    whether anything was captured, and the metadata answers it without
    touching the embeddings. An unreadable or absent manifest is treated as
    usable, so a parsing problem never silently clears real data.
    """

    meta_path = npz_path.with_suffix("").with_suffix(".meta.json")
    if not meta_path.exists():
        meta_path = npz_path.parent / f"{npz_path.stem}.meta.json"
    if not meta_path.exists():
        return True
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True
    if not meta.get("feature_dim"):
        return False
    valid = meta.get("integrity", {}).get("valid_gaze_frames")
    return valid is None or valid > 0


def _ensure_gitignore(root: Path) -> None:
    """Create the ignore file BEFORE the first sensitive write.

    The eval directory sits under a user-home git root, so an unguarded
    ``git add`` from there would otherwise sweep the recordings in.
    """

    root.mkdir(parents=True, exist_ok=True)
    gi = root / ".gitignore"
    if not gi.exists():
        gi.write_text("*\n!.gitignore\n", encoding="utf-8")


# --- CLI ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round", type=int, required=True, help="recording round id (0 = Phase 0)")
    parser.add_argument("--protocols", default="A,TUNE,GRID16,T1,T2", help="comma-separated, in order")
    parser.add_argument("--camera-x-cm", type=float, required=True)
    parser.add_argument("--camera-y-cm", type=float, required=True, help="down from the top-left corner; > screen height = below the screen")
    parser.add_argument("--screen-width-cm", type=float, required=True)
    parser.add_argument("--screen-height-cm", type=float, required=True)
    parser.add_argument("--device-w", type=int, default=None, help="display width in device px (default: taken from --monitor)")
    parser.add_argument("--device-h", type=int, default=None)
    parser.add_argument("--monitor", default=None, help="which display: an index, or part of its name (default: primary)")
    parser.add_argument("--list-monitors", action="store_true", help="print the displays the OS reports and exit")
    parser.add_argument("--viewing-distance-cm", type=float, default=None, help="eye to screen centre; distinct from the camera distance")
    parser.add_argument("--dpi-scale", type=float, default=1.0, help="OS scaling of the selected display (logical px = device px / scale)")
    parser.add_argument("--preset", default=None, help="calibration preset intended for this run; recorded in the manifest")
    parser.add_argument(
        "--overlay-model",
        type=Path,
        default=None,
        help=(
            "directory of a model saved by gf_fit.py (schema.json + svr_x.xml/svr_y.xml or ridge.npz), "
            "from an EARLIER round. Its live predictions are drawn as a moving blue crosshair on every "
            "frame of every protocol, so calibration accuracy can be watched, not just read afterwards. "
            "The raw (uncalibrated) model output is always shown as an orange dot, with or without this."
        ),
    )
    parser.add_argument(
        "--overlay-filter",
        choices=[kind.value for kind in GF.FilterKind],
        default=GF.FilterKind.ONE_EURO.value,
        help="temporal filter for the calibrated blue point (default: one-euro; off preserves the old behaviour)",
    )
    parser.add_argument("--overlay-show-unfiltered", action="store_true", help="also draw the unfiltered calibrated prediction in purple for diagnosis")
    parser.add_argument("--overlay-reset-gap-ms", type=float, default=250.0, help="reset filter history after this gap")
    parser.add_argument("--overlay-ema-cutoff-hz", type=float, default=2.0)
    parser.add_argument("--overlay-one-euro-min-cutoff-hz", type=float, default=1.2)
    parser.add_argument(
        "--overlay-one-euro-beta",
        type=float,
        default=0.0005,
        help="One Euro speed response in Hz per (pixel/second)",
    )
    parser.add_argument("--overlay-one-euro-derivative-cutoff-hz", type=float, default=1.0)
    parser.add_argument("--overlay-kalman-acceleration-noise", type=float, default=400.0, help="px^2/s^4")
    parser.add_argument("--overlay-kalman-measurement-noise", type=float, default=900.0, help="px^2")
    parser.add_argument("--targets", type=Path, default=PACKAGE_DIR / "targets.json", help="T1 targets (analyze.py geometry)")
    parser.add_argument("--targets-tune", type=Path, default=PACKAGE_DIR / "targets_tune.json")
    parser.add_argument("--targets-grid16", type=Path, default=PACKAGE_DIR / "targets_grid16.json")
    parser.add_argument("--eye-distance-cm", type=float, default=None, help="operator-measured eye-to-screen distance")
    parser.add_argument("--glasses", default=None, help="none | glasses | contacts")
    parser.add_argument("--lighting", default=None, help="short description, kept constant across sessions")
    parser.add_argument("--note", default=None)
    parser.add_argument("--skip-camera-probe", action="store_true", help="do not measure the camera's actual frame rate")
    parser.add_argument(
        "--x-range",
        nargs=2,
        type=float,
        metavar=("LO", "HI"),
        default=None,
        help=(
            "run everything inside this horizontal band (normalised, e.g. 0.3 0.7): the calibration "
            "grid narrows to LO/0.5/HI and the target files default to the *_central.json ones. "
            "Not a faithful reproduction of the library's calibration; recorded as such"
        ),
    )
    parser.add_argument("--out", type=Path, default=RECORDINGS_DIR)
    parser.add_argument(
        "--overwrite-round",
        action="store_true",
        help="discard an existing recording for this round id (refused by default)",
    )
    parser.add_argument("--dry-run", action="store_true", help="fake camera/face/model through the real process_frame")
    parser.add_argument("--headless", action="store_true", help="no window (dry-run only)")
    parser.add_argument("--speed", type=float, default=1.0, help="time scale for dry runs (10 = ten times faster)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Answered before parsing: listing displays is a question about the
    # machine, not about a run, and must not demand a run's arguments.
    if "--list-monitors" in (sys.argv[1:] if argv is None else list(argv)):
        print(GD.describe_monitors())
        return 0
    args = build_parser().parse_args(argv)
    if args.headless and not args.dry_run:
        print("--headless requires --dry-run")
        return 2
    if args.speed != 1.0 and not args.dry_run:
        print("--speed is for dry runs only: live protocol timing must match the library's")
        return 2
    try:
        monitor = GD.pick_monitor(args.monitor)
    except (ValueError, RuntimeError) as exc:
        print(f"{exc}\n\n{GD.describe_monitors()}")
        return 2
    device_w = args.device_w if args.device_w is not None else monitor.width_px
    device_h = args.device_h if args.device_h is not None else monitor.height_px
    print(f"display [{monitor.index}] {monitor.name}: {device_w}x{device_h} px at desktop {monitor.origin}"
          f"{' (PRIMARY)' if monitor.is_primary else ''}")
    rig = C.RigGeometry(args.camera_x_cm, args.camera_y_cm, args.screen_width_cm, args.screen_height_cm, device_w, device_h)
    try:
        overlay_filter_settings = GF.FilterSettings(
            width_px=device_w,
            height_px=device_h,
            kind=GF.FilterKind(args.overlay_filter),
            reset_gap_ms=args.overlay_reset_gap_ms,
            ema_cutoff_hz=args.overlay_ema_cutoff_hz,
            one_euro_min_cutoff_hz=args.overlay_one_euro_min_cutoff_hz,
            one_euro_beta_hz_per_px_s=args.overlay_one_euro_beta,
            one_euro_derivative_cutoff_hz=args.overlay_one_euro_derivative_cutoff_hz,
            kalman_acceleration_noise_px2_s4=args.overlay_kalman_acceleration_noise,
            kalman_measurement_noise_px2=args.overlay_kalman_measurement_noise,
        )
    except ValueError as exc:
        print(f"invalid overlay filter settings: {exc}")
        return 2
    if monitor.width_mm:
        for label, declared, reported in (("width", args.screen_width_cm, monitor.width_mm / 10.0),
                                          ("height", args.screen_height_cm, monitor.height_mm / 10.0)):
            if abs(declared - reported) > 2.0:
                print(f"SETUP WARNING: declared screen {label} {declared} cm vs OS-reported {reported:.1f} cm")
    if args.viewing_distance_cm:
        geom = GD.ViewingGeometry(args.screen_width_cm, args.screen_height_cm, device_w, device_h,
                                  args.viewing_distance_cm, args.eye_distance_cm)
        h, v = geom.half_angles_deg()
        print(f"viewing distance {args.viewing_distance_cm} cm -> half-angles h +/-{h:.1f} deg, v +/-{v:.1f} deg; "
              f"1.5 deg = {geom.deg_to_px(1.5):.0f} px")
    protocols = [p.strip() for p in args.protocols.split(",") if p.strip()]
    x_range: tuple[float, float] | None = None
    if args.x_range is not None:
        lo, hi = args.x_range
        if not (0.0 <= lo < 0.5 < hi <= 1.0):
            print("--x-range must satisfy 0 <= LO < 0.5 < HI <= 1")
            return 2
        x_range = (lo, hi)
        # Unless the operator pointed at specific files, use the central ones.
        if args.targets == PACKAGE_DIR / "targets.json":
            args.targets = PACKAGE_DIR / "targets_central.json"
        if args.targets_tune == PACKAGE_DIR / "targets_tune.json":
            args.targets_tune = PACKAGE_DIR / "targets_tune_central.json"
        if "GRID16" in protocols:
            print("GRID16 spans the full width and is dropped for a banded run")
            protocols = [p for p in protocols if p != "GRID16"]
        print(f"horizontal band {lo:.2f}..{hi:.2f}: calibration grid x = {lo:.2f}/0.50/{hi:.2f} (NOT the library's grid)")
    generated = bool({"FULL", "MOVE"} & set(protocols))
    if generated:
        # FULL/MOVE generate their own targets, so the geometry comes from the
        # display actually being driven -- including a --device-w/h override,
        # which is what a labelled simulation on another screen uses. A stale
        # targets.json describing a different monitor must not decide it.
        geometry = {
            "screen_id": monitor.name,
            "width_px": device_w,
            "height_px": device_h,
            "dpi_scale": args.dpi_scale,
            "orientation": "LANDSCAPE" if device_w >= device_h else "PORTRAIT",
        }
    else:
        _, geometry = C.load_targets(args.targets)
    # Every target file must describe the same screen, or the sets are not
    # scored on one ruler.
    checked = () if generated else (("targets-tune", args.targets_tune), ("targets-grid16", args.targets_grid16))
    for label, path in checked:
        if label == "targets-grid16" and "GRID16" not in protocols and not path.exists():
            continue
        if not path.exists():
            print(f"{path} is missing; generate it with: python gf_targets.py")
            return 2
        _, other = C.load_targets(path)
        if other != geometry:
            print(f"{path.name} carries a different screen geometry than {args.targets.name}")
            return 2

    # Print the geometry BEFORE anything that can block. The camera probe used
    # to run first and silently, so a camera that had dropped to a low frame
    # rate made the whole program look hung.
    print(
        f"camera at ({rig.camera_x_cm:.1f}, {rig.camera_y_cm:.1f}) cm -- "
        f"{'BELOW the screen' if rig.camera_below_screen else 'above or within the screen'}"
    )
    print(
        f"screen {rig.screen_w_cm} x {rig.screen_h_cm} cm, device {rig.device_w_px}x{rig.device_h_px}, "
        f"targets {geometry['width_px']}x{geometry['height_px']} logical"
    )
    print(f"protocols: {', '.join(protocols)}")
    probe = not (args.skip_camera_probe or args.dry_run)
    if probe:
        print("measuring the camera's actual frame rate (a few seconds)...", flush=True)
    manifest = GS.build_manifest(
        rig,
        declared_extra={
            "eye_distance_cm": args.eye_distance_cm,
            "glasses": args.glasses,
            "lighting": args.lighting,
            "operator_note": args.note,
            "viewing_distance_cm": args.viewing_distance_cm,
            "monitor": monitor.to_dict(),
            "preset_intended": args.preset,
        },
        skip_camera=not probe,
    )
    camera = manifest.measured.get("camera", {})
    if probe:
        verdict = camera_verdict(camera)
        print(verdict.message)
        if verdict.fatal:
            print(f"\nNot recording: {verdict.reason}")
            return 2
        if verdict.needs_confirmation and not confirm_low_frame_rate():
            return 2
    for warning in manifest.warnings:
        print(f"SETUP WARNING: {warning}")
    previous = args.out / f"round{args.round}" / "setup.json"
    if previous.exists():
        comparison = GS.compare_manifests(GS.SetupManifest.load(previous), manifest)
        if not comparison["comparable"]:
            print("SETUP CHANGED since the previous recording in this round:")
            for change in comparison["critical_changes"]:
                print(f"  {change['field']}: {change['before']} -> {change['after']}")
    if args.eye_distance_cm is None:
        print("NOTE: no --eye-distance-cm given; angular error will be unavailable for this session")
    result = run_session(
        protocols=protocols,
        round_id=args.round,
        rig=rig,
        targets_t1=args.targets,
        targets_tune=args.targets_tune,
        targets_grid16=args.targets_grid16,
        target_geometry=geometry,
        out_root=args.out,
        dry_run=args.dry_run,
        headless=args.headless,
        speed=args.speed,
        manifest=manifest,
        x_range=x_range,
        monitor=monitor,
        device_size=(device_w, device_h),
        allow_overwrite=args.overwrite_round,
        overlay_model_dir=args.overlay_model,
        overlay_filter_settings=overlay_filter_settings,
        show_unfiltered_overlay=args.overlay_show_unfiltered,
    )
    if result.recordings:
        manifest.save(args.out / f"round{args.round}" / "setup.json")
    print(
        json.dumps(
            {
                "recordings": {k: str(v) for k, v in result.recordings.items()},
                "aborted": result.aborted,
                "watchdog_tripped": result.watchdog_tripped,
                "failure": result.failure,
                "shutdown": result.shutdown,
            },
            indent=2,
        )
    )
    return 0 if (result.recordings and not result.aborted and not result.watchdog_tripped) else 1


if __name__ == "__main__":
    raise SystemExit(main())
