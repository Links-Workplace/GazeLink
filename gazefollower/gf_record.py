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
``--no-save`` runs the same protocols and overlay but writes nothing at all,
for showing the model to someone without spending a round number on it.

``gazefollower`` is imported only inside the functions that need it (its
import initialises native components), so the protocol logic here is unit
tested with fakes and no camera.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import random
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_capture as CAP  # noqa: E402
import gf_common as C  # noqa: E402
import gf_head_features as H  # noqa: E402
import gf_display as GD  # noqa: E402
import gf_gaze_filter as GF  # noqa: E402
import gf_setup as GS  # noqa: E402
import gf_targets as GT  # noqa: E402
import gf_schema as S  # noqa: E402
from gazelink_core.domain.observation import FrameObservation, HeadPolicy  # noqa: E402
from gazelink_core.gaze import sample_gate as GATE  # noqa: E402
from gazelink_core.tracking import face_landmarks as FL  # noqa: E402
from gazelink_core.tracking.gazefollower_dry_run import make_dry_run_components  # noqa: E402,F401
from gazelink_core.tracking import gazefollower_source as SRC  # noqa: E402

# Live infrastructure that used to be defined here now lives in the core and is
# shared with the live session (ARCH-01 stage D). Re-exported so every existing
# caller and patch keeps the names it used.
from gazelink_core.gaze.prediction import predict_one, predict_overlay_point  # noqa: E402,F401
from gazelink_core.gaze.preflight import (  # noqa: E402,F401
    PREFLIGHT_MAX_FRAMES,
    PREFLIGHT_MIN_ACTIVATION,
    preflight_verdict,
)
from gazelink_core.tracking.gazefollower_source import (  # noqa: E402,F401
    design_row_from_library as _overlay_design_row,
)
from gazelink_core.gaze.visibility import (  # noqa: E402,F401
    OVERLAY_STALE_S,
    visible_overlay_point,
    visible_point,
)
from gazelink_core.tracking.gazefollower_library import (  # noqa: E402,F401
    CAMERA_WARMUP_S,
    SHUTDOWN_STEP_TIMEOUT_S,
    _call_with_timeout,
    _library_dir,
    make_pass_through_calibration,
    shutdown_library,
)
from gazelink_core.ui.prompt_layout import (  # noqa: E402,F401
    PROMPT_ANCHORS,
    PROMPT_PITCH_PX,
    PROMPT_SIDE_BAND,
    PROMPT_SIDE_LINE_GAP_PX,
    PROMPT_TOP_PX,
    prompt_button_gaps,
    prompt_layout,
)
from gazelink_core.calibration import correction as CORR  # noqa: E402
from gazelink_core.calibration import poses as POSES  # noqa: E402
from gazelink_core.ui.pygame_display import (  # noqa: E402,F401
    _HEBREW_RANGE,
    HEBREW_FONTS,
    Display,
    _sleep_with_escape,
    has_hebrew,
    hebrew_font,
    rtl,
)

PACKAGE_DIR = Path(__file__).resolve().parent
RECORDINGS_DIR = PACKAGE_DIR / "recordings"

# A --no-save run writes nothing, so it spends no round number and the
# operator has none to invent. The id below only labels in-memory metadata; it
# is negative so a demo can never be mistaken for round 0 wherever it surfaces.
DEMO_ROUND_ID = -1

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
# After the last target, give any callback already inside the subscriber time
# to finish before the row list is frozen.
DRAIN_S = 0.25
# Protocols wait for a key press rather than a countdown: a three-second
# countdown did not leave time to read the instruction.
EXPECTED_FEATURE_DIM = 258

T2_TARGETS: tuple[tuple[float, float], ...] = ((0.5, 0.10), (0.5, 0.50), (0.5, 0.90))
T2_SECONDS = 20.0
T2_SETTLE_S = 1.5

# T3 holds each held-out target long enough to separate ARRIVING at it from
# STAYING on it. Measured on round34/T1, the raw prediction was still closing
# on the target at 2500-3000 ms after onset (median 2469 ms to settle within
# 200 px, against 546 ms on round19), so a 3.0 s presentation ends while the
# approach is still happening and the scored window carries it. 4.0 s leaves
# a second of settled hold after the slowest approach seen so far.
#
# Same targets and same 1.5 s settle as T1 on purpose: only the hold changes,
# so the two are comparable frame for frame over their shared first 3 s.
T3_SETTLE_S = C.DEFAULT_SETTLE_MS / 1000.0
T3_COLLECT_S = 2.5

INSTRUCTIONS = {
    "A": "Calibration. Keep your head STILL and look at each dot.",
    "FULL": "Full-screen targets: centre, sides, top, bottom and corners. Head still.",
    "MOVE": "Same targets. Sit naturally and let your head move a little as you normally would.",
    "TUNE": "Tuning targets. Keep your head still and look at each dot.",
    "GRID16": "16-point grid. Keep your head still and look at each dot.",
    "T1": "Held-out targets. Keep your head still and look at each dot.",
    "T2": "Keep your EYES on the dot and NOD your head SLOWLY up and down.",
    "T3": "Held-out targets, held longer. Keep your head still and look at each dot.",
    "B": "Same targets, several head positions. Follow the pose prompt before each block.",
}


# Pose blocks for protocol B, in presentation order.  Two properties matter
# and both come from the ORDER, not from any single pose:
#
#   * The centre pose recurs between every other pose, so "which pose" and
#     "how far into the session" are not the same thing.  Without the returns,
#     the last pose would also be the most fatigued and the two could not be
#     told apart -- the same confound that made the elapsed-time reading
#     unresolvable in round19..round26.
#   * Every block shows the SAME targets.  Pose therefore carries no
#     information about which target is on screen, which is the confound that
#     made head columns useless when fitted on protocol A: there pitch_a
#     correlated with target_y at r=+0.871, so the head numbers could be
#     learned as a shortcut to the answer instead of as a correction.
#
# Pose sequences for protocol B live in the core: which poses, in which order,
# and what the operator is told to do is a decision about the experiment, not
# about camera handling or file layout, and it is testable without hardware.
# The names below are kept so anything importing them keeps working.
POSE_SEQUENCE: tuple[tuple[str, str], ...] = POSES.STANDARD.as_tuples()
POSE_SEQUENCE_EXTENDED: tuple[tuple[str, str], ...] = POSES.EXTENDED.as_tuples()
POSE_SETS = POSES.POSE_SETS
DEFAULT_POSE_SET = POSES.DEFAULT_POSE_SET
resolve_poses = POSES.resolve_poses


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

    def observe(
        self, now_s: float, gaze_status: bool, left_openness: float, right_openness: float
    ) -> tuple[str, bool]:
        if self.done:
            return (S.PHASE_WAIT if self.stored else S.PHASE_WARMUP), False
        if now_s - self.onset_s < self.prepare_s:
            return (S.PHASE_PREPARE if self.stored else S.PHASE_WARMUP), False
        accepted = False
        if self.n_accepted < self.n_frames:
            ok = (
                bool(gaze_status)
                and left_openness > self.blink_threshold
                and right_openness > self.blink_threshold
            )
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
    # True when every block opens with a pose the operator must take up. The
    # runner prompts before the FIRST block too, not only on a change: in the
    # supplement set block one carries the actual instruction, and it was
    # silently never shown (round9, aborted after 24 rows).
    uses_poses: bool = False

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


def protocol_b(
    grid_x: tuple[float, float, float] | None = None,
    *,
    n_blocks: int = len(POSE_SEQUENCE),
    order_seed: int = 0,
) -> ProtocolSpec:
    """Multi-pose calibration: the A grid, repeated once per head pose.

    Protocol A holds the head still, so a model fitted on it has never seen
    the same gaze from two different poses and cannot learn to separate the
    two.  Measured on round18/A the entire yaw range was 0.060 wide while the
    drift observed between later sessions reached 0.061 -- as wide as
    everything the model was ever shown.  B exists to supply the missing
    examples: the same target from several poses, so gaze and pose vary
    independently.

    Targets keep their ids across blocks; ``block`` says which pose a row
    came from, and the schema already carries it.  The order is reshuffled
    per block so that target order and pose are not locked together either.
    """

    # The upper bound belongs to resolve_poses, which knows which pose set is
    # running; this only refuses a sequence with no blocks in it at all.
    if n_blocks < 1:
        raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")
    sequence = C.NINE_POINT_SEQUENCE if grid_x is None else C.nine_point_sequence(grid_x)
    points = list(sequence[1:])
    targets: list[Target] = []
    for block in range(n_blocks):
        # A warm-up opens every block, not only the session: the pose has just
        # changed and the tracker needs the same settling it gets at the start.
        targets.append(Target(-1, f"WARMUP_B{block}", *sequence[0], block=block))
        ordered = list(enumerate(points))
        random.Random(order_seed + block).shuffle(ordered)
        targets += [Target(i, f"CAL_{i}", x, y, block=block) for i, (x, y) in ordered]
    return ProtocolSpec(
        "B", targets, "calibration", instruction=INSTRUCTIONS["B"], uses_poses=True
    )


def protocol_timed(
    name: str,
    targets_file: Path,
    settle_s: float,
    collect_s: float,
    *,
    order_seed: int | None = None,
    repeat_first: int = 0,
) -> ProtocolSpec:
    raw, _ = C.load_targets(targets_file)
    targets = [
        Target(
            int(t["index"]),
            str(t["name"]),
            float(t["screen_position"]["x"]),
            float(t["screen_position"]["y"]),
        )
        for t in raw
    ]
    return ProtocolSpec(
        name,
        order_targets(targets, order_seed=order_seed, repeat_first=repeat_first),
        "timed",
        settle_s=settle_s,
        collect_s=collect_s,
        instruction=INSTRUCTIONS[name],
    )


def order_targets(
    targets: Sequence[Target],
    *,
    order_seed: int | None = None,
    repeat_first: int = 0,
) -> list[Target]:
    """Presentation order, optionally shuffled and with an anchor repeat.

    Every T1 recording so far showed targets 0..9 in that same fixed order, so
    "later in the run" and "a different place on the screen" were the same
    thing and no measurement could separate them. Measured over six runs, 53%
    of the horizontal error variance sits in that confounded term, which is
    why it matters.

    ``order_seed`` shuffles the order, seeded so it is reproducible and
    recorded in the manifest. Two runs with different seeds show whether the
    error travels with the position or stays with the elapsed time. This is
    the same device ``gf_targets.build_grid16`` already uses, for the same
    stated reason.

    ``repeat_first`` re-shows the first N targets again at the end, keeping
    their ids. That gives the same screen positions at two separated times
    inside ONE run, so position and time are separated without comparing runs
    at all -- a stronger design, because nothing else about the session can
    differ between the two visits.

    Ids are preserved on the repeat: the id names the position, and the two
    visits are told apart by presentation order, which the recording already
    carries as separate contiguous segments.
    """

    ordered = list(targets)
    if order_seed is not None:
        random.Random(order_seed).shuffle(ordered)
    if repeat_first < 0:
        raise ValueError("repeat_first must not be negative")
    if repeat_first > len(ordered):
        raise ValueError(
            f"repeat_first {repeat_first} exceeds the {len(ordered)} targets available"
        )
    return ordered + ordered[:repeat_first]


def protocol_t2() -> ProtocolSpec:
    targets = [Target(i, f"T2_{i}", x, y) for i, (x, y) in enumerate(T2_TARGETS)]
    return ProtocolSpec(
        "T2",
        targets,
        "timed",
        settle_s=T2_SETTLE_S,
        collect_s=T2_SECONDS - T2_SETTLE_S,
        instruction=INSTRUCTIONS["T2"],
    )


def protocol_t3(targets: Path) -> ProtocolSpec:
    """T1's targets, held long enough to see the approach finish."""

    loaded, _ = C.load_targets(targets)
    return ProtocolSpec(
        "T3",
        [
            Target(
                int(t["index"]),
                str(t["name"]),
                float(t["screen_position"]["x"]),
                float(t["screen_position"]["y"]),
            )
            for t in loaded
        ],
        "timed",
        settle_s=T3_SETTLE_S,
        collect_s=T3_COLLECT_S,
        instruction=INSTRUCTIONS["T3"],
    )


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
        head_builder: Callable[[Any], np.ndarray | None] = FL.head6_from_face,
        pnp: Callable[[Any], tuple[float, float, float] | None] = FL.pnp_from_face,
        overlay_model: Any = None,
        overlay_model_y: Any = None,
        overlay_filter_settings: GF.FilterSettings | None = None,
        shadow: Callable[[int], Any] | None = None,
        shadow_builder: S.RecordingBuilder | None = None,
    ) -> None:
        self.spec = spec
        # The second pipeline of a dual-resolution session (gf_capture). It
        # gets EXACTLY the rows the primary gets -- same frame, target, phase
        # and timing -- so the two recordings pair row for row; only what the
        # pipeline produced differs.
        if (shadow is None) != (shadow_builder is None):
            raise ValueError("shadow and shadow_builder come together or not at all")
        self.shadow = shadow
        self.shadow_builder = shadow_builder
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
        # Optional second model supplying only the vertical axis; see
        # _predict_overlay for why the axes come from different models.
        self.overlay_model_y = overlay_model_y
        self.overlay_filter = (
            None
            if overlay_model is None or overlay_filter_settings is None
            else GF.GazePointFilter(overlay_filter_settings)
        )
        self.lock = threading.Lock()
        # Set by the camera thread when a pose block boundary is reached,
        # cleared by the display thread once the operator has moved and
        # confirmed.  While it is set no gate is armed, so no rows are stored.
        self.awaiting_block: int | None = None
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

    def resume_block(self) -> None:
        """Arm the target the pose prompt was holding, timed from now.

        The onset must be taken here and not when the boundary was reached:
        the elapsed time in between is the operator moving their head, and
        counting it toward the settle window would score frames recorded
        mid-movement as if the pose were already held.
        """

        with self.lock:
            if self.awaiting_block is None:
                return
            self.awaiting_block = None
            target = self.current_target()
            if target is None:
                return
            now_s = self.clock()
            self._onset_s = now_s
            self._gate = CollectionGate(
                onset_s=now_s,
                stored=target.index >= 0,
                prepare_s=C.PREPARE_S / self.speed,
                wait_s=C.WAIT_S / self.speed,
            )

    def _advance(self, now_s: float) -> None:
        previous = self.current_target()
        self._index += 1
        if self._index >= len(self.spec.targets):
            self._gate = None
            self.finished = True
            return
        target = self.spec.targets[self._index]
        first_block = previous is None and self.spec.uses_poses
        if first_block or (previous is not None and target.block != previous.block):
            # A pose change needs the person to move, which cannot be timed
            # from here.  Park with no gate -- _on_frame already treats a
            # missing gate as idle, so nothing is stored and no embedding is
            # kept -- and let the display thread prompt and resume.
            self.awaiting_block = target.block
            self._gate = None
            self._onset_s = None
            return
        self._onset_s = now_s
        if self.spec.kind == "calibration":
            self._gate = CollectionGate(
                onset_s=now_s,
                stored=target.index >= 0,
                prepare_s=C.PREPARE_S / self.speed,
                wait_s=C.WAIT_S / self.speed,
            )
        else:
            self._gate = TimedGate(
                now_s, self.spec.settle_s / self.speed, self.spec.collect_s / self.speed
            )

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

    def on_observation(self, obs: FrameObservation) -> None:
        """The same per-frame step, for a source that already translated the frame."""

        try:
            self._on_observation(obs)
        except Exception as exc:  # noqa: BLE001 - must never propagate into the source's thread
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
        # The library's objects are translated by the tracking adapter, under
        # the recorder's head policy (pose only for a usable gaze, as always).
        # Translated lazily, INSIDE the lock and after the frame is counted, so
        # a pose builder that raises leaves the frame accounting exactly as it
        # always has.
        now_s = self.clock()
        self._on_observation(
            lambda: SRC.observe(
                face_info,
                gaze_info,
                observed_s=now_s,
                head_builder=self.head_builder,
                head_policy=HeadPolicy.GAZE,
                pnp=self.pnp,
            ),
            now_s,
        )

    def _on_observation(
        self,
        observation: FrameObservation | Callable[[], FrameObservation],
        now_s: float | None = None,
    ) -> None:
        if now_s is None:
            assert isinstance(observation, FrameObservation)
            now_s = observation.observed_s
        with self.lock:
            self.frames += 1
            self._frame_times.append(now_s)
            target = self.current_target()
            gate = self._gate
            obs = observation() if callable(observation) else observation
            gaze_status = obs.gaze_status
            features = GATE.features_for(obs, GATE.RECORD_OVERLAY)
            if features is not None:
                if self.feature_dim is None:
                    self.feature_dim = int(features.shape[0])
            # Stored in the LIBRARY's frame, deliberately: this is the raw
            # capture and every existing recording is in it, so flipping here
            # would silently change what old files mean. The camera faces the
            # person, so the "left" column is the eye on the left of the
            # IMAGE, which is their RIGHT one. Anything that interprets these
            # as the person's eyes must go through
            # gf_gesture.eyes_as_the_person_has_them first -- not doing so is
            # what made every wink detector watch the wrong eye.
            left, right = obs.openness_image
            head = obs.head6
            pnp = obs.pnp_deg
            raw_cm = obs.raw_gaze_cm
            tracking = obs.tracking_label

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
            overlay_valid = GATE.is_gaze_sample(obs, GATE.RECORD_OVERLAY, target_active=not idle)
            if overlay_valid:
                overlay_raw_norm = self._predict_overlay(features, head)
            # A blink, occlusion, tracking loss or bad model result resets the
            # filter, so it cannot pull a future valid point toward stale history.
            overlay_norm = GATE.filter_point(self.overlay_filter, overlay_raw_norm, now_s)
            if overlay_norm is not None:
                overlay_updated_s = now_s

            self.builder.append(
                frame_seq=self.frames - 1,
                timestamp_ns=obs.timestamp_ns,
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
            if self.shadow_builder is not None:
                self._append_shadow(
                    frame_seq=self.frames - 1,
                    timestamp_ns=obs.timestamp_ns,
                    elapsed_ms=elapsed_ms,
                    target=target,
                    phase=phase,
                    target_xy=target_xy,
                    label_cm=label_cm,
                    idle=idle,
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

    def _append_shadow(
        self,
        *,
        frame_seq: int,
        timestamp_ns: int,
        elapsed_ms: float | None,
        target: Target | None,
        phase: str,
        target_xy: tuple[float, float] | None,
        label_cm: tuple[float, float] | None,
        idle: bool,
        accepted: bool,
    ) -> None:
        """The shadow pipeline's row for this frame. Called under ``self.lock``.

        ``accepted`` is the primary's collection decision; the shadow row is
        accepted only if its own pipeline also produced a valid sample, so a
        frame one arm lost is never a training row for it. Pairing at
        analysis time then keeps only rows both arms accepted.
        """

        result = self.shadow(timestamp_ns) if self.shadow is not None else None
        obs = CAP.observe_shadow(result, head_builder=self.head_builder)
        status = obs.gaze_status
        features = GATE.features_for(obs, GATE.RECORD_OVERLAY)
        left, right = obs.openness_image
        head = obs.head6
        if result is None:
            tracking = "NO_SHADOW_RESULT"
        else:
            # The shadow rows have always rendered a missing state as "None".
            tracking = obs.tracking_state_name or obs.tracking_state_text
        assert self.shadow_builder is not None
        self.shadow_builder.append(
            frame_seq=frame_seq,
            timestamp_ns=timestamp_ns,
            elapsed_ms=elapsed_ms,
            target_id=-1 if target is None else target.index,
            block=0 if target is None else target.block,
            phase=phase,
            target_xy=target_xy,
            label_cm=label_cm,
            features=None if idle else features,
            head=None if idle else head,
            pnp_deg=None,
            raw_cm=None,
            openness=(left, right),
            tracking_state=tracking,
            gaze_status=status,
            accepted=bool(accepted and status and features is not None and not idle),
        )

    def _predict_overlay(
        self, features: np.ndarray | None, head: np.ndarray | None
    ) -> tuple[float, float] | None:
        return predict_overlay_point(
            self.overlay_model, self.overlay_model_y, features, head, self.rig
        )

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
        duration = (
            (self._frame_times[-1] - self._frame_times[0]) if len(self._frame_times) > 1 else 0.0
        )
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
        if not (
            math.isclose(ours[0], theirs[0], abs_tol=1e-9)
            and math.isclose(ours[1], theirs[1], abs_tol=1e-9)
        ):
            raise RuntimeError(f"px2cm mismatch at ({nx}, {ny}): ours {ours} vs library {theirs}")


# --- Display -----------------------------------------------------------------


# Fonts that carry Hebrew, in the order they are tried. MEASURED rather than
# assumed: rendered at the same size, pygame's built-in font produced
# byte-identical ink for aleph and shin (2508 == 2508) -- the same empty box
# twice -- while every system font gave different glyphs (Arial 7776/9504,
# Segoe UI 10578/12900). So ``pygame.font.Font(None, ...)`` cannot draw a
# Hebrew interface at all, and which font loaded has to be REPORTED: a
# fallback that silently draws boxes looks exactly like a rendering bug.
# --- Session -----------------------------------------------------------------


@dataclass
class SessionResult:
    recordings: dict[str, Path] = field(default_factory=dict)
    # Every protocol that ran to a frozen recording, saved or not. In a normal
    # run these are exactly the keys of ``recordings``; in a --no-save demo
    # nothing reaches the disk, so this is the only evidence a protocol ran and
    # the only thing the exit code can be judged on.
    protocols_run: list[str] = field(default_factory=list)
    saved: bool = True
    aborted: bool = False
    watchdog_tripped: bool = False
    failure: str | None = None
    shutdown: dict[str, Any] = field(default_factory=dict)


def held_condition_lines(hold: str | None) -> tuple[str, ...]:
    """The start-screen lines for a condition held across a whole run.

    Pure and separately testable because the alternative -- proving it by
    running a session -- takes minutes and did not, in fact, exercise the
    failure it was supposed to: a check protocol has no pose blocks, so the
    dry run that "verified" this feature never reached the code that broke it.
    """

    if not hold:
        return ()
    return (f"HOLD THIS THROUGHOUT:  {hold}", "")


def band_target_files(targets: Path, targets_tune: Path) -> tuple[Path, Path]:
    """Swap the default target sets for their central-band versions.

    A banded run narrows the CALIBRATION grid, but the held-out targets live
    in their own file and are not narrowed by the flag. Scoring full-width
    targets against a model calibrated on the central band measures the model
    outside everything it was ever shown: measured on round4, the targets
    outside the band read 964 px against 234 px for those inside it, and the
    all-targets median was mostly describing the ones the model had never
    seen.

    Shared with anything that drives run_session directly, because that is
    exactly how the swap gets missed -- it lived only in the CLI, and a
    programmatic caller silently recorded the wrong target set.
    """

    if targets == PACKAGE_DIR / "targets.json":
        targets = PACKAGE_DIR / "targets_central.json"
    if targets_tune == PACKAGE_DIR / "targets_tune.json":
        targets_tune = PACKAGE_DIR / "targets_tune_central.json"
    return targets, targets_tune


DUAL_LO_SUBDIR = "lo"
DUAL_MIN_WARMUP_FRAMES = 10


HI_RECORDINGS_DIR = RECORDINGS_DIR / "hi"


def check_hi_capture_allowed(
    *, dry_run: bool, overlay_model_dir: Path | None, out_root: Path, no_save: bool
) -> None:
    """Refuse a single-pipeline hi session anywhere but ``recordings/hi``.

    Until models carry their capture pipeline and every live tool refuses a
    mismatch (TASKS section 62, plan review blocker 1), hi recordings must not
    sit where ``gf_recalibrate`` or ``gf_pool.discover`` read by default: a hi
    calibration fitted there could be adopted into a 640x480 profile. The pool
    key separates pipelines as well; this is the second guard. No overlay:
    there is no hi model yet, and a lo model on hi features is a confident
    wrong dot.
    """

    if dry_run:
        raise SystemExit("--capture hi needs the real camera; it has no dry-run path")
    if overlay_model_dir is not None:
        raise SystemExit(
            "--capture hi cannot show an overlay yet: no saved model was fitted on hi features"
        )
    if no_save:
        return
    root = Path(out_root).resolve()
    allowed = HI_RECORDINGS_DIR.resolve()
    if root != allowed and allowed not in root.parents:
        raise SystemExit(f"--capture hi must write under {allowed}; use --out recordings/hi")


def check_dual_capture_allowed(
    *, dry_run: bool, overlay_model_dir: Path | None, out_root: Path, no_save: bool
) -> None:
    """Refuse a dual-resolution session that could be misread or mis-pooled.

    * dry run: the fake camera has no second pipeline;
    * overlay: every model on disk was fitted on 640x480-library features, so
      a dot drawn from hi features would be a confident wrong answer;
    * directly under recordings/: ``gf_pool.discover`` reads ``round*`` there,
      and these features must never train a production model. (The pool key
      also carries the pipeline, so this is the second of two guards.)
    """

    if dry_run:
        raise SystemExit("--capture dual needs the real camera; it has no dry-run path")
    if overlay_model_dir is not None:
        raise SystemExit(
            "--capture dual cannot show an overlay: every saved model was fitted on the "
            "library's 640x480 features"
        )
    if not no_save and Path(out_root).resolve() == RECORDINGS_DIR.resolve():
        raise SystemExit(
            "--capture dual must not write directly into recordings/ (the training pool reads "
            "round* there); use --out recordings/resolution"
        )


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
    overlay_model_y_dir: Path | None = None,
    overlay_filter_settings: GF.FilterSettings | None = None,
    show_unfiltered_overlay: bool = False,
    allow_overwrite: bool = False,
    skip_model_check: bool = False,
    target_order_seed: int | None = None,
    repeat_first: int = 0,
    pose_blocks: int | None = None,
    pose_set: str = DEFAULT_POSE_SET,
    pose_start: int = 1,
    hold: str | None = None,
    advance_timeout_s: float | None = None,
    no_save: bool = False,
    capture: str = "library",
) -> SessionResult:
    import gazefollower  # noqa: PLC0415  (initialises native components; unavoidable for the live path)
    from gazefollower import GazeFollower  # noqa: PLC0415
    from gazefollower.misc import DefaultConfig  # noqa: PLC0415

    result = SessionResult(saved=not no_save)
    session_stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    # None means this session persists nothing: every write below is guarded on
    # it, so a demo cannot leave a half-written round behind.
    round_dir = prepare_output_dir(
        out_root, round_id, allow_overwrite=allow_overwrite, no_save=no_save
    )

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
    # Raises before the camera is opened: a bad --pose-set or --pose-start is a
    # typo, and finding it after the operator has sat down costs a session.
    poses = resolve_poses(pose_set, pose_blocks, pose_start)

    specs: list[ProtocolSpec] = []
    for name in protocols:
        if name == "A":
            specs.append(protocol_a(grid_x))
        elif name == "B":
            specs.append(protocol_b(grid_x, n_blocks=len(poses)))
        elif name == "TUNE":
            specs.append(
                protocol_timed(
                    "TUNE",
                    targets_tune,
                    C.DEFAULT_SETTLE_MS / 1000.0,
                    C.DEFAULT_COLLECT_MS / 1000.0,
                )
            )
        elif name == "GRID16":
            specs.append(
                protocol_timed(
                    "GRID16",
                    targets_grid16,
                    C.DEFAULT_SETTLE_MS / 1000.0,
                    C.DEFAULT_COLLECT_MS / 1000.0,
                )
            )
        elif name == "T1":
            specs.append(
                protocol_timed(
                    "T1",
                    targets_t1,
                    C.DEFAULT_SETTLE_MS / 1000.0,
                    C.DEFAULT_COLLECT_MS / 1000.0,
                    order_seed=target_order_seed,
                    repeat_first=repeat_first,
                )
            )
        elif name in ("FULL", "MOVE"):
            # Generated for the display actually in use, so a change of monitor
            # cannot leave the targets describing the previous one.
            coverage = GD.coverage_targets(
                int(target_geometry["width_px"]),
                int(target_geometry["height_px"]),
                calibration=calibration_grid,
            )
            targets = [
                Target(
                    int(t["index"]),
                    str(t["name"]),
                    float(t["screen_position"]["x"]),
                    float(t["screen_position"]["y"]),
                )
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
        elif name == "T3":
            specs.append(protocol_t3(targets_t1))
        else:
            raise ValueError(
                f"protocol {name!r} is not known "
                "(A, TUNE, GRID16, T1, T2 for phase 0; T3 for the long-hold probe; "
                "B for the multi-pose calibration; FULL, MOVE for the full-screen sessions)"
            )

    if capture not in ("library", "dual", "hi"):
        raise ValueError(f"unknown capture mode {capture!r}")
    dual = capture == "dual"
    if dual:
        check_dual_capture_allowed(
            dry_run=dry_run, overlay_model_dir=overlay_model_dir, out_root=out_root, no_save=no_save
        )
    if capture == "hi":
        check_hi_capture_allowed(
            dry_run=dry_run, overlay_model_dir=overlay_model_dir, out_root=out_root, no_save=no_save
        )
    kwargs: dict[str, Any] = {"config": config, "calibration": make_pass_through_calibration()}
    dual_camera: Any = None
    if dry_run:
        camera, face_alignment, estimator = make_dry_run_components()
        kwargs.update(camera=camera, face_alignment=face_alignment, gaze_estimator=estimator)
    elif dual:
        dual_camera, face_alignment, estimator = CAP.make_dual_components()
        kwargs.update(camera=dual_camera, face_alignment=face_alignment, gaze_estimator=estimator)
    elif capture == "hi":
        # The same camera class with no second arm, so the warm-up shape
        # check, per-protocol stats and metadata below apply unchanged.
        dual_camera, face_alignment, estimator = CAP.make_hi_components()
        kwargs.update(camera=dual_camera, face_alignment=face_alignment, gaze_estimator=estimator)
    gf = GazeFollower(**kwargs)
    library_tmp = [str(getattr(gf, "_tmpSampleDataPath", ""))]

    base_meta = {
        "session": session_stamp,
        "rig": rig.to_dict(),
        "setup": None if manifest is None else manifest.to_dict(),
        "monitor": None if monitor is None else monitor.to_dict(),
        "target_geometry": dict(target_geometry),
        "library": {
            "name": "gazefollower",
            "version": getattr(gazefollower, "__version__", "1.0.2"),
            "screen_size": [lib_w, lib_h],
        },
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
        "overlay_filter": None
        if overlay_filter_settings is None
        else asdict(overlay_filter_settings),
        "capture": {
            "pipeline": {"dual": CAP.PIPELINE_HI, "hi": CAP.PIPELINE_HI_SINGLE}.get(
                capture, CAP.LIBRARY_PIPELINE
            )
        },
    }

    overlay_model = None
    overlay_model_y = None
    if overlay_model_dir is not None:
        import gf_fit as FIT  # noqa: PLC0415 - only needed when an overlay is requested

        # A correction.json beside the model is applied to the point that is
        # DRAWN, so the dot on screen and the numbers scored afterwards cannot
        # come from different things. Printed loudly: a silently corrected
        # overlay would look like a better model.
        overlay_model = CORR.load_with_correction(overlay_model_dir, FIT.FittedModel.load)
        print(
            f"overlay model loaded from {overlay_model_dir} (columns: {len(overlay_model.schema.columns)})"
        )
        if isinstance(overlay_model, CORR.CorrectedModel):
            print(f"  WITH A CORRECTION LAYER: {overlay_model.correction.describe()}")
            if overlay_model.correction.note:
                print(f"  {overlay_model.correction.note}")
        if overlay_model_y_dir is not None:
            overlay_model_y = CORR.load_with_correction(overlay_model_y_dir, FIT.FittedModel.load)
            print(f"vertical overlay model loaded from {overlay_model_y_dir}")
    # Which model produced the point is not recoverable from the recording
    # otherwise -- only the filter settings were stored, so a later reader
    # could not tell which model an operator actually watched.
    # The presentation order is the experiment here, so it is recorded rather
    # than left to be inferred from the frames.
    base_meta["target_order_seed"] = target_order_seed
    base_meta["repeat_first"] = repeat_first
    # What RAN, not what was asked for: --pose-blocks may be None ("all of
    # them") and --pose-start may have dropped the opening blocks, so these
    # come from the resolved sequence. The instruction TEXT is stored too --
    # it is the experimental variable here, and a later reader cannot
    # otherwise tell which wording the operator actually saw.
    has_b = "B" in protocols
    # Not gated on protocol B: the whole point is the protocols that have no
    # per-block prompt of their own.
    base_meta["hold_instruction"] = hold
    base_meta["pose_set"] = pose_set if has_b else None
    base_meta["pose_start"] = pose_start if has_b else None
    base_meta["pose_blocks"] = len(poses) if has_b else None
    base_meta["pose_sequence"] = poses.labels() if has_b else None
    base_meta["pose_instructions"] = poses.instructions() if has_b else None
    base_meta["pose_partial"] = poses.partial if has_b else None
    base_meta["overlay_model"] = None if overlay_model_dir is None else str(overlay_model_dir)
    base_meta["overlay_model_y"] = None if overlay_model_y_dir is None else str(overlay_model_y_dir)

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

    preflight: list[np.ndarray] = []

    def subscriber(face_info: Any, gaze_info: Any) -> None:
        runner = runner_ref["runner"]
        if runner is None:
            # Before the first protocol installs a runner, the warm-up frames
            # are otherwise thrown away. Keep their design rows so the overlay
            # model can be checked against today's conditions before the
            # operator spends a protocol on it. The row is assembled exactly
            # as ProtocolRunner._predict_overlay assembles it, head columns
            # included, or the check would not be measuring what will run.
            if overlay_model is not None and SRC.gaze_status_of(gaze_info):
                if len(preflight) < PREFLIGHT_MAX_FRAMES:
                    row = _overlay_design_row(overlay_model, face_info, gaze_info)
                    if row is not None:
                        preflight.append(row)
            return
        runner.on_frame(face_info, gaze_info)

    gf.add_subscriber(subscriber)

    try:
        # camera.start_sampling() directly: GazeFollower.start_sampling() would
        # also subscribe its CSV writer, which raises on status-False frames.
        gf.camera.start_sampling()
        display.draw_message(["Camera warming up..."])
        _sleep_with_escape(display, CAMERA_WARMUP_S / speed)
        if dual_camera is not None and dual_camera.usable_frames() < DUAL_MIN_WARMUP_FRAMES:
            # Caught here, not by the watchdog minutes later: a camera that
            # ignored 1920x1080 has every frame dropped and looks like silence.
            raise SystemExit(
                f"ABORTED before recording: the camera delivered {dual_camera.usable_frames()} "
                f"usable frames in {CAP.SOURCE_MODE[0]}x{CAP.SOURCE_MODE[1]} during warm-up: "
                f"{dual_camera.describe()['stats']}"
            )
        # No runner is installed yet, so nothing is recorded during this wait.
        if overlay_model is not None and not skip_model_check:
            # Both models are checked: a fresh x model paired with a stale y
            # model would freeze the vertical axis alone, which reads as a
            # dot sliding along a horizontal line rather than as a failure.
            ok, message = True, "no frames to check the model against; skipping"
            for label, candidate in (("x", overlay_model), ("y", overlay_model_y)):
                if candidate is None:
                    continue
                try:
                    activation = (
                        candidate.support_activation(np.vstack(preflight)) if preflight else None
                    )
                except AttributeError as exc:
                    # NOT swallowed. The preflight answers "ok" for anything it
                    # cannot measure, so a model object that has lost the method
                    # would turn the frozen-model abort off while printing a
                    # reassuring line. A missing method is a programming error
                    # in the model wrapper, not an unmeasurable condition.
                    raise RuntimeError(
                        f"overlay {label} model cannot report support activation: {exc}. "
                        "Refusing to record with the frozen-model check disabled."
                    ) from exc
                except Exception as exc:  # noqa: BLE001 - a check must not abort a good session
                    activation = None
                    print(f"overlay {label} model check could not run ({exc!r}); continuing")
                ok, message = preflight_verdict(activation)
                print(f"overlay {label} model check: {message}")
                if not ok:
                    break
            if not ok:
                display.draw_message(
                    ["Model does not fit current conditions.", "", "See the console."]
                )
                raise SystemExit(f"ABORTED before recording: {message}")
        if (
            display.wait_for_key(
                [
                    "Ready.",
                    "",
                    f"{len(specs)} protocols: {', '.join(s.name for s in specs)}",
                    "",
                    "Each one waits for you before it starts, so take the time to",
                    "read what it asks for. Esc aborts at any point.",
                ]
            )
            == "abort"
        ):
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
            shadow_builder = (
                S.RecordingBuilder(
                    spec.name, round_id, dict(meta, capture={"pipeline": CAP.PIPELINE_LO})
                )
                if dual
                else None
            )
            runner = ProtocolRunner(
                spec,
                builder,
                rig,
                speed=speed,
                overlay_model=overlay_model,
                overlay_model_y=overlay_model_y,
                overlay_filter_settings=overlay_filter_settings,
                shadow=None if not dual else dual_camera.shadow_for,
                shadow_builder=shadow_builder,
            )
            remaining = [s.name for s in specs[specs.index(spec) :]]
            # The wait is unbounded by design, so the recorder must NOT be
            # listening during it. The camera keeps running -- the tracker
            # needs to stay locked on -- but with no runner installed the
            # subscriber discards those frames instead of storing a face
            # embedding for every one of them for as long as the person is
            # away from the desk.
            # With a timeout the protocol starts on its own, which is what
            # makes an unattended, hands-free sequence possible at all. Esc
            # still aborts, so there is always a way out that needs no timing.
            start_line = (
                "Press SPACE or ENTER to start   (Esc to abort)"
                if advance_timeout_s is None
                else f"starts on its own in {advance_timeout_s:.0f}s   (Esc to abort)"
            )
            if (
                display.wait_for_key(
                    [
                        f"{spec.name}   ({specs.index(spec) + 1} of {len(specs)})",
                        "",
                        spec.instruction,
                        "",
                        # The held condition, if this run has one. Placed
                        # between the protocol's own instruction and the
                        # target count so it cannot be skipped over.
                        *held_condition_lines(hold),
                        f"{len(spec.exported_targets())} targets, about {_estimate_seconds(spec):.0f} seconds",
                        f"still to come: {', '.join(remaining[1:]) or 'nothing, this is the last one'}",
                        "",
                        start_line,
                    ],
                    timeout_s=advance_timeout_s,
                )
                == "abort"
            ):
                result.aborted = True
                break
            if dual_camera is not None:
                dual_camera.reset_stats()  # latency and frame counts per protocol
            runner_ref["runner"] = runner
            runner.start()
            # Armed here, not before the wait: the clock for "frames have
            # stopped" starts when this protocol does.
            dog = FrameWatchdog(limit_s=WATCHDOG_S, started_s=time.monotonic())
            display.play_beep()
            last_target: Target | None = runner.current_target()
            blocks_in_spec = len({t.block for t in spec.targets})
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
                pending = runner.awaiting_block
                if pending is not None:
                    pose = poses[pending]
                    label, pose_instruction = pose.label, pose.instruction
                    # rtl() reverses a whole string instead of running a bidi
                    # algorithm, so the closing line has to match the script of
                    # the instruction or one of the two is drawn backwards. The
                    # header is pure Latin and renders correctly either way.
                    # NOT named ``hold``: that is the run_session parameter
                    # carrying the operator's held condition, and reassigning
                    # it here would destroy it for every later protocol's
                    # start screen while the manifest -- written before this
                    # loop -- still recorded the right text. The operator
                    # would be shown boilerplate and the file would claim the
                    # condition, which is exactly the mismatch --hold exists
                    # to prevent.
                    pose_hold_line = (
                        "\u05d4\u05d7\u05d6\u05e7 \u05d0\u05ea \u05d4\u05ea\u05e0\u05d5\u05d7\u05d4 "
                        "\u05d4\u05d6\u05d0\u05ea \u05e2\u05d3 \u05e1\u05d5\u05e3 \u05d4\u05d1\u05dc\u05d5\u05e7"
                        if has_hebrew(pose_instruction)
                        else "Hold that position for the whole block."
                    )
                    if (
                        display.wait_for_key(
                            [
                                f"{spec.name}   pose {pending + 1} of {blocks_in_spec}   ({label})",
                                "",
                                pose_instruction,
                                "",
                                pose_hold_line,
                            ]
                        )
                        == "abort"
                    ):
                        result.aborted = True
                        break
                    # The watchdog measures silence from the camera, and the
                    # pose prompt is an unbounded wait during which frames are
                    # arriving but nothing is stored. Restart its clock so the
                    # operator taking their time cannot look like a stall.
                    dog = FrameWatchdog(limit_s=WATCHDOG_S, started_s=time.monotonic())
                    runner.resume_block()
                    display.play_beep()
                    continue
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
                    f"head {'ok' if state.head_valid else 'INVALID'}"
                    + (f"  pitch_a {state.pitch_a:+.3f}" if state.pitch_a is not None else ""),
                    f"errors {runner.errors}"
                    + (f"  LAST: {runner.last_error[:60]}" if runner.last_error else ""),
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
                rec_lo = None if runner.shadow_builder is None else runner.shadow_builder.freeze()
            rec.meta["integrity"] = runner.integrity(
                watchdog_tripped=watchdog, aborted=result.aborted
            )
            rec.meta["integrity"]["failure"] = result.failure
            rec.meta["integrity"]["fps_median"] = runner.fps_median()
            if dual_camera is not None:
                described = dual_camera.describe()
                rec.meta["capture"] = dict(rec.meta["capture"], **described)
                if rec_lo is not None:
                    rec_lo.meta["integrity"] = dict(rec.meta["integrity"])
                    rec_lo.meta["capture"] = dict(rec_lo.meta["capture"], **described)
            result.protocols_run.append(spec.name)
            if round_dir is not None:
                npz_path, _ = rec.save(round_dir)
                result.recordings[spec.name] = npz_path
                if rec_lo is not None:
                    rec_lo.save(round_dir / DUAL_LO_SUBDIR)
            dim = runner.feature_dim
            if dim is not None and dim != EXPECTED_FEATURE_DIM:
                print(f"WARNING: feature dim {dim} != expected {EXPECTED_FEATURE_DIM}")
            print(_summary_line(spec.name, rec, saved=round_dir is not None))
            if result.aborted or watchdog:
                break
        if not result.aborted and not result.watchdog_tripped:
            runner_ref["runner"] = None  # nothing recorded while this sits on screen
            display.wait_for_key(
                [
                    "Done. Thank you.",
                    "",
                    (
                        f"NOT SAVED (demo): {', '.join(result.protocols_run)}"
                        if no_save
                        else f"Recorded: {', '.join(result.recordings)}"
                    ),
                ],
                timeout_s=60.0,
            )
    finally:
        # stop_sampling() reaches the library's unbounded join; bound it too.
        result.shutdown = {
            "stop_sampling": _call_with_timeout(gf.camera.stop_sampling, SHUTDOWN_STEP_TIMEOUT_S)
        }
        result.shutdown.update(shutdown_library(gf))
        display.close()
    return result


def _summary_line(name: str, rec: S.Recording, *, saved: bool = True) -> str:
    integ = rec.meta.get("integrity", {})
    n_acc = int(np.sum(rec.rows_accepted()))
    n_col = int(np.sum(rec.rows_collecting()))
    gaze_rows = np.asarray(rec.gaze_status, dtype=bool)
    head_invalid = (
        float(np.mean(~rec.rows_head_valid()[gaze_rows])) if np.any(gaze_rows) else float("nan")
    )
    return (
        f"{name}: rows {rec.n_rows}, accepted {n_acc}, collecting {n_col}, fps {integ.get('fps_median')}, "
        f"errors {integ.get('subscriber_errors')}, head-invalid {100 * head_invalid:.1f}%, "
        f"aborted {integ.get('aborted')}, watchdog {integ.get('watchdog_tripped')}"
        # Said on the protocol's own line, not only in the closing summary: an
        # operator reading the scroll must not have to remember which flag the
        # run started with to know whether these rows still exist.
        + ("" if saved else "  [NOT SAVED]")
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


def resolve_round_id(round_id: int | None, *, no_save: bool) -> int:
    """The round id this session is labelled with, or a message to show instead.

    A saving run must always name its round: an id that defaulted silently
    could land on data an earlier session recorded. A --no-save run writes
    nothing, spends no round number, and gets the sentinel rather than making
    the operator invent an unused number for a demonstration.
    """

    if round_id is not None:
        return int(round_id)
    if no_save:
        return DEMO_ROUND_ID
    raise ValueError("--round is required unless --no-save is given")


def prepare_output_dir(
    out_root: Path,
    round_id: int,
    *,
    allow_overwrite: bool = False,
    no_save: bool = False,
) -> Path | None:
    """The directory this session writes into, or None when it writes nothing.

    One decision point for "does this session persist?", so a demonstration
    run cannot end up half-written: returning None skips the recordings-root
    check, the .gitignore, the round directory, and the overwrite guard as one
    unit. The guard is skipped rather than passed ``allow_overwrite``: it
    protects data from being replaced, and a run that writes nothing cannot
    replace anything, so a demo must not have to name an unused round.

    Nothing here is reached in no-save mode, which is also why ``out_root`` is
    not validated then -- there is no write for ``resolve_output_root`` to keep
    inside the approved directory.
    """

    if no_save:
        return None
    root = resolve_output_root(out_root)
    _ensure_gitignore(root)
    round_dir = root / f"round{round_id}"
    guard_existing_round(round_dir, allow_overwrite=allow_overwrite)
    return round_dir


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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--round",
        type=int,
        default=None,
        help="recording round id (0 = Phase 0); required unless --no-save is given",
    )
    parser.add_argument(
        "--protocols", default="A,TUNE,GRID16,T1,T2", help="comma-separated, in order"
    )
    parser.add_argument("--camera-x-cm", type=float, required=True)
    parser.add_argument(
        "--camera-y-cm",
        type=float,
        required=True,
        help="down from the top-left corner; > screen height = below the screen",
    )
    parser.add_argument("--screen-width-cm", type=float, required=True)
    parser.add_argument("--screen-height-cm", type=float, required=True)
    parser.add_argument(
        "--device-w",
        type=int,
        default=None,
        help="display width in device px (default: taken from --monitor)",
    )
    parser.add_argument("--device-h", type=int, default=None)
    parser.add_argument(
        "--monitor",
        default=None,
        help="which display: an index, or part of its name (default: primary)",
    )
    parser.add_argument(
        "--list-monitors", action="store_true", help="print the displays the OS reports and exit"
    )
    parser.add_argument(
        "--viewing-distance-cm",
        type=float,
        default=None,
        help="eye to screen centre; distinct from the camera distance",
    )
    parser.add_argument(
        "--dpi-scale",
        type=float,
        default=1.0,
        help="OS scaling of the selected display (logical px = device px / scale)",
    )
    parser.add_argument(
        "--preset",
        default=None,
        help="calibration preset intended for this run; recorded in the manifest",
    )
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
    parser.add_argument(
        "--overlay-show-unfiltered",
        action="store_true",
        help="also draw the unfiltered calibrated prediction in purple for diagnosis",
    )
    parser.add_argument(
        "--overlay-model-y",
        type=Path,
        default=None,
        help=(
            "second fitted model supplying ONLY the vertical axis of the blue dot; "
            "x still comes from --overlay-model. The two feature-scaling families are "
            "each accurate on a different axis, so a model per axis takes the better "
            "half of each. Measured offline only -- not yet validated live."
        ),
    )
    parser.add_argument(
        "--target-order-seed",
        type=int,
        default=None,
        help=(
            "shuffle T1's presentation order with this seed (recorded in the manifest). "
            "Every T1 run so far used the same fixed order, so time and screen position "
            "could not be told apart; two runs with different seeds separate them."
        ),
    )
    parser.add_argument(
        "--advance-timeout-s",
        type=float,
        default=None,
        help=(
            "start each protocol on its own after this many seconds instead of waiting for a "
            "key. Required for a hands-free sequence; Esc still aborts."
        ),
    )
    parser.add_argument(
        "--hold",
        default=None,
        help=(
            "a condition to hold for the WHOLE run, shown on the start screen of every "
            "protocol and recorded in the manifest. For check recordings that are "
            "supposed to be made in a particular head position -- without it the "
            "operator has to remember the condition, and a misremembered one is "
            "indistinguishable from a correctly recorded one afterwards."
        ),
    )
    parser.add_argument(
        "--pose-set",
        choices=sorted(POSE_SETS),
        default=DEFAULT_POSE_SET,
        help=(
            "protocol B only: which pose sequence to run. 'standard' is the original "
            f"{len(POSE_SEQUENCE)} blocks and stays the default so earlier runs reproduce "
            f"exactly; 'extended' is {len(POSE_SEQUENCE_EXTENDED)} blocks covering all six "
            "head components in both directions, with Hebrew instructions."
        ),
    )
    parser.add_argument(
        "--pose-blocks",
        type=int,
        default=None,
        help=(
            "protocol B only: how many pose blocks to run (default: all of the chosen "
            "set). Each block shows the whole calibration grid once, in its own order. "
            "Fewer blocks is a shorter session and a narrower pose range; whether the "
            "range was wide enough is decided afterwards from the recording, not here."
        ),
    )
    parser.add_argument(
        "--pose-start",
        type=int,
        default=1,
        help=(
            "protocol B only: start at this block of the chosen set (1-based). A safety "
            "valve for resuming after an aborted recording -- an aborted run is frozen as "
            "aborted and cannot be continued. A run that uses this is PARTIAL: coverage "
            "and pose weights are computed from the rows actually stored."
        ),
    )
    parser.add_argument(
        "--repeat-first",
        type=int,
        default=0,
        help=(
            "re-show the first N T1 targets again at the end, keeping their ids. Gives the "
            "same positions at two separated times inside ONE run, so position and time are "
            "separated without comparing runs."
        ),
    )
    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="record even if the overlay model does not fit the current conditions",
    )
    parser.add_argument(
        "--overlay-reset-gap-ms",
        type=float,
        default=250.0,
        help="reset filter history after this gap",
    )
    parser.add_argument("--overlay-ema-cutoff-hz", type=float, default=2.0)
    parser.add_argument("--overlay-one-euro-min-cutoff-hz", type=float, default=1.2)
    parser.add_argument(
        "--overlay-one-euro-beta",
        type=float,
        default=0.0005,
        help="One Euro speed response in Hz per (pixel/second)",
    )
    parser.add_argument("--overlay-one-euro-derivative-cutoff-hz", type=float, default=1.0)
    parser.add_argument(
        "--overlay-kalman-acceleration-noise", type=float, default=400.0, help="px^2/s^4"
    )
    parser.add_argument(
        "--overlay-kalman-measurement-noise", type=float, default=900.0, help="px^2"
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=PACKAGE_DIR / "targets.json",
        help="T1 targets (analyze.py geometry)",
    )
    parser.add_argument("--targets-tune", type=Path, default=PACKAGE_DIR / "targets_tune.json")
    parser.add_argument("--targets-grid16", type=Path, default=PACKAGE_DIR / "targets_grid16.json")
    parser.add_argument(
        "--eye-distance-cm",
        type=float,
        default=None,
        help="operator-measured eye-to-screen distance",
    )
    parser.add_argument("--glasses", default=None, help="none | glasses | contacts")
    parser.add_argument(
        "--lighting", default=None, help="short description, kept constant across sessions"
    )
    parser.add_argument("--note", default=None)
    parser.add_argument(
        "--skip-camera-probe",
        action="store_true",
        help="do not measure the camera's actual frame rate",
    )
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
    parser.add_argument(
        "--capture",
        choices=("library", "dual", "hi"),
        default="library",
        help=(
            "library (default): the library's own camera, 640x480. hi: read 1920x1080, take a "
            "centred 1440x1080 crop and run ONE pipeline on it (writes only under "
            "recordings/hi; no overlay). dual: read 1920x1080, take a "
            "centred 1440x1080 crop, and run TWO pipelines on every frame -- the crop itself "
            "(hi) and the crop reduced to 640x480 (lo, saved under round*/lo/) -- so the two "
            "resolutions are compared on identical frames. Experiment only: refuses an overlay "
            "and refuses to write directly into recordings/."
        ),
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help=(
            "demonstration run: show the protocols and the live overlay, but write NOTHING to "
            "disk -- no .npz, no .meta.json, no setup.json, no round directory. --round is then "
            "unnecessary, and the overwrite guard does not apply because nothing can be "
            "overwritten. Every safety check still runs: the overlay model preflight, the "
            "no-face stall guard, the frame watchdog and Esc"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fake camera/face/model through the real process_frame",
    )
    parser.add_argument("--headless", action="store_true", help="no window (dry-run only)")
    parser.add_argument(
        "--speed", type=float, default=1.0, help="time scale for dry runs (10 = ten times faster)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Answered before parsing: listing displays is a question about the
    # machine, not about a run, and must not demand a run's arguments.
    if "--list-monitors" in (sys.argv[1:] if argv is None else list(argv)):
        print(GD.describe_monitors())
        return 0
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        # argparse's own required=True used to enforce this; the rule moved
        # here because it now depends on another flag. Same exit code, same
        # moment: before anything opens a camera or a window.
        args.round = resolve_round_id(args.round, no_save=args.no_save)
    except ValueError as exc:
        parser.error(str(exc))
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
    print(
        f"display [{monitor.index}] {monitor.name}: {device_w}x{device_h} px at desktop {monitor.origin}"
        f"{' (PRIMARY)' if monitor.is_primary else ''}"
    )
    rig = C.RigGeometry(
        args.camera_x_cm,
        args.camera_y_cm,
        args.screen_width_cm,
        args.screen_height_cm,
        device_w,
        device_h,
    )
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
        for label, declared, reported in (
            ("width", args.screen_width_cm, monitor.width_mm / 10.0),
            ("height", args.screen_height_cm, monitor.height_mm / 10.0),
        ):
            if abs(declared - reported) > 2.0:
                print(
                    f"SETUP WARNING: declared screen {label} {declared} cm vs OS-reported {reported:.1f} cm"
                )
    if args.viewing_distance_cm:
        geom = GD.ViewingGeometry(
            args.screen_width_cm,
            args.screen_height_cm,
            device_w,
            device_h,
            args.viewing_distance_cm,
            args.eye_distance_cm,
        )
        h, v = geom.half_angles_deg()
        print(
            f"viewing distance {args.viewing_distance_cm} cm -> half-angles h +/-{h:.1f} deg, v +/-{v:.1f} deg; "
            f"1.5 deg = {geom.deg_to_px(1.5):.0f} px"
        )
    protocols = [p.strip() for p in args.protocols.split(",") if p.strip()]
    x_range: tuple[float, float] | None = None
    if args.x_range is not None:
        lo, hi = args.x_range
        if not (0.0 <= lo < 0.5 < hi <= 1.0):
            print("--x-range must satisfy 0 <= LO < 0.5 < HI <= 1")
            return 2
        x_range = (lo, hi)
        # Unless the operator pointed at specific files, use the central ones.
        args.targets, args.targets_tune = band_target_files(args.targets, args.targets_tune)
        if "GRID16" in protocols:
            print("GRID16 spans the full width and is dropped for a banded run")
            protocols = [p for p in protocols if p != "GRID16"]
        print(
            f"horizontal band {lo:.2f}..{hi:.2f}: calibration grid x = {lo:.2f}/0.50/{hi:.2f} (NOT the library's grid)"
        )
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
    checked = (
        ()
        if generated
        else (("targets-tune", args.targets_tune), ("targets-grid16", args.targets_grid16))
    )
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
            layout = comparison.get("code_layout_changed")
            if layout:
                print(f"  code layout: {layout['note']}")
    if args.eye_distance_cm is None:
        print(
            "NOTE: no --eye-distance-cm given; angular error will be unavailable for this session"
        )
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
        overlay_model_y_dir=args.overlay_model_y,
        overlay_filter_settings=overlay_filter_settings,
        show_unfiltered_overlay=args.overlay_show_unfiltered,
        skip_model_check=args.skip_model_check,
        target_order_seed=args.target_order_seed,
        repeat_first=args.repeat_first,
        pose_blocks=args.pose_blocks,
        pose_set=args.pose_set,
        hold=args.hold,
        pose_start=args.pose_start,
        advance_timeout_s=args.advance_timeout_s,
        no_save=args.no_save,
        capture=args.capture,
    )
    if result.saved and result.recordings:
        manifest.save(args.out / f"round{args.round}" / "setup.json")
    if not result.saved:
        print(
            "NOTHING WAS SAVED: this was a --no-save demonstration run. No recording, "
            "metadata or setup file was written to disk."
        )
    print(
        json.dumps(
            {
                "recordings": {k: str(v) for k, v in result.recordings.items()},
                "protocols_run": result.protocols_run,
                "saved": result.saved,
                "aborted": result.aborted,
                "watchdog_tripped": result.watchdog_tripped,
                "failure": result.failure,
                "shutdown": result.shutdown,
            },
            indent=2,
        )
    )
    # Judged on protocols that ran, not on files: in a normal run these are the
    # same set, and a demo that completed every protocol is not a failure just
    # because it deliberately wrote nothing.
    ran = result.protocols_run
    return 0 if (ran and not result.aborted and not result.watchdog_tripped) else 1


if __name__ == "__main__":
    raise SystemExit(main())
