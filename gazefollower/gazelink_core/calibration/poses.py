"""Head-pose sequences for multi-pose calibration.

Protocol B shows the calibration grid once per head pose so that gaze and
pose vary independently. WHICH poses, in which order, and what the operator
is told to do is a decision about the experiment -- not about camera
handling, frame gating or file layout -- so it lives here rather than inside
the recorder script, and it is testable without a camera, a display or a
model.

Two sets are defined:

``STANDARD`` -- the original seven blocks. Frozen. Earlier recordings are
    reproduced from it byte for byte, so nothing about it may change: a
    changed instruction or a changed order would silently redefine what an
    old recording means.

``EXTENDED`` -- fifteen blocks covering all six head components in BOTH
    directions. The standard set exercises only yaw (both ways) and chin
    down, and spends four of its seven blocks in the same centre pose, so a
    model fitted on it has never been shown what roll, translation or
    distance do to the features.

A ``PoseSet`` is immutable and slicing returns another ``PoseSet``: a run
that starts at block 9 gets a set of eight poses whose labels and
instructions are the ones it will actually present, and ``partial`` says so.
Coverage and pose weights are always computed downstream from the rows that
were recorded, never from the set that was planned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence


@dataclass(frozen=True)
class Pose:
    """One head pose: a label for the data, an instruction for the person."""

    label: str
    instruction: str

    def __iter__(self) -> Iterator[str]:
        """Unpacks as ``label, instruction`` so older tuple code still works."""

        yield self.label
        yield self.instruction


@dataclass(frozen=True)
class PoseSet:
    """An ordered, immutable sequence of poses, with a name and a provenance.

    ``partial`` is True when the set is a slice of a larger one -- a shortened
    run or a resume after an abort. It exists so a reader can tell a complete
    recording from a partial one without re-deriving it from row counts.
    """

    name: str
    poses: tuple[Pose, ...]
    partial: bool = False
    start: int = 1

    def __len__(self) -> int:
        return len(self.poses)

    def __iter__(self) -> Iterator[Pose]:
        return iter(self.poses)

    def __getitem__(self, index: int) -> Pose:
        return self.poses[index]

    def labels(self) -> list[str]:
        return [pose.label for pose in self.poses]

    def instructions(self) -> list[str]:
        """The exact strings the operator will be shown.

        Recorded in the manifest: the instruction is the experimental variable
        in a pose experiment, and a later reader cannot otherwise tell which
        wording produced the data.
        """

        return [pose.instruction for pose in self.poses]

    def as_tuples(self) -> tuple[tuple[str, str], ...]:
        """The legacy ``(label, instruction)`` shape."""

        return tuple((pose.label, pose.instruction) for pose in self.poses)

    def resolve(self, n_blocks: int | None = None, start: int = 1) -> "PoseSet":
        """The blocks a run will actually present, in order.

        ``n_blocks`` of None means all of them, so each set keeps its own
        length and no caller has to know how long any set is.

        ``start`` is 1-based and is a safety valve only: an aborted recording
        is frozen as aborted and cannot be resumed, so restarting from the
        block that failed is the alternative to recording everything again.
        The result is marked ``partial``.
        """

        if not 1 <= start <= len(self.poses):
            raise ValueError(
                f"start must be 1..{len(self.poses)} for pose set {self.name!r}, got {start}"
            )
        remaining = self.poses[start - 1 :]
        if n_blocks is None:
            chosen = remaining
        else:
            if not 1 <= n_blocks <= len(remaining):
                raise ValueError(
                    f"n_blocks must be 1..{len(remaining)} for pose set {self.name!r} "
                    f"starting at block {start}, got {n_blocks}"
                )
            chosen = remaining[:n_blocks]
        return PoseSet(
            name=self.name,
            poses=chosen,
            partial=len(chosen) != len(self.poses),
            start=start,
        )


def _build(name: str, entries: Sequence[tuple[str, str]]) -> PoseSet:
    return PoseSet(name=name, poses=tuple(Pose(label, text) for label, text in entries))


# --- the standard set: FROZEN -------------------------------------------------
#
# Instructions are deliberately plain and deliberately vague about magnitude
# ("a little"): the operator cannot see their own yaw_ratio, and asking for a
# number they cannot observe produces a guess recorded as a measurement.
STANDARD = _build(
    "standard",
    (
        ("centre", "Sit comfortably and face the screen straight on."),
        ("left", "Turn your head a LITTLE to the left. Stay comfortable."),
        ("centre_2", "Face the screen straight on again."),
        ("right", "Turn your head a LITTLE to the right. Stay comfortable."),
        ("centre_3", "Face the screen straight on again."),
        ("chin_down", "Tip your chin down a LITTLE. Keep your eyes on the dots."),
        ("centre_4", "Face the screen straight on again."),
    ),
)


# --- the extended set ---------------------------------------------------------
#
# Order is deliberate: the two directions of one axis are adjacent (2-3, 4-5,
# 6-7, 9-10, 11-12, 13-14) so that drift over the session pushes both
# directions of an axis the same way instead of masquerading as an axis bias.
# The three centre blocks (1, 8, 15) are rest, control, and a DIRECT drift
# measurement: comparing them says how much the operator moved over the
# recording, independently of the poses they were asked to hold. The break
# belongs at block 8; the prompt before every block is an unbounded wait that
# restarts the watchdog clock, so resting there costs nothing.
#
# The instructions are English at the operator's request. They were written in
# Hebrew first and the rendering was verified (real font, distinct glyphs, no
# mixed scripts), so the language here is a preference and not a limitation --
# but the person reading them mid-session is the one who has to act on them,
# so their preference decides. If they are ever put back into Hebrew, keep
# each line Hebrew-only: ``ui.pygame_display.rtl`` reverses the whole string
# rather than running a bidirectional algorithm, so a line mixing scripts is
# drawn with its Latin run backwards.
#
# TURN (blocks 2-3) and SLIDE (blocks 9-10) are capitalised on purpose. They
# are different movements producing different features, and if they are
# performed alike the analysis cannot separate them afterwards.
EXTENDED = _build(
    "extended",
    (
        ("centre", "Sit comfortably and face the screen straight on."),
        ("yaw_left", "TURN your head a LITTLE to the left. Keep your shoulders still."),
        ("yaw_right", "TURN your head a LITTLE to the right. Keep your shoulders still."),
        ("pitch_down", "Tip your chin down a LITTLE. Keep your eyes on the dots."),
        ("pitch_up", "Tip your chin up a LITTLE. Keep your eyes on the dots."),
        ("roll_left", "TILT your head a LITTLE toward your left shoulder. Do not turn it."),
        ("roll_right", "TILT your head a LITTLE toward your right shoulder. Do not turn it."),
        ("centre_mid", "Back to centre, facing the screen. Rest here as long as you need."),
        ("shift_left", "SLIDE your whole head a LITTLE to the left without turning it. Keep facing the screen."),
        ("shift_right", "SLIDE your whole head a LITTLE to the right without turning it. Keep facing the screen."),
        ("raise_head", "Sit up a LITTLE straighter so your head rises. Keep facing the screen."),
        ("lower_head", "Sink a LITTLE lower in the chair so your head drops. Keep facing the screen."),
        ("closer", "Lean a LITTLE closer to the screen. Keep facing it."),
        ("further", "Lean a LITTLE further back from the screen. Keep facing it."),
        ("centre_end", "Back to centre, facing the screen straight on."),
    ),
)


# --- the supplement set -------------------------------------------------------
#
# Added 2026-09-16 as a PLANNED IMPROVEMENT following analysis of the test
# recordings, and recorded as such: rounds 6, 7 and 8 were scored against the
# interval round5/B covered, and two components fell outside it systematically.
#
#   eye_mid_y  outside on ALL THREE tests (100 %, 100 %, 24 % of frames).
#              round5/B's own height blocks are why: "lower_head" produced a
#              median of 0.356 against the OPENING centre block's 0.371 -- the
#              operator sat HIGHER when asked to sit lower, because they reset
#              their posture first. The three centre blocks drifted 0.371 ->
#              0.350 -> 0.318 across the session, monotonically upward.
#   pitch_a    below the interval on round7 in 67 % of frames, where the
#              instruction involved no chin movement at all. The natural chin
#              position is simply lower than the calibration covered.
#
# So the instruction here names the failure directly: do NOT sit up first.
# The starting posture IS the thing being covered, and correcting it before
# the block is what made the original blocks miss.
#
# Two blocks only. This supplements a calibration, it does not replace one:
# the full 15-block recording stays valid and is pooled with this.
SUPPLEMENT = _build(
    "supplement",
    (
        (
            "settle_lower",
            "Do NOT sit up first. From exactly how you are sitting now, settle a LITTLE "
            "lower into the chair and stay there.",
        ),
        (
            "chin_natural_low",
            "Stay as you are and let your chin rest a LITTLE lower. Comfortable, not forced. "
            "Keep your eyes on the dots.",
        ),
    ),
)


POSE_SETS: Mapping[str, PoseSet] = {
    STANDARD.name: STANDARD,
    EXTENDED.name: EXTENDED,
    SUPPLEMENT.name: SUPPLEMENT,
}
DEFAULT_POSE_SET = STANDARD.name


def pose_set(name: str) -> PoseSet:
    try:
        return POSE_SETS[name]
    except KeyError:
        raise ValueError(
            f"unknown pose set {name!r}; known sets: {sorted(POSE_SETS)}"
        ) from None


def resolve_poses(
    name: str = DEFAULT_POSE_SET,
    n_blocks: int | None = None,
    start: int = 1,
) -> PoseSet:
    """Look a set up by name and slice it in one step."""

    return pose_set(name).resolve(n_blocks, start)
