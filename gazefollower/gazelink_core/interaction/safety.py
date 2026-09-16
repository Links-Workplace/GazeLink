"""Who may act, right now: the single owner of permission (ARCH-01 stage F).

``SafetyController`` owns the session's control mode (``ToggleMachine``), the
freshness of the face, the operator's activation (``--move-cursor --i-mean-it``
in wink mode) and the stopping flag. ``permission(now)`` answers from those,
at the moment it is asked -- the executor asks immediately before every input
it sends, never from a decision taken earlier (TECHNICAL_SPEC 15).

It makes no gaze decision and maps no gesture: it is told the events and the
face state and answers what they allow. Stopping is set under ``lock``, the
same re-entrant lock the executor holds while it sends, so an action already
being sent finishes before shutdown proceeds to the releases, and none starts
after (user amendment 3).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from gazelink_core.gaze.visibility import OVERLAY_STALE_S
from gazelink_core.interaction import control as CTL
from gazelink_core.interaction import gesture as GEST


@dataclass(frozen=True)
class Permission:
    """What the session allows at ``evaluated_s``. Re-read before every send."""

    evaluated_s: float
    input_enabled: bool
    controlled: bool
    cursor_enabled: bool
    selection_armed: bool
    face_ok: bool
    stopping: bool

    @property
    def may_select(self) -> bool:
        return self.input_enabled and self.selection_armed and not self.stopping

    @property
    def may_scroll(self) -> bool:
        return self.input_enabled and self.cursor_enabled and self.face_ok and not self.stopping

    @property
    def may_move_pointer(self) -> bool:
        return not self.stopping


class SafetyController:
    def __init__(
        self,
        *,
        control: CTL.ToggleMachine | None,
        input_enabled: bool,
        clock: Callable[[], float],
        stale_after_s: float = OVERLAY_STALE_S,
    ) -> None:
        self.control = control
        self.input_enabled = input_enabled
        self.clock = clock
        self.stale_after_s = stale_after_s
        self.lock = threading.RLock()
        self._stopping = False
        # The last face answer the loop computed this frame. The executor's
        # PAUSE/RESUME reads the same value the frame's tick used.
        self.face_ok = False

    # -- face -------------------------------------------------------------

    def evaluate_face(self, pipeline: Any) -> bool:
        """Is a face in front of the camera right now?

        Not the same question as "is there a usable gaze point": closing the
        eyes ends the point and not the face, and a caller that cannot tell
        them apart treats every deliberate close as a tracking failure and
        pauses in the same frame the close armed it. Frames must still be
        arriving, or a dead camera leaves the last answer standing and reads
        as a face for ever. Judged on the session's own clock.
        """

        state = pipeline.state
        if state.updated_s is None or self.clock() - state.updated_s > self.stale_after_s:
            ok = False
        else:
            ok = bool(state.face_present)
        self.face_ok = ok
        return ok

    # -- control mode -------------------------------------------------------

    @property
    def controlled(self) -> bool:
        return self.control is not None

    @property
    def label(self) -> str:
        return self.control.label if self.control is not None else ""

    @property
    def cursor_enabled(self) -> bool:
        return self.control is not None and self.control.mode.cursor_enabled

    @property
    def selection_armed(self) -> bool:
        return self.control is not None and self.control.mode.selection_armed

    def tick(self, events: Iterable[GEST.Event], *, face_ok: bool) -> list[CTL.Transition]:
        """ONE machine update per event, and at least one per frame."""

        assert self.control is not None
        return [
            self.control.update(event, tracking_ok=face_ok)
            for event in (list(events) or [GEST.Event.NONE])
        ]

    def toggle(self) -> CTL.Transition:
        """PAUSE/RESUME chosen as an action: the same event SPACE produces."""

        assert self.control is not None
        return self.control.update(GEST.Event.CONFIRM, tracking_ok=self.face_ok)

    # -- permission and stopping ------------------------------------------------

    def permission(self, now_s: float) -> Permission:
        with self.lock:
            return Permission(
                evaluated_s=now_s,
                input_enabled=self.input_enabled,
                controlled=self.controlled,
                cursor_enabled=self.cursor_enabled,
                selection_armed=self.selection_armed,
                face_ok=self.face_ok,
                stopping=self._stopping,
            )

    @property
    def stopping(self) -> bool:
        with self.lock:
            return self._stopping

    def begin_stop(self) -> None:
        """No new action after this returns; one being sent finishes first."""

        with self.lock:
            self._stopping = True
