"""Carry out what the router allowed, with the permission of this instant.

``ActionExecutor`` is the only caller of the input adapters in a live session
(ARCH-01 stage F). For every send it reads a FRESH ``Permission`` from the
``SafetyController`` under its lock and hands the adapter the real answer --
never a constant ``armed=True`` (TECHNICAL_SPEC 15). The adapters keep their
own local interlocks (minimum gaps, held-key checks, target lock) on top.

Releases are not here: they belong to shutdown (``CleanupStack``) and must
work whatever the permission says.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gazelink_core.interaction import actions as ACT
from gazelink_core.interaction.safety import SafetyController


class ActionExecutor:
    def __init__(
        self,
        *,
        safety: SafetyController,
        router: ACT.ActionRouter,
        cursor: Any,
        clicker: Any,
        scroller: Any,
        keys: Any,
        to_pixels: Callable[[tuple[float, float]], tuple[int, int]],
        telemetry: Any,
        clock: Callable[[], float],
        hold_after_click_s: float,
    ) -> None:
        self.safety = safety
        self.router = router
        self.cursor = cursor
        self.clicker = clicker
        self.scroller = scroller
        self.keys = keys
        self.to_pixels = to_pixels
        self.telemetry = telemetry
        self.clock = clock
        self.hold_after_click_s = hold_after_click_s

    # -- requests through the router -------------------------------------------

    def route(
        self,
        action: ACT.Action,
        source: ACT.Source,
        at: tuple[float, float] | None,
        issued_s: float,
        *,
        ui_mode: ACT.UiMode,
        text: str | None = None,
    ) -> bool:
        """Ask, then do. Every refusal is counted by the router.

        ``issued_s`` is when the thing that asked HAPPENED -- the moment the
        wink fired, not the moment the loop got round to it. The two differ by
        a queue, and the difference is the whole of the router's staleness
        rule.
        """

        with self.safety.lock:
            if self.safety.stopping:
                return False
            request = ACT.ActionRequest(action, source, issued_s, at=at, text=text)
            now_s = self.clock()
            permission = self.safety.permission(now_s)
            decision = self.router.decide(
                request,
                paused=not permission.selection_armed,
                ui_mode=ui_mode,
                now_s=now_s,
            )
            if not decision.accepted:
                return False
            return self._perform(action, at, text, permission.selection_armed)

    def _perform(
        self, action: ACT.Action, at: tuple[float, float] | None, text: str | None, armed: bool
    ) -> bool:
        """Carry out an action the router has already allowed.

        Split from the deciding on purpose, the same split ``gf_click`` makes
        between "who decides where" and "who presses". Each adapter still
        applies its own rules on top. The WHEEL is deliberately absent: a
        scroll repeats several times a second by design, so the router's
        duplicate rule would refuse almost every notch; its rate limit lives in
        ``ScrollAdapter``.
        """

        tally = self.telemetry.tally
        A = ACT.Action
        if action in ACT.CLICKS:
            if at is None:
                # The gaze is invalid while an eye is shut, and with nothing
                # remembered from before it there is no place the wink can
                # honestly mean. Counted, because "no wink was seen" and "a
                # wink was seen and had nowhere to go" need different fixes.
                tally["winks_with_no_aim"] += 1
                return False
            # Put the pointer where the eye was when the wink STARTED and click
            # there. Clicking wherever the pointer drifted to is how a click
            # lands next to what was being looked at.
            landing = self.to_pixels(at)
            # Release first: the pointer must reach the new place before the
            # hold pins it there, or a second wink would click wherever the
            # FIRST one landed.
            self.cursor.release_hold()
            # jump_to, not update: smoothing would leave the pointer short and
            # the dead zone would refuse small corrections.
            self.cursor.jump_to(landing)
            if action is A.DOUBLE_CLICK:
                landed = bool(self.clicker.double_click(armed=armed, at=self.cursor.last))
            elif action is A.RIGHT_CLICK:
                landed = bool(self.clicker.right_click(armed=armed, at=self.cursor.last))
            else:
                landed = bool(self.clicker.click(armed=armed, at=self.cursor.last))
            if landed:
                tally["clicks"] += 1
                tally["doubles"] = self.clicker.doubles
                tally["right_clicks"] = self.clicker.right_clicks
                # Hold after, not before: the click has landed, and now the
                # person needs a moment to see what happened.
                self.cursor.hold_for(self.hold_after_click_s)
            return landed
        if action is A.TYPE_TEXT:
            if self.keys.type_text(text or "", armed=armed):
                tally["chars_typed"] += len(text or "")
                return True
            return False
        if action is A.BACKSPACE:
            return bool(self.keys.backspace(armed=armed))
        if action is A.ENTER:
            return bool(self.keys.enter(armed=armed))
        if action in ACT.COMMANDS:
            # Locked immediately before, unlike the keyboard's target: a command
            # is one keystroke to whatever the person is looking at now.
            self.keys.lock_target()
            return bool({
                A.BACK: self.keys.back,
                A.FORWARD: self.keys.forward,
                A.SWITCH_WINDOW: self.keys.switch_window,
                A.ESCAPE: self.keys.escape,
            }[action](armed=armed))
        if action is A.DRAG_START:
            if at is None:
                # A pick with nowhere to mean would grab whatever the pointer
                # was last left on. Counted the same way a wink with no aim is.
                tally["winks_with_no_aim"] += 1
                return False
            self.cursor.release_hold()
            self.cursor.jump_to(self.to_pixels(at))
            if self.clicker.press(armed=armed):
                tally["drags_started"] += 1
                return True
            return False
        if action in (A.PAUSE, A.RESUME):
            # Through the machine that already owns this, with the event SPACE
            # produces. A second mode here would be a second answer to "how
            # much control does the person have".
            transition = self.safety.toggle()
            if transition.changed:
                # Nothing may stay held across a pause. Done BEFORE the label
                # is printed, so a failure to say so cannot skip the release.
                self.end_drag()
                self.telemetry.say(self.safety.label)
            if transition.cancel_selection:
                tally["cancelled"] += 1
            self.cursor.paused = not self.safety.cursor_enabled
            return bool(transition.changed)
        return False

    def reset_router(self) -> None:
        self.router.reset()

    # -- letting go, which is never an action ----------------------------------

    def end_drag(self) -> bool:
        """Release a held button. Deliberately NOT through the router.

        ``route`` refuses everything while stopping and everything but RESUME
        while paused -- and a pause, a lost face and a shutdown are exactly the
        moments a held button most needs to come up. A release undoes an
        action rather than being one, which is the same rule
        ``platform.real_input.require`` already applies to button-up.

        Idempotent: several safety paths can fire for one event.
        """

        if not self.clicker.dragging:
            return False
        released = bool(self.clicker.drag_release())
        if released:
            self.telemetry.tally["drags_released"] += 1
        else:
            # The up did not land. The adapter keeps naming the button in
            # ``stuck_button`` and shutdown reports it rather than losing it.
            self.telemetry.tally["drags_stuck"] += 1
        return released

    @property
    def dragging(self) -> bool:
        return bool(self.clicker.dragging)

    # -- the pointer ---------------------------------------------------------

    @property
    def pointer_enabled(self) -> bool:
        return bool(self.cursor.enabled)

    @property
    def pointer_held(self) -> Any:
        return self.cursor.held_until_s

    @property
    def pointer_paused(self) -> bool:
        return bool(self.cursor.paused)

    def set_pointer_paused(self, paused: bool) -> None:
        self.cursor.paused = paused

    def follow(self, pixels: tuple[int, int] | None) -> None:
        """Follow the gaze; None freezes. Nothing moves once stopping."""

        with self.safety.lock:
            if self.safety.stopping:
                return
            self.cursor.update(pixels)

    def return_to(self, pixels: tuple[int, int]) -> None:
        """Put the pointer back on the content (entering scroll mode)."""

        with self.safety.lock:
            if self.safety.stopping:
                return
            self.cursor.release_hold()
            self.cursor.jump_to(pixels)

    # -- the wheel -----------------------------------------------------------

    def scroll(self, notches: int) -> bool:
        with self.safety.lock:
            permission = self.safety.permission(self.clock())
            return bool(self.scroller.scroll(notches, armed=permission.may_scroll))

    # -- the keyboard's target -------------------------------------------------

    def lock_keyboard_target(self) -> None:
        self.keys.lock_target()

    @property
    def keyboard_target_lost(self) -> bool:
        return bool(self.keys.target_lost)

    @property
    def keyboard_target_name(self) -> str:
        return str(self.keys.target_name)
