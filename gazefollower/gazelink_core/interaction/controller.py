"""What the person is doing, and what their eyes ask for (ARCH-01 stage F).

``InteractionController`` is the single owner of the UI mode (cursor, menu,
scroll, keyboard) and of everything that belongs to it: the menu board, the
scanning keyboard, the scroll repeater, the content anchor, the seed rule and
the recovery menu of the no-click view. Each frame it turns the pipeline's
snapshot and events into control-machine ticks (through ``SafetyController``)
and into requests (through ``ActionExecutor``). It never touches an input
adapter, a window or the camera.

The existing interaction classes -- ``MenuModel``, ``ScanningKeyboard``,
``ScrollRepeater``, ``RecoveryMenu``, ``ActionRouter`` -- are used unchanged.
The order of every step below is the order the live loop has always used;
``tests/golden/live_trace.json`` holds it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from gazelink_core.gaze.visibility import OVERLAY_STALE_S, visible_point
from gazelink_core.interaction import actions as ACT
from gazelink_core.interaction import gesture as GEST
from gazelink_core.interaction import keyboard as KB
from gazelink_core.interaction import menu as MENU
from gazelink_core.interaction import scroll as SCR
from gazelink_core.interaction.executor import ActionExecutor
from gazelink_core.interaction.safety import SafetyController

VK_SPACE = 0x20


@dataclass(frozen=True)
class FrameView:
    """The points one frame shows: None whenever missing or stale."""

    fresh: tuple[float, float] | None
    raw_fresh: tuple[float, float] | None
    unfiltered: tuple[float, float] | None


class InteractionController:
    def __init__(
        self,
        *,
        options: Any,
        pipeline: Any,
        safety: SafetyController,
        executor: ActionExecutor,
        telemetry: Any,
        clock: Callable[[], float],
        key_is_down: Callable[[int], bool],
        to_pixels: Callable[[tuple[float, float]], tuple[int, int]] | None,
        board: MENU.MenuModel,
        keyboard: KB.ScanningKeyboard,
        scroll_cfg: SCR.ScrollConfig,
        recovery_menu: GEST.RecoveryMenu,
    ) -> None:
        self.options = options
        self.pipeline = pipeline
        self.safety = safety
        self.executor = executor
        self.telemetry = telemetry
        self.clock = clock
        self.key_is_down = key_is_down
        self.to_pixels = to_pixels
        self.board = board
        self.keyboard = keyboard
        self.scroll_cfg = scroll_cfg
        self.scroll_bands = SCR.scroll_zones(scroll_cfg)
        self.repeater = SCR.ScrollRepeater(scroll_cfg)
        self.recovery_menu = recovery_menu
        # Chosen in the no-click view's recovery menu (e.g. "recalibrate").
        self.chosen: list[str] = []
        # Edge, not level: a key held for half a second is one instruction,
        # and reading the level would toggle the mode on every frame it is
        # down -- sixty times a second, ending wherever the release happened.
        self.space_was_down = False
        # What the person is DOING, which is orthogonal to the ToggleMachine
        # on purpose and stays so. Adding modes to gf_control would change the
        # meaning of ``cursor_enabled`` everywhere to express something that is
        # not a kind of "how much control does the person have".
        #
        # Opening straight into scroll mode skips the menu, for when the
        # session is only for reading. The pointer is left exactly where it
        # was found -- there is no content point yet to jump back to.
        self.ui_mode = (
            ACT.UiMode.SCROLL
            if bool(options.start_scrolling) and options.click_by == "wink"
            else ACT.UiMode.CURSOR
        )
        # Where the pointer was while the gaze was last on CONTENT, in pixels.
        self.content_px: tuple[int, int] | None = None
        # The last thing that happened that the person needs to know about and
        # cannot see for themselves. Shown until the next one replaces it.
        self.notice: str | None = None
        # The same "seed the gaze first" rule the menu has, applied to the way
        # INTO scrolling: nothing scrolls until the gaze has been seen in the
        # dead middle at least once. True at the START of a session on purpose:
        # with --start-scrolling there is no tile the gaze was left on.
        self.scroll_seeded = True
        telemetry.seen_in.update({SCR.UP: 0, SCR.DOWN: 0, "the middle": 0, "no point": 0})

    # -- mode changes ----------------------------------------------------------

    def enter_mode(self, new_mode: ACT.UiMode, *, why: str = "") -> None:
        """Change what the person is doing, and clear everything in flight.

        Every one of these is a safety property already learned the hard way:
        the wheel stops, or it keeps repeating into the next mode; every wink
        is dropped -- fired AND in progress -- so a closure begun in one mode
        cannot act in the next; the router forgets what was asked recently;
        the scanning keyboard starts from the top with an empty echo.
        """

        if new_mode is self.ui_mode:
            return
        self.ui_mode = new_mode
        self.scroll_seeded = new_mode is not ACT.UiMode.SCROLL
        self.repeater.stop()
        self.executor.reset_router()
        self.telemetry.tally["winks_dropped_at_mode_edge"] += self.pipeline.cancel_wink()
        self.keyboard.reset(keep_text=False)
        if new_mode is not ACT.UiMode.MENU:
            self.board.close()
        if new_mode is ACT.UiMode.KEYBOARD:
            self.notice = None
            # Locked HERE, once, at the moment the person asks to type.
            self.executor.lock_keyboard_target()
            self.telemetry.say(f"typing into: {self.executor.keyboard_target_name}")
        if (
            new_mode is ACT.UiMode.SCROLL
            and self.content_px is not None
            and self.safety.cursor_enabled
        ):
            # Back to the content: the wheel goes to whatever is under the
            # pointer. Gated on the mode -- the jump ignores ``paused`` by
            # design, and nothing may move while paused.
            self.executor.return_to(self.content_px)
        self.telemetry.say(f"mode {self.ui_mode}{f' - {why}' if why else ''}")
        if new_mode is ACT.UiMode.SCROLL:
            self.telemetry.tally["scroll_sessions"] += 1

    def take_choice(self, choice: MENU.Choice, at: tuple[float, float] | None, now: float) -> None:
        """What a menu tile does, once the dwell has chosen it."""

        item = choice.item
        if item.effect is MENU.Effect.PAGE:
            # The only outcome that leaves the menu open.
            return
        if item.effect is MENU.Effect.CLICK_TYPE:
            self.telemetry.say(f"click type: {self.board.click_type}")
            self.enter_mode(ACT.UiMode.CURSOR, why="click type chosen")
            return
        if item.effect is MENU.Effect.MODE and item.mode is not None:
            self.enter_mode(item.mode, why="chosen in the menu")
            return
        if item.action is not None and self.executor.route(
            item.action, ACT.Source.DWELL, at, now, ui_mode=self.ui_mode
        ):
            self.telemetry.tally["commands"] += 1
        self.enter_mode(ACT.UiMode.CURSOR, why="menu closed")

    # -- one frame ---------------------------------------------------------------

    def tick_control(self, state: Any) -> bool:
        """Gestures, keys and the control machine. True when the session should end."""

        tally = self.telemetry.tally
        if self.safety.controlled:
            # ONE tick per frame, always, whether or not anything happened: the
            # machine clears the guard that follows a tracking loss on the first
            # clean frame it is GIVEN.
            events: list[GEST.Event] = []
            if self.options.toggle_by == "eyes":
                events = [e for _when, e in self.pipeline.drain_gesture_events()]
            else:
                # The long close is computed on the camera thread whatever the
                # toggle is; here it opens the menu (no new detector, no new
                # threshold: natural blinks reached 297 ms, deliberate holds
                # started at 1172 ms, and the confirm sits at 800 ms).
                long_closes = sum(
                    1
                    for _when, event in self.pipeline.drain_gesture_events()
                    if event is GEST.Event.CONFIRM
                )
                if long_closes and self.options.menu_enabled:
                    # Deliberately NOT gated on cursor_enabled: the menu has to
                    # open while paused or there is no way back without a
                    # keyboard. Nothing chosen in it reaches the OS while
                    # paused: the router refuses everything but RESUME.
                    paused_now = not self.safety.cursor_enabled
                    if self.ui_mode is ACT.UiMode.MENU:
                        # A long close OPENS the menu and does not close it
                        # (section 57: looking up at a tile raises the chin and
                        # reads as both eyes shut 800 ms later).
                        tally["menu_close_ignored"] += 1
                    else:
                        self.board.show(paused=paused_now)
                        tally["menu_opened"] += 1
                        self.enter_mode(ACT.UiMode.MENU, why="long close")
                down = self.key_is_down(VK_SPACE)
                if down and not self.space_was_down:
                    events = [GEST.Event.CONFIRM]
                self.space_was_down = down
            face_ok = self.safety.evaluate_face(self.pipeline)
            if state.openness_ratio is not None:
                # getattr: a test double may carry only the ratios, and a
                # missing pose must read as "unknown", never as a number.
                self.telemetry.ratios_seen.append(
                    (*state.openness_ratio, getattr(state, "head_pitch", None))
                )
            mode_changed = False
            for transition in self.safety.tick(events, face_ok=face_ok):
                if transition.changed:
                    mode_changed = True
                    self.telemetry.say(self.safety.label)
                if transition.cancel_selection:
                    tally["cancelled"] += 1
            if mode_changed:
                # A pause, a resume or a tracking loss/recovery invalidates
                # every wink in flight, exactly as a change of UI mode does.
                # Before (TECHNICAL_SPEC 15): a wink published while PAUSED and
                # drained in the frame SPACE resumed passed the now-armed check
                # and clicked -- an event from before the recovery acting after.
                tally["winks_dropped_at_mode_edge"] += self.pipeline.cancel_wink()
                self.executor.reset_router()
            self.executor.set_pointer_paused(not self.safety.cursor_enabled)
        else:
            for when, event in self.pipeline.drain_gesture_events():
                taken = self.recovery_menu.handle(event, when)
                if taken is None or taken.key == "dismiss":
                    continue
                if taken.key == "pause":
                    paused = not self.executor.pointer_paused
                    self.executor.set_pointer_paused(paused)
                    self.telemetry.say(f"cursor {'paused' if paused else 'resumed'}")
                    continue
                self.chosen.append(taken.key)
            self.recovery_menu.tick(self.clock())
        return bool(self.chosen)

    def tick_pointer(self, state: Any) -> FrameView:
        """Pointer, menu, keyboard, scroll and winks for this frame."""

        view = FrameView(
            visible_point(state.point, state.updated_s, self.clock(), OVERLAY_STALE_S),
            visible_point(state.raw_model, state.updated_s, self.clock(), OVERLAY_STALE_S),
            visible_point(state.unfiltered, state.updated_s, self.clock(), OVERLAY_STALE_S),
        )
        # None whenever the point is missing or stale: the adapter reads that
        # as "freeze", never as "move somewhere plausible".
        if not (self.executor.pointer_enabled and self.to_pixels is not None):
            return view
        tally = self.telemetry.tally
        fresh = view.fresh
        # None means freeze. An eye on its way down still passes the blink
        # gate, so ``fresh`` is a real point made from an eye already half
        # behind its lid; following it makes the pointer lurch as a wink
        # starts. Only a steady point may move the pointer or ask for anything.
        steady = not self.safety.controlled or state.eyes_steady
        aim = fresh if steady else None
        now = self.clock()
        if self.safety.controlled:
            zone = next(
                (z.key for z in self.scroll_bands if aim is not None and z.contains(aim)),
                None,
            )
            paused_now = not self.safety.cursor_enabled
            # The pause tile is the resume tile, so it has to keep telling the
            # truth about a mode that can change under an open menu.
            self.board.set_paused(paused_now)
            if not self.safety.face_ok or paused_now:
                # The wheel stops either way. The MODE is not cancelled with
                # it: the face is "not ok" for the first frames of EVERY
                # session, and cancelling on that switched --start-scrolling
                # off a moment after it started.
                self.repeater.stop()
            if self.ui_mode is ACT.UiMode.MENU:
                # Frozen while choosing: the pointer must not travel to the
                # tile the gaze is inspecting.
                self.executor.set_pointer_paused(True)
                choice = self.board.update(now, aim, fresh=aim is not None)
                if choice is not None:
                    tally["menu_choices"] += 1
                    self.take_choice(choice, aim, now)
            elif self.ui_mode is ACT.UiMode.KEYBOARD:
                # The gaze chooses nothing here; the pointer stays over the
                # window being typed into.
                self.executor.set_pointer_paused(True)
                if self.executor.keyboard_target_lost:
                    # Something else took the keyboard. Leave: in cursor mode
                    # the person can click the window they want and open the
                    # keyboard again, which locks the target they chose.
                    tally["keyboard_target_lost"] += 1
                    self.notice = "the window being typed into changed - the keyboard closed"
                    self.telemetry.say(self.notice)
                    self.enter_mode(ACT.UiMode.CURSOR, why=self.notice)
                else:
                    self.keyboard.update(now)
            elif self.ui_mode is ACT.UiMode.SCROLL:
                # Where the gaze actually went while scroll mode was on.
                seen_in = self.telemetry.seen_in
                if aim is None:
                    seen_in["no point"] += 1
                elif zone is not None:
                    seen_in[zone] += 1
                else:
                    seen_in["the middle"] += 1
                # Frozen for the whole of scroll mode, so the wheel keeps going
                # to the window the person aimed at.
                self.executor.set_pointer_paused(True)
                if aim is not None and zone is None:
                    # Seen in the dead middle: from here the bands are live.
                    self.scroll_seeded = True
                notches = self.repeater.update(
                    now, zone, usable=aim is not None and self.scroll_seeded
                )
                if notches and self.executor.scroll(notches):
                    tally["notches"] += abs(notches)
            elif aim is not None and zone is None:
                # The pointer's position while the gaze is on CONTENT.
                self.content_px = self.to_pixels(aim)
        if self.ui_mode is ACT.UiMode.CURSOR:
            self.executor.follow(
                self.to_pixels(fresh) if fresh is not None and steady else None
            )
        for when, aimed_at in self.pipeline.drain_wink_events():
            if not self.safety.controlled:
                continue
            self._take_wink(when, aimed_at, now)
        return view

    def _take_wink(self, when: float, aimed_at: tuple[float, float] | None, now: float) -> None:
        tally = self.telemetry.tally
        tally["winks"] += 1
        # The time the WINK happened, not the time it was looked at, and
        # screened before the keyboard's ``select`` changes state as it is
        # called: a stale wink refused afterwards would already have moved the
        # scan into a group nobody chose.
        refusal = self.executor.router.screen_gesture(
            issued_at_s=when,
            now_s=now,
            ui_mode=self.ui_mode,
            paused=not self.safety.selection_armed,
        )
        if refusal is ACT.Refusal.STALE:
            tally["winks_stale"] += 1
            return
        if refusal is ACT.Refusal.WRONG_MODE:
            if self.ui_mode is ACT.UiMode.MENU:
                tally["winks_in_menu"] += 1
            else:
                tally["winks_while_scrolling"] += 1
            return
        if refusal is ACT.Refusal.PAUSED:
            tally["suppressed_while_paused"] += 1
            return
        if self.ui_mode is ACT.UiMode.KEYBOARD:
            # The wink's OWN moment, not the moment the queue was drained: at
            # 1200 ms a cell, the difference moves the highlight on.
            key = self.keyboard.select(now, when_s=when)
            if key is None:
                # A group was opened, or a parked sweep restarted.
                return
            tally["keys_chosen"] += 1
            if key.control is KB.Control.CLOSE:
                self.enter_mode(ACT.UiMode.CURSOR, why="keyboard closed")
                return
            if key.control is not None:
                # Cancel and layout change the keyboard and nothing else.
                return
            action = ACT.Action.TYPE_TEXT if key.text is not None else key.action
            if action is not None and self.executor.route(
                action, ACT.Source.SCAN, aimed_at, when, ui_mode=self.ui_mode, text=key.text
            ):
                # Echoed only once it actually left.
                self.keyboard.on_sent(key)
            return
        self.executor.route(
            MENU.CLICK_ACTION[self.board.click_type],
            ACT.Source.WINK,
            aimed_at,
            when,
            ui_mode=self.ui_mode,
        )
