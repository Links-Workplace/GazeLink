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
from gazelink_core.interaction import bar as BAR
from gazelink_core.interaction import drag as DRAG
from gazelink_core.interaction import gesture as GEST
from gazelink_core.interaction import keyboard as KB
from gazelink_core.interaction import menu as MENU
from gazelink_core.interaction import scroll as SCR
from gazelink_core.interaction import steadiness as STEADY
from gazelink_core.interaction.executor import ActionExecutor
from gazelink_core.interaction.safety import SafetyController

VK_SPACE = 0x20

# Which counter a wink refused by mode belongs to. A dict rather than a chain
# of ifs so a new UiMode cannot be added without a place to count it.
WINK_REFUSED_IN = {
    ACT.UiMode.MENU: "winks_in_menu",
    ACT.UiMode.SCROLL: "winks_while_scrolling",
    ACT.UiMode.ZOOM: "winks_in_zoom",
    ACT.UiMode.DRAG: "winks_in_drag",
}


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
        bar: Any = None,
        spatial_keyboard: Any = None,
        zoom: Any = None,
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
        # The desk components, or None when --desk is off. Optional rather
        # than always built so the existing path constructs exactly what it
        # always constructed and the golden trace cannot move.
        self.bar = bar
        self.spatial_keyboard = spatial_keyboard
        self.zoom = zoom
        # Which bar modes this controller can actually carry out. Derived from
        # what was HANDED to it rather than hard-coded, so wiring the magnifier
        # in makes its target work without a second place to remember.
        self.bar_modes_wired = {ACT.UiMode.SCROLL, ACT.UiMode.KEYBOARD}
        if zoom is not None:
            self.bar_modes_wired.add(ACT.UiMode.ZOOM)
        # Dwell only. The pointer and every wink keep the strict steadiness
        # gate; a fill is the one thing a 32 ms blink must not send back to
        # zero. See ``steadiness`` for the measurement that made this needed.
        self.steady_hold = STEADY.SteadyHold(
            hold_ms=getattr(options, "steady_hold_ms", STEADY.DEFAULT_HOLD_MS)
        )
        # Owned here, like the menu and the keyboard: one place knows whether
        # something is being carried, and every exit path goes through it.
        self.drag = DRAG.DragSequence()
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
        # Before anything else. Every other in-flight mechanism is stopped
        # below; a held mouse button was the one that was not, and leaving
        # DRAG with the button down turns every later look into a selection.
        self._release_drag(DRAG.Ended.CANCELLED)
        self.ui_mode = new_mode
        # Nothing carried behind an eyelid may survive into the next mode, for
        # the same reason every wink in flight is dropped below.
        self.steady_hold.reset()
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
            if self.bar is not None:
                # One setting, shown in two places: the bar's tile must say
                # what the menu just switched.
                self.bar.click_kind = self.board.click_type
            self.telemetry.say(f"left wink: {self.board.click_type} click")
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

    def take_bar_pick(
        self, pick: BAR.Pick, at: tuple[float, float] | None, now: float
    ) -> None:
        """What a bar target does, once the dwell has chosen it.

        The bar has already changed ITSELF by the time this is called
        (``ControlBar._apply`` moves the view and records the click kind). What
        is left is the two things only the controller owns: the mode, and
        putting an Action through the router.

        Unlike ``take_choice`` this never returns to CURSOR at the end. The bar
        is not a trip: it is on screen the whole time, and a choice that did
        not ask for a mode leaves the person exactly where they were.
        """

        item = pick.item
        if item.effect is BAR.Effect.VIEW:
            # Compact/expanded is the bar's own business.
            return
        if item.effect is BAR.Effect.CLICK_KIND:
            # ONE source of truth for what a confirm does: ``_take_wink`` reads
            # ``board.click_type``, so the bar writes there rather than keeping
            # a second kind that nothing would read.
            self.board.click_type = self.bar.click_kind
            self.telemetry.say(f"click type: {self.board.click_type}")
            return
        if item.effect is BAR.Effect.MODE and item.mode is not None:
            if item.mode not in self.bar_modes_wired:
                # A target that leads nowhere is worse than a target that is
                # not offered: ZOOM has no renderer and nothing drives it, and
                # DRAG has no way to START a carry (a wink is refused in that
                # mode by design), so entering either would strand the person
                # in a mode with no way out but Esc. Refused OUT LOUD.
                self.telemetry.tally["bar_modes_not_ready"] += 1
                self.notice = f"{item.label} עדיין לא מחובר - לא נכנסנו למצב הזה"
                self.telemetry.say(f"bar: {item.mode} is not wired yet")
                return
            self.enter_mode(item.mode, why="chosen on the bar")
            return
        if item.action is not None and self.executor.route(
            item.action, ACT.Source.DWELL, at, now, ui_mode=self.ui_mode
        ):
            self.telemetry.tally["commands"] += 1

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
                # A pause, a resume or a tracking loss invalidates a held point
                # exactly as it invalidates a wink: after a loss the eyes may be
                # anywhere, and the last place they were open is not evidence.
                self.steady_hold.reset()
                self.executor.reset_router()
                # A pause, a resume or a tracking loss invalidates a carry as
                # surely as it invalidates a wink. The button comes up here
                # and not at shutdown, because shutdown may be minutes away.
                self._release_drag(
                    DRAG.Ended.PAUSED if not self.safety.cursor_enabled
                    else DRAG.Ended.TRACKING_LOST
                )
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
        # What a DWELL may use: ``aim``, or the last open-eyed point for as long
        # as the budget allows. ``aim`` itself is untouched, so everything that
        # moves the pointer or acts on a wink still sees only a steady point.
        dwell_aim = self.steady_hold.update(
            now, steady_point=aim, have_prediction=fresh is not None
        )
        if self.safety.controlled:
            zone = next(
                (z.key for z in self.scroll_bands if aim is not None and z.contains(aim)),
                None,
            )
            paused_now = not self.safety.cursor_enabled
            # The pause tile is the resume tile, so it has to keep telling the
            # truth about a mode that can change under an open menu.
            self.board.set_paused(paused_now)
            if self.bar is not None:
                # Same reason, same frame: the bar's pause target is the way
                # back, and it may not describe a mode that already changed.
                self.bar.set_paused(paused_now)
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
                choice = self.board.update(now, dwell_aim, fresh=dwell_aim is not None)
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
            elif self.bar is not None and self.ui_mode is ACT.UiMode.CURSOR:
                # The bar is on screen for the whole of cursor mode, so it is
                # driven every frame -- including while PAUSED, because the
                # resume target lives on it and the router already refuses
                # everything but RESUME while paused. Its own arming rule (see
                # ``bar`` module docstring) is what stops a glance downwards
                # from choosing something.
                pick = self.bar.update(now, dwell_aim, fresh=dwell_aim is not None)
                if pick is not None:
                    tally["bar_picks"] += 1
                    self.take_bar_pick(pick, aim, now)
                elif aim is not None and self.bar.hovered is None:
                    # On CONTENT rather than on a target: worth remembering as
                    # the place the wheel should go back to.
                    self.content_px = self.to_pixels(aim)
            elif aim is not None and zone is None:
                # The pointer's position while the gaze is on CONTENT.
                self.content_px = self.to_pixels(aim)
        if self.ui_mode in (ACT.UiMode.CURSOR, ACT.UiMode.DRAG):
            # DRAG follows too. Without it the pointer is frozen for the whole
            # carry and the item never goes anywhere -- a drag that cannot
            # move is a button held for nothing.
            self.executor.follow(
                self.to_pixels(fresh) if fresh is not None and steady else None
            )
        for event in self.pipeline.drain_wink_events():
            if not self.safety.controlled:
                continue
            # (when, aimed_at, eye). An event without an eye comes from a source
            # that only ever detected the right eye, and is read as that.
            when, aimed_at, *rest = event
            eye = GEST.Eye(rest[0]) if rest else GEST.Eye.RIGHT
            self._take_wink(when, aimed_at, now, eye=eye)
        return view

    def _release_drag(self, why: DRAG.Ended) -> None:
        """Let go, and record why. Safe to call when nothing is held."""

        held = self.drag.force_release(why)
        released = self.executor.end_drag()
        if held or released:
            self.notice = f"הגרירה שוחררה: {why}"
            self.telemetry.say(f"drag released - {why}")

    def _take_wink(
        self,
        when: float,
        aimed_at: tuple[float, float] | None,
        now: float,
        *,
        eye: GEST.Eye = GEST.Eye.RIGHT,
    ) -> None:
        tally = self.telemetry.tally
        tally["winks"] += 1
        tally["winks_left" if eye is GEST.Eye.LEFT else "winks_right"] += 1
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
            # One counter per mode. Before, everything that was not the menu
            # was counted as a wink while scrolling, so a report from a zoom
            # session blamed the wheel for winks the wheel never saw.
            tally[WINK_REFUSED_IN.get(self.ui_mode, "winks_while_scrolling")] += 1
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
            self.click_for(eye),
            ACT.Source.WINK,
            aimed_at,
            when,
            ui_mode=self.ui_mode,
        )

    def click_for(self, eye: GEST.Eye) -> ACT.Action:
        """The eye picks the button. RIGHT is always a right click; LEFT is a left
        click, single or double as the menu's toggle says (single by default).

        One mapping, in one place, so the menu label, the bar label and what a
        wink actually sends cannot drift apart.
        """

        if eye is GEST.Eye.RIGHT:
            return ACT.Action.RIGHT_CLICK
        return (
            ACT.Action.DOUBLE_CLICK
            if self.board.click_type == "double"
            else ACT.Action.LEFT_CLICK
        )
