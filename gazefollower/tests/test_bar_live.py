"""The desk bar drives something and is drawn — the wiring, not the geometry.

``test_desk_wiring`` proves the flag reaches the session and the components get
built. It cannot tell whether anything ever CALLS them, and for a while nothing
did: ``ControlBar`` was constructed, handed to the controller, stored on an
attribute and never read, and ``LivePresenter`` had no branch that drew it.
``--desk`` ran without error and showed nothing.

So the tests here are about being driven and being drawn:

* a pick reaches the thing that carries it out, and the CLICK KIND lands where
  a wink actually reads it (one source of truth, not two);
* a target whose machinery is missing is refused OUT LOUD instead of stranding
  the person in a mode with no renderer and no way out;
* the bar is driven in cursor mode only, so it cannot pick while the menu is up;
* the way back works while paused, because the resume target lives on the bar.

Fakes only: FakeClock, recorder senders, no camera, no window, no OS input.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from gazelink_core.interaction import actions as ACT  # noqa: E402
from gazelink_core.interaction import bar as BAR  # noqa: E402
from gazelink_core.interaction import desk_layout as L  # noqa: E402
from test_interaction_safety import _Rig  # noqa: E402

# Content means "outside every target this module owns", and it has to be
# outside them in BOTH views: the screen centre is inside the expanded menu's
# "settings" tile, so using it as content silently never re-armed the bar.
CONTENT = (0.15, 0.60)


def centre_of(key: str, buttons: list) -> tuple[float, float]:  # noqa: ANN001
    button = next(b for b in buttons if b.key == key)
    return (button.x0 + button.width / 2.0, button.y0 + button.height / 2.0)


class _Desk:
    """A rig whose controller has a bar, as ``--desk`` builds it."""

    def __init__(self, *, start_active: bool = True, zoom: object | None = None) -> None:
        self.rig = _Rig(start_active=start_active)
        self.controller = self.rig.controller
        self.bar = BAR.ControlBar(dwell_ms=900.0)
        self.controller.bar = self.bar
        if zoom is not None:
            self.controller.zoom = zoom
            self.controller.bar_modes_wired.add(ACT.UiMode.ZOOM)

    def look(
        self,
        point: tuple[float, float] | None,
        *,
        seconds: float = 0.1,
        steady: bool = True,
    ) -> None:
        """One frame of gaze, through the real per-frame path."""

        self.rig.pipeline.point = point
        self.rig.clock.advance(seconds)
        # ``updated_s`` is what makes the point CURRENT. Left unset, every
        # frame here was silently stale and the first version of these tests
        # was measuring nothing at all.
        self.controller.tick_pointer(
            _state(point, self.rig.clock.now(), steady=steady)
        )

    def stare(self, point: tuple[float, float], *, frames: int = 16) -> None:
        """Long enough for a 900 ms dwell to complete, one frame at a time."""

        for _ in range(frames):
            self.look(point)

    def picks(self) -> int:
        return int(self.rig.telemetry.tally["bar_picks"])

    def choose(self, key: str) -> None:
        """Arm on content, then hold the named target until it fires -- and STOP.

        Staring on past the pick is not what a person does, and it changes the
        answer: when the view flips under the gaze, the old target's rectangle
        may be content in the new layout, which re-arms the bar. Held here so
        the arming test measures the rule rather than that overshoot.
        """

        self.look(CONTENT)
        before = self.picks()
        target = centre_of(key, self.bar.buttons)
        for _ in range(16):
            self.look(target)
            if self.picks() > before:
                return


def _state(point: tuple[float, float] | None, now_s: float, *, steady: bool = True):  # noqa: ANN202
    from types import SimpleNamespace

    return SimpleNamespace(
        point=point,
        raw_model=point,
        unfiltered=point,
        updated_s=now_s,
        eyes_steady=steady,
        openness_ratio=None,
        frames=1,
        fps=30.0,
    )


class ThePickReachesSomethingTests(unittest.TestCase):
    def test_choosing_scroll_enters_scroll_mode(self) -> None:
        desk = _Desk()
        desk.choose("scroll")
        self.assertIs(desk.controller.ui_mode, ACT.UiMode.SCROLL)

    def test_choosing_keyboard_enters_keyboard_mode(self) -> None:
        desk = _Desk()
        desk.choose("keyboard")
        self.assertIs(desk.controller.ui_mode, ACT.UiMode.KEYBOARD)

    def test_the_click_kind_lands_where_a_wink_reads_it(self) -> None:
        # The property that matters: a LEFT wink reads board.click_type. A bar
        # that only updated its own click_kind would change the label and
        # nothing else, which is the silent kind of broken.
        from gazelink_core.interaction import gesture as GEST  # noqa: PLC0415

        desk = _Desk()
        desk.controller.board.click_type = "single"
        desk.choose("click")
        self.assertEqual(desk.bar.click_kind, "double")
        self.assertEqual(desk.controller.board.click_type, "double")
        self.assertIs(desk.controller.click_for(GEST.Eye.LEFT), ACT.Action.DOUBLE_CLICK)
        # The toggle is about the LEFT wink only.
        self.assertIs(desk.controller.click_for(GEST.Eye.RIGHT), ACT.Action.RIGHT_CLICK)

    def test_a_pick_is_counted(self) -> None:
        desk = _Desk()
        desk.choose("scroll")
        self.assertEqual(desk.rig.telemetry.tally["bar_picks"], 1)


class AModeWithNoMachineryIsRefusedTests(unittest.TestCase):
    def test_zoom_is_refused_while_nothing_drives_it(self) -> None:
        desk = _Desk()
        desk.choose("more")
        desk.choose("zoom")
        self.assertIs(desk.controller.ui_mode, ACT.UiMode.CURSOR)
        self.assertEqual(desk.rig.telemetry.tally["bar_modes_not_ready"], 1)

    def test_the_refusal_is_said_on_screen_and_not_only_counted(self) -> None:
        desk = _Desk()
        desk.choose("more")
        desk.choose("zoom")
        self.assertIsNotNone(desk.controller.notice)

    def test_drag_is_refused_while_a_carry_cannot_be_started(self) -> None:
        desk = _Desk()
        desk.choose("more")
        desk.choose("drag")
        self.assertIs(desk.controller.ui_mode, ACT.UiMode.CURSOR)

    def test_wiring_the_magnifier_in_makes_its_target_work(self) -> None:
        # The gate is derived from what the controller was GIVEN, so this test
        # fails the day the refusal outlives the thing it stands in for.
        desk = _Desk(zoom=object())
        desk.choose("more")
        desk.choose("zoom")
        self.assertIs(desk.controller.ui_mode, ACT.UiMode.ZOOM)


class TheBarIsDrivenInCursorModeOnlyTests(unittest.TestCase):
    def test_it_cannot_pick_while_the_menu_is_open(self) -> None:
        desk = _Desk()
        desk.controller.enter_mode(ACT.UiMode.MENU, why="test")
        desk.stare(centre_of("scroll", desk.bar.buttons), frames=30)
        self.assertIs(desk.controller.ui_mode, ACT.UiMode.MENU)
        self.assertEqual(desk.rig.telemetry.tally["bar_picks"], 0)

    def test_it_cannot_pick_while_scrolling(self) -> None:
        desk = _Desk()
        desk.controller.enter_mode(ACT.UiMode.SCROLL, why="test")
        desk.stare(centre_of("keyboard", desk.bar.buttons), frames=30)
        self.assertEqual(desk.rig.telemetry.tally["bar_picks"], 0)


class ABlinkDoesNotZeroTheFillTests(unittest.TestCase):
    """The defect that made ``--desk`` feel dead while the dot sat on the tile.

    Measured on ``recordings/round47/T1``: ``eyes_steady`` is true on 93.7% of
    frames, but the steady runs had a median of 5 and a 900 ms fill needs 29
    consecutive. ``gf_dwell_practice`` has no such gate and scored 17/18 on the
    same model and screen. Bridging turned those 31 fragments into one run of
    904 frames, with 30 gaps bridged and none too long.
    """

    def test_a_fill_survives_single_unsteady_frames(self) -> None:
        desk = _Desk()
        desk.look(CONTENT)
        target = centre_of("scroll", desk.bar.buttons)
        # One unsteady frame in every three: the real gaps were shorter still.
        for i in range(18):
            desk.look(target, steady=(i % 3 != 0))
        self.assertIs(desk.controller.ui_mode, ACT.UiMode.SCROLL)

    def test_the_pointer_still_refuses_an_unsteady_point(self) -> None:
        # The gate exists because an eye on its way down still yields a point.
        # Bridging is for the FILL only; if it leaked into the pointer, this
        # would record a move.
        desk = _Desk()
        desk.look(CONTENT)
        before = len(desk.rig.moves)
        for _ in range(6):
            desk.look((0.55, 0.45), steady=False)
        self.assertEqual(len(desk.rig.moves), before)

    def test_a_long_close_is_not_bridged_into_a_selection(self) -> None:
        # A deliberate close is the menu's signal. If the hold bridged it, the
        # gesture that opens the menu could finish a dwell on the way.
        desk = _Desk()
        desk.look(CONTENT)
        target = centre_of("scroll", desk.bar.buttons)
        desk.look(target)
        for _ in range(12):  # 1.2 s of closure, past the 800 ms confirm
            desk.look(target, steady=False)
        self.assertIs(desk.controller.ui_mode, ACT.UiMode.CURSOR)
        self.assertEqual(desk.rig.telemetry.tally["bar_picks"], 0)

    def test_a_tracking_loss_is_never_bridged(self) -> None:
        desk = _Desk()
        desk.look(CONTENT)
        target = centre_of("scroll", desk.bar.buttons)
        desk.look(target)
        for _ in range(16):
            desk.look(None)  # no point at all: a loss, not a blink
        self.assertEqual(desk.rig.telemetry.tally["bar_picks"], 0)

    def test_a_mode_change_forgets_the_held_point(self) -> None:
        desk = _Desk()
        desk.look(centre_of("scroll", desk.bar.buttons))
        desk.controller.enter_mode(ACT.UiMode.KEYBOARD, why="test")
        self.assertFalse(desk.controller.steady_hold.holding)


class TheWayBackWorksTests(unittest.TestCase):
    def test_the_pause_target_is_on_the_bar_in_every_view(self) -> None:
        bar = BAR.ControlBar()
        for view in (BAR.View.COMPACT, BAR.View.EXPANDED):
            bar.set_view(view)
            with self.subTest(view):
                self.assertIn(L.PAUSE_KEY, [b.key for b in bar.buttons])

    def test_the_bar_is_driven_while_paused_so_resume_can_be_chosen(self) -> None:
        # The reason the menu opens while paused applies here too: if the bar
        # went dead when the person paused, there would be no way back without
        # a keyboard -- which is the whole point of the pause target.
        desk = _Desk(start_active=False)
        self.assertFalse(desk.rig.safety.cursor_enabled)
        desk.look(CONTENT)
        desk.stare(centre_of(L.PAUSE_KEY, desk.bar.buttons))
        self.assertTrue(desk.rig.safety.cursor_enabled)

    def test_the_label_follows_the_mode_rather_than_the_key(self) -> None:
        bar = BAR.ControlBar()
        bar.set_paused(False)
        self.assertEqual(bar.pause_item.label, L.LABELS["pause"])
        bar.set_paused(True)
        self.assertEqual(bar.pause_item.label, L.LABELS["resume"])


class TheArmingRuleSurvivesTheWiringTests(unittest.TestCase):
    def test_a_second_target_cannot_be_taken_without_returning_to_content(self) -> None:
        desk = _Desk()
        desk.choose("more")
        before = desk.rig.telemetry.tally["bar_picks"]
        # No glance at content in between: staring straight at the next target.
        desk.stare(centre_of("drag", desk.bar.buttons), frames=30)
        self.assertEqual(desk.rig.telemetry.tally["bar_picks"], before)
        self.assertGreater(desk.bar.refused_unarmed, 0)


class TheRealSessionDrawsTheBarTests(unittest.TestCase):
    """Through ``run_live`` and the real composition root, not a hand-built rig.

    This is the test whose absence let the bar be built and forgotten: every
    earlier desk test either read the source as text or assembled its own
    controller, so nothing ever ran the session with ``desk=True`` and looked at
    what reached the screen. It also caught a stale fake -- ``_Display`` did not
    accept ``not_ready_line``, so the real presenter branch would have raised
    TypeError on its first frame while the suite stayed green.
    """

    def test_the_bar_is_drawn_when_the_flag_is_on(self) -> None:
        from test_live_loop import ON_SCREEN, _run

        display, _sends, _runner = _run(
            self, desk=True, points=[ON_SCREEN], max_seconds=0.2
        )
        self.assertIn("board", display.order)
        drawn = {key for board in display.boards for key in board}
        for key in ("click", "scroll", "keyboard", "more", L.PAUSE_KEY):
            with self.subTest(key):
                self.assertIn(key, drawn)

    def test_nothing_desk_shaped_is_drawn_when_the_flag_is_off(self) -> None:
        from test_live_loop import ON_SCREEN, _run

        display, _sends, _runner = _run(self, points=[ON_SCREEN], max_seconds=0.2)
        self.assertNotIn("board", display.order)

    def test_the_wording_that_reaches_the_screen_is_the_bars_rule(self) -> None:
        from test_live_loop import ON_SCREEN, _run

        display, _sends, _runner = _run(
            self, desk=True, points=[ON_SCREEN], max_seconds=0.2
        )
        for line in display.board_not_ready_lines:
            with self.subTest(line):
                self.assertNotIn("למרכז", line)


class ThePresenterDrawsItTests(unittest.TestCase):
    def test_the_bar_branch_uses_the_existing_tile_draw_call(self) -> None:
        source = (HERE.parent / "gazelink_core" / "ui" / "presenter.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("controller.bar is not None", source)
        self.assertIn("draw_board", source)

    def test_the_unready_line_is_the_bars_rule_and_not_the_menus(self) -> None:
        # draw_board's default tells the person to look at the CENTRE, which is
        # the menu's arming rule. The bar arms on content, so the default would
        # be an instruction that does not work.
        presenter = (HERE.parent / "gazelink_core" / "ui" / "presenter.py").read_text(
            encoding="utf-8"
        )
        display = (HERE.parent / "gazelink_core" / "ui" / "pygame_display.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("not_ready_line=", presenter)
        self.assertIn("not_ready_line: str", display)


if __name__ == "__main__":
    unittest.main()
