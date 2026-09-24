"""The bar is always on screen, so the thing it must not do is open by itself.

A persistent dwell target has no "it was just shown" moment to re-arm on, which
is what ``MenuModel`` relies on. Most of these tests are about the rule that
replaces it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gazelink_core.interaction import actions as A  # noqa: E402
from gazelink_core.interaction import bar as B  # noqa: E402
from gazelink_core.interaction import desk_layout as L  # noqa: E402

CONTENT = (0.5, 0.45)  # dead space: on the page, on no target


def centre(bar, key):
    return next(b.centre for b in bar.buttons if b.key == key)


def rest(bar, key, *, seconds=1.2, start=0.0, step=0.05):
    """Hold the gaze on a target and return the first pick it produces."""

    at = centre(bar, key)
    now = start
    while now < start + seconds:
        pick = bar.update(now, at)
        if pick is not None:
            return pick
        now += step
    return None


class ItOpensOnlyWhenAskedTests(unittest.TestCase):
    def test_a_glance_at_the_bottom_of_the_screen_chooses_nothing(self) -> None:
        bar = B.ControlBar()
        bar.disarm()
        self.assertIsNone(rest(bar, "scroll", seconds=3.0))
        self.assertGreater(bar.refused_unarmed, 0)

    def test_the_gaze_must_return_to_content_before_anything_can_be_chosen(self) -> None:
        bar = B.ControlBar()
        bar.disarm()
        rest(bar, "scroll", seconds=2.0)
        self.assertFalse(bar.armed)
        bar.update(5.0, CONTENT)
        self.assertTrue(bar.armed)
        self.assertIsNotNone(rest(bar, "scroll", start=5.0))

    def test_the_ring_shows_nothing_while_it_is_unarmed(self) -> None:
        # Tiles a person can see filling but cannot choose, with nothing
        # saying why, is the same experience as tiles that are broken.
        bar = B.ControlBar()
        bar.disarm()
        rest(bar, "scroll", seconds=0.5)
        self.assertEqual(bar.progress, 0.0)

    def test_leaving_the_bar_disarms_it_again(self) -> None:
        bar = B.ControlBar()
        bar.update(0.0, CONTENT)
        pick = rest(bar, "scroll")
        self.assertEqual(pick.key, "scroll")
        self.assertFalse(bar.armed)

    def test_a_view_change_re_arms_because_the_target_underneath_changed(self) -> None:
        bar = B.ControlBar()
        bar.update(0.0, CONTENT)
        self.assertEqual(rest(bar, "more").key, "more")
        self.assertIs(bar.view, B.View.EXPANDED)
        self.assertFalse(bar.armed)

    def test_a_lost_point_never_arms_anything(self) -> None:
        bar = B.ControlBar()
        bar.disarm()
        for i in range(40):
            bar.update(i * 0.05, None)
        self.assertFalse(bar.armed)

    def test_a_stale_point_never_arms_anything(self) -> None:
        bar = B.ControlBar()
        bar.disarm()
        for i in range(40):
            bar.update(i * 0.05, CONTENT, fresh=False)
        self.assertFalse(bar.armed)


class OneActivationPerRestTests(unittest.TestCase):
    def test_holding_far_longer_still_chooses_only_once(self) -> None:
        bar = B.ControlBar()
        bar.update(0.0, CONTENT)
        at = centre(bar, "scroll")
        picks = [bar.update(i * 0.05, at) for i in range(120)]
        self.assertEqual(len([p for p in picks if p is not None]), 1)

    def test_the_click_kind_is_a_setting_and_not_a_trip(self) -> None:
        # The "click" tile toggles single <-> double and leaves the bar where it
        # is: an ordinary click is a look and a wink, never a walk through a menu.
        bar = B.ControlBar()
        bar.update(0.0, CONTENT)
        pick = rest(bar, "click")
        self.assertEqual(pick.key, "click")
        self.assertEqual(bar.click_kind, "double")
        self.assertIs(bar.view, B.View.COMPACT)

    def test_right_and_double_are_no_longer_in_the_more_menu(self) -> None:
        # The right wink IS the right click, and double lives on the bar's toggle.
        keys = [i.key for i in B.MENU_ITEMS]
        self.assertNotIn("right-click", keys)
        self.assertNotIn("double-click", keys)
        self.assertEqual(B.CLICK_KINDS, ("single", "double"))

    def test_the_chosen_kind_decides_what_a_confirm_sends(self) -> None:
        bar = B.ControlBar()
        self.assertIs(B.CLICK_ACTION[bar.click_kind], A.Action.LEFT_CLICK)
        bar.click_kind = "double"
        self.assertIs(B.CLICK_ACTION[bar.click_kind], A.Action.DOUBLE_CLICK)

    def test_the_default_click_is_the_simple_one(self) -> None:
        self.assertEqual(B.ControlBar().click_kind, "single")


class PauseNeverMovesTests(unittest.TestCase):
    def test_pause_is_reachable_from_every_view(self) -> None:
        bar = B.ControlBar()
        for view in (B.View.COMPACT, B.View.EXPANDED, B.View.MINIMISED):
            bar.set_view(view)
            with self.subTest(view):
                self.assertIn(L.PAUSE_KEY, [b.key for b in bar.buttons])

    def test_pause_is_the_same_rectangle_in_every_view(self) -> None:
        bar = B.ControlBar()
        seen = set()
        for view in (B.View.COMPACT, B.View.EXPANDED, B.View.MINIMISED):
            bar.set_view(view)
            target = next(b for b in bar.buttons if b.key == L.PAUSE_KEY)
            seen.add((target.x0, target.y0, target.x1, target.y1))
        self.assertEqual(len(seen), 1)

    def test_the_pause_target_becomes_the_resume_target_in_place(self) -> None:
        bar = B.ControlBar()
        self.assertIs(bar.pause_item.action, A.Action.PAUSE)
        bar.set_paused(True)
        self.assertIs(bar.pause_item.action, A.Action.RESUME)

    def test_choosing_pause_returns_the_action_for_the_current_mode(self) -> None:
        bar = B.ControlBar()
        bar.set_paused(True)
        bar.update(0.0, CONTENT)
        pick = rest(bar, L.PAUSE_KEY)
        self.assertIs(pick.item.action, A.Action.RESUME)

    def test_pause_is_not_one_of_the_bar_actions(self) -> None:
        # In the bar it could be chosen in place of an action. Its own target,
        # with nothing beside it, is the whole point.
        self.assertNotIn(L.PAUSE_KEY, [i.key for i in B.BAR_ITEMS])


class MinimisingKeepsTheTargetTests(unittest.TestCase):
    def test_minimising_does_not_shrink_what_the_gaze_must_hit(self) -> None:
        bar = B.ControlBar()
        compact = {b.key: (b.width, b.height) for b in bar.buttons}
        bar.set_view(B.View.MINIMISED)
        small = {b.key: (b.width, b.height) for b in bar.buttons}
        self.assertEqual(compact, small)

    def test_coming_back_from_minimised_is_not_a_small_target(self) -> None:
        bar = B.ControlBar()
        bar.set_view(B.View.MINIMISED)
        for target in bar.buttons:
            self.assertGreaterEqual(target.width / 2.0, 0.030)


class GeometryTests(unittest.TestCase):
    def test_no_view_of_the_bar_can_activate_a_neighbour(self) -> None:
        self.assertEqual([w for w in B.layout_warnings() if w.startswith("NEIGHBOUR")], [])

    def test_every_item_has_a_hebrew_label(self) -> None:
        bar = B.ControlBar()
        for items in (B.BAR_ITEMS, B.MENU_ITEMS):
            for item in items:
                with self.subTest(item.key):
                    self.assertTrue(bar.label_for(item).strip())
                    self.assertNotEqual(bar.label_for(item), item.key)

    def test_the_click_button_names_the_kind_it_will_send(self) -> None:
        bar = B.ControlBar()
        click = next(i for i in B.BAR_ITEMS if i.key == "click")
        self.assertEqual(bar.label_for(click), "לחיצה כפולה: כבוי")
        bar.click_kind = "double"
        self.assertEqual(bar.label_for(click), "לחיצה כפולה: פעיל")

    def test_the_navigation_slot_of_the_bar_is_more_and_not_an_action(self) -> None:
        bar = B.ControlBar()
        targets = [b for b in bar.buttons if b.key != L.PAUSE_KEY]
        leftmost = min(targets, key=lambda b: b.x0)
        self.assertEqual(leftmost.key, "more")


class ReportTests(unittest.TestCase):
    def test_the_report_separates_reaching_a_target_from_choosing_it(self) -> None:
        # "I looked at it and nothing happened" and "it never saw me" are the
        # same sentence from the person and two different numbers here.
        bar = B.ControlBar()
        bar.disarm()
        rest(bar, "scroll", seconds=1.0)
        summary = bar.summary()
        self.assertIn("scroll", summary["targets the gaze reached"])
        self.assertGreater(summary["frames refused before the gaze returned to content"], 0)

    def test_an_unknown_click_kind_is_refused_at_construction(self) -> None:
        with self.assertRaises(ValueError):
            B.ControlBar(click_kind="middle")


if __name__ == "__main__":
    unittest.main()
