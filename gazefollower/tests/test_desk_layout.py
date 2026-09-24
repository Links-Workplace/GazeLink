"""The desk layouts must stay reachable, and must refuse to be tightened.

Every assertion here is about the two failures ``dwell.layout_warnings``
separates: MISSING is tolerable, reaching the NEIGHBOUR is not. A test that
only checked "the buttons fit" would pass on a layout that presses the wrong
thing.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gazelink_core.interaction import desk_layout as L  # noqa: E402


def _reach(buttons, index=0):
    """Half-width plus the gap to the next target in the same row."""

    row = sorted((b for b in buttons if b.y0 == buttons[index].y0), key=lambda b: b.x0)
    if len(row) < 2:
        return float("inf")
    return row[0].width / 2.0 + (row[1].x0 - row[0].x1)


class InsideTheBandTests(unittest.TestCase):
    def test_every_selectable_target_is_inside_the_validated_band(self) -> None:
        lo, hi = L.BAND
        groups = {
            "bar": L.bar(),
            "menu": L.menu(),
            "keyboard": L.keyboard(list("אבגדהוזח")),
            "scroll": L.scroll(),
            "zoom": L.zoom_controls(),
            "pause": [L.PAUSE],
        }
        outside = [
            f"{name}:{b.key}"
            for name, buttons in groups.items()
            for b in buttons
            if b.x0 < lo - 1e-9 or b.x1 > hi + 1e-9
        ]
        self.assertEqual(outside, [])

    def test_every_target_is_on_screen_vertically(self) -> None:
        for buttons in (L.bar(), L.menu(), L.keyboard(list("אבגדהוזח")), L.scroll(), [L.PAUSE]):
            for b in buttons:
                self.assertGreaterEqual(b.y0, 0.0)
                self.assertLessEqual(b.y1, 1.0, b.key)


class NeighbourRiskTests(unittest.TestCase):
    def test_no_layout_can_activate_the_neighbour_at_the_measured_bias(self) -> None:
        for name, buttons in (
            ("bar", L.bar()),
            ("menu", L.menu()),
            ("keyboard", L.keyboard(list("אבגדהוזח"))),
            ("zoom", L.zoom_controls()),
        ):
            with self.subTest(name):
                risky = [w for w in L.warnings(buttons) if w.startswith("NEIGHBOUR")]
                self.assertEqual(risky, [], f"{name} can reach its neighbour")

    def test_the_bar_keeps_a_margin_over_the_measured_bias(self) -> None:
        self.assertGreater(_reach(L.bar()), L.BIAS * 1.2)

    def test_a_row_that_is_too_tight_is_refused_rather_than_drawn(self) -> None:
        with self.assertRaises(L.LayoutTooTight):
            # Five slots at the density that sits on the measured line.
            L.row(list("abcde"), width=0.055, gap=0.020, top=0.76, height=0.15)

    def test_a_row_wider_than_the_band_is_refused(self) -> None:
        with self.assertRaises(L.LayoutTooTight):
            L.row(list("abcd"), width=0.12, gap=0.06, top=0.76, height=0.15)


class MissRiskIsReportedNotHiddenTests(unittest.TestCase):
    def test_the_bar_reports_its_miss_risk_explicitly(self) -> None:
        # Half-width 0.035 is under the 0.052 bias, so a miss IS expected and
        # the layout says so before a session rather than after a bad run.
        misses = [w for w in L.warnings(L.bar()) if w.startswith("MISS")]
        self.assertTrue(misses)

    def test_warnings_never_pair_a_target_with_the_one_below_it(self) -> None:
        # menu() is 3x2. Checked row by row, the only neighbours are the two
        # beside each tile; checked as one set, dwell would pair across rows.
        lines = L.warnings(L.menu())
        self.assertEqual([w for w in lines if w.startswith("NEIGHBOUR")], [])


class FixedFurnitureTests(unittest.TestCase):
    def test_pause_has_no_neighbour_at_all(self) -> None:
        others = L.bar() + L.menu() + L.keyboard(list("אבגדהוזח"))
        for b in others:
            horizontal = max(L.PAUSE.x0 - b.x1, b.x0 - L.PAUSE.x1)
            vertical = max(L.PAUSE.y0 - b.y1, b.y0 - L.PAUSE.y1)
            self.assertGreater(
                max(horizontal, vertical), L.BIAS,
                f"{b.key} sits within the measured bias of the pause target",
            )

    def test_pause_is_in_the_same_place_whatever_the_screen(self) -> None:
        # One constant, not a per-screen number: a pause that moves is a pause
        # the person has to hunt for at the moment they most need it.
        self.assertEqual((L.PAUSE.x0, L.PAUSE.y0, L.PAUSE.x1, L.PAUSE.y1),
                         (0.600, 0.025, 0.700, 0.125))

    def test_the_navigation_slot_is_the_same_rectangle_on_every_screen(self) -> None:
        root = L.bar()[L.NAV_SLOT]
        for keys in (["finish"], ["confirm", "zoom-in", "overview", "back"], ["drop", "cancel"]):
            slot = L.bottom_row(keys)[-1]
            self.assertEqual((slot.x0, slot.y0, slot.x1, slot.y1),
                             (root.x0, root.y0, root.x1, root.y1), keys)

    def test_the_navigation_slot_never_carries_an_action(self) -> None:
        self.assertEqual(L.bar()[L.NAV_SLOT].key, "more")
        self.assertEqual(L.bottom_row(["confirm", "back"])[-1].key, "back")


class OrderTests(unittest.TestCase):
    def test_the_first_item_is_the_rightmost_because_the_labels_are_hebrew(self) -> None:
        buttons = L.bar()
        self.assertEqual(buttons[0].key, "click")
        self.assertGreater(buttons[0].x0, buttons[-1].x0)

    def test_the_menu_reads_right_to_left_row_by_row(self) -> None:
        tiles = L.menu()
        self.assertEqual(tiles[0].key, "zoom")
        self.assertGreater(tiles[0].x0, tiles[1].x0)
        self.assertGreater(tiles[1].x0, tiles[2].x0)
        self.assertGreater(tiles[3].y0, tiles[0].y0)


class KeyboardTests(unittest.TestCase):
    def test_a_page_is_eight_letters_and_four_fixed_actions(self) -> None:
        buttons = L.keyboard(list("אבגדהוזח"))
        self.assertEqual(len(buttons), 12)
        self.assertEqual([b.key for b in buttons[-4:]], L.KEYBOARD_ACTIONS)

    def test_the_actions_are_in_the_same_place_whatever_the_page_holds(self) -> None:
        full = L.keyboard(list("אבגדהוזח"))[-4:]
        short = L.keyboard(list("ןףץ"))[-4:]
        self.assertEqual([(b.key, b.x0, b.y0) for b in full], [(b.key, b.x0, b.y0) for b in short])

    def test_a_page_refuses_to_be_overfilled(self) -> None:
        with self.assertRaises(ValueError):
            L.keyboard(list("אבגדהוזחט"))

    def test_every_label_used_by_a_layout_has_hebrew_text(self) -> None:
        keys = {b.key for b in L.bar() + L.menu() + L.scroll() + L.zoom_controls()}
        keys |= set(L.KEYBOARD_ACTIONS) | {L.PAUSE_KEY}
        missing = sorted(k for k in keys if k not in L.LABELS)
        self.assertEqual(missing, [])


class ScrollTests(unittest.TestCase):
    def test_the_bands_leave_a_dead_middle_to_stop_in(self) -> None:
        up, down = L.scroll()[0], L.scroll()[1]
        self.assertGreater(down.y0 - up.y1, 0.15)

    def test_the_bands_span_the_whole_band_so_there_is_no_neighbour_beside_them(self) -> None:
        up = L.scroll()[0]
        self.assertEqual((up.x0, up.x1), L.BAND)

    def test_the_scroll_screen_offers_nothing_but_up_down_and_the_way_out(self) -> None:
        self.assertEqual([b.key for b in L.scroll()], ["up", "down", "finish"])


class ZoomTests(unittest.TestCase):
    def test_the_panel_sits_inside_the_band(self) -> None:
        lo, hi = L.BAND
        self.assertGreaterEqual(L.ZOOM_PANEL.x0, lo)
        self.assertLessEqual(L.ZOOM_PANEL.x1, hi)

    def test_the_first_magnification_is_a_real_magnification(self) -> None:
        panel_width = L.ZOOM_PANEL.x1 - L.ZOOM_PANEL.x0
        self.assertGreaterEqual(panel_width / L.ZOOM_SPAN, 3.5)

    def test_confirm_is_first_and_back_is_in_the_navigation_slot(self) -> None:
        keys = [b.key for b in L.zoom_controls()]
        self.assertEqual(keys[0], "confirm")
        self.assertEqual(keys[-1], "back")


if __name__ == "__main__":
    unittest.main()
