"""The menu a long close opens: what it refuses, and the way back from a pause.

Pure state machine, so every one of these is deterministic.  The two rules
that carry its safety are asserted in both directions, because a menu that
chose nothing would pass every "it must not choose" test on its own.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_actions as A  # noqa: E402
import gf_menu as M  # noqa: E402

DEAD = (0.5, 0.5)


def _centres(page: str = "main", *, paused: bool = False) -> dict[str, tuple[float, float]]:
    items = M._pages(paused=paused)[page]
    return {b.key: b.centre for b in M.page_buttons(items)}


MAIN = _centres()


def _open(*, paused: bool = False, dwell: float = 100.0) -> M.MenuModel:
    board = M.MenuModel(dwell_ms=dwell)
    board.show(paused=paused)
    return board


def _choose(board: M.MenuModel, point, *, start: float = 0.0, seed: bool = True):
    """Seed the gaze in the dead zone, then rest on ``point`` long enough."""

    if seed:
        board.update(start, DEAD)
    board.update(start, point)
    return board.update(start + 1.0, point)


class GazeSeedingTests(unittest.TestCase):
    """A long close ENDS with the eyes opening, and the gaze lands wherever it
    lands.  A menu that appeared under it would start filling that tile
    immediately, and the person never asked for that tile."""

    def test_a_tile_is_not_chosen_before_the_gaze_is_seen_in_the_dead_zone(self) -> None:
        board = _open()
        chosen = None
        for i in range(20):
            chosen = board.update(i * 0.5, MAIN["scroll"]) or chosen
        self.assertIsNone(chosen, "the tile the eyes opened onto chose itself")
        self.assertFalse(board.seeded)
        self.assertTrue(board.refused_unseeded, "nothing was actually refused")

    def test_no_progress_is_drawn_while_it_cannot_be_chosen(self) -> None:
        """Showing a filling ring that cannot complete is worse than showing
        nothing: it says an activation is coming when none is."""

        board = _open()
        board.update(0.0, MAIN["scroll"])
        board.update(0.5, MAIN["scroll"])
        self.assertEqual(board.progress, 0.0)

    def test_the_same_tile_is_chosen_once_the_gaze_has_been_seen(self) -> None:
        """The other half. A menu that never chose anything would pass the
        first test and be useless."""

        board = _open()
        choice = _choose(board, MAIN["scroll"])
        self.assertIsNotNone(choice, "the tile could not be chosen even after seeding")
        self.assertEqual(choice.key, "scroll")

    def test_a_page_change_seeds_again(self) -> None:
        """The gaze is resting on the tile that was just chosen, and the tile
        that replaces it is a different tile."""

        board = _open()
        _choose(board, MAIN["more-1"])
        self.assertEqual(board.page, "nav")
        self.assertFalse(board.seeded, "the new page was live under a resting gaze")


class ChoosingTests(unittest.TestCase):
    def test_a_choice_closes_the_menu(self) -> None:
        """The operator asked for exactly this: pick a thing and get on with
        it, rather than pick and then have to close what you picked from."""

        board = _open()
        _choose(board, MAIN["keyboard"])
        self.assertFalse(board.open)

    def test_more_does_not_close_it(self) -> None:
        board = _open()
        choice = _choose(board, MAIN["more-1"])
        self.assertTrue(board.open, "the way to the next page closed the menu")
        self.assertIs(choice.effect, M.Effect.PAGE)

    def test_more_is_the_last_tile_of_every_page_and_always_leads_on(self) -> None:
        """Consistent so it can be learned: wherever the person is, the last
        tile takes them somewhere new and commits to nothing."""

        pages = M._pages(paused=False)
        for name in ("main", "nav", "system"):
            last = pages[name][-1]
            self.assertEqual(last.label, "עוד", f"{name} does not end with MORE")
            self.assertIs(last.effect, M.Effect.PAGE)

    def test_walking_the_more_tiles_comes_back_to_the_first_page(self) -> None:
        """A menu you can walk out of the end of is a menu with a dead end."""

        board = _open()
        seen = [board.page]
        for i in range(3):
            centres = _centres(board.page)
            key = board.items[-1].key
            _choose(board, centres[key], start=i * 10.0)
            seen.append(board.page)
        self.assertEqual(seen, ["main", "nav", "system", "main"])


class ClickTypeTests(unittest.TestCase):
    """What a LEFT wink does. The right wink is always a right click and is not
    chosen in the menu (operator, 24.9.2026)."""

    def test_the_default_is_single(self) -> None:
        """The operator's decision: an ordinary click is one click."""

        self.assertEqual(M.MenuModel().click_type, "single")

    def test_right_is_no_longer_a_type_to_choose(self) -> None:
        self.assertEqual(M.CLICK_TYPES, ("single", "double"))
        with self.assertRaises(ValueError):
            M.MenuModel(click_type="right")

    def test_the_toggle_switches_to_double_and_closes_the_menu(self) -> None:
        board = _open()
        _choose(board, MAIN["double-toggle"])
        self.assertEqual(board.click_type, "double")
        self.assertFalse(board.open, "choosing the toggle left the menu open")

    def test_the_toggle_names_the_state_it_is_in(self) -> None:
        # The label says what is ON now, so nothing has to be remembered.
        off = {i.key: i for i in M._pages(paused=False, click_type="single")["main"]}
        on = {i.key: i for i in M._pages(paused=False, click_type="double")["main"]}
        self.assertEqual(off["double-toggle"].label, "לחיצה כפולה: כבוי")
        self.assertEqual(on["double-toggle"].label, "לחיצה כפולה: פעיל")

    def test_choosing_it_twice_comes_back_to_single(self) -> None:
        board = _open()
        _choose(board, MAIN["double-toggle"])
        board.show()
        _choose(board, MAIN["double-toggle"], start=10.0)
        self.assertEqual(board.click_type, "single")

    def test_it_survives_the_menu_being_opened_again(self) -> None:
        board = _open()
        _choose(board, MAIN["double-toggle"])
        board.show()
        self.assertEqual(board.click_type, "double")

    def test_every_click_type_maps_to_an_action(self) -> None:
        for name in M.CLICK_TYPES:
            self.assertIn(M.CLICK_ACTION[name], A.CLICKS)


class PauseAndResumeTests(unittest.TestCase):
    """Getting back to control without hands.

    A pause tile with no resume beside it is a way out with no way back, which
    is the first priority in CLAUDE.md inverted.  The tile is the same tile.
    """

    def test_the_pause_tile_offers_resume_when_paused(self) -> None:
        items = {i.key: i for i in M._pages(paused=True)["system"]}
        self.assertIs(items["pause"].action, A.Action.RESUME)
        self.assertEqual(items["pause"].label, "המשך שליטה")

    def test_it_offers_pause_when_active(self) -> None:
        items = {i.key: i for i in M._pages(paused=False)["system"]}
        self.assertIs(items["pause"].action, A.Action.PAUSE)

    def test_it_is_in_the_same_place_either_way(self) -> None:
        """Same place, same gesture, both directions. A resume somewhere else
        is a resume the person has to find while they cannot act."""

        self.assertEqual(_centres("system", paused=True)["pause"],
                         _centres("system", paused=False)["pause"])

    def test_a_paused_menu_can_be_opened_and_walked_to_resume(self) -> None:
        board = _open(paused=True)
        _choose(board, MAIN["more-1"])
        _choose(board, _centres("nav")["more-2"], start=10.0)
        self.assertEqual(board.page, "system")
        choice = _choose(board, _centres("system", paused=True)["pause"], start=20.0)
        self.assertIsNotNone(choice)
        self.assertIs(choice.action, A.Action.RESUME)

    def test_the_tile_follows_a_mode_that_changes_underneath_it(self) -> None:
        """A lost face suspends the mode with nobody choosing anything, and a
        tile still saying "pause" would offer the one thing that cannot help."""

        board = _open(paused=False)
        _choose(board, MAIN["more-1"])
        _choose(board, _centres("nav")["more-2"], start=10.0)
        board.set_paused(True)
        items = {i.key: i for i in board.items}
        self.assertIs(items["pause"].action, A.Action.RESUME)


class LayoutTests(unittest.TestCase):
    def test_the_tiles_sit_inside_the_validated_band(self) -> None:
        lo, hi = M.BAND
        for button in M.page_buttons(M._pages(paused=False)["main"]):
            self.assertGreaterEqual(button.x0, lo)
            self.assertLessEqual(button.x1, hi)

    def test_a_biased_gaze_cannot_reach_the_wrong_tile(self) -> None:
        """The tolerable failure is a MISS and the dangerous one is choosing
        the neighbour. Missing is expected here; the neighbour must not be."""

        warnings = M.layout_warnings()
        self.assertTrue(
            any("MISS RISK" in w for w in warnings),
            "the vertical miss risk is real and should be reported",
        )
        self.assertFalse(
            [w for w in warnings if "NEIGHBOUR RISK" in w],
            f"a biased gaze can activate the wrong tile: {warnings}",
        )

    def test_the_first_tile_is_the_one_a_hebrew_reader_reaches_first(self) -> None:
        rects = M.tile_rects()
        self.assertGreater(rects[0][0], rects[1][0], "item one was not on the right")

    def test_there_is_a_dead_zone_between_the_rows(self) -> None:
        rects = M.tile_rects()
        gap = rects[2][1] - rects[0][3]
        self.assertGreater(gap, 0.38, f"the rows are {gap:.3f} apart, under the vertical bias")


class LostPointTests(unittest.TestCase):
    def test_a_stale_point_makes_no_progress(self) -> None:
        board = _open()
        board.update(0.0, DEAD)
        board.update(0.0, MAIN["scroll"])
        self.assertIsNone(board.update(5.0, MAIN["scroll"], fresh=False))
        self.assertEqual(board.progress, 0.0)

    def test_a_missing_point_is_counted_rather_than_ignored(self) -> None:
        board = _open()
        board.update(0.0, None)
        self.assertEqual(board.seen_in["no point"], 1)

    def test_a_closed_menu_chooses_nothing_however_long_the_gaze_rests(self) -> None:
        board = M.MenuModel(dwell_ms=10.0)
        for i in range(10):
            self.assertIsNone(board.update(i * 1.0, MAIN["scroll"]))


if __name__ == "__main__":
    unittest.main()
