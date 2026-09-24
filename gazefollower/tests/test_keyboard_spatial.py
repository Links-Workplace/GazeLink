"""The gaze keyboard: no wink required, no key taken that was not watched.

The two failures this guards are a key chosen by a glance that was passing
through, and a page whose contents nobody could predict before turning it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gazelink_core.interaction import actions as A  # noqa: E402
from gazelink_core.interaction import desk_layout as L  # noqa: E402
from gazelink_core.interaction import keyboard_spatial as K  # noqa: E402

CONTENT = (0.5, 0.35)  # between the letter rows: on no key


def centre(kb, key):
    return next(b.centre for b in kb.buttons if b.key == key)


def rest(kb, key, *, seconds=1.2, start=0.0, step=0.05):
    at = centre(kb, key)
    now = start
    while now < start + seconds:
        choice = kb.update(now, at)
        if choice is not None:
            return choice
        now += step
    return None


def armed(dwell_ms=900.0):
    kb = K.SpatialKeyboard(dwell_ms=dwell_ms)
    kb.update(0.0, CONTENT)
    return kb


class NoWinkIsRequiredTests(unittest.TestCase):
    def test_a_letter_is_chosen_by_looking_at_it(self) -> None:
        kb = armed()
        choice = rest(kb, "א")
        self.assertIsNotNone(choice)
        self.assertEqual(choice.text, "א")

    def test_the_keyboard_depends_on_no_gesture_detector_at_all(self) -> None:
        # Behaviour, not a word search: a full letter is produced from gaze
        # points alone, and the module imports nothing that reads an eyelid.
        import ast

        source = Path(K.__file__).read_text(encoding="utf-8")
        imported = {
            name.name
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom)
            for name in node.names
        }
        self.assertNotIn("gesture", imported)
        kb = armed()
        for letter in ("ש", "ל", "ו", "ם"):
            while kb.letters and letter not in kb.letters:
                kb.turn_page()
                kb.update(0.0, CONTENT)
            choice = rest(kb, letter, seconds=2.0)
            self.assertIsNotNone(choice, letter)
            kb.on_sent(choice)
            kb.update(0.0, CONTENT)
        self.assertEqual(kb.typed, "שלום")


class AGlanceTypesNothingTests(unittest.TestCase):
    def test_a_key_under_the_gaze_when_the_keyboard_opens_is_not_taken(self) -> None:
        kb = K.SpatialKeyboard()
        self.assertIsNone(rest(kb, "א", seconds=3.0))
        self.assertGreater(kb.refused_unarmed, 0)

    def test_the_gaze_must_be_seen_off_the_keys_first(self) -> None:
        kb = K.SpatialKeyboard()
        rest(kb, "א", seconds=2.0)
        kb.update(5.0, CONTENT)
        self.assertTrue(kb.armed)
        self.assertEqual(rest(kb, "א", start=5.0).text, "א")

    def test_a_page_change_re_arms_because_different_keys_are_there_now(self) -> None:
        kb = armed()
        rest(kb, "next-page")
        self.assertFalse(kb.armed)
        self.assertIsNone(rest(kb, "ט", seconds=2.0, start=3.0))

    def test_resting_far_longer_types_one_letter_and_not_a_stream(self) -> None:
        kb = armed()
        at = centre(kb, "א")
        chosen = [kb.update(i * 0.05, at) for i in range(120)]
        self.assertEqual(len([c for c in chosen if c is not None]), 1)

    def test_a_lost_point_types_nothing(self) -> None:
        kb = armed()
        for i in range(60):
            self.assertIsNone(kb.update(i * 0.05, None))

    def test_a_stale_point_types_nothing(self) -> None:
        kb = armed()
        for i in range(60):
            self.assertIsNone(kb.update(i * 0.05, centre(kb, "א"), fresh=False))


class TheEchoIsWhatLeftTests(unittest.TestCase):
    def test_choosing_a_letter_does_not_echo_it_on_its_own(self) -> None:
        # The line the person reads must be what was delivered, not what was
        # asked for: the adapter can still refuse.
        kb = armed()
        rest(kb, "א")
        self.assertEqual(kb.typed, "")

    def test_the_echo_appears_once_the_key_has_been_sent(self) -> None:
        kb = armed()
        choice = rest(kb, "א")
        kb.on_sent(choice)
        self.assertEqual(kb.typed, "א")

    def test_backspace_removes_one_character(self) -> None:
        kb = armed()
        for letter in ("א", "ב"):
            kb.on_sent(K.Choice(letter, text=letter))
        kb.on_sent(K.ACTIONS["backspace"])
        self.assertEqual(kb.typed, "א")

    def test_backspace_on_an_empty_line_is_harmless(self) -> None:
        kb = armed()
        kb.on_sent(K.ACTIONS["backspace"])
        self.assertEqual(kb.typed, "")

    def test_space_is_a_character_and_not_a_command(self) -> None:
        self.assertEqual(K.ACTIONS["space"].text, " ")
        self.assertIsNone(K.ACTIONS["space"].action)


class PagingIsPredictableTests(unittest.TestCase):
    def test_the_page_key_names_the_letters_it_leads_to(self) -> None:
        kb = K.SpatialKeyboard()
        preview = kb.next_preview
        self.assertTrue(preview.strip())
        after = kb.next_letters
        kb.turn_page()
        self.assertEqual(kb.letters, after)
        self.assertTrue(all(ch in preview for ch in kb.letters[:4]))

    def test_every_hebrew_letter_is_reachable_by_paging(self) -> None:
        seen: list[str] = []
        kb = K.SpatialKeyboard()
        for _ in range(len(K.PAGES)):
            seen.extend(kb.letters)
            kb.turn_page()
        for letter in K.HEBREW:
            self.assertIn(letter, seen, letter)

    def test_paging_returns_to_the_first_page_eventually(self) -> None:
        kb = K.SpatialKeyboard()
        first = kb.letters
        for _ in range(len(K.PAGES)):
            kb.turn_page()
        self.assertEqual(kb.letters, first)

    def test_english_is_reached_by_paging_and_not_by_a_fifth_target(self) -> None:
        kb = K.SpatialKeyboard()
        names = []
        for _ in range(len(K.PAGES)):
            names.append(kb.layout_name)
            kb.turn_page()
        self.assertIn("english", names)
        self.assertEqual(len(L.KEYBOARD_ACTIONS), 4)

    def test_paging_sends_nothing(self) -> None:
        kb = armed()
        choice = rest(kb, "next-page")
        self.assertIs(choice.control, K.Control.NEXT_PAGE)
        self.assertIsNone(choice.text)
        self.assertIsNone(choice.action)

    def test_a_short_last_page_still_carries_the_four_actions(self) -> None:
        kb = K.SpatialKeyboard()
        while len(kb.letters) == K.PER_PAGE:
            kb.turn_page()
        keys = [b.key for b in kb.buttons]
        for action in L.KEYBOARD_ACTIONS:
            self.assertIn(action, keys)


class FixedFurnitureTests(unittest.TestCase):
    def test_the_four_actions_are_in_the_same_place_on_every_page(self) -> None:
        kb = K.SpatialKeyboard()
        first = {b.key: (b.x0, b.y0) for b in kb.buttons if b.key in L.KEYBOARD_ACTIONS}
        for _ in range(3):
            kb.turn_page()
            now = {b.key: (b.x0, b.y0) for b in kb.buttons if b.key in L.KEYBOARD_ACTIONS}
            self.assertEqual(now, first)

    def test_close_is_in_the_navigation_slot(self) -> None:
        kb = K.SpatialKeyboard()
        actions = [b for b in kb.buttons if b.key in L.KEYBOARD_ACTIONS]
        self.assertEqual(min(actions, key=lambda b: b.x0).key, "close")

    def test_the_keyboard_cannot_reach_a_neighbour(self) -> None:
        self.assertEqual([w for w in K.layout_warnings() if w.startswith("NEIGHBOUR")], [])

    def test_a_page_never_holds_more_than_the_columns_allow(self) -> None:
        for _, letters in K.PAGES:
            self.assertLessEqual(len(letters), K.PER_PAGE)


class ContractTests(unittest.TestCase):
    def test_a_key_that_does_two_things_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            K.Choice("x", text="x", action=A.Action.ENTER)

    def test_a_key_that_does_nothing_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            K.Choice("x")

    def test_resetting_can_keep_or_clear_the_line(self) -> None:
        kb = armed()
        kb.on_sent(K.Choice("א", text="א"))
        kb.reset()
        self.assertEqual(kb.typed, "א")
        kb.reset(keep_text=False)
        self.assertEqual(kb.typed, "")

    def test_the_report_separates_reaching_a_key_from_taking_it(self) -> None:
        kb = K.SpatialKeyboard()
        rest(kb, "א", seconds=1.0)
        summary = kb.summary()
        self.assertIn("א", summary["keys the gaze reached"])
        self.assertGreater(summary["frames refused before the gaze returned to content"], 0)


if __name__ == "__main__":
    unittest.main()
