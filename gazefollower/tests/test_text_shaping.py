"""RTL shaping: Hebrew reverses, Latin does not, and the old screens are safe.

The equivalence test is the important one. ``pygame_display.rtl`` stays in use
on every existing protocol screen, and the two rules may only live side by side
while they agree on the lines those screens actually draw -- which are entirely
Hebrew.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gazelink_core.ui import text as T  # noqa: E402


def _old_rtl(value: str) -> str:
    """What ``pygame_display.rtl`` does, copied so the test needs no pygame."""

    return value[::-1] if T.has_hebrew(value) else value


class PureHebrewMatchesTheExistingRuleTests(unittest.TestCase):
    def test_the_two_rules_agree_on_every_pure_hebrew_label(self) -> None:
        for line in (
            "לחיצה", "גלילה", "מקלדת", "עוד", "חזרה", "ביטול", "השהיה",
            "המשך שליטה", "הבט במקום שעליו תרצה ללחוץ", "בחר מקום להנחה",
            "הפריט מוחזק", "בחר פעולה", "סגור תפריט",
        ):
            with self.subTest(line):
                self.assertEqual(T.shape_rtl(line), _old_rtl(line))


class MixedLinesTests(unittest.TestCase):
    def test_a_latin_word_is_not_reversed(self) -> None:
        shaped = T.shape_rtl("פריסה: English")
        self.assertIn("English", shaped)
        self.assertNotIn("hsilgnE", shaped)

    def test_the_hebrew_still_reads_right_to_left(self) -> None:
        # Rendered left to right, the Hebrew run must appear reversed so that
        # reading it from the right gives the original word.
        shaped = T.shape_rtl("פריסה: English")
        self.assertIn("פריסה"[::-1], shaped)

    def test_the_hebrew_ends_up_to_the_right_of_the_latin(self) -> None:
        shaped = T.shape_rtl("פריסה: English")
        self.assertGreater(shaped.index("פריסה"[::-1]), shaped.index("English"))

    def test_a_number_keeps_its_digits_in_order(self) -> None:
        shaped = T.shape_rtl("עמוד 12 מתוך 4")
        self.assertIn("12", shaped)
        self.assertNotIn("21", shaped)

    def test_a_letter_preview_keeps_its_letters_in_reading_order(self) -> None:
        shaped = T.shape_rtl("עמוד הבא")
        self.assertEqual(shaped, _old_rtl("עמוד הבא"))


class UntouchedTests(unittest.TestCase):
    def test_a_line_with_no_hebrew_is_returned_exactly(self) -> None:
        for line in ("Enter", "Escape", "GAZELINK", "1280x720", ""):
            self.assertEqual(T.shape_rtl(line), line)

    def test_shaping_is_stable_on_an_empty_string(self) -> None:
        self.assertEqual(T.shape_rtl(""), "")


class BracketTests(unittest.TestCase):
    def test_brackets_swap_hands_in_a_right_to_left_line(self) -> None:
        shaped = T.shape_rtl("(מהירות)")
        self.assertTrue(shaped.startswith("("), shaped)
        self.assertTrue(shaped.endswith(")"), shaped)


class RunTests(unittest.TestCase):
    def test_runs_cover_the_whole_line_without_losing_a_character(self) -> None:
        line = "עמוד 3: English (סוף)"
        self.assertEqual("".join(chunk for _, chunk in T.runs(line)), line)

    def test_shaping_never_loses_or_adds_characters(self) -> None:
        for line in ("עמוד 3: English (סוף)", "שלום", "Enter", "א1ב2ג3"):
            self.assertEqual(sorted(T.shape_rtl(line)), sorted(line), line)


if __name__ == "__main__":
    unittest.main()
