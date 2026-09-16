"""The scanning keyboard: two winks for one letter, and a sweep that stops.

Pure state machine on a monotonic clock, so all of this is deterministic.
Nothing here produces a keystroke -- it produces a ``Key``, and ``gf_keys`` is
the only thing that can send one.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_actions as A  # noqa: E402
import gf_keyboard as KB  # noqa: E402

STEP = 0.1  # 100 ms a step in these tests
SETTLE = 0.03  # 30 ms before a cell may be taken, scaled to STEP


def _kb(
    *,
    scan_ms: float = 100.0,
    sweeps: int = 3,
    layout: str = "hebrew",
    settle_ms: float = SETTLE * 1000.0,
    lag_ms: float = 0.0,
) -> KB.ScanningKeyboard:
    return KB.ScanningKeyboard(
        KB.ScanConfig(
            scan_ms=scan_ms, max_sweeps=sweeps, settle_ms=settle_ms, wink_lag_ms=lag_ms
        ),
        layout_name=layout,
    )


def _advance(kb: KB.ScanningKeyboard, steps: int, *, start: float = 0.0) -> float:
    """Move the highlight on ``steps`` times. Returns the clock afterwards."""

    now = start
    kb.update(now)
    for _ in range(steps):
        now += STEP + 0.001
        kb.update(now)
    return now


def _take(kb: KB.ScanningKeyboard, now: float) -> tuple[KB.Key | None, float]:
    """A wink on a cell that has been on screen long enough to be a choice.

    Every deliberate wink in these tests goes through here, because a wink
    that lands sooner than ``settle_ms`` after a cell appeared is not a
    reaction to that cell and the keyboard refuses it.
    """

    kb.update(now)
    at = now + SETTLE + 0.001
    return kb.select(at), at


class SweepTests(unittest.TestCase):
    def test_the_highlight_moves_on_by_itself(self) -> None:
        kb = _kb()
        self.assertEqual(kb.index, 0)
        _advance(kb, 2)
        self.assertEqual(kb.index, 2)

    def test_it_does_not_move_before_its_time_is_up(self) -> None:
        kb = _kb()
        kb.update(0.0)
        kb.update(0.05)
        self.assertEqual(kb.index, 0)

    def test_it_wraps_round_the_groups(self) -> None:
        kb = _kb()
        _advance(kb, len(kb.groups))
        self.assertEqual(kb.index, 0)


class TwoWinksOneLetterTests(unittest.TestCase):
    def test_the_first_wink_opens_a_group_and_types_nothing(self) -> None:
        """The same shape as the recovery menu: the first deliberate close
        only opens the list. A wink arriving with nothing on screen must not
        select anything."""

        kb = _kb()
        key, _ = _take(kb, 0.0)
        self.assertIsNone(key)
        self.assertIs(kb.level, KB.Level.KEYS)
        self.assertEqual(kb.typed, "")

    def test_the_second_wink_produces_one_key(self) -> None:
        kb = _kb()
        _, now = _take(kb, 0.0)
        key, _ = _take(kb, now)
        self.assertIsNotNone(key)
        self.assertEqual(key.text, KB.HEBREW[0])

    def test_two_winks_produce_exactly_one_character(self) -> None:
        kb = _kb()
        _, now = _take(kb, 0.0)
        key, _ = _take(kb, now)
        kb.on_sent(key)
        self.assertEqual(len(kb.typed), 1)

    def test_it_returns_to_the_groups_after_a_letter(self) -> None:
        """The next letter starts from the top rather than continuing inside
        whichever group the last one came from."""

        kb = _kb()
        _, now = _take(kb, 0.0)
        _take(kb, now)
        self.assertIs(kb.level, KB.Level.GROUPS)
        self.assertEqual(kb.index, 0)

    def test_the_letter_chosen_is_the_one_that_was_highlighted(self) -> None:
        kb = _kb()
        now = _advance(kb, 1)  # second group
        _, now = _take(kb, now)
        now = _advance(kb, 2, start=now)  # third key inside it
        key, _ = _take(kb, now)
        self.assertEqual(key.text, kb.groups[1].keys[2].text)


class WinkTimingTests(unittest.TestCase):
    """The cell a wink takes is the one the person was watching.

    Reported live: "I choose letters and it types other ones." A wink is
    stamped on a camera frame and acted on a queue and a display frame later,
    and over a 1200 ms cell that is enough for the highlight to have moved on.
    Everything here is about the gap between the two moments.
    """

    def test_a_wink_stamped_before_the_step_takes_the_earlier_cell(self) -> None:
        kb = _kb()
        now = _advance(kb, 1)  # the second group is live from `now`
        # The wink happened while the second group was still up; the loop only
        # gets to it after the sweep has stepped on to the third.
        stamped = now + STEP / 2.0
        kb.update(now + STEP + 0.001)
        self.assertEqual(kb.index, 2, "the fixture did not actually step")
        kb.select(now + STEP + 0.002, when_s=stamped)
        self.assertEqual(kb.group_index, 1, "it opened the group the sweep had moved to")
        self.assertEqual(kb.winks_resolved_back, 1)

    def test_without_a_stamp_it_still_takes_the_cell_that_is_live(self) -> None:
        """Callers with no stamp are unchanged: the old behaviour is the
        default, so nothing that cannot measure the moment is made worse."""

        kb = _kb()
        now = _advance(kb, 2)
        kb.select(now + SETTLE + 0.001)
        self.assertEqual(kb.group_index, 2)
        self.assertEqual(kb.winks_resolved_back, 0)

    def test_the_lag_moves_the_moment_the_wink_is_judged_by(self) -> None:
        """``wink_lag_ms`` is how long the lid had been moving before the
        wink was stamped. With it, a stamp just after a step still resolves to
        the cell the person was reacting to."""

        kb = _kb(lag_ms=40.0)  # more than the 1 ms overshoot below
        now = _advance(kb, 1)
        after_step = now + STEP + 0.001
        kb.update(after_step)
        self.assertEqual(kb.index, 2)
        kb.select(after_step + 0.001, when_s=after_step)
        self.assertEqual(kb.group_index, 1)

    def test_a_wink_too_soon_after_a_new_screen_takes_nothing(self) -> None:
        """Nothing has stepped yet, so this is a screen nobody has read."""

        kb = _kb()
        kb.update(0.0)
        self.assertIsNone(kb.select(SETTLE / 2.0))
        self.assertIs(kb.level, KB.Level.GROUPS, "it descended on a cell nobody had read")
        self.assertEqual(kb.winks_too_soon, 1)

    def test_a_quick_wink_part_way_through_a_sweep_is_not_refused(self) -> None:
        """The settle is charged on a NEW screen and not on every cell.

        Charging it on every cell would refuse a fast deliberate wink and cost
        the person a whole pass of the sweep; part-way through, the cell before
        this one was on screen and the lag already decides between the two.
        """

        kb = _kb()
        now = _advance(kb, 1)
        key_or_none = kb.select(now + SETTLE / 2.0)
        self.assertIsNone(key_or_none)  # a group opens and produces nothing
        self.assertIs(kb.level, KB.Level.KEYS, "a quick wink mid-sweep was thrown away")
        self.assertEqual(kb.winks_too_soon, 0)

    def test_a_refused_wink_disturbs_nothing(self) -> None:
        """It must cost the person no more than the rest of the cell they are
        already waiting through: not the highlight, not its clock, not the
        sweep count that decides when the keyboard parks itself."""

        kb = _kb(sweeps=2)
        kb.update(0.0)
        index_before = kb.index
        kb.select(SETTLE / 2.0)
        self.assertEqual(kb.index, index_before)
        # The clock was not restarted: the step still falls where it would have.
        kb.update(STEP + 0.001)
        self.assertEqual(kb.index, index_before + 1)

    def test_a_second_wink_from_one_closure_cannot_type_a_letter(self) -> None:
        """The detector can emit two winks out of the jitter around one
        reopening. The first opens a group; without a settle the second takes
        whatever sits first inside a group the person has not read yet."""

        kb = _kb(scan_ms=1200.0, settle_ms=350.0)
        kb.update(0.0)
        self.assertIsNone(kb.select(0.4))  # opens the first group
        self.assertIs(kb.level, KB.Level.KEYS)
        jitter = kb.select(0.6)  # 200 ms later, out of the same closure
        self.assertIsNone(jitter, "a stray second wink typed a letter")
        self.assertEqual(kb.typed, "")
        self.assertEqual(kb.winks_too_soon, 1)

    def test_a_refused_wink_says_so_on_screen(self) -> None:
        """CLAUDE.md 4.5: a wink that does nothing and shows nothing is
        indistinguishable from a wink the camera never saw."""

        kb = _kb()
        kb.update(0.0)
        self.assertFalse(kb.refused_recently(0.0))
        kb.select(SETTLE / 2.0)
        self.assertTrue(kb.refused_recently(SETTLE / 2.0, within_s=1.2))
        self.assertFalse(
            kb.refused_recently(SETTLE / 2.0 + 2.0, within_s=1.2),
            "the notice stayed up long after the wink it was about",
        )

    def test_a_lag_longer_than_a_cell_is_refused(self) -> None:
        """One step of history is all the machine keeps, so a lag past a whole
        cell would resolve to one it no longer remembers."""

        with self.assertRaises(ValueError):
            KB.ScanConfig(scan_ms=500.0, wink_lag_ms=500.0)

    def test_a_settle_as_long_as_the_cell_is_refused(self) -> None:
        """It would mean no cell is ever takeable and the keyboard silently
        types nothing at all."""

        with self.assertRaises(ValueError):
            KB.ScanConfig(scan_ms=500.0, settle_ms=500.0)

    def test_a_negative_lag_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            KB.ScanConfig(wink_lag_ms=-1.0)


class ActionKeyTests(unittest.TestCase):
    def _actions_group(self, kb: KB.ScanningKeyboard) -> int:
        return len(kb.groups) - 1

    def _pick(self, kb: KB.ScanningKeyboard, label: str):
        """Walk to the actions group and take the key with this label."""

        now = _advance(kb, self._actions_group(kb))
        _, now = _take(kb, now)
        labels = [k.label for k in kb.keys]
        now = _advance(kb, labels.index(label), start=now)
        return _take(kb, now)[0]

    def test_cancel_goes_back_to_the_groups_and_types_nothing(self) -> None:
        kb = _kb()
        key = self._pick(kb, "ביטול")
        self.assertIs(key.control, KB.Control.CANCEL)
        self.assertIs(kb.level, KB.Level.GROUPS)
        self.assertEqual(kb.index, 0)
        self.assertEqual(kb.typed, "")

    def test_space_is_an_ordinary_character(self) -> None:
        kb = _kb()
        key = self._pick(kb, "רווח")
        self.assertEqual(key.text, " ")

    def test_backspace_and_enter_are_actions_and_not_text(self) -> None:
        kb = _kb()
        self.assertIs(self._pick(kb, "מחיקה").action, A.Action.BACKSPACE)
        kb.reset()
        self.assertIs(self._pick(kb, "Enter").action, A.Action.ENTER)

    def test_changing_the_layout_changes_the_letters(self) -> None:
        kb = _kb()
        self._pick(kb, "פריסה")
        self.assertEqual(kb.layout_name, "english")
        self.assertIn(KB.ENGLISH[0], [k.text for g in kb.groups[:-1] for k in g.keys])

    def test_every_layout_carries_the_actions_group(self) -> None:
        """Consistent so it can be learned: the last group always does
        something other than type a letter."""

        for name in KB.LAYOUT_ORDER:
            groups = KB.layout(name)
            self.assertEqual(groups[-1].label, "פעולות")
            self.assertIn("סגירה", [k.label for k in groups[-1].keys])

    def test_close_is_reported_rather_than_acted_on_here(self) -> None:
        kb = _kb()
        key = self._pick(kb, "סגירה")
        self.assertIs(key.control, KB.Control.CLOSE)


class EchoTests(unittest.TestCase):
    """What the screen shows must be what actually left the machine.

    A keystroke can be refused after it is chosen -- the target window
    changed, the mode is paused -- and an echo that showed it anyway would
    tell the person their text is somewhere it is not.
    """

    def test_nothing_is_echoed_until_the_caller_confirms_it_was_sent(self) -> None:
        kb = _kb()
        _, now = _take(kb, 0.0)
        _take(kb, now)
        self.assertEqual(kb.typed, "", "the echo ran ahead of the keystroke")

    def test_backspace_takes_one_character_off_the_echo(self) -> None:
        kb = _kb()
        kb.on_sent(KB.Key("א", text="א"))
        kb.on_sent(KB.Key("ב", text="ב"))
        kb.on_sent(KB.Key("מחיקה", action=A.Action.BACKSPACE))
        self.assertEqual(kb.typed, "א")

    def test_enter_clears_it(self) -> None:
        kb = _kb()
        kb.on_sent(KB.Key("א", text="א"))
        kb.on_sent(KB.Key("Enter", action=A.Action.ENTER))
        self.assertEqual(kb.typed, "")


class ParkingTests(unittest.TestCase):
    """A highlight that blinks round a layout for ever is a screen the person
    cannot rest their eyes on, and it means every stray wink lands on whatever
    the timer happened to reach."""

    def test_it_stops_itself_after_the_configured_sweeps(self) -> None:
        kb = _kb(sweeps=2)
        _advance(kb, len(kb.groups) * 2 + 1)
        self.assertTrue(kb.parked)
        self.assertEqual(kb.parked_times, 1)

    def test_a_parked_sweep_stops_moving(self) -> None:
        kb = _kb(sweeps=1)
        now = _advance(kb, len(kb.groups) + 1)
        index = kb.index
        _advance(kb, 5, start=now)
        self.assertEqual(kb.index, index)

    def test_a_wink_restarts_it_without_choosing_anything(self) -> None:
        """The highlight was parked wherever the last sweep left it, and
        taking that would be taking a key the person never watched arrive."""

        kb = _kb(sweeps=1)
        now = _advance(kb, len(kb.groups) + 1)
        self.assertIsNone(kb.select(now))
        self.assertFalse(kb.parked)
        self.assertIs(kb.level, KB.Level.GROUPS)
        self.assertEqual(kb.typed, "")

    def test_choosing_something_resets_the_count(self) -> None:
        kb = _kb(sweeps=2)
        _advance(kb, len(kb.groups))
        _, now = _take(kb, 1.0)
        _take(kb, now)
        _advance(kb, len(kb.groups), start=2.0)
        self.assertFalse(kb.parked, "the sweep parked as if nothing had been chosen")


class LayoutContentTests(unittest.TestCase):
    def test_the_hebrew_layout_carries_the_final_forms(self) -> None:
        """A person typing Hebrew needs them, and nothing here is clever
        enough to substitute one at the end of a word."""

        letters = {k.text for g in KB.layout("hebrew")[:-1] for k in g.keys}
        for final in "ךםןףץ":
            self.assertIn(final, letters)

    def test_no_group_is_longer_than_the_letters_it_names(self) -> None:
        for name in KB.LAYOUT_ORDER:
            for group in KB.layout(name)[:-1]:
                self.assertLessEqual(len(group.keys), 6)

    def test_a_key_must_do_exactly_one_thing(self) -> None:
        with self.assertRaises(ValueError):
            KB.Key("bad", text="a", action=A.Action.ENTER)
        with self.assertRaises(ValueError):
            KB.Key("bad")


if __name__ == "__main__":
    unittest.main()
