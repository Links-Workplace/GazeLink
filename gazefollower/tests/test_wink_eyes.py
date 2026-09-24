"""Two eyes, two buttons -- and one closure is never two clicks.

The eye picks the button (operator, 24.9.2026): RIGHT wink = right click, LEFT
wink = left click. Both detectors run on every camera frame through the REAL
``LiveRunner``, with the real openness gates. What these tests pin down:

* each eye is detected as ITSELF, through the library's mirrored field names;
* a blink (both eyes together) is neither;
* the dangerous case: after a right wink, the eyes reopen unevenly -- the right
  opens first while the left is still low -- and for a few frames that looks
  exactly like a LEFT wink. It must not become a left click. The first version
  of the guard only cancelled the other detector, whose latch cleared on the
  very next frame because that eye was open; a mutation removing it passed
  every test. This file is what fails now;
* the guard does not swallow a deliberate left wink made a moment later.

Frames at 30 fps on a fed clock. No camera, no window, no OS input.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_common as C  # noqa: E402
import gf_live as L  # noqa: E402
from gazelink_core.interaction import gesture as GEST  # noqa: E402

FRAME_S = 1.0 / 30.0
OPEN = 100.0
SHUT = 10.0


class _Eyes:
    """Feeds the real runner. Arguments are the PERSON's eyes; the library's
    fields are the image's, so they are swapped on the way in."""

    def __init__(self) -> None:
        self.t = 1000.0
        rig = C.RigGeometry(60.0, -2.0, 120.0, 33.75, 5120, 1440)
        self.runner = L.LiveRunner(
            None, None, rig, None, head_builder=lambda _f: None, clock=lambda: self.t
        )
        self.events: list = []

    def frames(self, n: int, *, left: float, right: float) -> None:
        for _ in range(n):
            self.t += FRAME_S
            self.runner.on_frame(
                # image-left = the person's RIGHT eye
                SimpleNamespace(status=True, left_eye_openness=right, right_eye_openness=left),
                SimpleNamespace(status=False),
            )
            self.events.extend(self.runner.drain_wink_events())

    def eyes(self) -> list[GEST.Eye]:
        return [e[2] for e in self.events]


def _settled() -> _Eyes:
    eyes = _Eyes()
    eyes.frames(30, left=OPEN, right=OPEN)
    return eyes


class EachEyeIsItselfTests(unittest.TestCase):
    def test_a_right_wink_is_the_right_eye(self) -> None:
        eyes = _settled()
        eyes.frames(6, left=OPEN, right=SHUT)
        eyes.frames(20, left=OPEN, right=OPEN)
        self.assertEqual(eyes.eyes(), [GEST.Eye.RIGHT])

    def test_a_left_wink_is_the_left_eye(self) -> None:
        eyes = _settled()
        eyes.frames(6, left=SHUT, right=OPEN)
        eyes.frames(20, left=OPEN, right=OPEN)
        self.assertEqual(eyes.eyes(), [GEST.Eye.LEFT])

    def test_a_blink_is_neither(self) -> None:
        eyes = _settled()
        eyes.frames(6, left=SHUT, right=SHUT)
        eyes.frames(20, left=OPEN, right=OPEN)
        self.assertEqual(eyes.eyes(), [])


class OneClosureIsOneClickTests(unittest.TestCase):
    def test_an_uneven_reopening_after_a_right_wink_is_not_a_left_click(self) -> None:
        eyes = _settled()
        eyes.frames(6, left=OPEN, right=SHUT)  # the right wink: fires in here
        # The right eye opens first; the left lags low for ~130 ms. On its own
        # this is exactly what a left wink looks like.
        eyes.frames(4, left=SHUT, right=OPEN)
        eyes.frames(20, left=OPEN, right=OPEN)
        self.assertEqual(eyes.eyes(), [GEST.Eye.RIGHT], "one closure became two clicks")
        self.assertGreaterEqual(eyes.runner.winks_suppressed_other_eye, 1)

    def test_the_same_the_other_way_round(self) -> None:
        eyes = _settled()
        eyes.frames(6, left=SHUT, right=OPEN)
        eyes.frames(4, left=OPEN, right=SHUT)
        eyes.frames(20, left=OPEN, right=OPEN)
        self.assertEqual(eyes.eyes(), [GEST.Eye.LEFT])

    def test_a_deliberate_left_wink_a_moment_later_still_counts(self) -> None:
        # The guard is the closure plus its cooldown, not a ban on the other eye.
        eyes = _settled()
        eyes.frames(6, left=OPEN, right=SHUT)
        eyes.frames(15, left=OPEN, right=OPEN)  # 500 ms, both open
        eyes.frames(6, left=SHUT, right=OPEN)
        eyes.frames(20, left=OPEN, right=OPEN)
        self.assertEqual(eyes.eyes(), [GEST.Eye.RIGHT, GEST.Eye.LEFT])

    def test_a_mode_change_mid_wink_drops_both_eyes(self) -> None:
        eyes = _settled()
        eyes.frames(1, left=SHUT, right=OPEN)  # the left closure has begun
        eyes.runner.cancel_wink()
        eyes.frames(6, left=SHUT, right=OPEN)
        eyes.frames(20, left=OPEN, right=OPEN)
        self.assertEqual(eyes.eyes(), [], "a closure from the old mode fired in the new one")


NARROWED = 40.0  # under the 0.55 gate, as the other eye goes during a wink
DEEP = 5.0


def _long_closes(eyes: _Eyes) -> int:
    return sum(1 for _w, e in eyes.runner.drain_gesture_events() if e is GEST.Event.CONFIRM)


class AHeldWinkIsNotTheMenuTests(unittest.TestCase):
    """Measured 24.9: the other eye was under the gate as often as the winking
    one, so a held left wink read as both eyes shut and opened the menu."""

    def test_a_held_left_wink_with_the_right_narrowed_does_not_open_the_menu(self) -> None:
        eyes = _settled()
        eyes.frames(45, left=DEEP, right=NARROWED)  # 1.5 s, both under the gate
        eyes.frames(20, left=OPEN, right=OPEN)
        self.assertEqual(eyes.eyes(), [GEST.Eye.LEFT])
        self.assertEqual(_long_closes(eyes), 0, "one wink opened the menu as well")

    def test_the_same_for_the_right_eye(self) -> None:
        eyes = _settled()
        eyes.frames(45, left=NARROWED, right=DEEP)
        eyes.frames(20, left=OPEN, right=OPEN)
        self.assertEqual(eyes.eyes(), [GEST.Eye.RIGHT])
        self.assertEqual(_long_closes(eyes), 0)

    def test_a_real_long_close_still_opens_the_menu(self) -> None:
        eyes = _settled()
        eyes.frames(45, left=SHUT, right=SHUT)
        eyes.frames(20, left=OPEN, right=OPEN)
        self.assertEqual(eyes.eyes(), [])
        self.assertEqual(_long_closes(eyes), 1)

    def test_the_claim_ends_when_both_eyes_open(self) -> None:
        eyes = _settled()
        eyes.frames(10, left=DEEP, right=NARROWED)
        eyes.frames(10, left=OPEN, right=OPEN)
        eyes.frames(45, left=SHUT, right=SHUT)
        eyes.frames(20, left=OPEN, right=OPEN)
        self.assertEqual(eyes.eyes(), [GEST.Eye.LEFT])
        self.assertEqual(_long_closes(eyes), 1, "a past wink blocked the next long close")


if __name__ == "__main__":
    unittest.main()
