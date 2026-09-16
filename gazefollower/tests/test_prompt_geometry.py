"""Where the practice prompt is drawn, against the buttons it must not sit on.

No window: line sizes come from pygame's font metrics under the dummy video
driver. The real screen is drawn with those same metrics, so without pygame
the size-dependent checks are SKIPPED rather than run on a guessed width (a
guess either passes a layout that does not fit or fails one that does). The
question is geometric -- does reading the task put the gaze on or near a
button -- and needs no camera and no display.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_dwell as D  # noqa: E402
import gf_record as R  # noqa: E402

W, H = 5120, 1440
# Worst per-target bias the 15 Sep comparison sized its layout against
# (results/select_compare/20260915_073919.json, bias_used, pooled arm).
MEASURED_BIAS_X = 0.0724
MEASURED_BIAS_Y = 0.3282

TRIAL_LINES = ["trial 18/18    LOOK AT:  LOW-R", "dwell 900 ms    Esc to stop"]
MENU_LINES = [
    "",
    "MENU  (short close cycles, long close confirms)",
    "  > Keep going",
    "    Pause",
    "    Stop the session",
]
# Both the current fixed-window line and the longer one it replaced: the
# longer string is kept as a worst case for the side margin.
LONGEST_LINES = [
    "keep looking at the target until it ends    Esc to stop",
    "keep looking until it ends    Esc to stop",
]


def _sizes(lines: list[str], px: int) -> list[tuple[int, int]]:
    try:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame  # noqa: PLC0415
    except ImportError as exc:
        raise unittest.SkipTest("pygame not installed: real font metrics unavailable") from exc
    pygame.font.init()
    font = pygame.font.Font(None, px)
    return [font.size(line) for line in lines]


def _hits(gaps: list[tuple[str, float, float]]) -> set[str]:
    return {key for key, gx, gy in gaps if gx == 0.0 and gy == 0.0}


class TopAnchorDocumentsTheProblem(unittest.TestCase):
    def setUp(self) -> None:
        self.buttons = D.layout_zones()

    def test_top_prompt_with_menu_open_is_drawn_on_up_c(self) -> None:
        lines = TRIAL_LINES + MENU_LINES
        rects = R.prompt_layout(_sizes(lines, 64), "top", W, H)
        self.assertIn("UP-C", _hits(R.prompt_button_gaps(rects, self.buttons, W, H)))

    def test_top_prompt_without_menu_sits_within_the_vertical_bias_of_up_c(self) -> None:
        rects = R.prompt_layout(_sizes(TRIAL_LINES, 64), "top", W, H)
        gaps = [g for g in R.prompt_button_gaps(rects, self.buttons, W, H) if g[0] == "UP-C"]
        self.assertTrue(all(gx == 0.0 for _k, gx, _gy in gaps))
        self.assertLess(min(gy for _k, _gx, gy in gaps), MEASURED_BIAS_Y)

    def test_top_placement_is_the_original_formula(self) -> None:
        sizes = [(635, 44), (563, 46)]
        self.assertEqual(
            R.prompt_layout(sizes, "top", W, H),
            [(W // 2 - 635 // 2, 40, 635, 44), (W // 2 - 563 // 2, 110, 563, 46)],
        )


class SideAnchorClearsEveryButton(unittest.TestCase):
    def check(self, buttons: list[D.Button], lines: list[str]) -> None:
        rects = R.prompt_layout(_sizes(lines, 44), "sides", W, H)
        self.assertEqual(len(rects), 2 * len(lines))
        for x, y, w, h in rects:
            self.assertGreaterEqual(x, 0)
            self.assertLessEqual(x + w, W)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(y + h, H)
        gaps = R.prompt_button_gaps(rects, buttons, W, H)
        self.assertEqual(_hits(gaps), set())
        self.assertGreater(min(gx for _k, gx, _gy in gaps), MEASURED_BIAS_X)

    def test_zones_with_trial_lines_and_open_menu(self) -> None:
        self.check(D.layout_zones(), TRIAL_LINES + MENU_LINES)

    def test_zones_with_the_longest_prompt_line(self) -> None:
        self.check(D.layout_zones(), LONGEST_LINES)

    def test_the_two_copies_mirror_about_the_centre(self) -> None:
        rects = R.prompt_layout([(400, 40)], "sides", W, H)
        (lx, ly, lw, _), (rx, ry, rw, _) = rects
        self.assertEqual(ly, ry)
        self.assertAlmostEqual((lx + lw / 2 + rx + rw / 2) / 2, W / 2, delta=1)

    def test_sides_refuses_a_prompt_that_would_spill_into_the_band(self) -> None:
        # 1920 px wide: the menu title the harness really draws is wider than
        # the 576 px margin, and would otherwise be drawn into the band.
        with self.assertRaises(ValueError):
            R.prompt_layout(_sizes(TRIAL_LINES + MENU_LINES, 44), "sides", 1920, 1080)

    def test_unknown_anchor_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            R.prompt_layout([(10, 10)], "corner", W, H)


if __name__ == "__main__":
    unittest.main()
