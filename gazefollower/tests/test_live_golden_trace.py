"""Behaviour equivalence of the live session against a pre-refactor trace.

``tests/golden/live_trace.json`` was captured from ``run_live`` BEFORE the
ARCH-01 restructuring (stage C). Every later stage must reproduce it exactly:
inputs with their fake timestamps, draw calls, printed lines and return code.

A difference is either a regression or a documented, deliberate bug fix. For
a deliberate fix, regenerate with ``GAZELINK_UPDATE_LIVE_TRACE=1`` and record the
expected-before/after in TASKS.md (ARCH-01). Never regenerate to make a
refactor pass.

Drag is not characterised: the live path has no drag action.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_gesture as GEST  # noqa: E402
import gf_menu as MENU  # noqa: E402
import gf_scroll as SCR  # noqa: E402

from live_trace_harness import Script, run_scenario  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden" / "live_trace.json"
CENTRE = (0.5, 0.5)
CONFIRM = GEST.Event.CONFIRM


def _centres(page: str, *, paused: bool = False) -> dict[str, tuple[float, float]]:
    items = MENU._pages(paused=paused)[page]
    return {b.key: b.centre for b in MENU.page_buttons(items)}


MAIN = _centres("main")
NAV = _centres("nav")
CLICK = _centres("click")
SYSTEM_PAUSED = _centres("system", paused=True)
UP = SCR.scroll_zones()[0].centre


def legs(*spans: tuple[float, tuple[float, float] | None]):
    """Gaze by time: each (until_s, point) holds until that second."""

    def gaze(t: float):
        for until, point in spans:
            if t < until:
                return point
        return spans[-1][1]

    return gaze


def _moving(t: float) -> tuple[float, float]:
    return (0.3 + 0.4 * min(t, 0.5) / 0.5, 0.5)


SCENARIOS: dict[str, tuple[Script, dict]] = {
    "cursor_follow_loss_recovery": (
        Script(
            gaze=lambda t: _moving(t) if t < 0.5 else (0.5, 0.4),
            face=lambda t: not (0.5 <= t < 0.8),
        ),
        {"max_seconds": 1.2},
    ),
    "blink_then_wink_double_click": (
        Script(
            gaze=lambda t: (0.6, 0.5),
            steady=lambda t: not (0.3 <= t < 0.4),
            winks=[(0.7, (0.6, 0.5), 0.0)],
        ),
        {"max_seconds": 1.1},
    ),
    "pause_and_resume_with_space": (
        Script(
            gaze=lambda t: (0.4, 0.6),
            space_down=[(0.2, 0.25), (0.6, 0.65)],
            winks=[(0.4, (0.4, 0.6), 0.0), (0.9, (0.4, 0.6), 0.0)],
        ),
        {"max_seconds": 1.2},
    ),
    "stale_and_duplicate_winks": (
        Script(
            gaze=lambda t: CENTRE,
            winks=[(0.3, CENTRE, 0.6), (0.5, CENTRE, 0.0), (0.55, CENTRE, 0.0)],
        ),
        {"max_seconds": 1.0, "wink_click": "single"},
    ),
    "wink_with_no_aim": (
        Script(gaze=lambda t: CENTRE, winks=[(0.3, None, 0.0)]),
        {"max_seconds": 0.6},
    ),
    "menu_click_type_then_right_click": (
        Script(
            gaze=legs(
                (0.35, CENTRE),
                (0.7, MAIN["click-type"]),
                (0.85, CENTRE),
                (1.2, CLICK["right"]),
                (5.0, CENTRE),
            ),
            gestures=[(0.2, CONFIRM)],
            winks=[(1.7, CENTRE, 0.0)],
        ),
        {"max_seconds": 2.1, "menu_dwell_ms": 40.0},
    ),
    "menu_second_long_close_ignored_and_wink_in_menu": (
        Script(
            gaze=lambda t: CENTRE,
            gestures=[(0.2, CONFIRM), (0.5, CONFIRM)],
            winks=[(0.7, CENTRE, 0.0)],
        ),
        {"max_seconds": 1.0},
    ),
    "scroll_seed_notches_and_face_loss": (
        Script(
            gaze=legs((0.35, CENTRE), (0.7, MAIN["scroll"]), (1.0, UP), (1.4, CENTRE), (2.6, UP)),
            face=lambda t: not (2.0 <= t < 2.2),
            gestures=[(0.2, CONFIRM)],
            winks=[(2.4, UP, 0.0)],
        ),
        {"max_seconds": 2.8, "menu_dwell_ms": 40.0},
    ),
    "start_scrolling_down": (
        Script(gaze=legs((0.2, CENTRE), (1.5, SCR.scroll_zones()[1].centre))),
        {"max_seconds": 1.5, "start_scrolling": True},
    ),
    "keyboard_type_then_target_lost": (
        Script(
            gaze=legs((0.35, CENTRE), (9.0, MAIN["keyboard"])),
            gestures=[(0.2, CONFIRM)],
            winks=[(1.6, CENTRE, 0.0), (3.2, CENTRE, 0.0), (4.8, CENTRE, 0.0), (6.4, CENTRE, 0.0)],
            foreground=lambda t: 4242 if t < 4.0 else 9999,
        ),
        {"max_seconds": 7.5, "menu_dwell_ms": 40.0},
    ),
    "paused_menu_command_refused_then_resume": (
        Script(
            gaze=legs(
                (0.35, CENTRE),
                (0.7, MAIN["more-1"]),
                (0.85, CENTRE),
                (1.2, NAV["back"]),
                (1.4, CENTRE),
                (1.75, NAV["more-2"]),
                (1.9, CENTRE),
                (2.4, SYSTEM_PAUSED["pause"]),
                (5.0, CENTRE),
            ),
            gestures=[(0.2, CONFIRM)],
        ),
        {"max_seconds": 2.8, "menu_dwell_ms": 40.0, "start_active": False},
    ),
    "escape_stops_everything": (
        Script(gaze=lambda t: _moving(t), escape_at=0.4),
        {"max_seconds": 2.0},
    ),
    "cursor_only_no_clicks": (
        Script(gaze=lambda t: _moving(t), winks=[(0.3, CENTRE, 0.0)]),
        {"max_seconds": 0.6, "click_by": "off"},
    ),
    "refused_eyes_toggle_with_menu": (
        Script(),
        {"max_seconds": 0.2, "toggle_by": "eyes"},
    ),
}


def capture() -> dict[str, dict]:
    return {
        name: run_scenario(script, **kw).to_golden() for name, (script, kw) in SCENARIOS.items()
    }


class GoldenTraceTests(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        if os.environ.get("GAZELINK_UPDATE_LIVE_TRACE") == "1":
            GOLDEN.parent.mkdir(parents=True, exist_ok=True)
            GOLDEN.write_text(
                json.dumps(capture(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
            )
        cls.golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        # A file just written is compared with itself: refuse to call that a pass.
        cls.regenerated = os.environ.get("GAZELINK_UPDATE_LIVE_TRACE") == "1"

    def setUp(self) -> None:
        if self.regenerated:
            self.skipTest("golden file regenerated in this run; comparison skipped")

    def test_every_scenario_is_in_the_golden_file(self) -> None:
        self.assertEqual(sorted(self.golden), sorted(SCENARIOS))

    def test_scenarios_match_the_pre_refactor_trace(self) -> None:
        for name, (script, kw) in SCENARIOS.items():
            with self.subTest(scenario=name):
                got = run_scenario(script, **kw).to_golden()
                # JSON round-trip so tuples compare as the stored lists do.
                got = json.loads(json.dumps(got, ensure_ascii=False))
                expected = self.golden[name]
                self.assertEqual(got["returncode"], expected["returncode"])
                self.assertEqual(got["inputs"], expected["inputs"])
                self.assertEqual(got["stdout"], expected["stdout"])
                self.assertEqual(got["draw_kinds"], expected["draw_kinds"])
                self.assertEqual(got["draw_count"], expected["draw_count"])
                self.assertEqual(got["draw_sha256"], expected["draw_sha256"])

    def test_the_scenarios_exercise_what_they_are_named_for(self) -> None:
        """A trace that recorded nothing would match itself for ever."""

        g = self.golden
        kinds = lambda name: {e[0] for e in g[name]["inputs"]}  # noqa: E731
        self.assertIn("button", kinds("blink_then_wink_double_click"))
        self.assertIn(
            8,
            [e[2] for e in g["menu_click_type_then_right_click"]["inputs"] if e[0] == "button"],
            "the chosen right click never happened",
        )
        self.assertEqual(
            len([e for e in g["stale_and_duplicate_winks"]["inputs"] if e[0] == "button"]),
            2,
            "exactly one of the three winks may click: one is stale, one a duplicate",
        )
        self.assertIn("wheel", kinds("scroll_seed_notches_and_face_loss"))
        self.assertIn("wheel", kinds("start_scrolling_down"))
        self.assertIn("key", kinds("keyboard_type_then_target_lost"))
        self.assertNotIn("key", kinds("paused_menu_command_refused_then_resume"))
        self.assertNotIn("button", kinds("cursor_only_no_clicks"))
        self.assertTrue(
            any(k == "board" for k, _n in g["menu_click_type_then_right_click"]["draw_kinds"])
        )
        self.assertTrue(
            any("mode KEYBOARD" in line for line in g["keyboard_type_then_target_lost"]["stdout"])
        )
        self.assertTrue(
            str(g["refused_eyes_toggle_with_menu"]["returncode"]).startswith("SystemExit")
        )


if __name__ == "__main__":
    unittest.main()
