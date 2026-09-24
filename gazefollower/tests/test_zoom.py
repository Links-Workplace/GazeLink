"""The magnifier: what it refuses, and that it clicks where it says it does.

No screen is read here. The fake capture returns bytes a test controls, so the
freshness rule and the coordinate mapping are both checked exactly.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gazelink_core.interaction import zoom as Z  # noqa: E402
from gazelink_core.interaction.capture_port import DeviceRect  # noqa: E402

# The rig: one 5120x1440 panel at the desktop origin.
SCREEN_W, SCREEN_H = 5120, 1440


def to_device(point):
    x, y = point
    return (
        int(round(min(max(x, 0.0), 1.0) * (SCREEN_W - 1))),
        int(round(min(max(y, 0.0), 1.0) * (SCREEN_H - 1))),
    )


class FakeCapture:
    def __init__(self, rect, fill):
        self._rect = rect
        self._fill = fill

    @property
    def rect(self):
        return self._rect

    def crop_bytes(self, rect):
        if not (
            rect.x0 >= self._rect.x0 and rect.y0 >= self._rect.y0
            and rect.x1 <= self._rect.x1 and rect.y1 <= self._rect.y1
        ):
            raise ValueError("outside the capture")
        return bytes([self._fill]) * (rect.width * rect.height)


class FakeScreen:
    """Counts grabs and can be told to change what the screen now shows."""

    def __init__(self, fill=7):
        self.fill = fill
        self.grabs = []
        self.fail = False

    def grab(self, rect):
        if self.fail:
            raise RuntimeError("no screen")
        self.grabs.append(rect)
        return FakeCapture(rect, self.fill)


def build(**kw):
    screen = kw.pop("screen", None) or FakeScreen()
    zoom = Z.ZoomSelection(
        capture=screen,
        to_device=kw.pop("to_device", to_device),
        overlay_is_layered=kw.pop("layered", lambda: True),
        foreground=kw.pop("foreground", lambda: 4242),
        our_windows=kw.pop("our_windows", lambda: {99}),
        **kw,
    )
    return zoom, screen


class TheOverlayMustBeLayeredFirstTests(unittest.TestCase):
    def test_an_unlayered_overlay_refuses_to_capture_at_all(self) -> None:
        # Unlayered, our own window IS in the picture, and the magnifier would
        # show a photograph of itself.
        zoom, screen = build(layered=lambda: False)
        outcome = zoom.open((0.5, 0.5))
        self.assertFalse(outcome.allowed)
        self.assertIs(outcome.reason, Z.Refusal.NOT_LAYERED)
        self.assertEqual(screen.grabs, [])
        self.assertIs(zoom.phase, Z.Phase.CLOSED)

    def test_a_layered_overlay_captures_once(self) -> None:
        zoom, screen = build()
        self.assertTrue(zoom.open((0.5, 0.5)).allowed)
        self.assertEqual(len(screen.grabs), 1)


class OneCoordinateSpaceTests(unittest.TestCase):
    def test_the_captured_rectangle_comes_from_the_pointer_mapping(self) -> None:
        zoom, screen = build()
        zoom.open((0.5, 0.5))
        rect = screen.grabs[0]
        expected_left = to_device((0.5 - Z.SPAN_NEAR / 2, 0.0))[0]
        expected_right = to_device((0.5 + Z.SPAN_NEAR / 2, 0.0))[0]
        self.assertEqual((rect.x0, rect.x1), (expected_left, expected_right))

    def test_the_click_lands_inside_the_region_that_was_captured(self) -> None:
        zoom, screen = build()
        zoom.open((0.5, 0.5))
        zoom.look(0.5, 0.5)
        outcome = zoom.confirm()
        self.assertTrue(outcome.allowed, outcome.reason)
        self.assertTrue(screen.grabs[0].contains(*outcome.at_device))

    def test_a_point_at_the_panel_corner_maps_to_the_region_corner(self) -> None:
        zoom, screen = build()
        zoom.open((0.5, 0.5))
        zoom.look(0.0, 0.0)
        outcome = zoom.confirm()
        rect = screen.grabs[0]
        self.assertEqual(outcome.at_device, (rect.x0, rect.y0))

    def test_a_region_at_the_screen_edge_is_clamped_on_screen(self) -> None:
        zoom, screen = build()
        zoom.open((0.0, 0.0))
        rect = screen.grabs[0]
        self.assertGreaterEqual(rect.x0, 0)
        self.assertGreaterEqual(rect.y0, 0)
        self.assertLessEqual(rect.x1, SCREEN_W)


class TheFrozenPictureTests(unittest.TestCase):
    def test_looking_around_never_takes_a_new_picture(self) -> None:
        zoom, screen = build()
        zoom.open((0.5, 0.5))
        for i in range(20):
            zoom.look(i / 20.0, 0.5)
        self.assertEqual(len(screen.grabs), 1)

    def test_the_picture_changes_only_on_a_deliberate_step(self) -> None:
        zoom, screen = build()
        zoom.open((0.5, 0.5))
        zoom.look(0.5, 0.5)
        zoom.deeper()
        self.assertEqual(len(screen.grabs), 2)
        self.assertLess(zoom.region.span, Z.SPAN_NEAR)

    def test_going_back_restores_the_previous_region(self) -> None:
        zoom, _ = build()
        zoom.open((0.5, 0.5))
        before = zoom.region
        zoom.look(0.5, 0.5)
        zoom.deeper()
        zoom.back()
        self.assertEqual((zoom.region.cx, zoom.region.span), (before.cx, before.span))

    def test_back_at_the_top_does_nothing_rather_than_closing(self) -> None:
        zoom, _ = build()
        zoom.open((0.5, 0.5))
        self.assertFalse(zoom.back().allowed)
        self.assertIs(zoom.phase, Z.Phase.CHOOSING)

    def test_the_overview_shows_the_whole_screen(self) -> None:
        zoom, screen = build()
        zoom.open((0.5, 0.5))
        zoom.overview()
        self.assertEqual(zoom.region.span, Z.SPAN_OVERVIEW)
        self.assertEqual(screen.grabs[-1].width, SCREEN_W - 1)

    def test_cancel_drops_the_picture(self) -> None:
        zoom, _ = build()
        zoom.open((0.5, 0.5))
        zoom.cancel()
        self.assertIsNone(zoom.frame)
        self.assertIs(zoom.phase, Z.Phase.CLOSED)


class EnteringEmitsNothingTests(unittest.TestCase):
    def test_opening_produces_no_click(self) -> None:
        zoom, _ = build()
        outcome = zoom.open((0.5, 0.5))
        self.assertIsNone(outcome.at_device)

    def test_confirm_without_a_point_is_refused(self) -> None:
        zoom, _ = build()
        zoom.open((0.5, 0.5))
        outcome = zoom.confirm()
        self.assertFalse(outcome.allowed)
        self.assertIs(outcome.reason, Z.Refusal.NO_POINT)

    def test_looking_before_opening_does_nothing(self) -> None:
        zoom, _ = build()
        zoom.look(0.5, 0.5)
        self.assertIsNone(zoom.point)

    def test_confirm_while_closed_is_refused(self) -> None:
        zoom, _ = build()
        self.assertFalse(zoom.confirm().allowed)


class FreshnessTests(unittest.TestCase):
    def test_a_changed_foreground_window_blocks_the_click(self) -> None:
        window = {"id": 4242}
        zoom, _ = build(foreground=lambda: window["id"])
        zoom.open((0.5, 0.5))
        zoom.look(0.5, 0.5)
        window["id"] = 777
        outcome = zoom.confirm()
        self.assertFalse(outcome.allowed)
        self.assertIs(outcome.reason, Z.Refusal.FOREGROUND_CHANGED)

    def test_our_own_window_in_front_blocks_the_click(self) -> None:
        # Otherwise the window check passes against ourselves and means
        # nothing, which is exactly what happens in practice mode.
        window = {"id": 4242}
        zoom, _ = build(foreground=lambda: window["id"], our_windows=lambda: {99, 4242})
        zoom.open((0.5, 0.5))
        zoom.look(0.5, 0.5)
        outcome = zoom.confirm()
        self.assertFalse(outcome.allowed)
        self.assertIs(outcome.reason, Z.Refusal.FOREGROUND_IS_OURS)

    def test_changed_pixels_under_the_point_block_the_click(self) -> None:
        zoom, screen = build()
        zoom.open((0.5, 0.5))
        zoom.look(0.5, 0.5)
        screen.fill = 200
        outcome = zoom.confirm()
        self.assertFalse(outcome.allowed)
        self.assertIs(outcome.reason, Z.Refusal.CONTENT_CHANGED)

    def test_unchanged_pixels_allow_the_click(self) -> None:
        zoom, _ = build()
        zoom.open((0.5, 0.5))
        zoom.look(0.5, 0.5)
        self.assertTrue(zoom.confirm().allowed)

    def test_only_the_neighbourhood_is_compared_not_the_whole_region(self) -> None:
        zoom, screen = build()
        zoom.open((0.5, 0.5))
        zoom.look(0.5, 0.5)
        zoom.confirm()
        checked = screen.grabs[-1]
        self.assertLessEqual(checked.width, Z.FRESH_RADIUS_PX * 2 + 2)

    def test_a_failed_grab_during_the_check_refuses_rather_than_raises(self) -> None:
        zoom, screen = build()
        zoom.open((0.5, 0.5))
        zoom.look(0.5, 0.5)
        screen.fail = True
        outcome = zoom.confirm()
        self.assertFalse(outcome.allowed)
        self.assertIs(outcome.reason, Z.Refusal.CONTENT_CHANGED)


class FailuresAreReportedNotRaisedTests(unittest.TestCase):
    def test_a_screen_that_cannot_be_read_refuses_to_open(self) -> None:
        screen = FakeScreen()
        screen.fail = True
        zoom, _ = build(screen=screen)
        outcome = zoom.open((0.5, 0.5))
        self.assertFalse(outcome.allowed)
        self.assertIs(outcome.reason, Z.Refusal.NO_CAPTURE)

    def test_every_refusal_is_counted(self) -> None:
        zoom, _ = build(layered=lambda: False)
        zoom.open((0.5, 0.5))
        zoom.open((0.5, 0.5))
        self.assertEqual(zoom.refusals[Z.Refusal.NOT_LAYERED], 2)
        self.assertEqual(zoom.summary()["refused"], {str(Z.Refusal.NOT_LAYERED): 2})


class NothingReachesTheDiskTests(unittest.TestCase):
    def test_the_module_never_opens_a_file(self) -> None:
        import ast

        source = Path(Z.__file__).read_text(encoding="utf-8")
        called = {
            node.func.id
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertNotIn("open", called)
        self.assertNotIn("write_bytes", source)
        self.assertNotIn("save(", source)


class ArithmeticTests(unittest.TestCase):
    def test_a_neighbourhood_stays_inside_the_captured_rectangle(self) -> None:
        rect = DeviceRect(100, 100, 200, 200)
        box = rect.around(101, 101, 50)
        self.assertGreaterEqual(box.x0, rect.x0)
        self.assertLessEqual(box.x1, rect.x1)

    def test_an_empty_rectangle_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            DeviceRect(10, 10, 10, 20)

    def test_the_deep_span_really_is_a_further_magnification(self) -> None:
        self.assertLess(Z.SPAN_DEEP, Z.SPAN_NEAR)


if __name__ == "__main__":
    unittest.main()
